"""SAE model modules.

Two architectures per `.claude/rules/training.md`:

- ``BatchTopKSAE`` (Bussmann et al. 2024, Matryoshka/BatchTopK line,
  arXiv:2503.17547). Per-batch global top-(B*k) sparsity at train time;
  JumpReLU-style per-latent threshold used at inference (training rule 7 sets
  ``k = 60`` for level-0 Gemma, ``k = 4`` for meta-SAEs; the module reads the
  value from the caller rather than hard-coding it).
- ``JumpReLUSAE`` (Rajamanoharan et al. 2024, arXiv:2407.14435). Learned
  per-latent threshold ``theta`` applied via a straight-through estimator.

Both modules store weights fp32 for training stability and expose a helper
``decoder_directions()`` that returns the unit-normalized decoder matrix per
training rule 9 (meta-SAEs consume parent decoder directions as their input).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Shared config for all SAE variants."""

    arch: str  # "batchtopk" | "jumprelu"
    d_in: int  # input dimensionality (d_model for level-0/meta, = parent d_model)
    n_latents: int
    # For batchtopk: the per-token top-k count at inference; training uses B*k globally.
    # For jumprelu: target L0 (not enforced, used only for logging).
    sparsity: int
    # JumpReLU-only: initial threshold value (log-space).
    jumprelu_init_log_theta: float = -2.0


class _SAEBase(nn.Module):
    """Shared scaffolding for both variants."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.n_latents))
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_latents))
        self.W_dec = nn.Parameter(torch.empty(cfg.n_latents, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self._init_weights()

    def _init_weights(self) -> None:
        # Tied-init: W_enc = W_dec.T with unit-norm decoder rows, a standard
        # SAE init that keeps the encoder roughly aligned with the decoder at
        # step 0.
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
            self.W_enc.data = self.W_dec.data.T.clone()

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.W_dec + self.b_dec

    def decoder_directions(self, unit_norm: bool = True) -> torch.Tensor:
        """Return the (n_latents, d_in) decoder matrix.

        Per training rule 9, unit-normalized rows are fed to the next meta-SAE.
        """
        W = self.W_dec.detach()
        if unit_norm:
            W = F.normalize(W, dim=1)
        return W.contiguous()

    def encoder_directions(self, unit_norm: bool = True) -> torch.Tensor:
        """Return the (n_latents, d_in) encoder matrix (for PW-MCC joint sim)."""
        W = self.W_enc.detach().T  # -> (n_latents, d_in)
        if unit_norm:
            W = F.normalize(W, dim=1)
        return W.contiguous()


class BatchTopKSAE(_SAEBase):
    """BatchTopK SAE.

    Training-time activation: keep the global top (B * k) pre-activations
    across a batch, zero the rest. Inference-time: use a learned per-latent
    threshold fitted from the running max of the zeroed-out activations
    ("the last latent that survived the batch cut gives the threshold").
    """

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__(cfg)
        # Inference-time threshold, fit from training stats. Registered as a
        # buffer so it serializes with the checkpoint.
        self.register_buffer("jump_threshold", torch.zeros(cfg.n_latents))

    def forward(
        self,
        x: torch.Tensor,
        *,
        use_training_topk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.preact(x)
        pre = F.relu(pre)
        if use_training_topk:
            latents = _batch_top_k(pre, self.cfg.sparsity)
        else:
            latents = torch.where(pre > self.jump_threshold, pre, torch.zeros_like(pre))
        recon = self.decode(latents)
        return recon, latents

    @torch.no_grad()
    def fit_inference_threshold(
        self,
        x: torch.Tensor,
        *,
        quantile_floor: float = 0.0,
        chunk_size: int = 4096,
    ) -> None:
        """Set ``jump_threshold[i]`` to the min nonzero training activation for latent i.

        Processes x in chunks of ``chunk_size`` rows to bound peak GPU memory.
        Per-latent min is associative so the chunked result matches the
        unchunked result modulo the BatchTopK cut being applied per-chunk
        rather than globally — fine for threshold fitting and matches the
        per-batch regime the trained model was optimized under.

        Memory: O(chunk_size * n_latents) per chunk vs. O(N * n_latents)
        for the unchunked path. At width=16384, N=14745, the unchunked path
        materializes ~3 GiB of (latents, masked, mask) tensors on top of
        model + optimizer state, which OOMs on 16 GiB cards. Chunking caps
        that to ~800 MiB regardless of N.
        """
        n = x.shape[0]
        device = x.device
        ref_dtype = self.W_enc.dtype
        per_latent_min = torch.full(
            (self.cfg.n_latents,), float("inf"), device=device, dtype=ref_dtype
        )
        for start in range(0, n, chunk_size):
            chunk = x[start : start + chunk_size]
            pre = F.relu(self.preact(chunk))
            latents = _batch_top_k(pre, self.cfg.sparsity)
            nonzero_mask = latents > 0
            masked = torch.where(
                nonzero_mask, latents, torch.full_like(latents, float("inf"))
            )
            chunk_min, _ = masked.min(dim=0)
            per_latent_min = torch.minimum(per_latent_min, chunk_min)
        per_latent_min = torch.where(
            torch.isfinite(per_latent_min),
            per_latent_min,
            torch.zeros_like(per_latent_min),
        )
        if quantile_floor > 0:
            per_latent_min = torch.clamp(per_latent_min, min=quantile_floor)
        self.jump_threshold.copy_(per_latent_min)


def _batch_top_k(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Global top-(B*k) over the flattened (batch, n_latents) tensor.

    Exact Bussmann BatchTopK: one global cut across the whole batch.
    """
    B, L = pre.shape
    total = B * k
    if total >= pre.numel():
        return pre
    flat = pre.reshape(-1)
    topk = torch.topk(flat, total, sorted=False)
    mask = torch.zeros_like(flat)
    mask.scatter_(0, topk.indices, 1.0)
    return (flat * mask).reshape(B, L)


class _StraightThroughStep(torch.autograd.Function):
    """H(z - theta) with a straight-through gradient."""

    @staticmethod
    def forward(ctx, z, theta):  # type: ignore[override]
        ctx.save_for_backward(z, theta)
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        z, theta = ctx.saved_tensors
        # Rectangular STE: gradient flows through whenever z is near theta.
        # Bandwidth epsilon = 0.001 per Rajamanoharan et al. Appendix C.
        eps = 0.001
        mask = (z.sub(theta).abs() < eps).to(z.dtype) / (2 * eps)
        # d/dz = mask; d/dtheta = -mask
        return grad_out * mask, -grad_out * mask


class JumpReLUSAE(_SAEBase):
    """JumpReLU SAE (Rajamanoharan et al. 2024).

    Learnable per-latent threshold ``theta = exp(log_theta)`` kept positive via
    the exp parameterization. Forward: ``latent = z * H(z - theta)`` where H
    is the straight-through Heaviside step.
    """

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__(cfg)
        self.log_theta = nn.Parameter(
            torch.full((cfg.n_latents,), cfg.jumprelu_init_log_theta)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.preact(x))
        theta = torch.exp(self.log_theta)
        gate = _StraightThroughStep.apply(z, theta)
        latents = z * gate
        recon = self.decode(latents)
        return recon, latents


def build_sae(arch: str, *, d_in: int, n_latents: int, sparsity: int) -> _SAEBase:
    """Factory used by training bodies. Reads row config and returns a module."""
    cfg = SAEConfig(arch=arch, d_in=d_in, n_latents=n_latents, sparsity=sparsity)
    if arch == "batchtopk":
        return BatchTopKSAE(cfg)
    if arch == "jumprelu":
        return JumpReLUSAE(cfg)
    raise ValueError(f"unknown SAE arch {arch!r}")
