"""Recursive meta-SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows with ``depth >= 1``
(SPEC.md:43). Input is the decoder-direction matrix of the previous depth
(training rule 9).

Meta-SAEs are always BatchTopK per training rule 7 ("k=4 for all
meta-SAEs"). The ``level0_arch`` field in EXPERIMENTS.yaml identifies the
*parent* SAE's architecture (which determines how the parent is loaded),
not the meta-SAE's architecture.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.metrics import dead_latent_fraction, variance_explained
from src.training.harness import experiment_context
from src.training.loaders import load_level0
from src.training.metrics_io import write_metrics_tsv
from src.training.sae_models import build_sae

D_MODEL: dict[str, int] = {
    "google/gemma-2-2b": 2304,
    "openai-community/gpt2": 768,
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_parent_decoder(row: dict) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return (W_enc_parent, W_dec_parent, parent_source) for the given row.

    Both matrices are shaped ``(n_latents_parent, d_model)`` and unit-
    normalized by row per training rule 9.
    """
    depth = int(row["depth"])
    deps = row.get("dependencies") or []

    if depth == 1:
        # Parent is the level-0 SAE identified by `level0_source`.
        loaded = load_level0(
            level0_source=row["level0_source"],
            base_model=row["base_model"],
            layer=int(row["layer"]),
            site=row["site"],
        )
        W_enc = loaded.W_enc.to(torch.float32)
        W_dec = loaded.W_dec.to(torch.float32)
        # Ensure shape (n_latents, d_model)
        W_enc = _as_n_by_d(W_enc, loaded.W_dec.shape)
        W_dec = _as_n_by_d(W_dec, loaded.W_dec.shape)
        return (
            F.normalize(W_enc, dim=1),
            F.normalize(W_dec, dim=1),
            f"{loaded.source}@{loaded.revision}",
        )

    # depth >= 2: parent is a local meta-SAE checkpoint from a dependency id.
    if not deps:
        raise ValueError(f"depth={depth} row requires a dependency id in the matrix")
    parent_id = deps[0]
    parent_ckpt_path = Path("experiments/artifacts") / parent_id / "checkpoint.pt"
    if not parent_ckpt_path.exists():
        raise FileNotFoundError(
            f"parent checkpoint missing: {parent_ckpt_path}. "
            f"Dependency {parent_id!r} must be complete before this row runs."
        )
    parent = torch.load(parent_ckpt_path, map_location="cpu", weights_only=False)
    W_enc = parent["W_enc"].to(torch.float32)  # (d_in, n_latents)
    W_dec = parent["W_dec"].to(torch.float32)  # (n_latents, d_in)
    # Encoder is stored as (d_in, n_latents) in our checkpoint schema; transpose
    # so both matrices share the (n_latents, d_in) shape expected downstream.
    W_enc = W_enc.T.contiguous()
    return F.normalize(W_enc, dim=1), F.normalize(W_dec, dim=1), f"local:{parent_id}"


def _as_n_by_d(W: torch.Tensor, reference_shape: torch.Size) -> torch.Tensor:
    """Coerce an encoder or decoder matrix to ``(n_latents, d_model)``.

    Upstream SAE loaders vary in whether they return the encoder as
    ``(d_in, n_latents)`` or ``(n_latents, d_in)``. `loaders.py` guarantees
    ``W_dec`` is ``(n_latents, d_in)`` already and ``W_enc`` is
    ``(n_latents, d_in)`` (same shape) for the uniform return type.
    """
    if W.shape == reference_shape:
        return W
    if W.shape == (reference_shape[1], reference_shape[0]):
        return W.T.contiguous()
    raise ValueError(
        f"unexpected weight shape {tuple(W.shape)} vs reference {tuple(reference_shape)}"
    )


def main() -> None:
    with experiment_context(arch_hint="meta") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: meta-SAE harness ok"
            return
        ctx.init_wandb()

        row = ctx.row
        base = row["base_model"]
        if base not in D_MODEL:
            raise ValueError(f"meta-SAE: unknown base_model {base!r}")
        d_model = D_MODEL[base]
        parent_arch = row["level0_arch"]  # parent (level-0) arch; drives loader routing
        # Meta-SAEs are always BatchTopK per training rule 7; k is row.sparsity.
        arch = "batchtopk"
        width = int(row["width"])
        sparsity = int(row["sparsity"])

        if parent_arch not in ("batchtopk", "jumprelu"):
            raise ValueError(
                f"meta-SAE row {ctx.experiment_id} has level0_arch={parent_arch!r}; "
                f"expected 'batchtopk' or 'jumprelu'"
            )

        W_enc_parent, W_dec_parent, parent_source = _load_parent_decoder(row)
        if W_dec_parent.shape[1] != d_model:
            raise ValueError(
                f"parent decoder d={W_dec_parent.shape[1]} does not match "
                f"base_model d_model={d_model}"
            )
        parent_width = W_dec_parent.shape[0]

        device = _device()
        x_train = W_dec_parent.to(device)  # (parent_width, d_model), unit rows

        sae = build_sae(arch=arch, d_in=d_model, n_latents=width, sparsity=sparsity).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)

        # Hold out 10% for evaluation; training signals unit-normalized
        # decoder rows (not activations), so a random split within the parent
        # decoder matrix is fine.
        n_train = int(parent_width * 0.9)
        perm_seed = torch.Generator(device="cpu")
        perm_seed.manual_seed(ctx.seed)
        perm = torch.randperm(parent_width, generator=perm_seed)
        train_idx = perm[:n_train].to(device)
        eval_idx = perm[n_train:].to(device)

        # Step-budget-driven training: 20k gradient steps regardless of
        # dataset size, so tiny depth-3 inputs (parent_width=4096) still get
        # enough updates to converge. Adam lr=3e-4 per training rule 8.
        total_steps = 20000
        batch_size = min(4096, max(256, n_train // 4))
        log_every = max(1, total_steps // 100)

        curves_path: Path = ctx.artifact_dir / "curves.tsv"
        with curves_path.open("w") as cf:
            cf.write("step\tloss\n")
            step = 0
            while step < total_steps:
                # Reshuffle the training split each pass.
                gen = torch.Generator(device="cpu")
                gen.manual_seed(ctx.seed * 97 + step)
                epoch_perm = torch.randperm(n_train, generator=gen).to(device)
                for b in range(0, n_train - batch_size + 1, batch_size):
                    if step >= total_steps:
                        break
                    bidx = train_idx[epoch_perm[b : b + batch_size]]
                    batch = x_train[bidx]
                    recon, _ = sae(batch, use_training_topk=True)
                    loss = F.mse_loss(recon, batch)
                    if not torch.isfinite(loss):
                        _write_diverged(ctx, "NaN loss")
                        raise RuntimeError(f"meta-SAE diverged (NaN loss) step={step}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                    if grad_norm.item() > 1000:
                        _write_diverged(ctx, f"grad_norm={grad_norm.item():.3g}")
                        raise RuntimeError(f"meta-SAE diverged (grad norm) step={step}")
                    opt.step()
                    # Renormalize decoder rows so each feature stays on the unit
                    # sphere. Standard SAE hygiene (Bussmann et al., Bricken et
                    # al.); prevents W_dec from shrinking to dodge reconstruction
                    # cost, which lets W_enc pick up the slack.
                    with torch.no_grad():
                        sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
                    if step % log_every == 0:
                        cf.write(f"{step}\t{loss.item():.6g}\n")
                        if ctx.wandb_run is not None:
                            try:
                                ctx.wandb_run.log({"train/loss": loss.item(), "step": step})
                            except Exception:
                                pass
                    step += 1
            # Final loss line.
            cf.write(f"{step}\t{loss.item():.6g}\n")

        # Fit inference threshold on the full training split so the eval path
        # with `use_training_topk=False` behaves consistently with training.
        with torch.no_grad():
            sae.fit_inference_threshold(x_train[train_idx])

        # Held-out eval. Use training-topk for VE so the reported number
        # reflects the intended sparsity constraint (k=4 for meta-SAEs),
        # matching Leask et al.'s evaluation protocol.
        with torch.no_grad():
            x_eval = x_train[eval_idx]
            recon_eval, latents_eval = sae(x_eval, use_training_topk=True)
        ve = variance_explained(recon_eval, x_eval)
        dead_frac = dead_latent_fraction(latents_eval)

        if not math.isfinite(ve):
            ctx.notes = "variance_explained non-finite on eval"

        ckpt = {
            "experiment_id": ctx.experiment_id,
            "arch": arch,
            "d_in": d_model,
            "n_latents": width,
            "sparsity": sparsity,
            "W_enc": sae.W_enc.detach().to(torch.float16).cpu(),
            "b_enc": sae.b_enc.detach().to(torch.float16).cpu(),
            "W_dec": sae.W_dec.detach().to(torch.float16).cpu(),
            "b_dec": sae.b_dec.detach().to(torch.float16).cpu(),
            "jump_threshold": sae.jump_threshold.detach().to(torch.float16).cpu(),
            "parent_source": parent_source,
            "parent_arch": parent_arch,
            "row": row,
        }
        torch.save(ckpt, ctx.artifact_dir / "checkpoint.pt")

        metrics = {
            "variance_explained": ve,
            "pwmcc": None,            # computed post-hoc across sibling seeds
            "mmcs": None,
            "pwmcc_null_mean": None,  # populated post-hoc from null_* rows
            "pwmcc_null_std": None,
            "dead_latent_fraction": dead_frac,
            "variance_explained_heldout_tokens": int(x_eval.shape[0]),
        }
        write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)

        ctx.metrics["variance_explained"] = ve
        ctx.metrics["dead_latent_fraction"] = dead_frac
        ctx.notes = (
            f"meta-SAE arch=batchtopk (parent={parent_arch}) width={width} "
            f"k={sparsity} parent={parent_source} parent_width={parent_width}"
        )


def _write_diverged(ctx, reason: str) -> None:
    metrics = {
        "variance_explained": float("nan"),
        "pwmcc": None,
        "dead_latent_fraction": float("nan"),
    }
    write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)
    ctx.notes = f"diverged: {reason}"


if __name__ == "__main__":
    main()
