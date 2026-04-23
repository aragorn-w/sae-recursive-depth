"""Core metrics for SAE evaluation.

Implements the three mandatory metrics from `.claude/rules/metrics.md`:

1. ``pw_mcc`` (Pairwise Mean Max Cosine, Beiderman et al. 2025). Hungarian
   matching on the element-wise sum of encoder and decoder cosine similarity
   per Paulo & Belrose (arXiv:2501.16615 sec. 4), with the mean-of-two-cosines
   threshold 0.7 deciding matched vs unmatched.
2. ``mmcs`` (Mean Max Cosine Similarity, Sharkey et al. 2023,
   arXiv:2309.08600). For each feature in ``A``, take the maximum cosine with
   any feature in reference ``B``; return the mean.
3. ``isotropic_gaussian`` null draws seeded with ``null_seed = 1000 + seed``
   per rule 3.

Plus two utilities used across training bodies:

- ``variance_explained(recon, target)`` per rule 4
  (fraction of variance explained on a held-out batch).
- ``dead_latent_fraction(activations)`` per rule 5
  (fraction of latents that never activate across the provided batch).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# Reference: null_seed = 1000 + trained_seed so null and data draws never collide.
# See .claude/rules/metrics.md section "Mandatory behaviors" item 3.
NULL_SEED_OFFSET = 1000


def isotropic_gaussian(
    n: int,
    d: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw ``(n, d)`` standard-normal vectors with deterministic seeding."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    x = torch.randn(n, d, generator=g, dtype=dtype)
    return x.to(device)


def variance_explained(recon: torch.Tensor, target: torch.Tensor) -> float:
    """``1 - var(target - recon) / var(target)``.

    Per metrics rule 4 this is computed on a held-out shard. The caller
    decides which split to pass; this helper only does the math.
    """
    with torch.no_grad():
        resid = target - recon
        num = resid.pow(2).mean()
        den = (target - target.mean(dim=0, keepdim=True)).pow(2).mean()
        if den.item() == 0:
            return float("nan")
        return float(1.0 - (num / den).item())


def dead_latent_fraction(latent_batch: torch.Tensor) -> float:
    """Fraction of latents that never activate across ``latent_batch``.

    ``latent_batch`` is (N, n_latents). Per metrics rule 5 the caller is
    responsible for supplying roughly 10M tokens of held-out activations.
    """
    with torch.no_grad():
        active = (latent_batch != 0).any(dim=0)
        return float(1.0 - active.float().mean().item())


def _rows_unit_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.detach(), dim=1)


def pw_mcc(
    W_enc_a: torch.Tensor,
    W_dec_a: torch.Tensor,
    W_enc_b: torch.Tensor,
    W_dec_b: torch.Tensor,
    *,
    match_threshold: float = 0.7,
) -> dict:
    """PW-MCC with joint enc+dec Hungarian matching.

    Both encoder and decoder matrices must be shaped ``(n_latents, d_in)``.
    Returns a dict with keys:

    - ``pwmcc``: mean joint cosine across matched pairs (enc+dec)/2.
    - ``pwmcc_encoder_only``: mean encoder-only cosine across matched pairs.
    - ``pwmcc_decoder_only``: mean decoder-only cosine across matched pairs.
    - ``unmatched_fraction``: fraction of pairs with joint cosine below
      ``match_threshold``.

    Reference: Paulo & Belrose (arXiv:2501.16615 section 4) for the joint-sum
    Hungarian matching; Beiderman et al. 2025 for the PW-MCC formulation used
    in SAE seed-stability work. SPEC §9 uses the MEAN of enc and dec cos
    (i.e. joint sum / 2) so the statistic stays bounded in [-1, 1].
    """
    ea = _rows_unit_norm(W_enc_a)
    da = _rows_unit_norm(W_dec_a)
    eb = _rows_unit_norm(W_enc_b)
    db = _rows_unit_norm(W_dec_b)

    enc_sim = ea @ eb.T  # (n_a, n_b)
    dec_sim = da @ db.T
    joint_sum = enc_sim + dec_sim  # used for Hungarian cost (higher is better)

    # Handle unequal sizes: Hungarian over min(n_a, n_b) pairs.
    cost = -joint_sum.cpu().to(torch.float32).numpy()
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_joint_mean = (joint_sum[row_idx, col_idx] / 2.0).cpu()
    matched_enc = enc_sim[row_idx, col_idx].cpu()
    matched_dec = dec_sim[row_idx, col_idx].cpu()

    matched_mask = matched_joint_mean >= match_threshold
    if matched_mask.any():
        pwmcc_val = matched_joint_mean[matched_mask].mean().item()
        pwmcc_enc = matched_enc[matched_mask].mean().item()
        pwmcc_dec = matched_dec[matched_mask].mean().item()
    else:
        pwmcc_val = 0.0
        pwmcc_enc = 0.0
        pwmcc_dec = 0.0

    unmatched_fraction = float(1.0 - matched_mask.float().mean().item())

    return {
        "pwmcc": float(pwmcc_val),
        "pwmcc_encoder_only": float(pwmcc_enc),
        "pwmcc_decoder_only": float(pwmcc_dec),
        "unmatched_fraction": unmatched_fraction,
        "n_matched": int(matched_mask.sum().item()),
        "n_pairs": int(len(row_idx)),
    }


def mmcs(W_a: torch.Tensor, W_ref: torch.Tensor) -> float:
    """Mean Max Cosine Similarity.

    For each row of ``W_a``, take the max cosine to any row of ``W_ref``;
    return the mean. Both matrices are (n_features, d).

    Reference: Sharkey et al. 2023 (arXiv:2309.08600 figure 6 caption).
    """
    a = _rows_unit_norm(W_a)
    r = _rows_unit_norm(W_ref)
    sim = a @ r.T
    maxes, _ = sim.max(dim=1)
    return float(maxes.mean().item())


def pairwise_pwmcc_across_seeds(
    seeds_decoders: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    match_threshold: float = 0.7,
) -> dict:
    """Average PW-MCC over all C(N, 2) seed pairs.

    ``seeds_decoders`` is a list of ``(W_enc, W_dec)`` tuples, one per seed.
    Returns a dict with the mean / std / min / max of pairwise PW-MCC and
    the mean unmatched_fraction.
    """
    if len(seeds_decoders) < 2:
        raise ValueError("need at least two seeds for pairwise comparison")
    pairs = []
    unmatched = []
    for i in range(len(seeds_decoders)):
        for j in range(i + 1, len(seeds_decoders)):
            ea, da = seeds_decoders[i]
            eb, db = seeds_decoders[j]
            r = pw_mcc(ea, da, eb, db, match_threshold=match_threshold)
            pairs.append(r["pwmcc"])
            unmatched.append(r["unmatched_fraction"])
    arr = np.array(pairs)
    return {
        "pwmcc_mean": float(arr.mean()),
        "pwmcc_std": float(arr.std(ddof=0)),
        "pwmcc_min": float(arr.min()),
        "pwmcc_max": float(arr.max()),
        "unmatched_fraction_mean": float(np.mean(unmatched)),
        "n_pairs": len(pairs),
    }
