"""Metrics package.

Re-exports the public surface of `src.metrics.core` so that training and
analysis code can `from src.metrics import pw_mcc, mmcs, ...`.
"""

from src.metrics.core import (
    NULL_SEED_OFFSET,
    dead_latent_fraction,
    isotropic_gaussian,
    mmcs,
    pairwise_pwmcc_across_seeds,
    pw_mcc,
    variance_explained,
)

__all__ = [
    "NULL_SEED_OFFSET",
    "dead_latent_fraction",
    "isotropic_gaussian",
    "mmcs",
    "pairwise_pwmcc_across_seeds",
    "pw_mcc",
    "variance_explained",
]
