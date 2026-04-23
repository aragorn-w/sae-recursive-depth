"""Deterministic seed setup for SAE / meta-SAE training.

Implements the mandatory seed protocol from `.claude/rules/training.md` (rule 3):
every training entry point calls `set_all_seeds(seed)` before any model or data
code runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all RNG state and enable deterministic CUDA kernels.

    Per training rule 3: this must be called before any model or data code runs.
    `warn_only=True` lets ops without a deterministic implementation fall back to
    a nondeterministic one with a warning, rather than raising; nondeterminism at
    that layer is reported through W&B and `metrics.tsv` rather than blocking
    the run.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
