"""Read experiment rows out of `EXPERIMENTS.yaml`.

Every training entry point looks its row up through `load_row(experiment_id)`.
Training code must refuse to run if `experiment_id` is not in the matrix
(training rule 36).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "EXPERIMENTS.yaml"


class ExperimentNotFound(ValueError):
    """Raised when `--experiment-id` is not present in the locked matrix."""


def load_matrix() -> dict[str, Any]:
    with MATRIX_PATH.open() as f:
        return yaml.safe_load(f)


def load_row(experiment_id: str) -> dict[str, Any]:
    """Return the matrix row for `experiment_id`.

    Raises `ExperimentNotFound` if the id is missing. The runner must never
    invent experiments (CLAUDE.md rule 1); this refusal is the enforcement.
    """
    matrix = load_matrix()
    for row in matrix["experiments"]:
        if row["id"] == experiment_id:
            return row
    raise ExperimentNotFound(
        f"experiment_id {experiment_id!r} not in EXPERIMENTS.yaml. "
        f"Add it by editing the matrix (human-only; protected path)."
    )


def load_meta() -> dict[str, Any]:
    """Return the top-level `meta:` block (seeds, widths, ks, etc.)."""
    return load_matrix()["meta"]
