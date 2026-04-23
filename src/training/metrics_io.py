"""Writer for the per-experiment ``metrics.tsv`` sidecar.

Training rule 5: every training entry point writes
``experiments/artifacts/<experiment_id>/metrics.tsv`` with one header row and
one value row. ``scripts/evaluate_gates.py`` reads from this file to evaluate
decision gates, and ``scripts/run_loop.sh`` reads columns 1 and 2 (VE and
PW-MCC) via awk for ntfy / commit messages.

The canonical column order below MUST keep ``variance_explained`` in column 1
and ``pwmcc`` in column 2 for the shell awk reads to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

METRICS_COLUMNS: tuple[str, ...] = (
    "variance_explained",          # col 1 — shell awk reads this for ntfy
    "pwmcc",                        # col 2 — shell awk reads this for ntfy
    "mmcs",
    "pwmcc_null_mean",
    "pwmcc_null_std",
    "dead_latent_fraction",
    "median_detection_score",
    "null_detection_score_p95",
    "variance_explained_heldout_tokens",
    "unmatched_fraction",
    "pwmcc_encoder_only",
    "pwmcc_decoder_only",
)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("\t", " ").replace("\n", " ")


def write_metrics_tsv(path: Path, metrics: dict[str, Any]) -> None:
    """Write a single-row metrics.tsv with the canonical header.

    Unknown keys in ``metrics`` are ignored. Missing columns are emitted empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(METRICS_COLUMNS) + "\n")
        f.write("\t".join(_fmt(metrics.get(c)) for c in METRICS_COLUMNS) + "\n")
