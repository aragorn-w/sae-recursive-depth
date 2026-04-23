"""Append-only writer for `experiments/results.tsv`.

Training rule 2: every training entry point calls `append_result(...)` from a
`finally:` block so a partial result is recorded even on interrupt. Ground
truth is this TSV; W&B is visualization only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV = REPO_ROOT / "experiments" / "results.tsv"

RESULTS_TSV_HEADER: tuple[str, ...] = (
    "timestamp",
    "experiment_id",
    "status",
    "base_model",
    "level0_arch",
    "depth",
    "seed",
    "width",
    "variance_explained",
    "pwmcc",
    "mmcs",
    "pwmcc_null_mean",
    "pwmcc_null_std",
    "dead_latent_fraction",
    "gpu_hours",
    "commit_sha",
    "wandb_run_url",
    "notes",
)


def _ensure_header() -> None:
    if RESULTS_TSV.exists() and RESULTS_TSV.stat().st_size > 0:
        return
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_TSV.open("w") as f:
        f.write("\t".join(RESULTS_TSV_HEADER) + "\n")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v)
    # TSV cells must not contain tab or newline
    return s.replace("\t", " ").replace("\n", " ")


def append_result(row: dict[str, Any]) -> None:
    """Append one row to `experiments/results.tsv`.

    Unknown keys in `row` are ignored. Missing columns are emitted empty.
    Writes are done via `os.write(O_APPEND)` + `os.fsync` so concurrent
    appends from the shell runner and a Python `finally` block do not
    interleave within a single line (POSIX atomicity of writes under
    `PIPE_BUF` holds for a single-row TSV line).
    """
    _ensure_header()
    line = "\t".join(_fmt(row.get(col)) for col in RESULTS_TSV_HEADER) + "\n"
    data = line.encode("utf-8")
    if len(data) >= 4096:
        raise ValueError(
            f"results row exceeds 4096 bytes ({len(data)}); "
            f"atomic append no longer guaranteed. Shorten the notes field."
        )
    fd = os.open(RESULTS_TSV, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
