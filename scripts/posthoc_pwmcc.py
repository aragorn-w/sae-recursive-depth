#!/usr/bin/env python3
"""Compute pairwise PW-MCC across sibling seeds, write to pwmcc_posthoc.tsv.

For each (base_model, level0_arch, depth, width) cell where all 3 seeds have
the latest results.tsv status == "ok", load each seed's checkpoint, compute
joint enc+dec Hungarian PW-MCC pairwise, and append per-seed rows to
``experiments/pwmcc_posthoc.tsv``. Also computes the null distribution
(mean, std) for any null_* cell and emits per-non-null seed rows annotated
with the matching null mean/std.

Idempotent: skips cells whose 3 seed checkpoints are all older than the
posthoc row already written for them.

Run via cron, runner hook, or manually:

    uv run python scripts/posthoc_pwmcc.py
"""

from __future__ import annotations

import datetime
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import pairwise_pwmcc_across_seeds, pw_mcc  # noqa: E402

RESULTS_TSV = REPO_ROOT / "experiments" / "results.tsv"
POSTHOC_TSV = REPO_ROOT / "experiments" / "pwmcc_posthoc.tsv"
ARTIFACTS = REPO_ROOT / "experiments" / "artifacts"
MATRIX_FILE = REPO_ROOT / "EXPERIMENTS.yaml"

POSTHOC_HEADER = (
    "timestamp",
    "experiment_id",
    "cell_key",
    "pwmcc",
    "pwmcc_encoder_only",
    "pwmcc_decoder_only",
    "unmatched_fraction",
    "pwmcc_null_mean",
    "pwmcc_null_std",
    "n_seeds",
    "n_pairs",
    "notes",
)


def _ensure_header() -> None:
    if POSTHOC_TSV.exists() and POSTHOC_TSV.stat().st_size > 0:
        return
    POSTHOC_TSV.parent.mkdir(parents=True, exist_ok=True)
    with POSTHOC_TSV.open("w") as f:
        f.write("\t".join(POSTHOC_HEADER) + "\n")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open() as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 2:
        return (lines[0].split("\t") if lines else []), []
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return header, rows


def _latest_ok(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for r in rows:
        eid = r.get("experiment_id", "")
        if not eid:
            continue
        ts = r.get("timestamp", "")
        prev = latest.get(eid)
        if prev is None or ts > prev.get("timestamp", ""):
            latest[eid] = r
    return {eid: r for eid, r in latest.items() if r.get("status") == "ok"}


def _cell_key(row: dict[str, str]) -> str | None:
    """Group key used to discover sibling-seed cells.

    Two experiments are siblings when their (base_model, level0_arch, depth,
    width, name_root) match. The name_root strips the trailing seed and is
    the most reliable cross-check given the matrix's row id structure.
    """
    eid = row.get("experiment_id", "")
    if "_s" not in eid:
        return None
    name_root = eid.rsplit("_s", 1)[0]
    return (
        f"{row.get('base_model','')}|{row.get('level0_arch','')}|"
        f"{row.get('depth','')}|{row.get('width','')}|{name_root}"
    )


def _load_ckpt(eid: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(W_enc, W_dec)`` shaped (n_latents, d_in), fp32, on CPU."""
    ckpt_path = ARTIFACTS / eid / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W_enc = blob["W_enc"].to(torch.float32)  # stored as (d_in, n_latents)
    W_dec = blob["W_dec"].to(torch.float32)  # stored as (n_latents, d_in)
    if W_enc.shape == W_dec.shape:
        # Already (n_latents, d_in)
        pass
    elif W_enc.shape == (W_dec.shape[1], W_dec.shape[0]):
        W_enc = W_enc.T.contiguous()
    else:
        return None
    return W_enc, W_dec


def _ckpt_mtime(eid: str) -> float:
    p = ARTIFACTS / eid / "checkpoint.pt"
    return p.stat().st_mtime if p.exists() else 0.0


def _null_cell_key(real_row: dict[str, str]) -> str:
    """For a real meta-SAE/flat row, find the matching null cell key.

    Null cells are named `null_<base>_d<depth>_s<seed>`. Match on
    (base, depth, width).
    """
    base = "gemma" if "gemma" in real_row.get("base_model", "") else "gpt2"
    depth = real_row.get("depth", "")
    width = real_row.get("width", "")
    return (
        f"{real_row.get('base_model','')}|none|{depth}|{width}|null_{base}_d{depth}"
    )


def main() -> None:
    _ensure_header()
    _, results = _read_tsv(RESULTS_TSV)
    latest_ok = _latest_ok(results)

    # Group complete experiments by cell.
    by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for eid, r in latest_ok.items():
        key = _cell_key(r)
        if key:
            by_cell[key].append(r)

    # Read existing posthoc rows so we can skip up-to-date cells.
    _, post_rows = _read_tsv(POSTHOC_TSV)
    posthoc_seen: dict[str, float] = {}
    for r in post_rows:
        try:
            ts = datetime.datetime.fromisoformat(r.get("timestamp", "")).timestamp()
        except ValueError:
            ts = 0.0
        eid = r.get("experiment_id", "")
        if eid:
            posthoc_seen[eid] = max(posthoc_seen.get(eid, 0.0), ts)

    # First pass: compute null cell stats.
    null_stats: dict[str, tuple[float, float]] = {}  # cell_key -> (mean, std)
    for cell_key, members in by_cell.items():
        if not cell_key.split("|")[4].startswith("null_"):
            continue
        if len(members) < 2:
            continue
        decoders: list[tuple[torch.Tensor, torch.Tensor]] = []
        for r in members:
            ckpt = _load_ckpt(r["experiment_id"])
            if ckpt is None:
                break
            decoders.append(ckpt)
        if len(decoders) != len(members):
            continue
        agg = pairwise_pwmcc_across_seeds(decoders)
        null_stats[cell_key] = (agg["pwmcc_mean"], agg["pwmcc_std"])

    # Second pass: write per-seed posthoc rows for every cell with >=2 seeds.
    now = datetime.datetime.now().astimezone().isoformat()
    new_rows: list[list[str]] = []
    for cell_key, members in by_cell.items():
        if len(members) < 2:
            continue
        # Up-to-date check: skip if all seeds in this cell already have a
        # posthoc row newer than their checkpoint mtime.
        all_fresh = True
        for r in members:
            eid = r["experiment_id"]
            ck_t = _ckpt_mtime(eid)
            ph_t = posthoc_seen.get(eid, 0.0)
            if ph_t < ck_t:
                all_fresh = False
                break
        if all_fresh:
            continue

        decoders: list[tuple[torch.Tensor, torch.Tensor]] = []
        loaded_eids: list[str] = []
        for r in members:
            ckpt = _load_ckpt(r["experiment_id"])
            if ckpt is None:
                continue
            decoders.append(ckpt)
            loaded_eids.append(r["experiment_id"])
        if len(decoders) < 2:
            continue

        # Per-seed PW-MCC: average of pw_mcc against each other seed.
        # This gives a per-experiment value to put in the pwmcc column.
        n = len(decoders)
        per_seed: list[dict] = [
            {"pwmcc": [], "pwmcc_encoder_only": [], "pwmcc_decoder_only": [], "unmatched_fraction": []}
            for _ in range(n)
        ]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ea, da = decoders[i]
                eb, db = decoders[j]
                d = pw_mcc(ea, da, eb, db)
                per_seed[i]["pwmcc"].append(d["pwmcc"])
                per_seed[i]["pwmcc_encoder_only"].append(d["pwmcc_encoder_only"])
                per_seed[i]["pwmcc_decoder_only"].append(d["pwmcc_decoder_only"])
                per_seed[i]["unmatched_fraction"].append(d["unmatched_fraction"])

        # Look up matching null cell stats. For null cells themselves, no null
        # match (they ARE the null).
        is_null_cell = cell_key.split("|")[4].startswith("null_")
        nm, ns = (None, None)
        if not is_null_cell:
            null_key = _null_cell_key(members[0])
            if null_key in null_stats:
                nm, ns = null_stats[null_key]

        for idx, eid in enumerate(loaded_eids):
            row_dict = {
                "timestamp": now,
                "experiment_id": eid,
                "cell_key": cell_key,
                "pwmcc": _avg(per_seed[idx]["pwmcc"]),
                "pwmcc_encoder_only": _avg(per_seed[idx]["pwmcc_encoder_only"]),
                "pwmcc_decoder_only": _avg(per_seed[idx]["pwmcc_decoder_only"]),
                "unmatched_fraction": _avg(per_seed[idx]["unmatched_fraction"]),
                "pwmcc_null_mean": "" if nm is None else f"{nm:.6g}",
                "pwmcc_null_std": "" if ns is None else f"{ns:.6g}",
                "n_seeds": str(n),
                "n_pairs": str(n - 1),
                "notes": ("null cell" if is_null_cell else "real-data cell"),
            }
            new_rows.append([str(row_dict.get(c, "")) for c in POSTHOC_HEADER])

    if not new_rows:
        print("posthoc_pwmcc: nothing new to compute")
        return

    with POSTHOC_TSV.open("a") as f:
        for row in new_rows:
            f.write("\t".join(row) + "\n")
    print(f"posthoc_pwmcc: appended {len(new_rows)} rows -> {POSTHOC_TSV}")


def _avg(xs: list[float]) -> str:
    if not xs:
        return ""
    return f"{sum(xs) / len(xs):.6g}"


if __name__ == "__main__":
    main()
