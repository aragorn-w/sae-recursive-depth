"""Figures + summary tables.

Reads ``experiments/results.tsv`` (latest-row-wins per SPEC §8.2) and
``experiments/pwmcc_posthoc.tsv``. Writes plots and companion data files
under ``experiments/artifacts/_summary/`` per analysis-rules.md section 1
(every figure has a sibling .tsv with the exact numbers plotted) and
rule 7 (each data file carries a reproducibility stamp).

Scope is strictly: plots, tables, and analysis numerics. No outline, no
narrative, no captions beyond what-is-plotted. Writeup is out of scope.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV = REPO_ROOT / "experiments" / "results.tsv"
POSTHOC_TSV = REPO_ROOT / "experiments" / "pwmcc_posthoc.tsv"
SUMMARY_DIR = REPO_ROOT / "experiments" / "artifacts" / "_summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
TABLES_DIR = SUMMARY_DIR / "tables"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


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


def _latest_per_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for r in rows:
        eid = r.get("experiment_id", "")
        if not eid:
            continue
        ts = r.get("timestamp", "")
        if eid not in latest or ts > latest[eid].get("timestamp", ""):
            latest[eid] = r
    return latest


def _ftry(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _stamp_lines(experiment_ids: list[str], script_path: str) -> list[str]:
    """Reproducibility stamp lines per analysis rule 7."""
    return [
        f"# generated_at_utc\t{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"# git_commit_sha\t{_git_sha()}",
        f"# analysis_script\t{script_path}",
        f"# experiment_ids\t{','.join(sorted(experiment_ids))}",
    ]


def _write_data_file(path: Path, header: list[str], rows: list[list], stamp: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in stamp:
            f.write(line + "\n")
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(c) for c in r) + "\n")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        return f"{v:.6g}"
    return str(v)


# ---------- aggregation ----------

def _build_main_table() -> tuple[list[dict], list[dict]]:
    """Return (main_rows, posthoc_rows) where each main_row carries the latest
    headline metrics joined with the matching posthoc PW-MCC row."""
    _, results = _read_tsv(RESULTS_TSV)
    latest = _latest_per_id(results)
    _, posthoc = _read_tsv(POSTHOC_TSV)
    posthoc_latest = _latest_per_id(posthoc)

    main: list[dict] = []
    for eid, r in latest.items():
        if r.get("status") != "ok":
            continue
        p = posthoc_latest.get(eid, {})
        joined = {
            "experiment_id": eid,
            "base_model": r.get("base_model", ""),
            "level0_arch": r.get("level0_arch", ""),
            "depth": r.get("depth", ""),
            "seed": r.get("seed", ""),
            "width": r.get("width", ""),
            "variance_explained": _ftry(r.get("variance_explained", "")),
            "pwmcc": _ftry(p.get("pwmcc", "")) if p else None,
            "mmcs": _ftry(r.get("mmcs", "")),
            "pwmcc_null_mean": _ftry(p.get("pwmcc_null_mean", "")) if p else None,
            "pwmcc_null_std": _ftry(p.get("pwmcc_null_std", "")) if p else None,
            "dead_latent_fraction": _ftry(r.get("dead_latent_fraction", "")),
            "gpu_hours": _ftry(r.get("gpu_hours", "")),
        }
        main.append(joined)
    return main, list(posthoc_latest.values())


# ---------- plots ----------

def _save_fig(fig, fig_path: Path, data_path: Path, header, rows, stamp):
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    _write_data_file(data_path, header, rows, stamp)


def plot_ve_vs_depth(main: list[dict], script_path: str) -> None:
    import matplotlib.pyplot as plt

    # Group by (base_model, level0_arch); aggregate VE mean/std per depth.
    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in main:
        if r["level0_arch"] == "none":
            continue
        if r["variance_explained"] is None:
            continue
        try:
            d = int(r["depth"])
        except (ValueError, TypeError):
            continue
        groups[(r["base_model"], r["level0_arch"])][d].append(r["variance_explained"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    table_rows = []
    eids_used = set()
    for (base, arch), per_depth in sorted(groups.items()):
        depths = sorted(per_depth.keys())
        means = [float(np.mean(per_depth[d])) for d in depths]
        stds = [float(np.std(per_depth[d])) for d in depths]
        label = f"{_short_base(base)} / {arch}"
        ax.errorbar(depths, means, yerr=stds, marker="o", capsize=3, label=label)
        for d, m, s in zip(depths, means, stds):
            table_rows.append([base, arch, d, len(per_depth[d]), m, s])
    for r in main:
        eids_used.add(r["experiment_id"])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Variance explained")
    ax.set_title("Variance explained by depth (mean ± std across seeds)")
    ax.set_xticks(sorted({d for g in groups.values() for d in g.keys()}))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    stamp = _stamp_lines(sorted(eids_used), script_path)
    _save_fig(
        fig,
        FIGURES_DIR / "variance_explained_by_depth.pdf",
        FIGURES_DIR / "variance_explained_by_depth.tsv",
        ["base_model", "level0_arch", "depth", "n_seeds", "mean_ve", "std_ve"],
        table_rows,
        stamp,
    )


def plot_pwmcc_vs_depth(main: list[dict], script_path: str) -> None:
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"obs": [], "null_mean": [], "null_std": []}))
    for r in main:
        if r["level0_arch"] == "none":
            continue
        if r["pwmcc"] is None:
            continue
        try:
            d = int(r["depth"])
        except (ValueError, TypeError):
            continue
        cell = groups[(r["base_model"], r["level0_arch"])][d]
        cell["obs"].append(r["pwmcc"])
        if r["pwmcc_null_mean"] is not None:
            cell["null_mean"].append(r["pwmcc_null_mean"])
        if r["pwmcc_null_std"] is not None:
            cell["null_std"].append(r["pwmcc_null_std"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    table_rows = []
    eids_used = {r["experiment_id"] for r in main}
    for (base, arch), per_depth in sorted(groups.items()):
        depths = sorted(per_depth.keys())
        means = [float(np.mean(per_depth[d]["obs"])) for d in depths]
        nulls = [float(np.mean(per_depth[d]["null_mean"])) if per_depth[d]["null_mean"] else float("nan") for d in depths]
        null_sds = [float(np.mean(per_depth[d]["null_std"])) if per_depth[d]["null_std"] else float("nan") for d in depths]
        label = f"{_short_base(base)} / {arch}"
        line = ax.plot(depths, means, marker="o", label=label)[0]
        for d, m, n, ns in zip(depths, means, nulls, null_sds):
            table_rows.append([base, arch, d, m, n, ns])
            if not np.isnan(n) and not np.isnan(ns):
                ax.fill_between(
                    [d - 0.1, d + 0.1],
                    [n - 2 * ns, n - 2 * ns],
                    [n + 2 * ns, n + 2 * ns],
                    alpha=0.15,
                    color=line.get_color(),
                )
    ax.set_xlabel("Depth")
    ax.set_ylabel("PW-MCC across seeds")
    ax.set_title("Cross-seed PW-MCC by depth, with ±2σ null bands")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    stamp = _stamp_lines(sorted(eids_used), script_path)
    _save_fig(
        fig,
        FIGURES_DIR / "pwmcc_by_depth.pdf",
        FIGURES_DIR / "pwmcc_by_depth.tsv",
        ["base_model", "level0_arch", "depth", "mean_pwmcc", "mean_null_pwmcc", "mean_null_pwmcc_std"],
        table_rows,
        stamp,
    )


def plot_dead_fraction(main: list[dict], script_path: str) -> None:
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in main:
        if r["dead_latent_fraction"] is None:
            continue
        try:
            d = int(r["depth"])
        except (ValueError, TypeError):
            continue
        groups[(r["base_model"], r["level0_arch"])][d].append(r["dead_latent_fraction"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    eids = {r["experiment_id"] for r in main}
    for (base, arch), per_depth in sorted(groups.items()):
        depths = sorted(per_depth.keys())
        means = [float(np.mean(per_depth[d])) for d in depths]
        ax.plot(depths, means, marker="o", label=f"{_short_base(base)} / {arch}")
        for d, m in zip(depths, means):
            rows.append([base, arch, d, len(per_depth[d]), m])
    ax.set_xlabel("Depth")
    ax.set_ylabel("Dead latent fraction")
    ax.set_title("Fraction of dead latents by depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    stamp = _stamp_lines(sorted(eids), script_path)
    _save_fig(
        fig,
        FIGURES_DIR / "dead_latent_fraction_by_depth.pdf",
        FIGURES_DIR / "dead_latent_fraction_by_depth.tsv",
        ["base_model", "level0_arch", "depth", "n_seeds", "mean_dead_fraction"],
        rows,
        stamp,
    )


def _short_base(b: str) -> str:
    return {"google/gemma-2-2b": "Gemma-2-2B", "openai-community/gpt2": "GPT-2 Small"}.get(b, b)


# ---------- tables ----------

def write_results_pivot(main: list[dict], script_path: str) -> None:
    """Per-row table: every successful experiment with its headline metrics."""
    cols = [
        "experiment_id", "base_model", "level0_arch", "depth", "seed", "width",
        "variance_explained", "pwmcc", "pwmcc_null_mean", "pwmcc_null_std",
        "mmcs", "dead_latent_fraction", "gpu_hours",
    ]
    rows = []
    for r in sorted(main, key=lambda r: r["experiment_id"]):
        rows.append([r.get(c) for c in cols])
    stamp = _stamp_lines([r["experiment_id"] for r in main], script_path)
    _write_data_file(TABLES_DIR / "results_pivot.tsv", cols, rows, stamp)


def write_cell_summary(main: list[dict], script_path: str) -> None:
    """Per-cell table: aggregated mean ± std across seeds."""
    cells: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for r in main:
        try:
            d = int(r["depth"])
            w = int(r["width"])
        except (ValueError, TypeError):
            continue
        cells[(r["base_model"], r["level0_arch"], d, w)].append(r)
    cols = [
        "base_model", "level0_arch", "depth", "width", "n_seeds",
        "mean_variance_explained", "std_variance_explained",
        "mean_pwmcc", "std_pwmcc",
        "mean_dead_fraction",
    ]
    rows = []
    eids = []
    for (base, arch, d, w), members in sorted(cells.items()):
        ves = [m["variance_explained"] for m in members if m["variance_explained"] is not None]
        pws = [m["pwmcc"] for m in members if m["pwmcc"] is not None]
        deads = [m["dead_latent_fraction"] for m in members if m["dead_latent_fraction"] is not None]
        rows.append([
            base, arch, d, w, len(members),
            float(np.mean(ves)) if ves else None,
            float(np.std(ves)) if ves else None,
            float(np.mean(pws)) if pws else None,
            float(np.std(pws)) if pws else None,
            float(np.mean(deads)) if deads else None,
        ])
        eids.extend(m["experiment_id"] for m in members)
    stamp = _stamp_lines(sorted(set(eids)), script_path)
    _write_data_file(TABLES_DIR / "cell_summary.tsv", cols, rows, stamp)


# ---------- driver ----------

def build_all() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    main, _ = _build_main_table()
    if not main:
        print("[figures] no ok rows in results.tsv yet; nothing to plot")
        return

    script_path = "src/analysis/figures.py"
    plot_ve_vs_depth(main, script_path)
    plot_pwmcc_vs_depth(main, script_path)
    plot_dead_fraction(main, script_path)
    write_results_pivot(main, script_path)
    write_cell_summary(main, script_path)
    print(f"[figures] wrote figures + tables under {SUMMARY_DIR}")


if __name__ == "__main__":
    build_all()
