#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Bussmann/Leask reproduction reference. 0.5547 is the headline VE from
# Leask et al. arXiv:2502.04878, but it applies *only* to their specific
# config: GPT-2 Small + ReLU parent SAE (49,152 wide) + meta-SAE at
# dict_ratio 1/21. Our anchor rows use JumpReLU (Gemma) and BatchTopK
# (GPT-2) parents at dict_ratio 1/4, so this number is not a generalizable
# target. Kept here only as the auxk-trigger anchor below.
BUSSMANN_REPRO_REFERENCE_VE = 0.5547

# Auxk auto-trigger: when a BatchTopK anchor finishes with VE under this
# threshold, write the AUXK_ENABLED sentinel and re-queue the meta-SAE
# rows downstream of the anchor. Conservative floor; well below the
# Bussmann/Leask reference so spurious near-target results don't trip it.
AUXK_TRIGGER_VE = 0.50
AUXK_SENTINEL = Path("experiments/AUXK_ENABLED")
AUXK_TRIGGER_LOG = Path("experiments/AUXK_TRIGGER.log")
RESULTS_TSV = Path("experiments/results.tsv")
NTFY_SCRIPT = Path("scripts/ntfy_send.sh")

def load_metrics(metrics_file):
    metrics = {}
    if not os.path.exists(metrics_file):
        return metrics
    with open(metrics_file) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 2:
        return metrics
    headers = lines[0].split("\t")
    vals = lines[1].split("\t")
    for h, v in zip(headers, vals):
        try:
            metrics[h] = float(v)
        except ValueError:
            metrics[h] = v
    return metrics

def _num(v):
    if isinstance(v, (int, float)):
        return v
    return None

def compute_gate_value(metric, metrics):
    if metric == "variance_explained":
        return _num(metrics.get("variance_explained", metrics.get("VE")))
    if metric == "pwmcc_vs_null_sigma":
        obs = _num(metrics.get("pwmcc"))
        mu = _num(metrics.get("pwmcc_null_mean"))
        sd = _num(metrics.get("pwmcc_null_std"))
        if obs is None or mu is None or sd is None or sd == 0:
            return None
        return (obs - mu) / sd
    if metric == "variance_explained_deviation_from_leask":
        # Deprecated metric. Kept for back-compat in case any old row in
        # EXPERIMENTS.yaml still references it; new rows should use
        # `variance_explained` with an absolute floor instead. The reference
        # constant is config-specific (see BUSSMANN_REPRO_REFERENCE_VE
        # docstring above) and not a general target.
        ve = _num(metrics.get("variance_explained", metrics.get("VE")))
        if ve is None:
            return None
        return abs(ve - BUSSMANN_REPRO_REFERENCE_VE)
    if metric == "dead_latent_fraction":
        return _num(metrics.get("dead_latent_fraction"))
    if metric == "median_detection_score_vs_null_pct95":
        med = _num(metrics.get("median_detection_score"))
        null_p95 = _num(metrics.get("null_detection_score_p95"))
        if med is None or null_p95 is None or null_p95 == 0:
            return None
        return med / null_p95
    return None

def gate_triggered(metric, value, threshold):
    if value is None:
        return False
    if metric in ("variance_explained", "pwmcc_vs_null_sigma", "median_detection_score_vs_null_pct95"):
        return value < threshold
    if metric in ("variance_explained_deviation_from_leask", "dead_latent_fraction"):
        return value > threshold
    return False

def maybe_trigger_auxk(exp, metrics, matrix):
    """Auto-apply auxk if a BatchTopK anchor undershoots AUXK_TRIGGER_VE.

    Idempotent: bails out if AUXK_SENTINEL already exists. Touches the
    sentinel, appends `stale_auxk_fix` rows to results.tsv for every
    BatchTopK anchor + every meta-SAE row at depths 1-3 on either base
    model, and ntfys high-priority.
    """
    eid = exp["id"]
    if "batchtopk_anchor_d1" not in eid:
        return
    if AUXK_SENTINEL.exists():
        return
    ve = metrics.get("variance_explained", metrics.get("VE"))
    if ve is None or not isinstance(ve, (int, float)):
        return
    if ve >= AUXK_TRIGGER_VE:
        return

    AUXK_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    AUXK_SENTINEL.write_text(
        f"triggered by {eid} VE={ve:.4f} at {datetime.datetime.now().astimezone().isoformat()}\n"
    )

    # Re-queue every BatchTopK anchor + every depth-1/2/3 meta-SAE row, on
    # both base models and both arches. The runner picks them up under the
    # standard latest-row-wins rule (see scripts/run_loop.sh:104).
    requeue_ids = []
    for r in matrix["experiments"]:
        rid = r["id"]
        d = r.get("depth")
        if "batchtopk_anchor_d1" in rid:
            requeue_ids.append(rid)
            continue
        if d in (1, 2, 3) and r.get("level0_source") in ("train_from_scratch", "google/gemma-scope-2b-pt-res-canonical"):
            # Catches gemma_jumprelu_d{1,2,3}_*, gemma_batchtopk_d{1,2,3}_*,
            # gpt2_batchtopk_d{1,2,3}_*, and the ratio-ablation rows.
            requeue_ids.append(rid)

    now = datetime.datetime.now().astimezone().isoformat()
    with RESULTS_TSV.open("a") as f:
        for rid in requeue_ids:
            # 18-column TSV per SPEC §8.2; only timestamp/id/status are
            # populated for stale markers.
            f.write(f"{now}\t{rid}\tstale_auxk_fix\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n")

    log_line = f"{now}\t{eid}\tVE={ve:.4f}\trequeued={len(requeue_ids)}\n"
    with AUXK_TRIGGER_LOG.open("a") as f:
        f.write(log_line)

    title = f"[SAE auxk-trigger] {eid} VE={ve:.3f}<{AUXK_TRIGGER_VE}"
    msg = (
        f"BatchTopK anchor {eid} returned VE={ve:.4f}, below auxk trigger "
        f"threshold {AUXK_TRIGGER_VE}. Wrote {AUXK_SENTINEL}. Re-queued "
        f"{len(requeue_ids)} rows with stale_auxk_fix; runner will pick "
        f"them up with auxk loss enabled (Bussmann §3, alpha=1/32, k_aux=512). "
        f"To abort: rm {AUXK_SENTINEL}"
    )
    if NTFY_SCRIPT.exists():
        try:
            subprocess.run(
                ["bash", str(NTFY_SCRIPT), "high", title, msg, "warning"],
                timeout=15,
                check=False,
            )
        except Exception as e:
            print(f"[evaluate_gates] ntfy failed (non-fatal): {e!r}", file=sys.stderr)


def descendants_to_skip(matrix, exp, action):
    rows = matrix["experiments"]
    if action == "skip_experiment":
        return []
    if action == "skip_depth":
        lineage_key = (exp["base_model"], exp["level0_arch"], exp["seed"])
        return [r["id"] for r in rows
                if (r["base_model"], r["level0_arch"], r["seed"]) == lineage_key
                and r["depth"] > exp["depth"]
                and r["id"] != exp["id"]]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--metrics-file", required=True)
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--gates-tsv", required=True)
    args = ap.parse_args()

    with open("EXPERIMENTS.yaml") as f:
        matrix = yaml.safe_load(f)
    exp = next((e for e in matrix["experiments"] if e["id"] == args.experiment_id), None)
    if exp is None:
        print(f"unknown experiment {args.experiment_id}", file=sys.stderr)
        sys.exit(1)

    metrics = load_metrics(args.metrics_file)

    with open(args.state_file) as f:
        state = json.load(f)
    skipped = set(state.get("skipped_by_gate", []))

    now = datetime.datetime.now().astimezone().isoformat()
    triggered_any = False
    halt = False
    most_recent = None

    with open(args.gates_tsv, "a") as tsv:
        for gate in exp.get("decision_gates", []):
            metric = gate["metric"]
            threshold = gate["threshold"]
            action = gate["action"]
            value = compute_gate_value(metric, metrics)
            trig = gate_triggered(metric, value, threshold)
            tsv.write(f"{now}\t{exp['id']}\t{metric}\t{value}\t{threshold}\t{action}\t{int(trig)}\n")
            if not trig:
                continue
            triggered_any = True
            most_recent = {
                "experiment_id": exp["id"],
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "action": action,
                "timestamp": now,
            }
            if action == "halt_and_notify":
                halt = True
            elif action == "skip_depth":
                for sid in descendants_to_skip(matrix, exp, action):
                    skipped.add(sid)
            elif action == "skip_experiment":
                pass

    # Auxk auto-trigger: must run AFTER gate eval so the trigger ntfy and
    # the gate ntfy don't collide, and BEFORE state.json is written so any
    # state changes from the trigger (currently none, but reserved) compose.
    maybe_trigger_auxk(exp, metrics, matrix)

    state["skipped_by_gate"] = sorted(skipped)
    if most_recent is not None:
        state["most_recent_gate_outcome"] = most_recent
    state["last_heartbeat"] = now

    tmp = args.state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, args.state_file)

    # Decision-gate ntfys (skip_depth / skip_experiment) removed per operator
    # preference 2026-04-28: ntfy is reserved for human-judgment-required,
    # unrecoverable meta-blockers, and project completion. Gate fires are still
    # recorded in experiments/gates.tsv and state.most_recent_gate_outcome.

    if halt:
        sys.exit(42)
    sys.exit(0)

if __name__ == "__main__":
    main()
