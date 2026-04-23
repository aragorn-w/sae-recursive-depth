#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import sys

import yaml

LEASK_VE = 0.5547

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

def compute_gate_value(metric, metrics):
    if metric == "variance_explained":
        return metrics.get("variance_explained", metrics.get("VE"))
    if metric == "pwmcc_vs_null_sigma":
        obs = metrics.get("pwmcc")
        mu = metrics.get("pwmcc_null_mean")
        sd = metrics.get("pwmcc_null_std")
        if obs is None or mu is None or sd is None or sd == 0:
            return None
        return (obs - mu) / sd
    if metric == "variance_explained_deviation_from_leask":
        ve = metrics.get("variance_explained", metrics.get("VE"))
        if ve is None:
            return None
        return abs(ve - LEASK_VE)
    if metric == "dead_latent_fraction":
        return metrics.get("dead_latent_fraction")
    if metric == "median_detection_score_vs_null_pct95":
        med = metrics.get("median_detection_score")
        null_p95 = metrics.get("null_detection_score_p95")
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

    state["skipped_by_gate"] = sorted(skipped)
    if most_recent is not None:
        state["most_recent_gate_outcome"] = most_recent
    state["last_heartbeat"] = now

    tmp = args.state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, args.state_file)

    if halt:
        sys.exit(42)
    sys.exit(0)

if __name__ == "__main__":
    main()
