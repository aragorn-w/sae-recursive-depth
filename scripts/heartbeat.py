#!/usr/bin/env python3

import datetime
import json
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

MATRIX_FILE = "EXPERIMENTS.yaml"
RESULTS_TSV = "experiments/results.tsv"
STATE_FILE = "experiments/state.json"
NTFY_TOPIC = "sae-wanga-research"

def load_matrix():
    with open(MATRIX_FILE) as f:
        return yaml.safe_load(f)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)

def load_results():
    if not os.path.exists(RESULTS_TSV):
        return []
    with open(RESULTS_TSV) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows

def summarize():
    matrix = load_matrix()
    state = load_state()
    results = load_results()

    total = len(matrix["experiments"])
    nonstretch_total = sum(1 for e in matrix["experiments"] if not (e.get("conditional") or {}).get("stretch_if_time_permits"))

    complete_ids = {r["experiment_id"] for r in results if r.get("status") == "ok"}
    failed_ids = {r["experiment_id"] for r in results if r.get("status") == "failed"}
    skipped_ids = set(state.get("skipped_by_gate", []))

    gpu_seconds = 0.0
    for r in results:
        try:
            gpu_seconds += float(r.get("gpu_hours") or 0)
        except ValueError:
            pass
    gpu_hours = gpu_seconds / 3600.0

    current = state.get("current_experiment_id") or "none"
    runner_status = state.get("runner_status") or "unknown"
    last_gate = state.get("most_recent_gate_outcome") or {}
    blockers = state.get("blockers") or []

    last_heartbeat = state.get("last_heartbeat") or "never"

    gate_line = "none"
    if last_gate.get("experiment_id"):
        gate_line = f"{last_gate['experiment_id']} {last_gate.get('metric','')}={last_gate.get('value','')} -> {last_gate.get('action','')}"

    now_denver = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    msg = (
        f"Heartbeat {now_denver}\n"
        f"Runner: {runner_status} (current: {current})\n"
        f"Complete: {len(complete_ids)}/{total} ({len(complete_ids) - len(skipped_ids - complete_ids)}/{nonstretch_total} non-stretch)\n"
        f"Failed: {len(failed_ids)}  Skipped-by-gate: {len(skipped_ids)}\n"
        f"GPU-hours used: {gpu_hours:.2f}\n"
        f"Most recent gate: {gate_line}\n"
        f"Blockers: {', '.join(blockers) if blockers else 'none'}\n"
        f"Last state update: {last_heartbeat}"
    )

    title = f"[SAE heartbeat {datetime.datetime.now().strftime('%H:%M')}] {len(complete_ids)}/{total} complete, {gpu_hours:.1f} GPU-hrs"
    return title, msg, runner_status

def post(title, msg):
    try:
        subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-H", f"Title: {title}",
             "-H", "Priority: min",
             "-H", "Tags: heart",
             "-d", msg,
             f"https://ntfy.sh/{NTFY_TOPIC}"],
            check=False, timeout=10,
        )
    except Exception as e:
        print(f"ntfy post failed: {e}", file=sys.stderr)

def main():
    title, msg, _ = summarize()
    print(title)
    print(msg)
    post(title, msg)

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        state["last_heartbeat"] = datetime.datetime.now().astimezone().isoformat()
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)

if __name__ == "__main__":
    main()
