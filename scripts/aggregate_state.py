#!/usr/bin/env python3
"""aggregate_state.py — fold per-lane state files into legacy state.json.

Each lane worker writes experiments/state.lane.<label>.json with its own
runner_status, current_experiment_id, last_heartbeat. The legacy file
experiments/state.json must remain a single source of truth for downstream
consumers (heartbeat, dashboard, smoke checks). This script merges:

    runner_status:      "running" if any lane running, else "idle"|"paused"
    last_heartbeat:     max across lanes
    current_experiment_id:
                        dict {lane_label: exp_id} for live introspection
    most_recent_gate_outcome:
                        most recent across lanes (by timestamp)
    blockers:           union across lanes
    skipped_by_gate:    union across lanes
    lanes:              full per-lane state for debugging
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments")
LEGACY = os.path.join(EXP, "state.json")


def _parse_iso(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    lane_files = sorted(glob.glob(os.path.join(EXP, "state.lane.*.json")))
    lanes = {}
    for path in lane_files:
        try:
            with open(path) as f:
                lanes[os.path.basename(path).removeprefix("state.lane.").removesuffix(".json")] = json.load(f)
        except Exception as e:
            lanes[os.path.basename(path)] = {"_error": repr(e)}

    if not lanes:
        # No lane files yet; preserve legacy file untouched.
        return 0

    # Aggregate
    runner_status = "idle"
    statuses = [v.get("runner_status") for v in lanes.values() if isinstance(v, dict)]
    if any(s == "running" for s in statuses):
        runner_status = "running"
    elif any(s == "paused" for s in statuses):
        runner_status = "paused"

    last_heartbeat = max(
        (_parse_iso(v.get("last_heartbeat", "")) or datetime.min.replace(tzinfo=timezone.utc) for v in lanes.values() if isinstance(v, dict)),
        default=datetime.now(timezone.utc),
    ).isoformat()

    current = {label: v.get("current_experiment_id") for label, v in lanes.items() if isinstance(v, dict)}

    # Pick most recent gate outcome across lanes (by gate timestamp)
    gate_outcomes = []
    for v in lanes.values():
        if isinstance(v, dict):
            g = v.get("most_recent_gate_outcome") or {}
            ts = _parse_iso(g.get("timestamp", ""))
            if ts is not None and g.get("experiment_id"):
                gate_outcomes.append((ts, g))
    most_recent_gate = max(gate_outcomes, key=lambda x: x[0])[1] if gate_outcomes else {}

    blockers = sorted({b for v in lanes.values() if isinstance(v, dict) for b in (v.get("blockers") or [])})
    skipped = sorted({s for v in lanes.values() if isinstance(v, dict) for s in (v.get("skipped_by_gate") or [])})

    legacy = {
        "runner_status": runner_status,
        "pid": None,
        "started_at": None,
        "last_heartbeat": last_heartbeat,
        "current_experiment_id": current,
        "most_recent_gate_outcome": most_recent_gate,
        "blockers": blockers,
        "skipped_by_gate": skipped,
        "lanes": lanes,
    }

    # Preserve human_disposition if it exists on the legacy gate outcome
    if os.path.exists(LEGACY):
        try:
            with open(LEGACY) as f:
                prev = json.load(f)
            prev_gate = prev.get("most_recent_gate_outcome") or {}
            if prev_gate.get("experiment_id") == most_recent_gate.get("experiment_id"):
                hd = prev_gate.get("human_disposition")
                if hd and "human_disposition" not in legacy["most_recent_gate_outcome"]:
                    legacy["most_recent_gate_outcome"]["human_disposition"] = hd
        except Exception:
            pass

    tmp = LEGACY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(legacy, f, indent=2)
    os.replace(tmp, LEGACY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
