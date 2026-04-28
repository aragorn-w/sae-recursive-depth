#!/usr/bin/env python3
"""Heartbeat + watchdog for the autopilot.

Cron-driven (8am/8pm America/Denver). Speaks plain English to a research
manager. Only escalates to a high-priority notification when the autopilot
is genuinely stuck — operational errors auto-retry silently, gates fire
silently. Otherwise it's an informational status ping.

Escalation conditions (any one triggers high priority):
  - Runner crashed (process died unexpectedly).
  - Stalled: nothing has finished in 24h+ and there's still work to do.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from collections import Counter

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

MATRIX_FILE = "EXPERIMENTS.yaml"
RESULTS_TSV = "experiments/results.tsv"
STATE_FILE = "experiments/state.json"
NTFY_TOPIC = "sae-wanga-research"

STALL_HOURS = 24
MAX_ATTEMPTS = 10


def _read_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def _read_results() -> list[dict[str, str]]:
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


def _parse_iso(ts: str) -> datetime.datetime | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts)
    except ValueError:
        return None


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False


# ---------- humanizing ----------

def humanize_id(eid: str) -> str:
    """Turn a matrix id like ``gemma_jumprelu_anchor_d1_s2`` into a sentence."""
    if not eid:
        return "(none)"
    if eid == "autointerp_all":
        return "feature interpretation pass (Llama-3.1-8B over all SAEs)"
    if eid == "simplestories_stretch":
        return "SimpleStories stretch experiment"
    if eid.startswith("l0_"):
        # l0_gemma_batchtopk, l0_gpt2_batchtopk
        m = re.match(r"l0_(gemma|gpt2)_(batchtopk|jumprelu)", eid)
        if m:
            base = "Gemma-2-2B" if m.group(1) == "gemma" else "GPT-2 Small"
            arch = {"jumprelu": "JumpReLU", "batchtopk": "BatchTopK"}[m.group(2)]
            return f"level-0 {arch} SAE training on {base}"
    if eid.startswith("null_"):
        # null_gemma_d3_s2
        m = re.match(r"null_(gemma|gpt2)_d(\d+)_s(\d+)", eid)
        if m:
            base = "Gemma" if m.group(1) == "gemma" else "GPT-2"
            return f"null baseline ({base}, depth {m.group(2)}, seed {m.group(3)})"
    if eid.startswith("flat_"):
        # flat_gemma_w16384_s0
        m = re.match(r"flat_(gemma|gpt2)_w(\d+)_s(\d+)", eid)
        if m:
            base = "Gemma" if m.group(1) == "gemma" else "GPT-2"
            return f"flat-SAE control ({base}, width {m.group(2)}, seed {m.group(3)})"
    # Recursive meta-SAE: <base>_<arch>[_anchor]_d<depth>_s<seed>
    m = re.match(r"(gemma|gpt2)_(jumprelu|batchtopk)(_anchor)?_d(\d+)_s(\d+)", eid)
    if m:
        base = "Gemma" if m.group(1) == "gemma" else "GPT-2"
        arch = {"jumprelu": "JumpReLU", "batchtopk": "BatchTopK"}[m.group(2)]
        anchor = " (Bussmann ratio anchor)" if m.group(3) else ""
        return f"{base} {arch} meta-SAE depth {m.group(4)} seed {m.group(5)}{anchor}"
    return eid


def humanize_metric(metric: str) -> str:
    return {
        "variance_explained": "variance explained",
        "variance_explained_deviation_from_leask": "VE deviation from Bussmann/Leask reference (config-specific, deprecated)",
        "pwmcc_vs_null_sigma": "seed-stability vs null baseline (sigma)",
        "dead_latent_fraction": "dead latent fraction",
        "median_detection_score_vs_null_pct95": "interpretability score vs null 95th %ile",
    }.get(metric, metric.replace("_", " "))


def fmt_hours_ago(dt: datetime.datetime | None, now: datetime.datetime) -> str:
    if dt is None:
        return "no completions yet"
    delta = now - dt
    h = delta.total_seconds() / 3600.0
    if h < 1:
        return f"{int(delta.total_seconds() / 60)} minutes ago"
    if h < 48:
        return f"{h:.1f} hours ago"
    return f"{h / 24:.1f} days ago"


def fmt_pct(s: str) -> str:
    """Turn '0.184972' into '18%'. Leaves non-numeric as-is."""
    try:
        return f"{round(float(s) * 100)}%"
    except (ValueError, TypeError):
        return s


# ---------- summary ----------

def summarize() -> tuple[str, str, str]:
    matrix = _read_yaml(MATRIX_FILE)
    state = _read_state()
    rows = _read_results()

    # Latest-row-wins per SPEC §8.2.
    latest: dict[str, dict[str, str]] = {}
    attempts: Counter[str] = Counter()
    latest_ok: tuple[datetime.datetime, dict[str, str]] | None = None
    for r in rows:
        eid = r.get("experiment_id", "")
        if not eid:
            continue
        ts = r.get("timestamp", "")
        prev = latest.get(eid)
        if prev is None or ts > prev.get("timestamp", ""):
            latest[eid] = r
        if r.get("status") == "failed":
            attempts[eid] += 1
        if r.get("status") == "ok":
            t = _parse_iso(ts)
            if t is not None and (latest_ok is None or t > latest_ok[0]):
                latest_ok = (t, r)

    nonstretch_ids = {
        e["id"] for e in matrix["experiments"]
        if not (e.get("conditional") or {}).get("stretch_if_time_permits")
    }
    complete_ids = {eid for eid, r in latest.items() if r.get("status") == "ok"}
    n_done = len(complete_ids & nonstretch_ids)
    n_total = len(nonstretch_ids)
    n_remaining = n_total - n_done
    permanently_failed = {eid for eid, n in attempts.items() if n >= MAX_ATTEMPTS}

    gpu_seconds = 0.0
    for r in rows:
        try:
            gpu_seconds += float(r.get("gpu_hours") or 0)
        except ValueError:
            pass
    gpu_hours = gpu_seconds / 3600.0

    runner_status = state.get("runner_status") or "unknown"
    # Liveness check: with the parallel orchestrator, treat the runner as
    # alive if any per-lane pid is live, or if the legacy state.pid is live.
    lanes_state = state.get("lanes") or {}
    lane_pids = [v.get("pid") for v in lanes_state.values() if isinstance(v, dict)]
    runner_alive = _pid_alive(state.get("pid")) or any(_pid_alive(p) for p in lane_pids)
    runner_crashed = runner_status == "running" and not runner_alive

    now = datetime.datetime.now().astimezone()
    last_progress_dt = latest_ok[0] if latest_ok else None
    hours_since = (now - last_progress_dt).total_seconds() / 3600.0 if last_progress_dt else None
    stalled = (
        n_remaining > 0
        and hours_since is not None
        and hours_since > STALL_HOURS
    )

    # Pretty fragments. With the parallel orchestrator, current_experiment_id
    # is a dict {lane_label: exp_id}. Fall back to the legacy string form.
    _cur = state.get("current_experiment_id") or ""
    if isinstance(_cur, dict):
        active = [v for v in _cur.values() if v]
        if not active:
            current = ""
        elif len(active) == 1:
            current = humanize_id(active[0])
        else:
            current = " + ".join(humanize_id(a) for a in active)
    else:
        current = humanize_id(_cur)
    last_done_line = ""
    if latest_ok is not None:
        r = latest_ok[1]
        ve = fmt_pct(r.get("variance_explained", ""))
        last_done_line = (
            f"Last finished: {humanize_id(r.get('experiment_id', ''))}, "
            f"variance explained {ve} ({fmt_hours_ago(last_progress_dt, now)})."
        )

    last_gate_line = ""
    g = state.get("most_recent_gate_outcome") or {}
    if g.get("experiment_id"):
        try:
            v = f"{float(g.get('value', 0)):.3g}"
        except (ValueError, TypeError):
            v = str(g.get("value", ""))
        last_gate_line = (
            f"Most recent quality flag: {humanize_id(g['experiment_id'])} — "
            f"{humanize_metric(g.get('metric', ''))} = {v} "
            f"(threshold {g.get('threshold', '')})."
        )

    # Decide priority and headline.
    if runner_crashed:
        priority = "high"
        title = "Autopilot crashed — needs restart"
        lines = [
            "The autopilot process is no longer running, but its state file "
            "says it should be. To restart: tmux attach -t sae-loop, or kill "
            "and relaunch from the project directory.",
            f"Progress at crash: {n_done} of {n_total} experiments done.",
        ]
    elif stalled:
        priority = "high"
        title = f"Autopilot stalled — no progress in {hours_since:.0f}h"
        lines = [
            f"Nothing has finished in {hours_since:.1f} hours, but "
            f"{n_remaining} experiments are still pending.",
            f"Currently working on: {current}.",
            "This usually means a configuration issue (missing dependency, "
            "wrong model id) that the auto-retry can't resolve. Worth a look.",
        ]
    else:
        priority = "min"
        if n_remaining == 0:
            title = "Autopilot done — all experiments resolved"
        else:
            title = f"Autopilot status: {n_done} of {n_total} experiments done"
        lines = []
        if runner_status == "running":
            lines.append(f"Currently training: {current}.")
        elif runner_status == "idle":
            lines.append(
                "Runner is idle (no eligible work right now). It stays alive "
                "and will pick up retries or new dependencies automatically."
            )
        elif runner_status == "paused":
            lines.append("Runner paused (PAUSE file present).")
        elif runner_status == "halted":
            lines.append("Runner is halted. Restart needed.")
        if last_done_line:
            lines.append(last_done_line)
        if last_gate_line:
            lines.append(last_gate_line)
        if permanently_failed:
            sample = sorted(permanently_failed)[:3]
            extra = f" plus {len(permanently_failed) - 3} more" if len(permanently_failed) > 3 else ""
            lines.append(
                f"Experiments given up on after many retries: "
                f"{', '.join(humanize_id(x) for x in sample)}{extra}. "
                f"These need attention when convenient."
            )
        lines.append(f"Total compute used so far: {gpu_hours:.1f} GPU-hours.")

    body = "\n".join(lines)
    return title, body, priority


def post(title: str, body: str, priority: str) -> None:
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "-H", f"Title: {title}",
                "-H", f"Priority: {priority}",
                "-H", "Tags: heart" if priority == "min" else "Tags: warning",
                "-d", body,
                f"https://ntfy.sh/{NTFY_TOPIC}",
            ],
            check=False,
            timeout=10,
        )
    except Exception as e:
        print(f"ntfy post failed: {e}", file=sys.stderr)


def main() -> None:
    title, body, priority = summarize()
    print(title)
    print(body)
    post(title, body, priority)

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
