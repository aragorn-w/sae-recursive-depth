#!/usr/bin/env bash
# scripts/run_analysis.sh
# Analysis-lane worker. Polls for milestones:
#   1. autointerp_all dependencies satisfied -> dispatch autointerp_all
#      (no other lane picks it up because its gpu_preference is "2,3"
#       which doesn't match any standard lane filter)
#   2. matrix complete + autointerp complete -> run final analysis pass
#      (figures, tables, MMCS, summary stats), generate draft prose stubs,
#      write PRE_PAPER_COMPLETE sentinel, ntfy high-priority.
#
# Idempotent: each pass detects what's already done via results.tsv and
# the PRE_PAPER_COMPLETE sentinel. Safe to run repeatedly.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/gpu_policy.sh"

LOG_DIR="experiments/logs"
RESULTS_TSV="experiments/results.tsv"
GIT_LOCK="experiments/.git_commit.lock"
SENTINEL="experiments/PRE_PAPER_COMPLETE"
NTFY_TOPIC="sae-wanga-research"
mkdir -p "$LOG_DIR"

ntfy_send() {
    local priority="$1"; local title="$2"; local msg="$3"
    bash scripts/ntfy_send.sh "$priority" "$title" "$msg" >/dev/null 2>&1 || true
}

git_commit_locked() {
    local msg="$1"; shift
    (
        flock -x 9
        git add "$@" 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "$msg" --no-gpg-sign 2>&1 | tail -10 || true
        fi
    ) 9>"$GIT_LOCK"
}

# --- 1. autointerp dispatch ----------------------------------------------------

dispatch_autointerp() {
    local exp_id="autointerp_all"
    if uv run python - <<PY
import sys
from pathlib import Path
import yaml
m = yaml.safe_load(open("EXPERIMENTS.yaml"))
exp = next((e for e in m["experiments"] if e["id"] == "$exp_id"), None)
if exp is None:
    sys.exit(2)

# Latest status for autointerp_all
latest = {}
with open("experiments/results.tsv") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        latest[parts[1]] = parts[2]

if latest.get("$exp_id") == "ok":
    sys.exit(3)  # already done

# Dependencies: all must be latest=ok
unmet = [d for d in exp.get("dependencies", []) if latest.get(d) != "ok"]
if unmet:
    print("autointerp deps unmet:", len(unmet), "rows", file=sys.stderr)
    sys.exit(4)
sys.exit(0)
PY
    then
        local rc=0
    else
        local rc=$?
        case $rc in
            2) echo "[analysis] autointerp_all not in matrix (skip)"; return 0 ;;
            3) return 0 ;;  # already done
            4) return 1 ;;  # deps unmet, try later
            *) echo "[analysis] dep check error rc=$rc"; return 1 ;;
        esac
    fi

    echo "[analysis] dispatching autointerp_all"
    local log="$LOG_DIR/autointerp_all__lane-analysis_$(date +%Y%m%d_%H%M%S).log"
    # Pinning to CUDA 3 (3090) since Llama-3.1-8B fp16 needs ~16GB and the
    # 3090 is preferred over the 4060 Ti for KV-cache headroom.
    SAE_LANE_LABEL=analysis CUDA_VISIBLE_DEVICES=3 \
        uv run python -m src.analysis.run_autointerp --experiment-id autointerp_all 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "[analysis] autointerp rc=$rc"

    # Run gate eval + posthoc + figures + commit (mirrors run_loop's run_one tail)
    if [[ -f "experiments/artifacts/autointerp_all/metrics.tsv" ]]; then
        uv run python scripts/evaluate_gates.py \
            --experiment-id autointerp_all \
            --metrics-file experiments/artifacts/autointerp_all/metrics.tsv \
            --state-file experiments/state.lane.analysis.json \
            --gates-tsv experiments/gates.tsv 2>&1 | tail -20 || true
    fi
    uv run python scripts/posthoc_pwmcc.py >>"$log" 2>&1 || true
    uv run python -m src.analysis.figures >>"$log" 2>&1 || true
    git_commit_locked "[SAE-LOOP] autointerp_all complete" \
        experiments/artifacts/autointerp_all/ \
        experiments/artifacts/_summary/ \
        experiments/results.tsv experiments/gates.tsv experiments/pwmcc_posthoc.tsv \
        experiments/state.lane.analysis.json experiments/logs/

    return $rc
}

# --- 2. matrix completion check ----------------------------------------------

matrix_complete() {
    uv run python <<'PY'
import sys, yaml
m = yaml.safe_load(open("EXPERIMENTS.yaml"))
latest = {}
with open("experiments/results.tsv") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        latest[parts[1]] = parts[2]

# Skipped-by-gate also counts as resolved.
import json, os
state = {}
if os.path.exists("experiments/state.json"):
    try: state = json.load(open("experiments/state.json"))
    except Exception: pass
skipped = set(state.get("skipped_by_gate") or [])

def resolved(eid):
    if latest.get(eid) == "ok": return True
    if eid in skipped: return True
    return False

unresolved = []
for e in m["experiments"]:
    cond = e.get("conditional") or {}
    if cond.get("stretch_if_time_permits"):
        continue  # stretch rows are optional
    if not resolved(e["id"]):
        unresolved.append(e["id"])

if unresolved:
    print(f"unresolved: {len(unresolved)}", file=sys.stderr)
    print("\n".join(unresolved[:10]), file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
}

# --- 3. final analysis + draft prose ----------------------------------------

final_analysis_pass() {
    echo "[analysis] running final pass: posthoc PW-MCC, figures, summary tables"
    local log="$LOG_DIR/final_analysis_$(date +%Y%m%d_%H%M%S).log"

    # Re-run posthoc PW-MCC and figures one more time, with everything in.
    # NO PAPER WRITING — only quantitative artifacts (numbers, plots, tables).
    uv run python scripts/posthoc_pwmcc.py 2>&1 | tee -a "$log" || true
    uv run python -m src.analysis.figures 2>&1 | tee -a "$log" || true

    git_commit_locked "[SAE-LOOP] final analysis pass (figures/tables refreshed)" \
        experiments/artifacts/_summary/ \
        experiments/results.tsv experiments/gates.tsv experiments/pwmcc_posthoc.tsv \
        experiments/state.json experiments/state.lane.analysis.json experiments/logs/

    # Sentinel + ntfy
    if [[ ! -f "$SENTINEL" ]]; then
        printf "%s\nMatrix + autointerp + final analysis pipeline complete.\nFigures and summary tables refreshed under experiments/artifacts/_summary/.\nReady for human review. NO prose drafting performed.\n" \
            "$(date -Iseconds)" > "$SENTINEL"
        git_commit_locked "[SAE-LOOP] PRE_PAPER_COMPLETE sentinel raised" "$SENTINEL"
        ntfy_send "high" \
            "[SAE] PRE-PAPER COMPLETE" \
            "All non-stretch matrix rows resolved, autointerp scored, figures and summary tables refreshed. Workstation is idle and ready for paper-writing review. No drafting performed (per operator directive)."
        echo "[analysis] PRE_PAPER_COMPLETE sentinel raised"
    fi
}

# --- main ---------------------------------------------------------------------

main() {
    if [[ -f experiments/HALT ]]; then
        echo "[analysis] HALT present, exiting"
        return 0
    fi
    if [[ -f experiments/PAUSE ]]; then
        echo "[analysis] PAUSE present, sleeping"
        sleep 60
        return 0
    fi

    if [[ -f "$SENTINEL" ]]; then
        # Already done; nothing to do.
        echo "[analysis] $SENTINEL present; idle"
        sleep 1800
        return 0
    fi

    # Try to dispatch autointerp_all if eligible. Non-fatal if not yet ready.
    dispatch_autointerp || true

    # Always re-run incremental figures/posthoc each pass (cheap, idempotent).
    uv run python scripts/posthoc_pwmcc.py >/dev/null 2>&1 || true
    uv run python -m src.analysis.figures >/dev/null 2>&1 || true

    # Check matrix completion + autointerp; if both done, finalize.
    if matrix_complete; then
        # Verify autointerp_all is also done (it's part of "matrix" but only
        # runs from this lane, so double-check).
        local ai_status
        ai_status=$(awk -F'\t' '$2=="autointerp_all" {s=$3} END{print s}' "$RESULTS_TSV")
        if [[ "$ai_status" == "ok" ]]; then
            final_analysis_pass
        else
            echo "[analysis] matrix complete but autointerp_all not yet ok (status=$ai_status); waiting"
        fi
    fi

    # Idle pass period — short enough to be responsive, long enough to not hammer.
    sleep 300
}

main
