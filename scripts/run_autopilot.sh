#!/usr/bin/env bash
# scripts/run_autopilot.sh
# Top-level autopilot orchestrator. Spawns one tmux window per GPU lane,
# each running scripts/run_loop.sh under a pinned SAE_LANE + GPU remap.
# Re-spawns dead workers, periodically aggregates per-lane state into the
# legacy experiments/state.json, and triggers the post-matrix analysis lane
# when conditions hold.
#
# Usage:
#   bash scripts/run_autopilot.sh           # spawn workers, supervise forever
#   bash scripts/run_autopilot.sh stop      # kill all lane workers, leave session
#   bash scripts/run_autopilot.sh status    # one-shot health summary
#
# All five GPUs are SAE-owned during the cv-semseg freeze (2026-04-27 onward).
# Lane → CUDA mapping mirrors the post-freeze gpu_policy.sh. Lane-2 is split
# across both 3090s via two workers (2a→CUDA 1, 2b→CUDA 3).

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION="sae-loop"
SUPERVISOR_LOG="experiments/logs/orchestrator.log"
mkdir -p experiments/logs experiments/.row_claims

# Sprint-mode tunables. Aggressive cooldown so transient failures retry fast,
# deadline-gating disabled (cv-semseg-freeze sprint), retries unchanged.
export SAE_IGNORE_DEADLINE=1
export SAE_RETRY_COOLDOWN_SECONDS="${SAE_RETRY_COOLDOWN_SECONDS:-900}"   # 15 min
export SAE_MAX_ATTEMPTS="${SAE_MAX_ATTEMPTS:-10}"

# Lane definition table:
#   lane_label   SAE_LANE   GPU_REMAP env=value     description
LANES=(
    "lane-0|0|SAE_GPU_REMAP_0=2|meta-SAE primary (yaml '0' → CUDA 2 / 4080)"
    "lane-1|1|SAE_GPU_REMAP_1=0|meta-SAE parallel seed (yaml '1' → CUDA 0 / 4080)"
    "lane-2a|2|SAE_GPU_REMAP_2=1|24 GB jobs slot A (yaml '2' → CUDA 1 / 3090)"
    "lane-2b|2|SAE_GPU_REMAP_2=3|24 GB jobs slot B (yaml '2' → CUDA 3 / 3090)"
    "lane-4|4|SAE_GPU_REMAP_4=4|analysis & flat-SAE control (yaml '4' → CUDA 4 / 4060 Ti)"
)

ANALYSIS_LANE="analysis"  # special: not run_loop, runs scripts/run_analysis.sh

orch_log() {
    local stamp
    stamp=$(date -Iseconds)
    printf "[%s] %s\n" "$stamp" "$*" | tee -a "$SUPERVISOR_LOG"
}

ensure_session() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-session -d -s "$SESSION" -n supervisor
        orch_log "created tmux session $SESSION"
    fi
}

window_alive() {
    local win="$1"
    tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$win"
}

worker_busy() {
    # A worker is "busy" if its window has any python child running a training entrypoint.
    local win="$1"
    local pid
    pid=$(tmux list-panes -t "${SESSION}:${win}" -F '#{pane_pid}' 2>/dev/null | head -1)
    [[ -z "$pid" ]] && return 1
    pgrep -P "$pid" -f "train_meta_sae|train_level0_batchtopk|train_flat_sae|train_null_sae|run_autointerp" >/dev/null 2>&1
}

spawn_lane() {
    local label="$1"
    local lane_value="$2"
    local remap_env="$3"
    local desc="$4"

    if window_alive "$label"; then
        orch_log "lane $label already alive, skipping spawn"
        return 0
    fi

    tmux new-window -t "$SESSION" -n "$label" -d
    # Compose the launch command. Env vars live in the lane's shell so each
    # worker has independent SAE_LANE and SAE_GPU_REMAP_X without collision.
    local cmd
    cmd=$(printf 'export SAE_LANE=%q SAE_LANE_LABEL=%q %s SAE_IGNORE_DEADLINE=%q SAE_RETRY_COOLDOWN_SECONDS=%q SAE_MAX_ATTEMPTS=%q; cd %q; while true; do bash scripts/run_loop.sh 2>&1 | tee -a experiments/logs/lane-%s.log; echo "[lane=%s] run_loop exited rc=$? at $(date -Iseconds); restarting in 30s" | tee -a experiments/logs/lane-%s.log; sleep 30; done' \
        "$lane_value" "$label" "$remap_env" "$SAE_IGNORE_DEADLINE" "$SAE_RETRY_COOLDOWN_SECONDS" "$SAE_MAX_ATTEMPTS" "$REPO_ROOT" "$label" "$label" "$label")
    tmux send-keys -t "${SESSION}:${label}" "$cmd" C-m
    orch_log "spawned $label ($desc)"
}

spawn_analysis() {
    local label="$ANALYSIS_LANE"
    if window_alive "$label"; then
        return 0
    fi
    tmux new-window -t "$SESSION" -n "$label" -d
    local cmd
    cmd=$(printf 'export SAE_IGNORE_DEADLINE=1; cd %q; while true; do bash scripts/run_analysis.sh 2>&1 | tee -a experiments/logs/analysis.log; echo "[analysis] loop exited rc=$? at $(date -Iseconds); sleeping 600s before next pass" | tee -a experiments/logs/analysis.log; sleep 600; done' "$REPO_ROOT")
    tmux send-keys -t "${SESSION}:${label}" "$cmd" C-m
    orch_log "spawned analysis lane (post-matrix pipeline)"
}

stop_all() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "session $SESSION does not exist"; return 0
    fi
    for entry in "${LANES[@]}"; do
        IFS='|' read -r label _ _ _ <<<"$entry"
        if window_alive "$label"; then
            tmux send-keys -t "${SESSION}:${label}" C-c C-c 2>/dev/null || true
            sleep 1
            tmux kill-window -t "${SESSION}:${label}" 2>/dev/null || true
            orch_log "stopped $label"
        fi
    done
    if window_alive "$ANALYSIS_LANE"; then
        tmux send-keys -t "${SESSION}:${ANALYSIS_LANE}" C-c C-c 2>/dev/null || true
        sleep 1
        tmux kill-window -t "${SESSION}:${ANALYSIS_LANE}" 2>/dev/null || true
        orch_log "stopped $ANALYSIS_LANE"
    fi
    # Kill any orphaned training processes that may still hold GPU memory.
    pkill -TERM -f "train_meta_sae|train_level0_batchtopk|train_flat_sae|train_null_sae|run_autointerp" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "train_meta_sae|train_level0_batchtopk|train_flat_sae|train_null_sae|run_autointerp" 2>/dev/null || true
    # Also kill any lingering bash run_loop wrappers.
    pkill -TERM -f "bash scripts/run_loop.sh" 2>/dev/null || true
    rm -f experiments/.row_claims/*.claim 2>/dev/null || true
    orch_log "all lanes stopped, claim files cleared"
}

status_report() {
    echo "session: $SESSION"
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  (not running)"
        return 0
    fi
    echo "windows:"
    tmux list-windows -t "$SESSION" -F '  #W (pid=#{pane_pid})' 2>/dev/null
    echo "active claims:"
    ls -la experiments/.row_claims/ 2>/dev/null | tail -n +2
    echo "GPU snapshot:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "current training procs:"
    pgrep -af "train_meta_sae|train_level0_batchtopk|train_flat_sae|train_null_sae|run_autointerp" 2>/dev/null | head
}

aggregate_state() {
    uv run python scripts/aggregate_state.py 2>>"$SUPERVISOR_LOG" || true
}

supervise() {
    ensure_session
    orch_log "autopilot supervisor up (pid=$$, session=$SESSION)"
    for entry in "${LANES[@]}"; do
        IFS='|' read -r label lane_value remap_env desc <<<"$entry"
        spawn_lane "$label" "$lane_value" "$remap_env" "$desc"
        sleep 1
    done
    spawn_analysis

    while true; do
        for entry in "${LANES[@]}"; do
            IFS='|' read -r label lane_value remap_env desc <<<"$entry"
            if ! window_alive "$label"; then
                orch_log "WARN lane $label window dead — respawning"
                spawn_lane "$label" "$lane_value" "$remap_env" "$desc"
            fi
        done
        if ! window_alive "$ANALYSIS_LANE"; then
            orch_log "WARN analysis lane dead — respawning"
            spawn_analysis
        fi
        aggregate_state

        # Completion check: if PRE_PAPER_COMPLETE sentinel exists, we still
        # supervise (to keep the heartbeat alive) but stop logging "alive"
        # at the same cadence.
        if [[ -f experiments/PRE_PAPER_COMPLETE ]]; then
            sleep 1800
        else
            sleep 60
        fi
    done
}

case "${1:-supervise}" in
    supervise|run|"") supervise ;;
    stop) stop_all ;;
    status) status_report ;;
    *) echo "usage: $0 {supervise|stop|status}"; exit 2 ;;
esac
