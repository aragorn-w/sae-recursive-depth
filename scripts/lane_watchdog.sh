#!/usr/bin/env bash
# scripts/lane_watchdog.sh
# Detects and kills zombie SAE training processes. Conservative heuristic:
# all three must hold to flag a process as zombied:
#   1. Elapsed wall time > MIN_ELAPSED_S (past boot/load phase)
#   2. %CPU < CPU_THRESHOLD on two samples 30 s apart
#   3. curves.tsv missing OR empty OR mtime > CURVES_STALE_S old
# On confirmation: ntfy, SIGTERM, sleep 30, SIGKILL if still alive.
# Supervisor (run_autopilot.sh) respawns the lane.
#
# Knobs (env, with defaults):
#   SAE_WATCHDOG_INTERVAL=300      pass interval in seconds
#   SAE_WATCHDOG_MIN_ELAPSED=600   skip processes younger than this
#   SAE_WATCHDOG_CPU_THRESHOLD=1.0 %CPU below which a process is "idle"
#   SAE_WATCHDOG_CURVES_STALE=1800 curves.tsv mtime older than this = zombie
#   SAE_WATCHDOG_DRY_RUN=0         set 1 to log without killing

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WATCHDOG_LOG="experiments/logs/watchdog.log"
NTFY="${REPO_ROOT}/scripts/ntfy_send.sh"
mkdir -p experiments/logs

SLEEP_BETWEEN="${SAE_WATCHDOG_INTERVAL:-300}"
MIN_ELAPSED_S="${SAE_WATCHDOG_MIN_ELAPSED:-600}"
CPU_THRESHOLD="${SAE_WATCHDOG_CPU_THRESHOLD:-1.0}"
CURVES_STALE_S="${SAE_WATCHDOG_CURVES_STALE:-1800}"
DRY_RUN="${SAE_WATCHDOG_DRY_RUN:-0}"

log() {
    local stamp
    stamp=$(date -Iseconds)
    printf "[%s] %s\n" "$stamp" "$*" | tee -a "$WATCHDOG_LOG"
}

ntfy_kill() {
    local exp_id="$1" pid="$2" elapsed="$3" reason="$4"
    if [[ -x "$NTFY" ]]; then
        bash "$NTFY" high \
            "[WATCHDOG] killed zombie ${exp_id}" \
            "pid=${pid} elapsed=${elapsed}s reason=${reason}; supervisor will respawn lane" \
            "warning,skull" 2>/dev/null || true
    fi
}

extract_exp_id() {
    local pid="$1"
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null) || return 0
    awk -v c="$cmd" 'BEGIN{
        n=split(c, a, " ");
        for (i=1; i<=n; i++) if (a[i]=="--experiment-id" && (i+1)<=n) { print a[i+1]; exit }
    }'
}

sample_cpu() {
    local pid="$1"
    ps -p "$pid" -o %cpu= 2>/dev/null | awk 'NR==1{print $1+0}'
}

is_below() {
    awk -v c="$1" -v t="$2" 'BEGIN{ exit !(c < t) }'
}

watchdog_pass() {
    # Match only the child venv-python worker, not the `uv run` parent shim.
    # The parent always sits at %CPU=0 while waiting on the child, so matching
    # it would false-positive every healthy lane. Pattern requires the
    # `.venv/bin/python` prefix and the training module suffix.
    local pids
    pids=$(pgrep -f '/\.venv/bin/python[0-9.]* -m src\.training\.train_(level0_batchtopk|meta_sae|flat_sae|null_sae)' 2>/dev/null || true)
    [[ -z "$pids" ]] && return 0

    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        kill -0 "$pid" 2>/dev/null || continue

        local elapsed
        elapsed=$(ps -p "$pid" -o etimes= 2>/dev/null | awk '{print $1+0}')
        [[ -z "$elapsed" ]] && continue
        if (( elapsed < MIN_ELAPSED_S )); then
            continue
        fi

        local cpu1
        cpu1=$(sample_cpu "$pid")
        is_below "$cpu1" "$CPU_THRESHOLD" || continue

        sleep 30
        kill -0 "$pid" 2>/dev/null || continue
        local cpu2
        cpu2=$(sample_cpu "$pid")
        is_below "$cpu2" "$CPU_THRESHOLD" || continue

        local exp_id
        exp_id=$(extract_exp_id "$pid")
        [[ -z "$exp_id" ]] && continue
        local curves="experiments/artifacts/${exp_id}/curves.tsv"

        local reason=""
        if [[ ! -f "$curves" ]]; then
            reason="curves.tsv missing"
        elif [[ ! -s "$curves" ]]; then
            reason="curves.tsv empty"
        else
            local mtime now stale
            mtime=$(stat -c '%Y' "$curves" 2>/dev/null || echo 0)
            now=$(date +%s)
            stale=$(( now - mtime ))
            if (( stale > CURVES_STALE_S )); then
                reason="curves.tsv stale ${stale}s"
            fi
        fi

        if [[ -z "$reason" ]]; then
            continue
        fi

        log "ZOMBIE pid=${pid} exp=${exp_id} elapsed=${elapsed}s cpu=${cpu1}/${cpu2} reason=${reason}"
        if [[ "$DRY_RUN" == "1" ]]; then
            log "DRY_RUN: not killing"
            continue
        fi
        ntfy_kill "$exp_id" "$pid" "$elapsed" "$reason"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 30
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
            log "SIGKILL pid=${pid} (SIGTERM ignored)"
        else
            log "exited cleanly pid=${pid}"
        fi
    done <<< "$pids"
}

main() {
    log "lane_watchdog starting interval=${SLEEP_BETWEEN}s min_elapsed=${MIN_ELAPSED_S}s cpu<${CPU_THRESHOLD} curves_stale>${CURVES_STALE_S}s dry_run=${DRY_RUN}"
    while true; do
        watchdog_pass || log "pass error rc=$?"
        sleep "$SLEEP_BETWEEN"
    done
}

main "$@"
