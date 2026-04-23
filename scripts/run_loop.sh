#!/usr/bin/env bash
# scripts/run_loop.sh
# Top-level runner. Iterates EXPERIMENTS.yaml in dependency order, dispatches to
# the appropriate Python entry point, handles gates, commits, ntfys. Designed to
# run inside tmux window 'runner'. Robust to kill/restart.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STATE_FILE="experiments/state.json"
RESULTS_TSV="experiments/results.tsv"
GATES_TSV="experiments/gates.tsv"
LOG_DIR="experiments/logs"
PAUSE_FILE="experiments/PAUSE"
HALT_FILE="experiments/HALT"
NTFY_TOPIC="sae-wanga-research"

mkdir -p "$LOG_DIR" experiments/artifacts

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf "timestamp\texperiment_id\tstatus\tbase_model\tlevel0_arch\tdepth\tseed\twidth\tvariance_explained\tpwmcc\tmmcs\tpwmcc_null_mean\tpwmcc_null_std\tdead_latent_fraction\tgpu_hours\tcommit_sha\twandb_run_url\tnotes\n" > "$RESULTS_TSV"
fi

if [[ ! -f "$GATES_TSV" ]]; then
    printf "timestamp\texperiment_id\tmetric\tobserved_value\tthreshold\taction\ttriggered\n" > "$GATES_TSV"
fi

init_state() {
    local pid=$$
    local now
    now=$(date -Iseconds)
    python3 - "$STATE_FILE" "$pid" "$now" <<'PY'
import json, sys, os
path, pid, now = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if os.path.exists(path):
    with open(path) as f:
        state = json.load(f)
else:
    state = {}
state.update({
    "runner_status": "running",
    "pid": pid,
    "started_at": state.get("started_at", now),
    "last_heartbeat": now,
    "current_experiment_id": state.get("current_experiment_id"),
    "most_recent_gate_outcome": state.get("most_recent_gate_outcome", {}),
    "blockers": state.get("blockers", []),
})
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(state, f, indent=2)
os.replace(tmp, path)
PY
}

write_state_field() {
    local key="$1"
    local value="$2"
    python3 - "$STATE_FILE" "$key" "$value" <<'PY'
import json, sys, os
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    state = json.load(f)
try:
    parsed = json.loads(value)
    state[key] = parsed
except Exception:
    state[key] = value
state["last_heartbeat"] = __import__("datetime").datetime.now().astimezone().isoformat()
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(state, f, indent=2)
os.replace(tmp, path)
PY
}

ntfy_send() {
    local priority="$1"
    local title="$2"
    local msg="$3"
    bash scripts/ntfy_send.sh "$priority" "$title" "$msg" || true
}

select_next_pending() {
    python3 - <<'PY'
import yaml, json, sys, os, datetime
with open("EXPERIMENTS.yaml") as f:
    matrix = yaml.safe_load(f)
with open("experiments/results.tsv") as f:
    lines = f.read().strip().split("\n")
header = lines[0].split("\t")
complete = set()
failed_count = {}
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split("\t")
    rec = dict(zip(header, parts))
    if rec.get("status") == "ok":
        complete.add(rec["experiment_id"])
    elif rec.get("status") == "failed":
        failed_count[rec["experiment_id"]] = failed_count.get(rec["experiment_id"], 0) + 1

state = {}
if os.path.exists("experiments/state.json"):
    with open("experiments/state.json") as f:
        state = json.load(f)
skipped = set(state.get("skipped_by_gate", []))

today = datetime.date.today()
deadline = datetime.date(2026, 5, 5)
days_until_deadline = (deadline - today).days

rows = matrix["experiments"]

def dep_ok(exp):
    for d in exp.get("dependencies", []):
        if d not in complete:
            return False
    return True

eligible = []
for e in rows:
    if e["id"] in complete or e["id"] in skipped:
        continue
    if failed_count.get(e["id"], 0) >= 1:
        continue
    cond = e.get("conditional") or {}
    if cond.get("stretch_if_time_permits"):
        nonstretch_ids = {x["id"] for x in rows if not (x.get("conditional") or {}).get("stretch_if_time_permits")}
        if not nonstretch_ids.issubset(complete | skipped):
            continue
        if days_until_deadline < 4:
            continue
    if not dep_ok(e):
        continue
    eligible.append(e)

if not eligible:
    print("NONE")
    sys.exit(0)

def priority_key(e):
    has_halt = any(g.get("action") == "halt_and_notify" for g in e.get("decision_gates", []))
    return (0 if has_halt else 1, e.get("depth", 0), rows.index(e))

eligible.sort(key=priority_key)
chosen = eligible[0]
print(chosen["id"])
PY
}

get_experiment_field() {
    local exp_id="$1"
    local field="$2"
    python3 - "$exp_id" "$field" <<'PY'
import yaml, sys, json
exp_id, field = sys.argv[1], sys.argv[2]
with open("EXPERIMENTS.yaml") as f:
    matrix = yaml.safe_load(f)
for e in matrix["experiments"]:
    if e["id"] == exp_id:
        val = e.get(field, "")
        if isinstance(val, (dict, list)):
            print(json.dumps(val))
        else:
            print(val if val is not None else "")
        sys.exit(0)
sys.exit(1)
PY
}

dispatch_entrypoint() {
    local exp_id="$1"
    local level0_source
    level0_source=$(get_experiment_field "$exp_id" "level0_source")
    local level0_arch
    level0_arch=$(get_experiment_field "$exp_id" "level0_arch")
    local depth
    depth=$(get_experiment_field "$exp_id" "depth")

    if [[ "$exp_id" == "autointerp_all" ]]; then
        echo "src.analysis.run_autointerp"
        return
    fi
    if [[ "$exp_id" == "simplestories_stretch" ]]; then
        echo "src.training.train_level0_batchtopk"
        return
    fi
    if [[ "$level0_source" == "random_gaussian" ]]; then
        echo "src.training.train_null_sae"
        return
    fi
    if [[ "$level0_source" == "flat_sae_on_activations" ]]; then
        echo "src.training.train_flat_sae"
        return
    fi
    if [[ "$level0_source" == "train_from_scratch" && "$depth" == "0" ]]; then
        echo "src.training.train_level0_batchtopk"
        return
    fi
    echo "src.training.train_meta_sae"
}

evaluate_gates_and_apply() {
    local exp_id="$1"
    local metrics_file="$2"
    python3 scripts/evaluate_gates.py --experiment-id "$exp_id" --metrics-file "$metrics_file" --state-file "$STATE_FILE" --gates-tsv "$GATES_TSV"
    return $?
}

commit_artifacts() {
    local exp_id="$1"
    local ve="${2:-NA}"
    local pwmcc="${3:-NA}"
    git add experiments/artifacts/"$exp_id"/ experiments/results.tsv experiments/gates.tsv experiments/state.json experiments/logs/ 2>/dev/null || true
    git commit -m "[SAE-LOOP] $exp_id complete VE=$ve PW-MCC=$pwmcc" --no-gpg-sign 2>&1 | tail -20 || true
}

run_one() {
    local exp_id="$1"
    local gpu_pref
    gpu_pref=$(get_experiment_field "$exp_id" "gpu_preference")
    local entrypoint
    entrypoint=$(dispatch_entrypoint "$exp_id")
    local log_file="$LOG_DIR/${exp_id}_$(date +%Y%m%d_%H%M%S).log"

    write_state_field "current_experiment_id" "$exp_id"
    ntfy_send "low" "[SAE start] $exp_id" "entrypoint=$entrypoint gpu=$gpu_pref"

    local backoffs=(30 300 1800)
    local attempt=0
    local success=0
    local start_epoch
    start_epoch=$(date +%s)

    while [[ $attempt -lt 3 ]]; do
        attempt=$((attempt + 1))
        echo "=== attempt $attempt for $exp_id ===" >> "$log_file"
        if CUDA_VISIBLE_DEVICES="$gpu_pref" python3 -m "$entrypoint" --experiment-id "$exp_id" 2>&1 | tee -a "$log_file"; then
            success=1
            break
        fi
        local rc=${PIPESTATUS[0]}
        echo "attempt $attempt failed rc=$rc" >> "$log_file"
        if grep -qiE "CUDA error|cublas|NCCL|out of memory|OutOfMemoryError" "$log_file"; then
            nvidia-smi >> "$log_file" 2>&1 || true
        fi
        if [[ $attempt -lt 3 ]]; then
            local wait=${backoffs[$((attempt - 1))]}
            echo "backing off $wait seconds" >> "$log_file"
            sleep "$wait"
        fi
    done

    local end_epoch
    end_epoch=$(date +%s)
    local elapsed=$((end_epoch - start_epoch))

    if [[ $success -ne 1 ]]; then
        ntfy_send "high" "[SAE ERROR] $exp_id" "failed after 3 retries; see $log_file"
        local now
        now=$(date -Iseconds)
        printf "%s\t%s\tfailed\t\t\t\t\t\t\t\t\t\t\t\t%d\t\t\tretry_exhausted\n" "$now" "$exp_id" "$elapsed" >> "$RESULTS_TSV"
        return 1
    fi

    local metrics_file="experiments/artifacts/$exp_id/metrics.tsv"
    if [[ ! -f "$metrics_file" ]]; then
        ntfy_send "high" "[SAE ERROR] $exp_id" "entrypoint exited 0 but no metrics.tsv"
        local now
        now=$(date -Iseconds)
        printf "%s\t%s\tfailed\t\t\t\t\t\t\t\t\t\t\t\t%d\t\t\tno_metrics_file\n" "$now" "$exp_id" "$elapsed" >> "$RESULTS_TSV"
        return 1
    fi

    local ve pwmcc
    ve=$(awk -F'\t' 'NR==2 {print $1}' "$metrics_file" 2>/dev/null || echo "NA")
    pwmcc=$(awk -F'\t' 'NR==2 {print $2}' "$metrics_file" 2>/dev/null || echo "NA")

    evaluate_gates_and_apply "$exp_id" "$metrics_file"
    local gate_rc=$?

    commit_artifacts "$exp_id" "$ve" "$pwmcc"
    local sha
    sha=$(git rev-parse --short HEAD 2>/dev/null || echo "NA")

    local now
    now=$(date -Iseconds)
    local base_model depth seed width
    base_model=$(get_experiment_field "$exp_id" "base_model")
    depth=$(get_experiment_field "$exp_id" "depth")
    seed=$(get_experiment_field "$exp_id" "seed")
    width=$(get_experiment_field "$exp_id" "width")
    local level0_arch
    level0_arch=$(get_experiment_field "$exp_id" "level0_arch")
    printf "%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\t%d\t%s\t\t\n" "$now" "$exp_id" "$base_model" "$level0_arch" "$depth" "$seed" "$width" "$ve" "$pwmcc" "" "$elapsed" "$sha" >> "$RESULTS_TSV"

    ntfy_send "low" "[SAE done] $exp_id" "VE=$ve PW-MCC=$pwmcc elapsed=${elapsed}s"

    if [[ $gate_rc -eq 42 ]]; then
        ntfy_send "urgent" "[SAE HALT] $exp_id" "halt_and_notify gate triggered; runner stopping"
        write_state_field "runner_status" "halted"
        return 42
    fi

    return 0
}

main() {
    init_state
    ntfy_send "default" "[SAE milestone] runner starting" "pid=$$ $(date -Iseconds)"

    while true; do
        if [[ -f "$HALT_FILE" ]]; then
            ntfy_send "high" "[SAE runner halted by HALT file]" ""
            write_state_field "runner_status" "halted"
            break
        fi
        if [[ -f "$PAUSE_FILE" ]]; then
            write_state_field "runner_status" "paused"
            sleep 30
            continue
        fi
        write_state_field "runner_status" "running"

        local next
        next=$(select_next_pending)
        if [[ "$next" == "NONE" ]]; then
            ntfy_send "default" "[SAE milestone] matrix complete" "all eligible experiments have status=complete or skipped_by_gate"
            write_state_field "runner_status" "idle"
            break
        fi

        run_one "$next"
        local rc=$?
        if [[ $rc -eq 42 ]]; then
            break
        fi

        sleep 5
    done
}

main "$@"
