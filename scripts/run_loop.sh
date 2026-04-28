#!/usr/bin/env bash
# scripts/run_loop.sh
# Top-level runner. Iterates EXPERIMENTS.yaml in dependency order, dispatches to
# the appropriate Python entry point, handles gates, commits, ntfys. Designed to
# run inside tmux window 'runner'. Robust to kill/restart.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# GPU policy: translates EXPERIMENTS.yaml `gpu_preference` values to the
# current host's actual CUDA indices. Edit scripts/gpu_policy.sh (not
# EXPERIMENTS.yaml) when host layout or contention changes.
# shellcheck source=scripts/gpu_policy.sh
source "$REPO_ROOT/scripts/gpu_policy.sh"

# Lane-aware execution. SAE_LANE filters select_next_pending to only consider
# rows whose gpu_preference matches. Empty lane (legacy invocation) means
# match-everything, keeping single-runner behavior. Multiple workers running
# in parallel each pin SAE_LANE; row-level claim files prevent collision.
SAE_LANE="${SAE_LANE:-}"
SAE_LANE_LABEL="${SAE_LANE_LABEL:-${SAE_LANE:-default}}"
export SAE_LANE SAE_LANE_LABEL

STATE_FILE="experiments/state.lane.${SAE_LANE_LABEL}.json"
LEGACY_STATE_FILE="experiments/state.json"
RESULTS_TSV="experiments/results.tsv"
GATES_TSV="experiments/gates.tsv"
LOG_DIR="experiments/logs"
PAUSE_FILE="experiments/PAUSE"
HALT_FILE="experiments/HALT"
NTFY_TOPIC="sae-wanga-research"
CLAIM_DIR="experiments/.row_claims"
GIT_LOCK="experiments/.git_commit.lock"

mkdir -p "$LOG_DIR" experiments/artifacts "$CLAIM_DIR"

# In-session skip set. run_one appends an exp_id here on EXIT_SCAFFOLD_STUB
# (99); select_next_pending honors it. Cleared on every fresh invocation so
# scaffold stubs can be re-attempted after real bodies land without
# restarting anything heavier.
SESSION_SKIPFILE="${SESSION_SKIPFILE:-$(mktemp -t sae_session_skips.XXXXXX)}"
export SESSION_SKIPFILE
: > "$SESSION_SKIPFILE"
trap 'rm -f "$SESSION_SKIPFILE"' EXIT

EXIT_SCAFFOLD_STUB=99

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
    uv run python - "$STATE_FILE" "$pid" "$now" <<'PY'
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
    uv run python - "$STATE_FILE" "$key" "$value" <<'PY'
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

heartbeat() {
    # Bumps last_heartbeat without touching other fields. Used inside the
    # retry/backoff loop in run_one so lane state stays fresh while a row
    # is in inter-attempt sleep.
    uv run python - "$STATE_FILE" <<'PY' 2>/dev/null || true
import json, sys, os, datetime
p = sys.argv[1]
with open(p) as f:
    s = json.load(f)
s["last_heartbeat"] = datetime.datetime.now().astimezone().isoformat()
t = p + ".tmp"
with open(t, "w") as f:
    json.dump(s, f, indent=2)
os.replace(t, p)
PY
}

_failure_signature() {
    # Extract the dominant error class from the most recent "=== attempt N"
    # block in log_file. Empty if no recognizable signature. Used by
    # run_one to detect identical-traceback structural failures and
    # short-circuit further retries.
    local log="$1"
    awk '/^=== attempt /{block=""} {block=block"\n"$0} END{print block}' "$log" 2>/dev/null \
        | grep -oE '(OutOfMemoryError|CUDA error: [A-Za-z_]+|cublas[A-Za-z_]+|NCCL [A-Z_]+|BrokenPipeError|RuntimeError: [A-Za-z_]+|FileNotFoundError|ImportError)' \
        | tail -1
}

ntfy_send() {
    local priority="$1"
    local title="$2"
    local msg="$3"
    bash scripts/ntfy_send.sh "$priority" "$title" "$msg" || true
}

select_next_pending() {
    uv run python - <<'PY'
import yaml, json, sys, os, datetime, errno
SAE_LANE = os.environ.get("SAE_LANE", "")
# Optional spillover: comma-separated gpu_preference values this lane will
# also accept when its primary queue is empty, capped at
# SAE_LANE_FALLBACK_MAX_HOURS estimated_gpu_hours per row. Used to keep
# small-VRAM lanes (e.g. lane-4 / 4060 Ti) productive once their primary
# queue drains.
SAE_LANE_FALLBACK = [x.strip() for x in os.environ.get("SAE_LANE_FALLBACK", "").split(",") if x.strip()]
try:
    SAE_LANE_FALLBACK_MAX_HOURS = float(os.environ.get("SAE_LANE_FALLBACK_MAX_HOURS", "0.5"))
except ValueError:
    SAE_LANE_FALLBACK_MAX_HOURS = 0.5
CLAIM_DIR = "experiments/.row_claims"

def _claim_active(exp_id):
    """Return True if another worker holds an in-flight claim on this row."""
    p = os.path.join(CLAIM_DIR, f"{exp_id}.claim")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = {}
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    d[k.strip()] = v.strip()
        pid = int(d.get("pid", "0"))
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    except Exception:
        return False

with open("EXPERIMENTS.yaml") as f:
    matrix = yaml.safe_load(f)
with open("experiments/results.tsv") as f:
    lines = f.read().strip().split("\n")
header = lines[0].split("\t")
# Per SPEC §8.2 ("the later timestamp wins when querying"), each experiment
# is judged by the LATEST status row in results.tsv. This lets a row
# transition complete -> stale -> complete across recipe revisions without
# breaking the append-only contract.
#
# Auto-retry policy: a row whose latest status is "failed" becomes eligible
# again after RETRY_COOLDOWN_SECONDS, up to MAX_ATTEMPTS total. This means
# transient failures (OOM, NCCL blip, HF timeout) self-heal without ntfy,
# and only genuinely-broken configs accumulate enough failed rows to give up.
RETRY_COOLDOWN_SECONDS = int(os.environ.get("SAE_RETRY_COOLDOWN_SECONDS", "86400"))
MAX_ATTEMPTS = int(os.environ.get("SAE_MAX_ATTEMPTS", "10"))

# Pre-pass: find the most recent stale_* marker per row. A stale_* row is the
# operator's signal that a prior failure cause has been resolved (e.g. gpu_policy
# fix); failures predating it should not count toward MAX_ATTEMPTS, otherwise a
# requeue gives at most one retry before wedging in failed_blocked again.
stale_marker_ts = {}
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split("\t")
    rec = dict(zip(header, parts))
    eid = rec.get("experiment_id", "")
    ts = rec.get("timestamp", "")
    st = rec.get("status", "")
    if eid and st.startswith("stale_"):
        if eid not in stale_marker_ts or ts > stale_marker_ts[eid]:
            stale_marker_ts[eid] = ts

latest = {}  # exp_id -> (timestamp, status)
attempts = {}  # exp_id -> count of "failed" rows after the most recent stale_* marker
last_failed_ts = {}  # exp_id -> latest "failed" timestamp (post-stale-marker)
for line in lines[1:]:
    if not line.strip():
        continue
    parts = line.split("\t")
    rec = dict(zip(header, parts))
    eid = rec.get("experiment_id", "")
    ts = rec.get("timestamp", "")
    st = rec.get("status", "")
    if not eid:
        continue
    prev = latest.get(eid)
    if prev is None or ts > prev[0]:
        latest[eid] = (ts, st)
    if st == "failed":
        if eid in stale_marker_ts and ts <= stale_marker_ts[eid]:
            continue
        attempts[eid] = attempts.get(eid, 0) + 1
        if eid not in last_failed_ts or ts > last_failed_ts[eid]:
            last_failed_ts[eid] = ts
complete = {eid for eid, (_, st) in latest.items() if st == "ok"}
def _parse(ts):
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return None
now = datetime.datetime.now().astimezone()
failed_blocked = set()  # rows with latest=failed and not yet retry-eligible
for eid, (ts, st) in latest.items():
    # failed_structural: identical-traceback short-circuit from run_one.
    # These never auto-retry; they need an operator stale_* marker to
    # become eligible again.
    if st == "failed_structural":
        failed_blocked.add(eid)
        continue
    if st != "failed":
        continue
    if attempts.get(eid, 0) >= MAX_ATTEMPTS:
        failed_blocked.add(eid)
        continue
    last_ts = _parse(last_failed_ts.get(eid, ""))
    if last_ts is not None:
        try:
            elapsed = (now - last_ts).total_seconds()
        except TypeError:
            elapsed = RETRY_COOLDOWN_SECONDS
        if elapsed < RETRY_COOLDOWN_SECONDS:
            failed_blocked.add(eid)

state = {}
if os.path.exists("experiments/state.json"):
    with open("experiments/state.json") as f:
        state = json.load(f)
skipped = set(state.get("skipped_by_gate", []))

# Backfill skipped from durable gates.tsv. Per-lane skipped_by_gate lives
# only in lane state files and gets reset on lane respawn; the gates.tsv
# is the append-only ground truth for what gates have ever fired. Reading
# it here makes the skip set immune to lane state churn.
gates_tsv = "experiments/gates.tsv"
if os.path.exists(gates_tsv):
    exp_by_id = {e["id"]: e for e in matrix["experiments"]}
    with open(gates_tsv) as gf:
        ghead = None
        for gline in gf:
            cols = gline.rstrip("\n").split("\t")
            if ghead is None:
                ghead = cols
                continue
            if len(cols) < len(ghead):
                continue
            grec = dict(zip(ghead, cols))
            if grec.get("triggered") != "1":
                continue
            action = grec.get("action", "")
            if action not in ("skip_depth", "skip_experiment"):
                continue
            geid = grec.get("experiment_id", "")
            gexp = exp_by_id.get(geid)
            if gexp is None:
                continue
            if action == "skip_experiment":
                skipped.add(geid)
            elif action == "skip_depth":
                lineage = (gexp["base_model"], gexp["level0_arch"], gexp["seed"])
                gdepth = gexp.get("depth", 0)
                for r in matrix["experiments"]:
                    if (r["base_model"], r["level0_arch"], r["seed"]) == lineage \
                       and r.get("depth", 0) > gdepth:
                        skipped.add(r["id"])

# In-session skip set: ids that returned EXIT_SCAFFOLD_STUB during this
# runner invocation. Keeps the outer loop from picking the same stub twice.
session_skip = set()
skipfile = os.environ.get("SESSION_SKIPFILE")
if skipfile and os.path.exists(skipfile):
    with open(skipfile) as f:
        session_skip = {line.strip() for line in f if line.strip()}

today = datetime.date.today()
deadline = datetime.date(2026, 5, 5)
days_until_deadline = (deadline - today).days

# Hard freeze: after FREEZE_AT (America/Denver) no new rows are picked.
# In-flight runs finish naturally; the loop's next select_next_pending
# returns NONE and the runner exits its main loop.
try:
    from zoneinfo import ZoneInfo
    _now_local = datetime.datetime.now(tz=ZoneInfo("America/Denver"))
    FREEZE_AT = datetime.datetime(2026, 5, 4, 17, 0, tzinfo=ZoneInfo("America/Denver"))
except Exception:
    # Fallback if zoneinfo unavailable for any reason: use naive UTC and
    # subtract 6h to approximate MDT. Better to over-freeze than miss it.
    _now_local = datetime.datetime.utcnow() - datetime.timedelta(hours=6)
    FREEZE_AT = datetime.datetime(2026, 5, 4, 17, 0)
if _now_local >= FREEZE_AT and os.environ.get("SAE_IGNORE_DEADLINE", "0") != "1":
    print("NONE")
    sys.exit(0)

rows = matrix["experiments"]

def dep_ok(exp):
    for d in exp.get("dependencies", []):
        if d not in complete:
            return False
    return True

eligible = []
fallback_eligible = []  # rows accepted only via SAE_LANE_FALLBACK
for e in rows:
    if e["id"] in complete or e["id"] in skipped or e["id"] in session_skip:
        continue
    if e["id"] in failed_blocked:
        continue
    pref = str(e.get("gpu_preference", ""))
    is_primary = (not SAE_LANE) or pref == SAE_LANE
    is_fallback = (
        not is_primary
        and SAE_LANE_FALLBACK
        and pref in SAE_LANE_FALLBACK
        and (e.get("estimated_gpu_hours") or 0) <= SAE_LANE_FALLBACK_MAX_HOURS
    )
    if not (is_primary or is_fallback):
        continue
    if _claim_active(e["id"]):
        continue
    cond = e.get("conditional") or {}
    if cond.get("stretch_if_time_permits"):
        nonstretch_ids = {x["id"] for x in rows if not (x.get("conditional") or {}).get("stretch_if_time_permits")}
        if not nonstretch_ids.issubset(complete | skipped):
            continue
        # Stretch experiments are gated on calendar deadline by default.
        # When the user has explicitly disabled deadline-gating (autopilot
        # sprint mode, 2026-04-27), they run as soon as preconditions hold.
        if os.environ.get("SAE_IGNORE_DEADLINE", "0") != "1":
            if days_until_deadline < 4:
                continue
    if not dep_ok(e):
        continue
    if is_primary:
        eligible.append(e)
    else:
        fallback_eligible.append(e)

if not eligible and not fallback_eligible:
    print("NONE")
    sys.exit(0)

def priority_key(e):
    has_halt = any(g.get("action") == "halt_and_notify" for g in e.get("decision_gates", []))
    return (0 if has_halt else 1, e.get("depth", 0), rows.index(e))

# Primary queue wins; fallback only consulted when primary is empty so a
# spillover lane never starves its parent lane.
pool = eligible if eligible else fallback_eligible
pool.sort(key=priority_key)
chosen = pool[0]
print(chosen["id"])
PY
}

get_experiment_field() {
    local exp_id="$1"
    local field="$2"
    uv run python - "$exp_id" "$field" <<'PY'
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
    uv run python scripts/evaluate_gates.py --experiment-id "$exp_id" --metrics-file "$metrics_file" --state-file "$STATE_FILE" --gates-tsv "$GATES_TSV"
    return $?
}

append_notebook_entry() {
    local exp_id="$1"
    local ve="${2:-NA}"
    local pwmcc="${3:-NA}"
    local elapsed="${4:-0}"
    local gate_note="${5:-}"
    local stamp
    stamp=$(date -Iseconds)
    # lab_notebook.md is a protected path. The runner is forbidden from
    # writing to it (CLAUDE.md rule 3). Append per-experiment machine notes
    # to a sibling file the runner owns; the human promotes interesting
    # entries into lab_notebook.md manually.
    local runner_log="experiments/runner_notebook.md"
    if [[ ! -f "$runner_log" ]]; then
        printf "# Runner notebook\n\nAuto-appended by scripts/run_loop.sh. One row per completed experiment.\n\n" > "$runner_log"
    fi
    printf -- "- %s  **%s**  VE=%s  PW-MCC=%s  elapsed=%ss%s\n" "$stamp" "$exp_id" "$ve" "$pwmcc" "$elapsed" "${gate_note:+  gate=$gate_note}" >> "$runner_log"
}

commit_artifacts() {
    local exp_id="$1"
    local ve="${2:-NA}"
    local pwmcc="${3:-NA}"
    # Serialize git operations across parallel lane workers via flock.
    # Without this, two workers racing on `git add`/`git commit` can corrupt
    # the index. fd 9 holds the advisory lock for the duration of the block.
    (
        flock -x 9
        git add \
            "experiments/artifacts/$exp_id/" \
            "experiments/artifacts/_summary/" \
            experiments/results.tsv \
            experiments/gates.tsv \
            experiments/pwmcc_posthoc.tsv \
            experiments/state.json \
            experiments/state.lane.*.json \
            experiments/runner_notebook.md \
            experiments/logs/ 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "[SAE-LOOP] $exp_id complete VE=$ve PW-MCC=$pwmcc" --no-gpg-sign 2>&1 | tail -20 || true
        fi
    ) 9>"$GIT_LOCK"
}

claim_row() {
    local exp_id="$1"
    local claim_path="$CLAIM_DIR/$exp_id.claim"
    local tmp_path="$claim_path.$$.tmp"
    {
        printf "pid=%d\nlane=%s\nstarted_at=%s\n" "$$" "$SAE_LANE_LABEL" "$(date -Iseconds)"
    } > "$tmp_path"
    # O_EXCL via Python: rename only if claim doesn't already exist OR is stale.
    uv run python - "$claim_path" "$tmp_path" <<'PY'
import os, sys
target, src = sys.argv[1], sys.argv[2]
if os.path.exists(target):
    try:
        with open(target) as f:
            d = {k.strip(): v.strip() for k, v in (line.split("=", 1) for line in f if "=" in line)}
        pid = int(d.get("pid", "0"))
        if pid > 0:
            try:
                os.kill(pid, 0)
                # live claim — refuse
                os.unlink(src)
                sys.exit(1)
            except ProcessLookupError:
                pass  # stale, fall through
    except Exception:
        pass
os.replace(src, target)
sys.exit(0)
PY
    return $?
}

release_row() {
    local exp_id="$1"
    rm -f "$CLAIM_DIR/$exp_id.claim"
}

run_one() {
    local exp_id="$1"
    if ! claim_row "$exp_id"; then
        # Another lane worker grabbed this row between select_next_pending
        # and claim_row. Yield this iteration so the caller picks the next
        # eligible row on the next scan.
        echo "[lane=$SAE_LANE_LABEL] claim_lost $exp_id" >&2
        return 0
    fi
    trap "release_row '$exp_id'" RETURN
    local gpu_pref
    gpu_pref=$(get_experiment_field "$exp_id" "gpu_preference")
    local gpu_actual
    gpu_actual=$(resolve_gpu "$gpu_pref")
    local entrypoint
    entrypoint=$(dispatch_entrypoint "$exp_id")
    local log_file="$LOG_DIR/${exp_id}__lane-${SAE_LANE_LABEL}_$(date +%Y%m%d_%H%M%S).log"

    write_state_field "current_experiment_id" "$exp_id"
    # Per-experiment start/done notifications were too noisy. Heartbeat cron
    # already shows runner progress; the runner only ntfys on blockers,
    # gate fires, and milestone transitions now.

    local backoffs=(30 300 1800)
    local attempt=0
    local success=0
    local start_epoch
    start_epoch=$(date +%s)
    local last_signature=""
    local repeated_signature=0

    local rc=0
    while [[ $attempt -lt 3 ]]; do
        attempt=$((attempt + 1))
        heartbeat
        echo "=== attempt $attempt for $exp_id ===" >> "$log_file"
        CUDA_VISIBLE_DEVICES="$gpu_actual" uv run python -m "$entrypoint" --experiment-id "$exp_id" 2>&1 | tee -a "$log_file"
        rc=${PIPESTATUS[0]}
        if [[ $rc -eq 0 ]]; then
            success=1
            break
        fi
        if [[ $rc -eq $EXIT_SCAFFOLD_STUB ]]; then
            # Entry point is a scaffold stub. Silent-skip: no ntfy, no TSV
            # row from the shell side (Python already wrote its stub row),
            # no commit, no gate eval. Add the id to the session skip set
            # so the outer loop does not re-pick it.
            echo "$exp_id" >> "$SESSION_SKIPFILE"
            echo "scaffold_stub exit from $exp_id; skipping for this session" >> "$log_file"
            return 0
        fi
        echo "attempt $attempt failed rc=$rc" >> "$log_file"
        local signature
        signature=$(_failure_signature "$log_file")
        echo "failure_signature=${signature:-<none>}" >> "$log_file"
        if grep -qiE "CUDA error|cublas|NCCL|out of memory|OutOfMemoryError" "$log_file"; then
            nvidia-smi >> "$log_file" 2>&1 || true
        fi
        if [[ -n "$signature" && "$signature" == "$last_signature" ]]; then
            # Identical traceback across two consecutive attempts: structural
            # failure, no point in further retries with the same config.
            repeated_signature=1
            echo "structural failure detected: identical signature across attempts; aborting retries" >> "$log_file"
            break
        fi
        last_signature="$signature"
        if [[ $attempt -lt 3 ]]; then
            local wait=${backoffs[$((attempt - 1))]}
            echo "backing off $wait seconds" >> "$log_file"
            local half=$((wait / 2))
            sleep "$half"
            heartbeat
            sleep $((wait - half))
        fi
    done

    local end_epoch
    end_epoch=$(date +%s)
    local elapsed=$((end_epoch - start_epoch))

    if [[ $success -ne 1 ]]; then
        local now
        now=$(date -Iseconds)
        if [[ $repeated_signature -eq 1 ]]; then
            # Structural failure: identical traceback across attempts. Record
            # with a status the auto-retry policy explicitly excludes from
            # re-eligibility so we don't burn lane cycles on a config that
            # cannot succeed without operator intervention. Operator must
            # write a stale_* marker to re-stage after fixing the cause.
            local sig_tag="${last_signature:-unknown}"
            sig_tag="${sig_tag// /_}"
            printf "%s\t%s\tfailed_structural\t\t\t\t\t\t\t\t\t\t\t\t%d\t\t\trepeated_%s\n" \
                "$now" "$exp_id" "$elapsed" "$sig_tag" >> "$RESULTS_TSV"
            ntfy_send "high" "Structural failure: $exp_id" \
                "Two consecutive ${last_signature:-unknown-error} failures on $exp_id (lane $SAE_LANE_LABEL, gpu $gpu_actual). Marked failed_structural; will not auto-retry. Resolve the underlying cause and append a stale_* row to re-stage."
        else
            # Transient-looking failure (different signatures across
            # attempts). The auto-retry policy in select_next_pending
            # re-eligibilizes this row after RETRY_COOLDOWN_SECONDS.
            printf "%s\t%s\tfailed\t\t\t\t\t\t\t\t\t\t\t\t%d\t\t\tretry_exhausted\n" \
                "$now" "$exp_id" "$elapsed" >> "$RESULTS_TSV"
        fi
        return 1
    fi

    # The harness's finally: block has already appended exactly one row to
    # results.tsv before the Python process returned. Do NOT append another
    # row here. We only read the row back to populate ntfy + commit messages.
    local metrics_file="experiments/artifacts/$exp_id/metrics.tsv"
    local ve="NA" pwmcc="NA"
    if [[ -f "$metrics_file" ]]; then
        # metrics.tsv contract (training rule 5): first line is header, second
        # line is values. Training bodies put variance_explained in column 1
        # and pwmcc in column 2 so these two reads work generically.
        ve=$(awk -F'\t' 'NR==2 {print $1}' "$metrics_file" 2>/dev/null || echo "NA")
        pwmcc=$(awk -F'\t' 'NR==2 {print $2}' "$metrics_file" 2>/dev/null || echo "NA")
    fi
    # Missing metrics.tsv after a successful exit is a training-rule-5
    # violation. We don't ntfy: the row sits as ok with empty metrics, the
    # heartbeat catches the anomaly via "rows with status=ok but blank VE".

    local gate_rc=0
    if [[ -f "$metrics_file" ]]; then
        evaluate_gates_and_apply "$exp_id" "$metrics_file"
        gate_rc=$?
    fi

    # Post-hoc PW-MCC: fast, idempotent. Computes pairwise PW-MCC across
    # any (cell, all 3 seeds complete) tuples that haven't been processed yet,
    # and writes to experiments/pwmcc_posthoc.tsv. Failures here don't block
    # the runner.
    uv run python scripts/posthoc_pwmcc.py >> "$log_file" 2>&1 || true

    # Refresh figures + summary tables. Idempotent; reads results.tsv +
    # pwmcc_posthoc.tsv and rewrites _summary/{figures,tables}/*. Failures
    # here don't block the runner.
    uv run python -m src.analysis.figures >> "$log_file" 2>&1 || true

    append_notebook_entry "$exp_id" "$ve" "$pwmcc" "$elapsed" ""
    commit_artifacts "$exp_id" "$ve" "$pwmcc"

    # Per-experiment done ntfy removed (too noisy); heartbeat reports
    # progress every 12h.

    # halt_and_notify gates never actually halt the runner. They fire,
    # write the metric to gates.tsv and state.most_recent_gate_outcome, and
    # the runner moves on. Halts as a control-flow primitive were the wrong
    # design: a gate firing is research data ("this row deviated"), not an
    # operational error ("stop everything"). The heartbeat watchdog is the
    # only mechanism that escalates to a phone notification, and only when
    # progress has actually stalled.
    return 0
}

main() {
    init_state
    # Runner-start ntfy removed: the operator launched it themselves and
    # doesn't need a self-confirming notification.

    local idle_sleep=1800   # 30 min between scans when no eligible work

    while true; do
        if [[ -f "$HALT_FILE" ]]; then
            # User touched the HALT file deliberately. No ntfy: they know.
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
            write_state_field "runner_status" "idle"
            sleep "$idle_sleep"
            continue
        fi

        run_one "$next" || true

        sleep 5
    done
}

main "$@"
