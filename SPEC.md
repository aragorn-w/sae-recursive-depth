# SPEC.md

Authoritative specification for the recursive meta-SAE runner. Any divergence between this file and the implementation is a bug in the implementation.

## 1. Run loop (pseudocode)

```
while True:
    state = load_state("experiments/state.json")
    matrix = load_matrix("EXPERIMENTS.yaml")
    next_exp = select_next_pending(matrix, state)
    if next_exp is None:
        ntfy_milestone("matrix complete")
        break
    mark_running(next_exp, state)
    try:
        with retry_on_infra_error(max_tries=3, backoff_seconds=[30, 300, 1800]):
            result = dispatch(next_exp)
    except UnrecoverableFailure as e:
        mark_failed(next_exp, state, reason=str(e))
        ntfy_error(next_exp, e)
        continue
    append_results_tsv(next_exp, result)
    gate_outcome = evaluate_gates(next_exp, result)
    apply_gate_action(gate_outcome, matrix, state)
    commit_artifacts(next_exp, result)
    ntfy_update(next_exp, result, gate_outcome)
    save_state(state)
    sleep(5)
```

### 1.1 `select_next_pending`
- Filters matrix rows with `status == "pending"`.
- Keeps only rows whose every `dependencies` entry has `status == "complete"`.
- Excludes rows with `conditional.stretch_if_time_permits == true` unless all non-stretch experiments are complete AND the current date is at least 4 days before the 2026-05-05 deadline.
- Among eligible rows, preferred order is: rows with `halt_and_notify` gates first (anchors), then lower `depth` first, then the matrix's natural file order as a tiebreaker.

### 1.2 `dispatch`
Reads `level0_source` and routes to the correct Python entry point:
- `train_from_scratch` with `level0_arch == "batchtopk"` and `depth == 0` -> `python -m src.training.train_level0_batchtopk`
- `flat_sae_on_activations` -> `python -m src.training.train_flat_sae`
- `random_gaussian` -> `python -m src.training.train_null_sae`
- any huggingface id with `depth >= 1` -> `python -m src.training.train_meta_sae`
- id `autointerp_all` -> `python -m src.analysis.run_autointerp`
- id `simplestories_stretch` -> `python -m src.training.train_level0_batchtopk` with the simplestories config
Every entry point takes `--experiment-id <id>` and reads its config row from `EXPERIMENTS.yaml`. Every entry point sets `CUDA_VISIBLE_DEVICES` from the row's `gpu_preference` before importing torch.

## 2. Decision-gate evaluation

### 2.1 Gate actions
- `continue_depth`: the default non-action; runner proceeds to the next dependency-ready row.
- `skip_depth`: mark every matrix row whose `(base_model, level0_arch, seed)` lineage descends from this cell AND has `depth > this.depth` as `skipped_by_gate`. The cell itself is still `complete`.
- `skip_experiment`: mark only this row as `skipped_by_gate`. Do not touch descendants unless they depended on this row, in which case they become unreachable and are also marked `skipped_by_gate` with reason `unreachable_dependency`.
- `halt_and_notify`: set `status` of the runner in `state.json` to `halted`, ntfy urgent, exit the loop. Requires human intervention to restart.

### 2.2 Metric-to-gate mapping
Each row's `decision_gates` field is a list of `{metric, threshold, action}` dicts. At evaluation time:
- `variance_explained`: computed on held-out decoder directions (or activations for flat SAEs). Trigger action if `value < threshold`.
- `pwmcc_vs_null_sigma`: `(pwmcc_observed - pwmcc_null_mean) / pwmcc_null_std`. Trigger action if `value < threshold`.
- `variance_explained_deviation_from_leask`: `abs(value - 0.5547)`. Trigger action if `value > threshold`. Published value 0.5547 is from Leask et al. (arXiv:2502.04878), the depth-1 BatchTopK meta-SAE result on GPT-2 Small. On the Gemma-JumpReLU anchor this gate fires on any deviation >5pp from 0.5547; because no published JumpReLU anchor exists, a fired gate there means "investigate whether our number is plausible" rather than "our pipeline is broken."
- `dead_latent_fraction`: fraction of latents with zero activation across a 10M-token sample. Trigger action if `value > threshold`.
- `median_detection_score_vs_null_pct95`: `median(scores) / percentile_95(null_scores)`. Trigger action if `value < threshold`.

### 2.3 Logging
Every gate evaluation writes a row to `experiments/gates.tsv` with columns: `timestamp, experiment_id, metric, observed_value, threshold, action, triggered`. This is append-only.

## 3. Immutability guard

### 3.1 Protected paths
```
CLAUDE.md
EXPERIMENTS.yaml
DECISIONS.md
SPEC.md
proposal_v3.docx
lab_notebook.md
.claude/rules/**
.claude/agents/**
vendored/**
README.md
```

### 3.2 Hook location and install
- Hook lives at `scripts/pre_commit_immutability_guard.sh`, executable.
- Installed via `git config core.hooksPath scripts/git-hooks` where `scripts/git-hooks/pre-commit` is a symlink to `../pre_commit_immutability_guard.sh`. `bootstrap.sh` creates this.

### 3.3 Behavior
- Runs `git diff --cached --name-only` to get staged paths.
- For each staged path, checks against the protected-paths list (supporting `**` globs).
- If any match, prints a clear error and exits 1.
- Exits 0 otherwise.

### 3.4 Override
- Never override from the runner.
- Human override: `git commit --no-verify`, only when the human is intentionally editing a protected path.

## 4. ntfy protocol

Topic: `sae-wanga-research` (public ntfy.sh).

### 4.1 Priorities
- `min`: quiet heartbeats that don't wake the phone.
- `low`: routine progress (experiment start/end).
- `default`: milestones, gate outcomes that don't halt.
- `high`: failures that require attention within a few hours.
- `urgent`: halt_and_notify triggers and repeated infra failures.

### 4.2 Title conventions
- Heartbeat: `[SAE heartbeat 08:00] 24/75 complete, 6.2 GPU-hrs, no blockers`
- Experiment start: `[SAE start] gemma_jumprelu_d2_s0`
- Experiment done: `[SAE done] gemma_jumprelu_d2_s0 VE=0.42 PW-MCC=0.31`
- Gate skip: `[SAE gate SKIP_DEPTH] gemma_jumprelu_d3 VE<0.20`
- Gate halt: `[SAE HALT] gpt2_batchtopk_anchor_d1_s0 VE dev 0.09 from Leask`
- Milestone: `[SAE milestone] all depth-1 Gemma seeds complete`
- Error: `[SAE ERROR] OOM on gemma_batchtopk_d1_s1 after 3 retries`

### 4.3 Message format
Plain text, Markdown-lite. Heartbeats include: `matrix_row_current`, `complete / total`, `gpu_hours_used`, `most_recent_gate_outcome`, `blockers` (empty string if none).

### 4.4 Cron
```
0 8,20 * * * cd /home/wanga/school/math498c/sae-recursive-depth && /usr/bin/env TZ=America/Denver python scripts/heartbeat.py >> experiments/logs/heartbeat.log 2>&1
```

## 5. tmux layout

Session name `sae-loop`, 4 windows:
- Window 0 `runner`: main pane runs `bash scripts/run_loop.sh`.
- Window 1 `gpu`: `nvtop`.
- Window 2 `notebook`: `tail -F lab_notebook.md`.
- Window 3 `log`: `tail -F` on the most recently created file in `experiments/logs/`.

Create with:
```bash
tmux new-session -d -s sae-loop -n runner 'bash scripts/run_loop.sh'
tmux new-window -t sae-loop -n gpu 'nvtop'
tmux new-window -t sae-loop -n notebook 'tail -F lab_notebook.md'
tmux new-window -t sae-loop -n log 'bash -lc "tail -F $(ls -t experiments/logs/ | head -1)"'
```

Attach: `tmux attach -t sae-loop`.

## 6. Failure recovery

### 6.1 Retry policy
Infrastructure failures (OOM, CUDA init, file missing, network blip) are retried at 30s, 5min, 30min backoffs. After the third failure the row is marked `failed`, ntfy'd at `high`, and the runner moves on.

### 6.2 OOM handling
- Catch `torch.cuda.OutOfMemoryError`.
- On first OOM: halve the batch size for this run, log the change, retry.
- On second OOM at halved batch: halve again.
- On third OOM: mark `failed`, ntfy, move on.

### 6.3 CUDA error handling
- Capture stderr. If it contains `CUDA error` or `cublas` or `NCCL`, run `nvidia-smi` and log output, then retry.
- If `nvidia-smi` itself fails, mark the run `failed`, ntfy `urgent`, move on without retry.

### 6.4 W&B outage handling
- Runs always write to `results.tsv` first, then attempt W&B upload with a 30-second timeout.
- W&B failures do not mark the experiment as failed. They log a warning and queue a retry in `experiments/wandb_retry_queue.json` for a cron sweep.

### 6.5 Disk full
- Before each run, check free space on `/` and on the NVMe holding `experiments/`. If either is under 50 GB free, mark the run `failed` with reason `disk_full`, ntfy `urgent`, do not retry.

## 7. Interactive attach protocol

The human can attach to the tmux session at any time.
- Observation only: attach, switch windows, detach. Runner is not disturbed.
- Pair programming: human opens a new pane (Ctrl-b %) and runs `claude` interactively. The runner pane keeps going.
- Graceful pause: human signals the runner with `touch experiments/PAUSE`. Runner finishes the current experiment, writes results, commits, then polls for `experiments/PAUSE` absence before picking up the next row.
- Resume: `rm experiments/PAUSE`.
- Emergency stop: `touch experiments/HALT`. Runner exits cleanly at the next loop iteration after writing state.

## 8. State persistence

### 8.1 `experiments/state.json`
Single JSON object:
```json
{
  "runner_status": "running | halted | paused | idle",
  "pid": 123456,
  "started_at": "2026-04-14T08:00:00-06:00",
  "last_heartbeat": "2026-04-14T08:00:00-06:00",
  "current_experiment_id": "gemma_jumprelu_d2_s0",
  "most_recent_gate_outcome": {
    "experiment_id": "gemma_jumprelu_d3_s0",
    "metric": "variance_explained",
    "value": 0.17,
    "action": "skip_experiment",
    "timestamp": "2026-04-14T09:12:00-06:00"
  },
  "blockers": []
}
```
Written atomically (write to `.tmp` then rename).

### 8.2 `experiments/results.tsv`
Append-only. Header on first row:
```
timestamp	experiment_id	status	base_model	level0_arch	depth	seed	width	variance_explained	pwmcc	mmcs	pwmcc_null_mean	pwmcc_null_std	dead_latent_fraction	gpu_hours	commit_sha	wandb_run_url	notes
```
Each experiment appends exactly one row on completion. If re-run (because a previous attempt failed), a second row is appended; the later timestamp wins when querying. No in-place edits.

### 8.3 Recovery across restarts
On startup, the runner:
1. Reads `state.json`. If `runner_status == "running"` with a non-existent PID, treats it as a crash and continues from `current_experiment_id`.
2. Reads `results.tsv` and reconciles matrix `status` fields in memory. A row is `complete` iff the tsv contains at least one `status=ok` line for it.
3. Writes a new `state.json` with fresh PID and `runner_status=running`.

## 9. Experiment-specific rules

### 9.1 Anchor halt gate (Experiment 3)
The `halt_and_notify` gate on all anchor rows compares variance explained to the Leask et al. published value 0.5547 (arXiv:2502.04878 Table 1). A deviation above 0.05 on any single seed of the GPT-2 Small BatchTopK anchor halts the runner because this indicates a pipeline bug. The same gate on the Gemma-BatchTopK anchor halts on deviation >0.05 for the same reason. The gate on the Gemma-JumpReLU anchor fires for bookkeeping because JumpReLU has no published Bussmann reference, and a firing there is treated as "investigate" rather than "pipeline broken" - see DECISIONS.md entry from 2026-04-14.

### 9.2 Null baselines
Null-baseline experiments (`level0_source == "random_gaussian"`) train SAEs on isotropic Gaussian vectors of matching dimensionality (d_model = 2304 for Gemma, 768 for GPT-2) and matching sample count to the corresponding real-data depth. These never trigger gates; their `decision_gates` list is empty. Their outputs feed the `pwmcc_vs_null_sigma` computation for real-data rows.

## 10. What the runner must never do

- Modify any protected path in section 3.1.
- Invent new experiment rows.
- Interpret results in prose. Writing numbers to tsv is fine. Writing "this supports H2" is not.
- Skip seeds. If seed 1 fails infrastructurally, retry; don't fabricate it.
- Mutate `results.tsv` in place. Only append.
- Write to W&B without also writing to local tsv.
- Commit without the immutability guard running.
