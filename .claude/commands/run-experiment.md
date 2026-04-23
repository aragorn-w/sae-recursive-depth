---
description: Run a single experiment by ID from EXPERIMENTS.yaml with proper GPU pinning and logging.
argument-hint: <experiment_id>
---

# /run-experiment

Run exactly one experiment from `EXPERIMENTS.yaml` by its `id`. Intended for human-invoked interactive sessions (attach to tmux, pair-program, trigger a specific rerun). The autonomous `run_loop.sh` uses the same dispatch logic but this command is for targeted interventions.

## Arguments

`$1` = experiment id (e.g., `gemma_batchtopk_d1_s0`)

## Procedure

1. **Validate the id.** Parse `EXPERIMENTS.yaml` and confirm `$1` appears in the `experiments` list. If not, abort with a clear message listing the closest 3 ids by edit distance.

2. **Check dependencies.** For every id in the target's `dependencies` list, read `experiments/results.tsv` and confirm a row exists with `status = complete`. If any dependency is not complete, abort and list the missing ones.

3. **Pin GPUs.** Read `gpu_preference` from the experiment row and export `CUDA_VISIBLE_DEVICES` accordingly. Print the assignment to stdout.

4. **Dispatch to the correct entry point.** Based on `level0_arch`, `depth`, and `base_model`:
   - `level0_arch = batchtopk` and `depth = 0` → `python -m src.training.train_level0_batchtopk --config-id $1`
   - `depth >= 1` and `level0_arch in {batchtopk, jumprelu}` → `python -m src.training.train_meta_sae --config-id $1`
   - `level0_arch = none` (null baselines) → `python -m src.training.train_null_baseline --config-id $1`
   - `id == autointerp_all` → `python -m src.analysis.run_autointerp --config-id $1`

5. **Log to W&B.** Every dispatch sets `WANDB_PROJECT=sae-recursive-depth` and `WANDB_NAME=$1`. Resume if a run with that name already exists.

6. **Append to results.tsv.** On successful exit the entry point writes a row. Verify the row exists before returning. If missing, mark the experiment `failed` in `state.json` and exit non-zero.

7. **Send ntfy.** On success, call `scripts/ntfy_send.sh default "experiment-complete" "$1 finished" "experiment,manual"`. On failure, call with priority `high` and tag `failed,manual`.

## Flags to explain when running

- `CUDA_VISIBLE_DEVICES=<list>` restricts which physical GPUs PyTorch sees. Values are comma-separated indices matching `nvidia-smi`. For the 2x RTX 4080 meta-SAE runs the value is `0,1`. For 2x RTX 3090 level-0 and autointerp the value is `2,3`. For the RTX 4060 Ti evaluation GPU the value is `4`.
- `WANDB_MODE=online` is the default; set to `offline` if W&B is unreachable and sync later with `wandb sync`.

## Safety

This command does not bypass decision gates. It runs the experiment and lets the gate evaluator update state. It does not edit `EXPERIMENTS.yaml`, `SPEC.md`, or any other protected path.

If the experiment's `status` is already `complete`, prompt the user to confirm a rerun. A rerun appends a new row to `results.tsv` with a later timestamp; it does not delete the old row.
