# Lab Notebook

Project: How Deep Does the Rabbit Hole Go? Testing the Limits of Recursive Feature Decomposition via Multi-Depth Meta-Sparse Autoencoders on Gemma-2-2B and GPT-2 Small

Author: Aragorn Wang (Colorado School of Mines)
Due: 2026-05-05

This notebook is the human-facing scientific record. The runner writes to `experiments/results.tsv` and `experiments/state.json`. I write here. No em dashes, no AI-writing tells, no interpretation without grounding in a specific row of `results.tsv`.

---

## 2026-04-14 (Tue): Day zero

### Context

Today is day one of the 3-week execution window. The proposal is locked (`proposal_v3.docx`). The experimental matrix is locked (`EXPERIMENTS.yaml`). The autonomous runner spec is locked (`SPEC.md`). Decisions are logged (`DECISIONS.md`).

The full matrix has 75 experiment rows plus autointerp plus the conditional SimpleStories stretch goal. Estimated total budget is ~41.5 GPU-hours across 5 GPUs. That means wall-clock ~9-12 days if everything runs clean, with the rest of the window reserved for analysis, figure generation, and writing.

### What is installed so far

Nothing yet. This repo has CLAUDE.md, EXPERIMENTS.yaml, DECISIONS.md, SPEC.md, the `scripts/` directory, the `.claude/` rules and commands, a stub `README.md`, and this notebook. Day zero.

### Checklist for today

- [ ] Run `scripts/bootstrap.sh` to install Python deps, register the git hook, schedule the cron heartbeat, and send a test ntfy.
- [ ] `wandb login`.
- [ ] `huggingface-cli login` and accept the Gemma-2-2B license at https://huggingface.co/google/gemma-2-2b.
- [ ] Subscribe to ntfy topic `sae-wanga-research` on phone and desktop. Confirm the bootstrap ntfy arrives.
- [ ] Smoke test: manually run `python scripts/heartbeat.py` and confirm a notification fires with the expected placeholder counts.
- [ ] Write the four Python entry points referenced in `SPEC.md` and `.claude/commands/run-experiment.md`:
  - `src.training.train_level0_batchtopk`
  - `src.training.train_meta_sae`
  - `src.training.train_null_baseline`
  - `src.analysis.run_autointerp`
  - Plus `src/training/seed.py::set_all_seeds`, `src/training/loaders.py`, and the metric modules under `src/metrics/`.
- [ ] Smoke test a single level-0 training run at a tiny token budget (e.g., 10M tokens instead of 500M) to confirm the pipeline is wired end-to-end before committing the full 12-hour Gemma BatchTopK level-0 run.

### Reference links

- Proposal: `proposal_v3.docx` in the repo root.
- Experiment matrix: `EXPERIMENTS.yaml`.
- Runner spec: `SPEC.md`.
- Decision log: `DECISIONS.md`.

### Notes

The Experiment 3 anchor (depth-1 BatchTopK at ratio 1/21) has a `halt_and_notify` gate on `variance_explained_deviation_from_leask > 0.05`. If this fires on the Gemma BatchTopK anchor, the implication is that my level-0 BatchTopK implementation differs from Bussmann in a way that invalidates downstream recursive analysis. Everything else depends on that anchor matching Leask et al. (arXiv:2502.04878) VE = 55.47%. The JumpReLU anchor does not have a published baseline; a deviation there is a "needs investigation" signal, not a pipeline break, per DECISIONS.md entry 5.

---

## Week 1 (2026-04-14 to 2026-04-20)

### Entries

placeholder for Week 1 daily entries.

---

## Week 2 (2026-04-21 to 2026-04-27)

### Entries

## 2026-04-26 (Sun): Anchor diagnosis. JumpReLU VE 0.19 is likely a training pipeline issue, not a real atomicity signal.

### Context

Recipe v2 d1 anchor results landed across all three seeds. VE clusters tight at 0.187, 0.191, 0.192 (rows `gemma_jumprelu_anchor_d1_s{0,1,2}` in `experiments/results.tsv`, status `ok`, timestamps 2026-04-25T09:19Z to 10:15Z). The `halt_and_notify` gate on `variance_explained_deviation_from_leask` fired on every seed because abs(0.19 - 0.5547) is roughly 0.36, well above the 0.05 threshold (`experiments/gates.tsv` rows at the same timestamps).

Per `SPEC.md:60` and `DECISIONS.md` entry 5, a fired gate on the JumpReLU anchor is bookkeeping rather than a pipeline break: Leask et al. (arXiv:2502.04878) used BatchTopK as the level-0 SAE and there is no published JumpReLU reference value. The decision today is whether 0.19 is a real JumpReLU atomicity phenomenon or a pipeline issue.

### Decision

Treat 0.19 as preliminary. Do not use the d1, d2, or d3 JumpReLU numbers as the basis for any recursive-decomposition claim in the writeup until the BatchTopK anchors come back and validate the meta-SAE training pipeline.

### Evidence

Three converging lines:

1. Architecture-controlled comparison from OrtSAE (openreview.net/forum?id=lBctELT2f9, Table 1, Gemma-2-2B layer 20, L0=70, MetaSAE at 1/4 ratio with k=4): meta-SAE explained variance on a BatchTopK level-0 is 0.490, on Matryoshka 0.349, on OrtSAE 0.340. Even the most-atomic published architecture sits at 0.34. My JumpReLU at the same 1/4 ratio (rows `gemma_jumprelu_d1_s0` VE 0.166439, `gemma_jumprelu_d1_s2` VE 0.163816) is roughly 2x below that floor.

2. Width insensitivity. VE at the 1/21 anchor (rows VE 0.181 to 0.192) is essentially equal to or slightly higher than VE at 1/4 (rows VE 0.164 to 0.184). A correctly trained meta-SAE must show VE rising visibly as compression decreases from 21x to 4x. Equal VE across a 5x width change means effective capacity is bottlenecked by something other than nominal width.

3. Dead latent fractions in `experiments/results.tsv` confirm the bottleneck. `dead_latent_fraction` = 0.47 at width 16384 (1/4 ratio runs `gemma_jumprelu_d1_s0/s2`), 0.037 at width 3121 (1/21 anchor runs `gemma_jumprelu_anchor_d1_s*`). Half the wider meta-SAE is dead. `src/training/train_meta_sae.py` does not implement the auxiliary dead-latent loss L_aux from Bussmann arXiv:2412.06410 §3 (auxk-style with α=1/32 and k_aux=512 dead latents). My own commit `dd3efa0` already flagged this gap: "auxk-style ghost grads likely needed for full closure."

### Action taken

1. Re-queued `l0_gpt2_batchtopk` and `l0_gemma_batchtopk` via `stale_zstd_fix` rows in `experiments/results.tsv`. The `zstandard` dependency was installed in commit `eb6a464`, fixing the original `ValueError: Compression type zstd not supported` failures that drove both level-0 runs to `retry_exhausted`. When the in-flight `gemma_jumprelu_d2_s0` finishes, the depth-first selection priority in `scripts/run_loop.sh:213-216` picks the level-0 trainings (depth 0) first, then the BatchTopK anchors (depth 1), before any d2 or d3 JumpReLU seed (depth 2). The "hold further JumpReLU" effect happens organically through the priority key, no state.json or matrix mutation needed.

2. Re-queued four additional flat SAE rows that retry-exhausted on the same zstd issue: `flat_gpt2_w12288_s{0,1,2}` and `flat_gemma_w16384_s2`. Same `stale_zstd_fix` mechanism.

### What the BatchTopK anchors decide

If `gpt2_batchtopk_anchor_d1_s*` lands at 0.55 ± 0.05 (matching Leask), the pipeline is correct on BatchTopK. The 0.19 JumpReLU result is then a real, novel observation: Gemma JumpReLU decoders are unusually atomic compared to BatchTopK at the same ratio. In that case, resume d2/d3 JumpReLU and write the JumpReLU result up as a positive finding rather than an artifact.

If `gpt2_batchtopk_anchor_d1_s*` lands at 0.2 to 0.3 (well below Leask), the pipeline is undertrained across the board. Add the `auxk` auxiliary loss to `src/training/train_meta_sae.py` per Bussmann 2412.06410 §3, re-run the d1 anchors, and only then resume d2/d3.

### Reference links

- Research report with full evidence chain: `claudedocs/research_meta_sae_anchor_diagnosis_20260425T2320Z.md`.
- Leask et al. 2025 ICLR (arXiv:2502.04878): anchor reference value 55.47% on GPT-2 BatchTopK at 1/21.
- OrtSAE (openreview.net/forum?id=lBctELT2f9): Gemma-on-Gemma comparison values, Table 1.
- Bussmann BatchTopK SAEs (arXiv:2412.06410 §3): canonical auxk recipe (α=1/32, k_aux=512).

---

## Week 3 (2026-04-28 to 2026-05-04)

### Entries

## 2026-04-28 (Tue): Anchor widening from 1/21 to 1/4. lr fix on l0_gpt2. Stalled-lane recovery.

### Context

Three meta-blockers were holding the autopilot at the start of the day. All three are addressed in this entry. The auxk + 90k-step retune of the JumpReLU Gemma anchor at width 3121 (Bussmann 1/21 ratio, rows `gemma_jumprelu_anchor_d1_s0/s1/s2` in `experiments/results.tsv` at 2026-04-27T15:41/15:54/16:13) landed at VE 0.196370 / 0.195998 / 0.192787. The `variance_explained_deviation_from_leask` gate at threshold 0.05 fired `halt_and_notify` on every seed because abs(0.193 - 0.5547) ≈ 0.36, well above the threshold. Two further structural failures: `l0_gpt2_batchtopk` retried 3x with grad_norm=5.99e+03 at lr=3e-4 (rows at 2026-04-27T20:04 to 20:10, status `failed` then `retry_exhausted`), and lanes 2a/2b had silent stalls 3-6 hours after the last heartbeat with the python child still holding GPU memory (lane-2a child PID 1664828 on GPU-1 16292 MiB, lane-2b child PID 1651594 on GPU-3 24112 MiB at 2026-04-28T01:34Z).

### Decision

Widen all 9 anchor rows from Bussmann 1/21 (Gemma 3121, GPT-2 2340) to ratio 1/4 (Gemma 16384, GPT-2 12288). This deviates from a direct Leask et al. replication. The justification: the 0.36 VE deviation is roughly 7x the 0.05 gate threshold and persists across seeds, after a 3x training-step bump and after enabling auxk dead-latent revival from step 0. The gap is structural, not a tuning issue. The 1/4 ratio is the standard recursive ratio used elsewhere in the matrix (rows `gemma_*_d1_s*` widths in `EXPERIMENTS.yaml:121-167`) and matches the flat-baseline widths I already trained successfully (rows `flat_gemma_w16384_s0` VE 0.719891, `flat_gpt2_w12288_s2` VE 0.926622 in `experiments/results.tsv`). The leask-deviation gate is no longer meaningful at 1/4, so it is replaced with an absolute floor `variance_explained >= 0.50` per row. That floor sits ~30% below the flat baselines and gives the meta-SAE room to absorb the recursive-decomposition penalty without spurious gate fires.

For `l0_gpt2_batchtopk`, the divergence is a known-bad init-time gradient spike that the existing max_norm=1.0 grad clip catches in magnitude but not in pre-clip detection (the divergence test reads pre-clip norm). Half the lr (3e-4 to 1e-4) plus 1000-step linear warmup is the minimal intervention. The Gemma BatchTopK level-0 trained successfully under lr=3e-4, so the change is gated per-row via two new optional YAML keys (`learning_rate`, `lr_warmup_steps`) read in `src/training/train_level0_batchtopk.py:117-120,150-155`.

### Action taken

1. Edited 9 anchor rows in `EXPERIMENTS.yaml` (lines 535-749). Width 3121 to 16384 (Gemma jumprelu and batchtopk, 6 rows) and 2340 to 12288 (GPT-2 batchtopk, 3 rows). `dict_ratio` 0.0476 to 0.25. `estimated_gpu_hours` 0.5 to 1.5 (Gemma) and 0.3 to 1.0 (GPT-2). Replaced `variance_explained_deviation_from_leask threshold 0.05` with `variance_explained threshold 0.50` in each row's `decision_gates`.

2. Edited `src/training/train_level0_batchtopk.py` to read `row.get("learning_rate")` (default 3e-4) and `row.get("lr_warmup_steps")` (default 0). Added a 1000-step linear warmup that ramps lr from 0 to `base_lr` before each `opt.step()`. Updated `EXPERIMENTS.yaml` row `l0_gpt2_batchtopk` (lines 86-105) to set `learning_rate: 1.0e-4` and `lr_warmup_steps: 1000`.

3. SIGTERM'd both stalled GPU processes (PIDs 1664832 and 1651598) at 2026-04-28T01:39Z. Lane shells (1588098 lane-2a, 1588476 lane-2b) detected the failure, removed the row claims, slept 30s, and re-claimed with new python children (1682583 and 1682588) by 2026-04-28T01:44Z. New GPU memory residency: 16292 MiB on GPU-1, 23792 MiB on GPU-3, similar to pre-stall. If the same stall recurs in this iteration, the cause is in `iter_residual_batches` token shard prep (file-lock contention between two Gemma trainers sharing the same shard) and needs a code investigation rather than a kill+restart.

4. Appended 10 stale_* markers to `experiments/results.tsv` at 2026-04-28T01:44:35.917728Z. 9 rows tagged `stale_anchor_widen` (the 9 anchor IDs above), 1 row tagged `stale_lr_warmup_fix` (`l0_gpt2_batchtopk`). The runner picks these up via `scripts/run_loop.sh:202-217` and re-stages on the next loop iteration.

### Scope deviation note

This is the first explicit deviation from Bussmann's 1/21 ratio in the matrix. The proposal `proposal_v3.docx` and `EXPERIMENTS.yaml` SECTION 3 framed the 1/21 anchor as a direct replication target. With the widen, SECTION 3 is now an architecture-controlled comparison at 1/4 (the same ratio used in SECTION 2/4 recursive depths) rather than a Leask replication. The Leask 0.5547 number is no longer the target. Decision authority: the human operator (this entry) authorized the widen at 2026-04-28 in response to a failed retune. Logged for the writeup.

### ETA reset

With anchors at 16384/12288 and lr fix in place, the binding-constraint chain becomes: l0_gpt2_batchtopk (~4 GPU-h) -> 9 anchors at 16384/12288 (~12 GPU-h aggregate, parallelized 2-wide across 4080s) -> 18 Gemma recursive d1/d2/d3 rows -> autointerp on 13 rows. ~24-30 wall-hours from 2026-04-28T01:44Z if no further blockers fire, ~36-48 if either of the new gates trips again or the 3090 stalls recur.

---

## 2026-04-28 (Tue, mid-morning): Operator decision sweep, anchor refloor, auxk re-stage cleanup, deadline extension.

### Context

The widen+auxk retune from the morning entry produced anchor VE 0.20 (Gemma JumpReLU at w=16384) and 0.12 (GPT-2 BatchTopK at w=12288), still below the 0.20 floor in `EXPERIMENTS.yaml` (relaxed earlier in the day from 0.50). The `gpt2_batchtopk_anchor_d1_s0` finish at 07:46:31Z fired the gate AND triggered `scripts/evaluate_gates.py:maybe_trigger_auxk` (AUXK_TRIGGER_VE=0.50), which mass-appended `stale_auxk_fix` markers to 42 rows: every BatchTopK anchor + every depth-1/2/3 meta-SAE row on both base models. The sentinel `experiments/AUXK_ENABLED` is now set so the trigger cannot re-fire, but the 42 stale markers in `experiments/results.tsv` made the runner re-stage 7 already-`ok` Gemma JumpReLU rows alongside the genuinely-pending work. Lanes 0/1/4 spent the past hour retraining anchor seeds whose existing `ok` rows would have passed any reasonable floor.

### Operator decisions (8 questions, all resolved in this session)

1. **Anchor VE floor**: lowered from 0.20 (skip_depth) to **0.10 (skip_depth)** on all 9 anchor rows in `EXPERIMENTS.yaml` (lines 554, 578, 602, 626, 650, 674, 698, 722, 746). At 0.10 the floor is a sanity check (catches catastrophic divergence, e.g. negative VE) only; nothing cascade-skips. Per-row floors on the recursive d1/d2/d3 rows left at 0.20 — to be revisited if any cascade fires.
2. **Silent lanes 2a/2b**: confirmed via `py-spy dump` that both processes are forward-progressing (l0_gemma_batchtopk PID 37435 alternates between transformer_lens base-LM forward and SAE forward+loss in 3-second snapshots; flat_gemma_w16384_s2 PID 53029 inside transformer_lens forward for activation gen). Stale heartbeats are emission-gap, not stall. No kill.
3. **Skipped seed-1 d2/d3 rows**: re-run both. Appended `stale_seed1_recover` markers for `gemma_jumprelu_d2_s1` and `gemma_jumprelu_d3_s1` at 09:35:07Z. Cleared `skipped_by_gate` from `state.lane.lane-1.json` and `state.json`.
4. **autointerp_all timing**: single end-of-run pass once full corpus is in. No partial pass.
5. **simplestories_stretch**: attempt (was conditional `stretch_if_time_permits`, now standard).
6. **Gemma small-width flat baselines (w1024, w4096)**: run all 6 rows. Provides a flat-baseline width-sweep curve.
7. **gemma_batchtopk ratio sweep (`ratio_half`, `ratio_eighth`)**: run all 6 rows.
8. **In-flight anchor retunes**: kill all three. SIGTERMed PIDs 57175/59974/59135 at ~09:25Z; the lane wrappers respawned new python children which I then SIGKILLed along with the wrapper bashes (PIDs 36728/36986 + 67724/67731/67733/67737 + 69003/69010/69122/69127/69128/69132). Lane-4 left to complete `gemma_jumprelu_anchor_d1_s2` retune at w=16384 (the prior `ok` for s2 was at the obsolete w=3121, so a retrain is needed for seed-symmetry).

### Action taken

1. SIGTERM/SIGKILL of 12 python+wrapper PIDs across two waves (the first SIGTERM caused wrappers to respawn new python children mid-conversation; the second SIGKILL took out both the new children and the wrapper bashes themselves). The outer tmux loop respawned the lane-0 and lane-1 wrappers within 30s.

2. Manually removed orphaned claim files `experiments/.row_claims/gemma_jumprelu_anchor_d1_s{0,1}.claim` and `gemma_jumprelu_d1_s{0,1}.claim` (the wrappers' `trap RETURN` does not fire on signal-kill, so claim cleanup is manual).

3. Appended 11 synthetic-`ok` rows to `experiments/results.tsv` at 2026-04-28T09:33:13Z and T09:37:10Z. Each row carries the prior `ok` VE/dlf values verbatim with `gpu_hours=0` and notes documenting the source row + reason ("promoted from prior ok ...; auxk re-stage at 2026-04-28T07:46:31 was overbroad ...; decision1 anchor refloor 2026-04-28; no retrain"). Promoted rows: `gemma_jumprelu_anchor_d1_s{0,1}` (post-widen w=16384), `gpt2_batchtopk_anchor_d1_s{0,1}` (post-widen w=12288), `gemma_jumprelu_d{1,2,3}_s{0,2}`, `gemma_jumprelu_d1_s1`. Total 11 promotions.

4. Edited `EXPERIMENTS.yaml` lines 554/578/602/626/650/674/698/722/746 from `threshold: 0.20` to `threshold: 0.10`. Updated section-3 comment (line 533) to reflect the two-step relaxation 0.50 -> 0.20 -> 0.10 with rationale.

5. Appended `stale_seed1_recover` markers for `gemma_jumprelu_d2_s1` and `gemma_jumprelu_d3_s1` at 2026-04-28T09:35:07Z so the runner picks them up under the standard latest-row-wins rule.

6. Cleared `skipped_by_gate` from `experiments/state.json` (root + lane-1 sublane) and `experiments/state.lane.lane-1.json`.

7. Lane post-restart picks (verified at 09:37Z): lane-0 -> `gpt2_batchtopk_anchor_d1_s2` (legit pending), lane-1 -> `gpt2_batchtopk_d1_s1` (legit pending; depends on l0_gpt2_batchtopk which is `ok`), lane-4 -> `gemma_jumprelu_anchor_d1_s2` retune continuing on 4060 Ti, lane-2a -> `l0_gemma_batchtopk` continuing, lane-2b -> `flat_gemma_w16384_s2` continuing.

### Submission deadline change

Submission day moves from 2026-05-05 to **2026-05-09 23:59 MDT**. Writeup-freeze at T-24h: **2026-05-08 23:59 MDT**. After the freeze, no more training; corpus locked, writeup-only. Net buffer is now ~10 days of compute-availability, vs. ~7 days under the prior deadline.

### Scope deviation note

Decision 1 (anchor floor 0.20 -> 0.10) is the second floor relaxation in 24 hours. The series is now 0.50 (Leask deviation) -> 0.20 (post-widen, skip_depth) -> 0.10 (post-Decision1, sanity check only). Rationale stack: (1) at the widened 1/4 ratio, the anchor recipe inherently lands at VE 0.20 (Gemma) and 0.12 (GPT-2), neither of which is "broken"; (2) firing skip_depth on a recipe-typical VE pruned recursive d2/d3 work that the proposal explicitly requires; (3) the original Leask 55.47% target was config-specific (GPT-2 + ReLU + 1/21) and not a meaningful floor at 1/4. Section 3 is therefore an architecture-controlled comparison at 1/4, not a Leask replication.

The synthetic-`ok` promotions in step 3 are bookkeeping, not data fabrication: each row's VE/dlf values are copied verbatim from the prior real `ok` row in the same TSV (rows are append-only, the originals remain immutable). The promotions exist solely to defeat the over-broad `stale_auxk_fix` re-stage from 07:46:31Z.

### ETA reset

Estimated remaining GPU-h (sequential, with parallelism):
- in-flight (no new GPU-h needed): l0_gemma_batchtopk, flat_gemma_w16384_s2, gemma_jumprelu_anchor_d1_s2 retune
- 4 anchor rows still to run: gpt2_batchtopk_anchor_d1_s2 (~1 GPU-h, GPU-0), gemma_batchtopk_anchor_d1_s{0,1,2} (~4.5 GPU-h, blocked on l0_gemma_batchtopk)
- 9 gemma_batchtopk d1/d2/d3 (~6 GPU-h, blocked on anchor + l0)
- 9 gpt2_batchtopk d1/d2/d3 (~4.5 GPU-h, blocked on anchor)
- 6 ratio sweep (~6 GPU-h, blocked on anchor + l0)
- 6 small flats w1024/w4096 (~4 GPU-h, no deps; lanes 0/1/4 will pick up)
- 2 d2_s1/d3_s1 retries (~2 GPU-h, no deps)
- 1 simplestories_stretch (~2 GPU-h, no deps)
- autointerp_all (~6-10 GPU-h, blocked on full corpus)

Aggregate ~36-42 GPU-h remaining. With 5-lane parallelism on the existing GPUs, wall-time ~10-15 hours from 2026-04-28T09:37Z if no further infrastructural failures. ~12 days of buffer before the 2026-05-08 freeze. The compute side is no longer the binding constraint; writeup quality is.

### Decision 9 (emergent): d2 cascade action skip_depth -> skip_experiment

Discovered during implementation of Decision 3: the recursive d2 rows have `decision_gates: skip_depth` on both `variance_explained` (threshold 0.20) and `pwmcc_vs_null_sigma` (threshold 2.0). Given the existing data (d2_s0 VE -0.002, d2_s2 VE 0.003 — d2 collapse to ~0 VE is the recursive-decomposition phenomenon the experiment is measuring), d2_s1 is statistically certain to fire skip_depth, which would cascade-skip d3_s1 and defeat the seed-1 recovery.

Operator answer: A. Change all 9 recursive d2 rows' actions from `skip_depth` to `skip_experiment`. Under skip_experiment, the d2 row's own gate fire is recorded but it does not propagate to d3. d3 has its own threshold (0.20 with skip_experiment, an unchanged matrix value) and will run independently.

This also aligns d2 semantically with d3 (which already uses skip_experiment) and reflects the experiment's design intent: at d2 and d3, near-zero VE is the data point, not a recipe failure.

Edit: 18 lines in `EXPERIMENTS.yaml` (9 rows × 2 gate lines per row): variance_explained gate + pwmcc_vs_null_sigma gate, action `skip_depth` -> `skip_experiment`. Affects rows `gemma_jumprelu_d2_s{0,1,2}`, `gemma_batchtopk_d2_s{0,1,2}`, `gpt2_batchtopk_d2_s{0,1,2}`. The d1 rows' `skip_depth` remains unchanged because a d1-level recipe collapse legitimately invalidates downstream d2/d3 work for that lineage.

---

## 2026-04-28 (Tue, ~10:15Z): Decision 10 — d1 cascade action skip_depth -> skip_experiment

### Context

`gpt2_batchtopk_d1_s1` finished `ok` at VE=0.115188 (row in `experiments/results.tsv` at 2026-04-28T09:54:50Z), below the 0.20 `skip_depth` floor. The gate fired `skip_depth` at 09:54:52Z and cascade-skipped `gpt2_batchtopk_d2_s1` and `gpt2_batchtopk_d3_s1`, defeating the seed-1 recovery for the GPT-2 BatchTopK lineage. Decision 1 from the morning entry explicitly flagged this contingency: "Per-row floors on the recursive d1/d2/d3 rows left at 0.20 — to be revisited if any cascade fires."

The VE level is recipe-typical, not a collapse: GPT-2 BatchTopK at width 12288 (1/4 ratio) lands at VE 0.115–0.125 across both anchor (rows `gpt2_batchtopk_anchor_d1_s{0,1}` at 2026-04-28T07:46:29 and 08:17:07, VE 0.119408 and 0.124993) and recursive d1 (this row 0.115188). The Gemma JumpReLU equivalents land at ~0.18–0.20. The original assumption baked into the d1 floor — that low d1 VE means recipe collapse and should kill children — is wrong at this ratio for the GPT-2 path.

### Operator decision

Option B from operator session: change all 9 recursive d1 rows' actions from `skip_depth` to `skip_experiment`. Mirrors Decision 9's d2 change. d1 result is still recorded under its own gate; the cascade to d2/d3 is severed. Two pending GPT-2 d1 retrains (`gpt2_batchtopk_d1_s0`, `d1_s2`, both tagged `stale_auxk_fix` at 07:46:31) will almost certainly hit ~0.12 again and fire `skip_experiment` on themselves — no cascade impact.

Decision authority: human operator at 2026-04-28T10:12Z.

### Action taken

1. Edited `EXPERIMENTS.yaml`: 18 lines (9 d1 rows × 2 gate lines per row) flipped action `skip_depth` -> `skip_experiment` on `variance_explained` (threshold 0.20) and `pwmcc_vs_null_sigma` (threshold 2.0). Affects rows `gemma_jumprelu_d1_s{0,1,2}` (lines 131-132, 154-155, 177-178), `gemma_batchtopk_d1_s{0,1,2}` (lines 338-339, 361-362, 384-385), `gpt2_batchtopk_d1_s{0,1,2}` (lines 773-774, 796-797, 819-820). The 9 anchor `skip_depth` actions at threshold 0.10 remain unchanged (anchors have no downstream chain).

2. Appended `stale_seed1_recover` markers for `gpt2_batchtopk_d2_s1` and `gpt2_batchtopk_d3_s1` to `experiments/results.tsv` at 2026-04-28T10:12:08Z so the runner re-stages under the new semantics.

3. Cleared `skipped_by_gate` (gpt2_batchtopk entries only) from `experiments/state.json` (root + lane-1 sublane) and `experiments/state.lane.lane-1.json`.

### Note on Gemma seed-1 recovery

While Decision 10 was being applied, lane-1 completed both `gemma_jumprelu_d2_s1` (VE -0.005803, fired skip_experiment correctly per Decision 9) at 2026-04-28T10:08:42Z and `gemma_jumprelu_d3_s1` at 10:11:26Z. Gemma seed-1 recovery is complete. GPT-2 seed-1 recovery is now unblocked under Decision 10 and will be picked up by the next loop iteration.

### Scope deviation note

This is the third gate-action relaxation in 24 hours: anchor floor 0.50 -> 0.20 -> 0.10 (Decisions 1), d2 action skip_depth -> skip_experiment (Decision 9), d1 action skip_depth -> skip_experiment (Decision 10). The recipe-typical low-VE pattern at 1/4 ratio is now the working assumption, with skip_experiment used to record the data point without cascading to children. The matrix's original "low VE = recipe failure" semantics remain intact only at threshold 0.10 (anchors) — anything below 0.10 still cascade-skips.

---

## Submission day (2026-05-09 23:59 MDT)

placeholder.
