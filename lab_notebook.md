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

placeholder for Week 3 daily entries.

---

## Submission day (2026-05-05)

placeholder.
