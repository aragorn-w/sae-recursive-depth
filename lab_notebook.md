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

placeholder for Week 2 daily entries.

---

## Week 3 (2026-04-28 to 2026-05-04)

### Entries

placeholder for Week 3 daily entries.

---

## Submission day (2026-05-05)

placeholder.
