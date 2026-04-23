---
paths: src/analysis/**
---

# Analysis Code Rules

These rules apply to every file under `src/analysis/`. Analysis code turns `experiments/results.tsv` into figures, tables, and summary statistics. It does not train anything, and it does not invent new numbers.

## Mandatory behaviors

1. **Every plot has a matching data file.** If the script writes `experiments/artifacts/<exp_id>/figures/foo.pdf`, it must also write `experiments/artifacts/<exp_id>/figures/foo.tsv` or `foo.csv` containing the exact numbers plotted. A figure without a companion data file is not acceptable and will fail review.

2. **Every numeric claim traces to a row and column in `results.tsv`.** Analysis functions may compute derived quantities (means, regressions, slopes) but those derivations must be reproducible from `results.tsv` alone, and the script must document which rows and columns feed each derived number.

3. **Figures go under `experiments/artifacts/<experiment_id>/figures/`.** Cross-experiment summary figures go under `experiments/artifacts/_summary/figures/`. Never write figures to the repo root or to `src/`.

4. **No interpretation strings in figure output.** Titles and captions state what is plotted, not what it means. Example good title: "PW-MCC by depth, Gemma-2-2B BatchTopK, ratio 1/4". Example bad title: "Recursive decomposition fails at depth 3".

5. **Autointerp uses the local Llama-3.1-8B-Instruct on GPUs 2 and 3.** Exactly 200 features per SAE at depths 1 and 2 for both base models. Score using the Gemma Scope autointerp protocol from Paulo & Belrose (arXiv:2501.16615). Write per-feature scores to `experiments/artifacts/<exp_id>/autointerp/scores.tsv`.

6. **Null distribution comparisons use explicit columns.** When plotting against null baselines, read `null_pwmcc_mean` and `null_pwmcc_std` from `results.tsv` for the matching null experiment, and annotate the figure with the sigma distance.

7. **Reproducibility stamp on every figure.** Each figure's companion data file must carry a header row with: generation timestamp UTC, git commit SHA, analysis script path, and the set of `experiment_id` values consumed.

## Forbidden

- Hardcoding numbers that should come from `results.tsv`.
- Writing prose claims about results in code comments.
- Editing `results.tsv` from analysis scripts. Analysis is read-only against the tsv.
- Using matplotlib `plt.show()` in batch analysis paths. Save to disk only.
- Figures without a data-file sibling.

## Column dictionary (read-only from results.tsv)

Analysis code may assume these columns exist and are populated for relevant rows: `experiment_id`, `timestamp_utc`, `base_model`, `level0_arch`, `depth`, `seed`, `dict_ratio`, `width`, `variance_explained`, `dead_latent_fraction`, `pwmcc`, `mmcs`, `null_pwmcc_mean`, `null_pwmcc_std`, `unmatched_fraction`, `median_detection_score`, `null_median_detection_score_pct95`, `gpu_hours`, `wandb_run_id`.
