---
paths: src/metrics/**
---

# Metrics Code Rules

These rules apply to every file under `src/metrics/`. Violations must be fixed before the runner is allowed to dispatch metric computation.

## Mandatory behaviors

1. **Both PW-MCC and MMCS are always computed.** PW-MCC (pairwise mean maximum cosine similarity across seed pairs, Beiderman et al. 2025) and MMCS (mean max cosine similarity against a reference dictionary, Sharkey et al. 2023) serve different purposes. PW-MCC answers "are seeds converging to the same solution?" and MMCS answers "does this dictionary recover a target?". Both must exist in `results.tsv` columns `pwmcc` and `mmcs` for every trained SAE.

2. **Hungarian matching uses joint encoder + decoder cosine similarity.** Per Paulo & Belrose (arXiv:2501.16615), pair features on the element-wise sum of encoder cosine similarity and decoder cosine similarity, after L2-normalizing both directions. Matches with joint similarity below 0.7 are reported as unmatched and counted in the `unmatched_fraction` column.

3. **Null baselines are isotropic Gaussians of matching dimensionality.** For Gemma-2-2B use `d_model = 2304`. For GPT-2 Small use `d_model = 768`. Draw the same number of samples per null dictionary as the trained dictionary has features. Seed the Gaussian draws with `null_seed = 1000 + trained_seed` so nulls are reproducible and distinct from data seeds. Store null PW-MCC mean and std in `experiments/results.tsv` columns `pwmcc_null_mean` and `pwmcc_null_std` (per SPEC.md §8.2) so gates can compute sigma distances directly.

4. **Variance explained is computed on held-out activations.** Use a held-out shard of The Pile separate from the training shards. The headline value goes in `experiments/results.tsv` column `variance_explained` (SPEC.md §8.2). The held-out token count is recorded in the per-experiment `experiments/artifacts/<id>/metrics.tsv` sidecar under `variance_explained_heldout_tokens` (training rule 5).

5. **Dead latent fraction uses a 10M-token activation budget.** A latent is dead if it never activates across 10 million held-out tokens. Column: `dead_latent_fraction`.

6. **Every metric writes an append-only row to `experiments/results.tsv`.** Never overwrite. Never edit existing rows. If a metric is recomputed, append a new row with the same `experiment_id` and a later `timestamp` (per SPEC.md §8.2 column name); analysis picks the latest row per id.

7. **Metric code never interprets results.** It computes numbers and writes them. It does not print "looks good" or "this suggests". Interpretation is a human responsibility.

8. **Cite the source paper in the docstring** of every metric function: arXiv ID and the equation or section number where the metric is defined.

## Forbidden

- Silently dropping NaN or Inf values. If a metric produces NaN, record the NaN and attach a `metric_error` column explaining why.
- In-place modification of `results.tsv`.
- Computing PW-MCC without a matching null draw.
- Using encoder-only or decoder-only cosine similarity for Hungarian matching unless explicitly labeled as a separate diagnostic column. Diagnostic columns (`pwmcc_encoder_only`, `pwmcc_decoder_only`, `unmatched_fraction`) live in the per-experiment `metrics.tsv` sidecar; the headline `pwmcc` value goes in `results.tsv` per SPEC §8.2.

## Reference list

- Beiderman et al. 2025 (PW-MCC formulation as applied in SAE seed-stability work)
- Sharkey et al. 2023 (MMCS, arXiv:2309.08600 context)
- Paulo & Belrose 2025 (arXiv:2501.16615) for joint-similarity Hungarian matching
- Leask et al. 2025 (arXiv:2502.04878) for the 55.47% variance explained anchor value
