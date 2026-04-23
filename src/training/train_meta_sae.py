"""Recursive meta-SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows with `depth >= 1`
(SPEC.md:43). Input is the decoder-direction matrix of the previous depth
(training rule 9). Output is a BatchTopK SAE with `k=4`.

Training body not yet implemented; the harness handles `--smoke`.
"""

from __future__ import annotations

from src.training.harness import experiment_context


def main() -> None:
    with experiment_context(arch_hint="batchtopk_meta") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: meta-SAE harness ok"
            return

        # Training body goes here. Required pieces, from the rules:
        #   - resolve `dependencies[0]` to a prior-depth checkpoint or to
        #     the level-0 SAE via src/training/loaders.py::load_level0.
        #     Never load models by HF id here (rule 6).
        #   - extract decoder directions from the dependency, unit-normalize
        #     each column, cast fp16 for storage, fp32 for inner products
        #     (rule 9).
        #   - train a BatchTopK SAE with k = row['sparsity'] (rule 7; k=4
        #     for meta-SAEs).
        #   - Adam lr=3e-4, betas=(0.9, 0.999), weight_decay=0 (rule 8).
        #   - checkpoint + metrics.tsv + curves.tsv in ctx.artifact_dir
        #     (rule 5).
        #   - populate ctx.metrics: variance_explained, pwmcc, mmcs,
        #     pwmcc_null_mean, pwmcc_null_std, dead_latent_fraction.
        #   - on divergence, set ctx.status = "diverged" (rule 10).
        ctx.status = "scaffold_stub"
        ctx.notes = "train_meta_sae body not yet implemented"


if __name__ == "__main__":
    main()
