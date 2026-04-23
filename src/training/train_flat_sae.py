"""Flat SAE-on-activations training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "flat_sae_on_activations"` (SPEC.md:42). This is the
Experiment 6 comparator: a single-level SAE trained directly on base-model
activations with the same latent count as the recursive stack's leaf depth.
"""

from __future__ import annotations

from src.training.harness import experiment_context


def main() -> None:
    with experiment_context(arch_hint="flat") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: flat SAE harness ok"
            return

        # Training body goes here. Required pieces:
        #   - cache activations from row['base_model'] at row['layer'],
        #     row['site'] via src/data/ (not yet written).
        #   - train a single-level SAE matching `row['level0_arch']` with
        #     row['sparsity'] and row['width'].
        #   - Adam defaults per rule 8.
        #   - checkpoint + metrics.tsv + curves.tsv in ctx.artifact_dir.
        #   - populate ctx.metrics: variance_explained, dead_latent_fraction.
        ctx.status = "scaffold_stub"
        ctx.notes = "train_flat_sae body not yet implemented"


if __name__ == "__main__":
    main()
