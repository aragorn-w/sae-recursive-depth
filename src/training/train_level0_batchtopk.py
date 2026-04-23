"""Level-0 BatchTopK SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "train_from_scratch"` and `level0_arch == "batchtopk"`
and `depth == 0` (SPEC.md:40). Also reused for `simplestories_stretch`.

Training body is not yet implemented; the harness around it is wired up so
`--smoke` and scaffold-stub runs produce valid `results.tsv` rows. See
`.claude/rules/training.md` rules 7-10 for the implementation contract.
"""

from __future__ import annotations

from src.training.harness import experiment_context


def main() -> None:
    with experiment_context(arch_hint="batchtopk") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: level0 batchtopk harness ok"
            return

        # Training body goes here. Required pieces, from the rules:
        #   - read row['sparsity'] for k (rule 7; k=60 on Gemma-2-2B).
        #   - cache activations from row['base_model'] at row['layer'],
        #     row['site'] via src/data/ (not yet written).
        #   - Adam lr=3e-4, betas=(0.9, 0.999), weight_decay=0,
        #     500M tokens, batch 2048 (rule 8).
        #   - checkpoint to ctx.artifact_dir / "checkpoint.pt" (rule 5).
        #   - per-step curves to ctx.artifact_dir / "curves.tsv",
        #     final metrics to ctx.artifact_dir / "metrics.tsv" (rule 5).
        #   - populate ctx.metrics: variance_explained, dead_latent_fraction.
        #   - on divergence (NaN loss / grad norm > 1000), set
        #     ctx.status = "diverged" and return (rule 10).
        ctx.status = "scaffold_stub"
        ctx.notes = "train_level0_batchtopk body not yet implemented"


if __name__ == "__main__":
    main()
