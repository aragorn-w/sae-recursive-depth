"""Null-baseline SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "random_gaussian"` (SPEC.md:42 and SPEC.md:9.2). Trains an
SAE on isotropic Gaussian vectors of matching dimensionality and sample
count; feeds the `pwmcc_vs_null_sigma` computation for real-data rows.

Null baselines never trigger gates (their `decision_gates` list is empty).
"""

from __future__ import annotations

from src.training.harness import experiment_context


def main() -> None:
    with experiment_context(arch_hint="null") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: null SAE harness ok"
            return

        # Training body goes here. Required pieces:
        #   - generate N isotropic Gaussian vectors with d_model matching
        #     row['base_model'] (2304 Gemma, 768 GPT-2), N matching the
        #     corresponding real-data depth's sample count (SPEC.md:9.2).
        #   - train the SAE matching `row['level0_arch']` / sparsity / width.
        #   - checkpoint + metrics.tsv + curves.tsv in ctx.artifact_dir.
        #   - populate ctx.metrics: variance_explained, pwmcc (cross-seed
        #     against the other null seeds).
        ctx.status = "scaffold_stub"
        ctx.notes = "train_null_sae body not yet implemented"


if __name__ == "__main__":
    main()
