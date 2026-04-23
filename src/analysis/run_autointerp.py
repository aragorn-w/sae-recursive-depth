"""Autointerpretation entry point.

Dispatched from `scripts/run_loop.sh` for the `autointerp_all` row
(SPEC.md:44). Uses Llama-3.1-8B-Instruct on GPUs 3 to score feature
explanations at each depth of the recursive stack.
"""

from __future__ import annotations

from src.training.harness import experiment_context


def main() -> None:
    with experiment_context(arch_hint="autointerp") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: autointerp harness ok"
            return

        # Body goes here. Required pieces:
        #   - load Llama-3.1-8B-Instruct on GPU 3 (or 3,4 per row config).
        #   - sample top-activating contexts for a fixed set of features at
        #     each depth of the recursive stack.
        #   - score explanations; log to ctx.artifact_dir / "autointerp.tsv".
        #   - populate ctx.metrics: median_detection_score,
        #     median_detection_score_vs_null_pct95.
        ctx.status = "scaffold_stub"
        ctx.notes = "run_autointerp body not yet implemented"


if __name__ == "__main__":
    main()
