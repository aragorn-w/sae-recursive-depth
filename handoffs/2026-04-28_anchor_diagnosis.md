# Anchor Plateau Diagnosis — gemma_jumprelu_anchor_d1

Authored by CRON-2A at 2026-04-28T05:43Z. Diagnostic agent — no autonomous fix applied.

## Reproducibility table

The most recent **completed** runs are from 2026-04-27 at the original Bussmann ratio width=3121. The 03:49Z 2026-04-28 retries at the widened width=16384 are **still running** (PIDs 36795, 37053, 38331 alive as of this report), but have not produced curves.tsv writes or final metrics.tsv yet, so we have no new VE data points.

| seed | width | VE | gate outcome | run timestamp | wandb URL |
|---|---|---|---|---|---|
| 0 | 3121 | 0.19637  | halt_and_notify (deviation 0.358 > 0.05) | 2026-04-27T15:41:35Z | runs/ino64ook |
| 1 | 3121 | 0.195998 | halt_and_notify (deviation 0.359 > 0.05) | 2026-04-27T15:54:08Z | runs/x91sojlq |
| 2 | 3121 | 0.192787 | halt_and_notify (deviation 0.362 > 0.05) | 2026-04-27T16:13:21Z | runs/9kd63p95 |
| 0 | 16384 | (in progress) | — | started 03:49Z, no curves data at 05:43Z | runs/trmsdxqp (live) |
| 1 | 16384 | (in progress) | — | started 03:49Z, no curves data at 05:43Z | runs/6gfau0o6 (live) |
| 2 | 16384 | (in progress) | — | started 03:49Z, no curves data at 05:43Z | runs/tticrdfp (live) |

The width=16384 runs were preceded by `interrupted` rows at 02:57Z (KeyboardInterrupt during a prior supervisor restart) and a `stale_anchor_widen` audit note at 01:44Z documenting the matrix update from 3121 → 16384 plus replacement of the leask-deviation gate with a `VE >= 0.50` floor.

## Comparison to Leask reference

- LEASK_VE: 0.5547 (per scripts/evaluate_gates.py:13)
- Observed mean VE at width=3121 across s0/s1/s2: **0.19505** (sd 0.00146)
- Deviation: **0.3596 absolute** (35.96 percentage points). Far exceeds the 5pp Leask-deviation gate threshold (0.05) **and** the matrix's current VE>=0.50 floor.
- Plateau confirmed across the original (2026-04-23, run after recipe v1) and re-tuned (2026-04-27, with auxk + 90k steps) rounds — VE shifted ~0 between rounds despite training-recipe changes.
- Plateau confirmed at width=16384: **NOT YET KNOWN.** The 03:49Z retries are still running ~114 min in (estimated 1.5 GPU-h), with empty curves.tsv (likely Python file-buffer not flushed). Widening from 3121 → 16384 (5× capacity) is the only outstanding free parameter that has not yet produced a measured VE.

## Methodology confirmation

- Input to meta-SAE: parent decoder direction matrix (W_dec_parent), unit-normalized rows. Per src/training/train_meta_sae.py:151 (`x_train = W_dec_parent.to(device)  # (parent_width, d_model), unit rows`) and .claude/rules/training.md:27 (rule 9). Citation matches Leask et al. arXiv:2502.04878.
- Parent SAE: `google/gemma-scope-2b-pt-res-canonical` at HF revision `fd571b47c1c64851e9b1989792367b9babb4af63`, sae_lens 6.39.0. parent_width=65536 confirmed in results.tsv notes column.
- AuxK enabled (`enable_auxk: true` plumbed via force_auxk in train_meta_sae.py:178). The 2026-04-27 runs all carry the `auxk=on` tag and still plateaued at 0.195.
- Training budget: 90,000 gradient steps (3× the 30k default), unchanged for the 16384 retries.

## Hypotheses for the gap (none auto-resolvable; ranked by expected impact)

1. **Layer mismatch** — we use Gemma-2-2B layer 12 residual; Leask paper may report a different layer. Highest expected impact because the L0 decoder geometry varies sharply across layers; one wrong layer choice would explain a 35pp gap with no other tuning needed.
2. **Normalization convention** — rule 9 says "each decoder column unit-normalized"; code unit-normalizes ROWS. Under the project's `(n_latents, d_model)` shape convention these are the same vectors (each row is a feature direction), but Leask may use the transposed convention so the unit-norming axis differs.
3. **AuxK recipe** — `force_auxk` is plumbed but the specific values of AUXK_DEAD_STEPS, AUXK_K, AUXK_ALPHA in src/training/train_meta_sae.py may differ from Bussmann/Leask. The 2026-04-27 retune already enabled auxk and saw zero VE movement — weak evidence that auxk is not the binding constraint.
4. **Parent SAE source** — we use Google's canonical Gemma-Scope checkpoint; Leask may use their own L0 trained with different settings (token budget, sparsity, layer). Hardest to test because re-training a Gemma-Scope-equivalent L0 SAE costs many GPU-hours.

## Decision options for the human

(a) **Relax the gate threshold** from 0.50 to e.g. 0.20 in EXPERIMENTS.yaml lines 554, 578, 602, 626, 650, 674, 698, 722, 746. Lets the runner proceed; documents the deviation; cheapest path; does not require resolving the scientific question of whether 0.19 vs 0.55 represents a methodological mismatch.

(b) **Accept VE≈0.195 as the real result** and document the deviation from Leask in `lab_notebook.md` and `proposal_v3.docx`. Downstream depths d2/d3 continue with this baseline. Honest but reduces the persuasiveness of the recursive-decomposability claim relative to the proposal.

(c) **Re-target a different Leask configuration** that's tractable at our scale. Requires re-reading Leask et al. arXiv:2502.04878 to identify which specific (layer, ratio, auxk recipe, parent-SAE provenance) combination yields 0.5547 and updating `.claude/rules/training.md` rule 9 and/or `EXPERIMENTS.yaml` accordingly. Most expensive but addresses the root scientific question.

A recommended ordering would be: wait ≤1h for the width=16384 retries to complete (option 0); then if VE is still ~0.19, pick (a) to unblock the runner immediately, and queue (c) as a separate workstream after the matrix completes.

## Critical file pointers

- `src/training/train_meta_sae.py:151` — meta-SAE input feed (`x_train = W_dec_parent`)
- `src/training/train_meta_sae.py:178` — `force_auxk` plumbing from row config
- `.claude/rules/training.md:27` — rule 9 (decoder directions, Leask normalization)
- `scripts/evaluate_gates.py:13` — `LEASK_VE = 0.5547` reference constant
- `EXPERIMENTS.yaml:537–746` — anchor row gate definitions (current gate: VE >= 0.50 floor per the 01:44Z stale_anchor_widen audit note)
- `experiments/results.tsv` — appended history, look for `gemma_jumprelu_anchor_d1` rows

## Outstanding observation flagged by the agent

The width=16384 retries have been running 114 min with 0-byte `curves.tsv` files. The same symptom is visible on lane-2b's `l0_gpt2_batchtopk` run from the same 03:49Z spawn. Likely cause: Python's default file-buffer for `curves_path.open("w")` not flushing because no `flush()` call inside the per-step write loop. This is **not** a divergence and **not** a hang (PIDs alive, GPUs at 100%), but it does mean the agent cannot read live progress from disk. Recommend a separate (non-blocking) task to add a `cf.flush()` after each curves.tsv write in `src/training/train_level0_batchtopk.py` and `src/training/train_meta_sae.py` so future diagnostic agents have observable progress. This is purely an observability fix; it does not affect training correctness.
