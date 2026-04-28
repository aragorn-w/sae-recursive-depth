# Handoff — Leask retarget + skip-on-bad-anchor (protected paths)

**Author:** Claude (orchestrator session 2026-04-28)
**Triggered by:** Completed research investigation of arXiv:2502.04878 (Leask et al.) and arXiv:2412.06410 (Bussmann et al.). User approval recorded in conversation: "Re-target Leask. … if lowering the gate floor would mean trying more recursive levels upon a level that only gets 0.12 VE then let's just end it early there and accept it."

## Summary

The 0.5547 reproduction target was not the right target. Leask reports it for exactly one config (GPT-2 Small + ReLU parent + dict_ratio 1/21). Our anchors use JumpReLU/BatchTopK parents at dict_ratio 1/4 on different base models. They differ on at least four axes; comparing absolute VE to Leask's number is apples-to-oranges. Full research summary delivered in conversation.

Code-side changes (already applied by Claude in scripts/, non-protected):
- `scripts/evaluate_gates.py`: renamed `LEASK_VE` → `BUSSMANN_REPRO_REFERENCE_VE` with a docstring noting config-specificity. Added a deprecation note on the `variance_explained_deviation_from_leask` metric.
- `scripts/heartbeat.py`: amended the metric description to flag deprecation.
- `scripts/lane_watchdog.sh`: new — kills zombie training processes whose worker python is at %CPU=0 with stale curves.tsv.
- `scripts/run_autopilot.sh`: spawns the watchdog as a sidecar tmux window (`watchdog`).

This handoff covers the two **protected-path** changes that need your hands.

## Change 1 — `EXPERIMENTS.yaml`: anchor gate action `halt_and_notify` → `skip_depth`

**Why:** Currently when an anchor falls below the floor (e.g. `gpt2_batchtopk_anchor_d1_s0` at VE=0.119 vs the 0.20 floor), the gate fires `halt_and_notify` which doesn't actually halt anything (per `scripts/run_loop.sh:686-689` comment). The runner continues trying to train d2 and d3 meta-SAEs on top of a parent that already failed validation, wasting GPU-hours.

**Switching to `skip_depth`** uses the existing `descendants_to_skip` machinery in `scripts/evaluate_gates.py:191`, which matches `(base_model, level0_arch, seed)` and skips rows with `depth > exp.depth`. Since anchors are at depth=1, this skips d2 and d3 of the same lineage. Exactly the behavior you asked for.

**Diff (apply to all 9 anchor rows):** in `EXPERIMENTS.yaml`, replace each occurrence of:
```yaml
  decision_gates:
  - {metric: "variance_explained", threshold: 0.20, action: "halt_and_notify"}
```
with:
```yaml
  decision_gates:
  - {metric: "variance_explained", threshold: 0.20, action: "skip_depth"}
```

Affected lines (anchor rows): 554, 578, 602, 626, 650, 674, 698, 722, 746. Verify line numbers with `grep -n 'halt_and_notify' EXPERIMENTS.yaml | head -20` before editing — line numbers may shift after recent matrix updates.

**Note on threshold:** keeping the threshold at 0.20 (your relaxed value) is fine for the gemma jumprelu anchors (they came in at ~0.204, just above the floor). The gpt2 batchtopk anchor at 0.119 will fail this gate and now correctly skip d2/d3. If you'd rather give the gpt2 batchtopk lineage a fighting chance, lower the gpt2-anchor threshold to 0.10 specifically (per-row override). My recommendation: keep 0.20 globally — accept the early termination per your stated preference.

## Change 2 — `.claude/rules/training.md:27`: soften the Leask normalization attribution

**Why:** The current rule says *"Normalization is per Leask et al. (arXiv:2502.04878): each decoder column is unit-normalized before being fed to the next meta-SAE."* The web-research agent confirmed this attribution is **not in the Leask paper or the Bussmann LessWrong post**. Neither source explicitly endorses unit-normalizing decoder columns. The behavior in `src/training/train_meta_sae.py` (which actually unit-norms rows, not columns) is fine — it's standard SAE practice — but the citation as written is incorrect.

**Diff (one line):** in `.claude/rules/training.md`, replace line 27 (the rule 9 body):
```
9. Training data for recursive depths is the decoder direction matrix of the previous depth, cast to fp32 for the inner product and fp16 for storage. Normalization is per Leask et al. (arXiv:2502.04878): each decoder column is unit-normalized before being fed to the next meta-SAE.
```
with:
```
9. Training data for recursive depths is the decoder direction matrix of the previous depth, cast to fp32 for the inner product and fp16 for storage. Each decoder direction (row of W_dec under the project's `(n_latents, d_model)` convention) is unit-normalized before being fed to the next meta-SAE; this matches standard SAE practice (Bricken et al. 2023) but is not explicitly endorsed in Leask et al. arXiv:2502.04878 or Bussmann et al. arXiv:2412.06410, despite the recursive-meta-SAE methodology being adapted from those works.
```

This keeps the behavior, fixes the attribution, and adds the correct citation for the unit-normalization step.

## Optional change 3 — Add a true Bussmann-reproduction anchor

The existing 2026-04-25 runs at `width=3121` (which IS dict_ratio 1/21 for parent_width=65536) on the JumpReLU Gemma parent already gave VE=0.19 — same as the 1/4 retries. So **on Gemma**, dict_ratio is not the binding variable. Capacity is not the cap; parent SAE statistics are.

A true Bussmann reproduction would require a **GPT-2 Small + ReLU parent** SAE (49,152 wide). The repo does not currently load a ReLU GPT-2 SAE — `src/training/loaders.py::load_level0` would need a new branch for ReLU, plus a HuggingFace ReLU SAE source (e.g. Joseph Bloom's `jbloom/GPT2-Small-SAEs-Reformatted`). Cost: ~30 min compute for one anchor + a few hours of code work to add the loader. **My recommendation: skip this unless you specifically want a Bussmann reproduction calibration point.** The relative-VE story (1/21 ≈ 1/4 on our parents) is already informative without adding new infrastructure.

## Verification after applying

```bash
# 1. confirm gate action changed
grep -n 'halt_and_notify\|skip_depth' EXPERIMENTS.yaml | grep anchor

# 2. confirm rule 9 amended
sed -n '27p' .claude/rules/training.md

# 3. dry-run gate eval against an anchor metrics file (no commit)
uv run python scripts/evaluate_gates.py \
  --experiment-id gpt2_batchtopk_anchor_d1_s0 \
  --metrics-file experiments/artifacts/gpt2_batchtopk_anchor_d1_s0/metrics.tsv \
  --state-file /tmp/state-test.json \
  --gates-tsv /tmp/gates-test.tsv
# expect: gpt2_batchtopk_anchor_d1_s0 emits skip_depth (rc=0, not rc=42)
# and adds gpt2_batchtopk_d2_s0 + gpt2_batchtopk_d3_s0 to skipped_by_gate

# 4. apply via single human commit (uses --no-verify since both files are protected):
git add EXPERIMENTS.yaml .claude/rules/training.md
git commit --no-verify -m "matrix+rules: retarget Leask reproduction, skip lineage on bad anchor"
```

## Rollback

If anything misbehaves, revert with `git revert <sha>`. The watchdog and `evaluate_gates.py` constant rename are independent and can be reverted separately.
