---
title: "How Deep Does the Rabbit Hole Go? Testing the Limits of Recursive Feature Decomposition via Multi-Depth Meta-Sparse Autoencoders on Gemma-2-2B and GPT-2 Small"
author:
  - name: Aragorn Wang
    affiliation: Colorado School of Mines
    email: wanga@mines.edu
bibliography: references.bib
biblatex: true
---

# Progress Writeup (Living Section)

*This half of the document is a running log of research progress and will shrink or fold into the paper body as results come in. The IEEE conference template sections below are the eventual submission artifact. Current state: day 22 of 26, 2026-05-06. Submission deadline 2026-05-09 15:00 MDT / 21:00 UTC (extended from 2026-05-05; see `DECISIONS.md` 2026-04-28). The execution window has stretched from the planned three weeks to roughly four. The level-0 BatchTopK SAE on Gemma-2-2B is the active blocker; the downstream Gemma BatchTopK depth chain and `autointerp_all` are queued behind it.*

## 1. Question and relationship to existing literature

Sparse autoencoders decompose neural network activations into overcomplete dictionaries of nominally monosemantic features [@bricken2023monosemanticity; @cunningham2024sparse]. A recent line of work treats a trained SAE's decoder columns as signals in their own right and trains a second SAE on them, exposing finer-grained structure that the original SAE had composed into bundled features. This is the meta-SAE construction of Bussmann et al. [@bussmann2024showingsaeslearn] and the motivating apparatus for Leask et al. [@leask2025sparse], who report that meta-SAEs recover decomposition consistent with lower-width SAEs trained directly on activations.

The open question this project addresses is whether recursive meta-decomposition continues to surface meaningful structure past depth one. The meta-SAE literature has so far stopped at depth one. Either the recursion converges to a stable finer-grained basis at some depth, or it collapses into seed-dependent noise, or it exhibits some other failure mode (e.g., dead-latent explosion, total variance-explained collapse). Nobody has reported which. Answering that is the project's contribution.

The question reduces to a concrete empirical comparison. For each base model, level-0 architecture, and depth in $\{1, 2, 3\}$, train three seeds of a meta-SAE at dictionary ratio $1/4$ of the parent, apply the Paulo and Belrose joint-similarity Hungarian matching protocol [@paulo2025sparse], and compute PW-MCC [@beiderman2025pwmcc] and MMCS [@sharkey2023taking] against both matched seeds and an isotropic Gaussian null baseline. If PW-MCC at depth $d$ is not distinguishable from the null at $\geq 3\sigma$, the recursion has bottomed out at depth $d$. If PW-MCC remains separated from null at all three depths, the recursion is still productive and the true ceiling is beyond depth 3.

Relationship to existing methods: this project sits between the meta-SAE construction of Bussmann et al. [@bussmann2024showingsaeslearn] and the depth-1 reproduction of Leask et al. [@leask2025sparse]. The recipe uses Matryoshka-SAE conventions [@bussmann2025learning] on the meta-SAE side and Paulo and Belrose's joint-similarity protocol [@paulo2025sparse] for matching. Concrete choices (anchor ratio, BatchTopK $k$, autointerp judge, null-baseline construction) are in §2.

## 2. Method

The experimental matrix is locked in `EXPERIMENTS.yaml`: 75 training runs across two base models (Gemma-2-2B as primary, GPT-2 Small as replication), two level-0 architectures on Gemma (JumpReLU Gemma Scope weights and trained BatchTopK), three depths, three seeds, with flat-SAE width controls and isotropic Gaussian null baselines at every depth. Total estimated budget is 41.55 GPU-hours across five GPUs (two RTX 4080s for meta-SAE training, two RTX 3090s for level-0 BatchTopK training and autointerp, one RTX 4060 Ti for evaluation and metric computation).

The key methodological choices, with rationale logged in `DECISIONS.md`:

1. **Dictionary ratio $1/4$** for the main matrix. This is the ratio Bussmann et al. [@bussmann2024showingsaeslearn] use and the ratio at which meta-SAE structure is reported to be most interpretable. Two additional ratios ($1/2$ and $1/8$) are run at depth 1 only, as a sensitivity check.

2. **Dictionary ratio $1/21$** for the anchor experiment only. This matches Leask et al. [@leask2025sparse] exactly and is the pipeline's ground-truth reproducibility check: if the anchor BatchTopK SAE on GPT-2 Small does not reproduce their reported 55.47% variance explained to within 5 percentage points, the pipeline is considered broken and the runner halts for human review before continuing to recursive runs.

3. **PW-MCC and MMCS are both computed.** PW-MCC answers "are seeds converging?" by doing pairwise matching across three seed pairs; MMCS answers "does this dictionary recover some reference basis?" by matching against a fixed target. The two are complementary and the literature is split on which is the right stability metric, so this project reports both.

4. **Joint encoder-plus-decoder cosine similarity** for Hungarian matching, following Paulo and Belrose [@paulo2025sparse]. Matches below joint similarity 0.7 are reported as unmatched and contribute to an `unmatched_fraction` column rather than being force-paired.

5. **Three seeds per cell**, not five. Three is the minimum to get a sample standard deviation; five would roughly double training budget. The cost-benefit heavily favors three given the 41.55 GPU-hour ceiling.

6. **Isotropic Gaussian null baselines** at every depth, with matching $d_{\mathrm{model}}$ (2304 for Gemma-2-2B, 768 for GPT-2 Small) and matching sample count. This gives a direct sigma-distance test for whether observed PW-MCC exceeds what random dictionaries would produce.

7. **Autointerp at depths 1 and 2 only.** Running autointerp at depth 3 would consume roughly 1.5 GPU-hours per SAE and there are 18 depth-3 SAEs in the matrix. Autointerp at depth 3 is in the conditional stretch-goals section of `EXPERIMENTS.yaml` and will only run if the primary matrix completes with at least four days of slack.

## 3. Experiments run so far

As of 2026-05-06 (day 22 of 26), 54 of the 75 rows in `EXPERIMENTS.yaml` are `ok` in `experiments/results.tsv`. Numbers in this section are quoted directly from that file; row IDs are pasted verbatim. PW-MCC values are not yet in the TSV; they require the cross-seed Hungarian-matching pass that runs after the matrix completes, so this section reports variance explained (VE) only.

### 3.1 Status by cell

| Cell | Status | Source rows |
|---|---|---|
| Null baselines (Gemma + GPT-2, d1/d2/d3, s0/s1/s2) | 18/18 ok | `null_*` |
| Flat-SAE controls, Gemma | 8/9 ok | `flat_gemma_w{1024,4096,16384}_s*` (`w1024_s2` pending retry) |
| Flat-SAE controls, GPT-2 | 3/3 ok | `flat_gpt2_w12288_s*` |
| Anchor, GPT-2 BatchTopK at ratio 1/21 | 3/3 ok | `gpt2_batchtopk_anchor_d1_s*` |
| Anchor, Gemma JumpReLU at ratio 1/21 | 3/3 ok | `gemma_jumprelu_anchor_d1_s*` |
| Recursive Gemma JumpReLU, d1/d2/d3 × 3 seeds | 9/9 ok | `gemma_jumprelu_d{1,2,3}_s*` |
| Recursive GPT-2 BatchTopK, d1/d2/d3 × 3 seeds | 9/9 ok | `gpt2_batchtopk_d{1,2,3}_s*` |
| Level-0 BatchTopK, GPT-2 Small | 1/1 ok | `l0_gpt2_batchtopk` |
| Level-0 BatchTopK, Gemma-2-2B | in flight (~50%) | `l0_gemma_batchtopk` |
| Anchor, Gemma BatchTopK at ratio 1/21 | 0/3, blocked | `gemma_batchtopk_anchor_d1_s*` |
| Recursive Gemma BatchTopK, d1/d2/d3 × 3 seeds | 0/9, blocked | `gemma_batchtopk_d{1,2,3}_s*` |
| `autointerp_all` | 0/1, blocked on the 6 stale `gemma_batchtopk_d{1,2}` deps | `autointerp_all` |

### 3.2 Numeric snapshot, completed cells

Means $\pm$ standard deviations across $n=3$ seeds, computed from the `variance_explained` column of `experiments/results.tsv`.

**Level-0 SAEs on activations (parents of all meta-SAEs):**

- `l0_gpt2_batchtopk`: VE 0.951.
- `l0_gemma_batchtopk`: in flight; see Section 4.

**Flat-SAE controls** (direct SAE on activations at meta-SAE matched widths; $n=3$ seeds each unless noted):

- `flat_gpt2_w12288`: VE $0.926 \pm 0.001$.
- `flat_gemma_w16384`: VE $0.721 \pm 0.002$.
- `flat_gemma_w4096`: VE $0.685 \pm 0.003$.
- `flat_gemma_w1024` ($n=2$ only, `s2` retry pending): VE 0.628, 0.633.

**Anchor reproductions at ratio 1/21:**

- `gpt2_batchtopk_anchor_d1`: VE $0.124 \pm 0.004$. Leask et al. [@leask2025sparse] report 55.47% VE on the same architecture, ratio, and base model. Our deviation is approximately 43 percentage points, well outside the original 5pp `halt_and_notify` threshold (relaxed to 10pp during execution; see `DECISIONS.md` 2026-04-28 entries). The `l0_gpt2_batchtopk` and `flat_gpt2_w12288` cells (above) match published numbers, so the divergence is meta-SAE-specific. Three candidate causes are not yet bisected: a metric-definition mismatch with Leask et al., a level-0-source mismatch (we trained ours fresh, Leask uses Bussmann's released level-0 weights), or an undiscovered pipeline issue. The cheapest discriminator is to swap our level-0 for Bussmann's released BatchTopK [@bussmann2024batchtopk] and rerun a single anchor seed; if the anchor jumps toward 55%, the cause is level-0-source. The deviation is recorded as a measurement, not a verdict; see Section 3.3.
- `gemma_jumprelu_anchor_d1`: VE $0.203 \pm 0.002$. No published reference; this is a novel depth-1 configuration.

**Recursive Gemma JumpReLU, ratio 1/4:**

- d1: VE $0.184 \pm 0.001$ (null d1 = $0.0045 \pm 0.0004$; ~130 null-$\sigma$ in absolute units).
- d2: VE $-0.002 \pm 0.005$ (n=3). Null d2 = $0.0172 \pm 0.001$ (n=3, `null_gemma_d2_s*`). The recursive d2 mean sits 0.019 below the null mean; the gap is roughly $4\sigma$ on the recursive cell's std and $\sim 19\sigma$ on the null cell's std. The d2 cell is below null, not indistinguishable from it.
- d3: VE $-0.025 \pm 0.007$ (n=3). Null d3 $\approx 0$ (mean $-7 \times 10^{-6}$, std $5 \times 10^{-5}$ across `null_gemma_d3_s*`). The recursive d3 mean is 0.025 below null; the gap is roughly $4\sigma$ on the recursive cell's std. Negative VE means the reconstruction is worse than predicting the held-out mean.

**Recursive GPT-2 BatchTopK, ratio 1/4:**

- d1: VE $0.114 \pm 0.004$.
- d2: VE $0.048 \pm 0.004$.
- d3: VE $0.022 \pm 0.003$.

**Recursive Gemma BatchTopK, ratio 1/4:** all 9 cells pending `l0_gemma_batchtopk`.

### 3.3 Open items requiring human framing in the IEEE body

`HUMAN_JUDGMENT_NEEDED:` Two facts in this section will require interpretation when the IEEE Results section is populated. They are flagged here so they are not silently inherited as conclusions:

1. **GPT-2 BatchTopK anchor at ratio 1/21 lands at VE 0.124, not Leask et al.'s 55.47%.** The recipe is consistent across our ratio-1/21 anchor (VE 0.124) and our ratio-1/4 recursive d1 cells (VE 0.114), so the deviation is not ratio-specific. The level-0 (VE 0.95) and flat baseline (VE 0.93) match published numbers, so the deviation is meta-SAE-specific. Pending the level-0-swap test described above, the deviation is reported as a measurement rather than a verdict on the pipeline.
2. **Gemma JumpReLU recursive d2 collapses to null (VE $\approx 0$) and d3 is negative.** The numbers are stable across seeds (small $\sigma$). Whether this is the "depth-1 success, depth-2 collapse" answer shape (Section 5, shape 2) or a JumpReLU-specific recipe artifact is not yet decidable. The dispositive contrast is Gemma BatchTopK at d1/d2/d3, currently blocked on `l0_gemma_batchtopk`. If Gemma BatchTopK shows the same shape, the headline is shape 2; if Gemma BatchTopK looks like GPT-2 BatchTopK (above null at d2 but below null by d3), the headline is that depth-of-collapse depends on architecture, which is a different and more nuanced finding. The IEEE Results and Discussion sections cannot be written until this contrast is in.

### 3.4 Updated beliefs

What entered the project on day zero versus what the data have done to those priors, keyed to specific rows in `experiments/results.tsv`:

1. **Prior: the published recipe replicates.** Posterior: only at level-0. The GPT-2 BatchTopK anchor at ratio 1/21 lands at VE 0.124 against a Leask et al. target of 0.5547. The level-0 SAE (VE 0.951) and the flat width control (VE 0.926) reproduce, so the gap is meta-SAE-specific, not pipeline-wide. Confidence in the published recipe at the meta-SAE step is lower than it was on day zero; confidence in this pipeline's level-0 path is unchanged. The level-0-source swap (Bussmann's released BatchTopK weights in place of `l0_gpt2_batchtopk`) is the cheapest discriminator and would move the posterior more than any other available test.

2. **Prior: no strong opinion on which of the four shapes from §5 wins.** Posterior: both completed arms show a collapse, but at different depths, and that disagreement is the largest update to the working hypothesis so far. Gemma JumpReLU collapses between d1 (VE 0.184, $\sim$130 null-$\sigma$ above) and d2 (VE $-0.002$, below the null d2 baseline of 0.0172). GPT-2 BatchTopK collapses one depth later: d1 VE 0.114 and d2 VE 0.048 sit above their respective null baselines (0.016 and 0.023), but d3 VE 0.022 sits below null d3 of 0.028. Both arms point toward shape 2 ("collapse after some early-success depth"), but the depth-of-collapse differs: d=2 for Gemma JumpReLU, d=3 for GPT-2 BatchTopK. The question has narrowed from "where does it bottom out" to "is the depth-of-collapse base-model-driven or level-0-architecture-driven?" The Gemma BatchTopK chain (currently blocked on `l0_gemma_batchtopk`) is the discriminator: if it tracks Gemma JumpReLU (collapse at d=2), the ceiling is base-model-driven; if it tracks GPT-2 BatchTopK (collapse at d=3), the ceiling is level-0-architecture-driven.

3. **Prior: VE separation tracks PW-MCC separation.** Posterior: holding the prior, but flagging that the cross-seed Hungarian-matching pass that produces PW-MCC has not run yet. If PW-MCC at JumpReLU d2 turns out to be well above null while VE is below it, the VE-driven "collapse" reading is wrong and the d2 result is reconstruction-quality only, not feature-stability.

4. **Prior: level-0 BatchTopK on a 2B-parameter model is a one-day task.** Posterior: it is a multi-day task. The original 12 GPU-hour estimate in `EXPERIMENTS.yaml` was off by ~5x (observed ~65 hours wall-clock, sharded fp32 across two RTX 3090s). Sharded fp32 with atomic checkpointing is now the default rather than a contingency, baked into commit `7cf903e`.

## 4. Experiments left to run

Critical-path remaining work, in dependency order:

1. **`l0_gemma_batchtopk`**: in flight, ~50% as of 2026-05-06. Blocks 13 downstream rows. Estimated wall-clock to completion: ~33 hours from current step 120k at observed cadence ($\approx$ 16.4 minutes per 1k SAE steps; total budget 244,141 steps, derived from the 500M-token budget at sae batch size 2048).
2. **`gemma_batchtopk_anchor_d1_s{0,1,2}`**: 3 rows at ratio 1/21. Will dispatch immediately after `l0_gemma_batchtopk` writes its `eval_done.pt` sentinel.
3. **`gemma_batchtopk_d{1,2,3}_s{0,1,2}`**: 9 rows. Six of these are tagged `stale_auxk_fix` from the 2026-04-28 mass re-stage and are queued behind `l0_gemma_batchtopk`. Estimated 0.5h each at d1, 0.2h each at d2/d3, parallelizable across lanes 0/1/4.
4. **`flat_gemma_w1024_s2`**: single missing seed for the smallest flat-Gemma control.
5. **`autointerp_all`**: 5 GPU-h on lane 3 (Llama-3.1-8B-Instruct [@dubey2024llama], single-device). Blocked on the 6 `gemma_batchtopk_d{1,2}` rows above.
6. **`simplestories_stretch`**: conditional stretch-goal experiment. Will run only if time allows after `autointerp_all`.

Estimated time-to-matrix-completion: ~05:00 UTC 2026-05-08, leaving a ~38-hour cushion before the 21:00 UTC 2026-05-09 submission deadline. The cushion assumes no further infrastructural failures; one full restart of `l0_gemma_batchtopk` would consume nearly all of it.

The original "everything" framing (this section, day-zero) has inverted: 54 of 75 rows are now done, and what remains is concentrated on the Gemma BatchTopK depth chain plus autointerp.

## 5. How the answer depends on those experiments

The question "does the recursion bottom out, and if so, at what depth?" has four mutually exclusive answer shapes, each predicted by a specific pattern of PW-MCC values across depth:

1. **Immediate collapse.** PW-MCC at depth 1 is not distinguishable from the isotropic Gaussian null at $\geq 3\sigma$. This would imply that the meta-SAE construction is essentially learning noise even at depth 1, contradicting Bussmann et al. [@bussmann2024showingsaeslearn] and Leask et al. [@leask2025sparse]. If observed, the first explanation to rule out is a pipeline bug, starting with the anchor experiment at ratio $1/21$.

2. **Depth-1 success, depth-2 collapse.** PW-MCC at depth 1 is well-separated from null, PW-MCC at depth 2 is not. The recursion bottoms out at depth 1 and meta-SAEs should not be stacked.

3. **Graceful degradation.** PW-MCC decreases monotonically with depth but stays separated from null through depth 3. The recursion remains productive but each additional level is less stable than the last. In this case the answer is "deeper is still possible, but returns diminish", and I would predict a crossover depth beyond 3 where null is matched.

4. **Plateau.** PW-MCC stays roughly flat across depth, well above null. The recursion has essentially converged and depth 3 is close to the stable fine-grained basis. Autointerp scores should also plateau.

The four shapes above are formulated for a single (base model, level-0 architecture) arm. The matrix runs three such arms (Gemma JumpReLU, Gemma BatchTopK, GPT-2 BatchTopK), and the cross-arm comparison is its own finding. Three qualitatively different cases are possible. (i) All three arms agree: the answer shape is universal at the depths tested. (ii) The two Gemma arms agree and disagree with GPT-2: the depth ceiling is base-model-driven. (iii) The two BatchTopK arms agree and disagree with JumpReLU: the depth ceiling is level-0-architecture-driven. The currently-completed evidence (§3.2, §3.4) already shows JumpReLU and GPT-2 BatchTopK disagreeing on depth-of-collapse; the missing Gemma BatchTopK arm is the discriminator between (ii) and (iii).

Supporting evidence that will refine whichever shape wins: dead-latent fraction and variance-explained trends across depth, autointerp detection scores at depths 1 and 2 against the flat-SAE width controls and the null baseline, and ratio-sensitivity at depth 1 (1/2 and 1/8 against 1/4 on the BatchTopK arms).

## 6. Roadblocks so far

### Resolved

1. **Gemma-2-2B HuggingFace gating.** One-time `huggingface-cli login` plus license acceptance; resolved 2026-04-14.
2. **Gemma Scope JumpReLU file paths at width 65536, layer 12 residual.** No upstream changes during the execution window; loader paths in `src/data/loaders.py` work as written.
3. **`l0_gpt2_batchtopk` gradient divergence at lr=3e-4.** Three retries hit grad-norm ~6e+03 at the same step before each crash; the existing max-norm clip caught the magnitude but the divergence test reads pre-clip norm. Halved the learning rate to 1e-4 and added a 1000-step linear warmup, gated per-row via two new YAML keys (`learning_rate`, `lr_warmup_steps`) in `src/training/train_level0_batchtopk.py:117-120,150-155`. Gemma BatchTopK level-0 was already training cleanly under lr=3e-4, so the change is GPT-2-specific. Resolved 2026-04-28.
4. **Dead-latent explosion at width 16384.** `src/training/train_meta_sae.py` lacked the auxk dead-latent revival loss specified in Bussmann arXiv:2412.06410 §3. Dead-latent fraction at width 16384 ran at 0.47 (roughly half the dictionary unused). After adding auxk ($\alpha = 1/32$, $k_{\mathrm{aux}} = 512$) and gating it via the `experiments/AUXK_ENABLED` sentinel, recipe-typical dead-latent levels were restored. Auxk runs from step 0 on all post-fix runs. Resolved 2026-04-28.

### Active or partially mitigated

5. **`l0_gemma_batchtopk` underestimated by ~5x.** The original 12 GPU-hour estimate in `EXPERIMENTS.yaml` is roughly 5x lower than observed (~65 hours wall-clock at the current sharded `n_devices=2` fp32 configuration). This row is the persistent meta-blocker for the entire Gemma BatchTopK depth chain. Recovery work is documented in commit `7cf903e` ([SAE-LOOP] L0-Gemma sharded, n_devices=2, fp32, crash-resilient training): atomic checkpoint+resume+mirror plus a watchdog. Subsequent runs progress at the observed ~16.4 min/1k SAE steps cadence without further intervention.
6. **Auxk re-stage mass-staled 42 rows on 2026-04-28T07:46:31Z.** The `auxk_fix` re-stage logic was over-broad and re-queued already-completed Gemma JumpReLU recursive rows. Resolution: `Decision 12` (2026-04-28) introduces synthetic-`ok` promotion as a bookkeeping fix. Durable W&B-traceable `ok` rows are preserved (the TSV is append-only); a fresh-timestamp `ok` row is appended carrying the prior real-run's metrics so the runner sees the row as complete under its latest-row-wins rule.
7. **State-file races between concurrent lanes.** The original single-state-file design (`experiments/state.json`) was rewritten to per-lane state files (`experiments/state.lane.lane-*.json`) when the runner moved from one to five concurrent lanes. The root `state.json` now holds runner-level summary plus a `lanes` map of per-lane summaries.
8. **GPT-2 BatchTopK anchor deviates from Leask et al.** See Section 3.3, item 1. Open. Investigation pending; the cheapest discriminator is the level-0-swap test described there.

### Anticipated, not yet materialized

9. **Submission-day pressure if `l0_gemma_batchtopk` slips further.** Current ETA leaves a ~38-hour cushion (matrix completion ~05:00 UTC 2026-05-08 versus 21:00 UTC 2026-05-09 deadline). Another total-loss restart of `l0_gemma_batchtopk` would consume nearly all of it.
10. **Decision-gate threshold relaxations may have been over-permissive.** The anchor floor moved 0.50 → 0.20 → 0.10 across two days, and d1/d2 cascade actions were softened from `skip_depth` to `skip_experiment` (`Decisions 9, 10`). The relaxations were correct given the data, but they shift the burden of recipe-collapse detection from automated gates to manual review of `experiments/results.tsv`.

---

# Submission Paper (IEEE Conference Template)

*Everything below this line is the eventual conference-style paper and is intentionally a skeleton at the time of this submission. Sections fill in only after `experiments/results.tsv` closes (target: 2026-05-08 ~05:00 UTC). The skeleton is the standard IEEE conference template: abstract, index terms, introduction, related work, methods, experimental setup, results, discussion, limitations, conclusion, contributions, references, appendix. The progress writeup above is the contentful half of the document for the current submission.*

## Abstract

*To be written after results are in. Target length: 150 to 200 words. Must state: (i) the question (does recursive meta-SAE decomposition bottom out?), (ii) the setup (two base models, two level-0 architectures, three depths, three seeds, anchor to Leask et al.), (iii) the headline quantitative result (separation of PW-MCC from null at each depth), (iv) the answer (the depth at which the recursion bottoms out or a statement that it does not within the tested range), and (v) one sentence on implications for interpretability.*

## Index Terms

*To be finalized. Draft: sparse autoencoders, mechanistic interpretability, meta-sparse-autoencoders, feature decomposition, large language models, Gemma-2-2B, GPT-2.*

## I. Introduction

*Placeholder.*

Open with the interpretability-via-SAE research agenda [@bricken2023monosemanticity; @cunningham2024sparse]. Introduce feature absorption and the motivation for meta-decomposition [@chanin2024absorption]. Establish the single open question: whether recursion continues to produce meaningful decomposition past depth one. State contributions at a high level. Forward-reference each section.

## II. Related Work

*Placeholder. Subsections to write:*

*A. Sparse autoencoders on language model activations.* [@bricken2023monosemanticity; @cunningham2024sparse]

*B. Architectural variants: JumpReLU, TopK, BatchTopK, Matryoshka.* [@rajamanoharan2024jumping; @gao2024scaling; @bussmann2024batchtopk; @bussmann2025learning]

*C. Meta-SAEs and recursive decomposition.* [@bussmann2024showingsaeslearn; @leask2025sparse]

*D. Evaluation: stability metrics, autointerp, and benchmarks.* [@sharkey2023taking; @beiderman2025pwmcc; @paulo2025sparse; @karvonen2025saebench]

*E. Position of the present work.* What has not been measured. Gap this paper fills.

## III. Methods

*Placeholder. Subsections to write:*

*A. Base models and activation source.* Gemma-2-2B [@gemmateam2024gemma], layer 12 residual; GPT-2 Small [@radford2019language], layer 8 residual. The Pile [@gao2020pile] as the token source, 500M tokens.

*B. Level-0 SAEs.* Gemma Scope JumpReLU SAEs [@lieberum2024gemma], BatchTopK SAEs trained following Bussmann et al. [@bussmann2024batchtopk].

*C. Meta-SAE training.* Width progression at dictionary ratio $1/4$, BatchTopK activation with $k=4$, deterministic seeding, Adam at learning rate $3 \times 10^{-4}$.

*D. Matching and stability metrics.* Joint encoder-plus-decoder cosine similarity Hungarian matching [@paulo2025sparse]. PW-MCC and MMCS formulations [@beiderman2025pwmcc; @sharkey2023taking]. Isotropic Gaussian null baselines.

*E. Autointerp scoring.* Llama-3.1-8B-Instruct [@dubey2024llama] detection-score protocol [@paulo2025sparse], 200 features per SAE, depths 1 and 2.

*F. Anchor experiment.* Reproduction of Leask et al. [@leask2025sparse] at dictionary ratio $1/21$ on GPT-2 Small as pipeline validation.

## IV. Experimental Setup

### A. Hardware and software environment

All experiments run on a single workstation: AMD Ryzen Threadripper 7970X (32 cores), 128 GB DDR5, five NVIDIA GPUs (2x RTX 4080 16 GB, 2x RTX 3090 24 GB, 1x RTX 4060 Ti 16 GB; 96 GB total VRAM). Operating system: Ubuntu 24.04. CUDA 12.x. Python 3.11+ managed via [uv](https://docs.astral.sh/uv/). Core dependencies are SAELens [@bloom2024saelens] and TransformerLens [@nanda2022transformerlens] with version pins in `pyproject.toml`. Training and evaluation code lives in `src/`; orchestration in `scripts/`. Bootstrap procedure is documented in `README.md`.

GPU assignment is fixed per experiment via `CUDA_VISIBLE_DEVICES`, with the lane-to-GPU mapping documented in `CLAUDE.md`. Lanes 0 and 1 (RTX 4080) handle meta-SAE training; lanes 2a and 2b (RTX 3090, sharded via `transformer_lens` `n_devices=2` for `l0_gemma_batchtopk`) handle level-0 BatchTopK on Gemma-2-2B; lane 3 (RTX 3090) handles autointerp; lane 4 (RTX 4060 Ti) handles flat-SAE controls and metric computation.

### B. Experimental matrix

The 75-row matrix in `EXPERIMENTS.yaml` partitions into:

| Group | Rows | Notes |
|---|---|---|
| Level-0 SAEs | 2 | One BatchTopK trained from scratch on each base model. The Gemma JumpReLU level-0 is loaded from Gemma Scope [@lieberum2024gemma], not trained. |
| Anchor reproductions at ratio 1/21 | 9 | 3 seeds × 3 cells (Gemma JumpReLU, Gemma BatchTopK, GPT-2 BatchTopK). The GPT-2 BatchTopK anchor is the published Leask et al. [@leask2025sparse] configuration. |
| Recursive meta-SAEs at ratio 1/4 | 27 | 3 (base model, level-0 architecture) cells × 3 depths × 3 seeds. The primary scientific quantity. |
| Flat-SAE width controls | 12 | Direct SAEs on activations at widths matched to recursive d1/d2/d3 latent counts. |
| Isotropic Gaussian null baselines | 18 | 2 base models × 3 depths × 3 seeds. |
| Ratio-sensitivity ablations | 6 | Ratios 1/2 and 1/8 at d1 only, on Gemma BatchTopK and GPT-2 BatchTopK. |
| Conditional stretch goal | 1 | `simplestories_stretch`; runs only if matrix completes with slack. |

The matrix is locked: the runner does not invent experiments. Additions are made by human edit to `EXPERIMENTS.yaml` and committed under a non-`[SAE-LOOP]` prefix per the protected-paths guard in `scripts/pre_commit_immutability_guard.sh`. The full row-by-row breakdown (dependencies, decision gates, estimated GPU-hours per row) is in `EXPERIMENTS.yaml`.

Pile [@gao2020pile] tokens are the activation source for level-0 SAEs (500M tokens for Gemma-2-2B, 200M for GPT-2 Small, per `src/training/train_level0_batchtopk.py:TOKEN_BUDGET`). Decoder rows of each parent SAE are the input to the next-depth meta-SAE.

### C. Decision gates

Each row in `EXPERIMENTS.yaml` carries a `decision_gates` block specifying thresholds on `variance_explained` and `pwmcc_vs_null_sigma`, with an `action` per gate from {`continue_depth`, `skip_depth`, `skip_experiment`, `halt_and_notify`}. Gates fire after the row's metrics are appended to `experiments/results.tsv`. Action semantics are:

- `continue_depth`: record the deviation but proceed with the depth chain.
- `skip_depth`: cascade-skip rows whose dependencies include the firing row's lineage at deeper depth.
- `skip_experiment`: record the firing row's deviation but do not propagate to downstream rows.
- `halt_and_notify`: stop the runner, send an `ntfy` urgent message, await human review.

Threshold history during execution: anchor-deviation gates relaxed from 5pp → 10pp on 2026-04-26 (`DECISIONS.md` 2026-04-26 entries). Recursive d1 and d2 cascade actions changed from `skip_depth` to `skip_experiment` on 2026-04-28 (`Decisions 9` and `10` in the same file), severing cascades that would have prevented seed-1 recoveries. Anchor `variance_explained` floor relaxed in three steps: 0.50 → 0.20 → 0.10. The relaxations were necessary because the recipe-typical VE at ratio 1/4 turned out to be far below the prior assumption, but they shift recipe-collapse detection from automated gates to manual review (see Section 6, item 10).

### D. Reproducibility

`experiments/results.tsv` is the durable, append-only ground truth for every quantitative claim in this paper. Each row carries the experiment ID, status, base model, level-0 architecture, depth, seed, width, variance explained, PW-MCC, MMCS, dead-latent fraction, GPU-hours, commit SHA, W&B run URL, and free-form notes (header at line 1 of the file). Every figure in `experiments/artifacts/_summary/figures/` ships with a matching `.tsv` or `.csv` of source data carrying the generation timestamp, git commit SHA, the analysis script path, and the `experiment_id` set consumed.

[Weights & Biases](https://wandb.ai) mirrors results during execution but is not the canonical record; if a number appears in this paper without a corresponding row in `results.tsv`, it is a writing error.

The runner is deterministic at the row level: each row's `seed` field controls all random number generation via `src/training/seed.py`; the activation pipeline is deterministic under fixed token-shard input; decoder-row ordering is fixed by `np.argsort` over a deterministic norm. The set of completed rows at any given commit is reproducible from `EXPERIMENTS.yaml` plus the orchestration scripts in `scripts/`. The 5-lane parallel autopilot (`scripts/run_autopilot.sh`) introduces row-claim non-determinism in *which* lane picks up *which* row, but per-row computation remains seed-deterministic.

## V. Results

*Placeholder, to be filled in by the runner and summary analysis scripts as they complete.*

*A. Anchor reproduction.* Observed variance explained versus Leask et al. [@leask2025sparse] 55.47% target.

*B. PW-MCC versus depth.* Figure showing PW-MCC by depth for each (base model, level-0 architecture) cell, with null baseline shaded region.

*C. MMCS versus depth.* Same format as B, with MMCS.

*D. Variance explained and dead latent fraction versus depth.* Supporting metrics.

*E. Autointerp detection scores.* Comparison across depths, against flat-SAE width controls and null baselines.

*F. Sensitivity to dictionary ratio.* $1/2$ and $1/8$ at depth 1 on Gemma BatchTopK.

## VI. Discussion

*Placeholder. This is where I interpret results and explicitly state which of the four predicted answer shapes the data support, how confident the separation is, and how Gemma and GPT-2 Small compare. Also where I discuss what the depth-bottoming-out result (whatever it turns out to be) implies for the interpretability research agenda.*

## VII. Limitations

*Placeholder. Anticipated items: three seeds is a lower bound on reliable variance estimation, 500M-token budget is modest relative to production SAE training, two base models is thin replication, autointerp is judge-dependent and the judge (Llama-3.1-8B-Instruct) is itself fallible, the Pile is not a perfectly representative corpus.*

## VIII. Conclusion

*Placeholder. One paragraph restating the question, one paragraph stating the answer, one paragraph on what comes next.*

## Contributions

Aragorn Wang conceived the project, designed the experimental matrix, wrote the runner infrastructure and all training/evaluation code, and wrote the paper. The autonomous experimental runner was orchestrated using Claude Code (Anthropic) under direct supervision, executing the locked matrix in `EXPERIMENTS.yaml`; all decisions about experimental design, metrics, anchors, and interpretation are the author's.

The project advisor, MATH 498C course instructor at the Colorado School of Mines, provided feedback on the proposal.

The author thanks the authors of Gemma Scope [@lieberum2024gemma], SAELens [@bloom2024saelens], and TransformerLens [@nanda2022transformerlens] for releasing tools and weights that made this project tractable at undergraduate scale.

## References

*References are stored in `references.bib` and cited throughout with pandoc-resolvable `@key` citations. Pandoc compiles to LaTeX `\cite{key}` commands via the `--biblatex` flag. The full bibliography is rendered at compile time by biber.*

## Appendix

*Placeholder. Appendix is reserved for extra figures, derivations that do not fit in the main body, and additional discussion. No code goes in the appendix; the repository is the code artifact.*

*Planned appendix sections, subject to space:*

*A. Complete experimental matrix.* Table form of `EXPERIMENTS.yaml`.

*B. Per-experiment variance-explained and dead-latent values.* Table form of the corresponding columns in `results.tsv`.

*C. Derivation: null-baseline expected PW-MCC under isotropic Gaussian draws at matched $d_{\mathrm{model}}$ and sample count.*

*D. Additional autointerp score distributions.* Per-feature histograms for the SAEs most distant from null.
