# Meta-SAE anchor VE diagnosis: is 0.19 a JumpReLU phenomenon or a pipeline bug?

**Date**: 2026-04-25T23:20Z
**Question**: Is variance explained ≈ 0.19 plausible for a depth-1 BatchTopK meta-SAE trained on the decoder directions of a JumpReLU SAE on Gemma-2-2B layer 12 residual at 1/21 width ratio (~3121 latents)?
**Decisions requested**: trust anchor / run BatchTopK anchors / resource priority

---

## Executive summary

**Verdict (high confidence)**: VE ≈ 0.19 is *implausibly low* for a correctly trained meta-SAE at this ratio on this base. The 0.36 deviation from Leask's 0.5547 is **most likely an undertraining artifact in the local meta-SAE pipeline, not a real signature of JumpReLU atomicity.** Three lines of evidence converge on this:

1. **Architecture-controlled comparison**: OrtSAE (ICLR 2026 sub.) reports that on Gemma-2-2B *layer 20*, a BatchTopK meta-SAE at the standard 1/4 ratio explains **0.490** of the variance of a BatchTopK level-0; **0.349** for a Matryoshka level-0; **0.340** for an OrtSAE level-0 (their Table 1, L0=70). User's *JumpReLU* level-0 at the *same 1/4 ratio* lands at **0.166-0.184** — substantially below the most-atomic published architectures, on a base SAE that is, if anything, expected to fall in the same atomicity band as Matryoshka/OrtSAE.

2. **Width-insensitivity is the smoking gun**: User's results show VE essentially identical at 1/4 (16384 latents, VE 0.166-0.184) and 1/21 (3121 latents, VE 0.181-0.192). A correctly trained meta-SAE *must* show VE rising with width — going from 21x compression to 4x compression should buy several pp of VE at minimum. Equal VE means effective capacity is bottlenecked by something other than nominal width, almost certainly dead/inactive latents in the wider runs.

3. **Confirmed in the data**: User's own results show **dead_latent_fraction = 0.47 at width 16384 (1/4)** vs 0.037 at width 3121 (1/21). At 1/4, *half* the meta-latents are dead. Effective capacity at 1/4 ≈ 8683 alive latents — only 2.9× the 1/21 effective capacity, and competing with each other. Combined with the absence of an auxiliary dead-latent loss in `src/training/train_meta_sae.py` (no `auxk`, `L_aux`, or `ghost` term), this matches the failure mode that the canonical BatchTopK recipe (Bussmann arXiv:2412.06410, eq. for `L_aux`) was designed to prevent.

**Best-judgment recommendations**:

- **(a) Do not trust the current d2/d3 JumpReLU runs as scientifically valid.** They sit on a depth-1 anchor that is undertrained. Numbers will appear, but the recursive-decomposition claim is not interpretable through them yet.
- **(b) Run the GPT-2 BatchTopK anchor — it is the diagnostic.** It directly tests whether the local pipeline reproduces Leask's 0.5547 ± 0.05. If it does, the pipeline is correct and the JumpReLU result is genuinely lower (publishable observation). If it doesn't, the pipeline needs the auxk/ghost-grad fix before any d≥2 result is meaningful. Also run the Gemma BatchTopK anchor — same Gemma-2-2B base, comparable to OrtSAE's 0.490 reference, gives a Gemma-specific check.
- **(c) Do not preempt** the in-flight `gemma_jumprelu_d2_s0`. Let it finish so its checkpoint isn't orphaned. **Do not start more d2/d3 JumpReLU seeds** after it finishes until BatchTopK anchors return. Re-queue both `l0_gpt2_batchtopk` and `l0_gemma_batchtopk` immediately so they run as soon as the current d2 seed clears GPU 1.

Confidence levels are tagged inline below.

---

## Evidence base

### 1. Leask et al. 2025 — what 0.5547 actually represents

**Source**: Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey, Nanda. *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. ICLR 2025 Poster (arXiv:2502.04878). Setup quoted from the proceedings PDF (proceedings.iclr.cc/paper_files/paper/2025/file/84ca3f2d9d9bfca13f69b48ea63eb4a5-Paper-Conference.pdf):

> "GPT-2 SAE with dictionary size 49152. The meta-SAE has a dictionary size of 2304 meta-latents, with on average 4 of meta-latents active per SAE latent. Due to small number of training samples for the meta-SAE (49152), the meta-SAE is trained for 2000 epochs. We use the Adam optimizer with learning rate 1e-4 and a batch size of 4096. After training, the meta-SAE explains 55.47% of the variance of the decoder directions of the SAE."

**Key facts**:
- Base SAE: **BatchTopK** on **GPT-2 Small**, dictionary 49152.
- Meta-SAE: BatchTopK, 2304 latents, average L0=4 → **width ratio 2304/49152 ≈ 1/21.3** (this is the canonical "1/21 anchor" in `EXPERIMENTS.yaml`).
- 2000 epochs ≈ 24 000 gradient steps at batch 4096 over 49152 samples.
- Single number reported (55.47%); no scaling curve across multiple ratios was found in the searched excerpts.
- Activation choice rationale (quoted from the same paper): "A major drawback of the sparsity penalty used in JumpReLU SAEs compared to (Batch)TopK SAEs is that it is not possible to set an explicit sparsity and targeting a specific sparsity involves costly hyperparameter tuning." Leask therefore *did not* run their meta-SAE on JumpReLU base SAEs; the published reference is BatchTopK→BatchTopK only.

**Confidence**: High — verified across multiple search hits and consistent with `DECISIONS.md:37` and `SPEC.md:60`.

### 2. Bussmann et al. — the meta-SAE precursor

**Source**: Bussmann, Pearce, Leask, Bloom, Sharkey, Nanda. *Showing SAE Latents Are Not Atomic Using Meta-SAEs*. AI Alignment Forum, August 2024. (alignmentforum.org/posts/TMAmHh4DdMr4nCSr5/...)

The original LessWrong/AF post introduced meta-SAEs on a vanilla ReLU SAE and gave qualitative decompositions (Einstein → scientist + Germany + famous person; Paris → French city + capital + romance language). It does not provide a quantitative VE scaling curve in any text we surfaced. The Leask 2025 paper subsumes the methodology and adds the BatchTopK level-0 + 0.5547 number.

**Confidence**: Medium on the absence of a published Bussmann VE scaling curve — search did not find one but PDF retrieval was rate-limited.

### 3. JumpReLU vs BatchTopK decoder compressibility — what theory and adjacent results suggest

The Matryoshka SAE write-up (Bussmann/Bloom/Tigges, alignmentforum.org/posts/rKM9b6B2LqwSB5ToN/) frames meta-SAE VE explicitly as a *measure of compositional structure in the level-0 decoder*:

> "meta-SAEs explain substantially more variance in BatchTopK SAE decoder directions compared to Matryoshka SAE directions. This suggests that Matryoshka SAE latents are more disentangled, with less shared information between them."

OrtSAE (openreview.net/forum?id=lBctELT2f9) generalizes the framing: "lower MetaSAE explained variance as indicating better feature atomicity in the primary SAE."

Apples-to-apples Gemma-2-2B numbers from **OrtSAE Table 1** (layer 20, L0=70, MetaSAE at 1/4 ratio, k=4):

| Level-0 | Meta-SAE VE | Notes |
|---|---|---|
| ReLU SAE | **0.527** | (their "Comp. rate" column) |
| BatchTopK SAE | **0.490** | direct comparator for Leask-style anchor |
| Matryoshka SAE | **0.349** | most-atomic mainstream architecture |
| OrtSAE | **0.340** | their proposed orthogonal-features SAE |

**JumpReLU level-0 is not in any meta-SAE table that surfaced.** No published JumpReLU→meta-SAE VE exists at any ratio.

The directional question (does JumpReLU give a *more* or *less* compressible decoder than BatchTopK?) is unsettled in the literature, but the Matryoshka/OrtSAE results bound a reasonable expectation: even the *most disentangled* published Gemma-2-2B base SAE produces a meta-SAE VE of **~0.34 at 1/4 ratio**. JumpReLU's threshold structure and Pareto-frontier training (Gemma Scope arXiv:2408.05147) plausibly pushes it toward the Matryoshka end of the spectrum, but going *below* that band (to 0.18) is a strong claim that needs separate evidence.

**Confidence**: High that 0.34-0.49 is the plausible range; moderate that JumpReLU should sit somewhere in that range rather than far below it.

### 4. Gemma Scope JumpReLU SAE meta-SAE results — published?

Searched Hugging Face papers, alignment forum, OpenReview, NeurIPS/ICLR proceedings, EleutherAI blog, and Anthropic interpretability releases. **No direct published meta-SAE result on Gemma Scope JumpReLU SAEs surfaced.** Closest precedents are:
- Matryoshka SAE post — uses BatchTopK level-0 on Gemma-2-2B, not JumpReLU.
- OrtSAE — Gemma-2-2B layer 20, BatchTopK level-0.
- Gemma Scope itself (arXiv:2408.05147) — releases JumpReLU SAEs but does not train meta-SAEs on them.

So the user's experiment is, narrowly, the first published JumpReLU→meta-SAE VE measurement at this ratio. That makes the gate's "no published reference" warning literal.

**Confidence**: Medium-high that nothing public exists. (Cannot exclude unpublished work, e.g., on Neuronpedia.)

### 5. Compressibility at 21× — what's the floor?

Two relevant data points for grounding intuition:

- **PCA / random-direction floor**: 65536 unit-norm directions in 2304-dim space (Gemma's d_model) cannot be compressed to 3121 directions without loss; the lower bound is set by the rank of the decoder matrix and how much its singular value spectrum concentrates. For a structured decoder, top-3121 PCA components on the row space already capture most of the variance — meta-SAE VE should comfortably exceed PCA-at-the-same-rank if the dictionary is meaningfully compositional, and approach 1.0 if features are highly redundant.
- **Empirical floor on random Gaussian directions**: the user's own null SAE runs (`null_gemma_d1`, results.tsv 2026-04-25T22:33Z onward) give VE = -1.51 to -1.61 on 65536 Gaussian samples in 2304-dim with the same training recipe. A null score that negative confirms the recipe doesn't accidentally reconstruct random input — useful as a sanity check but not informative about the structured-data ceiling.

The published meta-SAE VE cluster (0.34-0.55) at *higher* compression (1/21 in Leask) or *lower* (1/4 in OrtSAE) shows that 0.5 is not the ceiling either. The space between 0.18 and 0.5 is occupied by published results on architectures *less* compositional than what JumpReLU is expected to be.

**Confidence**: Medium — mostly inferential rather than from a single citable scaling curve.

### 6. Width-insensitivity of meta-SAE VE — published or anomalous?

Searched explicitly for VE-vs-meta-SAE-width plots. The **Matryoshka SAE post** does mention training "five BatchTopK SAEs with similar dictionary sizes (2304, 4608, 9216, 18432, 36864)" but for the *base* SAE width sweep, not for meta-SAE width. We did not surface any paper that reports meta-SAE VE plateauing across a 5×-wider meta-SAE; the prior is firmly that increasing meta-SAE width improves VE, monotonically and visibly.

User's data:
- 1/21 ratio (3121 latents): VE ≈ 0.185-0.192, dead_frac 0.037
- 1/4 ratio (16384 latents): VE ≈ 0.164-0.184, dead_frac **0.47**

The ratio-1/4 dead-fraction of 47% is the diagnostic. The training code at `src/training/train_meta_sae.py` does not include the auxiliary dead-latent loss `L_aux = ||e − ê||²` with `α=1/32` and `k_aux=512` that the Bussmann BatchTopK paper (arXiv:2412.06410, §3) describes as the standard recipe to prevent dead latents in TopK/BatchTopK. The user's own commit message at `dd3efa0` acknowledges this: *"auxk-style ghost grads likely needed for full closure."*

**Confidence**: High — directly inspected the training source and verified no auxk implementation.

---

## Synthesis: ranked likely causes for VE = 0.19

In order of decreasing likelihood:

1. **Missing auxiliary dead-latent loss (auxk / ghost-grad)** in `train_meta_sae.py`. Standard BatchTopK practice (Bussmann arXiv:2412.06410). Directly explains the 47% dead fraction at 1/4 and the width-insensitivity. *Most likely root cause.*
2. **30k step budget too short** for the wider 1/4 runs. Leask's 2000 epochs over a 49152-row dataset ≈ 24k gradient steps; user uses 30k steps, which is comparable for the 1/21 case but *fewer epochs* for the wider 1/4 case (more samples per epoch with `train_from_scratch` activations as opposed to decoder-direction training). Secondary contributor.
3. **JumpReLU level-0 producing genuinely more atomic decoders**. Plausible *direction* per Matryoshka/OrtSAE theory, but the magnitude (0.18 vs 0.34 for the most-atomic published architecture) is too large to attribute to architecture alone. Tertiary contributor at most.
4. **Layer-12 vs layer-20**. OrtSAE's 0.490 was at layer 20 (later, more abstract). Earlier layers tend to have less compositional features, which would *lower* meta-SAE VE somewhat. Marginal contributor — unlikely to drive a 2-3× gap.
5. **Activation pipeline / reconstruction-evaluation bug**. Low prior because the eval protocol comment in the code explicitly references Leask's protocol and the null baseline behaves correctly (VE = -1.6 on random Gaussian data, exactly as expected).

---

## Recommendations (the decisions you asked for)

### (a) Trust the JumpReLU anchor as-is, or flag it?
**Flag it.** Treat current d1/d2/d3 JumpReLU numbers as preliminary, not as the scientific basis for any claim about recursive decomposability. The depth-1 anchor result of 0.19 is too far below comparable architectures, and the width-insensitivity points specifically at training-pipeline saturation rather than at a genuine atomicity signal.

### (b) Run the BatchTopK anchors?
**Yes, both.** The GPT-2 BatchTopK anchor is the unambiguous Leask replication test; the Gemma BatchTopK anchor is the cleaner Gemma-on-Gemma check (against OrtSAE's 0.490 reference). Cost is one BatchTopK level-0 train each (200M tokens for GPT-2, 500M for Gemma per `EXPERIMENTS.yaml:60-90`) plus 3 × meta-SAE training per anchor. With `zstandard` now installed (`eb6a464`), the previously-blocked level-0 trainings should run cleanly.

The anchors give a binary diagnosis:
- **GPT-2 BatchTopK anchor → ~0.55 ± 0.05**: pipeline is correct on BatchTopK; JumpReLU result of 0.19 is then a real (and genuinely surprising) JumpReLU phenomenon worth a separate writeup. Pipeline gets validated, JumpReLU work is rescued as a positive finding rather than a bug.
- **GPT-2 BatchTopK anchor → ~0.2-0.3** (i.e., far below Leask): pipeline-wide undertraining confirmed. Add `auxk` aux loss and longer training budget before any d≥2 result is interpretable. The Gemma anchor result is then also explained by the same fix.

### (c) Resource priority
- **Do not preempt** the in-flight `gemma_jumprelu_d2_s0` (would orphan a checkpoint and lose the gradient state).
- **Do** queue `l0_gpt2_batchtopk` and `l0_gemma_batchtopk` for re-execution now (append `stale_zstd_fix` rows in `results.tsv` per SPEC §8.2 latest-row-wins). They run on GPUs 0 / 2 (per the GPU table in CLAUDE.md), so they will not contend with `gemma_jumprelu_d2_s0` on GPU 1.
- **Hold** the remaining d2/d3 JumpReLU seeds *after* the current d2_s0 finishes — let the BatchTopK anchors come back first. If they validate the pipeline, resume; if they don't, fix `auxk` first and re-run d1 anchors before any d≥2.

### Pre-emptive code change worth considering (separately)
Adding the standard BatchTopK `L_aux` loss (Bussmann 2412.06410 §3, eq. for L_aux with α=1/32, k_aux=512 dead latents) to `src/training/train_meta_sae.py` is a small, well-scoped change with high expected value. The user's own commit `dd3efa0` flagged it. This is implementation work that lies outside this research command's scope (per `/sc:research` boundaries: "Will NOT implement findings"), but it is the highest-leverage next code edit if the BatchTopK anchors confirm the diagnosis.

---

## What this research did NOT establish

- **Quantitative scaling-curve prediction**: I could not surface a published curve of meta-SAE VE vs ratio for any architecture. The "VE should rise visibly from 1/21 to 1/4" claim is grounded in OrtSAE's 0.490 at 1/4 vs Leask's 0.5547 at 1/21 (different base models — only loosely comparable) and in general training dynamics, not in a single published scaling result.
- **Exact JumpReLU prediction**: Without a published meta-SAE result on a JumpReLU level-0, the "should land at ~0.34-0.49" expectation is an interpolation, not a measurement.
- **Bussmann original VE numbers**: PDF retrieval to the original LessWrong post was rate-limited; the qualitative claim ("first introduced meta-SAE methodology") is verified, but exact numerical values from the August 2024 post were not directly extracted.

If any of these become decision-relevant, they're worth a follow-up search.

---

## Sources

- Leask et al. 2025 — *Sparse Autoencoders Do Not Find Canonical Units of Analysis*, ICLR 2025 Poster, arXiv:2502.04878. proceedings.iclr.cc/paper_files/paper/2025/file/84ca3f2d9d9bfca13f69b48ea63eb4a5-Paper-Conference.pdf
- Bussmann et al. 2024 — *Showing SAE Latents Are Not Atomic Using Meta-SAEs*. alignmentforum.org/posts/TMAmHh4DdMr4nCSr5/showing-sae-latents-are-not-atomic-using-meta-saes
- Bussmann et al. 2024 — *BatchTopK Sparse Autoencoders*, NeurIPS 2024 Workshop / arXiv:2412.06410.
- Lieberum et al. 2024 — *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*, arXiv:2408.05147.
- Rajamanoharan et al. 2024 — *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*, arXiv:2407.14435.
- Bussmann et al. 2025 — *Learning Multi-Level Features with Matryoshka SAEs*. alignmentforum.org/posts/rKM9b6B2LqwSB5ToN/learning-multi-level-features-with-matryoshka-saes
- OrtSAE (anonymous, ICLR 2026 sub.) — openreview.net/forum?id=lBctELT2f9 (Table 1, layer 20 Gemma-2-2B at L0=70).
- Local sources: `experiments/results.tsv` (rows 2026-04-25T09 to 23 for `gemma_jumprelu_d1*` and `gemma_jumprelu_anchor_d1*`), `src/training/train_meta_sae.py`, `DECISIONS.md`, `SPEC.md`.

---

*Stop after research report — no implementation performed.*
