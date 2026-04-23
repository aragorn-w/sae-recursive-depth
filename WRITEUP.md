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

*This half of the document is a running log of research progress and will shrink or fold into the paper body as results come in. The IEEE conference template sections below are the eventual submission artifact. Current state: day zero of the three-week execution window, 2026-04-14.*

## 1. Question and relationship to existing literature

Sparse autoencoders decompose neural network activations into overcomplete dictionaries of nominally monosemantic features [@bricken2023monosemanticity; @cunningham2024sparse]. A recent line of work treats a trained SAE's decoder columns as signals in their own right and trains a second SAE on them, exposing finer-grained structure that the original SAE had composed into bundled features. This is the meta-SAE construction of Bussmann et al. [@bussmann2024showingsaeslearn] and the motivating apparatus for Leask et al. [@leask2025sparse], who report that meta-SAEs recover decomposition consistent with lower-width SAEs trained directly on activations.

The open question this project addresses is whether recursive meta-decomposition continues to surface meaningful structure past depth one. The meta-SAE literature has so far stopped at depth one. Either the recursion converges to a stable finer-grained basis at some depth, or it collapses into seed-dependent noise, or it exhibits some other failure mode (e.g., dead-latent explosion, total variance-explained collapse). Nobody has reported which. Answering that is the project's contribution.

The question reduces to a concrete empirical comparison. For each base model, level-0 architecture, and depth in $\{1, 2, 3\}$, train three seeds of a meta-SAE at dictionary ratio $1/4$ of the parent, apply the Paulo and Belrose joint-similarity Hungarian matching protocol [@paulo2025sparse], and compute PW-MCC [@beiderman2025pwmcc] and MMCS [@sharkey2023taking] against both matched seeds and an isotropic Gaussian null baseline. If PW-MCC at depth $d$ is not distinguishable from the null at $\geq 3\sigma$, the recursion has bottomed out at depth $d$. If PW-MCC remains separated from null at all three depths, the recursion is still productive and the true ceiling is beyond depth 3.

Relationship to existing methods: the level-0 SAEs reuse released Gemma Scope JumpReLU weights [@lieberum2024gemma] on Gemma-2-2B and a trained-from-scratch Bussmann-style BatchTopK SAE [@bussmann2024batchtopk] on GPT-2 Small (with an anchor experiment at dictionary ratio $1/21$ reproducing Leask et al. [@leask2025sparse]). Meta-SAE training follows Matryoshka-SAE conventions [@bussmann2025learning] with BatchTopK activation. Autointerp scoring uses Paulo and Belrose's detection protocol [@paulo2025sparse] with Llama-3.1-8B-Instruct [@dubey2024llama] running locally on two RTX 3090s. Cross-architecture and cross-dictionary-width null baselines follow SAEBench conventions [@karvonen2025saebench].

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

No experiments run yet. Today is day zero (2026-04-14). Infrastructure is complete (see `README.md` for the codebase map, `SPEC.md` for the runner contract, `CLAUDE.md` for agent guidance), but no training runs have been dispatched. `experiments/results.tsv` does not yet exist. No conclusions can be drawn.

This section will be expanded as the runner populates `results.tsv`. The plan is to report, in order of execution:

1. Anchor reproduction on GPT-2 Small BatchTopK at ratio $1/21$. This must match Leask et al. [@leask2025sparse] within 5 percentage points of variance explained or the matrix halts.
2. Level-0 BatchTopK on Gemma-2-2B at ratio $1/4$. This is the most expensive single run (estimated 12 GPU-hours) and feeds every Gemma BatchTopK meta-SAE downstream.
3. The 18 main-matrix Gemma meta-SAE runs (JumpReLU and BatchTopK, depths 1 through 3, three seeds each).
4. The 9 GPT-2 Small BatchTopK meta-SAE runs (depths 1 through 3, three seeds each).
5. Flat-SAE width controls (12 runs) and null baselines (18 runs).
6. Autointerp at depths 1 and 2.

## 4. Experiments left to run

Everything. The 75 rows in `EXPERIMENTS.yaml` plus autointerp plus the conditional SimpleStories stretch experiment. As rows complete and are checked in to `experiments/results.tsv`, they move from this section up into Section 3.

## 5. How the answer depends on those experiments

The question "does the recursion bottom out, and if so, at what depth?" has four mutually exclusive answer shapes, each predicted by a specific pattern of PW-MCC values across depth:

1. **Immediate collapse.** PW-MCC at depth 1 is not distinguishable from the isotropic Gaussian null at $\geq 3\sigma$. This would imply that the meta-SAE construction is essentially learning noise even at depth 1, contradicting Bussmann et al. [@bussmann2024showingsaeslearn] and Leask et al. [@leask2025sparse]. If observed, the first explanation to rule out is a pipeline bug, starting with the anchor experiment at ratio $1/21$.

2. **Depth-1 success, depth-2 collapse.** PW-MCC at depth 1 is well-separated from null, PW-MCC at depth 2 is not. The recursion bottoms out at depth 1 and meta-SAEs should not be stacked.

3. **Graceful degradation.** PW-MCC decreases monotonically with depth but stays separated from null through depth 3. The recursion remains productive but each additional level is less stable than the last. In this case the answer is "deeper is still possible, but returns diminish", and I would predict a crossover depth beyond 3 where null is matched.

4. **Plateau.** PW-MCC stays roughly flat across depth, well above null. The recursion has essentially converged and depth 3 is close to the stable fine-grained basis. Autointerp scores should also plateau.

Secondary evidence that will shape the conclusion regardless of which shape wins: (a) whether Gemma-2-2B and GPT-2 Small exhibit the same shape (replication across base model), (b) whether JumpReLU and BatchTopK level-0 architectures exhibit the same shape (replication across level-0 architecture), (c) whether dead-latent fraction and variance-explained trends corroborate the PW-MCC trend, and (d) whether autointerp detection scores at depth 2 exceed the depth-1 scores, flat-SAE control scores, and null baseline scores by statistically significant margins.

## 6. Roadblocks so far

1. **Gemma-2-2B is a gated HuggingFace repo.** Bootstrap requires `huggingface-cli login` plus acceptance of the license at https://huggingface.co/google/gemma-2-2b. Not a real blocker, but a manual step that has to happen before the runner dispatches any Gemma experiments.

2. **Gemma Scope JumpReLU SAE weight availability at width 65536 on layer 12 residual.** The published release covers this exact configuration, but any upstream change to the Gemma Scope HuggingFace repository structure would require loader edits. The `src/training/loaders.py` module is planned to pin the expected file paths and error clearly if they have moved.

3. **No published JumpReLU BatchTopK-style anchor.** The anchor experiment at ratio $1/21$ replicates Leask et al. [@leask2025sparse] on BatchTopK. There is no equivalent published result for JumpReLU meta-SAEs at that ratio, so a deviation on a hypothetical JumpReLU anchor is treated as "investigate" rather than "pipeline broken". See `DECISIONS.md` entry 5.

4. **41.55 GPU-hour estimate is tight.** Wall-clock of roughly 9 to 12 days on the 5-GPU rig if everything runs clean, leaving 9 to 12 days of slack in the three-week window. One stuck run that takes 24 hours to fail and recover eats meaningfully into that slack. The runner has three-attempt exponential backoff and halves batch size on CUDA OOM, but if level-0 BatchTopK on Gemma-2-2B takes substantially longer than the 12-hour estimate, the matrix will have to shed the stretch goal and possibly one of the sensitivity-check dictionary ratios.

5. **Autointerp dependency chain.** Autointerp runs on GPUs 2 and 3, which are also the GPUs used for level-0 BatchTopK training on Gemma. Autointerp cannot start until level-0 training has freed those GPUs. The dependency is correctly encoded in `EXPERIMENTS.yaml` but it means autointerp is the most schedule-sensitive step; any delay in level-0 training pushes autointerp closer to the submission deadline.

---

# Submission Paper (IEEE Conference Template)

*This half is the eventual conference-style paper. Sections below are blank until their corresponding experimental evidence is in `experiments/results.tsv`. Follows the standard IEEE conference template: abstract, index terms, introduction, related work, methods, experiments, results, discussion, conclusion, contributions, references, appendix.*

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

*Placeholder. Subsections to write:*

*A. Hardware and software environment.* Threadripper 7970X, 2x RTX 4080, 2x RTX 3090, 1x RTX 4060 Ti, 96 GB total VRAM. SAELens $\geq 6.0$ [@bloom2024saelens], TransformerLens $\geq 2.0$ [@nanda2022transformerlens]. Ubuntu 24.04.

*B. Experimental matrix.* 75 runs total: breakdown by base model, level-0 architecture, depth, seed, and control type.

*C. Decision gates.* Halt conditions and skip conditions applied by the autonomous runner per `SPEC.md`.

*D. Reproducibility.* All results appended to `experiments/results.tsv`. All figures generated from that file. Git commit SHA and timestamp recorded on every figure's data companion.

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

Aragorn Wang conceived the project, designed the experimental matrix, wrote the runner infrastructure and all training/evaluation code, and wrote the paper. The autonomous runner was orchestrated using Claude Code (Anthropic) under direct supervision; all decisions about experimental design, metrics, anchors, and interpretation are the author's. Claude Code executed the locked matrix in `EXPERIMENTS.yaml` and was not authorized to modify experimental design, results, or the writeup narrative during autonomous operation.

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
