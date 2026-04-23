# DECISIONS.md

Chronological log of architectural and scientific decisions. Each entry preserves context so the runner and my future self can reason without re-reading the full proposal discussion. New entries go at the top.

## Format
```
### YYYY-MM-DD: decision title
Decision: <what was decided>
Rationale: <why>
What would change our minds: <concrete evidence that would reverse this>
```

---

### 2026-04-14: Gemma-2-2B as primary base model, GPT-2 Small as replication arm
Decision: Gemma-2-2B is the primary track for all depth-1/2/3 experiments on two level-0 architectures (JumpReLU and BatchTopK). GPT-2 Small is retained as a parallel replication track.
Rationale: The 2025 SAE literature has shifted decisively to Gemma-2-2B. SAEBench (Karvonen et al., ICML 2025, arXiv:2503.09532) canonizes it as the reference base model. Gemma Scope (arXiv:2408.05147) ships SAEs up to 2^20 width. GPT-2 Small is the Leask et al. testbed and provides direct continuity with the only published meta-SAE result, which matters for validating the pipeline. Running both gives a model-generality signal (H4) at modest extra cost because GPT-2 SAE training is cheap (d_model=768).
What would change our minds: If Gemma Scope JumpReLU SAEs are revealed to have systematic issues that make them unsuitable for meta-SAE training, or if the BatchTopK level-0 training on Gemma-2-2B fails consistently, we fall back to GPT-2 Small as the primary track.

### 2026-04-14: Single headless Claude Code runner, not a multi-agent hierarchy
Decision: One runner iterates `EXPERIMENTS.yaml` deterministically. No search agents, no speculative branches, no automatic experiment invention.
Rationale: The experimental matrix is fixed by the proposal. Autonomy helps with search, not with fixed-matrix execution. A deterministic runner is more defensible for reproducibility and scientific rigor. Multi-agent orchestration introduces failure modes (agent disagreement, race conditions, non-deterministic ordering) that would consume more time to debug than the runner would save.
What would change our minds: If mid-project I identify a genuinely exploratory subquestion (e.g., characterizing dark matter accumulation across depths) that benefits from parallel search, I would consider a separate, short-lived agent for that subtask, still gated by my review.

### 2026-04-14: 3 seeds per cell, not 5
Decision: N=3 seeds per (base_model, level-0 arch, depth) cell.
Rationale: With 3 seeds we get C(3,2)=3 pairwise comparisons per cell, enough for a mean and a rough variance. Paulo & Belrose (arXiv:2501.16615) used 2 seeds at 32K width on Pythia and 2 seeds at 131K on Llama and reported distinguishable cross-seed overlap, so 3 is already an improvement. Going to 5 seeds triples the compute on the depth-1 bottleneck without proportionally tightening the PW-MCC confidence intervals. The random-Gaussian null baselines (also 3 seeds at every depth) provide the statistical reference.
What would change our minds: If the 3-seed cross-seed variance at depth 1 is so wide that depth-2 and depth-3 seed comparisons are uninterpretable, add 2 more seeds at the most informative cells.

### 2026-04-14: 1/4 dictionary-size ratio per recursive step, with ratio ablation
Decision: Primary recursive experiments use a 1/4 ratio per depth (65K to 16K to 4K to 1K for Gemma; 49K to 12K to 3K to 768 for GPT-2 Small). Ratios 1/2 and 1/8 are ablated at depth 1 on Gemma BatchTopK. The 1/21 Bussmann ratio is a distinct anchor experiment (see next decision).
Rationale: Bussmann et al. used ~1/21 (49,152 to 2,304) at depth 1 only. At 1/21 from a 65K level-0, depth 2 would have ~149 latents (below d_model=2304 for Gemma, so degenerate), so 1/21 cannot recurse to depth 2. A 1/4 ratio gives oversampling ratios of 7.1x (d1), 1.8x (d2) on Gemma, and much higher on GPT-2 Small (d_model=768). The 1.8x depth-3 oversampling on Gemma is tight and is flagged as the primary statistical risk in the proposal.
What would change our minds: If the 1/2 and 1/8 ablation at depth 1 shows that depth-1 PW-MCC and variance explained are highly sensitive to the ratio, then depth-2 and depth-3 at 1/4 are harder to interpret, and we'd focus the contribution on the ratio itself as the primary variable.

### 2026-04-14: Depth-1 Bussmann anchor (ratio 1/21) is a distinct experiment, depth-1 only
Decision: At ratio 1/21, run depth-1 meta-SAEs with 3 seeds for each of (Gemma-JumpReLU, Gemma-BatchTopK, GPT-2-BatchTopK), yielding 9 anchor rows. No depth-2 or depth-3 at this ratio.
Rationale: The 1/21 ratio is the published Bussmann/Leask setup. On GPT-2 Small it directly replicates Leask et al.'s 2,340-latent meta-SAE and lets us validate that our pipeline reproduces their 55.47% variance-explained number (arXiv:2502.04878). On Gemma-2-2B it is a novel depth-1 configuration that makes the two base models comparable at the published ratio. It stops at depth 1 because 1/21 collapses beyond depth 1 as described above. The Gemma-JumpReLU anchor is marked `halt_and_notify` on a >5pp deviation from Leask et al.'s 55.47% because such a deviation indicates a BatchTopK level-0 implementation bug that would invalidate all downstream analysis. Note that the JumpReLU anchor does not have a published reference value to compare to (Leask et al. used BatchTopK); the `halt_and_notify` gate on `variance_explained_deviation_from_leask` fires per-seed and is most informative on the BatchTopK anchors. The JumpReLU anchors serve as the JumpReLU-vs-BatchTopK-at-1/21 comparison, not a Leask replication.
What would change our minds: If the GPT-2 Small anchor fails to reproduce 55.47% within 5pp, halt immediately because the pipeline has a bug. If both architectures on Gemma produce the same anchor numbers, the architecture variable is less interesting than the ratio variable and we'd reweight our analysis accordingly.

### 2026-04-14: Train BatchTopK level-0 on Gemma-2-2B in addition to loading Gemma Scope JumpReLU
Decision: Train one BatchTopK level-0 SAE from scratch on Gemma-2-2B layer 12 residual, width 65536, k=60, 500M Pile tokens.
Rationale: Gemma Scope ships only JumpReLU. The meta-SAE literature is built on BatchTopK. Comparing recursive decomposition trajectories between JumpReLU-level-0 and BatchTopK-level-0 meta-SAEs on the same base model directly tests whether the recursive structure (or its absence) is architecture-dependent. This gap is explicitly flagged in the Matryoshka SAE paper (arXiv:2503.17547).
What would change our minds: If the 12 GPU-hour level-0 training on Gemma-2-2B fails or produces a checkpoint that doesn't match published BatchTopK performance on other models, we drop the Gemma-BatchTopK arm and keep only Gemma-JumpReLU.

### 2026-04-14: Decision-gate philosophy: auto-skip on structural signals, halt only on bugs
Decision: Decision gates are infrastructure, not science. Gates fire on pre-specified thresholds (variance explained < 20%, PW-MCC within 2 sigma of null) and take one of four actions: `continue_depth`, `skip_depth`, `skip_experiment`, or `halt_and_notify`. Only `halt_and_notify` stops the runner, and it is reserved for cases indicating a pipeline bug (e.g., the Experiment 3 anchor deviating >5pp from Leask et al.).
Rationale: An experiment that fails a gate has produced data that answers the question ("structure does not persist at depth N"). The right response is to record it, skip deeper cells for that lineage, and move on. Halting for every gate would waste 4 weeks. Halting on pipeline-bug signals is correct because continuing would produce garbage.
What would change our minds: If a gate is triggered and I review the data and conclude the threshold was wrong, the human edits the gate in `EXPERIMENTS.yaml` and the runner picks up the revision on the next iteration.

### 2026-04-14: ntfy topic and heartbeat schedule
Decision: ntfy topic is `sae-wanga-research` on the public ntfy.sh server. Heartbeats at 8am and 8pm America/Denver via cron. Immediate alerts on gate triggers, milestones, and unrecoverable failures.
Rationale: 12-hour heartbeats are frequent enough to notice a stuck runner within half a day but sparse enough not to burn ntfy's public quota. 8am/8pm aligns with my day. America/Denver handles DST automatically. Public ntfy.sh is free and the topic name is unique enough to avoid collisions; if privacy becomes a concern I'll migrate to a self-hosted instance.
What would change our minds: If public ntfy.sh becomes flaky or I start running experiments involving sensitive model outputs, migrate to a self-hosted ntfy on the workstation.

### 2026-04-14: GPU assignment
Decision: GPUs 0-1 (2x RTX 4080 16GB) = meta-SAE training. GPUs 2-3 (2x RTX 3090 24GB) = level-0 BatchTopK on Gemma and autointerp. GPU 4 (RTX 4060 Ti 16GB) = evaluation, plotting, null baselines.
Rationale: Meta-SAE training is compute-bound and fits in 16 GB because decoder directions are the training data (a 65K x 2304 FP16 matrix is about 300 MB), so the 4080s' faster tensor cores win. Level-0 BatchTopK on Gemma-2-2B needs activation caching headroom that only 24 GB provides. Llama-3.1-8B-Instruct at FP16 (about 16 GB weights plus KV cache) fits cleanly on a 3090 with room for batching; two 3090s let us parallelize the 6,000 autointerp calls to roughly 5 hours total. The 4060 Ti is the coolest-running card and is ideal for long analysis and plotting tasks that don't need fast tensor throughput.
What would change our minds: If Gemma-2-2B BatchTopK level-0 turns out to need more than 24 GB at batch 2048, drop the batch size or run it on two 3090s in parallel.

### 2026-04-14: Git auto-commits with immutability guard
Decision: The runner commits its own generated artifacts under a `[SAE-LOOP]` prefix. A pre-commit hook diffs the staged changes against a protected-paths list and aborts if any protected path is touched.
Rationale: Auto-commits give a per-experiment rollback point. The immutability guard prevents the runner from silently editing the proposal, the matrix, the spec, the decision log, the lab notebook, or path-scoped rules. These are human-curated artifacts; the runner must never touch them.
What would change our minds: If the guard fires often on legitimate runner commits, refine the protected-paths list rather than weakening the guard.

### 2026-04-14: experiments/results.tsv is ground truth, W&B is visualization
Decision: All quantitative results land in `experiments/results.tsv` as the durable append-only ground truth. W&B mirrors results for visualization and historical comparison.
Rationale: W&B is a third-party cloud service that can go down, change APIs, or retire free-tier features. The paper's defensibility depends on having the raw numbers locally. TSV is trivially parseable, diffable, and version-controllable.
What would change our minds: Nothing realistic. W&B is supplementary.

### 2026-04-14: Scope lock
Decision: The runner does not invent new experiments. `EXPERIMENTS.yaml` is the only source of truth for what runs.
Rationale: Feature creep is the single largest risk to a 4-week solo project. The proposal was scoped deliberately. Additions happen through a human edit to the YAML.
What would change our minds: After the primary matrix completes with clear findings, I may add follow-ups. Those follow-ups are explicit YAML edits, not runner decisions.

### 2026-04-14: Note on proposal text regarding Experiment 4 ratio 1/21 overlap
Observation: The proposal's Experiment 4 description says ratios {1/2, 1/4, 1/8, 1/21} are all ablated at depth 1, and that the 1/21 cell "shares data with the Experiment 3 anchor and does not require separate runs." The matrix reflects this by including only the {1/2, 1/8} ratio-ablation rows; 1/4 is covered by the primary experiments and 1/21 is covered by the Experiment 3 anchor. The proposal prompt counts this as 6 ratio-ablation rows, matching the matrix. This is consistent, not a bug. Flagging here to prevent accidental re-litigation.
What would change our minds: Nothing. This is a bookkeeping note.
