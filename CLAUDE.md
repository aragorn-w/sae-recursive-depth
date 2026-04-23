# CLAUDE.md

## Project summary
This project trains recursive meta-sparse-autoencoders at depths 1 through 3 on Gemma-2-2B and GPT-2 Small to test whether SAE features are atomic or recursively decomposable objects. Primary metrics are variance explained, cross-seed PW-MCC, and MMCS against isotropic Gaussian nulls. The full scientific scope is in `proposal_v3.docx` and the locked execution matrix is in `EXPERIMENTS.yaml`.

## Who is running this
Aragorn Wang, senior undergraduate, Colorado School of Mines. Single operator. Personal workstation "Hades": AMD Ryzen Threadripper 7970X, 128 GB DDR5, 2x RTX 4080, 2x RTX 3090, 1x RTX 4060 Ti, 96 GB VRAM total. Ubuntu 24.04, CUDA 12.x, Python 3.11+. Repo root `/home/wanga/school/math498c/sae-recursive-depth`.

## Repo structure
```
sae-recursive-depth/
  CLAUDE.md              this file
  EXPERIMENTS.yaml       locked experimental matrix (source of truth for what runs)
  DECISIONS.md           architectural decision log
  SPEC.md                runner specification
  README.md              public-facing overview
  proposal_v3.docx       immutable scientific proposal
  lab_notebook.md        chronological human notes
  scripts/               bash + python orchestration
  experiments/
    results.tsv          append-only durable results log (ground truth)
    state.json           runner position, PID, last heartbeat
    logs/                per-experiment stdout+stderr
    artifacts/           model checkpoints, metric tsvs, figures
  src/
    training/            SAE training code
    metrics/             PW-MCC, MMCS, Hungarian matching, nulls
    analysis/            plots, tables, reports
    data/                activation and decoder-direction pipelines
  .claude/
    rules/               path-scoped rules (training/metrics/analysis)
    commands/            custom slash commands
  handoffs/              per-session handoff markdowns
  vendored/              pinned third-party code (do not modify)
```

## Critical workflow rules

1. **Never invent experiments.** The runner executes rows from `EXPERIMENTS.yaml` in dependency order. If an experiment is not in the matrix, it does not run. New experiments are added by the human editing the YAML and committing it.

2. **Never interpret results.** Computing a metric is fine. Writing "this supports H2" is not. Any claim about what results mean is flagged with `HUMAN_JUDGMENT_NEEDED:` and ntfy'd. Autonomous runs produce numbers, not interpretations.

3. **Never modify protected paths.** The immutability guard (`scripts/pre_commit_immutability_guard.sh`) blocks commits that touch: `CLAUDE.md`, `EXPERIMENTS.yaml`, `DECISIONS.md`, `SPEC.md`, `proposal_v3.docx`, `lab_notebook.md`, `.claude/rules/**`, `.claude/agents/**`, `vendored/**`. If a change to any of these is needed, stop and ntfy.

4. **Update results before ending.** Before any session or experiment terminates, `experiments/results.tsv` must contain the row and `lab_notebook.md` must have a dated entry. TSV is immutable append-only; the notebook is chronological.

5. **Decision gates are final.** When a gate fires, take the configured action (`continue_depth`, `skip_depth`, `skip_experiment`, `halt_and_notify`), log it to `state.json`, ntfy, and proceed. The human reviews asynchronously.

6. **Infrastructural failures retry with backoff.** OOM, CUDA errors, W&B outages: retry 3 times at 30s / 5min / 30min. After three failures, ntfy urgent and mark `failed` in the matrix.

7. **Commit under `[SAE-LOOP]`.** All runner commits use the `[SAE-LOOP]` prefix. Human commits use any other prefix.

8. **Cite in prose.** When drafting Methods/Results prose, every quantitative claim must reference a specific row in `results.tsv`. No round numbers, no hand-waving.

## GPU assignment

| GPU | Device | Role | Rationale |
|-----|--------|------|-----------|
| 0   | RTX 4080 16GB | Meta-SAE training (depths 1-3) | Fast tensor cores, meta-SAE data fits in VRAM |
| 1   | RTX 4080 16GB | Meta-SAE training (parallel seed) | Parallelize seeds across cards |
| 2   | RTX 3090 24GB | Level-0 BatchTopK on Gemma-2-2B | Needs 24 GB headroom for activation caching |
| 3   | RTX 3090 24GB | Autointerp (Llama-3.1-8B-Instruct) | Llama-3.1-8B fp16 fits in 24 GB with room for KV cache |
| 4   | RTX 4060 Ti 16GB | Evaluation, metrics, plotting | Low contention for analysis pipeline |

Set via `CUDA_VISIBLE_DEVICES` on a per-experiment basis (see `EXPERIMENTS.yaml:gpu_preference`).

## Daily commands

```bash
tmux attach -t sae-loop                            # attach to runner session
bash scripts/run_loop.sh                            # start the runner (inside tmux)
python scripts/heartbeat.py                         # force a heartbeat now
bash scripts/ntfy_send.sh default "title" "msg"     # ad hoc ntfy
tail -f experiments/logs/$(ls -t experiments/logs/ | head -1)  # tail latest log
cat experiments/state.json                          # runner state
column -t -s $'\t' experiments/results.tsv | less -S  # results
```

## Imports

@import ./SPEC.md
@import ./DECISIONS.md
@import ./EXPERIMENTS.yaml
@import ./.claude/rules/training.md
@import ./.claude/rules/metrics.md
@import ./.claude/rules/analysis.md

## Scope discipline (final note)

The attached proposal `proposal_v3.docx` is the scope. `EXPERIMENTS.yaml` is the locked execution matrix derived from it. The runner does not invent, expand, or reprioritize experiments. If an experiment fails for an infrastructural reason, retry with backoff then ntfy. If a decision gate fires, take the configured action and ntfy. Never interpret. Always cite. Never modify protected paths.
