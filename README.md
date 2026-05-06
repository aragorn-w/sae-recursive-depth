# sae-recursive-depth

**How Deep Does the Rabbit Hole Go? Testing the Limits of Recursive Feature Decomposition via Multi-Depth Meta-Sparse Autoencoders on Gemma-2-2B and GPT-2 Small**

Aragorn Wang, Colorado School of Mines. MATH 498C final project. Execution window 2026-04-14 through 2026-05-05.

Repo: https://github.com/aragorn-w/sae-recursive-depth

---

## Where to find things

| What | Where |
|---|---|
| Current writeup | [`WRITEUP.md`](./WRITEUP.md) |
| Original project proposal | [`proposal_v3.docx`](./proposal_v3.docx) |
| Full experimental matrix (75 runs, locked) | [`EXPERIMENTS.yaml`](./EXPERIMENTS.yaml) |
| Runner specification | [`SPEC.md`](./SPEC.md) |
| Architectural decisions with rationale | [`DECISIONS.md`](./DECISIONS.md) |
| Agent guidance for Claude Code runs | [`CLAUDE.md`](./CLAUDE.md) |
| Day-by-day lab notebook | [`lab_notebook.md`](./lab_notebook.md) |
| Bibliography (BibTeX) | [`references.bib`](./references.bib) |
| Session handoffs between runner and human | [`handoffs/`](./handoffs/) |
| Raw training + metric results, append-only | `experiments/results.tsv` |
| Runner state, current matrix position | `experiments/state.json` (root summary) |
| Per-lane runner state | `experiments/state.lane.lane-{0,1,2a,2b,4}.json` |
| Per-experiment artifacts, checkpoints, figures | `experiments/artifacts/<experiment_id>/` |
| Figures for the writeup | `experiments/artifacts/_summary/figures/` |
| Tests | `tests/` (unit tests for metrics, loaders, gate evaluator) |
| W&B runs | project `sae-recursive-depth` under my account |

Nothing under `experiments/` is checkpointed into git yet. Those directories fill in as the runner executes.

## Codebase map

```
sae-recursive-depth/
|
|-- README.md                       (this file)
|-- WRITEUP.md                      (living writeup, IEEE conference template)
|-- references.bib                  (BibTeX bibliography for WRITEUP.md)
|-- proposal_v3.docx                (original proposal, locked)
|-- CLAUDE.md                       (agent instructions, loaded by Claude Code)
|-- SPEC.md                         (runner contract: gates, recovery, state schema)
|-- EXPERIMENTS.yaml                (75-row locked matrix, parsed by the runner)
|-- DECISIONS.md                    (dated architectural decisions with rationale)
|-- lab_notebook.md                 (human-facing daily log, no interpretation outside here)
|
|-- .claude/
|   |-- rules/                      (path-scoped rules auto-loaded by Claude Code)
|   |   |-- training.md             (applies to src/training/**)
|   |   |-- metrics.md              (applies to src/metrics/**)
|   |   |-- analysis.md             (applies to src/analysis/**)
|   |-- commands/                   (custom slash commands)
|       |-- run-experiment.md       (/run-experiment <id>)
|       |-- handoff.md              (/handoff generates handoffs/<ts>.md)
|
|-- scripts/
|   |-- bootstrap.sh                (one-shot setup: deps, git hooks, cron, ntfy)
|   |-- run_loop.sh                 (autonomous runner entry point, tmux-safe)
|   |-- evaluate_gates.py           (decision-gate evaluator, called per experiment)
|   |-- heartbeat.py                (twice-daily status ntfy, cron-scheduled)
|   |-- ntfy_send.sh                (thin wrapper over curl for ntfy.sh)
|   |-- pre_commit_immutability_guard.sh   (blocks edits to locked files)
|
|-- src/
|   |-- training/                   (level-0 SAEs, meta-SAEs, null baselines)
|   |   |-- train_level0_batchtopk.py
|   |   |-- train_meta_sae.py
|   |   |-- train_null_baseline.py
|   |   |-- seed.py                 (set_all_seeds, deterministic RNG)
|   |   |-- loaders.py              (Pile streaming, activation caching)
|   |-- metrics/                    (PW-MCC, MMCS, VE, dead latents, null baselines)
|   |-- analysis/                   (figure generation, autointerp, summary tables)
|   |   |-- run_autointerp.py       (Llama-3.1-8B-Instruct scoring on GPUs 2,3)
|   |-- data/                       (Pile shard configuration, held-out splits)
|
|-- tests/                          (pytest suite)
|   |-- test_metrics.py             (PW-MCC, MMCS, Hungarian matching correctness)
|   |-- test_loaders.py             (deterministic token streams under fixed seed)
|   |-- test_evaluate_gates.py      (gate action dispatch on synthetic results.tsv)
|
|-- experiments/                    (runtime output, .gitignored except results.tsv)
|   |-- state.json                  (runner state, current experiment, skipped list)
|   |-- results.tsv                 (append-only ground truth of every completed run)
|   |-- logs/                       (stdout/stderr per run, heartbeat log)
|   |-- artifacts/
|       |-- <experiment_id>/
|       |   |-- checkpoints/
|       |   |-- figures/            (figures + matching .tsv/.csv of source numbers)
|       |   |-- autointerp/scores.tsv
|       |-- _summary/figures/       (cross-experiment summary plots)
|
|-- handoffs/                       (session handoff markdown, one per session)
|-- vendored/                       (local copies of SAELens, TransformerLens if pinned)
```

### Where to look depending on what you want

- **"Does the recursion bottom out at depth 2 or 3?"** goes to `WRITEUP.md` sections IV and V once populated, then `experiments/artifacts/_summary/figures/pwmcc_vs_depth.pdf` and its companion tsv.
- **"What were the exact runs?"** goes to `EXPERIMENTS.yaml`.
- **"Did the pipeline reproduce Leask et al.'s 55.47%?"** goes to the anchor rows in `experiments/results.tsv` where `experiment_id` starts with `gpt2_batchtopk_anchor_` or `gemma_batchtopk_anchor_`, column `variance_explained`.
- **"Why was this decision made?"** goes to `DECISIONS.md`.
- **"What does the runner do when X happens?"** goes to `SPEC.md`.
- **"What happened in session Y?"** goes to `handoffs/YYYY-MM-DD-HHMM.md`.
- **"Was anything interesting observed that day?"** goes to `lab_notebook.md`.

## Status

Day 22 of 26 (2026-05-06; submission 2026-05-09 23:59 MDT, extended from 2026-05-05). 54 of the 75 matrix rows are `ok` in `experiments/results.tsv`. The remaining 21 rows plus `autointerp_all` are gated on `l0_gemma_batchtopk`, currently at SAE training step 120k of ~244k (~50% of its token budget). The runner has been refactored from the original single-loop design into a 5-lane parallel autopilot (`scripts/run_autopilot.sh`), with per-lane state files at `experiments/state.lane.lane-*.json` and a root `state.json` carrying lane summaries. Numeric progress is in `WRITEUP.md` Section 3; daily detail is in `lab_notebook.md`.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `bootstrap.sh` installs uv if it is missing, then resolves `pyproject.toml` into a project-local `.venv/`.

```bash
bash scripts/bootstrap.sh
```

This resolves Python dependencies via `uv sync`, registers the immutability-guard git hook, schedules the twice-daily cron heartbeat, verifies CUDA, and prints the remaining manual steps (W&B login, HuggingFace login for the gated Gemma-2-2B repo, ntfy subscription to topic `sae-wanga-research`).

Ad-hoc scripts run inside the managed environment with `uv run`, e.g. `uv run python scripts/heartbeat.py`. The full dependency list lives in `pyproject.toml`; `scripts/bootstrap.sh` has the non-Python setup details.

## Running the autonomous loop

```bash
tmux new-session -d -s sae-runner 'bash scripts/run_loop.sh'
tmux attach -t sae-runner
```

The runner iterates `EXPERIMENTS.yaml` one row at a time, respecting dependencies and decision gates. See `SPEC.md` for the full runner contract, gate actions, and recovery behavior.

To run a single experiment by hand inside an interactive Claude Code session:

```
/run-experiment gemma_batchtopk_d1_s0
```

To generate a session handoff:

```
/handoff
```

## Tests

```bash
pytest tests/ -v
```

Test coverage is limited to pieces with deterministic expected outputs: metric correctness against hand-computed cases, loader determinism under fixed seed, and gate-action dispatch on synthetic `results.tsv` rows. Training runs are not unit-tested; they are validated end-to-end by the anchor experiment reproducing Leask et al.

## Reproducibility

Every figure under `experiments/artifacts/` has a matching `.tsv` or `.csv` of source data next to it, carrying the generation timestamp, git commit SHA, analysis script path, and the set of `experiment_id` values consumed. The append-only `results.tsv` is the single source of truth; if a number in the writeup does not trace to a row and column there, it should not be in the writeup.

## Compiling the writeup to LaTeX

`WRITEUP.md` is authored to round-trip cleanly through Pandoc into the IEEE conference LaTeX template. A typical compile looks like:

```bash
pandoc WRITEUP.md \
    --from markdown \
    --to latex \
    --bibliography=references.bib \
    --biblatex \
    --template=ieee-conference.tex \
    --output=writeup.tex
pdflatex writeup.tex && biber writeup && pdflatex writeup.tex && pdflatex writeup.tex
```

The `--bibliography` flag points pandoc at `references.bib`. The `--biblatex` flag emits `\cite{key}` rather than resolving citations inline, so biber handles the bibliography at LaTeX compile time. The `ieee-conference.tex` template is a pandoc-compatible wrapper around the official IEEEtran document class.

## License

MIT (placeholder). A `LICENSE` file will be added before any public release.

## Contact

Aragorn Wang, Colorado School of Mines.
