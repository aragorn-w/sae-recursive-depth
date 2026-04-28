---
paths: src/training/**
---

# Training code rules

Scope: any Python module under `src/training/` that trains or fine-tunes an SAE, meta-SAE, or null baseline.

## Mandatory behaviors

1. W&B is always enabled. Use `wandb.init(project="sae-recursive-depth", name=experiment_id, config=<full row dict>)`. Never skip `wandb.init` even for null-baseline runs. If W&B init fails with network error, log a warning, set `wandb = None`, continue training; the local tsv remains the ground truth.

2. Results land in `experiments/results.tsv` via `src/training/results_io.py::append_result(...)` before the process exits. This call is in a `finally:` block so a partial result is always recorded even on interrupt. No exceptions.

3. Seeds are deterministic and set before any model or data code runs. Use `src.training.seed.set_all_seeds(seed)` which calls `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`, and sets `torch.use_deterministic_algorithms(True, warn_only=True)` plus `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`.

4. Never skip seed replication. If `seed` N fails, it is retried per the backoff policy in `SPEC.md:6.1`. Do not substitute a different seed.

5. Checkpoints go to `experiments/artifacts/<experiment_id>/checkpoint.pt`. Per-step metrics go to `experiments/artifacts/<experiment_id>/metrics.tsv` with a single header row and one value row (final metrics). Per-epoch curves go to `experiments/artifacts/<experiment_id>/curves.tsv`.

6. Never load any level-0 SAE or base model directly by huggingface id in training code. Go through `src/training/loaders.py::load_level0(level0_source, base_model, layer, site)` which handles SAELens, TransformerLens, and the bartbussmann/BatchTopK HuggingFace release uniformly and pins revision shas.

7. BatchTopK training uses `k=60` for level-0 on Gemma-2-2B and `k=4` for all meta-SAEs, as specified in `EXPERIMENTS.yaml:sparsity` per row. Never hard-code these values in training modules; always read from the row config.

8. Adam defaults for meta-SAE training: `lr=3e-4`, `betas=(0.9, 0.999)`, `weight_decay=0`. Level-0 BatchTopK on Gemma-2-2B uses the same Adam settings with 500M Pile tokens and batch 2048, matching the Matryoshka paper recipe (arXiv:2503.17547).

9. Training data for recursive depths is the decoder direction matrix of the previous depth, cast to fp32 for the inner product and fp16 for storage. Each decoder direction (row of W_dec under the project's `(n_latents, d_model)` convention) is unit-normalized before being fed to the next meta-SAE; this matches standard SAE practice (Bricken et al. 2023) but is not explicitly endorsed in Leask et al. arXiv:2502.04878 or Bussmann et al. arXiv:2412.06410, despite the recursive-meta-SAE methodology being adapted from those works.

10. If training diverges (NaN loss, explosive gradient norm >1000), do not retry with a different hyperparameter. Record the divergence in metrics.tsv with `status=diverged`, exit with a non-zero code, let the run_loop.sh backoff apply. Three divergences with the same config is a `failed` experiment, not an infrastructural problem.

## Forbidden behaviors

- No interpretation of results in logs, prose, or tsv notes.
- No in-place edits to `results.tsv`. Only append.
- No modifying `EXPERIMENTS.yaml` or any other protected path.
- No experiments not in the matrix. Training code refuses to run if `--experiment-id` is not present in `EXPERIMENTS.yaml`.
