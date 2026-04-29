"""Level-0 BatchTopK SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "train_from_scratch"` and `level0_arch == "batchtopk"`
and `depth == 0` (SPEC.md:40).

Trains a width-65536 (Gemma) / 49152 (GPT-2) BatchTopK SAE on residual-
stream activations from the base model. Adam lr=3e-4 betas=(0.9, 0.999)
weight_decay=0; SAE batch 2048; k from row['sparsity'] (60 for level-0).
Token budget: 500M for Gemma, 200M for GPT-2 per the matrix descriptions.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.activations import hook_name_for, iter_residual_batches
from src.data.pile_tokens import prepare_token_shard
from src.metrics import dead_latent_fraction, variance_explained
from src.training.harness import experiment_context
from src.training.loaders import load_base_model
from src.training.metrics_io import write_metrics_tsv
from src.training.sae_models import build_sae

# Per-base-model token budgets. Driven by the matrix descriptions
# ("500M Pile tokens" for Gemma, "200M" for GPT-2 in l0_gpt2_batchtopk).
TOKEN_BUDGET: dict[str, int] = {
    "google/gemma-2-2b": 500_000_000,
    "openai-community/gpt2": 200_000_000,
}

SEQ_LEN = 1024


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fwd_batch_size(base_model: str) -> int:
    # Empirical: Gemma-2-2B at seq_len=1024 fits ~8 sequences on 24 GB.
    # GPT-2 Small fits ~32. The SAE batch is independent (2048).
    if base_model == "google/gemma-2-2b":
        return 4
    return 24


def main() -> None:
    with experiment_context(arch_hint="batchtopk") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: level0 batchtopk harness ok"
            return
        ctx.init_wandb()

        row = ctx.row
        base_model = row["base_model"]
        if base_model not in TOKEN_BUDGET:
            raise ValueError(f"level-0 BatchTopK: unknown base_model {base_model!r}")
        layer = int(row["layer"])
        site = row["site"]
        width = int(row["width"])
        sparsity = int(row["sparsity"])
        n_tokens = TOKEN_BUDGET[base_model]
        device = _device()

        # Smoke-shrink for fast end-to-end validation: env var SAE_SMOKE_TOKENS
        # caps the token budget. Useful for the professor's quick-run mode.
        import os as _os
        smoke_cap = _os.environ.get("SAE_SMOKE_TOKENS")
        if smoke_cap:
            n_tokens = min(n_tokens, int(smoke_cap))

        # Stage 1: prepare deterministic token shard (pinned dataset).
        shard_path = prepare_token_shard(
            base_model, n_tokens=n_tokens, seq_len=SEQ_LEN, seed=ctx.seed
        )

        # Stage 2: load base model (revision-pinned) on this CUDA device.
        base = load_base_model(base_model, device=device)
        d_model = int(base.cfg.d_model)
        hook_name = hook_name_for(layer, site)

        # Stage 3: build SAE.
        sae = build_sae(
            arch="batchtopk", d_in=d_model, n_latents=width, sparsity=sparsity
        ).to(device)

        # Initialize b_dec to a streaming-mean estimate from the first ~64K
        # samples. Good init for raw activations whose mean is non-zero.
        sae_batch = 2048
        fwd_batch = _fwd_batch_size(base_model)
        init_buf: list[torch.Tensor] = []
        init_target = 65536
        seen = 0
        for batch in iter_residual_batches(
            base,
            hook_name=hook_name,
            token_shard_path=shard_path,
            fwd_batch_size=fwd_batch,
            sae_batch_size=sae_batch,
            device=device,
            n_tokens_target=init_target,
        ):
            init_buf.append(batch)
            seen += batch.shape[0]
            if seen >= init_target:
                break
        init_x = torch.cat(init_buf, dim=0)[:init_target]
        with torch.no_grad():
            sae.b_dec.data = init_x.mean(dim=0).to(sae.b_dec.dtype)
        del init_buf, init_x
        torch.cuda.empty_cache()

        base_lr = float(row.get("learning_rate") or 3e-4)
        warmup_steps = int(row.get("lr_warmup_steps") or 0)
        opt = torch.optim.Adam(
            sae.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0
        )

        curves_path: Path = ctx.artifact_dir / "curves.tsv"
        log_every = max(1, n_tokens // sae_batch // 100)

        step = 0
        last_loss = float("nan")
        with curves_path.open("w", buffering=1) as cf:
            cf.write("step\tloss\n")
            for batch in iter_residual_batches(
                base,
                hook_name=hook_name,
                token_shard_path=shard_path,
                fwd_batch_size=fwd_batch,
                sae_batch_size=sae_batch,
                device=device,
                n_tokens_target=n_tokens,
            ):
                recon, _ = sae(batch, use_training_topk=True)
                loss = F.mse_loss(recon, batch)
                if not torch.isfinite(loss):
                    _diverged(ctx, "NaN loss")
                    raise RuntimeError(f"level-0 diverged (NaN) step={step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Bricken parallel-grad removal on W_dec (matches meta-SAE recipe).
                if sae.W_dec.grad is not None:
                    with torch.no_grad():
                        W_unit = F.normalize(sae.W_dec.data, dim=1)
                        radial = (sae.W_dec.grad * W_unit).sum(dim=1, keepdim=True) * W_unit
                        sae.W_dec.grad.sub_(radial)
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                if warmup_steps > 0 and step < warmup_steps:
                    warmup_lr = base_lr * (step + 1) / warmup_steps
                    for pg in opt.param_groups:
                        pg["lr"] = warmup_lr
                # Divergence guard skips the warmup window: at random init the pre-clip
                # gradient norm is naturally O(thousands), and clip_grad_norm_(max=1.0)
                # plus a warmup lr ramping from ~0 keeps the actual update microscopic.
                # Real divergence (post-warmup blowup) still trips here.
                if step >= warmup_steps and grad_norm.item() > 1000:
                    _diverged(ctx, f"grad_norm={grad_norm.item():.3g}")
                    raise RuntimeError(f"level-0 diverged (grad) step={step}")
                opt.step()
                with torch.no_grad():
                    sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)

                last_loss = loss.item()
                if step % log_every == 0:
                    cf.write(f"{step}\t{last_loss:.6g}\n")
                    if ctx.wandb_run is not None:
                        try:
                            ctx.wandb_run.log({"train/loss": last_loss, "step": step})
                        except Exception:
                            pass
                step += 1
            cf.write(f"{step}\t{last_loss:.6g}\n")

        # Calibrate inference threshold + held-out eval on the next ~32K tokens.
        eval_n = 32768
        eval_buf: list[torch.Tensor] = []
        eval_seen = 0
        # Reuse the same shard but advance past the training position by
        # using a higher seed-derived offset; simplest is a fresh shard.
        eval_shard = prepare_token_shard(
            base_model, n_tokens=eval_n * 2, seq_len=SEQ_LEN, seed=10000 + ctx.seed
        )
        for batch in iter_residual_batches(
            base,
            hook_name=hook_name,
            token_shard_path=eval_shard,
            fwd_batch_size=fwd_batch,
            sae_batch_size=sae_batch,
            device=device,
            n_tokens_target=eval_n,
        ):
            eval_buf.append(batch)
            eval_seen += batch.shape[0]
            if eval_seen >= eval_n:
                break
        x_eval = torch.cat(eval_buf, dim=0)[:eval_n]
        del eval_buf

        # Calibrate threshold on training-style topk pass.
        with torch.no_grad():
            sae.fit_inference_threshold(x_eval[: min(8192, eval_n)])
            recon_eval, latents_eval = sae(x_eval, use_training_topk=True)
        ve = variance_explained(recon_eval, x_eval)
        dead_frac = dead_latent_fraction(latents_eval)

        if not math.isfinite(ve):
            ctx.notes = "variance_explained non-finite on eval"

        ckpt = {
            "experiment_id": ctx.experiment_id,
            "arch": "batchtopk",
            "d_in": d_model,
            "n_latents": width,
            "sparsity": sparsity,
            "W_enc": sae.W_enc.detach().to(torch.float16).cpu(),
            "b_enc": sae.b_enc.detach().to(torch.float16).cpu(),
            "W_dec": sae.W_dec.detach().to(torch.float16).cpu(),
            "b_dec": sae.b_dec.detach().to(torch.float16).cpu(),
            "jump_threshold": sae.jump_threshold.detach().to(torch.float16).cpu(),
            "base_model": base_model,
            "layer": layer,
            "site": site,
            "n_tokens_trained": n_tokens,
            "row": row,
        }
        torch.save(ckpt, ctx.artifact_dir / "checkpoint.pt")

        metrics = {
            "variance_explained": ve,
            "pwmcc": None,
            "mmcs": None,
            "pwmcc_null_mean": None,
            "pwmcc_null_std": None,
            "dead_latent_fraction": dead_frac,
            "variance_explained_heldout_tokens": int(x_eval.shape[0]),
        }
        write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)
        ctx.metrics["variance_explained"] = ve
        ctx.metrics["dead_latent_fraction"] = dead_frac
        ctx.notes = (
            f"level-0 BatchTopK on {base_model} layer {layer} {site}, "
            f"width={width} k={sparsity} tokens={n_tokens}"
        )


def _diverged(ctx, reason: str) -> None:
    metrics = {
        "variance_explained": float("nan"),
        "pwmcc": None,
        "dead_latent_fraction": float("nan"),
    }
    write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)
    ctx.notes = f"diverged: {reason}"


if __name__ == "__main__":
    main()
