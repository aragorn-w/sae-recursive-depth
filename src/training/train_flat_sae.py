"""Flat SAE-on-activations training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "flat_sae_on_activations"` (SPEC.md:42). This is the
Experiment 6 comparator: a single-level SAE trained directly on base-model
activations with the same latent count as the recursive stack's leaf depth
and k=4 to match the meta-SAE sparsity.
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

# Token budget for flat SAE training. Smaller than level-0 (the SAE is also
# smaller in width) but enough to converge; matches Anthropic-style flat SAE
# recipes for d_model = 2304 / 768.
TOKEN_BUDGET = 100_000_000
SEQ_LEN = 1024


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fwd_batch_size(base_model: str) -> int:
    if base_model == "google/gemma-2-2b":
        return 4
    return 24


def main() -> None:
    with experiment_context(arch_hint="flat") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: flat SAE harness ok"
            return
        ctx.init_wandb()

        row = ctx.row
        base_model = row["base_model"]
        layer = int(row["layer"])
        site = row["site"]
        width = int(row["width"])
        sparsity = int(row["sparsity"])
        device = _device()

        n_tokens = TOKEN_BUDGET
        import os as _os
        smoke_cap = _os.environ.get("SAE_SMOKE_TOKENS")
        if smoke_cap:
            n_tokens = min(n_tokens, int(smoke_cap))

        shard = prepare_token_shard(
            base_model, n_tokens=n_tokens, seq_len=SEQ_LEN, seed=ctx.seed
        )
        base = load_base_model(base_model, device=device)
        d_model = int(base.cfg.d_model)
        hook_name = hook_name_for(layer, site)

        # Flat SAE arch: BatchTopK with row's width and sparsity. The matrix
        # row's level0_arch is used only to identify the comparator family;
        # the underlying training is BatchTopK (parity with meta-SAE).
        sae = build_sae(
            arch="batchtopk", d_in=d_model, n_latents=width, sparsity=sparsity
        ).to(device)

        # Init b_dec on a streaming-mean estimate.
        sae_batch = 2048
        fwd_batch = _fwd_batch_size(base_model)
        init_buf: list[torch.Tensor] = []
        init_target = 32768
        seen = 0
        for batch in iter_residual_batches(
            base,
            hook_name=hook_name,
            token_shard_path=shard,
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

        opt = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)
        curves_path: Path = ctx.artifact_dir / "curves.tsv"
        log_every = max(1, n_tokens // sae_batch // 100)
        step = 0
        last_loss = float("nan")
        with curves_path.open("w", buffering=1) as cf:
            cf.write("step\tloss\n")
            for batch in iter_residual_batches(
                base,
                hook_name=hook_name,
                token_shard_path=shard,
                fwd_batch_size=fwd_batch,
                sae_batch_size=sae_batch,
                device=device,
                n_tokens_target=n_tokens,
            ):
                recon, _ = sae(batch, use_training_topk=True)
                loss = F.mse_loss(recon, batch)
                if not torch.isfinite(loss):
                    _diverged(ctx, "NaN loss")
                    raise RuntimeError(f"flat SAE diverged (NaN) step={step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if sae.W_dec.grad is not None:
                    with torch.no_grad():
                        W_unit = F.normalize(sae.W_dec.data, dim=1)
                        radial = (sae.W_dec.grad * W_unit).sum(dim=1, keepdim=True) * W_unit
                        sae.W_dec.grad.sub_(radial)
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                if grad_norm.item() > 1000:
                    _diverged(ctx, f"grad_norm={grad_norm.item():.3g}")
                    raise RuntimeError(f"flat SAE diverged (grad) step={step}")
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

        eval_n = 16384
        eval_shard = prepare_token_shard(
            base_model, n_tokens=eval_n * 2, seq_len=SEQ_LEN, seed=20000 + ctx.seed
        )
        eval_buf: list[torch.Tensor] = []
        eval_seen = 0
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
        with torch.no_grad():
            sae.fit_inference_threshold(x_eval[:8192])
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
            "kind": "flat_sae",
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
            f"flat SAE on {base_model} layer {layer} {site}, "
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
