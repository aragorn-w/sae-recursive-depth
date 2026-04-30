"""Level-0 BatchTopK SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "train_from_scratch"` and `level0_arch == "batchtopk"`
and `depth == 0` (SPEC.md:40).

Trains a width-65536 (Gemma) / 49152 (GPT-2) BatchTopK SAE on residual-
stream activations from the base model. Adam lr=3e-4 betas=(0.9, 0.999)
weight_decay=0; SAE batch 2048; k from row['sparsity'] (60 for level-0).
Token budget: 500M for Gemma, 200M for GPT-2 per the matrix descriptions.

Crash-resilience:
- Periodic checkpoints every CKPT_EVERY steps with atomic write
  (write tmp → fsync → rename) and off-volume mirror to /mnt/fast.
- Boundary markers ``training_done.pt`` and ``eval_done.pt`` so a
  resumed run can skip phases already completed.
- Cached eval activations (``eval_activations.pt``) so an eval-stage
  crash does not re-harvest the host LM forward pass.
- Resume scans the artifact dir at startup and picks up at the latest
  checkpoint.

Sharding:
- Gemma-2-2B uses transformer_lens ``n_devices=2`` to layer-split across
  GPUs visible at CUDA:0 and CUDA:1 (operationally GPU 1 + GPU 3 on
  Hades, exposed via ``CUDA_VISIBLE_DEVICES=1,3``). fp32 preserved.
- GPT-2 Small fits comfortably on one card; n_devices=1.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.activations import hook_name_for, iter_residual_batches
from src.data.pile_tokens import prepare_token_shard
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

# n_devices for transformer_lens layer-split sharding. Gemma-2-2B fp32
# (~10 GB) splits across two 3090s leaving ~14 GB headroom each for SAE
# weights, Adam state, and activations. GPT-2 Small is small enough that
# single-card is fine and avoids inter-GPU comms overhead.
N_DEVICES: dict[str, int] = {
    "google/gemma-2-2b": 2,
    "openai-community/gpt2": 1,
}

SEQ_LEN = 1024
CKPT_EVERY = int(os.environ.get("SAE_CKPT_EVERY", "10000"))
BACKUP_MIRROR_ROOT = Path("/mnt/fast/sae_backup")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fwd_batch_size(base_model: str) -> int:
    # Empirical: Gemma-2-2B at seq_len=1024 fits ~8 sequences on 24 GB.
    # GPT-2 Small fits ~32. The SAE batch is independent (2048).
    if base_model == "google/gemma-2-2b":
        return 4
    return 24


def _atomic_save(obj, path: Path) -> None:
    """Write ``obj`` to ``path`` atomically: write tmp, fsync, rename.

    Crash mid-write cannot corrupt the canonical file; a partially-written
    .tmp may be left behind and is overwritten on next save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _mirror_to_backup(src: Path, experiment_id: str) -> None:
    """Best-effort mirror of ``src`` to ``/mnt/fast/sae_backup/<exp>/...``.

    Same relative path under the mirror root. Failures are logged but never
    raise — primary save already succeeded.
    """
    try:
        rel = Path("checkpoints") / src.name if src.parent.name == "checkpoints" else Path(src.name)
        dst = BACKUP_MIRROR_ROOT / experiment_id / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:  # noqa: BLE001
        print(f"[ckpt] mirror failed src={src} err={e!r}", flush=True)


def _save_periodic(
    *,
    artifact_dir: Path,
    experiment_id: str,
    step: int,
    sae,
    opt,
    seq_cursor: int,
    b_dec_init_done: bool,
) -> None:
    """Save a periodic training checkpoint (atomic + mirrored)."""
    state = {
        "phase": "training",
        "step": step,
        "seq_cursor": seq_cursor,
        "b_dec_init_done": b_dec_init_done,
        "sae_state_dict": sae.state_dict(),
        "opt_state_dict": opt.state_dict(),
    }
    ckpt_dir = artifact_dir / "checkpoints"
    path = ckpt_dir / f"step_{step:08d}.pt"
    _atomic_save(state, path)
    _mirror_to_backup(path, experiment_id)
    # Trim old periodic checkpoints — keep only the latest 2 to bound disk.
    existing = sorted(ckpt_dir.glob("step_*.pt"))
    for old in existing[:-2]:
        try:
            old.unlink()
        except Exception:
            pass


def _save_training_done(
    *, artifact_dir: Path, experiment_id: str, step: int, sae
) -> None:
    """Write the post-training boundary marker (weights only, no opt state)."""
    state = {
        "phase": "training_done",
        "step": step,
        "sae_state_dict": sae.state_dict(),
    }
    path = artifact_dir / "checkpoints" / "training_done.pt"
    _atomic_save(state, path)
    _mirror_to_backup(path, experiment_id)


def _save_eval_activations(
    *, artifact_dir: Path, experiment_id: str, x_eval: torch.Tensor
) -> None:
    """Persist the harvested eval activations so an eval-stage crash skips re-harvest."""
    path = artifact_dir / "checkpoints" / "eval_activations.pt"
    _atomic_save({"x_eval": x_eval.detach().cpu()}, path)
    _mirror_to_backup(path, experiment_id)


def _save_eval_done(
    *,
    artifact_dir: Path,
    experiment_id: str,
    sae,
    ve: float,
    dead_frac: float,
    eval_n: int,
) -> None:
    """Boundary marker after eval — captures the post-threshold-fit state."""
    state = {
        "phase": "eval_done",
        "sae_state_dict": sae.state_dict(),
        "ve": ve,
        "dead_frac": dead_frac,
        "eval_n": eval_n,
    }
    path = artifact_dir / "checkpoints" / "eval_done.pt"
    _atomic_save(state, path)
    _mirror_to_backup(path, experiment_id)


def _scan_resume(artifact_dir: Path) -> dict | None:
    """Return the highest-priority resume payload (or None for fresh start).

    Priority: eval_done > training_done > latest step_*.pt.
    """
    ckpt_dir = artifact_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    eval_done = ckpt_dir / "eval_done.pt"
    if eval_done.exists():
        return {"path": eval_done, "phase": "eval_done"}
    training_done = ckpt_dir / "training_done.pt"
    if training_done.exists():
        # Also peek at any cached eval activations.
        return {"path": training_done, "phase": "training_done"}
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    if steps:
        return {"path": steps[-1], "phase": "training"}
    return None


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
        n_devices = N_DEVICES.get(base_model, 1)
        device = _device()

        # Smoke-shrink for fast end-to-end validation: env var SAE_SMOKE_TOKENS
        # caps the token budget. Useful for the professor's quick-run mode.
        smoke_cap = os.environ.get("SAE_SMOKE_TOKENS")
        if smoke_cap:
            n_tokens = min(n_tokens, int(smoke_cap))

        # Stage 1: prepare deterministic token shard (pinned dataset).
        shard_path = prepare_token_shard(
            base_model, n_tokens=n_tokens, seq_len=SEQ_LEN, seed=ctx.seed
        )

        # Stage 2: load base model with layer-split sharding when requested.
        # transformer_lens uses contiguous CUDA devices starting at ``device``
        # for layer-split. Caller controls the visible set via
        # CUDA_VISIBLE_DEVICES.
        base = load_base_model(base_model, device=device, n_devices=n_devices)
        d_model = int(base.cfg.d_model)
        hook_name = hook_name_for(layer, site)

        # Stage 3: build SAE on the residual-hook card (CUDA:0 of the visible
        # set; the residual hook fires on whichever shard hosts the target
        # block, but transformer_lens centralizes intermediates on the
        # primary device).
        sae = build_sae(
            arch="batchtopk", d_in=d_model, n_latents=width, sparsity=sparsity
        ).to(device)

        sae_batch = 2048
        fwd_batch = _fwd_batch_size(base_model)
        base_lr = float(row.get("learning_rate") or 3e-4)
        warmup_steps = int(row.get("lr_warmup_steps") or 0)
        opt = torch.optim.Adam(
            sae.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0
        )

        # Stage 4: scan for a resume point.
        resume = _scan_resume(ctx.artifact_dir)
        resume_step = 0
        resume_seq_cursor = 0
        b_dec_init_done = False
        skip_training = False
        skip_eval = False
        if resume is not None:
            print(f"[resume] found {resume['phase']} at {resume['path']}", flush=True)
            payload = torch.load(resume["path"], map_location=device, weights_only=False)
            if resume["phase"] == "eval_done":
                # Already-completed run: load SAE state and skip directly to
                # the metrics/checkpoint write block. Eval ve/dead_frac come
                # from the boundary marker.
                sae.load_state_dict(payload["sae_state_dict"])
                skip_training = True
                skip_eval = True
                resumed_ve = float(payload["ve"])
                resumed_dead = float(payload["dead_frac"])
                resumed_eval_n = int(payload["eval_n"])
            elif resume["phase"] == "training_done":
                sae.load_state_dict(payload["sae_state_dict"])
                skip_training = True
                b_dec_init_done = True
            else:  # mid-training
                sae.load_state_dict(payload["sae_state_dict"])
                opt.load_state_dict(payload["opt_state_dict"])
                resume_step = int(payload["step"])
                resume_seq_cursor = int(payload["seq_cursor"])
                b_dec_init_done = bool(payload.get("b_dec_init_done", True))

        # Stage 5: b_dec init from streaming-mean estimate (skipped on resume).
        if not skip_training and not b_dec_init_done:
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
            b_dec_init_done = True

        # Stage 6: training loop (skipped if training_done.pt or eval_done.pt
        # already exists).
        curves_path: Path = ctx.artifact_dir / "curves.tsv"
        log_every = max(1, n_tokens // sae_batch // 1000)
        step = resume_step
        seq_cursor = resume_seq_cursor
        last_loss = float("nan")

        if not skip_training:
            curves_mode = "a" if resume_step > 0 else "w"
            with curves_path.open(curves_mode, buffering=1) as cf:
                if curves_mode == "w":
                    cf.write("step\tloss\n")
                tokens_remaining = max(0, n_tokens - resume_step * sae_batch)
                if tokens_remaining > 0:
                    for batch in iter_residual_batches(
                        base,
                        hook_name=hook_name,
                        token_shard_path=shard_path,
                        fwd_batch_size=fwd_batch,
                        sae_batch_size=sae_batch,
                        device=device,
                        n_tokens_target=tokens_remaining,
                        seq_cursor_start=seq_cursor,
                    ):
                        recon, _ = sae(batch, use_training_topk=True)
                        loss = F.mse_loss(recon, batch)
                        if not torch.isfinite(loss):
                            _diverged(ctx, "NaN loss")
                            raise RuntimeError(f"level-0 diverged (NaN) step={step}")
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        # Bricken parallel-grad removal on W_dec.
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
                        # Divergence guard skips warmup (clip_grad_norm + tiny lr makes
                        # large pre-clip norms benign during ramp-in).
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
                        # Each SAE batch consumes sae_batch tokens = sae_batch/SEQ_LEN seqs.
                        seq_cursor += sae_batch // SEQ_LEN
                        step += 1
                        if step % CKPT_EVERY == 0:
                            _save_periodic(
                                artifact_dir=ctx.artifact_dir,
                                experiment_id=ctx.experiment_id,
                                step=step,
                                sae=sae,
                                opt=opt,
                                seq_cursor=seq_cursor,
                                b_dec_init_done=b_dec_init_done,
                            )
                cf.write(f"{step}\t{last_loss:.6g}\n")

            # Boundary marker: training complete.
            _save_training_done(
                artifact_dir=ctx.artifact_dir,
                experiment_id=ctx.experiment_id,
                step=step,
                sae=sae,
            )
            # Free Adam moments (~2.4 GiB at width=65536) before eval.
            del opt
            torch.cuda.empty_cache()

        # Stage 7: eval activations harvest, with cache reuse on retry.
        eval_n = 32768
        eval_cache_path = ctx.artifact_dir / "checkpoints" / "eval_activations.pt"
        if skip_eval:
            ve = resumed_ve
            dead_frac = resumed_dead
            x_eval_n = resumed_eval_n
        else:
            if eval_cache_path.exists():
                print(f"[resume] using cached eval_activations from {eval_cache_path}", flush=True)
                cached = torch.load(eval_cache_path, map_location=device, weights_only=False)
                x_eval = cached["x_eval"].to(device)
                # base may still be loaded — free it now if so.
                try:
                    del base
                except NameError:
                    pass
                torch.cuda.empty_cache()
            else:
                eval_shard = prepare_token_shard(
                    base_model, n_tokens=eval_n * 2, seq_len=SEQ_LEN, seed=10000 + ctx.seed
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
                # Free Gemma — eval-stage operations only need the SAE.
                del base
                torch.cuda.empty_cache()
                _save_eval_activations(
                    artifact_dir=ctx.artifact_dir,
                    experiment_id=ctx.experiment_id,
                    x_eval=x_eval,
                )

            # Stage 8: chunked eval. fit_inference_threshold + VE accumulation.
            with torch.no_grad():
                sae.fit_inference_threshold(x_eval[: min(8192, eval_n)], chunk_size=1024)
                target_mean = x_eval.mean(dim=0, keepdim=True)
                ss_resid = torch.zeros((), device=device, dtype=torch.float32)
                ss_target = torch.zeros((), device=device, dtype=torch.float32)
                latent_active_any = torch.zeros(width, dtype=torch.bool, device=device)
                for s in range(0, eval_n, sae_batch):
                    xb = x_eval[s : s + sae_batch]
                    recon_b, latents_b = sae(xb, use_training_topk=True)
                    ss_resid += (xb - recon_b).pow(2).sum()
                    ss_target += (xb - target_mean).pow(2).sum()
                    latent_active_any |= (latents_b != 0).any(dim=0)
                    del recon_b, latents_b, xb
                torch.cuda.empty_cache()

            ve = (
                float("nan")
                if ss_target.item() == 0
                else float(1.0 - (ss_resid / ss_target).item())
            )
            dead_frac = float(1.0 - latent_active_any.float().mean().item())
            x_eval_n = int(x_eval.shape[0])

            _save_eval_done(
                artifact_dir=ctx.artifact_dir,
                experiment_id=ctx.experiment_id,
                sae=sae,
                ve=ve,
                dead_frac=dead_frac,
                eval_n=x_eval_n,
            )

        if not math.isfinite(ve):
            ctx.notes = "variance_explained non-finite on eval"

        # Final canonical checkpoint (fp16-on-disk per SAE-paper convention,
        # consumed downstream by meta-SAE training and analysis pipelines).
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
        _atomic_save(ckpt, ctx.artifact_dir / "checkpoint.pt")

        metrics = {
            "variance_explained": ve,
            "pwmcc": None,
            "mmcs": None,
            "pwmcc_null_mean": None,
            "pwmcc_null_std": None,
            "dead_latent_fraction": dead_frac,
            "variance_explained_heldout_tokens": int(x_eval_n),
        }
        write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)
        ctx.metrics["variance_explained"] = ve
        ctx.metrics["dead_latent_fraction"] = dead_frac
        ctx.notes = (
            f"level-0 BatchTopK on {base_model} layer {layer} {site}, "
            f"width={width} k={sparsity} tokens={n_tokens} n_devices={n_devices}"
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
