"""Flat SAE-on-activations training entry point.

Handles rows where `level0_source == "flat_sae_on_activations"`. This is
the Experiment 6 comparator: a single-level SAE trained directly on
base-model activations with the same latent count as the recursive stack's
leaf depth and k=4 to match the meta-SAE sparsity.

Crash-resilience:
- Periodic checkpoints every CKPT_EVERY steps with atomic write
  (write tmp -> fsync -> rename) and off-volume mirror to /mnt/fast.
- Boundary markers ``training_done.pt`` and ``eval_done.pt`` so a
  resumed run can skip phases already completed.
- Cached eval activations (``eval_activations.pt``) so an eval-stage
  crash does not re-harvest the host LM forward pass.
- Resume scans the artifact dir at startup and picks up at the latest
  checkpoint.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from sae_recursive_depth.data.activations import hook_name_for, iter_residual_batches
from sae_recursive_depth.data.pile_tokens import prepare_token_shard
from sae_recursive_depth.metrics import dead_latent_fraction, variance_explained
from sae_recursive_depth.training.harness import experiment_context
from sae_recursive_depth.training.loaders import load_base_model
from sae_recursive_depth.training.metrics_io import write_metrics_tsv
from sae_recursive_depth.training.sae_models import build_sae

# Token budget for flat SAE training. Smaller than level-0 (the SAE is also
# smaller in width) but enough to converge; matches Anthropic-style flat SAE
# recipes for d_model = 2304 / 768.
TOKEN_BUDGET = 100_000_000
SEQ_LEN = 1024
CKPT_EVERY = int(os.environ.get("SAE_CKPT_EVERY", "5000"))
BACKUP_MIRROR_ROOT = Path("/mnt/fast/sae_backup")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fwd_batch_size(base_model: str) -> int:
    if base_model == "google/gemma-2-2b":
        return 4
    return 24


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _mirror_to_backup(src: Path, experiment_id: str) -> None:
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
    existing = sorted(ckpt_dir.glob("step_*.pt"))
    for old in existing[:-2]:
        try:
            old.unlink()
        except Exception:
            pass


def _save_training_done(*, artifact_dir: Path, experiment_id: str, step: int, sae) -> None:
    state = {
        "phase": "training_done",
        "step": step,
        "sae_state_dict": sae.state_dict(),
    }
    path = artifact_dir / "checkpoints" / "training_done.pt"
    _atomic_save(state, path)
    _mirror_to_backup(path, experiment_id)


def _save_eval_activations(*, artifact_dir: Path, experiment_id: str, x_eval: torch.Tensor) -> None:
    path = artifact_dir / "checkpoints" / "eval_activations.pt"
    _atomic_save({"x_eval": x_eval.detach().cpu()}, path)
    _mirror_to_backup(path, experiment_id)


def _save_eval_done(
    *, artifact_dir: Path, experiment_id: str, sae, ve: float, dead_frac: float, eval_n: int
) -> None:
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
    ckpt_dir = artifact_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    eval_done = ckpt_dir / "eval_done.pt"
    if eval_done.exists():
        return {"path": eval_done, "phase": "eval_done"}
    training_done = ckpt_dir / "training_done.pt"
    if training_done.exists():
        return {"path": training_done, "phase": "training_done"}
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    if steps:
        return {"path": steps[-1], "phase": "training"}
    return None


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
        smoke_cap = os.environ.get("SAE_SMOKE_TOKENS")
        if smoke_cap:
            n_tokens = min(n_tokens, int(smoke_cap))

        shard = prepare_token_shard(
            base_model, n_tokens=n_tokens, seq_len=SEQ_LEN, seed=ctx.seed
        )
        base = load_base_model(base_model, device=device)
        d_model = int(base.cfg.d_model)
        hook_name = hook_name_for(layer, site)

        sae = build_sae(
            arch="batchtopk", d_in=d_model, n_latents=width, sparsity=sparsity
        ).to(device)

        sae_batch = 2048
        fwd_batch = _fwd_batch_size(base_model)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)

        resume = _scan_resume(ctx.artifact_dir)
        resume_step = 0
        resume_seq_cursor = 0
        b_dec_init_done = False
        skip_training = False
        skip_eval = False
        resumed_ve = float("nan")
        resumed_dead = float("nan")
        resumed_eval_n = 0
        if resume is not None:
            print(f"[resume] found {resume['phase']} at {resume['path']}", flush=True)
            payload = torch.load(resume["path"], map_location=device, weights_only=False)
            if resume["phase"] == "eval_done":
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
            else:
                sae.load_state_dict(payload["sae_state_dict"])
                opt.load_state_dict(payload["opt_state_dict"])
                resume_step = int(payload["step"])
                resume_seq_cursor = int(payload["seq_cursor"])
                b_dec_init_done = bool(payload.get("b_dec_init_done", True))

        if not skip_training and not b_dec_init_done:
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
            b_dec_init_done = True

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
                        token_shard_path=shard,
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

            _save_training_done(
                artifact_dir=ctx.artifact_dir,
                experiment_id=ctx.experiment_id,
                step=step,
                sae=sae,
            )
            del opt
            torch.cuda.empty_cache()

        eval_n = 16384
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
                try:
                    del base
                except NameError:
                    pass
                torch.cuda.empty_cache()
            else:
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
                del base
                torch.cuda.empty_cache()
                _save_eval_activations(
                    artifact_dir=ctx.artifact_dir,
                    experiment_id=ctx.experiment_id,
                    x_eval=x_eval,
                )

            with torch.no_grad():
                sae.fit_inference_threshold(x_eval[:8192])
                recon_eval, latents_eval = sae(x_eval, use_training_topk=True)
            ve = variance_explained(recon_eval, x_eval)
            dead_frac = dead_latent_fraction(latents_eval)
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
