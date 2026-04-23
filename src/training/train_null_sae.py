"""Null-baseline SAE training entry point.

Dispatched from `scripts/run_loop.sh` for rows where
`level0_source == "random_gaussian"` (SPEC.md:42 and SPEC.md:9.2). Trains an
SAE on isotropic Gaussian vectors of matching dimensionality and sample
count; feeds the `pwmcc_vs_null_sigma` computation for real-data rows.

Null baselines never trigger gates (their `decision_gates` list is empty).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.metrics import (
    NULL_SEED_OFFSET,
    dead_latent_fraction,
    isotropic_gaussian,
    variance_explained,
)
from src.training.harness import experiment_context
from src.training.metrics_io import write_metrics_tsv
from src.training.sae_models import build_sae

# d_model by base model. The null SAE always sees d_model-dim vectors.
D_MODEL: dict[str, int] = {
    "google/gemma-2-2b": 2304,
    "openai-community/gpt2": 768,
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    with experiment_context(arch_hint="null") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: null SAE harness ok"
            return
        ctx.init_wandb()

        row = ctx.row
        base = row["base_model"]
        if base not in D_MODEL:
            raise ValueError(f"null SAE: unknown base_model {base!r}")
        d_model = D_MODEL[base]
        width = int(row["width"])
        sparsity = int(row["sparsity"])
        dict_ratio = float(row["dict_ratio"])
        # Sample count for a null matches the parent-depth sample count
        # (SPEC §9.2). At ratio 1/4 this is width * 4.
        n_samples = int(round(width / dict_ratio))

        # Null SAEs at meta-depth use the meta-SAE arch (BatchTopK k=4 is the
        # primary; sparsity=4 in the row, no ambiguity). The null baseline is
        # compared against meta-SAE seeds; matching the arch keeps the
        # comparison fair.
        sae = build_sae(
            arch="batchtopk",
            d_in=d_model,
            n_latents=width,
            sparsity=sparsity,
        ).to(_device())

        gen_seed = NULL_SEED_OFFSET + ctx.seed
        x_train = isotropic_gaussian(
            n_samples, d_model, seed=gen_seed, device=_device(), dtype=torch.float32
        )

        batch_size = min(4096, n_samples)
        n_epochs = 20
        n_batches = max(1, n_samples // batch_size)

        opt = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)

        curves_path: Path = ctx.artifact_dir / "curves.tsv"
        with curves_path.open("w") as cf:
            cf.write("epoch\tbatch\tloss\n")
            for epoch in range(n_epochs):
                # Shuffle indices per epoch.
                perm = torch.randperm(n_samples, device=_device())
                for b in range(n_batches):
                    idx = perm[b * batch_size : (b + 1) * batch_size]
                    batch = x_train[idx]
                    recon, _latents = sae(batch, use_training_topk=True)
                    loss = F.mse_loss(recon, batch)
                    if not torch.isfinite(loss):
                        _write_diverged(ctx, "NaN loss", curves_path)
                        raise RuntimeError(
                            f"null training diverged (NaN loss) epoch={epoch} batch={b}"
                        )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                    if grad_norm.item() > 1000:
                        _write_diverged(ctx, f"grad_norm={grad_norm.item():.3g}", curves_path)
                        raise RuntimeError(
                            f"null training diverged (grad norm) epoch={epoch} batch={b}"
                        )
                    opt.step()
                # One curve entry per epoch (final-batch loss).
                cf.write(f"{epoch}\t{n_batches - 1}\t{loss.item():.6g}\n")
                if ctx.wandb_run is not None:
                    try:
                        ctx.wandb_run.log({"train/loss": loss.item(), "epoch": epoch})
                    except Exception:
                        pass

        # Fit inference threshold so eval can run without batch-topk.
        sae.fit_inference_threshold(x_train[:batch_size])

        # Held-out evaluation on fresh Gaussians drawn from a distinct seed.
        eval_seed = NULL_SEED_OFFSET + ctx.seed + 500
        x_eval = isotropic_gaussian(
            min(16384, n_samples), d_model, seed=eval_seed, device=_device(), dtype=torch.float32
        )
        with torch.no_grad():
            recon_eval, latents_eval = sae(x_eval, use_training_topk=False)
        ve = variance_explained(recon_eval, x_eval)
        dead_frac = dead_latent_fraction(latents_eval)

        if not math.isfinite(ve):
            ctx.notes = "variance_explained non-finite on eval"

        # Save checkpoint: the decoder directions (and encoder) are what
        # downstream PW-MCC will read. Fp16 for storage per rule 9.
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
            "source": "null_isotropic_gaussian",
            "gen_seed": gen_seed,
            "row": row,
        }
        torch.save(ckpt, ctx.artifact_dir / "checkpoint.pt")

        # Metrics.tsv (training rule 5). PW-MCC is cross-seed so it cannot be
        # computed in a single-seed run; leave blank here. The post-hoc
        # analysis step fills in null_pwmcc_{mean,std} by pairing the 3 seeds.
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
        ctx.notes = f"null trained on {n_samples} Gaussian samples, d_model={d_model}"


def _write_diverged(ctx, reason: str, curves_path: Path) -> None:
    """Write a 'diverged' metrics.tsv per rule 10 before the caller raises."""
    metrics = {
        "variance_explained": float("nan"),
        "pwmcc": None,
        "dead_latent_fraction": float("nan"),
    }
    write_metrics_tsv(ctx.artifact_dir / "metrics.tsv", metrics)
    ctx.notes = f"diverged: {reason}"


if __name__ == "__main__":
    main()
