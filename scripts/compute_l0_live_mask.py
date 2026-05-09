"""One-off compute: per-latent firing rate of l0_gemma_batchtopk over 10M tokens.

Drives FIX 3 of the 2026-05-09 loop: write a live_mask for the dead-latent
live-only depth-1 meta-SAE. Reuses sae_recursive_depth/data/activations.py and sae_recursive_depth/training/
loaders.py per the loop prompt's "no new infra" rule.

Outputs:
  experiments/artifacts/l0_gemma_batchtopk/live_mask.pt
    {
        "firing_count": int64 tensor (n_latents,),
        "n_tokens": int,
        "firing_rate": fp32 tensor (n_latents,),
        "live_mask_strict": bool tensor (n_latents,),       # firing_rate > 0
        "live_mask_1e-5": bool tensor (n_latents,),         # firing_rate > 1e-5
        "live_count_strict": int,
        "live_count_1e-5": int,
    }
  experiments/artifacts/l0_gemma_batchtopk/live_mask_metadata.json
    Sidecar with firing-rate distribution, threshold reasoning, and
    config used to generate the mask.

Usage:
  uv run python scripts/compute_l0_live_mask.py --smoke   # 100k tokens, ~30s
  uv run python scripts/compute_l0_live_mask.py --full    # 10M tokens, ~1h

Inference uses ``use_training_topk=True`` to match the project's
``dead_latent_fraction`` definition (sae_recursive_depth/metrics/core.py:66) and the L0 eval
in sae_recursive_depth/training/train_level0_batchtopk.py:461. A latent counts as having
fired when its post-activation latent is nonzero in any batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sae_recursive_depth.data.activations import hook_name_for, iter_residual_batches  # noqa: E402
from sae_recursive_depth.data.pile_tokens import shard_path  # noqa: E402
from sae_recursive_depth.training.loaders import load_base_model  # noqa: E402
from sae_recursive_depth.training.sae_models import BatchTopKSAE, SAEConfig  # noqa: E402

L0_ARTIFACT = REPO_ROOT / "data" / "artifacts" / "l0_gemma_batchtopk"
L0_CHECKPOINT = L0_ARTIFACT / "checkpoint.pt"
OUT_MASK = L0_ARTIFACT / "live_mask.pt"
OUT_META = L0_ARTIFACT / "live_mask_metadata.json"


def load_l0_sae(device: torch.device) -> BatchTopKSAE:
    ckpt = torch.load(L0_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = SAEConfig(
        arch=ckpt["arch"],
        d_in=int(ckpt["d_in"]),
        n_latents=int(ckpt["n_latents"]),
        sparsity=int(ckpt["sparsity"]),
    )
    sae = BatchTopKSAE(cfg)
    sae.W_enc.data = ckpt["W_enc"].to(torch.float32)
    sae.b_enc.data = ckpt["b_enc"].to(torch.float32)
    sae.W_dec.data = ckpt["W_dec"].to(torch.float32)
    sae.b_dec.data = ckpt["b_dec"].to(torch.float32)
    sae.jump_threshold.copy_(ckpt["jump_threshold"].to(torch.float32))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae.to(device)


def find_token_shard(n_tokens_target: int) -> Path:
    cache = REPO_ROOT / "data" / "artifacts" / "_token_cache"
    candidates = sorted(cache.glob("google_gemma-2-2b__n*__seq1024__seed*.pt"))
    big = [p for p in candidates if "n100000000" in p.name or "n500000000" in p.name]
    if not big:
        raise FileNotFoundError(
            "no Gemma-2-2B token shard with >=100M tokens; "
            "regenerate via sae_recursive_depth/data/pile_tokens.py:prepare_token_shard"
        )
    seed1 = [p for p in big if "seed1" in p.name and "seed10000" not in p.name]
    pick = seed1[0] if seed1 else big[0]
    return pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="100k tokens, fast verification")
    ap.add_argument("--full", action="store_true", help="10M tokens, FIX 3 production")
    ap.add_argument("--n-tokens", type=int, default=None, help="override token count")
    ap.add_argument("--fwd-batch-size", type=int, default=8)
    ap.add_argument("--sae-batch-size", type=int, default=2048)
    args = ap.parse_args()

    if not (args.smoke or args.full or args.n_tokens):
        ap.error("pick one: --smoke, --full, or --n-tokens=N")
    if args.smoke and args.full:
        ap.error("--smoke and --full are exclusive")

    n_tokens = args.n_tokens or (100_000 if args.smoke else 10_000_000)
    mode = "smoke" if args.smoke else ("full" if args.full else "manual")
    print(f"[live_mask] mode={mode} n_tokens={n_tokens}", flush=True)

    if not L0_CHECKPOINT.exists():
        sys.exit(f"missing L0 checkpoint: {L0_CHECKPOINT}")

    if torch.cuda.device_count() < 2:
        print(
            f"[live_mask] WARN: only {torch.cuda.device_count()} CUDA devices visible; "
            "expected 2+ for Gemma sharding",
            flush=True,
        )

    device = torch.device("cuda:0")
    n_devices = min(2, torch.cuda.device_count())
    print(f"[live_mask] loading Gemma-2-2B with n_devices={n_devices}", flush=True)
    t0 = time.time()
    model = load_base_model("google/gemma-2-2b", device="cuda", n_devices=n_devices)
    print(f"[live_mask] base model loaded in {time.time() - t0:.1f}s", flush=True)

    print("[live_mask] loading L0 SAE checkpoint", flush=True)
    sae = load_l0_sae(device)
    n_latents = sae.cfg.n_latents
    print(f"[live_mask] SAE n_latents={n_latents} d_in={sae.cfg.d_in}", flush=True)

    token_shard = find_token_shard(n_tokens)
    print(f"[live_mask] using token shard: {token_shard.name}", flush=True)

    hook_name = hook_name_for(layer=12, site="residual")
    firing_count = torch.zeros(n_latents, dtype=torch.int64, device=device)

    streamed = 0
    t1 = time.time()
    for batch in iter_residual_batches(
        model,
        hook_name=hook_name,
        token_shard_path=token_shard,
        fwd_batch_size=args.fwd_batch_size,
        sae_batch_size=args.sae_batch_size,
        device=device,
        n_tokens_target=n_tokens,
        dtype=torch.float32,
    ):
        with torch.no_grad():
            _, latents = sae(batch, use_training_topk=True)
            firing_count += (latents != 0).sum(dim=0).to(torch.int64)
        streamed += batch.shape[0]
        if streamed % (args.sae_batch_size * 64) == 0:
            elapsed = time.time() - t1
            rate = streamed / max(elapsed, 1e-3)
            eta = (n_tokens - streamed) / max(rate, 1.0)
            n_live_strict = int((firing_count > 0).sum().item())
            print(
                f"[live_mask] {streamed}/{n_tokens} tokens "
                f"({100 * streamed / n_tokens:.1f}%) | "
                f"rate={rate:.0f} tok/s | eta={eta:.0f}s | "
                f"live_strict={n_live_strict}",
                flush=True,
            )

    elapsed = time.time() - t1
    print(f"[live_mask] streaming done in {elapsed:.1f}s, total {streamed} tokens", flush=True)

    firing_count_cpu = firing_count.detach().cpu()
    firing_rate = firing_count_cpu.to(torch.float32) / float(streamed)
    live_strict = firing_count_cpu > 0
    live_1em5 = firing_rate > 1e-5

    out = {
        "firing_count": firing_count_cpu,
        "n_tokens": int(streamed),
        "firing_rate": firing_rate,
        "live_mask_strict": live_strict,
        "live_mask_1e-5": live_1em5,
        "live_count_strict": int(live_strict.sum().item()),
        "live_count_1e-5": int(live_1em5.sum().item()),
    }
    OUT_MASK.parent.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        smoke_path = OUT_MASK.with_name("live_mask.smoke.pt")
        torch.save(out, smoke_path)
        print(f"[live_mask] SMOKE wrote {smoke_path}", flush=True)
    else:
        tmp = OUT_MASK.with_suffix(".pt.tmp")
        torch.save(out, tmp)
        os.replace(tmp, OUT_MASK)
        print(f"[live_mask] wrote {OUT_MASK}", flush=True)

    fr = firing_rate
    p50 = float(fr.median().item())
    p95 = float(fr.quantile(0.95).item())
    p99 = float(fr.quantile(0.99).item())
    fr_max = float(fr.max().item())
    quantile_thresholds = {
        "1e-7": int((fr > 1e-7).sum().item()),
        "1e-6": int((fr > 1e-6).sum().item()),
        "1e-5": int((fr > 1e-5).sum().item()),
        "1e-4": int((fr > 1e-4).sum().item()),
        "1e-3": int((fr > 1e-3).sum().item()),
    }
    metadata = {
        "experiment_id": "l0_gemma_batchtopk",
        "n_tokens": int(streamed),
        "n_latents": n_latents,
        "live_count_strict": out["live_count_strict"],
        "live_count_1e-5": out["live_count_1e-5"],
        "live_count_by_threshold": quantile_thresholds,
        "firing_rate_p50": p50,
        "firing_rate_p95": p95,
        "firing_rate_p99": p99,
        "firing_rate_max": fr_max,
        "dead_fraction_strict": 1.0 - out["live_count_strict"] / n_latents,
        "dead_fraction_1e-5": 1.0 - out["live_count_1e-5"] / n_latents,
        "elapsed_seconds": elapsed,
        "token_shard": token_shard.name,
        "fwd_batch_size": args.fwd_batch_size,
        "sae_batch_size": args.sae_batch_size,
        "use_training_topk": True,
        "mode": mode,
        "live_mask_path": str(
            OUT_MASK.with_name("live_mask.smoke.pt")
            if mode == "smoke"
            else OUT_MASK
        ),
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if mode != "smoke":
        with open(OUT_META, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[live_mask] wrote {OUT_META}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
