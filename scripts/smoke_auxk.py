#!/usr/bin/env python3
"""Structural smoke test for the auxk auxiliary loss in train_meta_sae.py.

Asserts: (a) the auxk path runs without crashing, (b) no NaN/inf, (c) loss
decreases roughly the same as without auxk. Does NOT assert that auxk
reduces dead_frac on this synthetic data — with k=4, n_lat=1024, batch=256
each latent's expected fire rate is ~1/batch, so most "dead" detections
inside the 1500-step budget are statistical noise rather than truly dead
latents. Real dead-latent rescue can only be measured on rich-structure
parent decoder data (Gemma Scope, BatchTopK level-0).

Run: uv run python scripts/smoke_auxk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.sae_models import build_sae

DEAD_STEPS = 20  # smaller than production AUXK_DEAD_STEPS so dead latents become eligible quickly
AUXK_K = 128
AUXK_ALPHA = 1.0 / 32.0
LOG_EVERY = 200


def run(auxk_on: bool, n_steps: int = 1500, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    # Heavy over-parameterization so many latents stay dead — the regime where auxk should help.
    d_in, n_lat, k = 64, 1024, 4
    parent_n = 512
    # Synthetic data with low-rank mixture structure: 200 latent atoms, each input is a
    # sparse mixture of 4 atoms. This gives revived dead latents a meaningful target
    # space (vs uniform random which has no structure for them to fit).
    n_atoms = 200
    atoms = F.normalize(torch.randn(n_atoms, d_in), dim=1)
    coeffs = torch.zeros(parent_n, n_atoms)
    for i in range(parent_n):
        idx = torch.randperm(n_atoms)[:4]
        coeffs[i, idx] = torch.randn(4).abs()
    x = F.normalize(coeffs @ atoms, dim=1).cuda()
    sae = build_sae(arch="batchtopk", d_in=d_in, n_latents=n_lat, sparsity=k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    dead_counter = torch.zeros(n_lat, dtype=torch.int32, device="cuda") if auxk_on else None
    batch_size = 256
    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, parent_n, (batch_size,), device="cuda")
        batch = x[idx]
        recon, latents = sae(batch, use_training_topk=True)
        loss = F.mse_loss(recon, batch)
        if auxk_on:
            fired = (latents > 0).any(dim=0)
            dead_counter = torch.where(fired, torch.zeros_like(dead_counter), dead_counter + 1)
            dead_mask = dead_counter > DEAD_STEPS
            n_dead = int(dead_mask.sum().item())
            if n_dead > 0:
                pre_all = F.relu(sae.preact(batch))
                dead_pre = pre_all[:, dead_mask]
                B = dead_pre.shape[0]
                k_aux = min(AUXK_K, n_dead)
                total_kept = B * k_aux
                flat = dead_pre.reshape(-1)
                if total_kept < flat.numel():
                    topk = torch.topk(flat, total_kept, sorted=False)
                    mask = torch.zeros_like(flat)
                    mask.scatter_(0, topk.indices, 1.0)
                    dead_pre = (flat * mask).reshape(B, n_dead)
                W_dec_dead = sae.W_dec[dead_mask]
                aux_recon = dead_pre @ W_dec_dead
                residual = (batch - recon).detach()
                aux_loss = F.mse_loss(aux_recon, residual)
                loss = loss + AUXK_ALPHA * aux_loss
        if not torch.isfinite(loss):
            return {"ok": False, "step": step, "loss": float("nan"), "auxk_on": auxk_on}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
        losses.append(float(loss.item()))
        if step % LOG_EVERY == 0:
            with torch.no_grad():
                _, lat = sae(x[:256], use_training_topk=True)
                df_now = ((lat > 0).sum(dim=0) == 0).float().mean().item()
            print(f"  [auxk={auxk_on}] step={step:4d} loss={loss.item():.5f} dead_frac={df_now:.3f}")
    with torch.no_grad():
        recon, latents = sae(x, use_training_topk=True)
        dead_frac = float(((latents > 0).sum(dim=0) == 0).float().mean().item())
        ve = 1.0 - (x - recon).pow(2).mean().item() / (x - x.mean(0, keepdim=True)).pow(2).mean().item()
    return {"ok": True, "loss_first": losses[0], "loss_last": losses[-1], "dead_frac": dead_frac, "ve": ve, "auxk_on": auxk_on}


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable; smoke test requires a GPU")
        return 2
    base = run(auxk_on=False)
    aux = run(auxk_on=True)
    print(f"baseline:  ok={base['ok']} loss {base['loss_first']:.4f} -> {base['loss_last']:.4f}  dead_frac={base['dead_frac']:.3f}  ve={base['ve']:.3f}")
    print(f"auxk on :  ok={aux['ok']}  loss {aux['loss_first']:.4f} -> {aux['loss_last']:.4f}  dead_frac={aux['dead_frac']:.3f}  ve={aux['ve']:.3f}")
    fail = []
    if not base["ok"] or not aux["ok"]:
        fail.append("non-finite loss")
    if not (aux["loss_last"] < base["loss_first"]):
        fail.append("auxk loss did not decrease from start")
    # Auxk loss should not blow up the optimization by more than 50% relative
    # to baseline final loss. If it does, the auxk gradient is fighting the
    # main loss in a way that suggests a sign or scale bug.
    if aux["loss_last"] > 1.5 * base["loss_last"]:
        fail.append(f"auxk loss diverged from baseline: {aux['loss_last']:.4f} vs {base['loss_last']:.4f}")
    if fail:
        print("FAIL:", "; ".join(fail))
        return 1
    print("PASS (structural; dead_frac improvement validated only on real data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
