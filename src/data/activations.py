"""On-the-fly residual-stream activation streaming.

Given a HookedTransformer base model + a pile-uncopyrighted token shard,
yield batched ``(sae_batch_size, d_model)`` tensors of layer-N residuals
suitable for SAE training. No fp16 activation cache lives on disk; the
token shard is cached (small) and the model forward pass runs each time.

The SAE batch size and the model forward batch size decouple: the model
forward runs at ``fwd_batch_size`` token-sequences, producing
``fwd_batch_size * seq_len`` residual vectors per pass; those are queued
and re-batched into ``sae_batch_size`` units before being yielded.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch


def hook_name_for(layer: int, site: str) -> str:
    """TransformerLens hook name for a residual-stream / mlp / attn output."""
    if site == "residual":
        return f"blocks.{layer}.hook_resid_post"
    if site == "mlp_out":
        return f"blocks.{layer}.hook_mlp_out"
    if site == "attn_out":
        return f"blocks.{layer}.hook_attn_out"
    raise ValueError(f"unknown site {site!r}")


def iter_residual_batches(
    model,
    *,
    hook_name: str,
    token_shard_path: Path,
    fwd_batch_size: int,
    sae_batch_size: int,
    device: torch.device | str,
    n_tokens_target: int,
    dtype: torch.dtype = torch.float32,
    seq_cursor_start: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield ``(sae_batch_size, d_model)`` residual tensors on ``device``.

    Stops once ``n_tokens_target`` residual vectors have been yielded across
    all batches. The model is run with ``torch.no_grad()`` and a single hook
    that captures the requested activation. Sequences are processed in shard
    order; reproducibility flows from the token shard's seed.

    ``seq_cursor_start`` lets a resumed run jump past sequences already
    consumed by an earlier crashed run. Token shards are deterministic given
    the seed, so resuming at sequence index k yields the same residuals the
    crashed run would have produced from step k onward.
    """
    from src.data.pile_tokens import load_token_shard

    tokens = load_token_shard(token_shard_path)  # (n_seqs, seq_len), int32
    n_seqs, seq_len = tokens.shape

    queue: list[torch.Tensor] = []
    queued = 0
    total_yielded = 0
    captured: dict[str, torch.Tensor] = {}

    def _hook(act, hook):  # noqa: ARG001
        captured["x"] = act.detach()

    model.eval()

    seq_cursor = max(0, int(seq_cursor_start))
    while total_yielded < n_tokens_target and seq_cursor < n_seqs:
        end = min(seq_cursor + fwd_batch_size, n_seqs)
        batch_tokens = tokens[seq_cursor:end].to(device=device, dtype=torch.long)
        seq_cursor = end

        captured.clear()
        with torch.no_grad():
            model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(hook_name, _hook)],
                return_type=None,
            )
        acts = captured["x"]  # (B, seq_len, d_model)
        flat = acts.reshape(-1, acts.shape[-1]).to(dtype)
        queue.append(flat)
        queued += flat.shape[0]

        # Drain into sae_batch_size-sized chunks.
        while queued >= sae_batch_size:
            stacked = torch.cat(queue, dim=0)
            chunk = stacked[:sae_batch_size]
            remainder = stacked[sae_batch_size:]
            queue = [remainder] if remainder.shape[0] > 0 else []
            queued = remainder.shape[0]
            total_yielded += chunk.shape[0]
            yield chunk
            if total_yielded >= n_tokens_target:
                return
