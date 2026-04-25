"""Pile-uncopyrighted tokenization + shard cache.

Streams ``monology/pile-uncopyrighted`` through a model's tokenizer, packs
tokens into ``(n_seqs, seq_len)`` int32 tensors, and writes them to
``experiments/artifacts/_token_cache/`` keyed by (model_id, n_tokens, seq_len,
seed). Idempotent: a second call with the same arguments returns the cached
shard without re-downloading or re-tokenizing.

Token shards are tiny relative to activations (500M int32 ≈ 2 GB, vs ≈ 1 TB
for materialized fp16 activations at d_model=2304). Activations are
generated on the fly from these shards at training time; see
``src.data.activations``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_CACHE_DIR = REPO_ROOT / "experiments" / "artifacts" / "_token_cache"

# The pile-uncopyrighted revision sha is pinned by the dataset id alone
# because monology/pile-uncopyrighted commits are stable and the dataset is
# small. If reproducibility issues surface, switch this to a sha-pinned form.
PILE_DATASET_ID = "monology/pile-uncopyrighted"


def shard_path(model_id: str, n_tokens: int, seq_len: int, seed: int) -> Path:
    """Canonical on-disk path for the (model, n_tokens, seq_len, seed) shard."""
    safe_model = model_id.replace("/", "_")
    name = f"{safe_model}__n{n_tokens}__seq{seq_len}__seed{seed}.pt"
    return TOKEN_CACHE_DIR / name


def prepare_token_shard(
    model_id: str,
    *,
    n_tokens: int,
    seq_len: int = 1024,
    seed: int = 0,
    dataset_split: str = "train",
) -> Path:
    """Tokenize Pile until ``n_tokens`` are produced; save as one shard.

    Output shard: ``(n_seqs, seq_len)`` int32, where ``n_seqs * seq_len >= n_tokens``.
    Sequences are EOS-separated concatenations of pile documents.
    """
    out = shard_path(model_id, n_tokens, seq_len, seed)
    if out.exists():
        return out

    # Local imports so the module can be imported without datasets/transformers
    # being available (e.g., in --smoke paths that don't touch real data).
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.eos_token_id is None:
        # GPT-2 tokenizer has eos_token; defensive fallback.
        tok.eos_token = tok.pad_token or "<|endoftext|>"

    n_seqs = (n_tokens + seq_len - 1) // seq_len
    out_buf = torch.empty(n_seqs, seq_len, dtype=torch.int32)
    seq_idx = 0
    cursor = 0  # position within the current sequence
    eos = tok.eos_token_id

    ds = load_dataset(PILE_DATASET_ID, split=dataset_split, streaming=True)
    # The streaming dataset is deterministic in document order; our seed
    # determines a starting offset so different shards see different slices.
    if seed:
        # `take` after `skip` is fine on streaming datasets; pile-uncopyrighted
        # is large enough that any shard offset stays within range.
        ds = ds.skip(seed * 1000)

    current_seq = out_buf[seq_idx]
    for example in ds:
        if seq_idx >= n_seqs:
            break
        text = example.get("text", "")
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eos)
        i = 0
        while i < len(ids) and seq_idx < n_seqs:
            take = min(seq_len - cursor, len(ids) - i)
            current_seq[cursor : cursor + take] = torch.tensor(
                ids[i : i + take], dtype=torch.int32
            )
            cursor += take
            i += take
            if cursor == seq_len:
                seq_idx += 1
                cursor = 0
                if seq_idx < n_seqs:
                    current_seq = out_buf[seq_idx]

    if seq_idx < n_seqs:
        # Truncate if Pile somehow ran out (will not happen in practice).
        out_buf = out_buf[:seq_idx]

    tmp = out.with_suffix(".pt.tmp")
    torch.save(
        {
            "tokens": out_buf,
            "model_id": model_id,
            "n_tokens": n_tokens,
            "seq_len": seq_len,
            "seed": seed,
            "dataset_id": PILE_DATASET_ID,
        },
        tmp,
    )
    os.replace(tmp, out)
    return out


def load_token_shard(path: Path) -> torch.Tensor:
    """Return the ``(n_seqs, seq_len)`` int32 tensor from ``path``."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["tokens"]
