"""Autointerpretation entry point.

Dispatched from `scripts/run_loop.sh` for the `autointerp_all` row
(SPEC.md:44). Uses Llama-3.1-8B-Instruct on GPU 3 to generate and score
feature explanations for every depth-1 and depth-2 SAE in the matrix.

Per analysis rule 5, exactly 200 features per SAE are scored using a
Paulo-and-Belrose-style detection protocol (arXiv:2501.16615): for each
feature, top-activating contexts are summarized into a one-line
explanation, then Llama is asked to discriminate held-out activating
contexts from non-activating ones using only the explanation.

Outputs:
  - experiments/artifacts/autointerp_all/scores.tsv  (one row per feature)
  - ctx.metrics.median_detection_score
  - ctx.metrics.null_detection_score_p95
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.data.activations import hook_name_for, iter_residual_batches
from src.data.pile_tokens import prepare_token_shard
from src.training.harness import experiment_context
from src.training.loaders import load_base_model
from src.training.metrics_io import write_metrics_tsv

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "experiments" / "artifacts"

LLAMA_ID = "meta-llama/Llama-3.1-8B-Instruct"
N_FEATURES_PER_SAE = 200
N_TOP_CONTEXTS = 6        # examples shown in the explanation prompt
N_HELD_OUT_POS = 6        # held-out high-activating contexts for scoring
N_HELD_OUT_NEG = 6        # held-out low-activating contexts for scoring
CONTEXT_WIN = 16          # tokens of context around the activation peak
SCORING_TOKENS = 200_000  # token budget per SAE for activation harvesting
SEQ_LEN = 1024


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sae_ckpt(eid: str):
    path = ARTIFACTS / eid / "checkpoint.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob


def _decoder_directions_in_dmodel(blob: dict) -> torch.Tensor:
    """Return ``(n_latents, d_model)`` unit-normalized feature directions.

    Both level-0 and meta-SAE checkpoints in this project store W_dec as
    (n_latents, d_in=d_model), so a row is a feature direction directly
    usable to project residual activations.
    """
    W = blob["W_dec"].to(torch.float32)
    if W.ndim != 2:
        raise ValueError(f"unexpected W_dec ndim {W.ndim}")
    return torch.nn.functional.normalize(W, dim=1)


def _build_llama(device: torch.device):
    """Return (model, tokenizer) for Llama-3.1-8B-Instruct, or (None, None)
    if it cannot be loaded (network down, weights missing, OOM)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tok
    except Exception as e:
        print(f"[autointerp] Llama load failed: {e!r}", flush=True)
        return None, None


def _llama_complete(
    model,
    tok,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    inp = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def _explanation_prompt(contexts: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in contexts)
    return (
        "You are analyzing a learned feature in a neural language model.\n"
        "The feature activates strongly on the following text snippets:\n"
        f"{bullets}\n\n"
        "In one short sentence, describe the concept this feature detects. "
        "Do not include the word 'feature' in your answer.\n"
        "Concept:"
    )


def _detection_prompt(explanation: str, snippet: str) -> str:
    return (
        "Concept: " + explanation.strip() + "\n"
        "Snippet: " + snippet + "\n"
        "Does the snippet match the concept? Answer with a single word: yes or no.\n"
        "Answer:"
    )


def _parse_yes_no(response: str) -> int:
    """Return 1 if the model said yes, 0 if no, -1 if undetermined."""
    s = response.strip().lower()
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return -1


def _harvest_activations(
    base_model_id: str,
    layer: int,
    site: str,
    feature_dirs: torch.Tensor,
    token_shard_path: Path,
    n_tokens: int,
    device: torch.device,
):
    """Run base model on tokens, return (activations, raw_tokens) tensors.

    ``activations`` is shape (n_tokens, n_latents) on CPU.
    ``raw_tokens`` is shape (n_tokens,) int32 on CPU (the input tokens, in
    the order they were processed).
    """
    base = load_base_model(base_model_id, device=device)
    hook = hook_name_for(layer, site)
    feature_dirs = feature_dirs.to(device)  # (n_latents, d_model)

    activations_buf: list[torch.Tensor] = []
    tokens_buf: list[torch.Tensor] = []
    yielded = 0
    fwd_batch = 4 if "gemma" in base_model_id else 24

    # Stream once through residual batches but keep the underlying tokens
    # too; iter_residual_batches yields (sae_batch_size, d_model). We need a
    # lower-level loop to keep tokens aligned, so re-use the same logic
    # inline.
    from src.data.pile_tokens import load_token_shard

    tokens = load_token_shard(token_shard_path)
    captured: dict[str, torch.Tensor] = {}

    def _hook(act, hook):  # noqa: ARG001
        captured["x"] = act.detach()

    seq_cursor = 0
    while yielded < n_tokens and seq_cursor < tokens.shape[0]:
        end = min(seq_cursor + fwd_batch, tokens.shape[0])
        batch = tokens[seq_cursor:end].to(device=device, dtype=torch.long)
        seq_cursor = end
        captured.clear()
        with torch.no_grad():
            base.run_with_hooks(batch, fwd_hooks=[(hook, _hook)], return_type=None)
        acts = captured["x"]  # (B, seq_len, d_model)
        flat = acts.reshape(-1, acts.shape[-1])  # (B*seq_len, d_model)
        proj = (flat @ feature_dirs.T).cpu()    # (B*seq_len, n_latents)
        activations_buf.append(proj)
        tokens_buf.append(batch.reshape(-1).cpu().to(torch.int32))
        yielded += proj.shape[0]
    activations = torch.cat(activations_buf, dim=0)[:n_tokens]
    raw_tokens = torch.cat(tokens_buf, dim=0)[:n_tokens]
    del base
    torch.cuda.empty_cache()
    return activations, raw_tokens


def _make_snippet(token_ids: torch.Tensor, peak_idx: int, base_tok, win: int) -> str:
    lo = max(0, peak_idx - win)
    hi = min(token_ids.shape[0], peak_idx + win + 1)
    return base_tok.decode(token_ids[lo:hi].tolist())


def _select_features(n_total: int, n_pick: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if n_total <= n_pick:
        return list(range(n_total))
    return sorted(rng.sample(range(n_total), n_pick))


def _score_one_feature(
    llama, llama_tok, *, explanation: str, pos_snips: list[str], neg_snips: list[str]
) -> float:
    """Detection score: fraction of correct yes/no answers."""
    correct = 0
    total = 0
    for s in pos_snips:
        out = _llama_complete(llama, llama_tok, _detection_prompt(explanation, s), max_new_tokens=4)
        v = _parse_yes_no(out)
        if v == 1:
            correct += 1
        if v != -1:
            total += 1
    for s in neg_snips:
        out = _llama_complete(llama, llama_tok, _detection_prompt(explanation, s), max_new_tokens=4)
        v = _parse_yes_no(out)
        if v == 0:
            correct += 1
        if v != -1:
            total += 1
    if total == 0:
        return float("nan")
    return correct / total


def _enumerate_target_saes(deps: list[str]) -> list[dict]:
    """Pick depth-1 and depth-2 SAEs from the dependency list."""
    out = []
    for d in deps:
        if "_d1_s" in d or "_d2_s" in d:
            out.append({"id": d})
    return out


def main() -> None:
    with experiment_context(arch_hint="autointerp") as ctx:
        if ctx.smoke:
            ctx.notes = "smoke: autointerp harness ok"
            return
        ctx.init_wandb()

        device = _device()
        deps = ctx.row.get("dependencies") or []
        targets = _enumerate_target_saes(deps)
        if not targets:
            ctx.status = "failed"
            ctx.notes = "no depth-1/2 dependencies in autointerp row"
            return

        llama, llama_tok = _build_llama(device)
        if llama is None:
            ctx.status = "failed"
            ctx.notes = "could not load Llama-3.1-8B-Instruct"
            return

        rng_seed = ctx.seed
        scores_path: Path = ctx.artifact_dir / "scores.tsv"
        with scores_path.open("w") as f:
            f.write("sae_id\tfeature\texplanation\tdetection_score\tnull_detection_score\n")

        all_scores: list[float] = []
        all_null: list[float] = []
        token_shard_cache: dict[str, Path] = {}
        base_tok_cache: dict[str, object] = {}

        for spec in targets:
            blob = _load_sae_ckpt(spec["id"])
            if blob is None:
                continue
            row = blob.get("row") or {}
            base_model_id = row.get("base_model")
            layer = int(row.get("layer", 0))
            site = row.get("site", "residual")
            if base_model_id not in token_shard_cache:
                token_shard_cache[base_model_id] = prepare_token_shard(
                    base_model_id, n_tokens=SCORING_TOKENS, seq_len=SEQ_LEN, seed=99
                )
            shard = token_shard_cache[base_model_id]

            if base_model_id not in base_tok_cache:
                from transformers import AutoTokenizer
                base_tok_cache[base_model_id] = AutoTokenizer.from_pretrained(base_model_id)
            base_tok = base_tok_cache[base_model_id]

            feature_dirs = _decoder_directions_in_dmodel(blob)
            n_latents = feature_dirs.shape[0]
            chosen = _select_features(n_latents, N_FEATURES_PER_SAE, seed=rng_seed)

            activations, token_ids = _harvest_activations(
                base_model_id, layer, site, feature_dirs, shard, SCORING_TOKENS, device
            )
            # For each chosen feature, get top-K tokens and held-out activating tokens.
            for feat_idx in chosen:
                col = activations[:, feat_idx]
                top_idx = torch.topk(col, k=N_TOP_CONTEXTS + N_HELD_OUT_POS, largest=True).indices.tolist()
                low_idx_pool = torch.topk(col.abs(), k=col.shape[0] // 4, largest=False).indices.tolist()
                low_idx_pool = [i for i in low_idx_pool if col[i].item() < col.median().item()]
                if len(low_idx_pool) < N_HELD_OUT_NEG:
                    continue
                random.Random(rng_seed * 7919 + feat_idx).shuffle(low_idx_pool)
                neg_idx = low_idx_pool[:N_HELD_OUT_NEG]

                top_for_explain = top_idx[:N_TOP_CONTEXTS]
                top_for_score = top_idx[N_TOP_CONTEXTS:]
                explain_snips = [_make_snippet(token_ids, i, base_tok, CONTEXT_WIN) for i in top_for_explain]
                pos_snips = [_make_snippet(token_ids, i, base_tok, CONTEXT_WIN) for i in top_for_score]
                neg_snips = [_make_snippet(token_ids, i, base_tok, CONTEXT_WIN) for i in neg_idx]

                expl_raw = _llama_complete(llama, llama_tok, _explanation_prompt(explain_snips), max_new_tokens=40)
                explanation = expl_raw.strip().split("\n")[0]
                if not explanation:
                    explanation = "(no explanation)"

                det_score = _score_one_feature(
                    llama, llama_tok,
                    explanation=explanation,
                    pos_snips=pos_snips,
                    neg_snips=neg_snips,
                )

                # Null score: same scoring with a random word as "explanation".
                null_explanation = random.Random(rng_seed + feat_idx + 13).choice(
                    ["xylophone", "scaffold", "tangerine", "monolith", "trellis", "embargo"]
                )
                null_score = _score_one_feature(
                    llama, llama_tok,
                    explanation=null_explanation,
                    pos_snips=pos_snips,
                    neg_snips=neg_snips,
                )

                with scores_path.open("a") as f:
                    safe_expl = explanation.replace("\t", " ").replace("\n", " ")[:200]
                    f.write(
                        f"{spec['id']}\t{feat_idx}\t{safe_expl}\t"
                        f"{det_score:.4g}\t{null_score:.4g}\n"
                    )

                if not np.isnan(det_score):
                    all_scores.append(det_score)
                if not np.isnan(null_score):
                    all_null.append(null_score)

        if all_scores:
            median_score = float(np.median(all_scores))
        else:
            median_score = float("nan")
        if all_null:
            null_p95 = float(np.percentile(all_null, 95))
        else:
            null_p95 = float("nan")

        write_metrics_tsv(
            ctx.artifact_dir / "metrics.tsv",
            {
                "variance_explained": None,
                "pwmcc": None,
                "mmcs": None,
                "dead_latent_fraction": None,
                "median_detection_score": median_score,
                "null_detection_score_p95": null_p95,
            },
        )
        ctx.metrics["median_detection_score"] = median_score
        ctx.metrics["null_detection_score_p95"] = null_p95
        ctx.notes = (
            f"autointerp scored {len(all_scores)} features across "
            f"{len(targets)} SAEs; median={median_score:.3g} null_p95={null_p95:.3g}"
        )


if __name__ == "__main__":
    main()
