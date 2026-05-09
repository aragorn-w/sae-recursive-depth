"""Autointerpretation entry point.

Handles the `autointerp_all` row. Uses Llama-3.1-8B-Instruct on GPU 3 to
generate and score feature explanations for every depth-1 and depth-2 SAE
in the matrix.

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

from sae_recursive_depth.data.activations import hook_name_for, iter_residual_batches
from sae_recursive_depth.data.pile_tokens import prepare_token_shard
from sae_recursive_depth.training.harness import experiment_context
from sae_recursive_depth.training.loaders import load_base_model
from sae_recursive_depth.training.metrics_io import write_metrics_tsv

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "data" / "artifacts"

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
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
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
    base,
    base_model_id: str,
    layer: int,
    site: str,
    feature_dirs: torch.Tensor,
    token_shard_path: Path,
    n_tokens: int,
    device: torch.device,
):
    """Run a *pre-loaded* base model on tokens; return (activations, raw_tokens).

    ``activations`` is shape (n_tokens, n_chosen_latents) on CPU — projected
    onto the supplied ``feature_dirs`` rows only, so callers that pre-slice
    ``feature_dirs`` to just the features they intend to score keep memory
    bounded by the slice rather than the full SAE width.
    ``raw_tokens`` is shape (n_tokens,) int32 on CPU (input tokens in order).

    The caller owns ``base``'s lifecycle so multiple SAEs sharing one base
    model don't reload it. Pass ``base=None`` to load+free internally
    (legacy single-SAE call path).
    """
    own_base = base is None
    if own_base:
        base = load_base_model(base_model_id, device=device)
    hook = hook_name_for(layer, site)
    feature_dirs = feature_dirs.to(device)  # (n_chosen, d_model)

    activations_buf: list[torch.Tensor] = []
    tokens_buf: list[torch.Tensor] = []
    yielded = 0
    fwd_batch = 4 if "gemma" in base_model_id else 24

    from sae_recursive_depth.data.pile_tokens import load_token_shard

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
        proj = (flat @ feature_dirs.T).cpu()    # (B*seq_len, n_chosen)
        activations_buf.append(proj)
        tokens_buf.append(batch.reshape(-1).cpu().to(torch.int32))
        yielded += proj.shape[0]
    activations = torch.cat(activations_buf, dim=0)[:n_tokens]
    raw_tokens = torch.cat(tokens_buf, dim=0)[:n_tokens]
    if own_base:
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
    """Detection score: fraction of correct yes/no answers (sequential)."""
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


def _score_one_feature_batched(
    llama, llama_tok, *, explanation: str, pos_snips: list[str], neg_snips: list[str]
) -> float:
    """Detection score with batched inference. Equivalent to _score_one_feature
    at temperature=0 modulo fp16 nondeterminism (~1-2 prompts per feature)."""
    prompts = [_detection_prompt(explanation, s) for s in pos_snips] + [
        _detection_prompt(explanation, s) for s in neg_snips
    ]
    if not prompts:
        return float("nan")
    device = next(llama.parameters()).device
    inp = llama_tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = llama.generate(
            **inp,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=llama_tok.eos_token_id,
        )
    new_tokens = out[:, inp["input_ids"].shape[1]:]
    decoded = llama_tok.batch_decode(new_tokens, skip_special_tokens=True)
    correct = 0
    total = 0
    n_pos = len(pos_snips)
    for i, resp in enumerate(decoded):
        v = _parse_yes_no(resp)
        is_pos = i < n_pos
        if (v == 1 and is_pos) or (v == 0 and not is_pos):
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


def _run(targets, scores_path: Path, device: torch.device, *, seed: int = 0, smoke_n: int = 0):
    """Phase 1+2+3 pipeline.

    Writes scores to ``scores_path`` (overwrites). Returns
    ``(all_scores, all_null)`` lists, or ``(None, None)`` if Llama failed
    to load.
    """
    all_scores: list[float] = []
    all_null: list[float] = []
    token_shard_cache: dict[str, Path] = {}
    base_tok_cache: dict[str, object] = {}

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w") as f:
        f.write("sae_id\tfeature\texplanation\tdetection_score\tnull_detection_score\n")

    # Phase 1: harvest activations for every target SAE while Gemma/GPT-2
    # is loaded. Group targets by base_model so each base only loads once.
    from collections import defaultdict
    targets_by_base: dict[str, list[dict]] = defaultdict(list)
    prepared: list[dict] = []
    for spec in targets:
        blob = _load_sae_ckpt(spec["id"])
        if blob is None:
            print(f"[autointerp] skip {spec['id']}: no checkpoint", flush=True)
            continue
        row = blob.get("row") or {}
        base_model_id = row.get("base_model")
        layer = int(row.get("layer", 0))
        site = row.get("site", "residual")
        feature_dirs_full = _decoder_directions_in_dmodel(blob)
        n_latents = feature_dirs_full.shape[0]
        chosen = _select_features(n_latents, N_FEATURES_PER_SAE, seed=seed)
        feature_dirs_chosen = feature_dirs_full[chosen].contiguous()
        entry = {
            "id": spec["id"],
            "base_model_id": base_model_id,
            "layer": layer,
            "site": site,
            "blob": blob,
            "chosen": chosen,
            "feature_dirs_chosen": feature_dirs_chosen,
        }
        prepared.append(entry)
        targets_by_base[base_model_id].append(entry)

    for base_model_id, entries in targets_by_base.items():
        if base_model_id not in token_shard_cache:
            token_shard_cache[base_model_id] = prepare_token_shard(
                base_model_id, n_tokens=SCORING_TOKENS, seq_len=SEQ_LEN, seed=99
            )
        shard = token_shard_cache[base_model_id]
        if base_model_id not in base_tok_cache:
            from transformers import AutoTokenizer
            base_tok_cache[base_model_id] = AutoTokenizer.from_pretrained(base_model_id)
        base = load_base_model(base_model_id, device=device)
        try:
            for entry in entries:
                activations, token_ids = _harvest_activations(
                    base,
                    base_model_id,
                    entry["layer"],
                    entry["site"],
                    entry["feature_dirs_chosen"],
                    shard,
                    SCORING_TOKENS,
                    device,
                )
                entry["activations"] = activations
                entry["token_ids"] = token_ids
        finally:
            del base
            torch.cuda.empty_cache()

    # Phase 2: load Llama after every base model has been freed.
    llama, llama_tok = _build_llama(device)
    if llama is None:
        return None, None

    # Phase 3: score each prepared SAE against its cached activations.
    for entry in prepared:
        activations = entry.get("activations")
        if activations is None:
            continue
        token_ids = entry["token_ids"]
        base_tok = base_tok_cache[entry["base_model_id"]]
        chosen = entry["chosen"]
        spec_id = entry["id"]
        chosen_iter = chosen if smoke_n <= 0 else chosen[:smoke_n]
        for col_idx, feat_idx in enumerate(chosen_iter):
            col = activations[:, col_idx]
            top_idx = torch.topk(col, k=N_TOP_CONTEXTS + N_HELD_OUT_POS, largest=True).indices.tolist()
            low_idx_pool = torch.topk(col.abs(), k=col.shape[0] // 4, largest=False).indices.tolist()
            col_median = col.median().item()
            low_idx_pool = [i for i in low_idx_pool if col[i].item() < col_median]
            if len(low_idx_pool) < N_HELD_OUT_NEG:
                continue
            random.Random(seed * 7919 + feat_idx).shuffle(low_idx_pool)
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

            det_score = _score_one_feature_batched(
                llama, llama_tok,
                explanation=explanation,
                pos_snips=pos_snips,
                neg_snips=neg_snips,
            )

            null_explanation = random.Random(seed + feat_idx + 13).choice(
                ["xylophone", "scaffold", "tangerine", "monolith", "trellis", "embargo"]
            )
            null_score = _score_one_feature_batched(
                llama, llama_tok,
                explanation=null_explanation,
                pos_snips=pos_snips,
                neg_snips=neg_snips,
            )

            with scores_path.open("a") as f:
                safe_expl = explanation.replace("\t", " ").replace("\n", " ")[:200]
                f.write(
                    f"{spec_id}\t{feat_idx}\t{safe_expl}\t"
                    f"{det_score:.4g}\t{null_score:.4g}\n"
                )

            if not np.isnan(det_score):
                all_scores.append(det_score)
            if not np.isnan(null_score):
                all_null.append(null_score)

    return all_scores, all_null


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

        scores_path: Path = ctx.artifact_dir / "scores.tsv"
        all_scores, all_null = _run(targets, scores_path, device, seed=ctx.seed)
        if all_scores is None:
            ctx.status = "failed"
            ctx.notes = "could not load Llama-3.1-8B-Instruct"
            return

        median_score = float(np.median(all_scores)) if all_scores else float("nan")
        null_p95 = float(np.percentile(all_null, 95)) if all_null else float("nan")
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


def main_standalone() -> None:
    """CLI entry for parallel workers / smoke tests. Bypasses experiment_context
    so it does not touch results.tsv — orchestrator does the merge + commit."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target-saes", required=True,
                   help="comma-separated list of SAE ids to score")
    p.add_argument("--output-tsv", required=True,
                   help="path to output scores.tsv")
    p.add_argument("--device", default=None,
                   help="cuda device override (e.g., 'cuda:1')")
    p.add_argument("--smoke-features", type=int, default=0,
                   help="if >0, only score this many features per SAE")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    target_ids = [s.strip() for s in args.target_saes.split(",") if s.strip()]
    targets = [{"id": s} for s in target_ids]
    scores_path = Path(args.output_tsv)
    device = torch.device(args.device) if args.device else _device()

    print(f"[autointerp-standalone] targets={target_ids} device={device} "
          f"smoke={args.smoke_features} output={scores_path}", flush=True)
    all_scores, all_null = _run(
        targets, scores_path, device,
        seed=args.seed,
        smoke_n=args.smoke_features,
    )
    if all_scores is None:
        print("[autointerp-standalone] FAILED: could not load Llama", flush=True)
        raise SystemExit(1)

    median_score = float(np.median(all_scores)) if all_scores else float("nan")
    null_p95 = float(np.percentile(all_null, 95)) if all_null else float("nan")
    print(f"[autointerp-standalone] done: {len(all_scores)} features, "
          f"median_det={median_score:.4g} null_p95={null_p95:.4g}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main_standalone()
    else:
        main()
