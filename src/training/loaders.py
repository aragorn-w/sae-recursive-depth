"""Uniform loader for level-0 SAEs and their base models.

Every training entry point that needs a level-0 SAE goes through
``load_level0(...)``. Per training rule 6, no module under ``src/training/``
may load an SAE or base model by HuggingFace id directly; all such loads
route here so revisions are pinned in one place and the return type is
uniform across SAELens, TransformerLens, and local checkpoints.

Routing:
  level0_source                                 arch         handler
  ------------------------------------------    ----------   --------------------
  "google/gemma-scope-2b-pt-res-canonical"      jumprelu     SAELens release
  "bartbussmann/BatchTopK"                      batchtopk    HF state_dict (404 — see PINNED_REVISIONS)
  "train_from_scratch"                          batchtopk    local checkpoint
  "flat_sae_on_activations"                     none         local checkpoint
  "random_gaussian"                             none         synthesized in memory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

Arch = Literal["jumprelu", "batchtopk", "none"]
Site = Literal["residual", "mlp_out", "attn_out"]

REPO_ROOT = Path(__file__).resolve().parents[2]

# The EXPERIMENTS.yaml matrix uses the HF-id form "google/gemma-scope-..." as
# `level0_source` while SAELens identifies the same release as just
# "gemma-scope-2b-pt-res-canonical" in SAE.from_pretrained. Keep both.
GEMMA_JUMPRELU_MATRIX_ID = "google/gemma-scope-2b-pt-res-canonical"
GEMMA_JUMPRELU_SAELENS_RELEASE = "gemma-scope-2b-pt-res-canonical"


# Pinned revisions for every external artifact. `_require_revision` refuses to
# load unpinned sources to prevent silent drift across upstream updates.
#
# The SAELens release id `gemma-scope-2b-pt-res-canonical` is an internal
# alias that maps to HF repo `google/gemma-scope-2b-pt-res` at the revision
# below. The pin is expressed as the pair of (HF sha, sae_lens version) since
# both influence which weights actually load.
PINNED_REVISIONS: dict[str, str] = {
    # Gemma Scope JumpReLU SAEs, residual stream, layer 12 canonical width 65k.
    # HF sha fetched on 2026-04-23 from google/gemma-scope-2b-pt-res.
    GEMMA_JUMPRELU_MATRIX_ID: (
        "hf:fd571b47c1c64851e9b1989792367b9babb4af63+sae_lens:6.39.0"
    ),
    # bartbussmann/BatchTopK is referenced by the matrix for the GPT-2 Small
    # anchor rows, but the repo returns 404 on HF. Left unpinned so the
    # loader explicitly raises with a clear error rather than silently
    # resolving to something else. Human must edit EXPERIMENTS.yaml to point
    # at the real release id.
}

# Base-model weight pins. Distinct from PINNED_REVISIONS to keep "level-0 SAE
# release" and "base LM weights" cleanly separated.
PINNED_BASE_MODEL_REVISIONS: dict[str, str] = {
    "google/gemma-2-2b": "c5ebcd40d208330abc697524c919956e692655cf",
    "openai-community/gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}


def load_base_model(model_id: str, *, device: str | torch.device = "cuda"):
    """Return a TransformerLens HookedTransformer for ``model_id``.

    Per training rule 6, all base-model loads go through this helper so the
    revision sha is pinned in one place. Returns the model in eval mode with
    gradients disabled for inference (callers re-enable as needed).
    """
    rev = PINNED_BASE_MODEL_REVISIONS.get(model_id)
    if rev is None:
        raise RuntimeError(
            f"base model {model_id!r} has no pinned revision in "
            f"PINNED_BASE_MODEL_REVISIONS (src/training/loaders.py)."
        )
    from transformer_lens import HookedTransformer

    # transformer_lens does not expose a revision kwarg in from_pretrained;
    # we pre-fetch the weights at the pinned sha so HF's cache resolves to
    # that revision when transformer_lens calls into it.
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, revision=rev)
    model = HookedTransformer.from_pretrained(model_id, device=str(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass(frozen=True)
class LoadedSAE:
    """Uniform handle for a level-0 SAE regardless of upstream format.

    Both W_enc and W_dec are shaped ``(n_latents, d_model)`` with fp16 storage
    on CPU. Row i is the direction for latent i. Per training rule 9, rows
    are unit-normalized by the consumer, not here (so MMCS is consistent with
    raw weights).
    """

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor | None
    b_dec: torch.Tensor | None
    arch: Arch
    source: str
    revision: str
    extra: dict


def _require_revision(source: str) -> str:
    sha = PINNED_REVISIONS.get(source)
    if sha is None:
        raise RuntimeError(
            f"level0_source {source!r} has no pinned revision in "
            f"PINNED_REVISIONS (src/training/loaders.py). Pin a SHA before "
            f"running; silent drift across upstream revisions would break "
            f"reproducibility."
        )
    return sha


def load_level0(
    level0_source: str,
    base_model: str,
    layer: int,
    site: Site,
) -> LoadedSAE:
    """Load a level-0 SAE for the given base model, layer, and site.

    Dispatches on ``level0_source`` as documented at the top of this module.
    Callers must not import SAELens / TransformerLens / HuggingFace APIs
    directly in training code; route through here (training rule 6).
    """
    if level0_source == GEMMA_JUMPRELU_MATRIX_ID:
        return _load_gemma_scope_jumprelu(base_model, layer, site)
    if level0_source == "bartbussmann/BatchTopK":
        return _load_bartbussmann_batchtopk(base_model, layer, site)
    if level0_source == "train_from_scratch":
        return _load_local_batchtopk(base_model, layer, site)
    if level0_source == "flat_sae_on_activations":
        return _load_local_flat_sae(base_model, layer, site)
    if level0_source == "random_gaussian":
        return _synthesize_random_gaussian(base_model, layer, site)
    raise ValueError(
        f"Unknown level0_source {level0_source!r}. Allowed values are "
        f"documented in EXPERIMENTS.yaml field semantics."
    )


def _gemma_scope_sae_id(layer: int, site: Site) -> str:
    if site != "residual":
        raise ValueError(
            f"gemma-scope-2b-pt-res-canonical only ships residual SAEs; got site={site!r}"
        )
    # Canonical width 65k on layer 12 is the matrix default. Other layers are
    # theoretically loadable but not in EXPERIMENTS.yaml.
    return f"layer_{layer}/width_65k/canonical"


def _load_gemma_scope_jumprelu(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """SAELens release ``gemma-scope-2b-pt-res-canonical``."""
    revision = _require_revision(GEMMA_JUMPRELU_MATRIX_ID)
    if base_model != "google/gemma-2-2b":
        raise ValueError(
            f"gemma-scope SAE requires base_model 'google/gemma-2-2b', got {base_model!r}"
        )
    sae_id = _gemma_scope_sae_id(layer, site)

    from sae_lens import SAE  # local import to avoid import cost when not needed

    sae = SAE.from_pretrained(
        release=GEMMA_JUMPRELU_SAELENS_RELEASE,
        sae_id=sae_id,
        device="cpu",
    )
    # SAELens returns W_enc=(d_in, n_latents) and W_dec=(n_latents, d_in).
    # Transpose encoder so both matrices share (n_latents, d_in) shape.
    W_enc = sae.W_enc.detach().to(torch.float16).T.contiguous()  # (n_latents, d_in)
    W_dec = sae.W_dec.detach().to(torch.float16).contiguous()    # (n_latents, d_in)
    b_enc = getattr(sae, "b_enc", None)
    b_enc_t = b_enc.detach().to(torch.float16) if b_enc is not None else None
    b_dec = getattr(sae, "b_dec", None)
    b_dec_t = b_dec.detach().to(torch.float16) if b_dec is not None else None
    extra: dict = {"d_in": int(sae.cfg.d_in), "d_sae": int(sae.cfg.d_sae)}
    threshold = getattr(sae, "threshold", None)
    if threshold is not None:
        extra["threshold"] = threshold.detach().to(torch.float16)
    return LoadedSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc_t,
        b_dec=b_dec_t,
        arch="jumprelu",
        source=GEMMA_JUMPRELU_MATRIX_ID,
        revision=revision,
        extra=extra,
    )


def _load_bartbussmann_batchtopk(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """HuggingFace ``bartbussmann/BatchTopK`` state_dict for GPT-2 Small layer 8 residual.

    As of 2026-04-23 this repo returns 404; author has no public HF models.
    The matrix references it for the Experiment 3 GPT-2 Small anchor rows;
    the correct id needs a human edit to EXPERIMENTS.yaml. Until then this
    function raises with a clear message so the runner marks those rows
    failed and moves on.
    """
    raise FileNotFoundError(
        "level0_source='bartbussmann/BatchTopK' does not resolve: the HF "
        "repo returns 404 and no public model exists under that author. "
        "Fix by editing EXPERIMENTS.yaml to point at the real release id "
        "(Leask et al. 2025 Bussmann BatchTopK GPT-2 Small 49k width). "
        f"[base_model={base_model}, layer={layer}, site={site}]"
    )


def _load_local_batchtopk(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """Locally trained BatchTopK level-0 checkpoint.

    Resolves to the ``l0_*`` artifact for the requested base_model.
    """
    base_to_artifact = {
        "google/gemma-2-2b": "l0_gemma_batchtopk",
        "openai-community/gpt2": "l0_gpt2_batchtopk",
    }
    artifact_id = base_to_artifact.get(base_model)
    if artifact_id is None:
        raise ValueError(
            f"train_from_scratch for base_model={base_model!r} is not defined in the matrix"
        )
    ckpt_path = REPO_ROOT / "experiments" / "artifacts" / artifact_id / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"level-0 BatchTopK checkpoint missing: {ckpt_path}. The "
            f"{artifact_id} row must be `complete` before its descendants can run."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Local checkpoints use the same schema as train_null_sae.py / train_meta_sae.py:
    # W_enc: (d_in, n_latents), W_dec: (n_latents, d_in)
    W_enc = ckpt["W_enc"].to(torch.float16)  # (d_in, n_latents)
    W_enc = W_enc.T.contiguous()             # -> (n_latents, d_in)
    W_dec = ckpt["W_dec"].to(torch.float16).contiguous()  # (n_latents, d_in)
    b_enc = ckpt.get("b_enc")
    b_dec = ckpt.get("b_dec")
    return LoadedSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc.to(torch.float16) if b_enc is not None else None,
        b_dec=b_dec.to(torch.float16) if b_dec is not None else None,
        arch="batchtopk",
        source="train_from_scratch",
        revision="local",
        extra={"experiment_id": ckpt.get("experiment_id", "")},
    )


def _load_local_flat_sae(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """Flat-SAE-on-activations local checkpoint (Experiment 6 comparator)."""
    raise NotImplementedError(
        "flat SAE comparator checkpoints are not produced yet. "
        "train_flat_sae body needs the activation-caching pipeline first."
    )


def _synthesize_random_gaussian(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """Null-baseline: isotropic Gaussian 'decoder directions'.

    The random_gaussian null is generated by the null training entry point
    itself against the matching d_model, not loaded. Routing through here
    keeps callers uniform but raises if something tries to load a null as a
    level-0 SAE.
    """
    raise RuntimeError(
        "random_gaussian is not loaded via load_level0. It is generated in "
        "src/training/train_null_sae.py from d_model and sample count. Callers "
        "should branch on level0_source before reaching the loader."
    )
