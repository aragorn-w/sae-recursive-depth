"""Uniform loader for level-0 SAEs and their base models.

Every training entry point that needs a level-0 SAE goes through
`load_level0(...)`. Per training rule 6, no module under `src/training/` may
load an SAE or base model by HuggingFace id directly; all such loads route
here so revisions are pinned in one place and the return type is uniform
across SAELens, TransformerLens, and the bartbussmann/BatchTopK release.

Routing:
  level0_source                                 arch         handler
  ------------------------------------------    ----------   --------------------
  "google/gemma-scope-2b-pt-res-canonical"      jumprelu     SAELens release
  "bartbussmann/BatchTopK"                      batchtopk    HF state_dict
  "train_from_scratch"                          batchtopk    local checkpoint
  "flat_sae_on_activations"                     none         local checkpoint
  "random_gaussian"                             none         synthesized in memory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Arch = Literal["jumprelu", "batchtopk", "none"]
Site = Literal["residual", "mlp_out", "attn_out"]


# Revision SHAs for every external artifact the loader may fetch. Pinning lives
# here (not in callers) so a single edit changes every downstream run. These
# are filled in by the human on first successful fetch; the loader refuses to
# load an unpinned source to prevent silent drift.
PINNED_REVISIONS: dict[str, str] = {
    # "google/gemma-scope-2b-pt-res-canonical": "<sha>",
    # "bartbussmann/BatchTopK": "<sha>",
}


@dataclass(frozen=True)
class LoadedSAE:
    """Uniform handle for a level-0 SAE regardless of upstream format.

    W_enc: (n_latents, d_model) encoder weight, fp16 on CPU.
    W_dec: (n_latents, d_model) decoder weight, fp16 on CPU. Row i is the
        decoder direction for latent i. Per training rule 9, rows are unit-
        normalized before being fed to a meta-SAE; normalization happens at
        the consumer, not here.
    b_enc: (n_latents,) encoder bias, fp16 on CPU, or None.
    b_dec: (d_model,) decoder bias, fp16 on CPU, or None.
    arch: "jumprelu" | "batchtopk" | "none".
    source: the level0_source string from EXPERIMENTS.yaml.
    revision: pinned SHA from PINNED_REVISIONS, or "local" for checkpoints.
    extra: arch-specific fields (e.g. {"k": 60} for BatchTopK,
        {"threshold": Tensor} for JumpReLU).
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
            f"PINNED_REVISIONS. Pin a SHA in src/training/loaders.py before "
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

    Dispatches on `level0_source` as documented at the top of this module.
    Callers must not import SAELens / TransformerLens / HuggingFace APIs
    directly in training code; route through here (training rule 6).
    """
    if level0_source == "google/gemma-scope-2b-pt-res-canonical":
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


def _load_gemma_scope_jumprelu(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """SAELens release `gemma-scope-2b-pt-res-canonical`, layer `layer`, width 65k."""
    revision = _require_revision("google/gemma-scope-2b-pt-res-canonical")
    raise NotImplementedError(
        "Wire up SAELens load: "
        "`SAE.from_pretrained('gemma-scope-2b-pt-res-canonical', "
        "sae_id=f'layer_{layer}/width_65k/canonical', device='cpu')`, "
        f"then wrap in LoadedSAE with revision={revision!r}. "
        "base_model, site are validated here before return."
    )


def _load_bartbussmann_batchtopk(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """HuggingFace `bartbussmann/BatchTopK` state_dict for GPT-2 Small layer 8 residual."""
    revision = _require_revision("bartbussmann/BatchTopK")
    raise NotImplementedError(
        "Wire up HF hub download of BatchTopK state_dict "
        f"(revision={revision!r}), then wrap in LoadedSAE with arch='batchtopk' "
        "and extra={'k': <from state_dict>}. Note: this release targets GPT-2 "
        "Small layer 8 residual; assert (base_model, layer, site) match."
    )


def _load_local_batchtopk(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """Locally trained BatchTopK level-0 checkpoint.

    Reads from experiments/artifacts/l0_<base_model_tag>_batchtopk/checkpoint.pt.
    """
    raise NotImplementedError(
        "Wire up torch.load of the local train_from_scratch checkpoint "
        "once src/training/train_level0_batchtopk.py exists. Expected path: "
        "experiments/artifacts/l0_<tag>_batchtopk/checkpoint.pt. "
        "revision='local' in the LoadedSAE."
    )


def _load_local_flat_sae(base_model: str, layer: int, site: Site) -> LoadedSAE:
    """Flat-SAE-on-activations local checkpoint (Experiment 6 comparator)."""
    raise NotImplementedError(
        "Wire up torch.load for flat SAE checkpoint once "
        "src/training/train_flat_sae.py exists. revision='local'."
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
