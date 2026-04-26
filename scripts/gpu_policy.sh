#!/usr/bin/env bash
# scripts/gpu_policy.sh
#
# Translates the EXPERIMENTS.yaml `gpu_preference` field into the actual
# CUDA_VISIBLE_DEVICES value used at dispatch time. Sourced by run_loop.sh.
#
# Why this file exists: EXPERIMENTS.yaml and the project CLAUDE.md GPU
# table are protected paths (CLAUDE.md rule 3). The physical CUDA ordering
# and host contention have shifted since those files were written. This
# file is the single editable point for that translation. Edit the
# SAE_GPU_REMAP entries below as the host layout changes, or export an
# env var (SAE_GPU_REMAP_<yaml>=<actual>) before launching the runner to
# override without editing.
#
# Current host layout (2026-04-26, post-diagnosis):
#   GPU 0  contended — cv-semseg evaluator-watch + P4 bench target this.
#   GPU 1  3090 24 GB — big-model slot.
#   GPU 2  16 GB 4080 — small-model slot.
#   GPU 3  3090 24 GB — big-model slot.
#   GPU 4  16 GB 4060 Ti — small-model slot.
#
# CRITICAL: indices in this file are nvidia-smi PCI_BUS_ID order. Without
# the export below, CUDA defaults to FASTEST_FIRST and swaps GPUs 1<->2
# silently — that bug routed every 24 GB intent to a 16 GB card from
# 2026-04-23 through 2026-04-26 and is the reason l0_gemma_batchtopk
# OOM'd at 15.57 GiB on what the YAML believed was a 3090.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#
# YAML field origins (intent labels, not raw indices):
#   "0" meta-SAE primary (needed 16 GB)
#   "1" meta-SAE parallel seed (needed 16 GB)
#   "2" Gemma level-0 activation cache (needed 24 GB)
#   "3" Llama-3.1-8B autointerp (needed 24 GB)
#   "4" analysis/plots (needed 16 GB)

declare -A SAE_GPU_REMAP=(
    ["0"]="2"       # meta-SAE (16 GB intent)          -> 16 GB non-contended
    ["1"]="4"       # meta-SAE parallel (16 GB intent) -> 16 GB non-contended
    ["2"]="1"       # Gemma activation cache (24 GB)   -> 24 GB 3090
    ["3"]="3"       # autointerp Llama-3.1-8B (24 GB)  -> 24 GB 3090
    ["4"]="4"       # analysis/plots (16 GB)           -> 16 GB (unchanged)
    ["0,1"]="2,4"
    ["2,3"]="1,3"
)

resolve_gpu() {
    local yaml_pref="$1"

    # Guardrail: refuse to dispatch if the parent shell already has
    # CUDA_VISIBLE_DEVICES set. Inline `CUDA_VISIBLE_DEVICES=...` in run_loop.sh
    # would normally override, but historically operators have started the
    # runner under shell envs that broke the remap in subtle ways. Hard fail
    # so misconfiguration surfaces immediately rather than silently routing
    # to the wrong card.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[gpu_policy] FATAL: CUDA_VISIBLE_DEVICES is set in parent shell ('${CUDA_VISIBLE_DEVICES}'); unset it before launching the runner so gpu_policy can pin per-experiment." >&2
        return 1
    fi

    if [[ -z "$yaml_pref" ]]; then
        echo "[gpu_policy] FATAL: row arrived with empty gpu_preference; refusing to launch (was previously masked by SAE_GPU_FALLBACK)." >&2
        return 1
    fi

    # Env override wins over the in-file table.
    local env_key="SAE_GPU_REMAP_$(echo "$yaml_pref" | tr ',' '_')"
    local env_val="${!env_key:-}"
    if [[ -n "$env_val" ]]; then
        echo "[gpu_policy] env override $env_key='$env_val' (yaml '$yaml_pref' bypassed)" >&2
        echo "$env_val"
        return 0
    fi

    local mapped="${SAE_GPU_REMAP[$yaml_pref]:-}"
    if [[ -n "$mapped" ]]; then
        echo "[gpu_policy] yaml '$yaml_pref' -> CUDA '$mapped' (PCI_BUS_ID order)" >&2
        echo "$mapped"
        return 0
    fi

    echo "[gpu_policy] FATAL: unknown gpu_preference '$yaml_pref'; add an entry to SAE_GPU_REMAP or set SAE_GPU_REMAP_${yaml_pref//,/_} env override." >&2
    return 1
}
