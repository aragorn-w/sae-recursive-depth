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
# Current host layout (2026-04-23, per operator):
#   GPU 0  contended — cv-semseg evaluator-watch + P4 bench target this.
#   GPU 1  3090 24 GB — big-model slot.
#   GPU 2  16 GB — small-model slot.
#   GPU 3  3090 24 GB — big-model slot.
#   GPU 4  16 GB — small-model slot.
#
# YAML field origins (from the original CLAUDE.md GPU table, now stale):
#   "0" meta-SAE primary (needed 16 GB)
#   "1" meta-SAE parallel seed (needed 16 GB)
#   "2" Gemma level-0 activation cache (needed 24 GB)
#   "3" Llama-3.1-8B autointerp (needed 24 GB)
#   "4" analysis/plots (needed 16 GB)
#
# So the translation routes each intent to a currently correct slot while
# avoiding the contended GPU 0.

declare -A SAE_GPU_REMAP=(
    ["0"]="2"       # meta-SAE (16 GB intent)          -> 16 GB non-contended
    ["1"]="4"       # meta-SAE parallel (16 GB intent) -> 16 GB non-contended
    ["2"]="1"       # Gemma activation cache (24 GB)   -> 24 GB 3090
    ["3"]="3"       # autointerp Llama-3.1-8B (24 GB)  -> 24 GB 3090
    ["4"]="4"       # analysis/plots (16 GB)           -> 16 GB (unchanged)
    # Hypothetical multi-GPU YAML values, in case rows add them later:
    ["0,1"]="2,4"
    ["2,3"]="1,3"
)

# If a row arrives with no gpu_preference, or a value we don't know, fall
# back to this safe default (a 16 GB non-contended card).
SAE_GPU_FALLBACK="${SAE_GPU_FALLBACK:-4}"

resolve_gpu() {
    local yaml_pref="$1"

    if [[ -z "$yaml_pref" ]]; then
        echo "[gpu_policy] no gpu_preference in row; using fallback '$SAE_GPU_FALLBACK'" >&2
        echo "$SAE_GPU_FALLBACK"
        return
    fi

    # Env override wins over the in-file table.
    local env_key="SAE_GPU_REMAP_$(echo "$yaml_pref" | tr ',' '_')"
    local env_val="${!env_key:-}"
    if [[ -n "$env_val" ]]; then
        echo "[gpu_policy] env remap '$yaml_pref' -> '$env_val'" >&2
        echo "$env_val"
        return
    fi

    local mapped="${SAE_GPU_REMAP[$yaml_pref]:-}"
    if [[ -n "$mapped" ]]; then
        if [[ "$mapped" != "$yaml_pref" ]]; then
            echo "[gpu_policy] remap '$yaml_pref' -> '$mapped'" >&2
        fi
        echo "$mapped"
        return
    fi

    echo "[gpu_policy] no entry for '$yaml_pref'; falling back to '$SAE_GPU_FALLBACK'" >&2
    echo "$SAE_GPU_FALLBACK"
}
