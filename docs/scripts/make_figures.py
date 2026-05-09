"""
Figure-generation script for the recursive-SAE-decomposition paper.

Run: `python make_figures.py` from any directory; outputs are written next
to this script under ``docs/figures/`` (edit OUTDIR below for a different
location).

The constants below are the per-seed and per-cell values consumed by the
paper's figures. They are derived from ``data/results.tsv``,
``data/pwmcc_posthoc.tsv``, and ``data/artifacts/_summary/tables/cell_summary.tsv``
as of 2026-05-09. Per-cell partitioning (main-recipe vs anchor-recipe vs
flat-vs-meta separation) was performed during the audit pass
against ``data/results.tsv``; rerunning that aggregation is the canonical
way to refresh these constants.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

GEMMA_BATCHTOPK_D1_MAIN = [0.122121, 0.125671, 0.130207]
GEMMA_BATCHTOPK_D1_ANCHOR = [0.142801, 0.153879, 0.157656]
GEMMA_BATCHTOPK_D1_LIVE = [0.205521, 0.207098, 0.208129]
GEMMA_BATCHTOPK_D1_RATIO_HALF = [0.116933]
GEMMA_BATCHTOPK_D1_RATIO_EIGHTH = [0.132006]
GEMMA_BATCHTOPK_D2_META = [-0.0111433, -0.0040139, 0.00254589]
GEMMA_BATCHTOPK_D3_META = [-0.0130301, -0.00702953, -0.00737667]

GEMMA_JUMPRELU_D1 = [0.191858]
GEMMA_JUMPRELU_D2 = [-0.00159755]
GEMMA_JUMPRELU_D3 = [-0.024678]

GPT2_BATCHTOPK_D1_MAIN = [0.114, 0.114, 0.114]
GPT2_BATCHTOPK_D1_ANCHOR = [0.124, 0.124, 0.124]
GPT2_BATCHTOPK_D2 = [0.048405]
GPT2_BATCHTOPK_D3 = [0.0220816]

L0_GEMMA_BATCHTOPK_VE = 0.742599
L0_GPT2_BATCHTOPK_VE = 0.950777
L0_GPT2_BATCHTOPK_LEASK_VE = 0.988033

NULL_GEMMA_D1_VE = 0.00448267
NULL_GEMMA_D2_VE = 0.0172015
NULL_GEMMA_D3_VE = -7.01348e-06
NULL_GEMMA_D1_VE_STD = 0.00035497
NULL_GEMMA_D2_VE_STD = 0.000834786
NULL_GEMMA_D3_VE_STD = 3.80623e-05

NULL_GPT2_D1_VE = 0.015851
NULL_GPT2_D2_VE = 0.0230906
NULL_GPT2_D3_VE = 0.0284098
NULL_GPT2_D1_VE_STD = 0.00132631
NULL_GPT2_D2_VE_STD = 0.00162843
NULL_GPT2_D3_VE_STD = 8.75787e-05

GEMMA_BATCHTOPK_D1_LIVE_PWMCC = 0.717175
GEMMA_JUMPRELU_D1_PWMCC = 0.725031
GPT2_BATCHTOPK_D1_PWMCC_FLAGGED = 0.724118
FLAT_GEMMA_W16384_PWMCC_JOINT = [0.804788, 0.801103, 0.804971]

FLAT_GEMMA_W16384_VE = [0.719891, 0.723404, 0.719076]
FLAT_GEMMA_W4096_VE = [0.684019, 0.688604, 0.682887]
FLAT_GEMMA_W1024_VE = [0.627885, 0.632998, 0.624026]
FLAT_GPT2_W12288_VE_MEAN = 0.926
FLAT_GEMMA_W4096_PWMCC = [0.902026, 0.8946, 0.902045]
FLAT_GEMMA_W1024_PWMCC = [0.905053, 0.896026, 0.903883]

GEMMA_BATCHTOPK_DEAD = {0: 0.690369, 1: np.mean([0.420044]), 2: 0.194865, 3: 0.137207}
GEMMA_JUMPRELU_DEAD = {1: 0.488648, 2: 0.472249, 3: 0.357747}
GPT2_BATCHTOPK_DEAD = {0: 0.210979, 1: 0.213972, 2: 0.260634, 3: 0.234375}
NULL_GEMMA_DEAD = {1: 0.813883, 2: 0.958822, 3: 0.771484}
NULL_GPT2_DEAD = {1: 0.598796, 2: 0.310221, 3: 0.00824649}

def mean_std(xs):
    a = np.array(xs, dtype=float)
    if len(a) == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

C_GEMMA_BATCHTOPK = '#1f77b4'
C_GEMMA_JUMPRELU = '#ff7f0e'
C_GPT2_BATCHTOPK = '#2ca02c'
C_NULL = '#888888'
C_FLAT = '#d62728'

def fig1_variance_explained_by_depth():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    depths = [0, 1, 2, 3]

    gemma_bt_means = []
    gemma_bt_stds = []
    main_d1_mean, main_d1_std = mean_std(GEMMA_BATCHTOPK_D1_MAIN)
    d2_mean, d2_std = mean_std(GEMMA_BATCHTOPK_D2_META)
    d3_mean, d3_std = mean_std(GEMMA_BATCHTOPK_D3_META)
    gemma_bt_means = [L0_GEMMA_BATCHTOPK_VE, main_d1_mean, d2_mean, d3_mean]
    gemma_bt_stds = [0.0, main_d1_std, d2_std, d3_std]

    gemma_jr_means = [np.nan, GEMMA_JUMPRELU_D1[0], GEMMA_JUMPRELU_D2[0], GEMMA_JUMPRELU_D3[0]]
    gemma_jr_stds = [0.0, 0.00988176, 0.00365819, 0.00548763]

    gpt2_bt_means = [L0_GPT2_BATCHTOPK_VE, np.mean(GPT2_BATCHTOPK_D1_MAIN), GPT2_BATCHTOPK_D2[0], GPT2_BATCHTOPK_D3[0]]
    gpt2_bt_stds = [0.0, 0.004, 0.00342447, 0.00230416]

    null_gemma = [np.nan, NULL_GEMMA_D1_VE, NULL_GEMMA_D2_VE, NULL_GEMMA_D3_VE]
    null_gpt2 = [np.nan, NULL_GPT2_D1_VE, NULL_GPT2_D2_VE, NULL_GPT2_D3_VE]

    ax.errorbar(depths, gemma_bt_means, yerr=gemma_bt_stds, marker='o', linewidth=1.8, markersize=6.5, capsize=3, color=C_GEMMA_BATCHTOPK, label='Gemma-2-2B / BatchTopK (meta-SAE)')
    ax.errorbar(depths, gemma_jr_means, yerr=gemma_jr_stds, marker='s', linewidth=1.8, markersize=6.5, capsize=3, color=C_GEMMA_JUMPRELU, label='Gemma-2-2B / JumpReLU (meta-SAE)')
    ax.errorbar(depths, gpt2_bt_means, yerr=gpt2_bt_stds, marker='^', linewidth=1.8, markersize=6.5, capsize=3, color=C_GPT2_BATCHTOPK, label='GPT-2 Small / BatchTopK (meta-SAE)')

    ax.plot([1, 2, 3], null_gemma[1:], linestyle='--', linewidth=1.2, color=C_NULL, alpha=0.8, label='Isotropic Gaussian null (Gemma)')
    ax.plot([1, 2, 3], null_gpt2[1:], linestyle=':', linewidth=1.2, color=C_NULL, alpha=0.8, label='Isotropic Gaussian null (GPT-2)')

    flat_d1_x = 1.0
    flat_d2_x = 2.0
    flat_d3_x = 3.0
    flat_means = [np.mean(FLAT_GEMMA_W16384_VE), np.mean(FLAT_GEMMA_W4096_VE), np.mean(FLAT_GEMMA_W1024_VE)]
    flat_stds = [np.std(FLAT_GEMMA_W16384_VE, ddof=1), np.std(FLAT_GEMMA_W4096_VE, ddof=1), np.std(FLAT_GEMMA_W1024_VE, ddof=1)]
    ax.errorbar([flat_d1_x, flat_d2_x, flat_d3_x], flat_means, yerr=flat_stds, marker='D', markersize=6.5, capsize=3, linestyle='none', color=C_FLAT, label='Flat-SAE control on activations (Gemma)', zorder=10)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Variance explained')
    ax.set_xticks(depths)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', framealpha=0.95)
    ax.set_title('Variance explained by depth, audit-corrected (mean ± std across seeds)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig1_variance_explained_by_depth.pdf'))
    plt.close(fig)

def fig2_pwmcc_by_depth():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    depths_d1 = [1]
    depths_d23 = [2, 3]

    ax.scatter([1], [GEMMA_BATCHTOPK_D1_LIVE_PWMCC], marker='o', s=70, color=C_GEMMA_BATCHTOPK, label='Gemma-2-2B / BatchTopK (live d1, n=3)', zorder=5)
    ax.scatter([1], [GEMMA_JUMPRELU_D1_PWMCC], marker='s', s=70, color=C_GEMMA_JUMPRELU, label='Gemma-2-2B / JumpReLU (d1, n=5)', zorder=5)
    ax.scatter([1], [GPT2_BATCHTOPK_D1_PWMCC_FLAGGED], marker='^', s=70, color=C_GPT2_BATCHTOPK, label='GPT-2 Small / BatchTopK (d1, n=6, meta-SAE only)', zorder=5)

    ax.scatter(depths_d23, [0, 0], marker='o', s=70, color=C_GEMMA_BATCHTOPK, zorder=5)
    ax.scatter(depths_d23, [0, 0], marker='s', s=70, color=C_GEMMA_JUMPRELU, zorder=5)
    ax.scatter(depths_d23, [0, 0], marker='^', s=70, color=C_GPT2_BATCHTOPK, zorder=5)

    ax.plot([1, 2, 3], [GEMMA_BATCHTOPK_D1_LIVE_PWMCC, 0, 0], linestyle='-', linewidth=1.5, color=C_GEMMA_BATCHTOPK, alpha=0.7)
    ax.plot([1, 2, 3], [GEMMA_JUMPRELU_D1_PWMCC, 0, 0], linestyle='-', linewidth=1.5, color=C_GEMMA_JUMPRELU, alpha=0.7)
    ax.plot([1, 2, 3], [GPT2_BATCHTOPK_D1_PWMCC_FLAGGED, 0, 0], linestyle='-', linewidth=1.5, color=C_GPT2_BATCHTOPK, alpha=0.7)

    ax.axhline(0, color=C_NULL, linewidth=1.0, linestyle='--', label='Isotropic Gaussian null (PW-MCC = 0)')

    flat_d1_pwmcc = np.mean(FLAT_GEMMA_W16384_PWMCC_JOINT)
    flat_d2_pwmcc = np.mean(FLAT_GEMMA_W4096_PWMCC)
    flat_d3_pwmcc = np.mean(FLAT_GEMMA_W1024_PWMCC)
    ax.scatter([1, 2, 3], [flat_d1_pwmcc, flat_d2_pwmcc, flat_d3_pwmcc], marker='D', s=80, color=C_FLAT, label='Flat-SAE control PW-MCC, joint Hungarian (Gemma)', zorder=6)

    ax.set_xlabel('Depth')
    ax.set_ylabel('Cross-seed PW-MCC')
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', framealpha=0.95)
    ax.set_title('Cross-seed PW-MCC by depth, audit-corrected')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig2_pwmcc_by_depth.pdf'))
    plt.close(fig)

def fig3_dead_latent_fraction_by_depth():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.plot([0, 1, 2, 3], [GEMMA_BATCHTOPK_DEAD[0], GEMMA_BATCHTOPK_DEAD[1], GEMMA_BATCHTOPK_DEAD[2], GEMMA_BATCHTOPK_DEAD[3]], marker='o', markersize=6.5, linewidth=1.8, color=C_GEMMA_BATCHTOPK, label='Gemma-2-2B / BatchTopK')
    ax.plot([1, 2, 3], [GEMMA_JUMPRELU_DEAD[1], GEMMA_JUMPRELU_DEAD[2], GEMMA_JUMPRELU_DEAD[3]], marker='s', markersize=6.5, linewidth=1.8, color=C_GEMMA_JUMPRELU, label='Gemma-2-2B / JumpReLU')
    ax.plot([0, 1, 2, 3], [GPT2_BATCHTOPK_DEAD[0], GPT2_BATCHTOPK_DEAD[1], GPT2_BATCHTOPK_DEAD[2], GPT2_BATCHTOPK_DEAD[3]], marker='^', markersize=6.5, linewidth=1.8, color=C_GPT2_BATCHTOPK, label='GPT-2 Small / BatchTopK')

    ax.plot([1, 2, 3], [NULL_GEMMA_DEAD[1], NULL_GEMMA_DEAD[2], NULL_GEMMA_DEAD[3]], marker='x', markersize=7, linewidth=1.2, linestyle='--', color=C_NULL, alpha=0.8, label='Null baseline (Gemma)')
    ax.plot([1, 2, 3], [NULL_GPT2_DEAD[1], NULL_GPT2_DEAD[2], NULL_GPT2_DEAD[3]], marker='+', markersize=8, linewidth=1.2, linestyle=':', color=C_NULL, alpha=0.8, label='Null baseline (GPT-2)')

    ax.set_xlabel('Depth')
    ax.set_ylabel('Dead-latent fraction')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_title('Dead-latent fraction by depth')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig3_dead_latent_fraction_by_depth.pdf'))
    plt.close(fig)

fig1_variance_explained_by_depth()
fig2_pwmcc_by_depth()
fig3_dead_latent_fraction_by_depth()
print(f'OK; figures written to {OUTDIR}')
