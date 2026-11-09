"""
make_fig2.py
=============
Figure 2 (per-subject accuracy distribution, collaborative ensemble) using
the corrected ("round" bin-selection + fixed 5.0s/1250-sample window)
Benchmark (results/ensemble_full_fixedwindow_round) and BETA
(results/ensemble_beta_fixedwindow_round) result files.

Usage:
    python make_fig2.py
"""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_accs(pattern):
    f = sorted(glob.glob(pattern))[-1]
    d = json.load(open(f))
    return np.array([x['best_acc'] for x in d['folds']]) * 100, f

bm_accs, bm_file = load_accs('../results/ensemble_full_fixedwindow_round/collaborative_loso_*.json')
beta_accs, beta_file = load_accs('../results/ensemble_beta_fixedwindow_round/collaborative_loso_*.json')
print('Benchmark file:', bm_file, '| n=', len(bm_accs), '| mean=%.2f' % bm_accs.mean())
print('BETA file:', beta_file, '| n=', len(beta_accs), '| mean=%.2f' % beta_accs.mean())

bm_sorted = np.sort(bm_accs)[::-1]
beta_sorted = np.sort(beta_accs)[::-1]

# Quiet, muted editorial palette (consistent with make_fig_ablation.py /
# make_fig_novelty_windows.py) instead of default Office blue/orange.
BENCHMARK_C = '#2a6fb0'   # deep blue
BENCHMARK_LINE = '#173f66'
BETA_C = '#c65d3b'        # muted terracotta
BETA_LINE = '#7a3620'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
MUTED = '#898781'
PRIMARY_INK = '#0b0b0b'

def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS)
    ax.spines['bottom'].set_color(AXIS)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)

fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), gridspec_kw={'width_ratios': [3, 3, 1.6]})

ax = axes[0]
ax.bar(np.arange(len(bm_sorted)), bm_sorted, color=BENCHMARK_C, width=0.85, zorder=3)
ax.axhline(bm_accs.mean(), color=BENCHMARK_LINE, linestyle='--', linewidth=1.4, zorder=4)
ax.annotate(f'mean = {bm_accs.mean():.2f}%', xy=(1, bm_accs.mean()), xycoords=('axes fraction', 'data'),
            xytext=(-4, 5), textcoords='offset points', ha='right', va='bottom',
            fontsize=9, color=BENCHMARK_LINE, fontweight='bold')
ax.set_title(f'(a) Benchmark (n={len(bm_accs)} subjects)', fontsize=11.5, color=PRIMARY_INK, pad=8)
ax.set_xlabel('Subject (sorted descending)', fontsize=10, color=PRIMARY_INK)
ax.set_ylabel('Accuracy (%)', fontsize=10, color=PRIMARY_INK)
ax.set_ylim(0, 100)
style_axis(ax)

ax = axes[1]
ax.bar(np.arange(len(beta_sorted)), beta_sorted, color=BETA_C, width=0.85, zorder=3)
ax.axhline(beta_accs.mean(), color=BETA_LINE, linestyle='--', linewidth=1.4, zorder=4)
ax.axhline(40, color=MUTED, linestyle=':', linewidth=1.1, zorder=2)
ax.annotate(f'mean = {beta_accs.mean():.2f}%', xy=(1, beta_accs.mean()), xycoords=('axes fraction', 'data'),
            xytext=(-4, 5), textcoords='offset points', ha='right', va='bottom',
            fontsize=9, color=BETA_LINE, fontweight='bold')
ax.annotate('40% threshold', xy=(1, 40), xycoords=('axes fraction', 'data'),
            xytext=(-4, -11), textcoords='offset points', ha='right', va='top',
            fontsize=8.5, color=MUTED,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.85))
ax.set_title(f'(b) BETA (n={len(beta_accs)} subjects)', fontsize=11.5, color=PRIMARY_INK, pad=8)
ax.set_xlabel('Subject (sorted descending)', fontsize=10, color=PRIMARY_INK)
ax.set_ylabel('Accuracy (%)', fontsize=10, color=PRIMARY_INK)
ax.set_ylim(0, 100)
style_axis(ax)

ax = axes[2]
parts = ax.violinplot([bm_accs, beta_accs], positions=[1, 2], showextrema=False, widths=0.8)
for pc, color in zip(parts['bodies'], [BENCHMARK_C, BETA_C]):
    pc.set_facecolor(color)
    pc.set_alpha(0.30)
    pc.set_edgecolor('none')
bp = ax.boxplot([bm_accs, beta_accs], positions=[1, 2], widths=0.25,
                 patch_artist=True, showfliers=True,
                 medianprops=dict(color=PRIMARY_INK, linewidth=1.5),
                 boxprops=dict(linewidth=0.8),
                 whiskerprops=dict(color=MUTED, linewidth=0.9),
                 capprops=dict(color=MUTED, linewidth=0.9),
                 flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor=MUTED, markersize=4))
for patch, color in zip(bp['boxes'], [BENCHMARK_C, BETA_C]):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
    patch.set_edgecolor(color)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Benchmark', 'BETA'], fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=10, color=PRIMARY_INK)
ax.set_title('(c) Distribution', fontsize=11.5, color=PRIMARY_INK, pad=8)
ax.set_ylim(0, 100)
ax.text(1, 5, f'σ={bm_accs.std():.1f}', ha='center', fontsize=9, color=BENCHMARK_LINE, fontweight='bold')
ax.text(2, 5, f'σ={beta_accs.std():.1f}', ha='center', fontsize=9, color=BETA_LINE, fontweight='bold')
style_axis(ax)
ax.spines['bottom'].set_visible(True)

plt.tight_layout()
plt.savefig('fig2_en.png', dpi=150, bbox_inches='tight')
print('Saved: fig2_en.png')
