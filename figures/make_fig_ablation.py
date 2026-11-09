"""
make_fig_ablation.py
======================
Bar chart for Figure "figablation": ensemble accuracy when each branch is
removed in turn, analysis on the original four-branch set (Benchmark,
corrected 5.0s/1250-sample window, "round" bin selection), from
results/ablation_full_fixedwindow_round/ablation_results_20261003_1343.json
(ablation_A_extractor_subset: all_4, drop_ext1..drop_ext4).

Usage:
    python make_fig_ablation.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = ['Full\n(4 branches)', 'Remove\ntemporal', 'Remove\nfrequency', 'Remove\nspatial-temporal', 'Remove\nShallowConvNet']
acc = [93.35, 92.64, 84.46, 92.96, 95.20]
delta = [None, -0.71, -8.89, -0.39, +1.85]

BASELINE_COLOR = '#a88535'   # muted gold, removal decreases accuracy
CRITICAL = '#b23a3a'         # removal decreases accuracy
GOOD = '#2f6fa8'             # removal increases accuracy (blue, not green:
                             # red/green would collide under deuteranopia,
                             # confirmed by scripts/validate_palette.js
                             # -- ΔE 5.1 deutan, below the 6 floor -- this
                             # blue/red pair passes all checks)
colors = [BASELINE_COLOR, CRITICAL, CRITICAL, CRITICAL, GOOD]

GRID = '#e1e0d9'
AXIS = '#c3c2b7'
MUTED = '#898781'
PRIMARY_INK = '#0b0b0b'

fig, ax = plt.subplots(figsize=(8.2, 4.6))

x = np.arange(len(labels))
bars = ax.bar(x, acc, color=colors, width=0.6, zorder=3)

baseline = acc[0]
ax.axhline(baseline, color=BASELINE_COLOR, linestyle='--', linewidth=1.2, zorder=2)

for i, (b, a, d) in enumerate(zip(bars, acc, delta)):
    label = f'{a:.2f}%' if d is None else f'{a:.2f}%\n({d:+.2f})'
    ax.annotate(label, xy=(b.get_x() + b.get_width() / 2, a), xytext=(0, 6),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=9, color=PRIMARY_INK, fontweight='bold', zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

ax.set_ylabel('Accuracy (%)', fontsize=10.5, color=PRIMARY_INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 100)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(AXIS)
ax.spines['bottom'].set_color(AXIS)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)

plt.tight_layout()
plt.savefig('fig_ablation_en.png', dpi=200, bbox_inches='tight')
print('Saved: fig_ablation_en.png')
