"""
make_fig_novelty_windows.py
=============================
Line chart for Figure "fignoveltywindows": accuracy of 4 methods
(harmonic_amp_linear, harmonic_stack_linear, FBCCA, proposed ensemble
[collaborative]) across 4 signal window lengths on Benchmark (LOSO),
corrected ("round" bin selection) windows 1.0/2.0/4.0/5.0s.

Values extracted directly from the result JSONs (not re-run here):
  harmonic_amp_linear / harmonic_stack_linear:
    results/baselines_novelty_{1s,2s,4s}_fixedwindow_round/*_loso_results.json
    results/baselines_novelty_fixedwindow_round/*_loso_results.json  (5.0s)
  FBCCA:
    results/fbcca_{1s,2s,4s}_fixedwindow_round/fbcca_results_*.json
    results/fbcca_fixedwindow_chfix/fbcca_results_20261003_0406.json  (5.0s)
  Proposed ensemble (collaborative):
    results/ensemble_{1s,2s,4s,full}_fixedwindow_round/collaborative_loso_*.json
    (= Table "tab5" collaborative-ensemble column)

Usage:
    python make_fig_novelty_windows.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Window length (s) - x axis is drawn to true value, unevenly spaced
windows = np.array([1.0, 2.0, 4.0, 5.0])
window_labels = ['1.0 s', '2.0 s', '4.0 s', '5.0 s']

# mean, std (%)
data = {
    'Proposed ensemble':     {'mean': [64.51, 82.01, 92.43, 95.20], 'std': [21.06, 17.95, 10.66, 8.44], 'color': '#2a6fb0'},
    'FBCCA':                 {'mean': [66.21, 90.01, 98.08, 98.74], 'std': [21.81, 13.09, 3.49, 2.89],   'color': '#c65d3b'},
    'harmonic_stack_linear': {'mean': [24.52, 56.99, 82.62, 91.77], 'std': [10.34, 19.59, 16.81, 11.18], 'color': '#1f9c78'},
    'harmonic_amp_linear':   {'mean': [13.27, 31.23, 60.61, 72.79], 'std': [6.84, 16.68, 23.59, 22.02],  'color': '#a67f1f'},
}

GRID = '#e1e0d9'
AXIS = '#c3c2b7'
MUTED = '#898781'
PRIMARY_INK = '#0b0b0b'

fig, ax = plt.subplots(figsize=(7.2, 4.6))

# At the 5.0s endpoint FBCCA (98.74), Proposed ensemble (94.52) and
# harmonic_stack_linear (91.77) are all within ~7pp of each other -> space
# their end labels apart vertically (15pt steps) so the three bold direct
# labels don't overlap; harmonic_amp_linear (72.79) is isolated and needs no nudge.
label_y_nudge = {'FBCCA': 12, 'Proposed ensemble': -3, 'harmonic_stack_linear': -18}

for name, d in data.items():
    mean = np.array(d['mean'])
    std = np.array(d['std'])
    ax.plot(windows, mean, marker='o', markersize=5, linewidth=2, color=d['color'], zorder=3)
    ax.fill_between(windows, mean - std, mean + std, color=d['color'], alpha=0.12, linewidth=0, zorder=1)
    # Direct end-of-line label instead of a legend box
    ax.annotate(name, xy=(windows[-1], mean[-1]), xytext=(8, label_y_nudge.get(name, 0)),
                textcoords='offset points', va='center', ha='left',
                fontsize=9.5, color=d['color'], fontweight='bold')

ax.set_xlabel('Signal window length', fontsize=10.5, color=PRIMARY_INK)
ax.set_ylabel('Accuracy (%)', fontsize=10.5, color=PRIMARY_INK)
ax.set_xticks(windows)
ax.set_xticklabels(window_labels, fontsize=9.5)
ax.set_xlim(0.7, 6.6)
ax.set_ylim(0, 105)
ax.tick_params(axis='y', labelsize=9.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(AXIS)
ax.spines['bottom'].set_color(AXIS)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)

plt.tight_layout()
plt.savefig('fig_novelty_windows_en.png', dpi=200, bbox_inches='tight')
print('Saved: fig_novelty_windows_en.png')
