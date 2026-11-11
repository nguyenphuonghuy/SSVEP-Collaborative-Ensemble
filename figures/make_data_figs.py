"""Data figures for the BSPC manuscript (Figure 2 = accuracy vs window, Figure S1;
the former four-branch removal figure is kept under _old/).

One style for all three: Arial/Liberation Sans, 7 pt text in a single dark ink,
one accent colour (blue) for the proposed model or its key branch, greys for
everything else.

Run:  python make_data_figs.py [--results-dir PATH]
      -> Figure_2.pdf, Figure_S1.pdf (+ .svg)

Figure S1 reads the per-subject LOSO results; Figures 2 and 3 use the summary
values reported in the manuscript (Fig. 2: four-branch ablation; Fig. 3 / Table 5).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SKILL_SCRIPTS = Path.home() / ".claude/skills/nature-figure/scripts"
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))

MM = 1 / 25.4
INK = "#1a1a1a"
ACCENT = "#2a6fb0"     # proposed ensemble / SSVEPFreqNet (same blue as Fig. 1)
GREY_D = "#4d4d4d"
GREY_M = "#8c8c8c"
GREY_L = "#bdbdbd"
GRID = "#e6e6e6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- Figure 2
def figure_2():
    """Branch ablation of the original four-branch design (Benchmark, 5.0 s)."""
    full = 93.35
    names = ["Temporal CNN", "SSVEPFreqNet", "Spatial-temporal CNN", "ShallowConvNet"]
    acc_without = np.array([92.64, 84.46, 92.96, 95.20])
    delta = acc_without - full                     # -0.71, -8.89, -0.39, +1.85
    colors = [GREY_M, ACCENT, GREY_M, GREY_L]

    fig, ax = plt.subplots(figsize=(90 * MM, 42 * MM), layout="constrained")
    y = np.arange(len(names))[::-1]
    ax.barh(y, delta, height=0.6, color=colors, zorder=3)
    ax.axvline(0, color=INK, lw=0.6, zorder=4)
    for yi, d in zip(y, delta):
        ax.text(d + (0.25 if d >= 0 else -0.25), yi, f"{d:+.2f}".replace("-", "\u2212"), va="center",
                ha="left" if d >= 0 else "right")
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-11, 3.5)
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2])
    ax.set_xlabel("Change in accuracy when the branch is removed (pp)")
    save(fig, "_old/Figure_2_branch_removal_4branch")


# --------------------------------------------------------------------------- Figure 3
def figure_3():
    """Accuracy vs window length (Benchmark, LOSO; mean +/- SD across 35 subjects)."""
    w = np.array([1.0, 2.0, 4.0, 5.0])
    series = [
        # label, mean, sd, colour, marker, linestyle, dodge (s)
        ("FBCCA", [66.21, 90.01, 98.08, 98.74], [21.81, 13.09, 3.49, 2.89],
         INK, "s", "-", -0.09),
        ("Ensemble (proposed)", [64.51, 82.01, 92.43, 95.20], [21.06, 17.95, 10.66, 8.44],
         ACCENT, "o", "-", -0.03),
        ("5-harmonic stack + linear", [24.52, 56.99, 82.62, 91.77], [10.34, 19.59, 16.81, 11.18],
         GREY_M, "^", "--", 0.03),
        ("Fundamental amplitude + linear", [13.27, 31.23, 60.61, 72.79], [6.84, 16.68, 23.59, 22.02],
         GREY_M, "v", ":", 0.09),
    ]
    fig, ax = plt.subplots(figsize=(120 * MM, 68 * MM))
    fig.subplots_adjust(left=0.09, right=0.66, bottom=0.14, top=0.97)
    label_y = {}
    for lab, m, sd, c, mk, ls, dx in series:
        m, sd = np.array(m), np.array(sd)
        ax.errorbar(w + dx, m, yerr=sd, color=c, lw=1.1 if c != GREY_M else 0.9, ls=ls,
                    marker=mk, ms=3.2, mfc=c if mk in "os" else "white", mew=0.8,
                    elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3 if c != GREY_M else 2)
        label_y[lab] = (m[-1], c)
    # direct labels at the right, nudged apart where the 5.0 s means are close
    ypos = {"FBCCA": 101.0, "Ensemble (proposed)": 95.0,
            "5-harmonic stack + linear": 89.0, "Fundamental amplitude + linear": 72.8}
    for lab, (yv, c) in label_y.items():
        ax.annotate(lab, xy=(5.12, yv), xytext=(5.45, ypos[lab]), va="center", ha="left",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GREY_L, lw=0.5, shrinkA=1, shrinkB=2))
    ax.set_xlim(0.7, 5.2)
    ax.set_xticks(w, ["1.0", "2.0", "4.0", "5.0"])
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    save(fig, "Figure_2")


# --------------------------------------------------------------------------- Figure S1
def find_results(results_dir):
    cands = [Path(results_dir)] if results_dir else [HERE.parents[1] / "results", HERE.parent / "results"]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit("results directory not found; pass --results-dir")


def per_subject(results, sub):
    f = sorted(glob.glob(str(results / sub / "collaborative_loso_*.json")))[-1]
    d = json.load(open(f))
    # 'baseline_acc' = probability averaging of the independently trained branches
    return np.array([x["baseline_acc"] for x in d["folds"]]) * 100


def figure_s1(results_dir=None):
    res = find_results(results_dir)
    bm = per_subject(res, "ensemble_full_fixedwindow_round")
    beta = per_subject(res, "ensemble_beta_fixedwindow_round")
    print(f"S1: Benchmark n={bm.size} mean={bm.mean():.2f}  BETA n={beta.size} mean={beta.mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 58 * MM), layout="constrained",
                             gridspec_kw={"width_ratios": [bm.size, beta.size]}, sharey=True)
    for ax, acc, title in zip(axes, (bm, beta),
                              ("a  Benchmark, 5.0 s window (n = 35)", "b  BETA, 2.0 s window (n = 70)")):
        s = np.sort(acc)[::-1]
        x = np.arange(1, s.size + 1)
        ax.bar(x, s, width=0.8, color=ACCENT, zorder=3)
        ax.axhline(acc.mean(), color=INK, lw=0.7, ls="--", zorder=4)
        ax.set_xlim(0.2, s.size + 0.8)
        ax.set_xticks([1] + list(range(10, s.size + 1, 10)))
        ax.set_xlabel("Subject (sorted by accuracy)")
        ax.set_title(f"{title}\nmean {acc.mean():.2f}% (dashed line)", loc="left", fontweight="bold", linespacing=1.4)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=str(HERE / "Figure_S1.alignment.json"),
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print("panel-alignment audit not available; skipped")
    save(fig, "Figure_S1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    figure_2()
    figure_3()
    figure_s1(a.results_dir)
