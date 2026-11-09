"""Figure 1: architecture of the multi-view ensemble (vector PDF).

Run:  python make_fig1.py   ->  Figure_1.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

# Colour-blind-safe (Okabe-Ito) branch colours
C_TEMP = "#0072B2"   # blue
C_FREQ = "#009E73"   # bluish green
C_SPAT = "#D55E00"   # vermillion
C_TRAIN = "#6A3D9A"  # purple (training-only)
INK = "#222222"
GREY = "#777777"

fig = plt.figure(figsize=(7.2, 4.75))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.0, ls="-", r=1.2, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       ec=ec, fc=fc, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color=INK, ls="-", lw=1.0, style="-|>", ms=7, conn=None):
    kw = dict(arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
              shrinkA=0, shrinkB=0, zorder=3)
    if conn:
        kw["connectionstyle"] = conn
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **kw))


def light(c, a=0.10):
    rgb = np.array(matplotlib.colors.to_rgb(c))
    return tuple(1 - a * (1 - rgb))


# ---------------------------------------------------------------- input
box(1, 22, 13.5, 15, INK)
ax.text(7.75, 35.3, "EEG trial", ha="center", va="center", fontsize=8, weight="bold")
rng = np.random.default_rng(3)
t = np.linspace(0, 1, 200)
for i in range(4):
    yb = 32.8 - i * 1.25
    sig = 0.45 * np.sin(2 * np.pi * 9 * t + i) + 0.25 * rng.standard_normal(200)
    ax.plot(2.6 + 10.3 * t, yb + 0.5 * sig, lw=0.45, color=GREY)
ax.text(7.75, 25.9, "9 channels × 1250\n(5.0 s, 250 Hz)", ha="center", va="center",
        fontsize=6.6, linespacing=1.15)
ax.text(7.75, 23.1, "z-normalized", ha="center", va="center", fontsize=6.2, color=GREY)

# ---------------------------------------------------------------- branches
BX, BW, BH = 19.5, 45, 14.2
branches = [
    (40.5, C_TEMP, "Temporal CNN", "temporal view (EEGNet-style)",
     ["Temporal conv\n1×25, 8 maps", "Depthwise spatial\nC×1, ×2", "Separable conv\n16 maps, pool"]),
    (22.4, C_FREQ, "SSVEPFreqNet", "frequency view (harmonic stacking)",
     None),
    (4.3, C_SPAT, "Spatial-temporal CNN", "spatial-temporal view (TRCA-inspired)",
     ["Spatial filter\nC×1, 8 virtual ch.", "Depthwise temporal\n1×25", "Conv 1×15\n16 maps, adapt. pool"]),
]

centers = []
for (y, col, name, sub, blocks) in branches:
    box(BX, y, BW, BH, col, fc=light(col, 0.07), lw=1.1)
    ax.text(BX + 1.4, y + BH - 1.9, name, ha="left", va="center", fontsize=8.2,
            weight="bold", color=col)
    ax.text(BX + 1.4, y + BH - 4.0, sub, ha="left", va="center", fontsize=6.4,
            color=GREY, style="italic")
    cy = y + 4.6
    centers.append(y + BH / 2)
    if blocks is not None:
        bw, gap = 13.0, 1.2
        for j, txt in enumerate(blocks):
            bx = BX + 1.4 + j * (bw + gap)
            box(bx, cy - 3.3, bw, 6.6, col, r=0.8, lw=0.8)
            ax.text(bx + bw / 2, cy, txt, ha="center", va="center", fontsize=6.3,
                    linespacing=1.2)
            if j < 2:
                arrow(bx + bw, cy, bx + bw + gap, cy, color=col, ms=5, lw=0.8)

# SSVEPFreqNet internals: FFT -> harmonic stacking tensor -> learned convs
y, col = branches[1][0], C_FREQ
cy = y + 4.6
# FFT block
box(BX + 1.4, cy - 3.3, 7.6, 6.6, col, r=0.8, lw=0.8)
ax.text(BX + 5.2, cy + 0.9, "|FFT|", ha="center", va="center", fontsize=6.6)
ax.text(BX + 5.2, cy - 1.3, "per\nchannel", ha="center", va="center", fontsize=5.8, color=GREY)
arrow(BX + 9.0, cy, BX + 12.6, cy, color=col, ms=5, lw=0.8)
# stacking tensor icon (targets x harmonics), fixed operation
sx, sy, cw, ch = BX + 12.8, cy - 2.4, 0.62, 0.95
vals = rng.random((5, 14)) * 0.5
vals[:, 6] = [1.0, 0.75, 0.55, 0.4, 0.3]
for h in range(5):
    for n in range(14):
        ax.add_patch(Rectangle((sx + n * cw, sy + (4 - h) * ch), cw, ch,
                               fc=matplotlib.colors.to_rgba(col, 0.15 + 0.85 * vals[h, n]),
                               ec="white", lw=0.25, zorder=2))
ax.text(sx + 7 * cw, sy + 5 * ch + 0.35, "harmonic stacking (fixed)", ha="center", va="bottom",
        fontsize=5.6, color=col, weight="bold")
ax.text(sx + 7 * cw, sy - 0.3, r"40 targets × 5 harmonics", ha="center", va="top",
        fontsize=5.4, color=GREY)
ex = sx + 14 * cw
arrow(ex + 0.3, cy, ex + 4.4, cy, color=col, ms=5, lw=0.8)
lx = ex + 4.4
lw_ = BX + BW - 1.4 - lx
box(lx, cy - 3.3, lw_, 6.6, col, r=0.8, lw=0.8)
ax.text(lx + lw_ / 2, cy, "Channel conv 1×1\n→ harmonic conv\n1×H, 16 maps",
        ha="center", va="center", fontsize=6.3, linespacing=1.2)

# input -> branches
fork_x = 16.8
ax.plot([14.5, fork_x], [29.5, 29.5], color=INK, lw=1.0, zorder=3)
ax.plot([fork_x, fork_x], [centers[2], centers[0]], color=INK, lw=1.0, zorder=3)
for c in centers:
    arrow(fork_x, c, BX, c)

# ---------------------------------------------------------------- logits
LX, LW = 68.5, 10.5
for (y, col, *_), c, k in zip(branches, centers, (1, 2, 3)):
    box(LX, c - 3.0, LW, 6.0, col, lw=1.0)
    ax.text(LX + LW / 2, c + 0.9, rf"$\mathrm{{softmax}}(z_{k})$", ha="center",
            va="center", fontsize=7.2)
    ax.text(LX + LW / 2, c - 1.4, "40 classes", ha="center", va="center", fontsize=5.9,
            color=GREY)
    arrow(BX + BW, c, LX, c, color=col)

# ---------------------------------------------------------------- averaging / decision
AX, AW = 83.5, 15.5
box(AX, 21.5, AW, 16.5, INK, fc="#F4F4F4", lw=1.1)
ax.text(AX + AW / 2, 35.4, "Inference", ha="center", va="center", fontsize=8, weight="bold")
ax.text(AX + AW / 2, 30.6, "average the three\nprobability vectors", ha="center",
        va="center", fontsize=6.5, linespacing=1.2)
ax.text(AX + AW / 2, 26.3, r"$\hat{y}=\arg\max_n \ \bar{p}_n$",
        ha="center", va="center", fontsize=7.0)
ax.text(AX + AW / 2, 23.2, "→ target frequency", ha="center", va="center", fontsize=6.5)
merge_x = 81.2
for c in centers:
    ax.plot([LX + LW, merge_x], [c, c], color=INK, lw=1.0, zorder=3)
ax.plot([merge_x, merge_x], [centers[2], centers[0]], color=INK, lw=1.0, zorder=3)
arrow(merge_x, centers[1], AX, centers[1])

# ---------------------------------------------------------------- optional training stage
tx, ty, tw, th = 55.0, 56.0, 44.0, 9.0
box(tx, ty, tw, th, C_TRAIN, fc=light(C_TRAIN, 0.05), ls=(0, (3, 2)), lw=0.9)
ax.text(tx + 1.1, ty + 7.0, "Optional collaborative fine-tuning (training only)",
        ha="left", va="center", fontsize=6.6, weight="bold", color=C_TRAIN)
ax.text(tx + 1.1, ty + 4.3,
        r"soft target $q_k$ = mean of the other two $\mathrm{softmax}(z_j/T)$, stop-gradient",
        ha="left", va="center", fontsize=6.2, color=INK)
ax.text(tx + 1.1, ty + 1.7,
        r"$L^{(k)}=\alpha\,\mathrm{CE}+(1-\alpha)\,T^2\,\mathrm{KL}(q_k\,\|\,\mathrm{softmax}(z_k/T))$,  $\alpha$=0.5, $T$=4",
        ha="left", va="center", fontsize=6.2, color=INK)
arrow(LX + LW / 2, centers[0] + 3.0, LX + LW / 2, ty, color=C_TRAIN, ls=(0, (3, 2)),
      style="<|-|>", ms=6, lw=0.9)

# ---------------------------------------------------------------- legend
ax.plot([20, 25], [1.3, 1.3], color=INK, lw=1.0)
ax.text(25.8, 1.3, "inference path (deployed model)", va="center", fontsize=6.3)
ax.plot([55, 60], [1.3, 1.3], color=C_TRAIN, lw=0.9, ls=(0, (3, 2)))
ax.text(60.8, 1.3, "training only; inference cost unchanged", va="center", fontsize=6.3)

fig.savefig("Figure_1.pdf")
print("wrote Figure_1.pdf")
