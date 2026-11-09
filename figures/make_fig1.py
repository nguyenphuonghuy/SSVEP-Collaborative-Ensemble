"""Figure 1: architecture of the multi-view ensemble.

Hand-built vector schematic (SVG, 1 unit = 1 pt, 183 mm double column).
Group titles are 8 pt by choice (Elsevier artwork range); all other text 6-7 pt.
Run:  python make_fig1.py   ->  Figure_1.svg, Figure_1.pdf (via rsvg-convert)

Element ids follow the figure brief:
  n:input, c:temporal / c:freq / c:spatial (branch containers), n:<branch>-<k>
  (blocks), n:p1..n:p3 (branch probabilities), n:average, c:finetune, e:* edges.
"""
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
FAMILY = "Arial, 'Liberation Sans', Helvetica, sans-serif"

# three type sizes only
S_GROUP, S_TEXT, S_SMALL = 8.0, 7.0, 6.0

INK = "#1a1a1a"
LINE = "#4d4d4d"       # outlines of generic blocks and connectors
FRAME = "#8c8c8c"      # outline-only containers
ACCENT = "#2a6fb0"     # the paper's new component (SSVEPFreqNet), same blue as Fig. 2
ACCENT_FILL = "#e8f0f8"

W, H = 518.7, 220.0    # 183 mm wide (double column)

out = []


def add(s):
    out.append(s)


def rect(x, y, w, h, stroke=LINE, fill="#ffffff", sw=0.8, rx=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=S_TEXT, weight="regular", anchor="middle", fill=INK, raw=False):
    body = s if raw else escape(s)
    fw = ' font-weight="bold"' if weight == "bold" else ""
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}"{fw} text-anchor="{anchor}" '
        f'fill="{fill}">{body}</text>')


def lines_in_box(cx, cy, rows):
    """Centre a list of (string, size) rows vertically around cy."""
    lh = [r[1] * 1.22 for r in rows]
    y = cy - sum(lh) / 2
    for (s, sz), h in zip(rows, lh):
        y += h
        text(cx, y - 0.28 * sz, s, size=sz)


def line(x0, y0, x1, y1, color=LINE, sw=0.9, arrow=False):
    m = ' marker-end="url(#ah)"' if arrow else ""
    add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" '
        f'stroke-width="{sw}"{m}/>')


def step(x, y, n):
    add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.6" fill="#ffffff" stroke="{INK}" stroke-width="0.6"/>')
    text(x, y + 2.1, str(n), size=S_SMALL)


# ------------------------------------------------------------------ geometry
IN_X, IN_W = 4.0, 66.0
FORK_X = 78.0
CX, CW = 86.0, 258.0               # branch containers
PAD = 7.0
ROW_H, ROW_GAP, TOP = 58.0, 12.0, 12.0
rows_y = [TOP + i * (ROW_H + ROW_GAP) for i in range(3)]
row_c = [y + ROW_H / 2 + 3.0 for y in rows_y]   # centre line of the block row
P_X, P_W, P_H = 364.0, 56.0, 26.0  # probability boxes
MERGE_X = 432.0
A_X, A_W = 442.0, W - 442.0 - 4.0  # averaging / decision
mid = row_c[1]

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
    f'viewBox="0 0 {W} {H}" font-family="{FAMILY}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5.5" '
    f'markerHeight="5.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>'
    '</marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

# ------------------------------------------------------------------ input
add('<g id="n:input">')
rect(IN_X, mid - 30, IN_W, 60)
lines_in_box(IN_X + IN_W / 2, mid, [("EEG trial", S_GROUP), ("9 channels × 1250", S_TEXT),
                                     ("samples (5.0 s,", S_TEXT), ("250 Hz)", S_TEXT)])
add('</g>')
add('<g id="e:input-branches">')
line(IN_X + IN_W, mid, FORK_X, mid)
line(FORK_X, row_c[0], FORK_X, row_c[2])
add(f'<circle cx="{FORK_X}" cy="{mid}" r="1.4" fill="{LINE}"/>')
for c in row_c:
    line(FORK_X, c, CX, c, arrow=True)
add('</g>')

# ------------------------------------------------------------------ branches
branches = [
    ("c:temporal", "Temporal CNN", False,
     [["Temporal conv", "1×25, 8 maps"], ["Depthwise spatial", "C×1, ×2"],
      ["Separable conv", "16 maps, pooling"]]),
    ("c:freq", "SSVEPFreqNet", True,
     [["|FFT|", "per channel"], ["Harmonic", "stacking, H = 5"],
      ["Channel conv", "1×1, 8 maps"], ["Harmonic conv", "1×H, 16 maps"]]),
    ("c:spatial", "Spatial-temporal CNN", False,
     [["Spatial filter", "C×1, 8 virtual ch."], ["Depthwise temporal", "1×25"],
      ["Conv 1×15", "16 maps, pooling"]]),
]
BLK_H, GAP = 26.0, 9.0
for (cid, title, accent, blocks), y, c in zip(branches, rows_y, row_c):
    add(f'<g id="{cid}">')
    rect(CX, y, CW, ROW_H, stroke=ACCENT if accent else FRAME,
         fill=ACCENT_FILL if accent else "#ffffff", sw=1.1 if accent else 0.8, rx=3)
    text(CX + PAD, y + 11.5, title, size=S_GROUP, weight="bold", anchor="start")
    n = len(blocks)
    bw = (CW - 2 * PAD - (n - 1) * GAP) / n
    for k, (a, b) in enumerate(blocks):
        bx = CX + PAD + k * (bw + GAP)
        add(f'<g id="n:{cid[2:]}-{k + 1}">')
        rect(bx, c - BLK_H / 2, bw, BLK_H, stroke=ACCENT if accent else LINE)
        lines_in_box(bx + bw / 2, c, [(a, S_TEXT), (b, S_SMALL)])
        add('</g>')
        if k < n - 1:
            line(bx + bw, c, bx + bw + GAP, c, arrow=True)
    if accent:
        # brackets: fixed operations vs learned layers
        yb = c + BLK_H / 2 + 3.5
        spans = [(CX + PAD, CX + PAD + 2 * bw + GAP, "fixed (known stimulation frequencies)"),
                 (CX + PAD + 2 * (bw + GAP), CX + CW - PAD, "learned")]
        for x0, x1, lab in spans:
            add(f'<path d="M{x0:.2f},{yb - 1.5:.2f} V{yb:.2f} H{x1:.2f} V{yb - 1.5:.2f}" '
                f'fill="none" stroke="{ACCENT}" stroke-width="0.6"/>')
            text((x0 + x1) / 2, yb + 7.0, lab, size=S_SMALL)
    add('</g>')

# ------------------------------------------------------------------ probabilities
for k, c in enumerate(row_c, start=1):
    line(CX + CW, c, P_X, c, arrow=True)
    add(f'<g id="n:p{k}">')
    rect(P_X, c - P_H / 2, P_W, P_H)
    text(P_X + P_W / 2, c - 1.0,
         f'softmax(<tspan font-style="italic">z</tspan><tspan font-size="{S_SMALL}" dy="1.6">{k}</tspan>'
         f'<tspan dy="-1.6">)</tspan>', raw=True)
    text(P_X + P_W / 2, c + 8.0, "40 classes", size=S_SMALL)
    add('</g>')

# ------------------------------------------------------------------ averaging / decision
add('<g id="e:p-average">')
for c in row_c:
    line(P_X + P_W, c, MERGE_X, c)
line(MERGE_X, row_c[0], MERGE_X, row_c[2])
add(f'<circle cx="{MERGE_X}" cy="{mid}" r="1.4" fill="{LINE}"/>')
line(MERGE_X, mid, A_X, mid, arrow=True)
add('</g>')
add('<g id="n:average">')
rect(A_X, mid - 30, A_W, 60)
lines_in_box(A_X + A_W / 2, mid, [("Average of the three", S_TEXT), ("probability vectors", S_TEXT),
                                   ("→ argmax", S_TEXT), ("→ target frequency", S_TEXT)])
add('</g>')

# ------------------------------------------------------------------ optional fine-tuning (dashed = training only)
fx0, fx1 = P_X - 6.0, P_X + P_W + 6.0
fy0, fy1 = row_c[0] - P_H / 2 - 6.0, row_c[2] + P_H / 2 + 6.0
add('<g id="c:finetune">')
rect(fx0, fy0, fx1 - fx0, fy1 - fy0, stroke=INK, fill="none", sw=0.7, rx=3, dash="3,2")
text((fx0 + fx1) / 2, fy0 - 12.5, "Optional collaborative", size=S_SMALL)
text((fx0 + fx1) / 2, fy0 - 5.0, "fine-tuning (training only)", size=S_SMALL)
add('</g>')

# ------------------------------------------------------------------ step numbers (main path)
step(IN_X + IN_W / 2, mid - 37.0, 1)
step((FORK_X + CX) / 2 + 1.0, row_c[0] - 10.0, 2)
step(P_X + P_W / 2, fy1 + 9.0, 3)
step(A_X + A_W / 2, mid - 37.0, 4)

add('</svg>')

svg = HERE / "Figure_1.svg"
svg.write_text("\n".join(out), encoding="utf-8")
print("wrote", svg.name)
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-d", "72", "-p", "72", "-f", "pdf", "-o", str(HERE / "Figure_1.pdf"), str(svg)], check=True)
    print("wrote Figure_1.pdf")
