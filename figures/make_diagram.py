#!/usr/bin/env python
"""Standalone raster of the project page's explainer diagram (score-based eviction vs
Random Attention), laid out for social posts: the two panels and nothing else — no title,
no result strip, no links, since the thread around it carries those.

    python figures/make_diagram.py          # -> assets/random_attention_diagram.png (+ .pdf)

Keep-sets and scores match the inline SVG in index.html, so the two never disagree.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, INK, MUTED = "#FBFAF7", "#1B2730", "#5E6B76"
BLUE, RED, GHOST, LINE = "#275A7A", "#C7423F", "#C9CFD4", "#DDD9CE"

W, H = 30, 30          # cell size
GAP = 4
ROWS = [188, 232, 276]  # three KV heads
NCELL, NPROMPT = 14, 4  # 1 sink + 3 prompt, then 10 trace tokens

# left panel: per-head scores for the 13 cells that compete (top 8 survive)
LEFT = [
    dict(p=[.6, .2, .1], g=[.8, .3, .7, .5, .2, .9, .4, .6, .1, .7]),
    dict(p=[.3, .5, .2], g=[.4, .7, .2, .8, .6, .1, .9, .3, .5, .6]),
    dict(p=[.1, .4, .6], g=[.7, .5, .3, .2, .8, .6, .3, .9, .2, .5]),
]
# right panel: which trace tokens each head's random draw kept (same budget, 9 cells)
RIGHT = [{0, 2, 3, 6, 9}, {1, 2, 5, 7, 8}, {0, 4, 5, 6, 8}]

fig = plt.figure(figsize=(11.88, 3.62), dpi=200, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(-6, 1182)
ax.set_ylim(422, 60)          # y grows downwards, like the SVG
ax.axis("off")


def cell(x, y, *, fill=None, alpha=1.0, label=None, struck=False):
    if fill:
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=5",
                                    linewidth=0, facecolor=fill, alpha=alpha, zorder=2))
    else:
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=5",
                                    linewidth=1.1, edgecolor=GHOST, facecolor="none",
                                    linestyle=(0, (3.5, 2.5)), zorder=2))
    if label is not None:
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", zorder=4,
                fontsize=7.4, fontweight="bold" if fill else "normal",
                color=BG if fill else MUTED)
    if struck:
        ax.plot([x + 6, x + W - 6], [y + H - 6, y + 6], color=MUTED, lw=.9, alpha=.75, zorder=5)


def bracket(x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y + 6, y, y, y + 6], color=color, lw=1.3, zorder=3)
    ax.text((x0 + x1) / 2, y - 8, text, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=color)


def panel_head(x, title, subtitle, color):
    ax.text(x, 96, title, fontsize=15, fontweight="bold", color=color, family="serif")
    ax.text(x, 118, subtitle, fontsize=9.5, color=MUTED)


# ----------------------------------------------------------------- left panel
LX = 45
panel_head(LX, "Score-based eviction", "H2O · SnapKV · R-KV · VaSE · TriAttention", INK)
bracket(LX, LX + W, 168, "sink", BLUE)
bracket(LX + W + GAP, LX + NCELL * (W + GAP) - GAP, 168, "top-K scores kept, per head", MUTED)

for hi, (y, hd) in enumerate(zip(ROWS, LEFT)):
    scores = hd["p"] + hd["g"]
    keep = set(sorted(range(13), key=lambda i: -scores[i])[:8])
    cell(LX, y, fill=BLUE, alpha=.85)
    for i, sc in enumerate(scores):
        x = LX + (i + 1) * (W + GAP)
        lab = f"{sc:.1f}".lstrip("0")
        if i in keep:
            cell(x, y, fill=BLUE, alpha=.85 if i < NPROMPT - 1 else .55, label=lab)
        else:
            cell(x, y, label=lab, struck=True)
    ax.text(LX + NCELL * (W + GAP) + 4, y + H / 2, f"head {hi+1}",
            fontsize=9, color=MUTED, va="center")

ax.text(LX, 352, "the question competes by score", fontsize=10.5, fontweight="bold", color=RED)
ax.text(LX, 374, "TriAttention pins the whole prompt; the others keep only sinks",
        fontsize=9.5, color=MUTED)
ax.text(LX, 404, "cost: a scoring pass at every eviction", fontsize=9.5, color=MUTED)

# ----------------------------------------------------------------- divider
ax.plot([578, 578], [86, 412], color=LINE, lw=1.3)

# ----------------------------------------------------------------- right panel
RX = 635
panel_head(RX, "Random Attention (ours)", "no scores, no calibration, nothing to tune", RED)
bracket(RX, RX + NPROMPT * (W + GAP) - GAP, 168, "prompt: always kept", BLUE)
bracket(RX + NPROMPT * (W + GAP), RX + NCELL * (W + GAP) - GAP, 168,
        "trace: uniform random, per head", RED)

for hi, (y, keep) in enumerate(zip(ROWS, RIGHT)):
    for i in range(NCELL):
        x = RX + i * (W + GAP)
        if i < NPROMPT:
            cell(x, y, fill=BLUE, alpha=.85)
        elif (i - NPROMPT) in keep:
            cell(x, y, fill=RED, alpha=.9)
        else:
            cell(x, y)
    ax.text(RX + NCELL * (W + GAP) + 4, y + H / 2, f"head {hi+1}",
            fontsize=9, color=MUTED, va="center")

ax.text(RX, 352, "every head keeps a different subset", fontsize=10.5, fontweight="bold", color=INK)
ax.text(RX, 374, "so some copy of what the model still needs survives", fontsize=9.5, color=MUTED)
ax.text(RX, 404, "cost: none — no scoring pass at all", fontsize=9.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
fig.savefig(os.path.join(out, "random_attention_diagram.png"), facecolor=BG)
fig.savefig(os.path.join(out, "random_attention_diagram.pdf"), facecolor=BG)
print("wrote assets/random_attention_diagram.png / .pdf")
