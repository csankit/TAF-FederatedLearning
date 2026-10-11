"""Schematic of TAF v3 server pipeline (not experimental data)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = "#1f1f1e", "#6b6a63"
BLUE, BLUE_BG = "#2a78d6", "#e6f0fb"
GREY_BG, GREY_EDGE = "#f3f2ee", "#bdbbb2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.6, 3.3)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, title, sub, fc=GREY_BG, ec=GREY_EDGE, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=8, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.6, sub, ha="center", va="top", fontsize=7, color=MUTED, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=9, shrinkA=0, shrinkB=0))


box(0, 16, 10, 14, "Clients", "updates\n$u_i$")
box(13, 16, 16, 14, "1  Norm bound", "scale to lower\nmedian norm $\\tau$")
box(32.5, 31, 20, 13, "2  Cluster gate $m_i$", "HDBSCAN on local\nmodels, majority only")
box(32.5, 16.5, 20, 13, "3  Anomaly $A_i$", "robust z of layer\nnorms and cosines")
box(32.5, 2, 20, 13, "4  Collusion $C_i$", "robust z of closest\npeer similarity")
box(56, 16, 13, 14, "Gate", "$g_i = m_i(1-A_i)$\n$\\times(1-C_i)$", fc=BLUE_BG, ec=BLUE, tc=BLUE)
box(72.5, 16, 15, 14, "5  Reputation", "$R_i$: slow up,\nfast down")
box(90.5, 16, 9.5, 14, "Mean", "weights\n$T_i = g_i R_i$")
arrow(10.5, 23, 13, 23)
for y in (37.5, 23, 8.5):
    arrow(29.5, 23, 32.5, y)
    arrow(53, y, 56, 23)
arrow(69.5, 23, 72.5, 23)
arrow(88, 23, 90.5, 23)
ax.text(95.2, 13.5, "next global\nmodel", ha="center", va="top", fontsize=7, color=MUTED)
out = sys.argv[1] if len(sys.argv) > 1 else "."
fig.savefig(f"{out}/fig_architecture.pdf", bbox_inches="tight")
fig.savefig(f"{out}/fig_architecture.png", dpi=200, bbox_inches="tight")
