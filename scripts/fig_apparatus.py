"""The experimental apparatus.

A block diagram separating what is measured on hardware from what is simulated over those
measurements, and showing where the live-replay validation closes the loop. House style matches
the other figures (see scripts/e11_real_trace.py).

    python scripts/fig_apparatus.py     ->  figures/apparatus.png   (set FIG_DIR to change)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
MEASURED_BG = "#eef2f1"
MEASURED_EDGE = "#9aa8a5"

plt.rcParams.update({"font.size": 9, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans"})

OUT = os.path.join(os.environ.get("FIG_DIR", "figures"), "apparatus.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
TITLE_FS, BODY_FS = 8.2, 7.0

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(cx, cy, w, h, title, body=None):
    """A titled block, (cx, cy) at its centre, w/h in axis units."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=SURF, edgecolor=INK, linewidth=0.9, zorder=3))
    if body:
        ax.text(cx, cy + h * 0.20, title, ha="center", va="center", color=INK,
                fontsize=TITLE_FS, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.22, body, ha="center", va="center", color=MUTED,
                fontsize=BODY_FS, linespacing=1.5, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", color=INK,
                fontsize=TITLE_FS, fontweight="bold", zorder=4)


def arrow(x1, y1, x2, y2, dashed=False, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=0.95, color=color, zorder=2,
                                 linestyle=(0, (4, 2.5)) if dashed else "solid"))


def line(x1, y1, x2, y2, color=MUTED):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.95, zorder=2,
            linestyle=(0, (4, 2.5)))


# ---------------------------------------------------------------- measured region
ax.add_patch(FancyBboxPatch((2, 55), 78, 42,
                            boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor=MEASURED_BG, edgecolor=MEASURED_EDGE,
                            linewidth=0.9, zorder=0))
ax.text(4.5, 93.5, "M E A S U R E D   O N   H A R D W A R E", ha="left", va="center",
        color=MUTED, fontsize=7.0, fontweight="bold", zorder=1)

box(22, 86, 34, 10, "Qwen3-8B under vLLM",
    "NVIDIA A100-SXM4-40GB\nNVIDIA L4")
box(60, 86, 34, 10, "Zeus / NVML",
    "board energy counter\n4 Hz metrics probe")
arrow(39.4, 86, 42.6, 86)

box(22, 67, 34, 13, "Batch-size sweep",
    "$L = 1 \\dots 64$, six repetitions each\n$\\Rightarrow$   $J(L)$,  $P(L)$,  tokens/s")
box(60, 67, 34, 13, "Paired generation",
    "every query twice: reasoning, direct\n"
    "judge $\\kappa = 0.824$  $\\Rightarrow$  $\\Delta$tok$(q)$, correctness")
arrow(22, 80.6, 22, 74.2)
arrow(60, 80.6, 60, 74.2)

# ---------------------------------------------------------------- simulated
arrow(22, 60.2, 33, 50.4, rad=0.05)
arrow(60, 60.2, 49, 50.4, rad=-0.05)

box(41, 43, 68, 13, "Event-driven serving simulator",
    "arrivals, queue and batch size only;\n"
    "per-token energy, power and decode rate are looked up, never modeled")
arrow(41, 36.2, 41, 29.0)

box(41, 22, 76, 13, "Policy comparison at matched energy",
    "random-selection control  ·  paired bootstrap intervals  ·  shared score grids\n"
    "identical arrival times and identical queries for every policy")
arrow(41, 15.4, 41, 9.0)

box(41, 5.5, 48, 6.6, "Energy–accuracy result, with intervals")

# ---------------------------------------------------------------- validation loop
line(75, 43, 84, 43)
line(84, 43, 84, 78)
arrow(84, 78, 80.6, 78, dashed=True, color=MUTED)
ax.text(85.5, 59, "validated:\nsimulated traces\nreplayed against\nthe live server",
        ha="left", va="center", color=MUTED, fontsize=6.9, linespacing=1.6)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor=SURF)
print("wrote", OUT)
