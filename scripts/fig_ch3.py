"""Chapter 3 figures: the measured J(L) curve, and its transfer to a second accelerator.

    python scripts/fig_ch3.py   ->  thesis/Images/jl_curve.png
                                    thesis/Images/cross_gpu.png
"""
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
TEAL, OCHRE = "#0b8f86", "#b0591a"          # validated CVD-safe pair

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans", "axes.titlesize": 8.5,
                     "legend.frameon": False})

A100 = "results/batching/concurrency.jsonl"
L4 = "results/batching/concurrency_nvidia_l4.jsonl"


def curve(path, field, l_out=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    agg = defaultdict(list)
    for r in rows:
        if l_out is None or r["l_out"] == l_out:
            agg[r["batch"]].append(r[field])
    b = sorted(agg)
    return np.array(b, float), np.array([st.mean(agg[k]) for k in b])


def style(ax, title, ylab, last=False):
    ax.set_title(title, color=INK, pad=6, loc="left")
    ax.set_ylabel(ylab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.set_xlabel("in-service concurrency")
    ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ============================================================ figure 3.1: the curve
b, j256 = curve(A100, "energy_per_out_token_j", 256)
_, j1024 = curve(A100, "energy_per_out_token_j", 1024)
_, pw = curve(A100, "power_w")
_, tp = curve(A100, "throughput_tok_s")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

ax = axes[0]
ax.plot(b, j256, color=TEAL, lw=1.8, marker="o", ms=3.6, label="256-token generations")
ax.plot(b, j1024, color=OCHRE, lw=1.8, marker="s", ms=3.6, label="1024-token generations")
ax.set_yscale("log")
ax.set_yticks([0.08, 0.25, 1.0, 3.5])
ax.set_yticklabels(["0.08", "0.25", "1.0", "3.5"])
style(ax, "Energy per output token", "joules")
ax.legend(loc="upper right", fontsize=6.8, handlelength=1.6, borderpad=.1)
ax.annotate("42x", xy=(20, 1.05), color=INK, fontsize=9, fontweight="bold", ha="center")

ax = axes[1]
ax.plot(b, pw, color=TEAL, lw=1.8, marker="o", ms=3.6)
ax.set_ylim(0, 300)
ax.set_yticks([0, 100, 200, 300])
style(ax, "GPU power", "watts")
ax.annotate("+14% end to end", xy=(2.2, 60), color=MUTED, fontsize=7)

ax = axes[2]
ax.plot(b, tp, color=TEAL, lw=1.8, marker="o", ms=3.6)
ax.set_yscale("log")
ax.set_yticks([68, 264, 992, 3236])
ax.set_yticklabels(["68", "264", "992", "3236"])
style(ax, "Aggregate throughput", "tokens/s")
ax.annotate("48x", xy=(8, 260), color=INK, fontsize=9, fontweight="bold", ha="center")

fig.tight_layout(pad=0.6, w_pad=1.6)
fig.savefig("thesis/Images/jl_curve.png", dpi=300, facecolor=SURF)
print("wrote thesis/Images/jl_curve.png")

# ============================================================ figure 3.2: cross-GPU
_, a_j = curve(A100, "energy_per_out_token_j")
_, l_j = curve(L4, "energy_per_out_token_j", 256)
_, a_p = curve(A100, "power_w")
_, l_p = curve(L4, "power_w", 256)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

ax = axes[0]
ax.plot(b, a_j, color=TEAL, lw=1.8, marker="o", ms=3.6, label="A100-SXM4-40GB")
ax.plot(b, l_j, color=OCHRE, lw=1.8, marker="s", ms=3.6, label="L4")
ax.set_yscale("log")
ax.set_yticks([0.08, 0.3, 1.0, 4.4])
ax.set_yticklabels(["0.08", "0.3", "1.0", "4.4"])
style(ax, "Absolute energy per token", "joules")
ax.legend(loc="upper right", fontsize=6.8, handlelength=1.6, borderpad=.1)

ax = axes[1]
ax.plot(b, a_j / a_j[0], color=TEAL, lw=1.8, marker="o", ms=3.6)
ax.plot(b, l_j / l_j[0], color=OCHRE, lw=1.8, marker="s", ms=3.6)
ax.set_yscale("log")
ax.set_yticks([0.02, 0.1, 0.5, 1.0])
ax.set_yticklabels(["0.02", "0.1", "0.5", "1.0"])
style(ax, "Normalized, $J(L)/J(1)$", "relative to $L=1$")
ax.annotate("mean error 4.6%\ncorrelation 0.9995", xy=(1.3, 0.035), color=MUTED, fontsize=7)

ax = axes[2]
ax.plot(b, a_p, color=TEAL, lw=1.8, marker="o", ms=3.6)
ax.plot(b, l_p, color=OCHRE, lw=1.8, marker="s", ms=3.6)
ax.set_ylim(0, 300)
ax.set_yticks([0, 100, 200, 300])
style(ax, "GPU power", "watts")

fig.tight_layout(pad=0.6, w_pad=1.6)
fig.savefig("thesis/Images/cross_gpu.png", dpi=300, facecolor=SURF)
print("wrote thesis/Images/cross_gpu.png")
