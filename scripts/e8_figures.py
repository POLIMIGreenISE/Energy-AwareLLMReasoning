"""Regenerate E8's bar figures from the committed rows -- no re-simulation.

`scripts/e8_consolidated.py` takes ~40 minutes because it re-runs the whole sweep. Figures 2-4 are
just views of `results/e8/e8_rows.json`, so a caption or label fix should not cost a sweep. This
redraws them from the committed numbers, which also makes the point that every figure in the thesis
is reproducible from data in the repository.

(fig1, the frontiers, needs the full parameter sweep and is still produced by the main script.)

Out: results/e8/{fig2_gap_by_regime.png, fig3_slo_reversal.png, fig4_decomposition.png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/e8"
D = json.load(open(f"{OUT}/e8_rows.json", encoding="utf-8"))
REGIMES = ["near-idle", "moderate", "saturated", "bursty"]

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_POS, C_NEG = "#2a78d6", "#e34948"          # diverging pair: polarity, not identity
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURF})


def bars(groups, xlabels, ylabel, suptitle, path, figsize=(11.2, 3.6), rect=0.92, fs=8):
    fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    for ax, (title, g) in zip(np.atleast_1d(axes), groups):
        x = np.arange(len(g)); v = [d["gap"] for d in g]
        ax.bar(x, v, width=0.62, color=[C_POS if t >= 0 else C_NEG for t in v], zorder=3)
        ax.errorbar(x, v, yerr=[[d["gap"] - d["lo"] for d in g], [d["hi"] - d["gap"] for d in g]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
        ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(xlabels(g), fontsize=fs)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
        ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, fontsize=10.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, rect)); fig.savefig(path, dpi=200); plt.close(fig)
    print("wrote", path)


# --- fig 2: gap by regime and cap ------------------------------------------------
bars([(r, D["gap"][r]) for r in REGIMES],
     lambda g: [f"{int(d['frac']*100)}%" for d in g],
     "accuracy gap (pp)\nload-aware − static",
     "Where dynamic thresholding pays  ·  deployable load-aware vs a densely swept static threshold, "
     "matched energy, 95% paired CIs  ·  blue = dynamic wins, red = static",
     f"{OUT}/fig2_gap_by_regime.png")

# --- fig 3: the SLO reversal -----------------------------------------------------
bars([(r, [d for d in D["slo"][r] if d["gap"] == d["gap"]]) for r in REGIMES],
     lambda g: [d["mult"] for d in g],
     "accuracy gap (pp)\nload-aware − static",
     "The reversal: under a tight tail-latency target the static threshold wins  ·  "
     "40% energy cap throughout",
     f"{OUT}/fig3_slo_reversal.png")

# --- fig 4: what earns the win ---------------------------------------------------
PRETTY = {"static-token": "+ tokens\n(oracle)", "load-aware-const": "+ load\n(deployable)",
          "load-aware": "+ both\n(oracle)"}
groups = []
for r in REGIMES:
    g = [d for d in D["d"] if d["regime"] == r and d["frac"] == 0.40 and d["policy"] in PRETTY]
    groups.append((r, sorted(g, key=lambda d: list(PRETTY).index(d["policy"]))))
bars(groups, lambda g: [PRETTY[d["policy"]] for d in g],
     "accuracy gap (pp)\nvs static threshold",
     "What earns the win: live load, or per-query token knowledge?  ·  40% energy cap  ·  "
     "each ingredient added to the static threshold",
     f"{OUT}/fig4_decomposition.png", figsize=(11.2, 3.8), rect=0.90, fs=8)
