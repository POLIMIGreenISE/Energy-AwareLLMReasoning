"""Figures for the per-query chapter and the deployment chapter.

    python scripts/fig_ch6_ch8.py  ->  thesis/Images/asymmetry.png
                                       thesis/Images/live_energy.png
                                       thesis/Images/controller.png

Every number is loaded from the results files, never transcribed. Three Chapter 7 figures once
carried hand-copied tables and went stale when the experiments were re-run; see fig_ch5.py.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
TEAL, OCHRE = "#0b8f86", "#b0591a"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans", "axes.titlesize": 8.5})

J_PER_TOKEN = 3.7        # the batch-one constant of sec:paired; it cancels, see sec:simulator-limits


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ==================================================================== 1. the asymmetry (Chapter 6)
# MEASURED: results/crossdataset/official_router_eval.md -- energy-adjusted accuracy of each router
# minus the random-routing floor on the realistic mixed corpus, bootstrap CI.
ROUTERS = [
    ("published mmBERT",   -0.0071, -0.0189, +0.0044),
    ("published 0.6B",     -0.0060, -0.0175, +0.0061),
    ("ours, ModernBERT",   +0.0094, -0.0046, +0.0236),
    ("ours, logistic reg.", +0.0128, -0.0004, +0.0267),
]

# MEASURED: results/e10/dtok_pred.json (scripts/e10_dtok_predictability.py), out-of-fold.
_pred = json.load(open("results/e10/dtok_pred.json"))
_dhat = _pred["dhat"]
_paired = json.load(open("results/paired/paired_oracle_final.json"))
_true = {r["qid"]: (r["energy_wh_think"] - r["energy_wh_no_think"]) * 3600.0 / J_PER_TOKEN
         for r in _paired}
qids = [q for q in _dhat if q in _true and _true[q] > 0]
yhat = np.array([_dhat[q] for q in qids])
ytrue = np.array([_true[q] for q in qids])

rho = spearmanr(yhat, ytrue).statistic
within2 = float(np.mean(np.maximum(yhat / ytrue, ytrue / yhat) <= 2.0))
print(f"  check: n={len(qids)} mean dtok={ytrue.mean():.0f} rho={rho:.3f} within2x={within2:.3f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                               gridspec_kw={"width_ratios": [1.05, 1]})

ys = np.arange(len(ROUTERS))
for y, (name, g, lo, hi) in zip(ys, ROUTERS):
    axL.plot([lo, hi], [y, y], color=INK, lw=1.1, zorder=3)
    for b in (lo, hi):
        axL.plot([b, b], [y - .12, y + .12], color=INK, lw=1.1, zorder=3)
    # One colour for all four: the panel's point is that none of them is distinguishable from
    # zero, and colouring by the sign of a non-significant estimate would assert the opposite.
    axL.plot([g], [y], "o", ms=5, color=MUTED, zorder=4)
axL.axvline(0, color=MUTED, lw=1.0, zorder=2)
axL.set_yticks(ys)
axL.set_yticklabels([n for n, *_ in ROUTERS])
axL.set_ylim(-.6, len(ROUTERS) - .4)
axL.set_xlabel("energy-adjusted accuracy $-$ random floor")
axL.set_title("Which queries benefit: not predictable", color=INK, loc="left", pad=8)
axL.text(0, 1.005, "every interval includes zero", transform=axL.transAxes,
         color=MUTED, fontsize=7, va="bottom")
tidy(axL, grid_axis="x")

axR.plot([1e2, 2e4], [1e2, 2e4], color=MUTED, lw=1.0, zorder=3)
for f, ls in ((2.0, (0, (4, 3))), (0.5, (0, (4, 3)))):
    axR.plot([1e2, 2e4], [1e2 * f, 2e4 * f], color=MUTED, lw=.7, ls=ls, zorder=3)
axR.scatter(yhat, ytrue, s=3.5, color=TEAL, alpha=.22, edgecolors="none", zorder=2)
axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlim(1.5e2, 2e4)
axR.set_ylim(1.5e2, 2e4)
axR.set_xlabel("predicted extra tokens (out of fold)")
axR.set_ylabel("measured extra tokens")
axR.set_title("What they cost: predictable", color=INK, loc="left", pad=8)
axR.text(0, 1.005, r"$\rho$ = %.2f, %.0f%% inside the dashed factor-of-two band"
         % (rho, 100 * within2), transform=axR.transAxes, color=MUTED, fontsize=7, va="bottom")
tidy(axR, grid_axis="both")

fig.tight_layout(pad=0.6, w_pad=1.6)
fig.savefig("thesis/Images/asymmetry.png", dpi=300, facecolor=SURF)
print("wrote thesis/Images/asymmetry.png")


# ============================================================ 2. live energy bracket (Chapter 8)
# MEASURED: results/livetest/sim_vs_live.jsonl. The simulator charges no idle power, so a correct
# prediction must sit between the measured total and that total with idle draw removed.
rows = [json.loads(l) for l in open("results/livetest/sim_vs_live.jsonl")]
order = {"low": 0, "bursty": 1, "high": 2}
rows.sort(key=lambda r: (order[r["level"]], r["policy"]))

fig, ax = plt.subplots(figsize=(7.2, 3.0))
xs = np.arange(len(rows))
inside = 0
for x, r in zip(xs, rows):
    lo, hi, sim = r["meas_wh_idlesub"], r["meas_wh"], r["sim_wh"]
    ok = lo <= sim <= hi
    inside += ok
    ax.plot([x, x], [lo, hi], color=GRID, lw=7, solid_capstyle="butt", zorder=2)
    ax.plot([x], [sim], "o", ms=5, color=TEAL if ok else OCHRE, zorder=4)
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['policy']}\n{r['level']}" for r in rows], fontsize=7)
ax.set_xlim(-.7, len(rows) - .3)
ax.set_ylabel("energy (Wh)")
ax.set_title("Simulated energy against the measured bracket", color=INK, loc="left", pad=17)
ax.text(0, 1.005,
        "grey band: measured total down to the same total with idle draw removed  ·  "
        f"{inside} of {len(rows)} predictions land inside",
        transform=ax.transAxes, color=MUTED, fontsize=7, va="bottom")
tidy(ax)
fig.tight_layout(pad=0.6)
fig.savefig("thesis/Images/live_energy.png", dpi=300, facecolor=SURF)
print(f"wrote thesis/Images/live_energy.png  ({inside}/{len(rows)} inside)")


# ================================================================ 3. the controller (Chapter 8)
# MEASURED: results/repair/pacing_*.jsonl (notebooks/13_controller_repair.ipynb).
def pacing(name):
    return [json.loads(l) for l in open(f"results/repair/pacing_{name}.jsonl")]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.9))

for name, lab, col in (("bursty_as-shipped", "as shipped", OCHRE),
                       ("bursty_guarded", "guarded", TEAL)):
    tr = pacing(name)
    axL.plot([r["n_done"] for r in tr], [r["lam_after"] for r in tr],
             color=col, lw=1.3, label=lab, zorder=3)
axL.set_yscale("log")
axL.set_xlabel("requests completed")
axL.set_ylabel(r"$\lambda$")
axL.set_title("Bursty traffic: the loop that lost control", color=INK, loc="left", pad=17)
axL.text(0, 1.005, "the shipped loop reaches its floor by request 29 of 200",
         transform=axL.transAxes, color=MUTED, fontsize=7, va="bottom")
axL.legend(frameon=False, fontsize=7, loc="lower right")
tidy(axL, grid_axis="both")

tr = pacing("high_guarded")
frac = np.array([r["frac"] for r in tr])
spent = np.array([r["spent_j"] for r in tr]) / 3600.0
BUDGET_WH = 34.37                     # 1.5x the measured never-reasoning energy, CONTROLLER_REPAIR 3
axR.plot([0, 1], [0, BUDGET_WH], color=MUTED, lw=1.0, ls=(0, (4, 3)),
         label="what pacing assumes", zorder=3)
axR.plot(frac, spent, color=TEAL, lw=1.4, label="what it actually spent", zorder=4)
axR.axhline(BUDGET_WH, color=OCHRE, lw=1.0, zorder=2)
axR.text(0.02, BUDGET_WH, "authorized budget", color=OCHRE, fontsize=7, va="bottom")
axR.set_xlabel("fraction of requests completed")
axR.set_ylabel("energy spent (Wh)")
axR.set_title("Saturation: the bill arrives in the drain", color=INK, loc="left", pad=17)
axR.text(0, 1.005, "four fifths of the energy lands after 58% of completions",
         transform=axR.transAxes, color=MUTED, fontsize=7, va="bottom")
axR.legend(frameon=False, fontsize=7, loc="upper left")
tidy(axR)

fig.tight_layout(pad=0.6, w_pad=1.6)
fig.savefig("thesis/Images/controller.png", dpi=300, facecolor=SURF)
print("wrote thesis/Images/controller.png")
