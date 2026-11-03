"""Figure 2.2 -- how the energy-adjusted accuracy metric is constructed.

Draws, from the real paired OptimalThinkingBench run: a policy's operating points, the
best-feasible envelope EAA integrates, the fixed budget range every policy is scored over, and
EAA itself as the mean height of that envelope.

    python scripts/fig_eaa.py     ->  thesis/Images/eaa_metric.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
FILL, ACCENT = "#e7ecea", "#0b6e66"

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.family": "DejaVu Sans"})

OUT = os.path.join("thesis", "Images", "eaa_metric.png")

# ------------------------------------------------------------------ data
paired = json.load(open("results/paired/paired_oracle_final.json"))
fte = {json.loads(l)["qid"]: json.loads(l)["p_think"]
       for l in open("results/paired/router_fte_modernbert.jsonl", encoding="utf-8")}
rows = [p for p in paired if p["qid"] in fte]
N = len(rows)

cn = np.array([r["correct_no_think"] for r in rows], float)
ct = np.array([r["correct_think"] for r in rows], float)
en = np.array([r["energy_wh_no_think"] for r in rows], float) * 1000.0   # mWh
et = np.array([r["energy_wh_think"] for r in rows], float) * 1000.0
score = np.array([fte[r["qid"]] for r in rows], float)


def acc(choose):
    return float(np.mean(np.where(choose, ct, cn)))


def ene(choose):
    return float(np.mean(np.where(choose, et, en)))


lo, hi = ene(np.zeros(N, bool)), ene(np.ones(N, bool))
a_never, a_always = acc(np.zeros(N, bool)), acc(np.ones(N, bool))

# the router's operating points: one per threshold
taus = np.unique(np.concatenate([np.linspace(0, 1, 201), np.unique(score)]))
pts = [(ene(score >= t), acc(score >= t)) for t in taus]

# the oracle frontier, for reference
d_acc, d_ene = ct - cn, et - en
posm = (d_acc > 0) & (d_ene > 0)
lams = sorted(set([0.0] + (d_acc[posm] / d_ene[posm]).tolist()))
front = [(ene((d_acc - l * d_ene) > 1e-12), acc((d_acc - l * d_ene) > 1e-12)) for l in lams]
front += [(lo, a_never), (hi, a_always)]


def envelope(points, grid):
    """Best accuracy affordable at each budget; never-reason accuracy where nothing is."""
    out = []
    for e in grid:
        feasible = [a for pe, a in points if pe <= e + 1e-9]
        out.append(max(feasible) if feasible else a_never)
    return np.array(out)


budgets = np.linspace(lo, hi, 200)
env_router = envelope(pts, budgets)
env_oracle = envelope(front, budgets)
tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
eaa_router = tz(env_router, budgets) / (hi - lo)
eaa_oracle = tz(env_oracle, budgets) / (hi - lo)

print(f"N={N} | never {a_never:.4f} @ {lo:.0f} mWh | always {a_always:.4f} @ {hi:.0f} mWh")
print(f"EAA router {eaa_router:.4f} | EAA oracle {eaa_oracle:.4f} | "
      f"cheapest router point {min(p[0] for p in pts):.0f} mWh")

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(6.6, 3.5))

ax.fill_between(budgets, a_never - 0.02, env_router, color=FILL, zorder=1)
ax.plot(budgets, env_oracle, color=MUTED, lw=1.2, ls=(0, (5, 2.5)), zorder=3,
        label="hindsight oracle (ceiling)")
ax.plot(budgets, env_router, color=ACCENT, lw=2.0, zorder=4,
        label="best affordable setting of the router")
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, color=MUTED, alpha=0.45,
           zorder=2, label="the router's operating points, one per threshold")

ax.axhline(eaa_router, color=ACCENT, lw=1.0, ls=(0, (1.5, 2)), zorder=5)
ax.text(hi - 0.02 * (hi - lo), eaa_router - 0.006,
        "EAA: the mean height\nof the solid curve",
        fontsize=8, color=ACCENT, ha="right", va="top", linespacing=1.45, zorder=6)

for x, lab, ha, dx in ((lo, "never reason", "left", 0.012), (hi, "always reason", "right", -0.012)):
    ax.axvline(x, color=GRID, lw=0.9, zorder=0)
    ax.text(x + dx * (hi - lo), a_always + 0.016, lab, fontsize=7.6, color=MUTED, ha=ha)

ax.scatter([lo, hi], [a_never, a_always], s=26, color=INK, zorder=6)

ax.set_xlim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
ax.set_ylim(a_never - 0.02, a_always + 0.035)
ax.set_xlabel("energy budget for the workload (mWh)")
ax.set_ylabel("accuracy")
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=7.6, handletextpad=0.6, borderpad=0.2)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor=SURF)
print("wrote", OUT)
