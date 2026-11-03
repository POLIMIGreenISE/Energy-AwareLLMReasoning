"""Corrected head-to-head: load-aware vs a PROPERLY swept fixed threshold.

The fairness audit found that the previous threshold grid (quantiles of the score VALUES) collapsed
to 26 operating points because 81.5% of scores are near zero -- while the router actually admits 684
distinct think-rates. That handicapped the baseline. Here the grid is built from the sorted UNIQUE
scores by index, so every sampled threshold yields a distinct think-set and the full think-rate range
is covered.

Everything else is held identical between policies: same arrival traces (same seeds), same queries in
the same order, all questions answered, and accuracy read at MATCHED energy.
Out: results/audit/fair_headtohead.md
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
S = wl.scores["ModernBERT"]
sv = np.array([S[q] for q in wl.qids])
N, SEEDS = 500, 6
REGIMES = {"bimodal": dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0),
           "stable":  dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0)}

# --- grids -------------------------------------------------------------------
uniq = np.unique(sv)
step = max(1, len(uniq) // 160)
TAU_DENSE = np.unique(np.concatenate([[0.0], uniq[::step], uniq[-3:], [1.01]]))
TAU_OLD = np.unique(np.concatenate([[0.0], np.quantile(sv, np.linspace(0.002, 0.998, 26)), [1.01]]))
LAM = np.logspace(-7, 3, 40)
rates = lambda g: sorted({float((sv >= t).mean()) for t in g})
print(f"router: {len(uniq)} distinct scores -> {len(rates(np.r_[0.0, uniq, 1.01]))} achievable think-rates")
print(f"  OLD grid   : {len(TAU_OLD):3d} thresholds -> {len(rates(TAU_OLD)):3d} distinct think-rates")
print(f"  DENSE grid : {len(TAU_DENSE):3d} thresholds -> {len(rates(TAU_DENSE)):3d} distinct think-rates")

def pol_static(t): return lambda L, q, c: S[q] >= t
def pol_load(m):   return lambda L, q, c: S[q] > m * hw.jtok(L) * wl.dtok(q)

def sweep(grid, fn, reg, seeds):
    pts = []
    for v in grid:
        E, A, T = [], [], []
        for s in seeds:
            r = simulate(make_trace(wl, N, s, **reg), fn(v), wl, hw)
            E.append(r["energy_wh"]); A.append(r["acc"]); T.append(r["think_rate"])
        pts.append((float(np.mean(E)), float(np.mean(A)), float(np.mean(T)), float(v)))
    return pts

def best_at(pts, cap):
    f = [p for p in pts if p[0] <= cap + 1e-9]
    return max(f, key=lambda p: p[1]) if f else None

lines = []
def say(x): print(x); lines.append(x)

for reg_name, reg in REGIMES.items():
    seeds = list(range(SEEDS))
    e_nev = np.mean([simulate(make_trace(wl, N, s, **reg), lambda L, q, c: False, wl, hw)["energy_wh"] for s in seeds])
    e_alw = np.mean([simulate(make_trace(wl, N, s, **reg), lambda L, q, c: True, wl, hw)["energy_wh"] for s in seeds])
    st_old = sweep(TAU_OLD, pol_static, reg, seeds)
    st_new = sweep(TAU_DENSE, pol_static, reg, seeds)
    la = sweep(LAM, pol_load, reg, seeds)
    say(f"\n### {reg_name}: never {e_nev:.0f} Wh, always {e_alw:.0f} Wh over {N} questions "
        f"({SEEDS} identical traces per policy)")
    say(f"  {'cap':>7} | {'fixed-thr (OLD grid)':>21} | {'fixed-thr (DENSE grid)':>23} | "
        f"{'load-aware':>19} | {'gap vs DENSE':>12}")
    say(f"  {'':>7} | {'acc':>9} {'Wh':>10} | {'acc':>9} {'think':>6} {'Wh':>6} | "
        f"{'acc':>9} {'think':>4} {'Wh':>4} | ")
    for frac in [0.25, 0.40, 0.55, 0.70]:
        cap = e_nev + frac * (e_alw - e_nev)
        bo, bn, bl = best_at(st_old, cap), best_at(st_new, cap), best_at(la, cap)
        if not (bo and bn and bl): continue
        gap = bl[1] - bn[1]
        say(f"  {cap:>6.0f}W | {bo[1]:>9.4f} {bo[0]:>10.1f} | {bn[1]:>9.4f} {bn[2]*100:>5.0f}% {bn[0]:>6.1f} | "
            f"{bl[1]:>9.4f} {bl[2]*100:>3.0f}% {bl[0]:>4.0f} | {gap*100:>+11.2f}pp")
        lines.append("")
        lines[-1] = lines[-1]  # keep list shape

with open(f"{OUT}/fair_headtohead.md", "w", encoding="utf-8") as f:
    f.write("# Corrected head-to-head: load-aware vs a properly swept fixed threshold\n\n")
    f.write(f"The previous grid sampled quantiles of the score VALUES and collapsed to "
            f"{len(rates(TAU_OLD))} operating points; the router actually admits "
            f"{len(rates(np.r_[0.0, uniq, 1.01]))}. The dense grid samples the sorted unique scores by "
            f"index, reaching {len(rates(TAU_DENSE))} distinct think-rates. Same traces, same queries, "
            f"all {N} questions answered, accuracy read at matched energy.\n\n```\n")
    f.write("\n".join(l for l in lines if l))
    f.write("\n```\n")
print(f"\nwrote {OUT}/fair_headtohead.md")
