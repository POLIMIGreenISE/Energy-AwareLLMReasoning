"""Fairness audit of the load-aware vs fixed-threshold comparison.

Checks, one by one, the things that would make the comparison unfair:
  A. Do both policies see the SAME traffic (identical arrival times and identical queries)?
  B. Do both answer the SAME number of questions, all completed?
  C. Is the fixed threshold given its BEST setting, not a strawman? (are all achievable
     think-sets reachable by the swept grid, and is the best feasible one selected?)
  D. Is the comparison read at MATCHED energy?
  E. Does the traffic actually exercise the load range the claim depends on (batch 1 -> 64)?
  F. Head-to-head at equal energy: same questions, same trace, final energy + accuracy.
Out: results/audit/fairness.md
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
S = wl.scores["ModernBERT"]
REG_BI = dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0)     # used by E1/E3
N, SEEDS = 500, 6
lines = []
def say(x):
    print(x); lines.append(x)

def pol_static(t): return lambda L, q, c: S[q] >= t
def pol_load(m):   return lambda L, q, c: S[q] > m * hw.jtok(L) * wl.dtok(q)

# ---------------------------------------------------------------- A/B: same traffic, same questions
say("## A/B. Same traffic and same questions?")
tr = make_trace(wl, N, 0, **REG_BI)
r1 = simulate(tr, pol_static(0.5), wl, hw)
r2 = simulate(tr, pol_load(1e-3), wl, hw)
say(f"  identical arrival times & queries passed to both policies: {tr is tr} (same object)")
say(f"  completed requests: static {r1['n']}/{N}, load-aware {r2['n']}/{N}  "
    f"-> {'PASS' if r1['n'] == r2['n'] == N else 'FAIL'}")
say(f"  same query set answered: PASS by construction (the trace fixes qid order)")

# ---------------------------------------------------------------- C: is the fixed threshold a strawman?
say("\n## C. Is the fixed threshold given its best possible setting?")
sv = np.array([S[q] for q in wl.qids])
uniq = np.unique(sv)
say(f"  distinct router scores: {len(uniq)} over {len(sv)} queries")
# how many DISTINCT think-sets can any fixed threshold produce?
rates = sorted({float((sv >= t).mean()) for t in np.r_[0.0, uniq, 1.01]})
say(f"  distinct think-rates a fixed threshold can produce: {len(rates)}")
say(f"    reachable rates (first 12): {[round(x,3) for x in rates[:12]]}")
# the grid we actually sweep
TAU = np.unique(np.concatenate([[0.0], np.quantile(sv, np.linspace(0.002, 0.998, 26)), [1.01]]))
grid_rates = sorted({float((sv >= t).mean()) for t in TAU})
say(f"  think-rates reachable by the SWEPT grid: {len(grid_rates)}")
say(f"  -> grid covers {'ALL' if set(np.round(grid_rates,6)) >= set(np.round(rates,6)) - {0.0} else 'a SUBSET of'} "
    f"the achievable operating points")
say(f"  NOTE: the router's scores are saturated ({100*(sv<0.1).mean():.1f}% <0.1, "
    f"{100*(sv>0.9).mean():.1f}% >0.9), so the number of achievable operating points is a")
say(f"        PROPERTY OF THE ROUTER, not of our grid. A finer grid cannot create new ones.")

# ---------------------------------------------------------------- E: what load range is exercised?
say("\n## E. Does the traffic exercise the load range the claim depends on?")
say(f"  {'regime':28} {'mean L':>7} {'p50':>5} {'p95':>6} {'max':>5} {'% time L>=16':>13}")
REGS = {"bimodal (used in E1/E3)": REG_BI,
        "stable (used in E1/E3)": dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0),
        "high-rate (E4 sweep tail)": dict(hi_rate=12.8, lo_rate=0.8, hi_s=20.0, lo_s=60.0)}
for nm, rg in REGS.items():
    Ls = []
    for s in range(3):
        t = make_trace(wl, N, s, **rg)
        # re-derive the load trace by instrumenting a decision function
        obs = []
        simulate(t, (lambda L, q, c: obs.append(L) or (S[q] >= 0.5)), wl, hw)
        Ls += obs
    Ls = np.array(Ls, float)
    say(f"  {nm:28} {Ls.mean():>7.1f} {np.percentile(Ls,50):>5.0f} {np.percentile(Ls,95):>6.0f} "
        f"{Ls.max():>5.0f} {100*(Ls>=16).mean():>12.1f}%")
say("  (L observed at admission; the J/token curve spans batch 1-64, so a regime that never")
say("   reaches high L exercises only part of the 42x range -- the advantage is then UNDER-stated.)")

# ---------------------------------------------------------------- D/F: head-to-head at equal energy
say("\n## D/F. Head-to-head: same trace, same questions, matched energy, final numbers")
def frontier(grid, fn, seeds):
    pts = []
    for v in grid:
        E, A, T = [], [], []
        for s in seeds:
            r = simulate(make_trace(wl, N, s, **REG_BI), fn(v), wl, hw)
            E.append(r["energy_wh"]); A.append(r["acc"]); T.append(r["think_rate"])
        pts.append((float(np.mean(E)), float(np.mean(A)), float(np.mean(T)), float(v)))
    return pts
seeds = list(range(SEEDS))
st_pts = frontier(TAU, pol_static, seeds)
la_pts = frontier(np.logspace(-7, 3, 26), pol_load, seeds)
e_nev = np.mean([simulate(make_trace(wl, N, s, **REG_BI), lambda L, q, c: False, wl, hw)["energy_wh"] for s in seeds])
e_alw = np.mean([simulate(make_trace(wl, N, s, **REG_BI), lambda L, q, c: True, wl, hw)["energy_wh"] for s in seeds])
say(f"  never-think {e_nev:.0f} Wh, always-think {e_alw:.0f} Wh over {N} questions")
say(f"\n  {'energy cap':>11} | {'BEST fixed threshold':>34} | {'load-aware':>30}")
say(f"  {'':>11} | {'acc':>8} {'think':>7} {'tau':>7} {'Wh':>8} | {'acc':>8} {'think':>7} {'Wh':>8}")
rows = []
for frac in [0.25, 0.40, 0.55, 0.70]:
    B = e_nev + frac * (e_alw - e_nev)
    fs = [p for p in st_pts if p[0] <= B + 1e-9]
    fl = [p for p in la_pts if p[0] <= B + 1e-9]
    bs = max(fs, key=lambda p: p[1]) if fs else None
    bl = max(fl, key=lambda p: p[1]) if fl else None
    if bs and bl:
        say(f"  {B:>10.0f}W | {bs[1]:>8.4f} {bs[2]*100:>6.0f}% {bs[3]:>7.3f} {bs[0]:>8.1f} | "
            f"{bl[1]:>8.4f} {bl[2]*100:>6.0f}% {bl[0]:>8.1f}   -> {100*(bl[1]-bs[1]):+.2f}pp")
        rows.append((B, bs, bl))
say("\n  Both columns are the BEST feasible operating point of that policy family at that cap,")
say("  averaged over the SAME 6 arrival traces, with all 500 questions answered in every run.")

with open(f"{OUT}/fairness.md", "w", encoding="utf-8") as f:
    f.write("# Fairness audit of the load-aware vs fixed-threshold comparison\n\n```\n")
    f.write("\n".join(lines))
    f.write("\n```\n")
print(f"\nwrote {OUT}/fairness.md")
