"""E1 -- Given an ENERGY QUOTA, which think/no-think allocation policy delivers most accuracy?

Split into the two questions an operator actually has, because conflating them is unfair
(policies that land at different energies cannot be compared on accuracy alone):

  E1a  MATCHED-ENERGY FRONTIER. Sweep each policy family's control parameter, simulate on
       held-out traces, build its mean energy-accuracy frontier, and read accuracy AT the cap.
       Everyone is compared at exactly the same energy -> pure allocation quality.

  E1b  CAP ADHERENCE. Tune each policy offline on historical traces (as a real deployment does),
       deploy on held-out traces, and measure whether it actually respects the cap.

Policies: random@rate (the online random floor) / static-tau (today's practice: one fixed
threshold) / load-aware (threshold on score > lam*J(L)*dtok) / load-aware+feedback (lam adapts
on the meter, counting COMMITTED in-flight energy -- without that correction naive feedback
overshoots badly, because a think-decision's energy arrives long after the decision) /
hindsight oracle (knapsack: flip-the-answer queries, cheapest first).

Free (no GPU). Out: results/e1/{e1_quota.md, e1_quota.png}
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH

OUT = "results/e1"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
REGIMES = {"bimodal": dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0),
           "stable":  dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0)}
N_TRACE, N_EVAL, N_HIST = 600, 8, 3
BUDGETS = [0.25, 0.40, 0.55, 0.70]
print(f"queries {len(wl.qids)} | J/tok spread {max(hw.JTOK.values())/min(hw.JTOK.values()):.0f}x")

FLIP = sorted([q for q in wl.qids if wl.dacc(q) > 0], key=lambda q: wl.dtok(q))

# ---------------------------------------------------------------- policy families (param -> decider)
def pol_static(tau):   return lambda L, q, c: SCORE[q] >= tau
def pol_random(rate, seed=0):
    rng = np.random.default_rng(seed)
    return lambda L, q, c: rng.random() < rate
def pol_load(lam):     return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * wl.dtok(q)
def pol_oracle(k):
    chosen = set(FLIP[:int(k)])
    return lambda L, q, c: q in chosen

def pol_feedback(budget_wh, lam0=1e-3, kp=0.5, n_total=N_TRACE):
    """lam adapts on the meter. Counts COMMITTED in-flight energy (ctx['inflight_j']) so the
    controller does not over-think while the bill for earlier decisions is still in flight."""
    state = {"lam": lam0}
    def decide(L, q, c):
        return SCORE[q] > state["lam"] * hw.jtok(L) * wl.dtok(q)
    def on_complete(c):
        f = c["n_done"] / n_total
        if f < 0.05: return
        committed_wh = (c["energy_j"] + c["inflight_j"]) * WH      # lag correction
        pace = committed_wh / max(budget_wh * f, 1e-9)
        state["lam"] = float(np.clip(state["lam"] * (pace ** kp), 1e-8, 1e4))
    return decide, on_complete

# Grids must RESOLVE each family's usable operating points, or we handicap it artificially:
#  - the energy response to thinking is steeply non-linear (queueing): in the stable regime a
#    think-rate of 0.18 already costs 72% of the whole never->always range, so sample densely low.
#  - the fine-tuned router's scores are SATURATED (81% <0.1, 11% >0.9, almost nothing between), so
#    a threshold grid must follow the score QUANTILES rather than be uniform on [0,1].
_sv = np.array(sorted(SCORE.values()))
TAU_GRID = np.unique(np.concatenate([[0.0], np.quantile(_sv, np.linspace(0.002, 0.998, 26)), [1.01]]))
RATE_GRID = np.unique(np.concatenate([np.linspace(0, 0.2, 11), np.linspace(0.25, 1.0, 8)]))
FAMILIES = {
    "random@rate":  ("linear", RATE_GRID, pol_random),
    "static-tau":   ("quantile", TAU_GRID, pol_static),
    "load-aware":   ("log", np.logspace(-7, 3, 20), pol_load),
    "hindsight oracle": ("linear", np.unique(np.linspace(0, len(FLIP), 14).astype(int)), pol_oracle),
}
print(f"score saturation: frac<0.1 {np.mean(_sv < 0.1):.3f}, frac>0.9 {np.mean(_sv > 0.9):.3f} "
      f"-> a fixed threshold has few distinct operating points")

def run_mean(trace_seeds, regime_p, decider_fn, param, fb=False):
    """Mean (energy, accuracy, p95) over held-out traces for one parameter value."""
    E, A, P, T = [], [], [], []
    for s in trace_seeds:
        tr = make_trace(wl, N_TRACE, s, **regime_p)
        if fb:
            d, oc = decider_fn(param)
            r = simulate(tr, d, wl, hw, on_complete=oc)
        else:
            r = simulate(tr, decider_fn(param), wl, hw)
        E.append(r["energy_wh"]); A.append(r["acc"]); P.append(r["lat_p95"]); T.append(r["think_rate"])
    return np.mean(E), np.mean(A), np.mean(P), np.mean(T)

def frontier_acc_at(front, e):
    """Best accuracy achievable at energy <= e on a policy's frontier."""
    feas = [a for pe, a in front if pe <= e + 1e-9]
    return max(feas) if feas else np.nan

rows_a, rows_b, fronts = [], [], {}
for reg, p in REGIMES.items():
    eval_seeds = list(range(N_EVAL))
    hist_seeds = [100 + i for i in range(N_HIST)]
    nev = run_mean(eval_seeds, p, lambda _: (lambda L, q, c: False), None)
    alw = run_mean(eval_seeds, p, lambda _: (lambda L, q, c: True), None)
    print(f"\n### {reg}: never {nev[1]:.3f}@{nev[0]:.1f}Wh  always {alw[1]:.3f}@{alw[0]:.1f}Wh")

    # ---------- E1a: build each family's frontier on held-out traces
    fr = {}
    for name, (kind, grid, fn) in FAMILIES.items():
        pts = [run_mean(eval_seeds, p, fn, v) for v in grid]
        pts_ea = sorted([(e, a) for e, a, _, _ in pts])
        best, out = -1, []
        for e, a in pts_ea:
            if a > best + 1e-12: out.append((e, a)); best = a
        fr[name] = out
    # feedback: its "parameter" IS the target budget, so sweep targets
    fb_targets = [nev[0] + f * (alw[0] - nev[0]) for f in np.linspace(0.05, 1.0, 12)]
    pts = [run_mean(eval_seeds, p, lambda b: pol_feedback(b), b, fb=True) for b in fb_targets]
    pts_ea = sorted([(e, a) for e, a, _, _ in pts]); best, out = -1, []
    for e, a in pts_ea:
        if a > best + 1e-12: out.append((e, a)); best = a
    fr["load-aware+feedback"] = out
    fronts[reg] = fr

    print("  E1a  accuracy at MATCHED energy (read off each policy's held-out frontier)")
    print(f"    {'cap':>6} " + " ".join(f"{n:>20}" for n in fr))
    for frac in BUDGETS:
        B = nev[0] + frac * (alw[0] - nev[0])
        vals = {n: frontier_acc_at(fr[n], B) for n in fr}
        base = vals.get("random@rate", np.nan)
        print(f"    {frac:>5.0%} " + " ".join(f"{vals[n]:>20.4f}" for n in fr))
        for n in fr:
            cap_pct = 100 * (vals[n] - nev[1]) / max(vals["hindsight oracle"] - nev[1], 1e-9)
            rows_a.append((reg, frac, n, vals[n], vals[n] - base, cap_pct))

    # ---------- E1b: tune offline on historical traces, deploy, measure cap adherence
    def tune(fn, grid, fb=False):
        """largest-thinking parameter whose HISTORICAL mean energy still fits the cap"""
        best = None
        for v in grid:
            e, a, _, _ = run_mean(hist_seeds, p, fn, v, fb=fb)
            if e <= B_cur and (best is None or a > best[1]): best = (v, a)
        return best[0] if best else grid[0]
    for frac in BUDGETS:
        B_cur = nev[0] + frac * (alw[0] - nev[0])
        for name, (kind, grid, fn) in FAMILIES.items():
            v = tune(fn, grid)
            E, A, P, T = [], [], [], []
            for s in eval_seeds:
                tr = make_trace(wl, N_TRACE, s, **p)
                r = simulate(tr, fn(v), wl, hw)
                E.append(r["energy_wh"]); A.append(r["acc"]); P.append(r["lat_p95"]); T.append(r["think_rate"])
            E = np.array(E)
            rows_b.append((reg, frac, name, np.mean(A), np.mean(E), 100 * (np.mean(E) - B_cur) / B_cur,
                           100 * np.mean(E > B_cur * 1.02), np.mean(T), np.mean(P)))
        # feedback targets the cap directly
        E, A, P, T = [], [], [], []
        for s in eval_seeds:
            tr = make_trace(wl, N_TRACE, s, **p)
            d, oc = pol_feedback(B_cur)
            r = simulate(tr, d, wl, hw, on_complete=oc)
            E.append(r["energy_wh"]); A.append(r["acc"]); P.append(r["lat_p95"]); T.append(r["think_rate"])
        E = np.array(E)
        rows_b.append((reg, frac, "load-aware+feedback", np.mean(A), np.mean(E),
                       100 * (np.mean(E) - B_cur) / B_cur, 100 * np.mean(E > B_cur * 1.02),
                       np.mean(T), np.mean(P)))
    print("  E1b  cap adherence (tuned offline on historical traces, deployed held-out)")
    print(f"    {'cap':>5} {'policy':22} {'acc':>7} {'energy':>8} {'vs cap':>8} {'breach':>7} {'p95 lat':>8}")
    for r in [r for r in rows_b if r[0] == reg]:
        print(f"    {r[1]:>4.0%} {r[2]:22} {r[3]:.4f} {r[4]:8.1f} {r[5]:+7.1f}% {r[6]:6.0f}% {r[8]:8.1f}")

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
COL = {"random@rate": "tab:gray", "static-tau": "tab:orange", "load-aware": "tab:blue",
       "load-aware+feedback": "tab:green", "hindsight oracle": "k"}
for ax, reg in zip(axes, ["bimodal", "stable"]):
    for name, c in COL.items():
        fr = fronts[reg].get(name)
        if not fr: continue
        ax.plot([e for e, a in fr], [a for e, a in fr],
                "--" if name == "hindsight oracle" else "o-", color=c, ms=4, lw=1.8, label=name)
    ax.set_xlabel("energy over the trace (Wh, measured power integrated)")
    ax.set_ylabel("accuracy delivered")
    ax.set_title(f"E1a: energy–accuracy frontier under load — {reg}")
    ax.grid(alpha=.3); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/e1_quota.png", dpi=130)

with open(f"{OUT}/e1_quota.md", "w", encoding="utf-8") as f:
    f.write("# E1 — allocating think/no-think under an energy quota\n\n")
    f.write(f"Frontiers built on {N_EVAL} held-out traces; offline tuning uses {N_HIST} separate "
            "historical traces. Energy = measured GPU power integrated over time; load emerges from "
            "arrivals+service. Prefill time omitted (decode-dominated).\n\n")
    f.write("## E1a — accuracy at MATCHED energy (fair comparison)\n\n")
    f.write("|regime|cap|policy|accuracy|vs random@rate|% of oracle headroom|\n|--|--|--|--|--|--|\n")
    for reg, frac, n, v, d, cp in rows_a:
        f.write(f"|{reg}|{frac:.0%}|{n}|{v:.4f}|{d*100:+.2f}pp|{cp:.0f}%|\n")
    f.write("\n## E1b — cap adherence (tuned offline, deployed on held-out traces)\n\n")
    f.write("|regime|cap|policy|accuracy|energy (Wh)|vs cap|breach rate|think-rate|p95 lat (s)|\n")
    f.write("|--|--|--|--|--|--|--|--|--|\n")
    for reg, frac, n, a, e, over, br, th, lat in rows_b:
        f.write(f"|{reg}|{frac:.0%}|{n}|{a:.4f}|{e:.1f}|{over:+.1f}%|{br:.0f}%|{th*100:.0f}%|{lat:.1f}|\n")
print(f"\nwrote {OUT}/e1_quota.md, e1_quota.png")
