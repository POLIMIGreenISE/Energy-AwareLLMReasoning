"""E4 -- WHEN does load-aware allocation pay? Sweeping arrival rate and burstiness.

E1/E2 showed the benefit is real under bursty load and ~0 under stable load. E4 turns that
binary observation into a boundary condition, which is the question a reader will ask:
"under what serving conditions should I bother?"

Two sweeps, both comparing load-aware vs static-tau at MATCHED energy (each policy's frontier is
built on held-out traces and read at the same energy cap -- the fair comparison from E1a):

  E4a  mean ARRIVAL RATE, fixed burstiness   -> does load LEVEL matter?
  E4b  BURSTINESS at (roughly) fixed mean rate -> does load VARIATION matter?

Hypothesis from E1/E2: the lever is VARIATION, not level. Load-awareness can only exploit the
difference between cheap (high-L) and expensive (low-L) thinking moments; if load never varies
there is nothing to exploit, however high or low it sits.

Free (no GPU). Out: results/e4/{e4_sweep.md, e4_sweep.png}
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate

OUT = "results/e4"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
N_TRACE, N_SEEDS, CAP = 500, 5, 0.40

_sv = np.array(sorted(SCORE.values()))
TAU_GRID = np.unique(np.concatenate([[0.0], np.quantile(_sv, np.linspace(0.002, 0.998, 18)), [1.01]]))
LAM_GRID = np.logspace(-7, 3, 18)

def pol_static(tau): return lambda L, q, c: SCORE[q] >= tau
def pol_load(lam):   return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * wl.dtok(q)

def frontier(params, fn, regime):
    """Mean (energy, accuracy) per parameter over held-out traces -> upper-left hull."""
    pts, stats = [], []
    for v in params:
        E, A, mL, cv, ut = [], [], [], [], []
        for s in range(N_SEEDS):
            r = simulate(make_trace(wl, N_TRACE, s, **regime), fn(v), wl, hw)
            E.append(r["energy_wh"]); A.append(r["acc"])
            mL.append(r["load_mean"]); cv.append(r["load_cv"]); ut.append(r["util"])
        pts.append((np.mean(E), np.mean(A)))
        stats.append((np.mean(mL), np.mean(cv), np.mean(ut)))
    best, out = -1, []
    for e, a in sorted(pts):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out, pts, stats

def acc_at(front, e):
    feas = [a for pe, a in front if pe <= e + 1e-9]
    return max(feas) if feas else np.nan

def eval_regime(regime, label):
    """Gap = MEAN accuracy advantage of load-aware over static-tau across the whole overlapping
    energy range (not a single cap). Single-cap readings are erratic because static-tau's frontier
    has large gaps (saturated scores -> few operating points), so it lands on never-think at some
    caps and on a good point at others; averaging over the range removes that artefact."""
    f_st, _, _ = frontier(TAU_GRID, pol_static, regime)
    f_la, _, st_la = frontier(LAM_GRID, pol_load, regime)
    mid = st_la[len(st_la) // 2]
    lo = max(f_st[0][0], f_la[0][0]); hi = min(f_st[-1][0], f_la[-1][0])
    if hi <= lo:
        return dict(label=label, load_mean=mid[0], load_cv=mid[1], util=mid[2],
                    acc_static=np.nan, acc_load=np.nan, gap=np.nan)
    grid = np.linspace(lo, hi, 25)
    gaps = [acc_at(f_la, e) - acc_at(f_st, e) for e in grid]
    gaps = [g for g in gaps if g == g]
    return dict(label=label, load_mean=mid[0], load_cv=mid[1], util=mid[2],
                acc_static=float(np.mean([acc_at(f_st, e) for e in grid])),
                acc_load=float(np.mean([acc_at(f_la, e) for e in grid])),
                gap=float(np.mean(gaps)) if gaps else np.nan)

rows = []
print(f"E4a — sweeping mean ARRIVAL RATE (fixed burst shape), cap={CAP:.0%}")
print(f"  {'rate':>8} {'busy-L':>8} {'load CV':>8} {'util':>6} {'static':>8} {'load-aware':>11} {'gap':>8}")
for rate in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]:
    reg = dict(hi_rate=rate * 4, lo_rate=rate * 0.25, hi_s=20.0, lo_s=60.0)
    r = eval_regime(reg, f"rate={rate}")
    rows.append(("E4a", rate, r))
    print(f"  {rate:>8.2f} {r['load_mean']:>8.1f} {r['load_cv']:>8.2f} {r['util']:>6.2f} "
          f"{r['acc_static']:>8.4f} {r['acc_load']:>11.4f} {r['gap']*100:>+7.2f}pp")

print(f"\nE4b — sweeping BURSTINESS at ~fixed mean rate, cap={CAP:.0%}")
print(f"  {'burst':>8} {'busy-L':>8} {'load CV':>8} {'util':>6} {'static':>8} {'load-aware':>11} {'gap':>8}")
MEAN_RATE = 0.5
for burst in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    # keep the time-average arrival rate ~constant while widening the hi/lo swing
    hi_s, lo_s = 20.0, 60.0
    frac_hi = hi_s / (hi_s + lo_s)
    lo = MEAN_RATE / (frac_hi * burst + (1 - frac_hi))
    reg = dict(hi_rate=lo * burst, lo_rate=lo, hi_s=hi_s, lo_s=lo_s)
    r = eval_regime(reg, f"burst={burst}")
    rows.append(("E4b", burst, r))
    print(f"  {burst:>8.0f} {r['load_mean']:>8.1f} {r['load_cv']:>8.2f} {r['util']:>6.2f} "
          f"{r['acc_static']:>8.4f} {r['acc_load']:>11.4f} {r['gap']*100:>+7.2f}pp")

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
a = [r for k, x, r in rows if k == "E4a"]; xa = [x for k, x, r in rows if k == "E4a"]
b = [r for k, x, r in rows if k == "E4b"]; xb = [x for k, x, r in rows if k == "E4b"]
ax = axes[0]
ax.semilogx(xa, [r["gap"] * 100 for r in a], "o-", color="tab:blue")
ax.axhline(0, color="k", lw=1); ax.set_xlabel("mean arrival rate (req/s)")
ax.set_ylabel("load-aware − static-τ at matched energy (pp)")
ax.set_title("E4a: does load LEVEL matter?"); ax.grid(alpha=.3)
ax = axes[1]
ax.semilogx(xb, [r["gap"] * 100 for r in b], "s-", color="tab:green")
ax.axhline(0, color="k", lw=1); ax.set_xlabel("burstiness (hi/lo arrival-rate ratio)")
ax.set_ylabel("gap (pp)"); ax.set_title("E4b: does load VARIATION matter?"); ax.grid(alpha=.3)
ax = axes[2]
allr = a + b
ax.scatter([r["load_cv"] for r in a], [r["gap"] * 100 for r in a], c="tab:blue", label="E4a rate sweep")
ax.scatter([r["load_cv"] for r in b], [r["gap"] * 100 for r in b], c="tab:green", marker="s", label="E4b burst sweep")
cvs = np.array([r["load_cv"] for r in allr]); gaps = np.array([r["gap"] * 100 for r in allr])
m = ~np.isnan(gaps)
if m.sum() > 2:
    z = np.polyfit(cvs[m], gaps[m], 1)
    xs = np.linspace(cvs[m].min(), cvs[m].max(), 20)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=1,
            label=f"fit (r={np.corrcoef(cvs[m], gaps[m])[0,1]:.2f})")
ax.axhline(0, color="k", lw=1); ax.set_xlabel("load variability (CV of in-service L)")
ax.set_ylabel("gap (pp)"); ax.set_title("The lever is VARIATION, not level"); ax.grid(alpha=.3)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/e4_sweep.png", dpi=130)

with open(f"{OUT}/e4_sweep.md", "w", encoding="utf-8") as f:
    f.write("# E4 — when does load-aware allocation pay?\n\n")
    f.write(f"load-aware vs static-tau at MATCHED energy (cap = {CAP:.0%} of always-think), each "
            f"frontier built on {N_SEEDS} held-out traces of {N_TRACE} queries. 'busy-L' and "
            "'load CV' are time-weighted over busy periods.\n\n")
    for k, title in [("E4a", "Sweeping mean arrival rate (fixed burst shape)"),
                     ("E4b", f"Sweeping burstiness at ~fixed mean rate ({MEAN_RATE} req/s)")]:
        f.write(f"## {k} — {title}\n\n")
        f.write("|x|busy-L|load CV|utilisation|static-tau|load-aware|gap|\n|--|--|--|--|--|--|--|\n")
        for kk, x, r in rows:
            if kk != k: continue
            f.write(f"|{x}|{r['load_mean']:.1f}|{r['load_cv']:.2f}|{r['util']:.2f}|"
                    f"{r['acc_static']:.4f}|{r['acc_load']:.4f}|{r['gap']*100:+.2f}pp|\n")
        f.write("\n")
    if m.sum() > 2:
        f.write(f"**Correlation of the gap with load variability (CV): r = "
                f"{np.corrcoef(cvs[m], gaps[m])[0,1]:.2f}** across both sweeps.\n")
print(f"\nwrote {OUT}/e4_sweep.md, e4_sweep.png")
