"""E5 -- do the RQ4 conclusions survive the simulator's modelling assumptions?

E1/E2/E4 compose MEASURED hardware curves with a SIMULATED queue. Four assumptions are doing
work, and each is an extrapolation we cannot currently justify from the measurements:

  kv_alpha    KV-cache growth. Curves were measured at l_out=256; thinking requests average
              ~2,700 tokens. kv_alpha>0 prices long generations up by (1 + a*tokens/1000).
  prefill_j   prompt processing energy, currently omitted entirely.
  max_batch   the KV-limited in-service cap (48 assumed).
  interp      how the gaps between measured batch sizes are filled (linear vs step).

For every variant we re-test the TWO headline conclusions, and report whether each still holds:
  C1 (E2a): a token-blind controller misses its energy budget while the load-aware one hits it.
  C2 (E1a): load-aware allocation beats a fixed threshold at MATCHED energy.

A conclusion that survives every variant can be stated without the simulator caveat; one that
does not must be reported as assumption-dependent. Free (no GPU).
Out: results/e5/sensitivity.md
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH

OUT = "results/e5"; os.makedirs(OUT, exist_ok=True)
wl = Workload()
SCORE = wl.scores["ModernBERT"]
REGIME = dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0)   # bimodal
N_TRACE, N_SEEDS = 500, 5
_sv = np.array(sorted(SCORE.values()))
TAU_GRID = np.unique(np.concatenate([[0.0], np.quantile(_sv, np.linspace(0.002, 0.998, 16)), [1.01]]))
LAM_GRID = np.logspace(-7, 3, 16)

VARIANTS = [
    ("baseline",                dict()),
    ("KV growth a=0.15",        dict(kv_alpha=0.15)),
    ("KV growth a=0.40",        dict(kv_alpha=0.40)),
    ("prefill 200 J/req",       dict(prefill_j=200.0)),
    ("prefill 800 J/req",       dict(prefill_j=800.0)),
    ("max_batch=24",            dict(max_batch=24)),
    ("max_batch=96",            dict(max_batch=96)),
    ("step interpolation",      dict(interp="step")),
    ("KV .4 + prefill 800",     dict(kv_alpha=0.40, prefill_j=800.0)),
]

def build(v):
    hw = Hardware(kv_alpha=v.get("kv_alpha", 0.0), interp=v.get("interp", "linear"))
    kw = dict(max_batch=v.get("max_batch", 48), prefill_j=v.get("prefill_j", 0.0))
    return hw, kw

def run(hw, kw, dec, seeds):
    E, A = [], []
    for s in seeds:
        r = simulate(make_trace(wl, N_TRACE, s, **REGIME), dec, wl, hw, **kw)
        E.append(r["energy_wh"]); A.append(r["acc"])
    return float(np.mean(E)), float(np.mean(A))

def frontier(hw, kw, grid, fn, seeds):
    pts = [run(hw, kw, fn(v), seeds) for v in grid]
    best, out = -1, []
    for e, a in sorted(pts):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out

def acc_at(front, e):
    f = [a for pe, a in front if pe <= e + 1e-9]
    return max(f) if f else np.nan

def pol_static(t): return lambda L, q, c: SCORE[q] >= t
def pol_load(lam): return lambda L, q, c: SCORE[q] > lam * hw_cur.jtok(L) * wl.dtok(q)

rows = []
print(f"{'variant':22} {'never Wh':>9} {'always Wh':>10} | {'C1 token-blind':>15} {'C1 load-aware':>14} "
      f"| {'C2 static':>10} {'C2 load':>9} {'C2 gap':>8}")
for name, v in VARIANTS:
    hw, kw = build(v)
    hw_cur = hw                                   # used by pol_load
    seeds = list(range(N_SEEDS))
    e_nev, a_nev = run(hw, kw, lambda L, q, c: False, seeds)
    e_alw, a_alw = run(hw, kw, lambda L, q, c: True, seeds)
    B = e_nev + 0.40 * (e_alw - e_nev)

    # ---- C1: plan lambda on a historical trace with each cost model, deploy, measure budget miss
    hist = [100, 101]
    qs = [q for _, q in make_trace(wl, N_TRACE, 100, **REGIME)]
    dtoks = np.array([wl.dtok(q) for q in qs]); base = np.array([wl.tok_nt[q] for q in qs])
    s_arr = np.array([SCORE[q] for q in qs])
    c_const = hw.jtok(8)                          # a plausible operator calibration point
    lo, hi = 1e-10, 1e6
    for _ in range(60):                           # token-blind plans in closed form (its belief)
        lam = (lo * hi) ** 0.5
        think = s_arr > lam * (c_const * dtoks)
        pred = ((base + think * dtoks) * c_const).sum() * WH
        if pred > B: lo = lam
        else: hi = lam
    lam_tb = (lo * hi) ** 0.5
    lo, hi = 1e-10, 1e6
    for _ in range(14):                           # load-aware plans by replaying the true physics
        lam = (lo * hi) ** 0.5
        pred = run(hw, kw, pol_load(lam), hist)[0]
        if pred > B: lo = lam
        else: hi = lam
    lam_la = (lo * hi) ** 0.5
    e_tb = run(hw, kw, (lambda L, q, c: SCORE[q] > lam_tb * c_const * wl.dtok(q)), seeds)[0]
    e_la = run(hw, kw, pol_load(lam_la), seeds)[0]
    miss_tb = 100 * (e_tb - B) / B; miss_la = 100 * (e_la - B) / B

    # ---- C2: accuracy at matched energy (mean gap over the overlapping frontier range)
    f_st = frontier(hw, kw, TAU_GRID, pol_static, seeds)
    f_la = frontier(hw, kw, LAM_GRID, pol_load, seeds)
    lo_e = max(f_st[0][0], f_la[0][0]); hi_e = min(f_st[-1][0], f_la[-1][0])
    if hi_e > lo_e:
        g = np.linspace(lo_e, hi_e, 20)
        a_st = float(np.nanmean([acc_at(f_st, e) for e in g]))
        a_la = float(np.nanmean([acc_at(f_la, e) for e in g]))
        gap = a_la - a_st
    else:
        a_st = a_la = gap = np.nan
    rows.append((name, e_nev, e_alw, miss_tb, miss_la, a_st, a_la, gap))
    print(f"{name:22} {e_nev:9.1f} {e_alw:10.1f} | {miss_tb:+14.1f}% {miss_la:+13.1f}% "
          f"| {a_st:10.4f} {a_la:9.4f} {gap*100:+7.2f}pp")

c1_ok = all(abs(r[4]) < 15 and abs(r[3]) > 25 for r in rows)
c2_ok = all(r[7] > 0 for r in rows if r[7] == r[7])
print(f"\nC1 (token proxy misses the budget, load-aware hits it): "
      f"{'HOLDS in every variant' if c1_ok else 'ASSUMPTION-DEPENDENT'}")
print(f"C2 (load-aware beats a fixed threshold at matched energy): "
      f"{'HOLDS in every variant' if c2_ok else 'ASSUMPTION-DEPENDENT'}")

with open(f"{OUT}/sensitivity.md", "w", encoding="utf-8") as f:
    f.write("# E5 — sensitivity of the RQ4 conclusions to the simulator's assumptions\n\n")
    f.write(f"Bimodal regime, {N_SEEDS} held-out traces of {N_TRACE} queries, budget = 40% of "
            "always-think. **C1** = a token-blind controller misses its energy budget while the "
            "load-aware one hits it. **C2** = load-aware beats a fixed threshold at matched energy "
            "(mean gap across the overlapping frontier range).\n\n")
    f.write("|variant|never (Wh)|always (Wh)|C1 token-blind miss|C1 load-aware miss|C2 static|C2 load-aware|C2 gap|\n")
    f.write("|--|--|--|--|--|--|--|--|\n")
    for n, en, ea, mt, ml, ast_, ala, g in rows:
        f.write(f"|{n}|{en:.1f}|{ea:.1f}|{mt:+.1f}%|{ml:+.1f}%|{ast_:.4f}|{ala:.4f}|{g*100:+.2f}pp|\n")
    f.write(f"\n**C1: {'holds in every variant' if c1_ok else 'ASSUMPTION-DEPENDENT'}. "
            f"C2: {'holds in every variant' if c2_ok else 'ASSUMPTION-DEPENDENT'}.**\n")
print(f"\nwrote {OUT}/sensitivity.md")
