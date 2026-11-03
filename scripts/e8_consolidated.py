"""E8 -- THE consolidated comparison: static threshold vs dynamic (load-aware) threshold,
broken down by CONCURRENCY REGIME.

This experiment replaces the scattered E1/E3/E4 tables with one protocol, one results file and
one figure set, per the 5 Aug supervisor direction ("stop exploring, converge"). Everything a
reader needs to judge the claim is on the same page:

  * FOUR CONCURRENCY REGIMES as an explicit axis -- near-idle, moderate, saturated, and the
    bursty case that swings between them. Load is never set by hand: it emerges from the arrival
    process, so each policy's own thinking decisions feed back into the batch size it then sees.
  * FLOOR AND CEILING ON EVERY COMPARISON -- never-think, always-think, random-at-rate (the honest
    floor: same think-rate, no query information) and the hindsight oracle (the ceiling). A gap
    reported without them is uninterpretable.
  * A DENSELY SWEPT BASELINE. The fairness correction (results/CORRECTIONS.md S1) showed a coarse
    grid silently strands the static baseline. Both families are swept densely here -- the static
    grid is sampled from the sorted unique router scores BY INDEX, and the load-aware grid gets a
    comparable number of points -- so neither family is handicapped by resolution.
  * ALL FOUR METRICS TOGETHER: energy, accuracy, p95 latency, budget adherence, with paired CIs.

Three sub-experiments, in the order an operator meets the questions:

  E8a  ALLOCATION QUALITY. At a matched energy cap, which policy delivers most accuracy?
       Both families get their best feasible operating point -- symmetric, so the comparison is
       about allocation, not about tuning luck.
  E8b  THE SLO REVERSAL. Add a p95 latency target alongside the energy cap. The ranking flips.
  E8d  WHAT EARNS THE WIN. The full rule uses two things a static threshold does not -- live load,
       and per-query token counts. The latter is NOT observable before generating (the simulator
       reads it from the dataset; a live router cannot), so granting it to one side only is unfair
       AND undeployable. Each ingredient is added separately to see which carries the result.
  E8c  DEPLOYMENT. Tune offline on historical traces, deploy on held-out traces: does the policy
       actually respect the cap? (Includes the closed-loop controller that meters its own spend.)

Free -- no GPU. Runtime ~1 h single-core.
Out: results/e8/{CONSOLIDATED.md, fig1_frontiers.png, fig2_gap_by_regime.png, fig3_slo_reversal.png,
                 fig4_decomposition.png, e8_rows.json}
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH
from src.capfit import refine_to_bound

OUT = "results/e8"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]

N_TRACE  = 500                      # queries per trace
EVAL     = list(range(20))          # held-out evaluation traces
HIST     = [100 + i for i in range(5)]   # separate "historical" traces, for offline tuning (E8c)
CAPS     = [0.25, 0.40, 0.55, 0.70]      # fraction of the never->always energy span
SLO_MULT = [1.5, 3.0, 6.0]               # p95 targets, as multiples of the never-think p95
B_BOOT   = 20000

# ---------------------------------------------------------------- the concurrency axis
# Arrival processes chosen so that the offered load spans the measured J(L) curve: near-idle
# (where a token costs ~3.4 J) through saturated (where it costs ~0.1 J). The bursty regime has a
# LOW mean rate but swings between the two extremes -- that swing is what a load-aware policy can
# exploit and a static one cannot see.
REGIMES = {
    "near-idle":  dict(hi_rate=0.20, lo_rate=0.20, hi_s=30.0, lo_s=30.0),
    "moderate":   dict(hi_rate=0.80, lo_rate=0.80, hi_s=30.0, lo_s=30.0),
    "saturated":  dict(hi_rate=3.00, lo_rate=3.00, hi_s=30.0, lo_s=30.0),
    "bursty":     dict(hi_rate=6.00, lo_rate=0.03, hi_s=6.0,  lo_s=110.0),
}

# ---------------------------------------------------------------- policy families
FLIP = sorted([q for q in wl.qids if wl.dacc(q) > 0], key=lambda q: wl.dtok(q))

def mk_never(_p, _s):  return lambda L, q, c: False
def mk_always(_p, _s): return lambda L, q, c: True
def mk_static(tau, _s): return lambda L, q, c: SCORE[q] >= tau
def mk_load(lam, _s):   return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * wl.dtok(q)
def mk_random(rate, s):
    rng = np.random.default_rng(1000 + s)
    return lambda L, q, c: rng.random() < rate
def mk_oracle(k, _s):
    chosen = set(FLIP[:int(k)])          # hindsight: the queries thinking actually flips, cheapest first
    return lambda L, q, c: q in chosen

# --- the 2x2 decomposition: which ingredient actually earns the win? ----------------------------
# The full load-aware rule uses TWO things a plain threshold does not: the live load L, and the
# per-query token count dtok(q). dtok is NOT observable before generating -- the simulator can read
# it from the paired dataset, a live router cannot -- so granting it to one side only is both unfair
# and undeployable. These two extra families separate the ingredients:
#
#                  | token-blind                     | per-query tokens
#   load-blind     | static-tau (the incumbent)      | static-token (the field's token proxy)
#   load-aware     | load-aware-const (DEPLOYABLE)   | load-aware (needs oracle token counts)
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))          # 2370 tokens

def mk_static_tok(t, _s):   return lambda L, q, c: SCORE[q] > t * wl.dtok(q)
def mk_load_const(lam, _s): return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * DT_MEAN

# --- grids: every family must RESOLVE its own operating points, or the comparison is decided by
# grid resolution rather than by policy. The fairness correction (CORRECTIONS.md S1) fixed this for
# the static baseline; the same discipline has to apply to the load-aware families, and a naive
# log-spaced lambda grid does NOT satisfy it. Because 81.5% of router scores sit below 0.1, one
# 1.16x step in lambda can sweep the effective threshold across a large slice of the score mass --
# a dry run showed the load-aware family jumping straight from a 0.3% to a >5% think-rate, landing
# it at never-think under a cap it could actually have used. That would hand the win to the static
# baseline for exactly the wrong reason.
#
# So each family's grid is built as the IMAGE OF ITS OWN score-space grid under its price map:
#     load-aware-const : threshold = lam * J(L) * E[dtok]   ->  lam = score / (J(L) * E[dtok])
#     load-aware       : threshold = lam * J(L) * dtok(q)   ->  lam = (score/dtok) / J(L)
# evaluated across the measured range of L, then unioned with a wide log grid for coverage.
_uniq = np.unique([SCORE[q] for q in wl.qids])
_step = max(1, len(_uniq) // 180)
TAU_GRID  = np.unique(np.concatenate([[0.0], _uniq[::_step], _uniq[-3:], [1.01]]))
RATE_GRID = np.unique(np.concatenate([np.linspace(0.0, 0.30, 31), np.linspace(0.35, 1.0, 14)]))
ORAC_GRID = np.unique(np.linspace(0, len(FLIP), 41).astype(int))
# static-token thresholds live on score/dtok, sampled by INDEX of the sorted unique ratios -- the
# same recipe that fixed the fairness bug, so this baseline reaches its full think-rate range too.
_ratio = np.unique([SCORE[q] / max(wl.dtok(q), 1.0) for q in wl.qids])
TOKT_GRID = np.unique(np.concatenate([[0.0], _ratio[::max(1, len(_ratio) // 180)],
                                      _ratio[-3:], [_ratio[-1] * 1.01]]))

_LREF = (1, 4, 16, 48)                       # the concurrencies the regimes actually visit
_WIDE = np.logspace(-8, 4, 40)               # coverage outside the mapped band

def _price_grid(score_grid, scale_of_L, target=240):
    """Map a dense score-space grid into lambda space through the price at several loads."""
    cand = np.concatenate([np.asarray(score_grid, float) / max(scale_of_L(L), 1e-12) for L in _LREF])
    cand = np.unique(cand[np.isfinite(cand) & (cand > 0)])
    if len(cand) > target: cand = cand[:: max(1, len(cand) // target)]
    return np.unique(np.concatenate([cand, _WIDE]))

LAM_CONST_GRID = _price_grid(TAU_GRID,  lambda L: hw.jtok(L) * DT_MEAN)
LAM_TOK_GRID   = _price_grid(TOKT_GRID, lambda L: hw.jtok(L))

FAMILIES = {                               # name -> (grid, factory)
    "static-tau":       (TAU_GRID,  mk_static),
    "load-aware":       (LAM_TOK_GRID,   mk_load),
    "random@rate":      (RATE_GRID, mk_random),
    "oracle":           (ORAC_GRID, mk_oracle),
    "static-token":     (TOKT_GRID, mk_static_tok),
    "load-aware-const": (LAM_CONST_GRID, mk_load_const),
}
# How each family's parameter is refined at a cap (src/capfit.py). A grid dense in score space is
# not dense in energy, and a family with no point just below the cap would otherwise be scored on
# budget it was never offered -- which is what the published run did to load-aware-const.
SCALE = {"static-tau": "linear", "load-aware": "log", "random@rate": "linear",
         "oracle": "int", "static-token": "log", "load-aware-const": "log"}

CORE = ("static-tau", "load-aware-const", "random@rate", "oracle")   # what fig1 draws
DECOMP = ("static-tau", "static-token", "load-aware-const", "load-aware")
KEYS = ("energy_wh", "acc", "lat_p95", "think_rate", "load_mean", "load_cv", "util", "makespan")

# The simulator charges power only while the GPU is BUSY (L>0) and nothing during idle gaps. A real
# A100 with the weights resident still draws 37.2 W idle (measured in the E6 replay,
# results/validation/replay_summary.jsonl). That matters most in the near-idle regime, where the
# server spends much of the wall clock doing nothing -- so report both figures. It does not affect
# any policy COMPARISON within a regime beyond the second order: idle power is charged over the
# wall clock, and a policy that thinks more keeps the GPU busy longer, i.e. idles LESS.
IDLE_W = 37.2

def wall_energy(m):
    """Busy energy + measured idle draw over the idle fraction of the wall clock."""
    idle_s = m["makespan"].mean() * (1.0 - m["util"].mean())
    return m["energy_wh"].mean() + IDLE_W * idle_s / 3600.0


def run(regime, factory, param, seeds):
    """Per-seed metric arrays for one operating point. Identical traces across every policy."""
    out = {k: [] for k in KEYS}
    for s in seeds:
        r = simulate(make_trace(wl, N_TRACE, s, **regime), factory(param, s), wl, hw)
        for k in KEYS: out[k].append(r[k])
    return {k: np.asarray(v, float) for k, v in out.items()}


def boot_ci(d, b=B_BOOT, seed=7):
    """95% CI of the mean of a PAIRED per-trace difference (both policies saw the same traces)."""
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def best_feasible(points, cap, slo=None):
    """Highest MEAN accuracy among operating points meeting the constraints. None if stranded."""
    feas = [p for p in points
            if p["energy_wh"].mean() <= cap + 1e-9
            and (slo is None or p["lat_p95"].mean() <= slo + 1e-9)]
    return max(feas, key=lambda p: p["acc"].mean()) if feas else None


# ================================================================= sweep
t0 = time.time()
print(f"queries {len(wl.qids)} | J/token spread {max(hw.JTOK.values())/min(hw.JTOK.values()):.0f}x "
      f"| router: {len(_uniq)} distinct scores")
print(f"grids: static {len(TAU_GRID)}  load-aware {len(LAM_TOK_GRID)}  "
      f"load-aware-const {len(LAM_CONST_GRID)}  random {len(RATE_GRID)}  "
      f"oracle {len(ORAC_GRID)}  static-token {len(TOKT_GRID)}   | {len(EVAL)} eval traces "
      f"x {N_TRACE} queries | E[dtok] {DT_MEAN:.0f} tok\n")

SW, REF = {}, {}
for rname, rp in REGIMES.items():
    REF[rname] = {"never": run(rp, mk_never, None, EVAL), "always": run(rp, mk_always, None, EVAL)}
    SW[rname] = {}
    for fname, (grid, fac) in FAMILIES.items():
        SW[rname][fname] = [dict(param=float(v), **run(rp, fac, v, EVAL)) for v in grid]
    n, a = REF[rname]["never"], REF[rname]["always"]
    print(f"[{time.time()-t0:5.0f}s] {rname:10s} never {n['acc'].mean():.4f} @ {n['energy_wh'].mean():6.1f} Wh "
          f"(L {n['load_mean'].mean():4.1f}, CV {n['load_cv'].mean():.2f}, p95 {n['lat_p95'].mean():5.1f}s)  ->  "
          f"always {a['acc'].mean():.4f} @ {a['energy_wh'].mean():6.1f} Wh (L {a['load_mean'].mean():4.1f})")

# ================================================================= E8a + E8b
rows_a, rows_b, gapdata, slodata = [], [], {}, {}
for rname in REGIMES:
    rp = REGIMES[rname]
    nev, alw = REF[rname]["never"], REF[rname]["always"]
    e_lo, e_hi = nev["energy_wh"].mean(), alw["energy_wh"].mean()
    p95_base = nev["lat_p95"].mean()
    gapdata[rname], slodata[rname] = [], []

    for frac in CAPS:
        cap = e_lo + frac * (e_hi - e_lo)
        for f, (_, fac) in FAMILIES.items():       # let every family reach this cap, not just the
            refine_to_bound(SW[rname][f], cap,     # ones whose price map happens to land near it
                            lambda v, fac=fac: run(rp, fac, v, EVAL), scale=SCALE[f])
        sel = {f: best_feasible(SW[rname][f], cap) for f in FAMILIES}
        for f, p in sel.items():
            if p is None: continue
            rows_a.append(dict(regime=rname, frac=frac, cap=cap, policy=f, param=p["param"],
                               acc=p["acc"].mean(), wh=p["energy_wh"].mean(), think=p["think_rate"].mean(),
                               p95=p["lat_p95"].mean(), L=p["load_mean"].mean()))
        # The headline gap is the DEPLOYABLE policy's: load-aware with a constant dtok. The
        # oracle-token variant is reported alongside as an upper bound on what per-query token
        # knowledge could add, never as the claim.
        st, dep, orc = sel["static-tau"], sel["load-aware-const"], sel["load-aware"]
        if st is not None and dep is not None:
            d = (dep["acc"] - st["acc"]) * 100
            lo, hi = boot_ci(d)
            g = dict(frac=frac, cap=cap, gap=float(d.mean()), lo=lo, hi=hi, wins=float((d > 0).mean()))
            if orc is not None:
                do = (orc["acc"] - st["acc"]) * 100
                olo, ohi = boot_ci(do)
                g.update(gap_oracle_tok=float(do.mean()), lo_ot=olo, hi_ot=ohi)
            gapdata[rname].append(g)

    for mult in SLO_MULT + [None]:
        cap = e_lo + 0.40 * (e_hi - e_lo)                     # the 40% cap throughout E8b
        slo = None if mult is None else mult * p95_base
        for f in ("static-tau", "load-aware-const"):     # the latency target is a second boundary,
            if slo is not None:                          # and needs the same treatment as the cap
                refine_to_bound(SW[rname][f], slo,
                                lambda v, fac=FAMILIES[f][1]: run(rp, fac, v, EVAL),
                                metric="lat_p95", scale=SCALE[f])
        sel = {f: best_feasible(SW[rname][f], cap, slo) for f in ("static-tau", "load-aware-const")}
        st, la = sel["static-tau"], sel["load-aware-const"]
        lab = "unconstrained" if mult is None else f"{mult:g}x"
        if st is None or la is None:
            slodata[rname].append(dict(mult=lab, gap=np.nan, lo=np.nan, hi=np.nan)); continue
        d = (la["acc"] - st["acc"]) * 100
        lo, hi = boot_ci(d)
        slodata[rname].append(dict(mult=lab, slo=(np.nan if slo is None else slo), gap=float(d.mean()),
                                   lo=lo, hi=hi, st_acc=st["acc"].mean(), la_acc=la["acc"].mean(),
                                   st_p95=st["lat_p95"].mean(), la_p95=la["lat_p95"].mean()))
        rows_b.append(dict(regime=rname, slo_label=lab, st_acc=st["acc"].mean(), la_acc=la["acc"].mean(),
                           st_p95=st["lat_p95"].mean(), la_p95=la["lat_p95"].mean(), gap=float(d.mean()),
                           lo=lo, hi=hi))

# ================================================================= E8d -- what earns the win?
# At a matched cap, price each ingredient separately: per-query token knowledge (which a live
# router does NOT have) versus load-awareness (which it does). If load-awareness carries the
# result, the deployable rule keeps it; if token knowledge carries it, the result is a simulation
# artefact and must be reported as one.
rows_d, decomp = [], {}
for rname in REGIMES:
    nev, alw = REF[rname]["never"], REF[rname]["always"]
    decomp[rname] = {}
    for frac in (0.25, 0.40):
        cap = nev["energy_wh"].mean() + frac * (alw["energy_wh"].mean() - nev["energy_wh"].mean())
        sel = {f: best_feasible(SW[rname][f], cap) for f in DECOMP}
        base = sel["static-tau"]
        if base is None: continue
        for f in DECOMP:
            p = sel[f]
            if p is None: continue
            d = (p["acc"] - base["acc"]) * 100
            lo, hi = (0.0, 0.0) if f == "static-tau" else boot_ci(d)
            rec = dict(regime=rname, frac=frac, policy=f, acc=p["acc"].mean(), wh=p["energy_wh"].mean(),
                       think=p["think_rate"].mean(), p95=p["lat_p95"].mean(),
                       gap=float(d.mean()), lo=lo, hi=hi)
            rows_d.append(rec)
            decomp[rname].setdefault(frac, []).append(rec)

# ================================================================= E8c -- deployment
def pol_feedback(budget_wh, lam0=1e-3, kp=0.5):
    """Closed loop: lambda adapts on the meter, counting energy already COMMITTED by in-flight
    requests -- a think-decision's bill arrives long after the decision, so spend-only feedback
    over-thinks badly (76% vs 14% think-rate at the same cap; CORRECTIONS.md S4)."""
    st = {"lam": lam0}
    def decide(L, q, c): return SCORE[q] > st["lam"] * hw.jtok(L) * wl.dtok(q)
    def on_complete(c):
        f = c["n_done"] / N_TRACE
        if f < 0.05: return
        pace = ((c["energy_j"] + c["inflight_j"]) * WH) / max(budget_wh * f, 1e-9)
        st["lam"] = float(np.clip(st["lam"] * (pace ** kp), 1e-8, 1e4))
    return decide, on_complete

DEPLOY = ("static-tau", "load-aware", "load-aware-const", "random@rate")
rows_c = []
for rname, rp in REGIMES.items():
    nev, alw = REF[rname]["never"], REF[rname]["always"]
    e_lo, e_hi = nev["energy_wh"].mean(), alw["energy_wh"].mean()
    # the offline sweep depends only on (regime, family) -- do it once, reuse for every cap
    HSW = {f: [dict(param=float(v), **run(rp, FAMILIES[f][1], v, HIST)) for v in FAMILIES[f][0]]
           for f in DEPLOY}
    for frac in (0.25, 0.40):
        cap = e_lo + frac * (e_hi - e_lo)
        for fname in DEPLOY:
            grid, fac = FAMILIES[fname]
            refine_to_bound(HSW[fname], cap,                                       # same fix, on the
                            lambda v, fac=fac: run(rp, fac, v, HIST),              # historical sweep
                            scale=SCALE[fname])
            pick = best_feasible(HSW[fname], cap)                                  # tuned offline
            if pick is None: continue
            dep = run(rp, fac, pick["param"], EVAL)                                # deploy held-out
            rows_c.append(dict(regime=rname, frac=frac, cap=cap, policy=fname, param=pick["param"],
                               acc=dep["acc"].mean(), wh=dep["energy_wh"].mean(),
                               vs_cap=100 * (dep["energy_wh"].mean() - cap) / cap,
                               breach=float((dep["energy_wh"] > cap).mean()),
                               think=dep["think_rate"].mean(), p95=dep["lat_p95"].mean()))
        E, A, T, P = [], [], [], []
        for s in EVAL:                                                             # closed loop
            d, oc = pol_feedback(cap)
            r = simulate(make_trace(wl, N_TRACE, s, **rp), d, wl, hw, on_complete=oc)
            E.append(r["energy_wh"]); A.append(r["acc"]); T.append(r["think_rate"]); P.append(r["lat_p95"])
        E = np.asarray(E)
        rows_c.append(dict(regime=rname, frac=frac, cap=cap, policy="load-aware+feedback", param=np.nan,
                           acc=float(np.mean(A)), wh=float(E.mean()), vs_cap=100 * (E.mean() - cap) / cap,
                           breach=float((E > cap).mean()), think=float(np.mean(T)), p95=float(np.mean(P))))
    print(f"[{time.time()-t0:5.0f}s] E8c done: {rname}")

json.dump(dict(a=rows_a, b=rows_b, c=rows_c, d=rows_d,
               gap={k: v for k, v in gapdata.items()}, slo={k: v for k, v in slodata.items()}),
          open(f"{OUT}/e8_rows.json", "w"), indent=1, default=float)

# ================================================================= figures
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_STATIC, C_LOAD, C_ORACLE = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
C_POS, C_NEG, C_RAND = "#2a78d6", "#e34948", "#9a9a95"            # diverging pair; neutral floor
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURF})

def pareto(points):
    """Upper-left envelope: cheapest point achieving each accuracy level."""
    pts = sorted((p["energy_wh"].mean(), p["acc"].mean()) for p in points)
    out, best = [], -1
    for e, a in pts:
        if a > best + 1e-12: out.append((e, a)); best = a
    return np.array(out) if out else np.empty((0, 2))

# --- fig 1: the frontiers, one panel per concurrency regime -------------------
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
for ax, rname in zip(axes.ravel(), REGIMES):
    nev, alw = REF[rname]["never"], REF[rname]["always"]
    for fname, col, lw, z in (("oracle", C_ORACLE, 1.4, 2), ("random@rate", C_RAND, 1.4, 2),
                              ("static-tau", C_STATIC, 2.0, 3), ("load-aware-const", C_LOAD, 2.0, 4)):
        P = pareto(SW[rname][fname])
        if len(P): ax.plot(P[:, 0], P[:, 1], color=col, lw=lw, zorder=z,
                           label={"static-tau": "static threshold",
                                  "load-aware-const": "load-aware (deployable)",
                                  "random@rate": "random at rate (floor)",
                                  "oracle": "hindsight oracle (ceiling)"}[fname])
    for ref, mk in ((nev, "o"), (alw, "s")):
        ax.plot(ref["energy_wh"].mean(), ref["acc"].mean(), mk, ms=6, mfc=SURF, mec=INK, mew=1.4, zorder=5)
    ax.annotate("never-think", (nev["energy_wh"].mean(), nev["acc"].mean()), textcoords="offset points",
                xytext=(7, -9), color=MUTED, fontsize=8)
    ax.annotate("always-think", (alw["energy_wh"].mean(), alw["acc"].mean()), textcoords="offset points",
                xytext=(-8, 7), color=MUTED, fontsize=8, ha="right")
    for frac in (0.25, 0.40):
        cap = nev["energy_wh"].mean() + frac * (alw["energy_wh"].mean() - nev["energy_wh"].mean())
        ax.axvline(cap, color=GRID, lw=0.8, zorder=1)
        ax.annotate(f"{int(frac*100)}% cap", (cap, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(3, 4), color=MUTED, fontsize=7.5, rotation=90, va="bottom")
    ax.set_title(f"{rname}  ·  L {nev['load_mean'].mean():.1f}→{alw['load_mean'].mean():.1f}"
                 f"  ·  CV {nev['load_cv'].mean():.2f}", fontsize=9.5, color=INK, loc="left", pad=7)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.set_xlabel(f"energy (Wh, {N_TRACE} queries)"); ax.set_ylabel("accuracy")
h, lb = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lb, frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0.008, 0.955))
fig.suptitle("Energy–accuracy frontiers by concurrency regime  ·  Qwen3-8B, measured A100 curves, "
             f"{len(EVAL)} traces × {N_TRACE} queries", fontsize=10.5, color=INK, x=0.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.925)); fig.savefig(f"{OUT}/fig1_frontiers.png", dpi=200); plt.close(fig)

# --- fig 2: the gap, by regime and energy cap --------------------------------
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.5), sharey=True)
for ax, rname in zip(axes, REGIMES):
    g = gapdata[rname]
    x = np.arange(len(g)); v = [d["gap"] for d in g]
    ax.bar(x, v, width=0.62, color=[C_POS if t >= 0 else C_NEG for t in v], zorder=3)
    ax.errorbar(x, v, yerr=[[d["gap"] - d["lo"] for d in g], [d["hi"] - d["gap"] for d in g]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(d['frac']*100)}%" for d in g])
    ax.set_title(rname, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xlabel("energy cap (% of always-think)")
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
axes[0].set_ylabel("accuracy gap (pp)\nload-aware − static")
fig.suptitle("Where dynamic thresholding pays  ·  deployable load-aware vs a densely swept static "
             "threshold, matched energy, 95% paired CIs  ·  blue = dynamic wins, red = static", fontsize=10.5, color=INK, x=0.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig2_gap_by_regime.png", dpi=200); plt.close(fig)

# --- fig 3: the SLO reversal -------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.5), sharey=True)
for ax, rname in zip(axes, REGIMES):
    g = [d for d in slodata[rname] if np.isfinite(d["gap"])]
    x = np.arange(len(g)); v = [d["gap"] for d in g]
    ax.bar(x, v, width=0.62, color=[C_POS if t >= 0 else C_NEG for t in v], zorder=3)
    ax.errorbar(x, v, yerr=[[d["gap"] - d["lo"] for d in g], [d["hi"] - d["gap"] for d in g]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([d["mult"] for d in g], fontsize=8)
    ax.set_title(rname, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xlabel("p95 latency target (× never-think)")
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
axes[0].set_ylabel("accuracy gap (pp)\nload-aware − static")
fig.suptitle("The reversal: under a tight tail-latency target the static threshold wins  ·  "
             "40% energy cap throughout", fontsize=10.5, color=INK, x=0.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig3_slo_reversal.png", dpi=200); plt.close(fig)

# --- fig 4: what earns the win -- load-awareness or token knowledge? ---------
PRETTY = {"static-token": "+ tokens\n(oracle)", "load-aware-const": "+ load\n(deployable)",
          "load-aware": "+ both\n(oracle)"}
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.9), sharey=True)
for ax, rname in zip(axes, REGIMES):
    g = [r for r in decomp[rname].get(0.40, []) if r["policy"] != "static-tau"]
    x = np.arange(len(g)); v = [r["gap"] for r in g]
    ax.bar(x, v, width=0.6, color=[C_POS if t >= 0 else C_NEG for t in v], zorder=3)
    ax.errorbar(x, v, yerr=[[r["gap"] - r["lo"] for r in g], [r["hi"] - r["gap"] for r in g]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[r["policy"]] for r in g], fontsize=7.5)
    ax.set_title(rname, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
axes[0].set_ylabel("accuracy gap (pp)\nvs static threshold")
fig.suptitle("What earns the win: live load, or per-query token knowledge?  ·  40% energy cap, "
             "each ingredient added to the static threshold", fontsize=10.5, color=INK, x=0.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(f"{OUT}/fig4_decomposition.png", dpi=200); plt.close(fig)

# ================================================================= write-up
def tab(hdr, rows):
    return "\n".join(["|" + "|".join(hdr) + "|", "|" + "|".join("--" for _ in hdr) + "|"] + rows) + "\n"

L = []
L.append("# E8 — the consolidated comparison: static vs dynamic thresholding, by concurrency regime\n")
L.append(f"*One protocol, one results file. Supersedes the headline numbers in `results/e1`, `results/e3`, "
         f"`results/e4` and the C2 column of `results/e5` (see `results/CORRECTIONS.md`). "
         f"{len(EVAL)} held-out traces × {N_TRACE} queries per operating point; identical arrival times and "
         f"identical queries for every policy; accuracy read at matched energy. "
         f"Static grid {len(TAU_GRID)} thresholds, load-aware grids {len(LAM_CONST_GRID)} / "
         f"{len(LAM_TOK_GRID)} — every family's grid is the image of its own dense score-space grid "
         f"under its price map, so none is handicapped by grid resolution.*\n")

L.append("## 0. The concurrency axis\n")
L.append("Load is not set by hand — it emerges from arrivals versus service, so each policy's own "
         "thinking decisions feed back into the batch size it then sees. `L` is the time-weighted "
         "in-service concurrency over busy periods; it is reported for never-think → always-think to "
         "show the span each regime exercises on the measured J(L) curve.\n")
rows = []
for rname, rp in REGIMES.items():
    n, a = REF[rname]["never"], REF[rname]["always"]
    rows.append(f"|{rname}|{rp['hi_rate']:g} / {rp['lo_rate']:g} req/s|{n['load_mean'].mean():.1f} → "
                f"{a['load_mean'].mean():.1f}|{n['load_cv'].mean():.2f}|{n['util'].mean():.2f}|"
                f"{n['energy_wh'].mean():.1f}|{a['energy_wh'].mean():.1f}|{a['energy_wh'].mean()/n['energy_wh'].mean():.1f}×|"
                f"{wall_energy(n):.1f}|{wall_energy(a):.1f}|{n['acc'].mean():.4f}|{a['acc'].mean():.4f}|")
L.append(tab(["regime", "arrival (hi/lo)", "concurrency L never→always", "load CV", "utilisation",
              "never (Wh)", "always (Wh)", "energy ratio", "never +idle (Wh)", "always +idle (Wh)",
              "never acc", "always acc"], rows))
_ni, _sat = REF["near-idle"]["never"], REF["saturated"]["never"]
L.append(f"\n**The same {N_TRACE} questions, answered identically, cost "
         f"{_ni['energy_wh'].mean() / _sat['energy_wh'].mean():.1f}× more energy on a near-idle server "
         f"than on a saturated one** ({_ni['energy_wh'].mean():.0f} vs {_sat['energy_wh'].mean():.0f} Wh, "
         f"never-think in both). Nothing about the work changed — only the queue. This is the same "
         f"physics as the 42× J/token spread, stated at the level an operator bills at.\n")
L.append(f"The `+idle` columns add the **measured 37.2 W idle draw** of an A100 with the weights "
         f"resident (from the E6 replay, `results/validation/replay_summary.jsonl`) over the idle "
         f"fraction of the wall clock. The simulator itself charges power only while the GPU is busy, "
         f"which understates near-idle regimes most. It does not distort the policy comparisons: idle "
         f"power is charged over wall-clock time, and a policy that thinks *more* keeps the GPU busy "
         f"longer, so it idles *less* — correcting for it makes thinking look cheaper, not dearer.\n")

L.append("## 1. E8a — allocation quality at matched energy\n")
L.append("Every policy family gets its best feasible operating point at the cap, so the comparison is "
         "about *allocation*, not tuning luck. The floor (random at the same think-rate) and the ceiling "
         "(hindsight oracle) bracket what is achievable.\n")
for rname in REGIMES:
    rows = []
    for frac in CAPS:
        sel = [r for r in rows_a if r["regime"] == rname and r["frac"] == frac]
        if not sel: continue
        by = {r["policy"]: r for r in sel}
        g = next((d for d in gapdata[rname] if d["frac"] == frac), None)
        cell = lambda f: (f"{by[f]['acc']:.4f} ({by[f]['think']*100:.0f}%)" if f in by else "—")
        rows.append(f"|{int(frac*100)}% ({by[list(by)[0]]['cap']:.0f} Wh)|{cell('random@rate')}|"
                    f"{cell('static-tau')}|{cell('load-aware-const')}|{cell('load-aware')}|"
                    f"{cell('oracle')}|"
                    + (f"**{g['gap']:+.2f}pp** [{g['lo']:+.2f}, {g['hi']:+.2f}]|" if g else "—|"))
    L.append(f"**{rname}**\n")
    L.append(tab(["energy cap", "random@rate (floor)", "static threshold",
                  "load-aware **(deployable)**", "load-aware + oracle tokens",
                  "oracle (ceiling)", "deployable − static, 95% CI"], rows))
L.append("Cells are `accuracy (think-rate)`. The CI is a bootstrap over the paired per-trace difference "
         "— both policies ran on the same traces, so the difference is paired.\n")
L.append("**The headline column is `load-aware (deployable)`** — the rule with a *constant* Δtokens, "
         "which is what a live router can actually compute and what `src/router.py` ships. "
         "`load-aware + oracle tokens` is the same rule given per-query ground-truth token counts; it "
         "is an upper bound on what better token estimation could buy, not a claim. §3 separates the "
         "two ingredients directly.\n")
L.append("![frontiers](fig1_frontiers.png)\n\n![gap by regime](fig2_gap_by_regime.png)\n")

L.append("## 2. E8b — the reversal under a tail-latency target\n")
L.append("Same deployable load-aware rule as §1. Energy cap held at 40% throughout; a p95 SLO is added "
         "alongside it, expressed as a multiple of the never-think p95 in that regime.\n")
rows = []
for r in rows_b:
    rows.append(f"|{r['regime']}|{r['slo_label']}|{r['st_acc']:.4f}|{r['st_p95']:.0f} s|{r['la_acc']:.4f}|"
                f"{r['la_p95']:.0f} s|{r['gap']:+.2f}pp [{r['lo']:+.2f}, {r['hi']:+.2f}]|")
L.append(tab(["regime", "p95 target", "static acc", "static p95", "load-aware acc", "load-aware p95",
              "gap, 95% CI"], rows))
L.append("![slo reversal](fig3_slo_reversal.png)\n")

L.append("## 3. E8d — what earns the win: live load, or per-query token knowledge?\n")
L.append("The full rule `score > λ·J(L)·Δtokens(q)` uses two things a plain threshold does not: the "
         "**live load** `L`, and the **per-query token count** `Δtokens(q)`. Δtokens is *not "
         "observable before generating* — the simulator reads it from the paired dataset, a live "
         "router cannot — so a comparison that grants it to one side only is both unfair to the "
         "baseline and undeployable. Adding each ingredient separately to the static threshold "
         "settles which one carries the result, and therefore whether the deployable router keeps it.\n")
L.append(f"`load-aware-const` replaces Δtokens(q) with the constant E[Δtokens] = {DT_MEAN:.0f} tokens — "
         "load-aware, per-query token-blind. It is exactly what `src/router.py` ships by default.\n")
rows = []
for r in rows_d:
    lbl = {"static-tau": "static threshold (baseline)", "static-token": "+ per-query tokens (token proxy)",
           "load-aware-const": "+ live load — **deployable**", "load-aware": "+ both (oracle tokens)"}[r["policy"]]
    rows.append(f"|{r['regime']}|{int(r['frac']*100)}%|{lbl}|{r['acc']:.4f}|{r['think']*100:.0f}%|"
                f"{r['p95']:.0f} s|" + ("—|" if r["policy"] == "static-tau"
                                        else f"{r['gap']:+.2f}pp [{r['lo']:+.2f}, {r['hi']:+.2f}]|"))
L.append(tab(["regime", "cap", "rule", "accuracy", "think-rate", "p95", "gap vs static, 95% CI"], rows))
L.append("![decomposition](fig4_decomposition.png)\n")

L.append("## 4. E8c — deployment: tuned offline, does it hold the cap?\n")
L.append(f"Each policy is tuned on {len(HIST)} *historical* traces and then deployed on the "
         f"{len(EVAL)} held-out ones — the realistic procedure. `vs cap` is the realised energy against the "
         f"authorised budget; `breach` is the fraction of traces that exceeded it. "
         f"`load-aware+feedback` needs no offline tuning: it adapts λ against the meter, counting energy "
         f"already committed by in-flight requests.\n")
rows = []
for r in rows_c:
    rows.append(f"|{r['regime']}|{int(r['frac']*100)}%|{r['policy']}|{r['acc']:.4f}|{r['wh']:.1f}|"
                f"{r['vs_cap']:+.1f}%|{r['breach']*100:.0f}%|{r['think']*100:.0f}%|{r['p95']:.0f} s|")
L.append(tab(["regime", "cap", "policy", "accuracy", "energy (Wh)", "vs cap", "breach rate",
              "think-rate", "p95"], rows))

L.append("\n*Repro: `python scripts/e8_consolidated.py` (CPU, a few minutes). Raw per-row numbers in "
         "`e8_rows.json`.*\n")
open(f"{OUT}/CONSOLIDATED.md", "w", encoding="utf-8").write("\n".join(L))
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/CONSOLIDATED.md + 4 figures")
