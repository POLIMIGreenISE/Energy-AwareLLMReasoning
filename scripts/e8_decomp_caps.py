"""E8d at every cap -- which ingredient earns the advantage at the 55% and 70% energy caps?

E8d (results/e8/CONSOLIDATED.md section 3) separated the load-aware rule's two ingredients, live load
and per-query token cost, only at the 25% and 40% caps. The near-idle regime is where the deployable
rule wins at the two LOOSER caps (55%, 70%), so the earlier statement that "wherever the two
ingredients were separated, the accuracy came from token cost" leaves exactly that case untested.
This runs the same decomposition at all four caps.

Protocol: E8 exactly -- same regimes, 20 evaluation traces x 500 queries, same factories, same
price-mapped grids, same best-feasible selection and paired bootstrap. Only the four decomposition
families are swept.

GUARDS (the script stops if either fails):
  1. at 25% and 40%, all four policies reproduce e8_rows.json "d" (gap, lo, hi) exactly;
  2. at 55% and 70%, "+ load" (the deployable rule minus the static threshold) reproduces the E8a
     headline gaps in e8_rows.json "gap" exactly -- an independent published check on the new caps.

Free -- no GPU; (regime, family) sweeps run in parallel.
    python scripts/e8_decomp_caps.py           -> results/e8/DECOMP_CAPS.md, e8_decomp_caps.json
    python scripts/e8_decomp_caps.py --smoke   tiny grids, 2 traces, no guards, no files
"""
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.serving_sim import Hardware, Workload, make_trace, simulate
from src.capfit import refine_to_bound

SMOKE = "--smoke" in sys.argv
OUT = "results/e8"

# ---------------------------------------------------------------- verbatim from e8_consolidated.py
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
N_TRACE = 500
EVAL = list(range(20))
CAPS = [0.25, 0.40, 0.55, 0.70]
B_BOOT = 20000

REGIMES = {
    "near-idle":  dict(hi_rate=0.20, lo_rate=0.20, hi_s=30.0, lo_s=30.0),
    "moderate":   dict(hi_rate=0.80, lo_rate=0.80, hi_s=30.0, lo_s=30.0),
    "saturated":  dict(hi_rate=3.00, lo_rate=3.00, hi_s=30.0, lo_s=30.0),
    "bursty":     dict(hi_rate=6.00, lo_rate=0.03, hi_s=6.0,  lo_s=110.0),
}

def mk_never(_p, _s):  return lambda L, q, c: False
def mk_always(_p, _s): return lambda L, q, c: True
def mk_static(tau, _s): return lambda L, q, c: SCORE[q] >= tau
def mk_load(lam, _s):   return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * wl.dtok(q)

DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))

def mk_static_tok(t, _s):   return lambda L, q, c: SCORE[q] > t * wl.dtok(q)
def mk_load_const(lam, _s): return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * DT_MEAN

_uniq = np.unique([SCORE[q] for q in wl.qids])
_step = max(1, len(_uniq) // 180)
TAU_GRID = np.unique(np.concatenate([[0.0], _uniq[::_step], _uniq[-3:], [1.01]]))
_ratio = np.unique([SCORE[q] / max(wl.dtok(q), 1.0) for q in wl.qids])
TOKT_GRID = np.unique(np.concatenate([[0.0], _ratio[::max(1, len(_ratio) // 180)],
                                      _ratio[-3:], [_ratio[-1] * 1.01]]))
_LREF = (1, 4, 16, 48)
_WIDE = np.logspace(-8, 4, 40)

def _price_grid(score_grid, scale_of_L, target=240):
    cand = np.concatenate([np.asarray(score_grid, float) / max(scale_of_L(L), 1e-12) for L in _LREF])
    cand = np.unique(cand[np.isfinite(cand) & (cand > 0)])
    if len(cand) > target: cand = cand[:: max(1, len(cand) // target)]
    return np.unique(np.concatenate([cand, _WIDE]))

LAM_CONST_GRID = _price_grid(TAU_GRID,  lambda L: hw.jtok(L) * DT_MEAN)
LAM_TOK_GRID   = _price_grid(TOKT_GRID, lambda L: hw.jtok(L))

FAMILIES = {
    "static-tau":       (TAU_GRID,       mk_static),
    "load-aware":       (LAM_TOK_GRID,   mk_load),
    "static-token":     (TOKT_GRID,      mk_static_tok),
    "load-aware-const": (LAM_CONST_GRID, mk_load_const),
}
SCALE = {"static-tau": "linear", "load-aware": "log", "static-token": "log",
         "load-aware-const": "log"}                      # how each parameter is refined at a cap
DECOMP = ("static-tau", "static-token", "load-aware-const", "load-aware")
LABEL = {"static-tau": "static threshold (baseline)", "static-token": "+ per-query tokens",
         "load-aware-const": "+ live load (deployable)", "load-aware": "+ both"}
KEYS = ("energy_wh", "acc", "lat_p95", "think_rate", "load_mean")


def run(regime, factory, param, seeds):
    out = {k: [] for k in KEYS}
    for s in seeds:
        r = simulate(make_trace(wl, N_TRACE, s, **regime), factory(param, s), wl, hw)
        for k in KEYS: out[k].append(r[k])
    return {k: np.asarray(v, float) for k, v in out.items()}


def boot_ci(d, b=B_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def best_feasible(points, cap):
    feas = [p for p in points if p["energy_wh"].mean() <= cap + 1e-9]
    return max(feas, key=lambda p: p["acc"].mean()) if feas else None


# ---------------------------------------------------------------- parallel sweep
SEEDS = EVAL[:2] if SMOKE else EVAL


def task(args):
    rname, fname = args
    rp = REGIMES[rname]
    if fname in ("never", "always"):
        return rname, fname, run(rp, mk_never if fname == "never" else mk_always, None, SEEDS)
    grid, fac = FAMILIES[fname]
    g = grid[::30] if SMOKE else grid
    return rname, fname, [dict(param=float(v), **run(rp, fac, v, SEEDS)) for v in g]


def main():
    t0 = time.time()
    print(f"grids: static {len(TAU_GRID)}  static-token {len(TOKT_GRID)}  load-aware-const "
          f"{len(LAM_CONST_GRID)}  load-aware {len(LAM_TOK_GRID)} | {len(SEEDS)} traces x {N_TRACE}"
          f"{'  [SMOKE]' if SMOKE else ''}", flush=True)
    jobs = [(r, f) for r in REGIMES for f in ("never", "always") + DECOMP]
    SW = {r: {} for r in REGIMES}
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, (os.cpu_count() or 4) - 2))) as ex:
        for rname, fname, res in ex.map(task, jobs):
            SW[rname][fname] = res
    print(f"[{time.time()-t0:5.0f}s] sweeps done", flush=True)

    rows = []
    for rname in REGIMES:
        nev, alw = SW[rname]["never"], SW[rname]["always"]
        e_lo, e_hi = nev["energy_wh"].mean(), alw["energy_wh"].mean()
        for frac in CAPS:
            cap = e_lo + frac * (e_hi - e_lo)
            for f in DECOMP:                       # every family must be able to reach the cap; a
                fac = FAMILIES[f][1]               # grid dense in score space is not dense in energy
                refine_to_bound(SW[rname][f], cap,
                                lambda v, fac=fac: run(REGIMES[rname], fac, v, SEEDS),
                                scale=SCALE[f])
            sel = {f: best_feasible(SW[rname][f], cap) for f in DECOMP}
            base = sel["static-tau"]
            if base is None:
                continue
            for f in DECOMP:
                p = sel[f]
                if p is None:
                    rows.append(dict(regime=rname, frac=frac, policy=f, stranded=True)); continue
                d = (p["acc"] - base["acc"]) * 100
                lo, hi = (0.0, 0.0) if f == "static-tau" else boot_ci(d)
                rows.append(dict(regime=rname, frac=frac, policy=f, stranded=False, acc=float(p["acc"].mean()),
                                 wh=float(p["energy_wh"].mean()), think=float(p["think_rate"].mean()),
                                 p95=float(p["lat_p95"].mean()), gap=float(d.mean()), lo=lo, hi=hi))

    if SMOKE:
        for r in rows:
            if r["frac"] in (0.55, 0.70) and not r["stranded"]:
                print("  %-10s %.0f%% %-18s %+.2f [%+.2f, %+.2f]" % (r["regime"], 100*r["frac"], r["policy"], r["gap"], r["lo"], r["hi"]))
        print(f"[{time.time()-t0:5.0f}s] smoke ok"); return

    # ---- guards
    pub = json.load(open(os.path.join(OUT, "e8_rows.json")))
    bad = []
    for d in pub["d"]:
        r = next((x for x in rows if x["regime"] == d["regime"] and abs(x["frac"] - d["frac"]) < 1e-12
                  and x["policy"] == d["policy"]), None)
        if r is None or r["stranded"] or any(abs(r[k] - d[k]) > 1e-9 for k in ("gap", "lo", "hi")):
            bad.append(("E8d", d["regime"], d["frac"], d["policy"], d["gap"], None if r is None else r.get("gap")))
    for rname, gl in pub["gap"].items():
        for g in gl:
            if g["frac"] not in (0.55, 0.70):
                continue
            r = next(x for x in rows if x["regime"] == rname and abs(x["frac"] - g["frac"]) < 1e-12
                     and x["policy"] == "load-aware-const")
            if r["stranded"] or any(abs(r[k] - g[k]) > 1e-9 for k in ("gap", "lo", "hi")):
                bad.append(("E8a", rname, g["frac"], g["gap"], r.get("gap")))
    if bad:
        for b in bad: print("GUARD FAILED:", b)
        raise SystemExit("does not reproduce e8_rows.json -- not the same experiment")
    print(f"[{time.time()-t0:5.0f}s] guards passed: 25%/40% reproduce E8d, 55%/70% load ingredient reproduces E8a",
          flush=True)

    json.dump(rows, open(os.path.join(OUT, "e8_decomp_caps.json"), "w"), indent=1)
    L = ["# E8d at every cap: which ingredient earns the advantage", "",
         "E8's protocol exactly; each ingredient of the rule added separately to a densely swept static "
         "threshold, at all four energy caps. `*` marks a 95% paired bootstrap interval excluding zero. "
         "The 25% and 40% rows reproduce `CONSOLIDATED.md` section 3 exactly, and the `+ live load` rows "
         "at 55% and 70% reproduce section 1 exactly (both checked in the script).", ""]
    for rname in REGIMES:
        L += [f"## {rname}", "", "|cap|policy|accuracy|reasons|p95|gap vs static (pp), 95% CI|", "|--|--|--|--|--|--|"]
        for r in rows:
            if r["regime"] != rname: continue
            if r["stranded"]:
                L.append(f"|{r['frac']:.0%}|{LABEL[r['policy']]}|stranded||||"); continue
            ci = "baseline" if r["policy"] == "static-tau" else "%+.2f [%+.2f, %+.2f]%s" % (
                r["gap"], r["lo"], r["hi"], " *" if (r["lo"] > 0 or r["hi"] < 0) else "")
            L.append(f"|{r['frac']:.0%}|{LABEL[r['policy']]}|{r['acc']:.4f}|{100*r['think']:.1f}%|{r['p95']:.0f} s|{ci}|")
        L.append("")
    L += [f"*Repro: `python scripts/e8_decomp_caps.py` (CPU, {time.time()-t0:.0f} s on this machine).*", ""]
    open(os.path.join(OUT, "DECOMP_CAPS.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"[{time.time()-t0:5.0f}s] wrote {OUT}/DECOMP_CAPS.md and e8_decomp_caps.json")


if __name__ == "__main__":
    main()
