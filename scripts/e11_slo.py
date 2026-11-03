"""E11 x E8b -- does the load-aware advantage on real bursty traffic survive a tail-latency target?

E8b (results/e8/CONSOLIDATED.md section 2) showed that on the synthetic bursty regime the deployable
load-aware rule's lead over a static threshold disappears at a p95 target of six times the
never-reasoning p95 and reverses below it. E11 (results/e11/REAL_TRACE.md) found the arrival shape the
rule needs in real code-completion traffic, which is interactive. The thesis leaves the combination
open: does the gain on real code-completion arrivals survive a latency target?

Protocol: E11 exactly (same traces, same 12 evaluation windows x 500 queries, same parameter grids,
same energy-cap selection), plus E8b's latency constraint. At each energy cap, each family's best
operating point is the highest mean accuracy among points meeting BOTH the cap and the p95 target;
targets are multiples of that arrival process's never-reasoning p95. Only the two policies compared
are swept (static threshold, deployable load-aware rule).

GUARD: the unconstrained selection must reproduce e11_rows.json's published gaps and CIs exactly, or
the script stops -- otherwise this would not be the same experiment with one constraint added.

Free -- no GPU; the four arrival processes run in parallel.
    python scripts/e11_slo.py            full run  -> results/e11/SLO.md, e11_slo_rows.json
    python scripts/e11_slo.py --smoke    tiny grids, 2 traces, no guard, no files
"""
import os, sys, csv, json, time, datetime as dt
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.serving_sim import Hardware, Workload, make_trace, simulate
from src.capfit import refine_to_bound

SMOKE = "--smoke" in sys.argv
OUT = "results/e11"
TRACE_DIR = os.environ.get("AZURE_TRACE_DIR", "data/azure")

# ---------------------------------------------------------------- verbatim from e11_real_trace.py
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))

N_TRACE, EVAL = 500, list(range(12))
CAPS = [0.25, 0.40, 0.55, 0.70]
B_BOOT = 20000
SLO_MULT = [1.5, 3.0, 6.0]                  # verbatim from e8_consolidated.py


def load_azure(name):
    path = os.path.join(TRACE_DIR, f"AzureLLMInferenceTrace_{name}.csv")
    ts = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts.append(dt.datetime.fromisoformat(r["TIMESTAMP"][:26]).timestamp())
    t = np.sort(np.asarray(ts, float))
    return t - t[0]


def dilate_to(t, target_rate):
    return t * (len(t) / (t[-1] - t[0]) / target_rate)


def thin_to(t, target_rate, rng):
    p = min(1.0, target_rate * (t[-1] - t[0]) / len(t))
    return t[rng.random(len(t)) < p]


def real_trace(t_all, n, seed, target_rate, mode="dilate"):
    rng = np.random.default_rng(seed)
    kept = dilate_to(t_all, target_rate) if mode == "dilate" else thin_to(t_all, target_rate, rng)
    if len(kept) <= n:
        raise RuntimeError(f"trace too short after {mode}: {len(kept)} <= {n}")
    i = int(rng.integers(0, len(kept) - n))
    arr = kept[i:i + n] - kept[i]
    qs = [wl.qids[j] for j in rng.integers(0, len(wl.qids), n)]
    return list(zip(arr.tolist(), qs))


def mk_never(_p, _s):       return lambda L, q, c: False
def mk_always(_p, _s):      return lambda L, q, c: True
def mk_static(tau, _s):     return lambda L, q, c: SCORE[q] >= tau
def mk_load_const(lam, _s): return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * DT_MEAN


_uniq = np.unique([SCORE[q] for q in wl.qids])
TAU_GRID = np.unique(np.concatenate([[0.0], _uniq[:: max(1, len(_uniq) // 180)], _uniq[-3:], [1.01]]))
_cand = np.concatenate([TAU_GRID / max(hw.jtok(L) * DT_MEAN, 1e-12) for L in (1, 4, 16, 48)])
_cand = np.unique(_cand[np.isfinite(_cand) & (_cand > 0)])
LAM_GRID = np.unique(np.concatenate([_cand[:: max(1, len(_cand) // 240)], np.logspace(-8, 4, 40)]))

FAMILIES = {"static-tau": (TAU_GRID, mk_static), "load-aware": (LAM_GRID, mk_load_const)}
SCALE = {"static-tau": "linear", "load-aware": "log"}   # how each parameter is refined at a boundary
KEYS = ("energy_wh", "acc", "lat_p95", "think_rate", "load_mean")

REGIME_NAMES = ["Poisson @0.8 (control)", "Azure conv @0.8", "Azure code @0.8", "Azure code @0.8 (thinned)"]


def regimes():
    conv, code = load_azure("conv"), load_azure("code")
    return {
        "Poisson @0.8 (control)":    dict(kind="poisson", rate=0.8),
        "Azure conv @0.8":           dict(kind="real", t=conv, rate=0.8, mode="dilate"),
        "Azure code @0.8":           dict(kind="real", t=code, rate=0.8, mode="dilate"),
        "Azure code @0.8 (thinned)": dict(kind="real", t=code, rate=0.8, mode="thin"),
    }


def trace_for(rp, seed):
    if rp["kind"] == "poisson":
        return make_trace(wl, N_TRACE, seed, hi_rate=rp["rate"], lo_rate=rp["rate"],
                          hi_s=30.0, lo_s=30.0)
    return real_trace(rp["t"], N_TRACE, seed, rp["rate"], rp.get("mode", "dilate"))


def boot(d, b=B_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    return tuple(float(x) for x in np.percentile(
        rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1), [2.5, 97.5]))


# ---------------------------------------------------------------- the one change: a p95 constraint
def best(pts, cap, slo=None):
    f = [p for p in pts if p["energy_wh"].mean() <= cap + 1e-9
         and (slo is None or p["lat_p95"].mean() <= slo + 1e-9)]
    return max(f, key=lambda p: p["acc"].mean()) if f else None


def sweep_regime(rname):
    """Runs in a worker process: never/always anchors plus both families' full grids."""
    rp = regimes()[rname]
    seeds = EVAL[:2] if SMOKE else EVAL

    def run(fac, v):
        out = {k: [] for k in KEYS}
        for s in seeds:
            r = simulate(trace_for(rp, s), fac(v, s), wl, hw)
            for k in KEYS:
                out[k].append(r[k])
        return {k: np.asarray(x, float) for k, x in out.items()}

    t0 = time.time()
    nev, alw = run(mk_never, None), run(mk_always, None)
    sw = {}
    for f, (grid, fac) in FAMILIES.items():
        g = grid[::25] if SMOKE else grid
        sw[f] = [dict(param=float(v), **run(fac, v)) for v in g]
    return rname, nev, alw, sw, time.time() - t0


def main():
    t0 = time.time()
    print(f"grids: static {len(TAU_GRID)}  load-aware {len(LAM_GRID)} | {len(EVAL)} traces x {N_TRACE}"
          f"{'  [SMOKE]' if SMOKE else ''}", flush=True)
    with ProcessPoolExecutor(max_workers=len(REGIME_NAMES)) as ex:
        results = list(ex.map(sweep_regime, REGIME_NAMES))
    print(f"[{time.time()-t0:5.0f}s] sweeps done: " + ", ".join(f"{r[0]} {r[4]:.0f}s" for r in results),
          flush=True)

    published = None if SMOKE else json.load(open(os.path.join(OUT, "e11_rows.json")))["gaps"]
    table, guard_bad = [], []
    for rname, nev, alw, sw, _ in results:
        rp = regimes()[rname]
        e_lo, e_hi = nev["energy_wh"].mean(), alw["energy_wh"].mean()
        p95_base = float(nev["lat_p95"].mean())

        def run_here(fac, v, rp=rp):
            out = {k: [] for k in KEYS}
            for s in (EVAL[:2] if SMOKE else EVAL):
                r = simulate(trace_for(rp, s), fac(v, s), wl, hw)
                for k in KEYS:
                    out[k].append(r[k])
            return {k: np.asarray(x, float) for k, x in out.items()}

        for ci, frac in enumerate(CAPS):
            cap = e_lo + frac * (e_hi - e_lo)
            for mult in [None] + SLO_MULT:
                slo = None if mult is None else mult * p95_base
                # Both constraints are boundaries a grid can straddle, so both need refining, for
                # both families (src/capfit.py, results/CORRECTIONS.md).
                for f, (_, fac) in FAMILIES.items():
                    refine_to_bound(sw[f], cap, lambda v, fac=fac: run_here(fac, v), scale=SCALE[f])
                    if slo is not None:
                        refine_to_bound(sw[f], slo, lambda v, fac=fac: run_here(fac, v),
                                        metric="lat_p95", scale=SCALE[f])
                st, la = best(sw["static-tau"], cap, slo), best(sw["load-aware"], cap, slo)
                row = dict(regime=rname, frac=frac, mult=("none" if mult is None else f"{mult:g}x"),
                           p95_never=p95_base, slo=slo, stranded=(st is None or la is None))
                if not row["stranded"]:
                    d = (la["acc"] - st["acc"]) * 100
                    lo, hi = boot(d)
                    row.update(gap=float(d.mean()), lo=lo, hi=hi,
                               st_acc=float(st["acc"].mean()), la_acc=float(la["acc"].mean()),
                               st_p95=float(st["lat_p95"].mean()), la_p95=float(la["lat_p95"].mean()),
                               st_think=float(st["think_rate"].mean()), la_think=float(la["think_rate"].mean()),
                               st_wh=float(st["energy_wh"].mean()), la_wh=float(la["energy_wh"].mean()))
                table.append(row)
                if mult is None and published is not None:
                    pub = published[rname][ci]
                    same = (not row["stranded"] and abs(pub["frac"] - frac) < 1e-12
                            and all(abs(pub[k] - row[k]) < 1e-9 for k in ("gap", "lo", "hi")))
                    if not same:
                        guard_bad.append((rname, frac, pub, {k: row.get(k) for k in ("gap", "lo", "hi")}))

    if SMOKE:
        for r in table:
            if r["frac"] == 0.40:
                print("  %-26s %-5s %s" % (r["regime"], r["mult"],
                      "stranded" if r["stranded"] else "gap %+.2f [%+.2f, %+.2f]  p95 st %.0f la %.0f"
                      % (r["gap"], r["lo"], r["hi"], r["st_p95"], r["la_p95"])))
        print(f"[{time.time()-t0:5.0f}s] smoke ok")
        return

    if guard_bad:
        for g in guard_bad:
            print("GUARD FAILED:", g)
        raise SystemExit("unconstrained selection does not reproduce REAL_TRACE.md -- not the same experiment")
    print(f"[{time.time()-t0:5.0f}s] guard passed: unconstrained gaps reproduce REAL_TRACE.md exactly", flush=True)

    json.dump(table, open(os.path.join(OUT, "e11_slo_rows.json"), "w"), indent=1)

    def fmt(r):
        if r["stranded"]:
            return "|%s|%s|%s|stranded|||||" % (r["regime"], r["mult"], "-" if r["slo"] is None else "%.0f s" % r["slo"])
        sig = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
        return ("|%s|%s|%s|%.4f|%.0f s|%.1f%%|%.4f|%.0f s|%.1f%%|%+.2f [%+.2f, %+.2f]%s|" % (
            r["regime"], r["mult"], "-" if r["slo"] is None else "%.0f s" % r["slo"],
            r["st_acc"], r["st_p95"], 100 * r["st_think"], r["la_acc"], r["la_p95"], 100 * r["la_think"],
            r["gap"], r["lo"], r["hi"], sig))

    L = ["# E11 x E8b: production arrival traces under a tail-latency target", "",
         "E11's protocol exactly (same traces, 12 windows x 500 queries, same grids, same selection), with "
         "E8b's p95 constraint added: at each energy cap, each policy's best operating point must meet the "
         "cap **and** the p95 target. Targets are multiples of the never-reasoning p95 of that arrival "
         "process. `*` marks a 95% paired bootstrap interval that excludes zero. The unconstrained rows "
         "reproduce `REAL_TRACE.md` exactly (checked in the script).", "",
         "## Never-reasoning p95 by arrival process", "",
         "|arrival process|never-reasoning p95|1.5x|3x|6x|", "|--|--|--|--|--|"]
    for rname in REGIME_NAMES:
        b = next(r["p95_never"] for r in table if r["regime"] == rname)
        L.append("|%s|%.1f s|%.0f s|%.0f s|%.0f s|" % (rname, b, 1.5 * b, 3 * b, 6 * b))
    for frac in CAPS:
        L += ["", f"## Energy cap {frac:.0%}", "",
              "|arrival process|p95 target|target|static acc|static p95|static reasons|load-aware acc|"
              "load-aware p95|load-aware reasons|gap (pp), 95% CI|",
              "|--|--|--|--|--|--|--|--|--|--|"]
        for rname in REGIME_NAMES:
            for r in table:
                if r["regime"] == rname and r["frac"] == frac:
                    L.append(fmt(r))
    L += ["", f"*Repro: `python scripts/e11_slo.py` (CPU, {time.time()-t0:.0f} s on this machine).*", ""]
    open(os.path.join(OUT, "SLO.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"[{time.time()-t0:5.0f}s] wrote {OUT}/SLO.md and e11_slo_rows.json")


if __name__ == "__main__":
    main()
