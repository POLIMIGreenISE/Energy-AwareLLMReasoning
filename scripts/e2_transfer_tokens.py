"""E2 rebuilt -- can an energy budget be held when the controller does not know each query's token cost?

WHY A REBUILD. The thesis table tab:calibration-transfer (results/e2/calibration_transfer.md, log
results/e2xfer.log) was produced by a script that was never committed, on "steady" and "heavy" regimes
whose parameters were not recorded. Its budget is also mislabelled there: the E2 code defines it as
never + 0.35 x (always - never), not as 35% of always-reasoning energy (e2spec.log is only consistent
with the span definition). And both of its cost models were given each query's TRUE token count, so it
says nothing about the deployable rule, which must estimate it.

WHAT THIS DOES. E2's open-loop procedure (scripts/e2_token_proxy_controller.py, plan_lambda) rebuilt in
the shared simulator (src/serving_sim.py) on the four documented E8 regimes, so it is reproducible and
consistent with the rest of Chapter 5:
  * budget  B = never + f x (always - never), f in {0.35, 0.60}; anchors are means over the 20 E8
    evaluation traces;
  * plan    lambda by bisection on the 5 E8 historical traces (seeds 100-104). A fixed-price controller
    predicts energy in closed form as C x tokens (E2's belief, 60 steps); the load-aware controller
    replays the historical traces through the simulator with its own decision rule (14 steps).
    Historical tokens are logged, so planning uses the true tokens of the historical queries; only the
    DECISION uses the controller's token estimate;
  * deploy  the frozen lambda on the 20 E8 evaluation traces, and measure energy against B;
  * sweep   E2's nine prices C (J/token) and the load-aware model, each under three token models:
            true (as in E2), constant E[dtok] = 2370 (the shipped rule), and the out-of-fold text
            estimate of results/e10/dtok_pred.json.

GUARD: never/always anchors must reproduce results/e8/CONSOLIDATED.md section 0 (to its 0.1 Wh
rounding), i.e. same simulator, regimes and traces.

Free -- no GPU; regimes run in parallel.
    python scripts/e2_transfer_tokens.py          -> results/e2/TRANSFER_TOKENS.md, e2_transfer_tokens.json
    python scripts/e2_transfer_tokens.py --smoke  2 eval traces, 1 historical, no guard, no files
"""
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH

SMOKE = "--smoke" in sys.argv
OUT = "results/e2"

hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
N_TRACE = 500
EVAL = list(range(2)) if SMOKE else list(range(20))          # E8's evaluation traces
HIST = [100] if SMOKE else [100 + i for i in range(5)]      # E8's historical traces
FRACS = [0.35, 0.60]                                         # E2's budgets (span definition)
C_SWEEP = [3.42, 2.40, 1.70, 1.20, 0.865, 0.60, 0.454, 0.244, 0.13]   # E2's nine prices

REGIMES = {                                                  # verbatim from e8_consolidated.py
    "near-idle":  dict(hi_rate=0.20, lo_rate=0.20, hi_s=30.0, lo_s=30.0),
    "moderate":   dict(hi_rate=0.80, lo_rate=0.80, hi_s=30.0, lo_s=30.0),
    "saturated":  dict(hi_rate=3.00, lo_rate=3.00, hi_s=30.0, lo_s=30.0),
    "bursty":     dict(hi_rate=6.00, lo_rate=0.03, hi_s=6.0,  lo_s=110.0),
}
# MEASURED: results/e8/CONSOLIDATED.md section 0 (never, always Wh)
PUBLISHED_ANCHORS = {"near-idle": (100.7, 172.8), "moderate": (39.9, 56.7),
                     "saturated": (12.0, 47.3), "bursty": (81.6, 417.1)}

DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
DHAT = json.load(open("results/e10/dtok_pred.json"))["dhat"]
TOKEN_MODELS = {
    "true":      lambda q: wl.dtok(q),
    "constant":  lambda q: DT_MEAN,
    "estimated": lambda q: max(DHAT[q], 1.0),
}


def traces(regime, seeds):
    return [make_trace(wl, N_TRACE, s, **REGIMES[regime]) for s in seeds]


def fixed_rule(lam, c, est):   return lambda L, q, ctx: SCORE[q] > lam * c * est(q)
def load_rule(lam, est):       return lambda L, q, ctx: SCORE[q] > lam * hw.jtok(L) * est(q)


def mean_energy(trs, decide_factory):
    return float(np.mean([simulate(tr, decide_factory(), wl, hw)["energy_wh"] for tr in trs]))


def plan_fixed(hist, c, est, budget):
    """E2's closed-form belief: energy = C x (direct tokens + reasoning tokens of queries it reasons on)."""
    qs = [q for tr in hist for _, q in tr]
    s = np.array([SCORE[q] for q in qs])
    base = np.array([wl.tok_nt[q] for q in qs])
    dtrue = np.array([wl.dtok(q) for q in qs])
    dest = np.array([est(q) for q in qs])
    lo, hi = 1e-10, 1e6
    for _ in range(60):
        lam = (lo * hi) ** 0.5
        think = s > lam * c * dest
        pred = ((base + think * dtrue) * c).sum() * WH / len(hist)
        if pred > budget: lo = lam
        else: hi = lam
    return (lo * hi) ** 0.5


def plan_load(hist, est, budget):
    """The load-aware controller has the physics: replay history through the simulator."""
    lo, hi = 1e-10, 1e6
    for _ in range(14):
        lam = (lo * hi) ** 0.5
        if mean_energy(hist, lambda: load_rule(lam, est)) > budget: lo = lam
        else: hi = lam
    return (lo * hi) ** 0.5


def deploy(evals, factory, budget):
    res = [simulate(tr, factory(), wl, hw) for tr in evals]
    e = np.array([r["energy_wh"] for r in res])
    return dict(vs_budget=float(100 * (e.mean() - budget) / budget),
                vs_budget_sd=float(np.std(100 * (e - budget) / budget)),
                breach=float((e > budget).mean()),
                acc=float(np.mean([r["acc"] for r in res])),
                think=float(np.mean([r["think_rate"] for r in res])))


def run_regime(regime):
    t0 = time.time()
    evals, hist = traces(regime, EVAL), traces(regime, HIST)
    never = mean_energy(evals, lambda: (lambda L, q, c: False))
    always = mean_energy(evals, lambda: (lambda L, q, c: True))
    rows = []
    for f in FRACS:
        budget = never + f * (always - never)
        for tname, est in TOKEN_MODELS.items():
            for c in C_SWEEP:
                lam = plan_fixed(hist, c, est, budget)
                rows.append(dict(regime=regime, frac=f, budget_wh=budget, tokens=tname, model="fixed",
                                 C=c, lam=lam, **deploy(evals, lambda: fixed_rule(lam, c, est), budget)))
            lam = plan_load(hist, est, budget)
            rows.append(dict(regime=regime, frac=f, budget_wh=budget, tokens=tname, model="load-aware",
                             C=None, lam=lam, **deploy(evals, lambda: load_rule(lam, est), budget)))
    return regime, never, always, rows, time.time() - t0


def main():
    t0 = time.time()
    print(f"{len(REGIMES)} regimes x {len(FRACS)} budgets x {len(TOKEN_MODELS)} token models x "
          f"({len(C_SWEEP)} prices + load-aware) | eval {len(EVAL)} hist {len(HIST)}"
          f"{'  [SMOKE]' if SMOKE else ''}", flush=True)
    with ProcessPoolExecutor(max_workers=len(REGIMES)) as ex:
        results = list(ex.map(run_regime, REGIMES))
    print(f"[{time.time()-t0:5.0f}s] done: " + ", ".join(f"{r[0]} {r[4]:.0f}s" for r in results), flush=True)

    anchors = {r[0]: (r[1], r[2]) for r in results}
    rows = [x for r in results for x in r[3]]

    if SMOKE:
        for x in rows:
            if x["frac"] == 0.35 and (x["model"] == "load-aware" or x["C"] in (1.70, 0.13)):
                print("  %-9s %-9s %-10s C=%-5s %+7.1f%%" % (x["regime"], x["tokens"], x["model"], x["C"], x["vs_budget"]))
        print("smoke ok"); return

    bad = [(k, anchors[k], v) for k, v in PUBLISHED_ANCHORS.items()
           if abs(anchors[k][0] - v[0]) > 0.051 or abs(anchors[k][1] - v[1]) > 0.051]
    if bad:
        for b in bad: print("GUARD FAILED:", b)
        raise SystemExit("anchors do not reproduce CONSOLIDATED.md section 0")
    print(f"[{time.time()-t0:5.0f}s] guard passed: never/always anchors reproduce E8 section 0", flush=True)

    json.dump(dict(anchors=anchors, rows=rows), open(os.path.join(OUT, "e2_transfer_tokens.json"), "w"), indent=1)

    L = ["# E2 rebuilt: holding a budget without knowing each query's token cost", "",
         "E2's open-loop plan-then-deploy procedure in the shared simulator on the four E8 regimes. Budget = "
         "never + f x (always - never). lambda is bisected on 5 historical traces and frozen; the table gives "
         "energy actually spent on the 20 evaluation traces against the budget (mean, %). Token models: "
         "**true** per-query counts (as in the original E2), **constant** E[dtok] = 2370 (the shipped rule), "
         "**estimated** out-of-fold text prediction (results/e10). Anchors reproduce CONSOLIDATED.md section 0.",
         ""]
    for f in FRACS:
        for tname in TOKEN_MODELS:
            L += [f"## Budget {f:.0%} of the span, {tname} token counts", "",
                  "|regime|budget (Wh)|" + "|".join(f"C={c:g}" for c in C_SWEEP) + "|C closest|load-aware|breach|",
                  "|--|--|" + "|".join("--" for _ in C_SWEEP) + "|--|--|--|"]
            for rg in REGIMES:
                rr = [x for x in rows if x["regime"] == rg and x["frac"] == f and x["tokens"] == tname]
                fx = [x for x in rr if x["model"] == "fixed"]
                la = next(x for x in rr if x["model"] == "load-aware")
                best = min(fx, key=lambda x: abs(x["vs_budget"]))
                L.append(f"|{rg}|{rr[0]['budget_wh']:.1f}|" + "|".join(f"{x['vs_budget']:+.0f}" for x in fx)
                         + f"|{best['C']:g} ({best['vs_budget']:+.1f})|**{la['vs_budget']:+.1f}%**|{100*la['breach']:.0f}%|")
            L.append("")
    L += ["## Load-aware controller only: effect of the token model", "",
          "|budget|regime|true tokens|constant|estimated|", "|--|--|--|--|--|"]
    for f in FRACS:
        for rg in REGIMES:
            v = {t: next(x for x in rows if x["regime"] == rg and x["frac"] == f and x["tokens"] == t
                         and x["model"] == "load-aware") for t in TOKEN_MODELS}
            L.append(f"|{f:.0%}|{rg}|" + "|".join(
                f"{v[t]['vs_budget']:+.1f}% (acc {v[t]['acc']:.4f}, reasons {100*v[t]['think']:.1f}%)"
                for t in TOKEN_MODELS) + "|")
    L += ["", f"*Repro: `python scripts/e2_transfer_tokens.py` (CPU, {time.time()-t0:.0f} s on this machine).*", ""]
    open(os.path.join(OUT, "TRANSFER_TOKENS.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"[{time.time()-t0:5.0f}s] wrote {OUT}/TRANSFER_TOKENS.md and e2_transfer_tokens.json")


if __name__ == "__main__":
    main()
