"""E15 -- why the closed-loop controller blew its budget live, and what fixes it.

E12 ran `BudgetController` against a real energy meter on an A100 and it FAILED: +36.7% over budget
on bursty traffic and +134.4% at saturation, worse than the offline tuning it was meant to repair.
The pacing trace (`results/closedloop/pacing_bursty.jsonl`) shows the mechanism precisely:

    n_done=10   spent 0.34 Wh   target 1.86 Wh   pace 0.305   ->  lambda 6.50e-04
    n_done=29   spent 1.00 Wh   target 5.38 Wh   pace 0.301   ->  lambda 1.00e-08   (the FLOOR)

Within 29 of 200 requests the controller had driven lambda five orders of magnitude down to its
clamp, i.e. think-on-everything; the second quartile of arrivals ran at an 82% think-rate, and the
run finished 37% over budget with lambda back up at 2.8e-02, far too late.

THE DIAGNOSIS. The pacing signal is `committed / (budget * n_done/N)`, which assumes energy should
accrue in proportion to COMPLETED REQUESTS. It does not, because completion order is correlated with
cost: short no-think requests finish first, so early in any window the realised spend is *always*
below the linear target regardless of whether the policy is on track. The controller reads that
structural surplus as headroom and loosens. It is a selection effect, not a measurement error, and
no amount of metering accuracy fixes it -- E12's meter agreed with Zeus to 0.0%.

Three candidate fixes, all cheap and all testable here:

  RATE-LIMIT   cap how far lambda may move per update, so no burst of cheap completions can walk it
               to the clamp before the expensive consequences land.
  TIME-PACED   pace against elapsed time against the known workload window instead of completed
               requests. An operator knows the window; completion order is what is biased.
  ASYMMETRIC   tighten fast, loosen slowly. Over-spending breaches a cap; under-spending merely
               leaves accuracy on the table, so the two errors do not deserve equal gain.

Free -- no GPU. Out: results/e15/{CONTROLLER_FIX.md, fig_controller.png}
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH

OUT = "results/e15"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
N, SEEDS = 500, list(range(12))
REGIMES = {"bursty":    dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0,  lo_s=110.0),
           "saturated": dict(hi_rate=3.0, lo_rate=3.0,  hi_s=30.0, lo_s=30.0)}
t0 = time.time()


def make_controller(budget_j, span_s, lam0=1e-3, kp=0.5, rate_limit=None,
                    time_paced=False, asym=False, warmup=0.05):
    """Returns (decide, on_complete). `rate_limit` caps the per-update multiplicative step."""
    st = {"lam": lam0}
    def decide(L, q, c):
        return SCORE[q] > st["lam"] * hw.jtok(L) * DT_MEAN
    def on_complete(c):
        frac = (c["t"] / span_s) if time_paced else (c["n_done"] / N)
        if frac < warmup or frac <= 0:
            return
        pace = (c["energy_j"] + c["inflight_j"]) / max(budget_j * frac, 1e-9)
        k = kp
        if asym and pace < 1.0:                 # loosening is the dangerous direction
            k = kp * 0.25
        step = pace ** k
        if rate_limit:
            step = float(np.clip(step, 1.0 / rate_limit, rate_limit))
        st["lam"] = float(np.clip(st["lam"] * step, 1e-8, 1e4))
    return decide, on_complete


VARIANTS = {
    "as-shipped (E12)":            dict(),
    "+ rate-limit (1.5x/update)":  dict(rate_limit=1.5),
    "+ time-paced":                dict(time_paced=True),
    "+ asymmetric gain":           dict(asym=True),
    "all three":                   dict(rate_limit=1.5, time_paced=True, asym=True),
}

rows = []
for rname, reg in REGIMES.items():
    # the authorised budget: 1.5x never-think, matching E12's construction
    nev = np.mean([simulate(make_trace(wl, N, s, **reg), lambda L, q, c: False, wl, hw)["energy_wh"]
                   for s in SEEDS])
    budget_wh = 1.5 * nev
    spans = {s: max(t for t, _ in make_trace(wl, N, s, **reg)) for s in SEEDS}
    print(f"\n=== {rname}: never {nev:.1f} Wh -> budget {budget_wh:.1f} Wh")
    for vname, kw in VARIANTS.items():
        E, A, T, brk = [], [], [], []
        for s in SEEDS:
            tr = make_trace(wl, N, s, **reg)
            d, oc = make_controller(budget_wh * 3600, spans[s], **kw)
            r = simulate(tr, d, wl, hw, on_complete=oc)
            E.append(r["energy_wh"]); A.append(r["acc"]); T.append(r["think_rate"])
            brk.append(r["energy_wh"] > budget_wh)
        E = np.array(E)
        row = dict(regime=rname, variant=vname, budget_wh=float(budget_wh),
                   energy_wh=float(E.mean()), vs_budget=float(100 * (E.mean() - budget_wh) / budget_wh),
                   worst=float(100 * (E.max() - budget_wh) / budget_wh),
                   breach=float(np.mean(brk)), acc=float(np.mean(A)), think=float(np.mean(T)))
        rows.append(row)
        print(f"  {vname:28s} {row['energy_wh']:6.1f} Wh  {row['vs_budget']:+7.1f}%  "
              f"(worst {row['worst']:+7.1f}%)  breach {row['breach']*100:3.0f}%  "
              f"acc {row['acc']:.4f}  think {row['think']*100:4.1f}%")

json.dump(rows, open(f"{OUT}/controller_fix.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_OK, C_BAD = "#2a78d6", "#e34948"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, len(REGIMES), figsize=(5.6 * len(REGIMES), 4.0), sharey=True)
for ax, rname in zip(np.atleast_1d(axes), REGIMES):
    g = [r for r in rows if r["regime"] == rname]
    y = np.arange(len(g)); v = [r["vs_budget"] for r in g]
    ax.barh(y, v, height=.62, color=[C_BAD if t > 0 else C_OK for t in v], zorder=3)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r["variant"] for r in g], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("energy vs authorised budget (%)")
    ax.set_title(f"{rname} — budget {g[0]['budget_wh']:.0f} Wh", fontsize=9.5, color=INK,
                 loc="left", pad=6)
    ax.grid(True, axis="x", color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
fig.suptitle("E15 — the closed-loop controller's pacing bias, and what fixes it  ·  "
             "red = over budget, blue = under", fontsize=10.5, color=INK, x=.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(f"{OUT}/fig_controller.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- write-up
def tab(h, rs): return "\n".join(["|" + "|".join(h) + "|", "|" + "|".join("--" for _ in h) + "|"] + rs) + "\n"
L = ["# E15 — why the closed-loop controller overspent, and what the simulator can and cannot say\n"]
L.append("E12 ran `BudgetController` against a real NVML energy meter on an A100 and it **failed** — "
         "+36.7% over budget on bursty traffic, +134.4% at saturation, worse than the offline tuning "
         "it was built to repair. The meter was not at fault: it agreed with Zeus to 0.0% on every "
         "run. The controller was.\n")
L.append("## The mechanism, from E12's own pacing trace\n")
L.append("```\nn_done=10   spent 0.34 Wh   target 1.86 Wh   pace 0.305  ->  lambda 6.50e-04\n"
         "n_done=29   spent 1.00 Wh   target 5.38 Wh   pace 0.301  ->  lambda 1.00e-08  (the FLOOR)\n```\n")
L.append("Within 29 of 200 requests the controller drove λ five orders of magnitude down to its clamp "
         "— think on everything. The second quartile of arrivals ran at an **82% think-rate**, and the "
         "run ended 37% over budget with λ back at 2.8e-02, far too late to matter.\n")
L.append("**The diagnosis is a selection effect, not a measurement error.** The pacing signal is "
         "`committed / (budget × n_done/N)`, which assumes energy accrues in proportion to *completed "
         "requests*. It does not: completion order is correlated with cost, because short no-think "
         "requests finish first. Early in any window the realised spend is therefore **structurally** "
         "below the linear target, whether or not the policy is on track. The controller reads that as "
         "headroom and loosens. No improvement in metering can fix it.\n")
L.append("## Three candidate fixes, tested in simulation\n")
L.append(f"Same protocol as E8: {len(SEEDS)} traces × {N} queries, budget = 1.5× never-think, matching "
         f"E12's construction. `worst` is the worst single trace — a cap that holds on average but "
         f"breaches half the time is not a cap.\n")
for rname in REGIMES:
    rs = [f"|{r['variant']}|{r['energy_wh']:.1f}|{r['vs_budget']:+.1f}%|{r['worst']:+.1f}%|"
          f"{r['breach']*100:.0f}%|{r['acc']:.4f}|{r['think']*100:.1f}%|"
          for r in rows if r["regime"] == rname]
    L.append(f"**{rname}** — authorised {[r for r in rows if r['regime']==rname][0]['budget_wh']:.1f} Wh\n")
    L.append(tab(["controller", "energy (Wh)", "vs budget", "worst trace", "breach rate",
                  "accuracy", "think-rate"], rs))
L.append("![controller](fig_controller.png)\n")
L.append("## Reading — including what this experiment failed to establish\n")
L.append("**The simulator does not reproduce the live failure, so it cannot validate the diagnosis.** "
         "As-shipped, the controller *under*-spends on bursty traffic here (−3.7% on average) where "
         "live it overspent by +36.7%. Whatever drove λ to its floor within 29 requests on the real "
         "server is not in this model — which is consistent with E9 and E13: the simulator "
         "under-predicts energy by 43% at load and does not model the preemption E13 measured, and "
         "those are precisely the conditions under which the controller broke. **The completion-order "
         "bias remains the best available explanation of the live trace, but it is not confirmed "
         "here.**\n")
L.append("**Time-pacing does not help, which is evidence against the simplest version of that story.** "
         "If the completion-order bias were the whole mechanism, pacing against a clock would remove "
         "it. It does not — in either regime. So either the bias is not dominant in the simulator's "
         "dynamics, or it is not the dominant term live either.\n")
L.append("**Only the asymmetric gain helps, and only partly.** Making the loosening gain a quarter of "
         "the tightening gain is the one variant that improves adherence in both regimes: on bursty "
         "it cuts the breach rate from 42% to 17% and the worst trace from +18.1% to +10.2%; at "
         "saturation it pulls the mean from +39.9% to +2.3%. But the worst trace at saturation is "
         "still +46.6% and 42% of traces still breach. **A cap that holds on average and breaches on "
         "four traces in ten is not a cap.**\n")
L.append("**What this licenses saying.** That the shipped controller overspends is measured (E12). "
         "That an asymmetric gain is the most promising of three candidate repairs is a simulation "
         "result. That it *fixes* the live failure is **not** established, and the honest next step is "
         "to re-run E12 with it on real hardware rather than to report a fix that has only ever been "
         "tested against a model known to be wrong in the relevant regime.\n")
L.append("\n*Repro: `python scripts/e15_controller_fix.py` (CPU).*\n")
open(f"{OUT}/CONTROLLER_FIX.md", "w", encoding="utf-8").write("\n".join(L))
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/CONTROLLER_FIX.md")
