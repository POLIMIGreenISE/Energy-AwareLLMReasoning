"""E2 -- Does the TOKEN cost proxy MISLEAD an energy-budgeted controller?

RQ1 showed token count does not predict energy under batched serving (J/token falls ~43x from
batch 1 to 64). E2 asks the consequence: an operator running a hybrid-reasoning model under an
ENERGY BUDGET must estimate "what will it cost to let this request think?". The field's universal
proxy says cost = c * tokens (load-blind). Our measurement says cost = J(L) * tokens (load-aware).

Same controller, same trace, same physics -- only the controller's BELIEF about cost differs:
  token-blind : cost_est = c * dtokens          (c calibrated at some reference batch)
  load-aware  : cost_est = J(L) * dtokens       (live in-service count L, = vLLM num_requests_running)

Two experiments:
  E2a OPEN-LOOP  : controller picks its price lambda from its own cost model to hit budget B,
                   then runs. Measure ACTUAL energy -> budget overshoot/undershoot.
  E2b CLOSED-LOOP: lambda adapts on the real meter so BOTH land on budget B. Compare ACCURACY
                   at matched energy -> pure allocation quality (does the proxy spend the budget
                   in the wrong places?).

Physics is always the measured curve; only the controller's belief changes. Free (no GPU).
Out: results/e2/{e2_token_proxy.md, e2_token_proxy.png}
"""
import json, os, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/e2"; os.makedirs(OUT, exist_ok=True)
J_B1 = 3.7; WH = 1 / 3600.0

# ---------------------------------------------------------------- measured hardware curves
conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
agg = defaultdict(lambda: defaultdict(list))
for r in conc:
    agg[r["batch"]]["tput"].append(r["throughput_tok_s"])
    agg[r["batch"]]["power"].append(r["power_w"])
    agg[r["batch"]]["jtok"].append(r["energy_per_out_token_j"])
BATCHES = sorted(agg)
TPUT = {b: st.mean(agg[b]["tput"]) for b in BATCHES}
POWER = {b: st.mean(agg[b]["power"]) for b in BATCHES}
JTOK = {b: st.mean(agg[b]["jtok"]) for b in BATCHES}
print("measured J/token by batch:", {b: round(JTOK[b], 3) for b in BATCHES})
print(f"  -> spread {max(JTOK.values())/min(JTOK.values()):.0f}x  (this is what the token proxy ignores)")

def _interp(table, L):
    if L <= BATCHES[0]: return table[BATCHES[0]]
    if L >= BATCHES[-1]: return table[BATCHES[-1]]
    return float(np.interp(L, BATCHES, [table[b] for b in BATCHES]))
def req_rate(L): return _interp(TPUT, L) / L if L > 0 else 0.0
def gpu_power(L): return _interp(POWER, L) if L > 0 else 0.0
def jtok(L): return _interp(JTOK, L)

# ---------------------------------------------------------------- per-query data (OTB paired run)
paired = json.load(open("results/paired/paired_oracle_final.json"))
probes = {json.loads(l)["qid"]: json.loads(l) for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
fte = {json.loads(l)["qid"]: json.loads(l)["p_think"]
       for l in open("results/paired/router_fte_modernbert.jsonl", encoding="utf-8")}
by_qid = {p["qid"]: p for p in paired}
common = sorted(set(by_qid) & set(probes) & set(fte))
cn = {q: by_qid[q]["correct_no_think"] for q in common}
ct = {q: by_qid[q]["correct_think"] for q in common}
tok_nt = {q: by_qid[q]["energy_wh_no_think"] * 3600 / J_B1 for q in common}
tok_th = {q: by_qid[q]["energy_wh_think"] * 3600 / J_B1 for q in common}
score_fte = {q: fte[q] for q in common}
RNG0 = np.random.default_rng(11)
score_rand = {q: float(RNG0.random()) for q in common}   # router-quality control
print(f"queries: {len(common)}")

# ---------------------------------------------------------------- trace + simulator with a budget controller
def make_trace(n, seed, hi_rate, lo_rate, hi_s, lo_s):
    rng = np.random.default_rng(seed)
    times, t = [], 0.0
    while len(times) < n:
        phase = t % (hi_s + lo_s)
        t += rng.exponential(1.0 / (hi_rate if phase < hi_s else lo_rate))
        times.append(t)
    qs = [common[i] for i in rng.integers(0, len(common), n)]
    return list(zip(times, qs))

MAX_BATCH = 48
def simulate(trace, decide, on_complete=None):
    """decide(L, qid, energy_j_so_far, n_done) -> bool think."""
    arrivals = sorted(trace); ai, t = 0, 0.0
    inservice, waitq, done = [], [], []
    energy_j = 0.0
    def start(q, arr_t):
        think = decide(len(inservice), q, energy_j, len(done))
        inservice.append(dict(rem=(tok_th[q] if think else tok_nt[q]),
                              correct=(ct[q] if think else cn[q]), think=think, arr=arr_t))
    while ai < len(arrivals) or inservice or waitq:
        L = len(inservice); rate = req_rate(L)
        t_comp = min((r["rem"] / rate for r in inservice), default=float("inf")) if rate > 0 else float("inf")
        t_arr = (arrivals[ai][0] - t) if ai < len(arrivals) else float("inf")
        dt = min(t_comp, t_arr)
        if dt == float("inf"): break
        if L > 0:
            energy_j += gpu_power(L) * dt
            for r in inservice: r["rem"] -= rate * dt
        t += dt
        if t_arr <= t_comp:
            _, q = arrivals[ai]; ai += 1
            if len(inservice) < MAX_BATCH: start(q, t)
            else: waitq.append((t, q))
        else:
            fin = [r for r in inservice if r["rem"] <= 1e-6]
            for r in fin:
                r["latency"] = t - r["arr"]; done.append(r)
                if on_complete: on_complete(energy_j, len(done))
            inservice = [r for r in inservice if r["rem"] > 1e-6]
            while waitq and len(inservice) < MAX_BATCH:
                arr_t, q = waitq.pop(0); start(q, arr_t)
    lat = np.array([r["latency"] for r in done])
    return dict(acc=float(np.mean([r["correct"] for r in done])), energy_wh=energy_j * WH,
                think_rate=float(np.mean([r["think"] for r in done])), n=len(done),
                lat_p95=float(np.percentile(lat, 95)))

# ---------------------------------------------------------------- the two cost models
def make_decider(model, lam, score, c_const=None, state=None):
    """think iff  score(q)  >  lam * cost_est.  cost_est in JOULES for the EXTRA think tokens."""
    def decide(L, q, e_j, n_done):
        dtok = max(tok_th[q] - tok_nt[q], 0.0)
        if model == "load-aware":
            cost = jtok(L) * dtok
        else:                                    # token-blind: constant J/token
            cost = c_const * dtok
        if state is not None:
            lam_eff = state["lam"]
        else:
            lam_eff = lam
        return score[q] > lam_eff * cost
    return decide

def run_closed_loop(trace, model, score, budget_wh, c_const=None, lam0=None, kp=0.55):
    """lambda adapts on the REAL meter so the run lands on budget_wh; returns the run + final lam."""
    n = len(trace)
    state = {"lam": lam0}
    def on_complete(e_j, n_done):
        frac_done = n_done / n
        if frac_done < 0.04: return
        pace = (e_j * WH) / max(budget_wh * frac_done, 1e-9)     # >1 = overspending
        state["lam"] = float(np.clip(state["lam"] * (pace ** kp), 1e-8, 1e4))
    dec = make_decider(model, None, score, c_const=c_const, state=state)
    r = simulate(trace, dec, on_complete=on_complete)
    r["lam_final"] = state["lam"]
    return r

REGIMES = {"bimodal": dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0),
           "stable":  dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0)}
N_TRACE, N_SEEDS = 600, 12

# anchors: never / always energy for budget scaling
anchor = {}
for reg, p in REGIMES.items():
    tr = make_trace(N_TRACE, 0, **p)
    nev = simulate(tr, lambda L, q, e, n: False); alw = simulate(tr, lambda L, q, e, n: True)
    anchor[reg] = (nev, alw)
    print(f"{reg}: never {nev['acc']:.3f} @ {nev['energy_wh']:.2f} Wh | always {alw['acc']:.3f} @ {alw['energy_wh']:.2f} Wh")

# calibration points a token-proxy operator might pick
C_CAL = {"J(batch=1)": JTOK[BATCHES[0]], "J(batch=8)": jtok(8), "J(batch=32)": jtok(32)}
print("token-proxy calibration constants (J/tok):", {k: round(v, 3) for k, v in C_CAL.items()})

# ---------------------------------------------------------------- E2a: open-loop budget adherence
# Each controller PLANS its price lambda offline on a HISTORICAL trace using its own cost model,
# then is deployed on held-out traces. Realistic: an operator tunes on past traffic, then runs.
#   token-blind plans with a closed-form  E = c * tokens          (its belief; no load term)
#   load-aware  plans by replaying the historical trace through the true physics (it HAS the model)
print("\n=== E2a  OPEN-LOOP: plan lambda on a historical trace, deploy on held-out traces ===")
def plan_lambda(hist_trace, model, score, budget_wh, c_const=None, steps=14):
    """Bisect lambda so the controller's OWN predicted energy on the historical trace == budget."""
    qs = [q for _, q in hist_trace]
    dtoks = np.array([max(tok_th[q] - tok_nt[q], 0.0) for q in qs])
    base = np.array([tok_nt[q] for q in qs]); s = np.array([score[q] for q in qs])
    lo, hi = 1e-10, 1e6
    for _ in range(steps if model == "load-aware" else 60):
        lam = (lo * hi) ** 0.5
        if model == "token-blind":                      # closed form under its (wrong) belief
            think = s > lam * (c_const * dtoks)
            pred_wh = ((base + think * dtoks) * c_const).sum() * WH
        else:                                            # replays the real physics it models
            pred_wh = simulate(hist_trace, make_decider("load-aware", lam, score))["energy_wh"]
        if pred_wh > budget_wh: lo = lam
        else: hi = lam
    return (lo * hi) ** 0.5

rows_a = []
for reg, p in REGIMES.items():
    nev, alw = anchor[reg]
    hist = make_trace(N_TRACE, 100, **p)                 # "last week's traffic"
    for frac in [0.35, 0.60]:
        B = nev["energy_wh"] + frac * (alw["energy_wh"] - nev["energy_wh"])
        for cname, c in list(C_CAL.items()) + [("—", None)]:
            model = "load-aware" if c is None else "token-blind"
            lam = plan_lambda(hist, model, score_fte, B, c_const=c)
            errs, accs = [], []
            for seed in range(N_SEEDS):
                tr = make_trace(N_TRACE, seed, **p)
                r = simulate(tr, make_decider(model, lam, score_fte, c_const=c))
                errs.append(100 * (r["energy_wh"] - B) / B); accs.append(r["acc"])
            label = "load-aware" if c is None else f"token-blind {cname}"
            rows_a.append((reg, frac, label, np.mean(errs), np.std(errs), np.mean(accs)))
            print(f"  {reg:8} budget {frac:.0%} | {label:24} actual-vs-budget "
                  f"{np.mean(errs):+7.1f}% (sd {np.std(errs):4.1f}) acc {np.mean(accs):.3f}")

# ---------------------------------------------------------------- E2b: closed-loop, matched budget
print("\n=== E2b  CLOSED-LOOP: both controllers meter actual energy and hit the budget; compare ACCURACY ===")
rows_b = []
for reg, p in REGIMES.items():
    nev, alw = anchor[reg]
    for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
        B = nev["energy_wh"] + frac * (alw["energy_wh"] - nev["energy_wh"])
        for sname, score in [("ModernBERT", score_fte), ("random", score_rand)]:
            # PAIRED design: both controllers run on the SAME traces, so the CI must be on the
            # per-seed DIFFERENCE (an unpaired CI would drown a real effect in trace-to-trace noise).
            per = {m: dict(a=[], e=[], t=[]) for m in ("token-blind", "load-aware")}
            for seed in range(N_SEEDS):
                trace = make_trace(N_TRACE, seed, **p)
                lam0 = 1.0 / max(jtok(8) * 1000, 1e-9)
                for model, c in [("token-blind", C_CAL["J(batch=8)"]), ("load-aware", None)]:
                    r = run_closed_loop(trace, model, score, B, c_const=c, lam0=lam0)
                    per[model]["a"].append(r["acc"]); per[model]["e"].append(r["energy_wh"])
                    per[model]["t"].append(r["think_rate"])
            dif = np.array(per["load-aware"]["a"]) - np.array(per["token-blind"]["a"])
            aT, eT, rT = (np.mean(per["token-blind"][k]) for k in ("a", "e", "t"))
            aL, eL, rL = (np.mean(per["load-aware"][k]) for k in ("a", "e", "t"))
            d = float(dif.mean()); ci = float(1.96 * dif.std(ddof=1) / np.sqrt(len(dif)))
            rows_b.append((reg, frac, sname, aT, eT, rT, aL, eL, rL, d, ci, float((dif > 0).mean())))
            print(f"  {reg:8} B={frac:.0%} {sname:11} | token-blind acc {aT:.4f}@{eT:.2f}Wh (think {rT:.0%}) "
                  f"| load-aware acc {aL:.4f}@{eL:.2f}Wh (think {rL:.0%}) | PAIRED delta {d*100:+.2f} "
                  f"+/-{ci*100:.2f}pp (pos in {(dif>0).mean()*100:.0f}% of traces)")

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
bl = [r for r in rows_a if r[0] == "bimodal"]
labels = sorted(set(r[2] for r in bl))
xs = np.arange(len(labels)); w = 0.38
for i, frac in enumerate([0.35, 0.60]):
    vals = [next(r[3] for r in bl if r[2] == lb and r[1] == frac) for lb in labels]
    ax.bar(xs + (i - 0.5) * w, vals, w, label=f"budget = {frac:.0%} of always-think")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([l.replace("token-blind ", "token\n") for l in labels], fontsize=8)
ax.set_ylabel("actual energy vs budget (%)"); ax.set_title("E2a: open-loop budget adherence (bimodal load)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
ax = axes[1]
for reg, mk in [("bimodal", "o-"), ("stable", "s--")]:
    for sname, c in [("ModernBERT", "tab:green"), ("random", "tab:gray")]:
        rr = [r for r in rows_b if r[0] == reg and r[2] == sname]
        ax.plot([r[1] for r in rr], [r[9] * 100 for r in rr], mk, color=c, ms=5,
                label=f"{reg}, {sname} router")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("energy budget (fraction of always-think)")
ax.set_ylabel("accuracy gain of load-aware over token-blind (pp)")
ax.set_title("E2b: same budget, same meter — allocation quality"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/e2_token_proxy.png", dpi=130)

with open(f"{OUT}/e2_token_proxy.md", "w", encoding="utf-8") as f:
    f.write("# E2 — Does the token cost proxy mislead an energy-budgeted controller?\n\n")
    f.write(f"Measured J/token spans **{max(JTOK.values())/min(JTOK.values()):.0f}x** across batch "
            f"({min(BATCHES)}→{max(BATCHES)}): {({b: round(JTOK[b],3) for b in BATCHES})}. "
            "The controller's *belief* about cost is the only thing that differs; the physics "
            "(measured power integrated over time) is identical.\n\n")
    f.write("## E2a — open-loop: the controller plans with its own cost model\n\n")
    f.write("|regime|budget|cost model|actual vs budget|accuracy|\n|--|--|--|--|--|\n")
    for reg, frac, name, m, sd, acc in rows_a:
        f.write(f"|{reg}|{frac:.0%}|{name}|{m:+.1f}%|{acc:.3f}|\n")
    f.write("\n## E2b — closed-loop: both meter real energy and land on budget; compare accuracy\n\n")
    f.write("Both controllers run on the SAME traces, so the CI is on the **per-seed paired "
            f"difference** over {N_SEEDS} traces (an unpaired CI would drown the effect in "
            "trace-to-trace noise).\n\n")
    f.write("|regime|budget|router|token-blind acc|load-aware acc|paired delta (pp)|95% CI|positive traces|\n"
            "|--|--|--|--|--|--|--|--|\n")
    for reg, frac, sname, aT, eT, rT, aL, eL, rL, d, ci, fp in rows_b:
        f.write(f"|{reg}|{frac:.0%}|{sname}|{aT:.4f}|{aL:.4f}|{d*100:+.2f}|±{ci*100:.2f}|{fp*100:.0f}%|\n")
print(f"\nwrote {OUT}/e2_token_proxy.md, e2_token_proxy.png")
