"""RQ4-lite: are energy-optimal think/no-think labels load-dependent?

Fully offline. Joins two things we already measured:
  1. Per-query paired outcomes (accuracy + tokens per mode) from
     results/paired/paired_oracle_final.json  (tokens = b1eq_wh * 3600 / 3.7).
  2. The measured token->energy price curve j(L) [J/output token] vs batch size
     from results/batching/concurrency.jsonl (A100, Qwen3-8B).

Decision model (Lagrangian): think on query q iff
    dAcc(q) > lambda * dE(q, L),   dE(q,L) = j(L) * (tok_think - tok_no_think)
lambda (accuracy value per Wh) is CALIBRATED ONCE at idle (batch=1) to best
reproduce our existing energy-aware oracle labels, then held fixed while the
load L varies. A token-priced policy uses the same rule with a fixed per-token
price (calibrated identically at batch=1) — by construction it cannot react to
load. Any divergence between the two as L grows is the RQ4 result.

Scenario B additionally applies the *measured* mode-specific tail pricing at
high concurrency (think batches drain to low concurrency; measured in the
scaled paired run: j_think=0.545, j_no_think=0.269 J/tok at batch 64).

Run:  python scripts/analyze_load_labels.py
Out:  results/load_labels/{summary.md, label_flips.png, frontier_family.png}
"""
import json, os, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J_B1 = 3.7                      # pilot batch-1 J/output-token (A100)
OUT = "results/load_labels"; os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- inputs
paired = json.load(open("results/paired/paired_oracle_final.json"))
N = len(paired)
cn = np.array([p["correct_no_think"] for p in paired])
ct = np.array([p["correct_think"] for p in paired])
tok_nt = np.array([p["energy_wh_no_think"] for p in paired]) * 3600 / J_B1
tok_th = np.array([p["energy_wh_think"]    for p in paired]) * 3600 / J_B1
dA = ct - cn
dTok = tok_th - tok_nt
oracle0 = np.array([p["oracle"] == "think" for p in paired])

conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for r in conc:
    jt[r["batch"]].append(r["energy_per_out_token_j"])
LOADS = sorted(jt)
J_OF_L = {L: st.mean(jt[L]) for L in LOADS}
print("measured j(L) J/token:", {L: round(v, 3) for L, v in J_OF_L.items()})

# measured mode-specific prices at batch 64 (tail effect, from gen_v2 amortized run)
J64_THINK, J64_NOTHINK = 0.545, 0.269

WH = 1 / 3600.0  # J -> Wh

def energies(L, tail=False):
    """Per-query (E_no_think, E_think) in Wh at load L."""
    if tail and L == 64:
        return tok_nt * J64_NOTHINK * WH, tok_th * J64_THINK * WH
    j = J_OF_L[L]
    return tok_nt * j * WH, tok_th * j * WH

def policy(lam, L, tail=False):
    e_nt, e_th = energies(L, tail)
    return dA > lam * (e_th - e_nt)          # think iff accuracy gain beats energy price

def metrics(choose, L, tail=False):
    e_nt, e_th = energies(L, tail)
    acc = np.where(choose, ct, cn).mean()
    e = np.where(choose, e_th, e_nt).mean() * 1000     # mWh/query
    return acc, e

# ------------------------------------------------- calibrate lambda at idle
# pick lambda maximizing agreement with our existing energy-aware oracle at L=1
lams = np.logspace(-3, 1.5, 400)              # 1/Wh
agree = [(policy(l, 1) == oracle0).mean() for l in lams]
lam_star = float(lams[int(np.argmax(agree))])
print(f"calibrated lambda* = {lam_star:.3f} /Wh  "
      f"(agreement with existing oracle at batch=1: {max(agree):.3f})")

# token policy: same calibration but priced in tokens (load-blind by construction)
mus = np.logspace(-6, 0, 400)                 # 1/token
agree_t = [((dA > m * dTok) == oracle0).mean() for m in mus]
mu_star = float(mus[int(np.argmax(agree_t))])
tok_choose = dA > mu_star * dTok              # fixed forever

# ---------------------------------------------------------------- sweep load
rows = []
base = policy(lam_star, 1)
for L in LOADS:
    for tail in ([False, True] if L == 64 else [False]):
        ch = policy(lam_star, L, tail)
        flips = int((ch != base).sum())
        acc, e = metrics(ch, L, tail)
        acc_t, e_t = metrics(tok_choose, L, tail)
        # utility = what the Lagrangian objective actually values
        u  = acc  - lam_star * e  / 1000
        ut = acc_t - lam_star * e_t / 1000
        rows.append(dict(load=L, tail=tail, think_share=float(ch.mean()),
                         flips_vs_idle=flips, acc=acc, mWh=e,
                         tok_acc=acc_t, tok_mWh=e_t,
                         acc_gap_pp=(acc - acc_t) * 100, util_gap=u - ut))

print(f"\n{'load':>5} {'tail':>5} {'think%':>7} {'flips':>6} | {'E-policy acc/mWh':>18} | "
      f"{'tok-policy acc/mWh':>18} | {'acc gap':>7}")
for r in rows:
    print(f"{r['load']:>5} {str(r['tail']):>5} {r['think_share']*100:6.1f}% {r['flips_vs_idle']:>6} | "
          f"{r['acc']:.3f} / {r['mWh']:7.1f} | {r['tok_acc']:.3f} / {r['tok_mWh']:7.1f} | "
          f"{r['acc_gap_pp']:+5.1f}pp")

# ---------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
uni = [r for r in rows if not r["tail"]]
ax = axes[0]
ax.plot([r["load"] for r in uni], [r["think_share"] * 100 for r in uni],
        "-o", color="tab:green", label="energy-optimal policy")
ax.axhline(tok_choose.mean() * 100, ls="--", color="tab:orange",
           label="token-optimal policy (load-blind)")
t64 = [r for r in rows if r["tail"]]
if t64:
    ax.scatter([64], [t64[0]["think_share"] * 100], marker="*", s=170, color="tab:red",
               zorder=5, label="batch 64, measured tail pricing")
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("% of queries labeled THINK")
ax.set_title("Energy-optimal think-share grows with load"); ax.legend()

ax = axes[1]
ax.plot([r["load"] for r in uni], [r["acc_gap_pp"] for r in uni], "-o", color="tab:blue")
if t64:
    ax.scatter([64], [t64[0]["acc_gap_pp"]], marker="*", s=170, color="tab:red")
ax.axhline(0, color="grey", lw=.8)
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("accuracy gap vs token policy (pp)")
ax.set_title("Accuracy left on the table by token-based routing")
plt.tight_layout(); plt.savefig(f"{OUT}/label_flips.png", dpi=120)

# frontier family
def frontier(L, tail=False):
    e_nt, e_th = energies(L, tail)
    dE_wh = e_th - e_nt
    ratios = sorted({dA[i] / dE_wh[i] for i in range(N) if dE_wh[i] > 0 and dA[i] > 0})
    pts = []
    for lam in [0.0] + ratios + ([max(ratios) * 1.5] if ratios else []):
        choose = dA > lam * dE_wh
        acc = np.where(choose, ct, cn).mean()
        e = np.where(choose, e_th, e_nt).mean() * 1000
        pts.append((e, acc))
    for a, e in [metrics(np.zeros(N, bool), L, tail), metrics(np.ones(N, bool), L, tail)]:
        pts.append((e, a))
    ul, best = [], -1
    for e, a in sorted(pts, key=lambda p: (p[0], -p[1])):
        if a > best + 1e-12: ul.append((e, a)); best = a
    return ul

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for L, c in [(1, "tab:red"), (8, "tab:orange"), (64, "tab:green")]:
    f = frontier(L)
    ax.plot([e for e, a in f], [a for e, a in f], "-o", ms=3, color=c,
            label=f"oracle frontier @ batch {L}")
    a_t, e_t = metrics(tok_choose, L)
    a_e, e_e = metrics(policy(lam_star, L), L)
    ax.scatter([e_t], [a_t], marker="x", s=80, color=c)
    ax.scatter([e_e], [a_e], marker="*", s=140, color=c)
ax.set_xscale("log")
ax.set_xlabel("energy per query (mWh, log scale)"); ax.set_ylabel("accuracy")
ax.set_title("Frontier family: * = energy policy, x = token policy (load-blind)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/frontier_family.png", dpi=120)

# ------------------------------------------- lambda sweep: where do labels flip?
# For any operator energy price lambda, compare think-share at idle vs busy.
lam_grid = np.logspace(-3, 1.2, 200)
share_idle = np.array([policy(l, 1).mean() for l in lam_grid])
share_busy = np.array([policy(l, 64).mean() for l in lam_grid])
flip_frac = np.array([(policy(l, 64) != policy(l, 1)).mean() for l in lam_grid])
i_max = int(np.argmax(flip_frac))
print(f"\nlambda sweep: max label-flip fraction {flip_frac[i_max]*100:.1f}% "
      f"at lambda={lam_grid[i_max]:.2f}/Wh "
      f"(idle thinks {share_idle[i_max]*100:.0f}%, busy thinks {share_busy[i_max]*100:.0f}%)")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(lam_grid, share_idle * 100, color="tab:red", label="idle server (batch 1)")
ax.plot(lam_grid, share_busy * 100, color="tab:green", label="busy server (batch 64)")
ax.fill_between(lam_grid, share_idle * 100, share_busy * 100, alpha=.15, color="tab:blue",
                label="load-dependent zone (labels differ)")
ax.axvline(lam_star, ls=":", color="grey")
ax.text(lam_star, 5, " calibrated λ*", rotation=90, va="bottom", fontsize=8, color="grey")
ax.set_xscale("log"); ax.set_xlabel("operator energy price λ (accuracy per Wh)")
ax.set_ylabel("% of queries labeled THINK")
ax.set_title("The same query set, priced at idle vs busy: labels diverge with load")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/lambda_sweep.png", dpi=120)

# ------------------------------------------- budget framing: what a Wh buys
def best_acc_under_budget(B_mwh, L, tail=False):
    f = frontier(L, tail)
    feas = [a for e, a in f if e <= B_mwh + 1e-9]
    return max(feas) if feas else float("nan")

budgets = [50, 100, 200, 500, 1000, 2000, 4000]
print(f"\naccuracy achievable under a fixed energy budget (mWh/query):")
print(f"{'budget':>8} | " + " | ".join(f"batch {L:>2}" for L in [1, 8, 64]))
budget_rows = []
for B in budgets:
    accs = [best_acc_under_budget(B, L) for L in [1, 8, 64]]
    budget_rows.append((B, accs))
    print(f"{B:>8} | " + " | ".join(f"{a:8.3f}" if a == a else "   --   " for a in accs))

# ---------------------------------------------------------------- summary
with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
    f.write("# RQ4-lite: load-dependent energy-aware labels\n\n")
    f.write(f"- N={N} paired queries; j(L) measured on A100 (concurrency sweep); "
            f"lambda*={lam_star:.3f}/Wh calibrated at batch=1 "
            f"(agreement with existing oracle {max(agree):.3f}).\n")
    f.write(f"- token policy calibrated identically at batch=1 (mu*={mu_star:.2e}/token), "
            "load-blind by construction.\n\n")
    f.write("|load|tail|think share|label flips vs idle|E-policy acc @ mWh|token policy acc @ mWh|acc gap|\n")
    f.write("|--|--|--|--|--|--|--|\n")
    for r in rows:
        f.write(f"|{r['load']}|{'yes' if r['tail'] else ''}|{r['think_share']*100:.1f}%"
                f"|{r['flips_vs_idle']}|{r['acc']:.3f} @ {r['mWh']:.1f}"
                f"|{r['tok_acc']:.3f} @ {r['tok_mWh']:.1f}|{r['acc_gap_pp']:+.1f}pp|\n")
    f.write(f"\n## Lambda sweep\nMax label-flip fraction between idle and busy: "
            f"**{flip_frac[i_max]*100:.1f}%** of queries at λ={lam_grid[i_max]:.2f}/Wh "
            f"(idle {share_idle[i_max]*100:.0f}% think vs busy {share_busy[i_max]*100:.0f}% think). "
            f"At the accuracy-dominant calibrated λ*={lam_star:.3f}/Wh flips are minimal "
            f"({rows[-2]['flips_vs_idle']}/{N}) — note dAcc is quantized at 0.5 with REPS=2, "
            "which widens the dead zone; more reps would sharpen this.\n")
    f.write("\n## Budget framing (accuracy achievable under fixed mWh/query)\n\n")
    f.write("|budget (mWh/q)|batch 1|batch 8|batch 64|\n|--|--|--|--|\n")
    for B, accs in budget_rows:
        f.write(f"|{B}|" + "|".join(f"{a:.3f}" if a == a else "--" for a in accs) + "|\n")
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
print(f"\nwrote {OUT}/summary.md, label_flips.png, frontier_family.png")
