"""Step 2 — load-adaptive threshold router vs static router (offline, no GPU).

Architecture under test:  decision = [P(think) from query]  vs  [threshold tau(load)].
The classifier is trained ONCE (text-only, out-of-fold P(think), identical to
notebook 03). What adapts is only the threshold:

  static   : tau* fixed at idle calibration (what a token-trained router does —
             token prices never change, so its operating point can't move)
  adaptive : tau*(L) = argmax_tau [ accuracy - lambda * E(L) ]  re-solved per load

Two operator scenarios: lambda* = 0.04/Wh (accuracy-first, calibrated in step 1)
and lambda = 1.0/Wh (energy-conscious, inside the step-1 flip zone).
Plus the budget framing: best feasible accuracy under a fixed mWh/query cap.

Run:  python scripts/analyze_load_router.py
Out:  results/load_labels/{router_summary.md, load_router.png}
"""
import json, os, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J_B1 = 3.7
OUT = "results/load_labels"; os.makedirs(OUT, exist_ok=True)

paired = json.load(open("results/paired/paired_oracle_final.json"))
N = len(paired)
cn = np.array([p["correct_no_think"] for p in paired])
ct = np.array([p["correct_think"] for p in paired])
tok_nt = np.array([p["energy_wh_no_think"] for p in paired]) * 3600 / J_B1
tok_th = np.array([p["energy_wh_think"]    for p in paired]) * 3600 / J_B1
y = np.array([p["oracle"] == "think" for p in paired]).astype(int)

conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for r in conc:
    jt[r["batch"]].append(r["energy_per_out_token_j"])
LOADS = sorted(jt)
J_OF_L = {L: st.mean(jt[L]) for L in LOADS}

# ------------------------------------------------ the router (same as notebook 03)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([p["question"] for p in paired], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(p["question"]) for p in paired]).reshape(-1, 1) / 1000.0
X = np.hstack([Xe, qlen])
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p_think = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

WH = 1 / 3600.0
def E(choose, L):
    j = J_OF_L[L]
    return np.where(choose, tok_th, tok_nt).mean() * j * WH * 1000   # mWh/q

def A(choose):
    return np.where(choose, ct, cn).mean()

TAUS = np.linspace(0, 1, 201)
CHOICES = [p_think >= t for t in TAUS]

def best_tau(L, lam):
    utils = [A(c) - lam * E(c, L) / 1000 for c in CHOICES]
    i = int(np.argmax(utils))
    return TAUS[i], CHOICES[i]

# ------------------------------------------------ scenario sweeps
results = {}
for lam, name in [(0.04, "lambda=0.04/Wh (accuracy-first)"),
                  (1.0,  "lambda=1.0/Wh (energy-conscious)")]:
    tau_idle, ch_idle = best_tau(1, lam)          # static router frozen here
    rows = []
    for L in LOADS:
        tau_L, ch_L = best_tau(L, lam)
        rows.append(dict(load=L, tau_static=tau_idle, tau_adaptive=tau_L,
                         acc_static=A(ch_idle), acc_adaptive=A(ch_L),
                         e_static=E(ch_idle, L), e_adaptive=E(ch_L, L),
                         think_static=float(ch_idle.mean()), think_adaptive=float(ch_L.mean())))
    results[name] = (lam, tau_idle, rows)
    print(f"\n--- {name} (static tau*={tau_idle:.2f}, thinks {ch_idle.mean()*100:.0f}%)")
    print(f"{'load':>5} | {'tau(L)':>7} {'think%':>7} | {'acc adap':>8} {'acc stat':>8} {'gap':>6} | "
          f"{'mWh adap':>8} {'mWh stat':>8}")
    for r in rows:
        print(f"{r['load']:>5} | {r['tau_adaptive']:7.2f} {r['think_adaptive']*100:6.0f}% | "
              f"{r['acc_adaptive']:.3f}   {r['acc_static']:.3f}  {(r['acc_adaptive']-r['acc_static'])*100:+5.1f}pp | "
              f"{r['e_adaptive']:8.1f} {r['e_static']:8.1f}")

# ------------------------------------------------ budget framing with the real router
def best_acc_under_budget(B, L, static_tau=None):
    """Best router accuracy with mean energy <= B at load L.
    static_tau: evaluate only that threshold (frozen policy)."""
    if static_tau is not None:
        c = p_think >= static_tau
        return A(c) if E(c, L) <= B + 1e-9 else float("nan")
    feas = [A(c) for c in CHOICES if E(c, L) <= B + 1e-9]
    return max(feas) if feas else float("nan")

budgets = [50, 100, 200, 500, 1000, 2000]
lam, tau_idle, _ = results["lambda=1.0/Wh (energy-conscious)"]
print("\nBudget framing (router, not oracle): best accuracy under mean-energy cap")
print(f"{'budget':>8} | " + " | ".join(f"b{L:>2} adap/static" for L in [1, 8, 64]))
budget_rows = []
for B in budgets:
    cells = []
    for L in [1, 8, 64]:
        a_ad = best_acc_under_budget(B, L)
        a_st = best_acc_under_budget(B, L, static_tau=tau_idle)
        cells.append((a_ad, a_st))
    budget_rows.append((B, cells))
    print(f"{B:>8} | " + " | ".join(
        f"{(f'{a:.3f}' if a == a else ' --  ')}/{(f'{s:.3f}' if s == s else ' --  ')}"
        for a, s in cells))

# ------------------------------------------------ plot (energy-conscious scenario)
lam1, tau1, rows1 = results["lambda=1.0/Wh (energy-conscious)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.plot([r["load"] for r in rows1], [r["tau_adaptive"] for r in rows1], "-o",
        color="tab:green", label="adaptive τ(load)")
ax.axhline(tau1, ls="--", color="tab:orange", label="static τ (idle-calibrated)")
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("router threshold τ"); ax.set_title("λ=1.0/Wh: the threshold should fall as load rises")
ax.legend()
ax = axes[1]
ax.plot([r["load"] for r in rows1], [(r["acc_adaptive"] - r["acc_static"]) * 100 for r in rows1],
        "-o", color="tab:blue")
ax.axhline(0, color="grey", lw=.8)
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("accuracy gain of τ(load) vs static τ (pp)")
ax.set_title("Accuracy recovered by load-awareness (same classifier)")
plt.tight_layout(); plt.savefig(f"{OUT}/load_router.png", dpi=120)

with open(f"{OUT}/router_summary.md", "w", encoding="utf-8") as f:
    f.write("# Step 2 — load-adaptive threshold router\n\n"
            "Same trained classifier P(think); only the threshold adapts: "
            "tau*(L) = argmax acc - lambda*E(L).\n\n")
    for name, (lam, tau_idle, rows) in results.items():
        f.write(f"## {name} (static tau*={tau_idle:.2f})\n\n")
        f.write("|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|\n")
        f.write("|--|--|--|--|--|--|--|--|\n")
        for r in rows:
            f.write(f"|{r['load']}|{r['tau_adaptive']:.2f}|{r['think_adaptive']*100:.0f}%"
                    f"|{r['acc_adaptive']:.3f}|{r['acc_static']:.3f}"
                    f"|{(r['acc_adaptive']-r['acc_static'])*100:+.1f}pp"
                    f"|{r['e_adaptive']:.1f}|{r['e_static']:.1f}|\n")
        f.write("\n")
    f.write("## Budget framing (energy-conscious static tau)\n\n")
    f.write("|budget mWh/q|" + "|".join(f"batch {L} adaptive/static" for L in [1, 8, 64]) + "|\n")
    f.write("|--|--|--|--|\n")
    for B, cells in budget_rows:
        f.write(f"|{B}|" + "|".join(
            f"{(f'{a:.3f}' if a == a else '--')} / {(f'{s:.3f}' if s == s else '--')}"
            for a, s in cells) + "|\n")
print(f"\nwrote {OUT}/router_summary.md, load_router.png")
