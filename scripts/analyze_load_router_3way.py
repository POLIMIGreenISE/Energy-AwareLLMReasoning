"""Completes the system-aware router finale: adds the FINE-TUNED ModernBERT
router (results/paired/router_fte_modernbert.jsonl, built after the original
system-aware-router run in results/system_router/summary.md) into the SAME
tau(load) mechanism already used there for p_need and results/load_labels/
router_summary.md's LogReg+MiniLM, so all three strategies are directly
comparable side by side.

Same mechanism throughout: tau*(L) = argmax_tau [ acc(tau) - lambda*E(tau,L) ],
re-solved per load L from the measured J(L) decode-token curve. Classifier
generation cost scales with load; classifier/probe cost is a FIXED measured
constant (see results/system_router/summary.md for why -- unmeasured
prefill-only concurrency curve, and the fine-tuned encoder is an
architecturally separate small model).

Run:  python scripts/analyze_load_router_3way.py
Out:  results/system_router/load_3way.md, load_3way.png
"""
import json, os, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/system_router"; os.makedirs(OUT, exist_ok=True)
J_B1 = 3.7
WH = 1 / 3600.0

paired = json.load(open("results/paired/paired_oracle_final.json"))
probes = {json.loads(l)["qid"]: json.loads(l) for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
fte = {json.loads(l)["qid"]: json.loads(l)["p_think"]
       for l in open("results/paired/router_fte_modernbert.jsonl", encoding="utf-8")}
fte_energy = json.load(open("results/paired/router_fte_energy.json"))

by_qid = {p["qid"]: p for p in paired}
common = sorted(set(by_qid) & set(probes) & set(fte))
rows = [by_qid[q] for q in common]
N = len(rows)
print(f"N={N}")

cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]) * 1000
et = np.array([r["energy_wh_think"] for r in rows]) * 1000
tok_nt, tok_th = en / 1000 * 3600 / J_B1, et / 1000 * 3600 / J_B1
y = np.array([r["oracle"] == "think" for r in rows]).astype(int)

p_need = np.array([probes[q]["p_need_think"] for q in common])
need_fixed = float(np.mean([probes[q]["probe_energy_mwh"] / 4.0 for q in common]))
p_fte = np.array([fte[q] for q in common])
fte_fixed = float(fte_energy["modernbert"])

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([by_qid[q]["question"] for q in common], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(by_qid[q]["question"]) for q in common]).reshape(-1, 1) / 1000.0
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
p_logreg = cross_val_predict(clf, np.hstack([Xe, qlen]), y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]

conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for r in conc:
    jt[r["batch"]].append(r["energy_per_out_token_j"])
LOADS = sorted(jt)
J_OF_L = {L: st.mean(v) for L, v in jt.items()}

STRATEGIES = {
    "LogReg+MiniLM": (p_logreg, 0.0),
    "Fine-tuned ModernBERT": (p_fte, fte_fixed),
    "p_need rule": (p_need, need_fixed),
}

def A(choose): return np.where(choose, ct, cn).mean()
def E(choose, L, fixed): return np.where(choose, tok_th, tok_nt).mean() * J_OF_L[L] * WH * 1000 + fixed

LAM = 1.0   # energy-conscious scenario -- the one where load-awareness matters
results = {}
for name, (score, fixed) in STRATEGIES.items():
    taus = np.unique(np.concatenate([np.linspace(0, 1, 201), np.unique(score)]))
    def best_tau(L):
        utils = [A(score >= t) - LAM * E(score >= t, L, fixed) / 1000 for t in taus]
        i = int(np.argmax(utils))
        return taus[i], score >= taus[i]
    tau_idle, ch_idle = best_tau(1)
    rows_L = []
    for L in LOADS:
        tau_L, ch_L = best_tau(L)
        rows_L.append(dict(load=L, tau=tau_L, acc_static=A(ch_idle), acc_adaptive=A(ch_L),
                           e_static=E(ch_idle, L, fixed), e_adaptive=E(ch_L, L, fixed)))
    results[name] = (tau_idle, rows_L)
    print(f"\n{name} (static tau*={tau_idle:.3f}):")
    for r in rows_L:
        if r["load"] in (1, 8, 64):
            gap = (r["acc_adaptive"] - r["acc_static"]) * 100
            print(f"  L={r['load']:>3} acc adap={r['acc_adaptive']:.3f} static={r['acc_static']:.3f} "
                  f"({gap:+.1f}pp) | mWh adap={r['e_adaptive']:.1f} static={r['e_static']:.1f}")

print("\nheadline @ batch=64, lambda=1.0/Wh:")
for name in STRATEGIES:
    _, rows_L = results[name]
    r64 = next(r for r in rows_L if r["load"] == 64)
    gain = (r64["acc_adaptive"] - r64["acc_static"]) * 100
    print(f"  {name:22} {r64['acc_static']:.3f} -> {r64['acc_adaptive']:.3f}  ({gain:+.1f}pp)")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
colors = {"LogReg+MiniLM": "tab:blue", "Fine-tuned ModernBERT": "tab:green", "p_need rule": "tab:orange"}
ax = axes[0]
for name, c in colors.items():
    _, rows_L = results[name]
    ax.plot([r["load"] for r in rows_L], [r["acc_adaptive"] for r in rows_L], "-o", ms=3, color=c, label=f"{name} adaptive")
    ax.plot([r["load"] for r in rows_L], [r["acc_static"] for r in rows_L], "--", color=c, alpha=.5)
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)"); ax.set_ylabel("accuracy")
ax.set_title("Solid=adaptive tau(L), dashed=static (lambda=1.0/Wh)"); ax.legend(fontsize=7)
ax = axes[1]
for name, c in colors.items():
    _, rows_L = results[name]
    ax.plot([r["load"] for r in rows_L], [(r["acc_adaptive"] - r["acc_static"]) * 100 for r in rows_L],
            "-o", ms=3, color=c, label=name)
ax.axhline(0, color="grey", lw=.8)
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("accuracy gain, adaptive vs static (pp)"); ax.set_title("Load-responsiveness by strategy")
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/load_3way.png", dpi=120)

with open(f"{OUT}/load_3way.md", "w", encoding="utf-8") as f:
    f.write("# tau(load) 3-way: LogReg+MiniLM vs fine-tuned ModernBERT vs p_need\n\n")
    f.write(f"N={N}, lambda=1.0/Wh (energy-conscious). Completes results/system_router/summary.md "
            "(which only covered p_need) with the fine-tuned encoder added.\n\n")
    f.write("|strategy|static tau*|static acc|adaptive acc @ batch64|gain (pp)|\n|--|--|--|--|--|\n")
    for name in STRATEGIES:
        tau_idle, rows_L = results[name]
        r64 = next(r for r in rows_L if r["load"] == 64)
        gain = (r64["acc_adaptive"] - r64["acc_static"]) * 100
        f.write(f"|{name}|{tau_idle:.3f}|{r64['acc_static']:.3f}|{r64['acc_adaptive']:.3f}|{gain:+.1f}pp|\n")
print(f"\nwrote {OUT}/load_3way.md, load_3way.png")
