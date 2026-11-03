"""Workload-mix sensitivity: headline numbers under a BALANCED 50/50 easy/hard mix.

The full OTB natural mix is ~78% easy (1,327/383), which mechanically shrinks
every workload-level number (EAA, tau(load) gains) relative to the old ~60/40
248-query runs — for reasons unrelated to router quality. This script re-evaluates
the key policies under a balanced mix by WEIGHTING each easy query by
n_hard/n_easy (deterministic, uses all data; equivalent in expectation to
subsampling easy down to n_hard).

Run:  python scripts/analyze_balanced_mix.py
Out:  results/audit/balanced_mix.md
"""
import json, os, statistics as st
from collections import defaultdict

import numpy as np

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)

paired = json.load(open("results/paired/paired_oracle_final.json"))
probes = {json.loads(l)["qid"]: json.loads(l) for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
rows = [p for p in paired if p["qid"] in probes]
N = len(rows)
cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]) * 1000; et = np.array([r["energy_wh_think"] for r in rows]) * 1000
hard = np.array([r["subset"] == "underthinkingbench" for r in rows])
p_need = np.array([probes[r["qid"]]["p_need_think"] for r in rows])
need_wh = np.array([probes[r["qid"]]["probe_energy_mwh"] / 4.0 for r in rows])
y = np.array([r["oracle"] == "think" for r in rows]).astype(int)

W = np.where(hard, 1.0, hard.sum() / (~hard).sum())   # easy down-weighted -> 50/50
W = W / W.sum()
print(f"N={N} ({(~hard).sum()} easy, {hard.sum()} hard) | balanced weights: "
      f"easy {W[~hard][0]:.4f} hard {W[hard][0]:.4f}")

def wacc(choose):  return float(np.sum(W * np.where(choose, ct, cn)))
def wene(choose, extra=None):
    e = np.where(choose, et, en)
    if extra is not None: e = e + extra
    return float(np.sum(W * e))

# ------------------------------------------------ router P(think) (same pipeline)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([r["question"] for r in rows], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(r["question"]) for r in rows]).reshape(-1, 1) / 1000.0
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
p_think = cross_val_predict(clf, np.hstack([Xe, qlen]), y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]

def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda q: (q[0], -q[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out

never, always = np.zeros(N, bool), np.ones(N, bool)
e_lo, e_hi = wene(never), wene(always)
a_never, a_always = wacc(never), wacc(always)

def eaa(pts, lo, hi, g=200):
    # FAIR EAA (2026-07-22): fill budgets below a policy's cheapest point with the true
    # Never-Think accuracy and normalize over the fixed [lo,hi] -- see analyze_probe_router.
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(f) if f else a_never)
    acc = np.array(acc)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(acc, es) / (hi - lo)

taus = np.linspace(0, 1, 101)
curve_A = [(wene(p_think >= t), wacc(p_think >= t)) for t in taus]
taus_p = np.unique(np.concatenate([taus, np.unique(p_need)]))
curve_P = [(wene(p_need >= t, need_wh), wacc(p_need >= t)) for t in taus_p]

dA, dE = ct - cn, et - en
pos = (dA > 0) & (dE > 0)
lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
front = pareto_ul([(wene((dA - l * dE) > 1e-12), wacc((dA - l * dE) > 1e-12)) for l in lams]
                  + [(e_lo, a_never), (e_hi, a_always)])

r = {"A text router": eaa(curve_A, e_lo, e_hi), "P p_need rule": eaa(curve_P, e_lo, e_hi),
     "oracle frontier": eaa(front, e_lo, e_hi)}
print(f"\nBALANCED 50/50 mix: Never {a_never:.3f} @ {e_lo:.0f} mWh | Always {a_always:.3f} @ {e_hi:.0f} mWh")
for k, v in r.items():
    gain = 100 * (v - a_never) / (r["oracle frontier"] - a_never + 1e-12)
    print(f"  {k:16} EAA {v:.4f}  ({gain:.0f}% of oracle gain)")

# ------------------------------------------------ tau(load) under balanced mix
conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for c in conc: jt[c["batch"]].append(c["energy_per_out_token_j"])
J_OF_L = {L: st.mean(v) for L, v in jt.items()}
J_B1 = 3.7
tok_nt, tok_th = en / 1000 * 3600 / J_B1, et / 1000 * 3600 / J_B1
WH = 1 / 3600.0

def wene_L(choose, L):
    return float(np.sum(W * np.where(choose, tok_th, tok_nt))) * J_OF_L[L] * WH * 1000

LAM = 1.0
CH = [p_think >= t for t in np.linspace(0, 1, 201)]
def best(L):
    utils = [wacc(c) - LAM * wene_L(c, L) / 1000 for c in CH]
    return CH[int(np.argmax(utils))]
ch_static, ch_adapt = best(1), best(64)
gap = (wacc(ch_adapt) - wacc(ch_static)) * 100
print(f"\ntau(load) lambda=1/Wh balanced mix: static acc {wacc(ch_static):.3f} -> "
      f"adaptive {wacc(ch_adapt):.3f} at batch 64 (**{gap:+.1f}pp**)")

with open(f"{OUT}/balanced_mix.md", "w", encoding="utf-8") as f:
    f.write("# Workload-mix sensitivity — balanced 50/50 easy/hard\n\n"
            f"N={N} ({(~hard).sum()} easy weighted down to match {hard.sum()} hard).\n\n"
            f"| policy | EAA (balanced) | % oracle gain |\n|--|--|--|\n")
    for k, v in r.items():
        gain = 100 * (v - a_never) / (r["oracle frontier"] - a_never + 1e-12)
        f.write(f"| {k} | {v:.4f} | {gain:.0f}% |\n")
    f.write(f"\nNever {a_never:.3f} @ {e_lo:.0f} mWh; Always {a_always:.3f} @ {e_hi:.0f} mWh.\n"
            f"tau(load) lambda=1/Wh: static {wacc(ch_static):.3f} -> adaptive {wacc(ch_adapt):.3f} "
            f"at batch 64 ({gap:+.1f}pp).\n")
print(f"wrote {OUT}/balanced_mix.md")
