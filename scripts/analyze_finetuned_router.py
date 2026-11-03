"""Compare fine-tuned encoder routers (notebook 07) against the existing
frozen-MiniLM+LogReg router and the p_need rule, on the same honest-energy
EAA/Pareto machinery used everywhere else.

Classifier inference energy (measured with Zeus in notebook 07,
router_fte_energy.json) is charged into every fine-tuned-router decision, the
same way the p_need probe's measured 1.5 mWh/query is charged into its curve.

Run (after notebook 07 has run and pushed its outputs):
  python scripts/analyze_finetuned_router.py
Out:  results/system_router/finetuned_comparison.md, pareto_finetuned.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/system_router"; os.makedirs(OUT, exist_ok=True)

paired = {p["qid"]: p for p in json.load(open("results/paired/paired_oracle_final.json"))}
probes = {json.loads(l)["qid"]: json.loads(l) for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}

fte_files = {f.replace("router_fte_", "").replace(".jsonl", ""): f
             for f in os.listdir("results/paired") if f.startswith("router_fte_") and f.endswith(".jsonl")}
if not fte_files:
    raise SystemExit("No router_fte_*.jsonl found -- run notebook 07 on vast.ai first, then pull.")
energy_path = "results/paired/router_fte_energy.json"
fte_energy = json.load(open(energy_path)) if os.path.exists(energy_path) else {}
print("found fine-tuned router outputs:", list(fte_files), "| energy:", fte_energy)

fte_scores = {}
for name, path in fte_files.items():
    fte_scores[name] = {json.loads(l)["qid"]: json.loads(l)["p_think"]
                        for l in open(f"results/paired/{path}", encoding="utf-8")}

common = sorted(set(paired) & set(probes) & set.intersection(*[set(v) for v in fte_scores.values()]))
N = len(common)
print(f"N={N} queries with paired labels + probes + all fine-tuned scores")

cn = np.array([paired[q]["correct_no_think"] for q in common])
ct = np.array([paired[q]["correct_think"] for q in common])
en = np.array([paired[q]["energy_wh_no_think"] for q in common]) * 1000
et = np.array([paired[q]["energy_wh_think"] for q in common]) * 1000
y = np.array([paired[q]["oracle"] == "think" for q in common]).astype(int)
p_need = np.array([probes[q]["p_need_think"] for q in common])
need_wh = np.array([probes[q]["probe_energy_mwh"] / 4.0 for q in common])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([paired[q]["question"] for q in common], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(paired[q]["question"]) for q in common]).reshape(-1, 1) / 1000.0
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
p_logreg = cross_val_predict(clf, np.hstack([Xe, qlen]), y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]

def metrics(choose, extra_mwh=0.0):
    choose = np.asarray(choose, bool)
    acc = np.where(choose, ct, cn).mean()
    e = (np.where(choose, et, en) + extra_mwh).mean()
    return e, acc

def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda q: (q[0], -q[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out

NEVER_ACC = None   # set once below to the true Never-Think accuracy

def eaa(pts, lo, hi, g=200):
    # FAIR EAA: every router can trivially also Never-Think (reject all queries) at the
    # floor energy, so for budgets below a router's cheapest think-policy its best
    # achievable accuracy is Never-Think accuracy -- NOT undefined. Filling those budgets
    # with the TRUE Never-Think accuracy and normalizing over the SAME fixed [lo,hi] for
    # every policy is the only fair comparison. (The earlier version masked those budgets
    # out and normalized over each policy's own range, which unfairly flattered
    # saturated-score policies like p_need that cannot reach low budgets -- caught
    # 2026-07-22. NB: a saturated policy's own curve may not even CONTAIN a never-think
    # point, so we must use the externally-supplied NEVER_ACC, not min(pts).)
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(f) if f else NEVER_ACC)
    acc = np.array(acc)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(acc, es) / (hi - lo)

e_lo, e_hi = metrics(np.zeros(N, bool))[0], metrics(np.ones(N, bool))[0]
NEVER_ACC = metrics(np.zeros(N, bool))[1]   # true Never-Think accuracy (fair-EAA floor)
dA, dE = ct - cn, et - en
pos = (dA > 0) & (dE > 0)
lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
front = pareto_ul([metrics((dA - l * dE) > 1e-12) for l in lams]
                  + [metrics(np.zeros(N, bool)), metrics(np.ones(N, bool))])
oracle_auc = eaa(front, e_lo, e_hi)

taus = np.linspace(0, 1, 101)
curves = {
    "LogReg+MiniLM (existing)": (np.array([metrics(p_logreg >= t) for t in taus]),
                                 roc_auc_score(y, p_logreg), 0.0),
}
taus_need = np.unique(np.concatenate([taus, np.unique(p_need)]))
curves["p_need rule"] = (np.array([metrics(p_need >= t, need_wh) for t in taus_need]),
                         roc_auc_score(y, p_need), float(need_wh.mean()))
for name, scores in fte_scores.items():
    p = np.array([scores[q] for q in common])
    e_cost = fte_energy.get(name, 0.0)   # mWh/query, measured
    curves[f"fine-tuned {name}"] = (np.array([metrics(p >= t, e_cost) for t in taus]),
                                    roc_auc_score(y, p), e_cost)

print(f"\nEAA-AUC over [{e_lo:.0f}, {e_hi:.0f}] mWh (oracle frontier: {oracle_auc:.4f}):")
rows_out = []
for name, (curve, auc, cost) in curves.items():
    e = eaa([tuple(c) for c in curve], e_lo, e_hi)
    pct = 100 * (e - metrics(np.zeros(N, bool))[1]) / (oracle_auc - metrics(np.zeros(N, bool))[1])
    rows_out.append((name, auc, cost, e, pct))
    print(f"  {name:28} label AUC {auc:.3f} | classifier cost {cost:7.4f} mWh/q | EAA {e:.4f} ({pct:.1f}% of oracle gain)")

fig, ax = plt.subplots(figsize=(8.5, 5.8))
colors = plt.cm.tab10.colors
for i, (name, (curve, auc, cost)) in enumerate(curves.items()):
    ax.plot(curve[:, 0], curve[:, 1], "-", lw=1.7, color=colors[i % 10],
            label=f"{name} (EAA {eaa([tuple(c) for c in curve], e_lo, e_hi):.3f})")
ax.plot([e for e, a in front], [a for e, a in front], "--", color="black", lw=1.3, label="oracle frontier")
ax.set_xlabel("energy per query (mWh, b1eq; classifier/probe cost included)")
ax.set_ylabel("accuracy"); ax.set_title("Fine-tuned encoder routers vs existing router vs p_need")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_finetuned.png", dpi=120)

with open(f"{OUT}/finetuned_comparison.md", "w", encoding="utf-8") as f:
    f.write("# Fine-tuned encoder router vs existing router vs p_need\n\n")
    f.write(f"N={N}. Oracle frontier EAA: {oracle_auc:.4f}.\n\n")
    f.write("|policy|label AUC|classifier/probe cost (mWh/q)|EAA-AUC|% of oracle gain|\n|--|--|--|--|--|\n")
    for name, auc, cost, e, pct in rows_out:
        f.write(f"|{name}|{auc:.3f}|{cost:.4f}|{e:.4f}|{pct:.1f}%|\n")
print(f"\nwrote {OUT}/finetuned_comparison.md, pareto_finetuned.png")
