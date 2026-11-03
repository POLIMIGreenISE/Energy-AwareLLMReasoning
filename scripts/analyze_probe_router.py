"""Step 3c — router with logit-confidence probes (probes_v1.jsonl from notebook 06).

Feature sets (cumulative cost accounting, all honest):
  A text-only                 : no probe cost
  B text+drafts               : + 3 short drafts        (~25 mWh)
  D text+drafts+probes        : + 4 forward-pass probes (measured, ~few mWh)
  E probes+drafts (no text)   : same probe cost, no embedding
  R p_true rule (no training) : think iff mean P(true) < theta  (threshold sweep)
  P p_need rule (no training) : think iff p_need_think > theta  (HRBench's prompt-
      strategy analog; ONE forward pass, no drafts needed -> charged probe_wh/4)

Run:  python scripts/analyze_probe_router.py   (after notebook 06 data is pulled)
Out:  results/draft_router/{summary3.md, pareto_probe.png}
"""
import json, os, re, sys, statistics as st
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "results/draft_router"; os.makedirs(OUT, exist_ok=True)
SEED = 0

# ---------------------------------------------------------------- load probes
probes = {}
for l in open("results/paired/probes_v1.jsonl", encoding="utf-8"):
    d = json.loads(l); probes[d["qid"]] = d

# ---------------------------------------------------------------- draft feats (as 3b)
drafts = defaultdict(dict)
for l in open("results/paired/drafts_v1.jsonl", encoding="utf-8"):
    d = json.loads(l); drafts[d["qid"]][d["rep"]] = d

def norm(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", str(s))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())
def as_num(s):
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", norm(s).replace(" ", ""))
    return float(m.group()) if m else None
def pair_agree(a, b):
    va, vb = as_num(a), as_num(b)
    if va is not None and vb is not None:
        return float(abs(va - vb) <= 1e-6 * max(1.0, abs(va)))
    na, nb = norm(a), norm(b)
    return float(na == nb and na != "")
def jac(a, b):
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

paired = json.load(open("results/paired/paired_oracle_final.json"))
rows = []
for p in paired:
    reps = drafts.get(p["qid"], {}); pr = probes.get(p["qid"])
    if len(reps) < 3 or pr is None:
        continue
    txt = [reps[i]["draft_text"] for i in range(3)]
    pairs = list(combinations(range(3), 2))
    rows.append({**p,
        "agree_frac": st.mean(pair_agree(txt[i], txt[j]) for i, j in pairs),
        "jacm": st.mean(jac(txt[i], txt[j]) for i, j in pairs),
        "tok_mean": st.mean(reps[i]["completion_tokens"] for i in range(3)) / 64.0,
        "draft_wh": sum(reps[i]["energy_b1eq_wh"] for i in range(3)),
        "pt_mean": st.mean(pr["p_true"]), "pt_min": min(pr["p_true"]),
        "pt_std": st.pstdev(pr["p_true"]), "p_need": pr["p_need_think"],
        "probe_wh": pr["probe_energy_mwh"] / 1000.0})
N = len(rows)
print(f"queries with drafts+probes: {N}/{len(paired)}")

cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]); et = np.array([r["energy_wh_think"] for r in rows])
draft_wh = np.array([r["draft_wh"] for r in rows]); probe_wh = np.array([r["probe_wh"] for r in rows])
y = np.array([r["oracle"] == "think" for r in rows]).astype(int)
sub = np.array([r["subset"] for r in rows])
pt_mean = np.array([r["pt_mean"] for r in rows]); p_need = np.array([r["p_need"] for r in rows])
print(f"probe cost: drafts {draft_wh.mean()*1000:.0f} mWh + logit probes {probe_wh.mean()*1000:.1f} mWh")

# ---------------------------------------------------------------- diagnostics
from sklearn.metrics import roc_auc_score
hard = sub == "underthinkingbench"
solv = (ct > 0.5) & (cn <= 0.5)
print("\ndiagnostics:")
print(f"  P(true) calibration (predicts no_think correct):   AUC {roc_auc_score((cn>0.5).astype(int), pt_mean):.3f}")
print(f"  solvability among HARD via 1-P(true):              AUC {roc_auc_score(solv[hard].astype(int), -pt_mean[hard]):.3f}  (drafts: 0.434)")
print(f"  p_need_think separates easy/hard:                  AUC {roc_auc_score(hard.astype(int), p_need):.3f}")
print(f"  oracle-label AUC of 1-P(true) alone:               {roc_auc_score(y, -pt_mean):.3f}")

# ---------------------------------------------------------------- routers
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([r["question"] for r in rows], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(r["question"]) for r in rows]).reshape(-1, 1) / 1000.0
Xd = np.column_stack([[r["agree_frac"] for r in rows], [r["jacm"] for r in rows],
                      [r["tok_mean"] for r in rows]])
Xp = np.column_stack([pt_mean, [r["pt_min"] for r in rows], [r["pt_std"] for r in rows], p_need])

def oof(X):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

P = {"A text-only": (oof(np.hstack([Xe, qlen])), 0),
     "B text+drafts": (oof(np.hstack([Xe, qlen, Xd])), 1),
     "D text+drafts+probes": (oof(np.hstack([Xe, qlen, Xd, Xp])), 2),
     "E drafts+probes only": (oof(np.hstack([Xd, Xp])), 2)}
for k, (p, _) in P.items():
    print(f"{k:22} label AUC: {roc_auc_score(y, p):.3f}")

def overhead(level):
    if level == 0: return np.zeros(N)
    if level == 1: return draft_wh
    if level == "need": return probe_wh / 4.0   # single difficulty probe, no drafts
    return draft_wh + probe_wh

def metrics(choose, level=0):
    choose = np.asarray(choose, bool)
    acc = np.where(choose, ct, cn).mean()
    e = (np.where(choose, et, en) + overhead(level)).mean() * 1000
    return e, acc

taus = np.linspace(0, 1, 101)
curves = {k: np.array([metrics(p >= t, lvl) for t in taus]) for k, (p, lvl) in P.items()}
curves["R p_true rule"] = np.array([metrics(pt_mean < t, 2) for t in taus])
# p_need clusters near 0/1 (AUC 0.989 separator), so sweep its exact values too
taus_need = np.unique(np.concatenate([taus, np.unique(p_need)]))
curves["P p_need rule"] = np.array([metrics(p_need >= t, "need") for t in taus_need])

def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda q: (q[0], -q[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out

dA, dE = ct - cn, (et - en) * 1000
ratios = sorted({dA[i] / dE[i] for i in range(N) if dE[i] > 0 and dA[i] > 0})
of = pareto_ul([metrics((dA - l * dE) > 1e-12) for l in [0] + ratios]
               + [metrics(np.zeros(N, bool)), metrics(np.ones(N, bool))])

NEVER_ACC = None   # set below to true Never-Think accuracy (fair-EAA floor)

def eaa(pts, lo, hi, g=200):
    # FAIR EAA (fixed 2026-07-22): fill budgets below a policy's cheapest operating point
    # with the TRUE Never-Think accuracy (every policy can reject all queries) and
    # normalize over the SAME fixed [lo,hi]. The old version masked those budgets out and
    # renormalized per-policy, unfairly flattering saturated-score policies (p_need,
    # p_true) that cannot reach low energy budgets.
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(f) if f else NEVER_ACC)
    acc = np.array(acc)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(acc, es) / (hi - lo)

e_lo, e_hi = metrics(np.zeros(N, bool))[0], metrics(np.ones(N, bool))[0]
NEVER_ACC = metrics(np.zeros(N, bool))[1]
res = {k: eaa([tuple(q) for q in c], e_lo, e_hi) for k, c in curves.items()}
print(f"\nEAA-AUC over [{e_lo:.0f}, {e_hi:.0f}] mWh (oracle {eaa(of, e_lo, e_hi):.4f}):")
for k, v in res.items():
    print(f"  {k:22} {v:.4f}")

fig, ax = plt.subplots(figsize=(8.2, 5.6))
colors = {"A text-only": "tab:green", "B text+drafts": "tab:blue",
          "D text+drafts+probes": "tab:red", "E drafts+probes only": "tab:purple",
          "R p_true rule": "tab:gray", "P p_need rule": "tab:orange"}
for k, c in colors.items():
    ax.plot(curves[k][:, 0], curves[k][:, 1], "-", lw=1.6, color=c, label=f"{k} (EAA {res[k]:.3f})")
ax.plot([e for e, a in of], [a for e, a in of], "--", color="black", lw=1.2, label="oracle frontier")
ax.set_xlabel("energy per query (mWh, b1eq; probe overhead included)")
ax.set_ylabel("accuracy"); ax.set_title("Router with logit-confidence probes")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_probe.png", dpi=120)

with open(f"{OUT}/summary3.md", "w", encoding="utf-8") as f:
    f.write("# Logit-confidence probe router (step 3c)\n\n")
    f.write(f"N={N}. Probe cost: drafts {draft_wh.mean()*1000:.0f} + logits {probe_wh.mean()*1000:.1f} mWh.\n\n")
    f.write(f"Diagnostics: P(true)->no_think-correct AUC {roc_auc_score((cn>0.5).astype(int), pt_mean):.3f}; "
            f"solvability-among-hard AUC {roc_auc_score(solv[hard].astype(int), -pt_mean[hard]):.3f}; "
            f"p_need easy/hard AUC {roc_auc_score(hard.astype(int), p_need):.3f}\n\n")
    f.write("|policy|label AUC|EAA-AUC (honest)|\n|--|--|--|\n")
    for k, (p, _) in P.items():
        f.write(f"|{k}|{roc_auc_score(y, p):.3f}|{res[k]:.4f}|\n")
    f.write(f"|R p_true rule (no training)|{roc_auc_score(y, -pt_mean):.3f}|{res['R p_true rule']:.4f}|\n")
    f.write(f"|P p_need rule (no training)|{roc_auc_score(y, p_need):.3f}|{res['P p_need rule']:.4f}|\n")
    f.write(f"|oracle frontier||{eaa(of, e_lo, e_hi):.4f}|\n")
    f.write("\np_need rule = think iff p_need_think > tau (single forward-pass difficulty "
            "probe, no drafts, no embedding, no training; overhead probe_wh/4 ~ "
            f"{probe_wh.mean()/4*1000:.1f} mWh/q). Completes the strategy comparison: "
            "learned router vs draft-agreement rule vs self-assessment rules.\n")
print(f"\nwrote {OUT}/summary3.md, pareto_probe.png")
