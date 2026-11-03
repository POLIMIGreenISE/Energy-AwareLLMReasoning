"""Step 3b — router with FORMAT-CONSTRAINED draft features (drafts_v1.jsonl).

Same evaluation as analyze_draft_router.py, but the confidence probe is now
3 short "final answer only" drafts (mean ~12 tokens) instead of 2 free-form
no_think answers (~540 tokens). Honest accounting per query:
    probe = sum of the 3 drafts' energy (b1eq)  [~35 mWh]
    no_think branch: probe + E_no_think ; think branch: probe + E_think

Run:  python scripts/analyze_draft_router2.py
Out:  results/draft_router/{summary2.md, pareto_draft2.png}
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

# ---------------------------------------------------------------- draft features
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

def feats(qid):
    reps = drafts.get(qid, {})
    if len(reps) < 3:
        return None
    txt = [reps[i]["draft_text"] for i in range(3)]
    pairs = list(combinations(range(3), 2))
    agree_frac = st.mean(pair_agree(txt[i], txt[j]) for i, j in pairs)
    jacm = st.mean(jac(txt[i], txt[j]) for i, j in pairs)
    toks = [reps[i]["completion_tokens"] for i in range(3)]
    probe_wh = sum(reps[i]["energy_b1eq_wh"] for i in range(3))
    return dict(agree_frac=agree_frac, all_agree=float(agree_frac == 1.0), jacm=jacm,
                tok_mean=st.mean(toks) / 64.0, tok_std=(st.pstdev(toks)) / 64.0,
                probe_wh=probe_wh)

paired = json.load(open("results/paired/paired_oracle_final.json"))
rows = []
for p in paired:
    f = feats(p["qid"])
    if f is not None:
        rows.append({**p, **f})
N = len(rows)
print(f"queries with 3 constrained drafts: {N}/{len(paired)}")

cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]); et = np.array([r["energy_wh_think"] for r in rows])
probe = np.array([r["probe_wh"] for r in rows])
y = np.array([r["oracle"] == "think" for r in rows]).astype(int)
agreef = np.array([r["agree_frac"] for r in rows]); jacm = np.array([r["jacm"] for r in rows])
sub = np.array([r["subset"] for r in rows])
print(f"mean probe cost: {probe.mean()*1000:.0f} mWh (vs no_think {en.mean()*1000:.0f}, think {et.mean()*1000:.0f})")

# ---------------------------------------------------------------- diagnostics
solv = (ct > 0.5) & (cn <= 0.5)
hard = sub == "underthinkingbench"
easy = ~hard
print("\nconstrained-draft agreement by group:")
for name, m in [("easy (overthinking)", easy), ("hard-solvable", solv & hard),
                ("hard-unsolvable", hard & ~solv)]:
    print(f"  {name:20} n={int(m.sum()):3}  agree_frac={agreef[m].mean():.2f}  jaccard={jacm[m].mean():.2f}")
from sklearn.metrics import roc_auc_score
print(f"  (1-agree_frac) predicts solvable among HARD: AUC "
      f"{roc_auc_score(solv[hard], -agreef[hard]):.3f}  (free-form was 0.595)")

# ---------------------------------------------------------------- routers
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([r["question"] for r in rows], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(r["question"]) for r in rows]).reshape(-1, 1) / 1000.0
Xd = np.column_stack([agreef, [r["all_agree"] for r in rows], jacm,
                      [r["tok_mean"] for r in rows], [r["tok_std"] for r in rows]])

def oof(X):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

P = {"A text-only": oof(np.hstack([Xe, qlen])),
     "B text+drafts": oof(np.hstack([Xe, qlen, Xd])),
     "C drafts-only": oof(Xd)}
for k, p in P.items():
    print(f"{k:15} label AUC: {roc_auc_score(y, p):.3f}")

def metrics(choose, pay_probe=False):
    choose = np.asarray(choose, bool)
    acc = np.where(choose, ct, cn).mean()
    e = np.where(choose, et, en) + (probe if pay_probe else 0)
    return e.mean() * 1000, acc

taus = np.linspace(0, 1, 101)
curves = {k: np.array([metrics(p >= t, pay_probe=not k.startswith("A")) for t in taus])
          for k, p in P.items()}
dart = metrics(agreef < 1.0, pay_probe=True)              # think unless ALL 3 agree
dart_m = metrics(agreef < 0.5, pay_probe=True)            # think iff majority disagree

def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda q: (q[0], -q[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out

dA, dE = ct - cn, (et - en) * 1000
ratios = sorted({dA[i] / dE[i] for i in range(N) if dE[i] > 0 and dA[i] > 0})
of = pareto_ul([metrics((dA - l * dE) > 1e-12) for l in [0] + ratios]
               + [metrics(np.zeros(N, bool)), metrics(np.ones(N, bool))])

e_lo, e_hi = metrics(np.zeros(N, bool))[0], metrics(np.ones(N, bool))[0]
_NEVER_ACC = metrics(np.zeros(N, bool))[1]

def eaa(pts, lo, hi, g=200):
    # Fair EAA (2026-07-22): fill sub-cheapest budgets with true Never-Think accuracy,
    # normalize over fixed [lo,hi].
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(f) if f else _NEVER_ACC)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(np.array(acc), es) / (hi - lo)
res = {k: eaa([tuple(q) for q in curves[k]], e_lo, e_hi) for k in P}
print(f"\nEAA-AUC over [{e_lo:.0f}, {e_hi:.0f}] mWh (oracle {eaa(of, e_lo, e_hi):.4f}):")
for k in P: print(f"  {k:15} {res[k]:.4f}" + ("  <- probe overhead included" if not k.startswith("A") else ""))
print(f"  DART all-3 rule : acc={dart[1]:.3f} @ {dart[0]:.0f} mWh (honest)")
print(f"  DART majority   : acc={dart_m[1]:.3f} @ {dart_m[0]:.0f} mWh (honest)")

fig, ax = plt.subplots(figsize=(8, 5.5))
for k, c in {"A text-only": "tab:green", "B text+drafts": "tab:blue",
             "C drafts-only": "tab:purple"}.items():
    ax.plot(curves[k][:, 0], curves[k][:, 1], "-", lw=1.6, color=c, label=f"{k} (EAA {res[k]:.3f})")
ax.plot([e for e, a in of], [a for e, a in of], "--", color="tab:red", label="oracle frontier")
ax.scatter([dart[0]], [dart[1]], marker="D", s=70, color="black", zorder=5, label="DART all-3 (honest)")
ax.scatter([metrics(np.ones(N, bool))[0]], [metrics(np.ones(N, bool))[1]], marker="^", s=90,
           color="tab:orange", label="Always-Think")
ax.scatter([metrics(np.zeros(N, bool))[0]], [metrics(np.zeros(N, bool))[1]], marker="v", s=90,
           color="tab:brown", label="Never-Think")
ax.set_xlabel("energy per query (mWh, b1eq; draft policies pay the 3-draft probe)")
ax.set_ylabel("accuracy"); ax.set_title("Router with format-constrained draft features")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_draft2.png", dpi=120)

with open(f"{OUT}/summary2.md", "w", encoding="utf-8") as f:
    f.write("# Constrained-draft router (step 3b)\n\n")
    f.write(f"N={N}; probe = 3 drafts, mean {probe.mean()*1000:.0f} mWh/query.\n\n")
    f.write(f"Agreement: easy {agreef[easy].mean():.2f} | hard-solvable "
            f"{agreef[solv & hard].mean():.2f} | hard-unsolvable {agreef[hard & ~solv].mean():.2f}; "
            f"solvability AUC among hard: {roc_auc_score(solv[hard], -agreef[hard]):.3f}\n\n")
    f.write("|policy|label AUC|EAA-AUC (honest)|\n|--|--|--|\n")
    for k in P:
        f.write(f"|{k}|{roc_auc_score(y, P[k]):.3f}|{res[k]:.4f}|\n")
    f.write(f"|oracle frontier||{eaa(of, e_lo, e_hi):.4f}|\n")
    f.write(f"\nDART all-3: acc {dart[1]:.3f} @ {dart[0]:.0f} mWh; "
            f"majority: acc {dart_m[1]:.3f} @ {dart_m[0]:.0f} mWh.\n")
print(f"\nwrote {OUT}/summary2.md, pareto_draft2.png")
