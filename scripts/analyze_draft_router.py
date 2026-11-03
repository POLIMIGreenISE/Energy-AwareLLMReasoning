"""Step 3 — DART-style draft-agreement features for the router (offline, no GPU).

Key insight: our scaled run already contains two independent no_think generations
per query (REPS=2). Those are exactly DART's "two cheap drafts": if they agree,
the query is probably easy (answer directly); if they disagree, it's uncertain.
Agreement uses only model outputs — legal at inference time (no gold involved).

We evaluate:
  A. text-only router (MiniLM embedding + qlen)          [reproduces notebook 03]
  B. text + draft features
  C. draft features only (embedding-free, like DART)
  D. DART rule, zero training: think iff drafts disagree
with TWO energy accountings:
  naive : chosen mode's energy only (drafts free) — what token papers implicitly do
  honest: draft-using policies pay both drafts: no_think -> 2*E_nt, think -> 2*E_nt + E_th

Run:  python scripts/analyze_draft_router.py
Out:  results/draft_router/{summary.md, pareto_draft.png}
"""
import json, os, re, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scoring

OUT = "results/draft_router"; os.makedirs(OUT, exist_ok=True)
SEED = 0

# ---------------------------------------------------------------- draft features
gens = [json.loads(l) for l in open("results/paired/gen_v2.jsonl", encoding="utf-8") if l.strip()]
nt = defaultdict(dict)
for r in gens:
    if r["mode"] == "no_think":
        nt[r["qid"]][r["rep"]] = r

def norm(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", str(s))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())

def as_num(s):
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", norm(s).replace(" ", ""))
    return float(m.group()) if m else None

def draft_feats(qid, answer_type):
    reps = nt.get(qid, {})
    if len(reps) < 2:
        return None
    a = [scoring.extract_answer(reps[i]["answer_text"], answer_type) for i in sorted(reps)][:2]
    n0, n1 = norm(a[0]), norm(a[1])
    v0, v1 = as_num(a[0]), as_num(a[1])
    if v0 is not None and v1 is not None:
        exact = float(abs(v0 - v1) <= 1e-6 * max(1.0, abs(v0)))
    else:
        exact = float(n0 == n1 and n0 != "")
    t0, t1 = set(n0.split()), set(n1.split())
    jac = len(t0 & t1) / len(t0 | t1) if (t0 | t1) else 0.0
    toks = [reps[i]["completion_tokens"] for i in sorted(reps)][:2]
    return dict(agree=exact, jaccard=jac,
                nt_tokens=st.mean(toks) / 1000.0, nt_tok_diff=abs(toks[0] - toks[1]) / 1000.0)

paired = json.load(open("results/paired/paired_oracle_final.json"))
rows = []
for p in paired:
    at = next((g["answer_type"] for g in gens if g["qid"] == p["qid"]), "unknown")
    f = draft_feats(p["qid"], at)
    if f is not None:
        rows.append({**p, **f})
print(f"queries with both drafts: {len(rows)}/{len(paired)}")

cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]); et = np.array([r["energy_wh_think"] for r in rows])
y = np.array([r["oracle"] == "think" for r in rows]).astype(int)
agree = np.array([r["agree"] for r in rows]); jac = np.array([r["jaccard"] for r in rows])
sub = np.array([r["subset"] for r in rows])
N = len(rows)

# ---------------------------------------------------------------- diagnostics
solv = (ct > 0.5) & (cn <= 0.5)
unsolv = (sub == "underthinkingbench") & ~solv
easy = sub == "overthinkingbench"
print("\ndraft-agreement rate by group:")
for name, m in [("easy (overthinking)", easy),
                ("hard-solvable (oracle=think)", solv & (sub == "underthinkingbench")),
                ("hard-unsolvable", unsolv)]:
    print(f"  {name:30} n={int(m.sum()):3}  agree={agree[m].mean():.2f}  jaccard={jac[m].mean():.2f}")

# ---------------------------------------------------------------- energy accounting
def metrics(choose, drafts=False):
    choose = np.asarray(choose, bool)
    acc = np.where(choose, ct, cn).mean()
    if drafts:  # honest: both drafts always generated
        e = np.where(choose, 2 * en + et, 2 * en).mean() * 1000
    else:
        e = np.where(choose, et, en).mean() * 1000
    return e, acc

# ---------------------------------------------------------------- routers
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([r["question"] for r in rows], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(r["question"]) for r in rows]).reshape(-1, 1) / 1000.0
Xd = np.column_stack([agree, jac, [r["nt_tokens"] for r in rows], [r["nt_tok_diff"] for r in rows]])

def oof_probs(X):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

P = {"A text-only": oof_probs(np.hstack([Xe, qlen])),
     "B text+drafts": oof_probs(np.hstack([Xe, qlen, Xd])),
     "C drafts-only": oof_probs(Xd)}
for k, p in P.items():
    print(f"{k:15} label AUC (out-of-fold): {roc_auc_score(y, p):.3f}")

taus = np.linspace(0, 1, 101)
curves = {}
for k, p in P.items():
    drafts = not k.startswith("A")
    curves[k] = np.array([metrics(p >= t, drafts=drafts) for t in taus])
dart = metrics(agree < 0.5, drafts=True)                    # think iff disagree
dart_naive = metrics(agree < 0.5, drafts=False)

# oracle frontier (naive accounting — the ceiling is probe-free)
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
    # normalize over fixed [lo,hi] -- see analyze_probe_router for rationale.
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(f) if f else _NEVER_ACC)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(np.array(acc), es) / (hi - lo)
print(f"\nEAA-AUC over [{e_lo:.0f}, {e_hi:.0f}] mWh  (oracle frontier: {eaa(of, e_lo, e_hi):.4f})")
res = {}
for k in P:
    res[k] = eaa([tuple(q) for q in curves[k]], e_lo, e_hi)
    print(f"  {k:15} {res[k]:.4f}")
print(f"  DART rule       point: acc={dart[1]:.3f} @ {dart[0]:.0f} mWh (honest) / {dart_naive[0]:.0f} mWh (naive)")

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"A text-only": "tab:green", "B text+drafts": "tab:blue", "C drafts-only": "tab:purple"}
for k, c in colors.items():
    ax.plot(curves[k][:, 0], curves[k][:, 1], "-", lw=1.6, color=c,
            label=f"{k} (EAA {res[k]:.3f})")
ax.plot([e for e, a in of], [a for e, a in of], "--", color="tab:red", label="oracle frontier")
ax.scatter([dart[0]], [dart[1]], marker="D", s=70, color="black", zorder=5,
           label="DART rule (honest energy)")
ax.scatter([metrics(np.ones(N,bool))[0]], [metrics(np.ones(N,bool))[1]], marker="^", s=90, color="tab:orange", label="Always-Think")
ax.scatter([metrics(np.zeros(N,bool))[0]], [metrics(np.zeros(N,bool))[1]], marker="v", s=90, color="tab:brown", label="Never-Think")
ax.set_xlabel("energy per query (mWh, batch-1-equiv; draft policies pay probe overhead)")
ax.set_ylabel("accuracy"); ax.set_title("Router with DART-style draft features")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pareto_draft.png", dpi=120)

with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
    f.write("# Draft-agreement router (step 3)\n\n")
    f.write(f"N={N} queries with 2 no_think drafts. Agreement by group: "
            f"easy {agree[easy].mean():.2f}, hard-solvable {agree[solv & (sub=='underthinkingbench')].mean():.2f}, "
            f"hard-unsolvable {agree[unsolv].mean():.2f}\n\n")
    f.write("|policy|label AUC|EAA-AUC|\n|--|--|--|\n")
    for k in P:
        f.write(f"|{k}|{roc_auc_score(y, P[k]):.3f}|{res[k]:.4f}|\n")
    f.write(f"|oracle frontier||{eaa(of, e_lo, e_hi):.4f}|\n")
    f.write(f"\nDART rule (think iff drafts disagree): acc {dart[1]:.3f} @ {dart[0]:.0f} mWh honest "
            f"({dart_naive[0]:.0f} naive). Draft policies pay 2*E_no_think always.\n")
print(f"\nwrote {OUT}/summary.md, pareto_draft.png")
