"""WHY does the OTB-trained router fail on Mohseni? The "easy-math trap" hypothesis.

OTB's HF release contains NO hard math: its math items (312 `numeric`) are all EASY and ~all
labeled no_think (3.2% think). Its think-half is Reasoning-Gym puzzles. So an OTB-trained router
plausibly learns "math/numeric-looking -> NO think" and "puzzle-looking -> think".

If so, on Mohseni's Big-Math/AIME (HARD math that genuinely needs thinking) it should assign
SYSTEMATICALLY LOW p(think) -- i.e. be inverted, which explains a below-chance/at-chance transfer
AUC better than "no signal".

Also probes whether OTB's within-under-half signal (AUC 0.76) is really puzzle-TEMPLATE routing.
Free/local. Out: results/pooled/transfer_inversion.md
"""
import json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# ---------- OTB (train) ----------
lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
t = pq.read_table("data/otb/train.parquet").to_pydict()
meta = {" ".join(str(q).split())[:200]: (json.loads(m).get("answer_type"), s)
        for q, m, s in zip(t["question"], t["metadata"], t["subset"])}
otb_q, otb_y, otb_at, otb_sub = [], [], [], []
for e in lab:
    k = " ".join(str(e["question"]).split())[:200]
    at, s = meta.get(k, ("unknown", e["subset"]))
    otb_q.append(e["question"]); otb_y.append(1 if e["oracle"] == "think" else 0)
    otb_at.append(at); otb_sub.append(s)
otb_y = np.array(otb_y); otb_at = np.array(otb_at); otb_sub = np.array(otb_sub)

# ---------- Mohseni (test) ----------
CD = "results/crossdataset"
rows = [json.loads(l) for l in open(f"{CD}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows: byq[r["qid"]][r["mode"]] = r
cache = JudgeCache(f"{CD}/judge_cache_mohseni.jsonl")
mo_q, mo_y, mo_src = [], [], []
for q, mm in byq.items():
    if "think" not in mm or "no_think" not in mm: continue
    nt, th = mm["no_think"], mm["think"]
    v = cache.get(nt["answer_text"], th["answer_text"])
    if v == "__miss__": continue
    mo_q.append(nt["question"]); mo_y.append(1 if (int(v[1]) and not int(v[0])) else 0)
    mo_src.append(nt.get("source") or "?")
mo_y = np.array(mo_y); mo_src = np.array(mo_src)

enc = SentenceTransformer("all-MiniLM-L6-v2")
Xo = enc.encode(otb_q, batch_size=64, normalize_embeddings=True).astype(np.float32)
Xm = enc.encode(mo_q, batch_size=64, normalize_embeddings=True).astype(np.float32)

# router trained on ALL of OTB, applied to Mohseni
router = LogisticRegression(max_iter=2000).fit(Xo, otb_y)
p_mo = router.predict_proba(Xm)[:, 1]

print("=== What did the OTB router learn? (its own training data, by answer type) ===")
p_otb_in = router.predict_proba(Xo)[:, 1]
for name, m in [("over NUMERIC (easy math)", (otb_sub == "overthinkingbench") & (otb_at == "numeric")),
                ("over other (non-math easy)", (otb_sub == "overthinkingbench") & (otb_at != "numeric")),
                ("under (reasoning puzzles)", otb_sub == "underthinkingbench")]:
    print(f"  {name:28} n={int(m.sum()):4d}  oracle think {otb_y[m].mean()*100:5.1f}%  router p(think) {p_otb_in[m].mean():.3f}")

print("\n=== THE TRAP: OTB router applied to Mohseni, by source ===")
print(f"  {'source':22} {'n':>4} {'ORACLE think%':>14} {'router p(think)':>16} {'AUC':>7}")
for s in sorted(set(mo_src)):
    m = mo_src == s
    if m.sum() < 20: continue
    a = roc_auc_score(mo_y[m], p_mo[m]) if len(set(mo_y[m])) > 1 else float("nan")
    print(f"  {s:22} {int(m.sum()):>4} {mo_y[m].mean()*100:>13.1f}% {p_mo[m].mean():>16.3f} {a:>7.3f}")
math_m = np.isin(mo_src, ["Big-Math-RL", "AIME-1983-2024"])
chat_m = ~math_m
print(f"\n  MATH  (needs thinking most): oracle think {mo_y[math_m].mean()*100:.0f}%  but router says p={p_mo[math_m].mean():.3f}")
print(f"  CHAT                       : oracle think {mo_y[chat_m].mean()*100:.0f}%  and router says p={p_mo[chat_m].mean():.3f}")
print(f"  -> router ranks MATH {'LOWER' if p_mo[math_m].mean() < p_mo[chat_m].mean() else 'higher'} than chat, "
      f"though math needs thinking {'MORE' if mo_y[math_m].mean() > mo_y[chat_m].mean() else 'less'}.")
print(f"  overall transfer AUC {roc_auc_score(mo_y, p_mo):.3f}  |  'route math->think' oracle-free rule AUC "
      f"{roc_auc_score(mo_y, math_m.astype(float)):.3f}")

print("\n=== Is OTB's within-under-half signal (AUC .76) just puzzle-TEMPLATE routing? ===")
un = otb_sub == "underthinkingbench"
qs_un = [q for q, u in zip(otb_q, un) if u]; y_un = otb_y[un]; X_un = Xo[un]
sig = np.array([" ".join(re.sub(r"\d+", "#", q).split()[:6]).lower() for q in qs_un])
cnt = Counter(sig); big = [s for s, c in cnt.items() if c >= 15]
print(f"  {len(cnt)} distinct 6-word templates among {len(qs_un)} puzzles; {len(big)} templates with n>=15")
oof_t = np.zeros(len(y_un))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X_un, y_un):
    rate = defaultdict(lambda: y_un[tr].mean())
    for s_ in set(sig[tr]): rate[s_] = y_un[tr][sig[tr] == s_].mean()
    oof_t[te] = [rate[s_] for s_ in sig[te]]
print(f"  'which TEMPLATE?' lookup-table router          AUC {roc_auc_score(y_un, oof_t):.3f}")
oof_x = np.zeros(len(y_un))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X_un, y_un):
    oof_x[te] = LogisticRegression(max_iter=2000).fit(X_un[tr], y_un[tr]).predict_proba(X_un[te])[:, 1]
print(f"  full text router within under-half            AUC {roc_auc_score(y_un, oof_x):.3f}")
# within the biggest templates only (topic held constant)
inner = []
for s_ in big:
    m = sig == s_
    if len(set(y_un[m])) < 2 or m.sum() < 25: continue
    oof_i = np.zeros(int(m.sum()))
    Xi, yi = X_un[m], y_un[m]
    try:
        for tr, te in StratifiedKFold(min(5, int(min(Counter(yi).values()))), shuffle=True, random_state=42).split(Xi, yi):
            oof_i[te] = LogisticRegression(max_iter=2000).fit(Xi[tr], yi[tr]).predict_proba(Xi[te])[:, 1]
        inner.append((s_[:38], int(m.sum()), yi.mean(), roc_auc_score(yi, oof_i)))
    except Exception:
        pass
if inner:
    print("  WITHIN a single puzzle template (topic held constant):")
    for s_, n_, r_, a_ in sorted(inner, key=lambda z: -z[1])[:6]:
        print(f"    {s_:40} n={n_:3d} think {r_*100:4.0f}%  AUC {a_:.3f}")
    print(f"    mean within-template AUC {np.mean([a for *_ , a in inner]):.3f} (n-weighted "
          f"{np.average([a for *_ , a in inner], weights=[n for _, n, _, _ in inner]):.3f})")
