"""Is Mohseni's data unroutable, or is OUR label just a harder target than the published one?

Same Mohseni text, same features (MiniLM), same CV. Predict three different labels:
  - OUR energy-aware oracle : think iff (think-correct AND no-think-wrong)   [does thinking FLIP it]
  - Mohseni's published label: the his_label field of the dataset          [published think/no_think]
  - a pure DIFFICULTY label  : think iff (no-think is wrong)  == Mohseni's math rule incl. both-wrong->think
If published-label / difficulty AUC >> our-oracle AUC (~0.55), the gap is the LABEL DEFINITION, not our
method: the published target ~ "is this hard" (text-predictable); ours ~ "does thinking help" (model-behavioral).
Free/local.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CD = "results/crossdataset"
rows = [json.loads(l) for l in open(f"{CD}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows: byq[r["qid"]][r["mode"]] = r
cache = JudgeCache(f"{CD}/judge_cache_mohseni.jsonl")

Q, SRC, OUR, HIS, DIFF = [], [], [], [], []
for q, mm in byq.items():
    if "think" not in mm or "no_think" not in mm: continue
    nt, th = mm["no_think"], mm["think"]
    v = cache.get(nt["answer_text"], th["answer_text"])
    if v == "__miss__": continue
    cn, ct = int(v[0]), int(v[1])
    Q.append(nt["question"]); SRC.append(nt.get("source") or "?")
    OUR.append(1 if (ct and not cn) else 0)                 # ours: thinking flips wrong->right
    DIFF.append(1 if (cn == 0) else 0)                      # difficulty: no-think is wrong (= Mohseni's math rule)
    HIS.append(1 if nt.get("his_label") == "think" else 0)  # Mohseni's published label
Q = np.array(Q, dtype=object); SRC = np.array(SRC)
OUR = np.array(OUR); HIS = np.array(HIS); DIFF = np.array(DIFF)
print(f"Mohseni N={len(Q)} | think-rate  ours {OUR.mean()*100:.0f}%  published {HIS.mean()*100:.0f}%  difficulty {DIFF.mean()*100:.0f}%")
print(f"our-vs-published agreement {np.mean(OUR==HIS)*100:.0f}% | our-vs-difficulty {np.mean(OUR==DIFF)*100:.0f}% | published-vs-difficulty {np.mean(HIS==DIFF)*100:.0f}%")

from sentence_transformers import SentenceTransformer
X = SentenceTransformer("all-MiniLM-L6-v2").encode(list(Q), batch_size=64, normalize_embeddings=True).astype(np.float32)

def cv_auc(y, mask=None):
    idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
    yy = y[idx]
    if len(set(yy)) < 2: return float("nan")
    oof = np.zeros(len(idx))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X[idx], yy):
        oof[te] = LogisticRegression(max_iter=2000).fit(X[idx][tr], yy[tr]).predict_proba(X[idx][te])[:, 1]
    return roc_auc_score(yy, oof)

print("\n=== Can a text router predict each label? (5-fold CV AUC, all of Mohseni) ===")
print(f"  OUR energy-aware oracle (does thinking FLIP it) : AUC {cv_auc(OUR):.3f}")
print(f"  Mohseni's published label (think/no_think)  : AUC {cv_auc(HIS):.3f}")
print(f"  pure DIFFICULTY (is no-think wrong)             : AUC {cv_auc(DIFF):.3f}")

print("\n=== Per source: think-rate and text-predictability of EACH label ===")
print(f"  {'source':20} {'n':>4} | {'our%':>5} {'pub%':>5} {'diff%':>5} | {'ourAUC':>6} {'pubAUC':>6} {'diffAUC':>7}")
for s in sorted(set(SRC)):
    m = SRC == s
    if m.sum() < 30: continue
    print(f"  {s:20} {int(m.sum()):>4} | {OUR[m].mean()*100:>4.0f}% {HIS[m].mean()*100:>4.0f}% {DIFF[m].mean()*100:>4.0f}% | "
          f"{cv_auc(OUR,m):>6.3f} {cv_auc(HIS,m):>6.3f} {cv_auc(DIFF,m):>7.3f}")
