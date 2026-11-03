"""Is OTB's high routability real difficulty signal, or just TOPIC/SOURCE routing?

Uses OTB's own `domain` + `answer_type` metadata (fetched from HF) to decompose the 0.91 AUC:
  1. "which half?" (over vs under) as a ONE-feature router  -> how much of 0.91 is the coarse split
  2. "which domain?" as a lookup-table router (CV mean think-rate per domain)
  3. WITHIN-half / WITHIN-domain text routing -> is there signal beyond topic?
  4. the easy-math question: OTB over-half has 312 `numeric` items (easy math). Are they routed
     correctly (no_think), or does the router send all math to think?
Free/local. Out: results/pooled/otb_decomposition.md
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---- OTB labels (ours) + domain metadata (theirs), joined on question text
lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
t = pq.read_table("data/otb/train.parquet").to_pydict()
meta = {}
for q, m, s in zip(t["question"], t["metadata"], t["subset"]):
    d = json.loads(m)
    meta[" ".join(str(q).split())[:200]] = (d.get("domain"), d.get("answer_type"), s)

rows = []
miss = 0
for e in lab:
    k = " ".join(str(e["question"]).split())[:200]
    if k not in meta:
        miss += 1; continue
    dom, atype, sub = meta[k]
    rows.append(dict(q=e["question"], dom=dom or "?", atype=atype or "unknown", sub=sub,
                     y=1 if e["oracle"] == "think" else 0,
                     cn=e["correct_no_think"], ct=e["correct_think"]))
print(f"OTB joined {len(rows)} / {len(lab)} labeled (unmatched {miss})")
y = np.array([r["y"] for r in rows])
sub = np.array([r["sub"] for r in rows]); dom = np.array([r["dom"] for r in rows])
at = np.array([r["atype"] for r in rows])
is_under = (sub == "underthinkingbench").astype(float)

print("\n=== 1. OTB composition ===")
for s in ["overthinkingbench", "underthinkingbench"]:
    m = sub == s
    print(f"  {s:22} n={m.sum():4d}  think {y[m].mean()*100:4.1f}%   answer_types: {dict(Counter(at[m]))}")
print(f"  distinct domains: over {len(set(dom[sub=='overthinkingbench']))}, under {len(set(dom[sub=='underthinkingbench']))}")
print("  under-half domains:", dict(Counter(dom[sub == 'underthinkingbench']).most_common(8)))

print("\n=== 2. How much of OTB's AUC is just the coarse over/under split? ===")
print(f"  ONE-feature router 'is it the under-half?'      AUC {roc_auc_score(y, is_under):.3f}")

from sentence_transformers import SentenceTransformer
X = SentenceTransformer("all-MiniLM-L6-v2").encode([r["q"] for r in rows], batch_size=64,
                                                    normalize_embeddings=True).astype(np.float32)
def cv_auc(X_, y_, groups=None):
    if len(set(y_)) < 2: return float("nan")
    oof = np.zeros(len(y_))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X_, y_):
        oof[te] = LogisticRegression(max_iter=2000).fit(X_[tr], y_[tr]).predict_proba(X_[te])[:, 1]
    return roc_auc_score(y_, oof), oof
full_auc, oof_full = cv_auc(X, y)
print(f"  FULL text router (MiniLM+LogReg, 5-fold CV)     AUC {full_auc:.3f}")

# domain lookup-table router: score = train-fold think-rate of that domain
oof_dom = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    rate = defaultdict(lambda: y[tr].mean())
    for d in set(dom[tr]):
        m = dom[tr] == d
        rate[d] = y[tr][m].mean()
    oof_dom[te] = [rate[d] for d in dom[te]]
print(f"  'which DOMAIN?' lookup-table router             AUC {roc_auc_score(y, oof_dom):.3f}")
print(f"  'which ANSWER-TYPE?' lookup-table router        AUC ", end="")
oof_at = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    rate = defaultdict(lambda: y[tr].mean())
    for a in set(at[tr]):
        rate[a] = y[tr][at[tr] == a].mean()
    oof_at[te] = [rate[a] for a in at[te]]
print(f"{roc_auc_score(y, oof_at):.3f}")

print("\n=== 3. Is there signal BEYOND topic? (text routing WITHIN each half) ===")
for s in ["overthinkingbench", "underthinkingbench"]:
    m = sub == s
    a, _ = cv_auc(X[m], y[m])
    print(f"  within {s:22} n={m.sum():4d} think {y[m].mean()*100:4.1f}%  ->  text AUC {a:.3f}")

print("\n=== 4. THE EASY-MATH QUESTION: are OTB's easy math items routed correctly? ===")
print(f"  {'slice':34} {'n':>4} {'oracle think%':>13} {'router mean p(think)':>21}")
for name, m in [("over-half NUMERIC (easy math)", (sub == "overthinkingbench") & (at == "numeric")),
                ("over-half MCQ", (sub == "overthinkingbench") & (at == "mcq")),
                ("over-half open-ended", (sub == "overthinkingbench") & (at == "open-ended")),
                ("over-half open-ended-long", (sub == "overthinkingbench") & (at == "open-ended-long")),
                ("under-half (reasoning puzzles)", sub == "underthinkingbench")]:
    print(f"  {name:34} {int(m.sum()):>4} {y[m].mean()*100:>12.1f}% {oof_full[m].mean():>21.3f}")
# at a deployment threshold, what fraction of easy math gets sent to think?
for tau in [0.3, 0.5]:
    em = (sub == "overthinkingbench") & (at == "numeric")
    un = sub == "underthinkingbench"
    print(f"  tau={tau}: easy-math routed to think {100*(oof_full[em]>=tau).mean():.0f}%  |  "
          f"hard-puzzle routed to think {100*(oof_full[un]>=tau).mean():.0f}%")

lines = ["# Decomposing OTB's routability: real signal or topic routing?\n",
         f"Joined {len(rows)} OTB items with their own `domain`/`answer_type` metadata (HF parquet).\n",
         "## Composition", "```"]
for s in ["overthinkingbench", "underthinkingbench"]:
    m = sub == s
    lines.append(f"{s:22} n={m.sum():4d}  think {y[m].mean()*100:4.1f}%  types {dict(Counter(at[m]))}")
lines += ["```\n", "## How much of the AUC is topic?", "```",
          f"'is it the under-half?' 1-feature router : AUC {roc_auc_score(y, is_under):.3f}",
          f"'which domain?' lookup table            : AUC {roc_auc_score(y, oof_dom):.3f}",
          f"'which answer-type?' lookup table       : AUC {roc_auc_score(y, oof_at):.3f}",
          f"FULL text router (MiniLM+LogReg)        : AUC {full_auc:.3f}", "```\n",
          "## Signal beyond topic (within-half text routing)", "```"]
for s in ["overthinkingbench", "underthinkingbench"]:
    m = sub == s; a, _ = cv_auc(X[m], y[m])
    lines.append(f"within {s:22} n={m.sum():4d} think {y[m].mean()*100:4.1f}% -> AUC {a:.3f}")
lines += ["```\n", "## Easy math in OTB", "```"]
for name, m in [("over NUMERIC (easy math)", (sub=="overthinkingbench")&(at=="numeric")),
                ("under (reasoning puzzles)", sub=="underthinkingbench")]:
    lines.append(f"{name:26} n={int(m.sum()):4d} oracle think {y[m].mean()*100:5.1f}%  router mean p {oof_full[m].mean():.3f}")
lines.append("```")
os.makedirs("results/pooled", exist_ok=True)
open("results/pooled/otb_decomposition.md", "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote results/pooled/otb_decomposition.md")
