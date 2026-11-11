"""Cross-dataset generalization (FREE, no GPU): does our OTB-energy-aware-trained
router's DECISIONS agree with Mohseni's independently-built routing labels?

We train the LogReg+MiniLM router on ALL of OTB (paired_oracle_final.json), then
predict think/no_think on Mohseni's held-out test prompts (data/mohseni/test.parquet)
and compare to the dataset `label`. This is a decision-AGREEMENT test, not an accuracy test:
those labels use a different definition (other models, token/reward-based, not our
energy-aware Qwen3-8B), so agreement is a directional generalization signal, not
ground truth. Reported overall and per source (math sources should align more,
since OTB's hard half is math-heavy; chat sources are the true out-of-distribution).

Run:  python scripts/test_crossdataset_router.py
Out:  results/crossdataset/mohseni_agreement.md
"""
import json, os
import numpy as np
import pyarrow.parquet as pq
from collections import Counter, defaultdict

OUT = "results/crossdataset"; os.makedirs(OUT, exist_ok=True)

# ---- train the OTB router on ALL OTB labels
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

otb = json.load(open("results/paired/paired_oracle_final.json"))
emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([p["question"] for p in otb], normalize_embeddings=True, show_progress_bar=False)
qlen = np.array([len(p["question"]) for p in otb]).reshape(-1, 1) / 1000.0
Xtr = np.hstack([Xe, qlen])
ytr = np.array([p["oracle"] == "think" for p in otb]).astype(int)
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
print(f"trained OTB router on {len(otb)} queries ({ytr.mean()*100:.0f}% think)")

# ---- load Mohseni test split
t = pq.read_table("data/mohseni/test.parquet").to_pydict()
prompts, labels, sources = t["prompt"], t["label"], t["source"]
y_moh = np.array([l == "think" for l in labels]).astype(int)
N = len(prompts)
print(f"Mohseni test: {N} prompts ({y_moh.mean()*100:.0f}% think), sources {dict(Counter(sources))}")

# ---- predict with the OTB router
Xe_m = emb.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
qlen_m = np.array([len(p) for p in prompts]).reshape(-1, 1) / 1000.0
Xm = np.hstack([Xe_m, qlen_m])
p_think = clf.predict_proba(Xm)[:, 1]

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def report_block(mask, name):
    yy, pp = y_moh[mask], p_think[mask]
    if len(set(yy)) < 2:
        return f"{name:20} n={mask.sum():4}  (single-class, AUC undefined)"
    auc = roc_auc_score(yy, pp)
    # decision agreement at the router's default 0.5 and at the OTB-tuned operating threshold
    acc50 = ((pp >= 0.5).astype(int) == yy).mean()
    bal = balanced_accuracy_score(yy, (pp >= 0.5).astype(int))
    return f"{name:20} n={mask.sum():4}  AUC {auc:.3f}  agree@0.5 {acc50:.3f}  balanced {bal:.3f}  (Mohseni think% {yy.mean()*100:.0f})"

allm = np.ones(N, bool)
print("\n--- decision agreement (our OTB router vs Mohseni labels) ---")
print(report_block(allm, "ALL"))
math_mask = np.array([s in ("Big-Math-RL", "AIME-1983-2024") for s in sources])
chat_mask = np.array([s in ("Nectar", "WildChat-filtered") for s in sources])
print(report_block(math_mask, "MATH (Big-Math+AIME)"))
print(report_block(chat_mask, "CHAT (Nectar+WildChat)"))
for src in ["Big-Math-RL", "AIME-1983-2024", "Nectar", "WildChat-filtered"]:
    print(report_block(np.array([s == src for s in sources]), src))

# best achievable agreement by sweeping the threshold (does the signal exist even if miscalibrated?)
auc_all = roc_auc_score(y_moh, p_think)
best_acc, best_tau = max(((((p_think >= t).astype(int) == y_moh).mean(), t) for t in np.linspace(0, 1, 101)))
print(f"\noverall AUC {auc_all:.3f}; best threshold-swept agreement {best_acc:.3f} at tau={best_tau:.2f}")
print("(AUC is the calibration-free signal; agree@0.5 depends on OTB's 88% no_think prior "
      "carrying over, which mismatches Mohseni's balanced 50/50)")

with open(f"{OUT}/mohseni_agreement.md", "w", encoding="utf-8") as f:
    f.write("# Cross-dataset generalization: OTB router vs Mohseni labels (decision agreement)\n\n")
    f.write(f"OTB router (LogReg+MiniLM, trained on all {len(otb)} OTB queries) predicting on "
            f"Mohseni's {N} held-out test prompts. Its labels use a DIFFERENT definition "
            "(other models, not our energy-aware Qwen3-8B), so this is decision AGREEMENT, "
            "a directional generalization signal -- not accuracy.\n\n")
    f.write("|subset|n|AUC|agree@0.5|balanced acc|Mohseni think%|\n|--|--|--|--|--|--|\n")
    def row(mask, name):
        yy, pp = y_moh[mask], p_think[mask]
        if len(set(yy)) < 2: return f"|{name}|{mask.sum()}|--|--|--|{yy.mean()*100:.0f}|\n"
        return (f"|{name}|{mask.sum()}|{roc_auc_score(yy,pp):.3f}|"
                f"{((pp>=0.5).astype(int)==yy).mean():.3f}|"
                f"{balanced_accuracy_score(yy,(pp>=0.5).astype(int)):.3f}|{yy.mean()*100:.0f}|\n")
    f.write(row(allm, "ALL"))
    f.write(row(math_mask, "MATH (Big-Math+AIME)"))
    f.write(row(chat_mask, "CHAT (Nectar+WildChat)"))
    for src in ["Big-Math-RL", "AIME-1983-2024", "Nectar", "WildChat-filtered"]:
        f.write(row(np.array([s == src for s in sources]), src))
    f.write(f"\nOverall AUC {auc_all:.3f}; best threshold-swept agreement {best_acc:.3f} @ tau={best_tau:.2f}.\n")
    f.write("\n**Reading:** AUC is the calibration-free signal (does our router RANK Mohseni's "
            "think-queries above its no_think ones?). agree@0.5 also depends on our OTB prior "
            "(88% no_think) transferring, which mismatches Mohseni's balanced 50/50 mix -- so the "
            "threshold-swept best agreement is the fairer point estimate of the transferable signal.\n")
print(f"\nwrote {OUT}/mohseni_agreement.md")
