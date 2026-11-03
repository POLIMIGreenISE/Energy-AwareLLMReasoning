"""Is Mohseni unroutable, or is our ROUTER just too weak? Throw many routers at it.

Same Mohseni text + labels, 5-fold CV, but sweep across feature spaces AND classifiers:
  features : MiniLM (384d), mpnet (768d, stronger), TF-IDF word n-grams (very different space)
  models   : LogReg, MLP, HistGradientBoosting, RBF-SVM, LinearSVC
If the BEST of all combinations still can't clear ~0.60 on Mohseni, the ceiling is a property of
the data (low mutual information between text and label), not of the LogReg+MiniLM choice.
Reports our energy-aware label AND Mohseni's own label. Free/local (mpnet downloads ~420MB once).
"""
import json, os, sys, warnings
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CD = "results/crossdataset"
rows = [json.loads(l) for l in open(f"{CD}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows: byq[r["qid"]][r["mode"]] = r
cache = JudgeCache(f"{CD}/judge_cache_mohseni.jsonl")
Q, OUR, HIS = [], [], []
for q, mm in byq.items():
    if "think" not in mm or "no_think" not in mm: continue
    nt, th = mm["no_think"], mm["think"]
    v = cache.get(nt["answer_text"], th["answer_text"])
    if v == "__miss__": continue
    Q.append(nt["question"]); OUR.append(1 if (int(v[1]) and not int(v[0])) else 0)
    HIS.append(1 if nt.get("his_label") == "think" else 0)
OUR = np.array(OUR); HIS = np.array(HIS)
print(f"Mohseni N={len(Q)} | our think {OUR.mean()*100:.0f}% | his think {HIS.mean()*100:.0f}%")

from sentence_transformers import SentenceTransformer
feats = {}
feats["MiniLM-384"] = SentenceTransformer("all-MiniLM-L6-v2").encode(Q, batch_size=64, normalize_embeddings=True).astype(np.float32)
try:
    feats["mpnet-768"] = SentenceTransformer("all-mpnet-base-v2").encode(Q, batch_size=32, normalize_embeddings=True).astype(np.float32)
except Exception as e:
    print("mpnet unavailable:", repr(e)[:60])
feats["tfidf"] = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2).fit_transform(Q)

def auc_cv(X, y, clf):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    if hasattr(clf, "predict_proba"):
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        p = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
    return roc_auc_score(y, p)

def models(sparse):
    m = {"LogReg": LogisticRegression(max_iter=2000, C=1.0),
         "MLP": make_pipeline(StandardScaler(with_mean=not sparse),
                              MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True, random_state=0)),
         "LinearSVC": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3)}
    if not sparse:
        m["HistGBM"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
        m["RBF-SVM"] = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=0)
    return m

for label_name, y in [("our energy-aware oracle", OUR), ("Mohseni's own labels", HIS)]:
    print(f"\n=== Mohseni CV AUC predicting: {label_name}  (best per cell) ===")
    best = 0
    header = f"  {'features':12} | " + " ".join(f"{n:>9}" for n in ["LogReg", "MLP", "LinearSVC", "HistGBM", "RBF-SVM"])
    print(header)
    for fname, X in feats.items():
        sparse = fname == "tfidf"
        row = {}
        for mname, clf in models(sparse).items():
            try:
                a = auc_cv(X, y, clf); row[mname] = a; best = max(best, a)
            except Exception as e:
                row[mname] = float("nan")
        cells = " ".join(f"{row.get(n, float('nan')):>9.3f}" for n in ["LogReg", "MLP", "LinearSVC", "HistGBM", "RBF-SVM"])
        print(f"  {fname:12} | {cells}")
    print(f"  >>> BEST across ALL features x classifiers: AUC {best:.3f}")
