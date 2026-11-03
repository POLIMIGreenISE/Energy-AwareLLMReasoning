"""Why can't the router route Mohseni even when trained on it?

Contrast, same features / same model / same CV protocol:
  - OTB-specialist     : LogReg trained ONLY on OTB, CV within OTB          (is OTB routable?)
  - Mohseni-specialist : LogReg trained ONLY on Mohseni, CV within Mohseni  (is Mohseni routable?)
  - + within-Big-Math class separability (can text tell think from no_think among uniform hard math?)

If the Mohseni-specialist (all capacity on Mohseni, zero dilution) is still ~chance, then pooling
did NOT "break" Mohseni -- Mohseni's think/no_think label just isn't a function of the query text.
Reuses the cached pooled embeddings; free/local.
"""
import json, os, re, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CD = "results/crossdataset"
def numeric_correct(ans, gold):
    g = str(gold).replace(",", ""); nums = re.findall(r"-?\d+(?:\.\d+)?", str(ans).replace(",", "")[-200:])
    try: return any(abs(float(x)-float(g)) <= 1e-6*max(1,abs(float(g))) for x in nums)
    except ValueError: return None
def rc(q, cn, ct, en, et, ds, src, pn):
    return dict(q=q, cn=float(cn), ct=float(ct), en=en*1000, et=et*1000, ds=ds, src=src,
                oracle=("think" if (ct and not cn) else "no_think"), pn=pn)
def load_otb():
    lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
    pn = {json.loads(l)["qid"]: json.loads(l)["p_need_think"] for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
    return [rc(e["question"], e["correct_no_think"], e["correct_think"], e["energy_wh_no_think"],
              e["energy_wh_think"], "otb", "otb_"+e["subset"][:4], pn[e["qid"]]) for e in lab if e["qid"] in pn]
def load_cd(ds):
    rows = [json.loads(l) for l in open(f"{CD}/gen_test_{ds}.jsonl", encoding="utf-8") if l.strip()]
    byq = defaultdict(dict)
    for r in rows: byq[r["qid"]][r["mode"]] = r
    pn = {json.loads(l)["qid"]: json.loads(l)["p_need_think"] for l in open(f"{CD}/pneed_test_{ds}.jsonl", encoding="utf-8")}
    cache = JudgeCache(f"{CD}/judge_cache_{ds}.jsonl") if ds == "mohseni" else None
    out = []
    for q, mm in byq.items():
        if "think" not in mm or "no_think" not in mm or q not in pn: continue
        nt, th = mm["no_think"], mm["think"]
        if ds == "gsm8k":
            cn, ct = numeric_correct(nt["answer_text"], nt["gold"]), numeric_correct(th["answer_text"], th["gold"])
            if cn is None or ct is None: continue
        else:
            v = cache.get(nt["answer_text"], th["answer_text"])
            if v == "__miss__": continue
            cn, ct = v
        out.append(rc(nt["question"], cn, ct, nt["energy_b1eq_wh"], th["energy_b1eq_wh"], ds, nt.get("source") or ds, pn[q]))
    return out

DATA = load_otb() + load_cd("mohseni") + load_cd("gsm8k")
qs = [d["q"] for d in DATA]
cache_files = [f for f in os.listdir("results/pooled") if f.startswith("emb_MiniLM_")]
X = np.load(f"results/pooled/{cache_files[0]}")["X"] if cache_files else None
if X is None or X.shape[0] != len(DATA):
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer("all-MiniLM-L6-v2").encode(qs, batch_size=64, normalize_embeddings=True).astype(np.float32)
ds_arr = np.array([d["ds"] for d in DATA]); src_arr = np.array([d["src"] for d in DATA])
y = np.array([d["oracle"] == "think" for d in DATA]).astype(int)
cn = np.array([d["cn"] for d in DATA]); ct = np.array([d["ct"] for d in DATA])
en = np.array([d["en"] for d in DATA]); et = np.array([d["et"] for d in DATA])

def cv_specialist(mask):
    """5-fold CV LogReg trained AND tested only within mask. Returns OOF proba (full-length, nan outside)."""
    idx = np.where(mask)[0]; yy = y[idx]
    oof = np.full(len(DATA), np.nan)
    if len(set(yy)) < 2: return oof
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X[idx], yy):
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[idx][tr], yy[tr])
        oof[idx[te]] = lr.predict_proba(X[idx][te])[:, 1]
    return oof

def eaa_vs_random(idx, score):
    c0, c1, e0, e1 = cn[idx], ct[idx], en[idx], et[idx]; n = len(idx)
    def metrics(ch): ch = np.asarray(ch, bool); return float(np.where(ch, e1, e0).mean()), float(np.where(ch, c1, c0).mean())
    e_lo, never = metrics(np.zeros(n, bool)); e_hi, always = metrics(np.ones(n, bool))
    def eaa(pts):
        best, o = -1, []
        for e, a in sorted(pts):
            if a > best+1e-12: o.append((e, a)); best = a
        es = np.linspace(e_lo, e_hi, 200); acc = [max([a for pe, a in o if pe <= e+1e-9] or [never]) for e in es]
        tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(tz(np.array(acc), es)/(e_hi-e_lo+1e-12))
    dA, dE = c1-c0, e1-e0; pos = (dA > 0) & (dE > 0)
    lams = sorted(set([0.0]+(dA[pos]/dE[pos]).tolist())) if pos.any() else [0.0]
    orc = eaa([metrics((dA-l*dE) > 1e-12) for l in lams]+[(e_lo, never), (e_hi, always)])
    rng = np.random.default_rng(0)
    rnd = np.mean([eaa([metrics(rng.random(n) >= t) for t in np.linspace(0,1,60)]) for _ in range(20)])
    ts = np.unique(np.concatenate([np.linspace(0,1,101), np.unique(score)]))
    e = eaa([metrics(score >= t) for t in ts])
    g = lambda v: 100*(v-never)/(orc-never+1e-12)
    return dict(auc=roc_auc_score(y[idx], score), eaa=e, vs_rand=g(e)-g(rnd), orc=orc, rnd_gain=g(rnd),
                never=never, always=always)

print("dataset sizes:", dict(Counter(ds_arr)))
print("\n=== Is each dataset routable from text? (specialist: train & CV WITHIN the dataset) ===")
for ds in ["otb", "mohseni", "gsm8k"]:
    m = ds_arr == ds; oof = cv_specialist(m); idx = np.where(m)[0]
    r = eaa_vs_random(idx, oof[idx])
    print(f"  {ds:8} specialist:  AUC {r['auc']:.3f} | EAA {r['eaa']:.4f} ({r['vs_rand']:+.0f}pp vs random) "
          f"| think {y[idx].mean()*100:.0f}%")

print("\n=== Mohseni routing: does MORE Mohseni-specific training help? ===")
mo = np.where(ds_arr == "mohseni")[0]
# OTB-only -> mohseni (transfer, 0 mohseni in train)
oof_otbonly = np.full(len(DATA), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, np.array([f"{d}_{v}" for d, v in zip(ds_arr, y)])):
    otr = tr[ds_arr[tr] == "otb"]
    oof_otbonly[te] = LogisticRegression(max_iter=2000).fit(X[otr], y[otr]).predict_proba(X[te])[:, 1]
oof_pool = np.full(len(DATA), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, np.array([f"{d}_{v}" for d, v in zip(ds_arr, y)])):
    oof_pool[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
oof_spec = cv_specialist(ds_arr == "mohseni")
for name, oof in [("OTB-only (0 Mohseni in train)", oof_otbonly), ("pooled (OTB+Moh+GSM)", oof_pool),
                  ("Mohseni-specialist (Mohseni only)", oof_spec)]:
    print(f"  {name:34} Mohseni-slice AUC {roc_auc_score(y[mo], oof[mo]):.3f}")

print("\n=== WHY: within EACH source, is think/no_think textually separable? (kind = math vs chat) ===")
kind = {"Big-Math-RL": "math", "AIME-1983-2024": "math", "Nectar": "chat", "WildChat-filtered": "chat",
        "otb_over": "simple-Q", "otb_unde": "hard-math", "gsm8k": "math"}
for src in ["otb_over", "otb_unde", "Big-Math-RL", "AIME-1983-2024", "Nectar", "WildChat-filtered", "gsm8k"]:
    m = src_arr == src
    if m.sum() < 20 or len(set(y[m])) < 2:
        print(f"  {src:18} n={int(m.sum()):4d}  think {y[m].mean()*100:4.0f}%  (too few of one class for AUC)"); continue
    oof = cv_specialist(m); idx = np.where(m)[0]
    print(f"  {src:18} [{kind.get(src,'?'):9}] n={m.sum():4d}  think {y[idx].mean()*100:4.0f}%  ->  within-source text->label AUC {roc_auc_score(y[idx], oof[idx]):.3f}")
# does the coarse math-vs-chat topic at least predict the label across Mohseni?
mo = ds_arr == "mohseni"
is_math = np.array([kind.get(s) == "math" for s in src_arr])
print(f"\n  Across Mohseni: think-rate  math {y[mo & is_math].mean()*100:.0f}%  vs  chat {y[mo & ~is_math].mean()*100:.0f}%")
print(f"  'is it math?' as a 1-feature router on Mohseni -> AUC {roc_auc_score(y[mo], is_math[mo].astype(float)):.3f}")
