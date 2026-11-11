"""Labeling both-fail queries as `think` (Mohseni's rule): does it help?

Two label definitions differing ONLY on both-wrong queries:
  A  energy-aware (ours) : think iff (think correct AND no-think wrong)   -> both-wrong = no_think
  B  difficulty (Mohseni): think iff (no-think wrong)                     -> both-wrong = think

Hypothesis: B is easier for a text router to learn (it is "is this hard?", which correlates
with topic/surface form). Counter-hypothesis (energy): B routes hopeless queries to thinking, burning
energy for zero accuracy gain.

So we measure BOTH: (1) learnability = CV AUC predicting each label; (2) DEPLOYMENT VALUE = train a
router on A vs on B, then score both on the SAME energy-accuracy metric (EAA vs random floor), which
is label-independent (it uses the real cn/ct/en/et). Bootstrap CIs on the difference.
Free/local. Out: results/pooled/bothfail_label.md
"""
import json, os, re, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

CD = "results/crossdataset"
def numeric_correct(ans, gold):
    g = str(gold).replace(",", ""); nums = re.findall(r"-?\d+(?:\.\d+)?", str(ans).replace(",", "")[-200:])
    try: return any(abs(float(x)-float(g)) <= 1e-6*max(1, abs(float(g))) for x in nums)
    except ValueError: return None

def load_all():
    D = []
    lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
    for e in lab:
        D.append(dict(q=e["question"], ds="otb", cn=float(e["correct_no_think"]), ct=float(e["correct_think"]),
                      en=e["energy_wh_no_think"]*1000, et=e["energy_wh_think"]*1000))
    for ds in ["mohseni", "gsm8k"]:
        rows = [json.loads(l) for l in open(f"{CD}/gen_test_{ds}.jsonl", encoding="utf-8") if l.strip()]
        byq = defaultdict(dict)
        for r in rows: byq[r["qid"]][r["mode"]] = r
        cache = JudgeCache(f"{CD}/judge_cache_{ds}.jsonl") if ds == "mohseni" else None
        for q, mm in byq.items():
            if "think" not in mm or "no_think" not in mm: continue
            nt, th = mm["no_think"], mm["think"]
            if ds == "gsm8k":
                cn, ct = numeric_correct(nt["answer_text"], nt["gold"]), numeric_correct(th["answer_text"], th["gold"])
                if cn is None or ct is None: continue
            else:
                v = cache.get(nt["answer_text"], th["answer_text"])
                if v == "__miss__": continue
                cn, ct = v
            D.append(dict(q=nt["question"], ds=ds, cn=float(cn), ct=float(ct),
                          en=nt["energy_b1eq_wh"]*1000, et=th["energy_b1eq_wh"]*1000))
    return D

D = load_all()
ds = np.array([d["ds"] for d in D])
cn = np.array([d["cn"] for d in D]); ct = np.array([d["ct"] for d in D])
en = np.array([d["en"] for d in D]); et = np.array([d["et"] for d in D])
yA = ((ct > 0.5) & (cn <= 0.5)).astype(int)          # energy-aware (ours)
yB = (cn <= 0.5).astype(int)                          # difficulty  (both-wrong -> think)
bothwrong = ((cn <= 0.5) & (ct <= 0.5))
print(f"N={len(D)} | both-wrong {bothwrong.mean()*100:.1f}%")
X = SentenceTransformer("all-MiniLM-L6-v2").encode([d["q"] for d in D], batch_size=64,
                                                   normalize_embeddings=True).astype(np.float32)

def cv_scores(y, idx):
    oof = np.zeros(len(idx))
    yy = y[idx]
    if len(set(yy)) < 2: return oof
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X[idx], yy):
        oof[te] = LogisticRegression(max_iter=2000).fit(X[idx][tr], yy[tr]).predict_proba(X[idx][te])[:, 1]
    return oof

def eaa_pack(idx):
    c0, c1, e0, e1 = cn[idx], ct[idx], en[idx], et[idx]; n = len(idx)
    def m(ch): ch = np.asarray(ch, bool); return float(np.where(ch, e1, e0).mean()), float(np.where(ch, c1, c0).mean())
    elo, nev = m(np.zeros(n, bool)); ehi, alw = m(np.ones(n, bool))
    def E(pts):
        best, o = -1, []
        for e, a in sorted(pts):
            if a > best+1e-12: o.append((e, a)); best = a
        es = np.linspace(elo, ehi, 160)
        acc = [max([a for pe, a in o if pe <= e+1e-9] or [nev]) for e in es]
        tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(tz(np.array(acc), es)/(ehi-elo+1e-12))
    dA, dE = c1-c0, e1-e0; pos = (dA > 0) & (dE > 0)
    lams = sorted(set([0.0]+(dA[pos]/dE[pos]).tolist())) if pos.any() else [0.0]
    orc = E([m((dA-l*dE) > 1e-12) for l in lams]+[(elo, nev), (ehi, alw)])
    sweep = lambda s: E([m(s >= t) for t in np.linspace(0, 1, 90)])
    return sweep, orc, nev, m

print(f"\n{'slice':10} {'both-wrong':>11} | {'AUC(A energy)':>14} {'AUC(B difficulty)':>18} | "
      f"{'EAA router-A':>13} {'EAA router-B':>13} {'random':>8}")
rows_out = []
rng = np.random.default_rng(0)
for name in ["otb", "mohseni", "gsm8k", "POOLED"]:
    idx = np.arange(len(D)) if name == "POOLED" else np.where(ds == name)[0]
    sA, sB = cv_scores(yA, idx), cv_scores(yB, idx)
    aucA = roc_auc_score(yA[idx], sA) if len(set(yA[idx])) > 1 else float("nan")
    aucB = roc_auc_score(yB[idx], sB) if len(set(yB[idx])) > 1 else float("nan")
    sweep, orc, nev, m = eaa_pack(idx)
    eA, eB = sweep(sA), sweep(sB)
    er = float(np.mean([sweep(rng.random(len(idx))) for _ in range(15)]))
    print(f"  {name:8} {bothwrong[idx].mean()*100:>10.1f}% | {aucA:>14.3f} {aucB:>18.3f} | "
          f"{eA:>13.4f} {eB:>13.4f} {er:>8.4f}")
    rows_out.append((name, bothwrong[idx].mean(), aucA, aucB, eA, eB, er, orc, nev))

# bootstrap the router-A vs router-B EAA difference on the pooled set and on mohseni
print("\n=== Does the more-learnable label produce a BETTER ROUTER? (bootstrap B=200) ===")
for name in ["mohseni", "POOLED"]:
    idx = np.arange(len(D)) if name == "POOLED" else np.where(ds == name)[0]
    sA, sB = cv_scores(yA, idx), cv_scores(yB, idx)
    diffs = []
    for _ in range(200):
        bs = rng.integers(0, len(idx), len(idx))
        sub = idx[bs]
        c0, c1, e0, e1 = cn[sub], ct[sub], en[sub], et[sub]
        def m2(ch): ch = np.asarray(ch, bool); return float(np.where(ch, e1, e0).mean()), float(np.where(ch, c1, c0).mean())
        elo, nev2 = m2(np.zeros(len(sub), bool)); ehi, _ = m2(np.ones(len(sub), bool))
        def E2(pts):
            best, o = -1, []
            for e, a in sorted(pts):
                if a > best+1e-12: o.append((e, a)); best = a
            es = np.linspace(elo, ehi, 100)
            acc = [max([a for pe, a in o if pe <= e+1e-9] or [nev2]) for e in es]
            tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            return float(tz(np.array(acc), es)/(ehi-elo+1e-12))
        sw = lambda s: E2([m2(s >= t) for t in np.linspace(0, 1, 70)])
        diffs.append(sw(sB[bs]) - sw(sA[bs]))
    diffs = np.array(diffs)
    print(f"  {name:8} EAA(router-B) - EAA(router-A) = {diffs.mean():+.4f} "
          f"[{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]"
          f"  {'B BETTER' if np.percentile(diffs,2.5)>0 else ('A BETTER' if np.percentile(diffs,97.5)<0 else 'NO SIG DIFF')}")

# the energy price of the both-wrong rule
print("\n=== Energy price of routing both-wrong queries to THINK (they fail either way) ===")
for name in ["otb", "mohseni", "gsm8k", "POOLED"]:
    idx = np.arange(len(D)) if name == "POOLED" else np.where(ds == name)[0]
    bw = idx[bothwrong[idx]]
    if len(bw) == 0: continue
    waste = (et[bw] - en[bw]).sum() / len(idx)
    print(f"  {name:8} both-wrong {len(bw):4d}/{len(idx):4d} | extra energy if all routed to think: "
          f"{waste:6.1f} mWh/query avg ({100*waste/en[idx].mean():.0f}% of the never-think budget), 0 accuracy gain")

with open("results/pooled/bothfail_label.md", "w", encoding="utf-8") as f:
    f.write("# Labeling both-fail queries as `think`?\n\n")
    f.write("A = energy-aware (ours, both-wrong -> no_think); B = difficulty (both-wrong -> think).\n\n")
    f.write("|slice|both-wrong|AUC label A|AUC label B|EAA router-A|EAA router-B|random|\n|--|--|--|--|--|--|--|\n")
    for n_, bw, aA, aB, eA, eB, er, orc, nev in rows_out:
        f.write(f"|{n_}|{bw*100:.1f}%|{aA:.3f}|{aB:.3f}|{eA:.4f}|{eB:.4f}|{er:.4f}|\n")
print("\nwrote results/pooled/bothfail_label.md")
