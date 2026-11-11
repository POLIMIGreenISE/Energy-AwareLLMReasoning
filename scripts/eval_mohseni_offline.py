"""Offline re-score of the Mohseni cross-dataset eval from the CACHED judge verdicts.

The full eval (scripts/eval_testset.py) re-attempts the 2 permanently-failed (safety-blocked)
grade-both pairs on every run, which blocks for minutes on retry back-off. All 1,806 real
verdicts are already in results/crossdataset/judge_cache_mohseni.jsonl, so this reconstructs
the labels from disk (no API calls) and recomputes everything, including the random-router
EAA floor. Pure analysis; safe to re-run.
"""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache

OUT = "results/crossdataset"
CACHE = JudgeCache(f"{OUT}/judge_cache_mohseni.jsonl")
rows = [json.loads(l) for l in open(f"{OUT}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows:
    byq[r["qid"]][r["mode"]] = r
qids = [q for q in byq if "think" in byq[q] and "no_think" in byq[q]]

lab = {}
for q in qids:
    nt, th = byq[q]["no_think"], byq[q]["think"]
    v = CACHE.get(nt["answer_text"], th["answer_text"])     # key(gold=nt_ans, answer=th_ans)
    if v == "__miss__":
        continue
    lab[q] = (bool(v[0]), bool(v[1]))
print(f"mohseni: {len(qids)} paired queries | reconstructed {len(lab)} labels from cache")

data = {}
for q in lab:
    cn, ct = lab[q]
    nt, th = byq[q]["no_think"], byq[q]["think"]
    data[q] = dict(cn=float(cn), ct=float(ct),
                   en=nt["energy_b1eq_wh"] * 1000, et=th["energy_b1eq_wh"] * 1000,
                   oracle="think" if (ct and not cn) else "no_think",
                   his=nt.get("his_label"), source=nt.get("source"))
oracle = {q: data[q]["oracle"] for q in data}
print("oracle label dist:", dict(Counter(oracle.values())))

scores = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{OUT}/routerscores_test_mohseni.jsonl", encoding="utf-8")}
pneed = {json.loads(l)["qid"]: json.loads(l)["p_need_think"] for l in open(f"{OUT}/pneed_test_mohseni.jsonl", encoding="utf-8")}
common = [q for q in data if q in scores and q in pneed]
N = len(common)
cn = np.array([data[q]["cn"] for q in common]); ct = np.array([data[q]["ct"] for q in common])
en = np.array([data[q]["en"] for q in common]); et = np.array([data[q]["et"] for q in common])
ROUTERS = {"LogReg+MiniLM": np.array([scores[q]["p_logreg"] for q in common]),
           "ModernBERT": np.array([scores[q]["p_modernbert"] for q in common]),
           "p_need": np.array([pneed[q] for q in common])}

def metrics(ch): ch = np.asarray(ch, bool); return np.where(ch, et, en).mean(), np.where(ch, ct, cn).mean()
def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda p: (p[0], -p[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out
e_lo, e_hi = metrics(np.zeros(N, bool))[0], metrics(np.ones(N, bool))[0]
never_acc = metrics(np.zeros(N, bool))[1]; always_acc = metrics(np.ones(N, bool))[1]
def eaa(pts):
    es = np.linspace(e_lo, e_hi, 200); acc = []
    for e in es:
        f = [a for pe, a in pts if pe <= e + 1e-9]; acc.append(max(f) if f else never_acc)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(np.array(acc), es) / (e_hi - e_lo)
dA, dE = ct - cn, et - en
pos = (dA > 0) & (dE > 0)
lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
of = pareto_ul([metrics((dA - l * dE) > 1e-12) for l in lams] + [metrics(np.zeros(N, bool)), metrics(np.ones(N, bool))])
oracle_eaa = eaa(of)

rng = np.random.default_rng(0)
rand = []
for _ in range(30):
    s = rng.random(N); ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
    rand.append(eaa([metrics(s >= t) for t in ts]))
random_eaa = float(np.mean(rand))
gainpct = lambda e: 100 * (e - never_acc) / (oracle_eaa - never_acc + 1e-12)
random_pct = gainpct(random_eaa)

from sklearn.metrics import roc_auc_score
y = np.array([oracle[q] == "think" for q in common]).astype(int)
print(f"\n=== mohseni: OTB-trained routers on the new distribution (N={N}) ===")
print(f"Never {never_acc:.3f}@{e_lo:.0f}mWh  Always {always_acc:.3f}@{e_hi:.0f}mWh  Oracle EAA {oracle_eaa:.4f}")
print(f"Random-router EAA {random_eaa:.4f} ({random_pct:.0f}% of oracle gain) <- floor any real router must beat")
rt = {}
for name, s in ROUTERS.items():
    ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
    e = eaa([metrics(s >= t) for t in ts]); auc = roc_auc_score(y, s)
    rt[name] = (auc, e, gainpct(e))
    print(f"  {name:14} label-AUC {auc:.3f} | EAA {e:.4f} ({gainpct(e):.0f}% of gain, {gainpct(e)-random_pct:+.0f}pp vs random)")

his = np.array([1 if data[q]["his"] == "think" else 0 for q in common])
agree = (y == his).mean()
print(f"\nour energy-aware oracle vs Mohseni's labels: agreement {agree:.3f} (our think% {y.mean()*100:.0f}, Mohseni {his.mean()*100:.0f})")
src_lines = []
for src in sorted(set(data[q]["source"] for q in common)):
    m = np.array([data[q]["source"] == src for q in common])
    if m.sum() > 5:
        line = f"  {src:20} agree {(y[m]==his[m]).mean():.3f} (n={int(m.sum())})"; print(line); src_lines.append(line.strip())

with open(f"{OUT}/eval_mohseni.md", "w", encoding="utf-8") as f:
    f.write(f"# Generalization eval: OTB-trained routers on mohseni (N={N})\n\n")
    f.write(f"Never {never_acc:.3f} @ {e_lo:.0f} mWh; Always {always_acc:.3f} @ {e_hi:.0f} mWh; "
            f"Oracle EAA {oracle_eaa:.4f}. Judge: reference-free grade-both (gemini-2.5-flash, certified).\n\n")
    f.write(f"**Random-router EAA {random_eaa:.4f} ({random_pct:.0f}% of oracle gain)** — the floor any "
            f"real router must beat (Always ≫ Never here, so random routing already captures gain).\n\n")
    f.write("|router (trained on OTB)|label-AUC vs our oracle|EAA on new data|% oracle gain|vs random|\n|--|--|--|--|--|\n")
    for name, (auc, e, pct) in rt.items():
        f.write(f"|{name}|{auc:.3f}|{e:.4f}|{pct:.0f}%|{pct-random_pct:+.0f}pp|\n")
    f.write(f"\nOur energy-aware oracle vs Mohseni's own labels: **agreement {agree:.3f}** "
            f"(our think {y.mean()*100:.0f}% vs Mohseni's 50%). By source:\n\n")
    for l in src_lines:
        f.write(f"- {l}\n")
print(f"\nwrote {OUT}/eval_mohseni.md")
