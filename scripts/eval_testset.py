"""Evaluate the OTB-trained routers on a held-out test dataset (after notebook 08).

Judges Qwen3-8B's think/no_think answers, builds energy-aware oracle labels, then
scores our OTB-trained routers on the NEW distribution:
  - GSM8K  : answers judged against gold (numeric/equivalence, certified Gemini pipeline)
  - Mohseni: NO gold -> reference-free grading (a strong judge grades both answers,
             generating its own ground truth via src.judge.judge_grade_both)

Reports, per router (LogReg, ModernBERT, p_need): the energy-accuracy Pareto / EAA on
the new data, plus (Mohseni only) agreement of our energy-aware oracle with Mohseni's
own labels -- a direct read on whether the two think/no_think definitions even align.

Usage: GEMINI_API_KEY=... GEMINI_MODEL=gemini-2.5-pro python scripts/eval_testset.py --dataset mohseni
Out:   results/crossdataset/eval_{dataset}.md
"""
import json, os, re, sys, argparse, statistics as st
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("GEMINI_THINKING_BUDGET", "0")
os.environ.setdefault("GEMINI_MAX_TOKENS", "3072")
os.environ.setdefault("JUDGE_MIN_INTERVAL", "0.5")
from src import scoring
from src.judge import judge_answer, judge_grade_both, JudgeCache

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["mohseni", "gsm8k"])
args = ap.parse_args()
DS = args.dataset
OUT = "results/crossdataset"
GEN = f"{OUT}/gen_test_{DS}.jsonl"
CACHE = JudgeCache(f"{OUT}/judge_cache_{DS}.jsonl")

rows = [json.loads(l) for l in open(GEN, encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows:
    byq[r["qid"]][r["mode"]] = r      # REPS=1 -> one row per (qid, mode)
qids = [q for q in byq if "think" in byq[q] and "no_think" in byq[q]]
print(f"{DS}: {len(qids)} queries with both modes")

def numeric_correct(ans, gold):
    g = str(gold).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(ans).replace(",", "")[-200:])
    try:
        return any(abs(float(x) - float(g)) <= 1e-6 * max(1, abs(float(g))) for x in nums)
    except ValueError:
        return None

lab = {}                # qid -> (nt_correct, th_correct)
n_judge = n_cache = n_fail = 0
for q in qids:
    nt, th = byq[q]["no_think"], byq[q]["think"]
    if DS == "gsm8k":
        cn = numeric_correct(nt["answer_text"], nt["gold"])
        ct = numeric_correct(th["answer_text"], th["gold"])
        if cn is None or ct is None:   # fall back to the equivalence judge
            cn = judge_answer(nt["question"], nt["gold"], nt["answer_text"], provider="gemini") if cn is None else cn
            ct = judge_answer(th["question"], th["gold"], th["answer_text"], provider="gemini") if ct is None else ct
            n_judge += 1
    else:  # mohseni: reference-free grade-both, dedup by (nt_ans, th_ans)
        key = CACHE.key(nt["answer_text"], th["answer_text"])
        hit = CACHE.cache.get(key, "__miss__")
        if hit != "__miss__":
            cn, ct = hit; n_cache += 1
        else:
            v = judge_grade_both(nt["question"], nt["answer_text"], th["answer_text"], provider="gemini")
            n_judge += 1
            if v is None: n_fail += 1; continue
            cn, ct = v
            CACHE.cache[key] = list(v)
            with open(CACHE.path, "a") as f:
                f.write(json.dumps({"k": key, "v": list(v)}) + "\n")
    if cn is None or ct is None: n_fail += 1; continue
    lab[q] = (bool(cn), bool(ct))
    if n_judge and n_judge % 100 == 0:
        print(f"  judged {n_judge} (cache {n_cache}, fail {n_fail})")
print(f"labeled {len(lab)} | judge {n_judge}, cache {n_cache}, fail {n_fail}")

# energy-aware oracle (same rule as OTB) + energies
J_B1 = 3.7
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

# router scores
scores = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{OUT}/routerscores_test_{DS}.jsonl", encoding="utf-8")}
pneed = {json.loads(l)["qid"]: json.loads(l)["p_need_think"] for l in open(f"{OUT}/pneed_test_{DS}.jsonl", encoding="utf-8")}
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
never_acc = metrics(np.zeros(N, bool))[1]
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

# random-router EAA baseline: uniform-random scores, threshold-swept, averaged over seeds.
# When Always >> Never (big +Acc for thinking), even random routing captures a chunk of the
# oracle gain just by partially sweeping toward always-think -- so a router's EAA is only
# meaningful RELATIVE to this. gain-over-random = router%_of_gain - random%_of_gain.
rng = np.random.default_rng(0)
rand_eaas = []
for _ in range(30):
    s = rng.random(N)
    ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
    rand_eaas.append(eaa([metrics(s >= t) for t in ts]))
random_eaa = float(np.mean(rand_eaas))
random_pct = 100 * (random_eaa - never_acc) / (oracle_eaa - never_acc + 1e-12)

from sklearn.metrics import roc_auc_score
y = np.array([oracle[q] == "think" for q in common]).astype(int)
print(f"\n=== {DS}: OTB-trained routers on the new distribution (N={N}) ===")
print(f"Never {never_acc:.3f}@{e_lo:.0f}mWh  Always {metrics(np.ones(N,bool))[1]:.3f}@{e_hi:.0f}mWh  Oracle EAA {oracle_eaa:.4f}")
print(f"Random-router EAA {random_eaa:.4f} ({random_pct:.0f}% of oracle gain) <- floor any real router must beat")
taus = np.linspace(0, 1, 101)
rt = {}
for name, s in ROUTERS.items():
    ts = np.unique(np.concatenate([taus, np.unique(s)]))
    e = eaa([metrics(s >= t) for t in ts])
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    pct = 100 * (e - never_acc) / (oracle_eaa - never_acc + 1e-12)
    rt[name] = (auc, e, pct)
    print(f"  {name:14} label-AUC(vs our oracle) {auc:.3f} | EAA {e:.4f} "
          f"({pct:.0f}% of oracle gain, {pct-random_pct:+.0f}pp vs random)")

# Mohseni only: does our energy-aware oracle agree with Mohseni's own labels?
extra = ""
if DS == "mohseni":
    his = np.array([1 if data[q]["his"] == "think" else 0 for q in common])
    agree = (y == his).mean()
    from sklearn.metrics import roc_auc_score as auc2
    print(f"\nour energy-aware oracle vs Mohseni's labels: agreement {agree:.3f} "
          f"(our think% {y.mean()*100:.0f}, his {his.mean()*100:.0f})")
    extra = f"\nOur energy-aware oracle vs Mohseni labels: agreement {agree:.3f}.\n"
    # by source
    for src in sorted(set(data[q]["source"] for q in common)):
        m = np.array([data[q]["source"] == src for q in common])
        if m.sum() > 5:
            print(f"  {src:20} agree {(y[m]==his[m]).mean():.3f} (n={m.sum()})")

with open(f"{OUT}/eval_{DS}.md", "w", encoding="utf-8") as f:
    f.write(f"# Generalization eval: OTB-trained routers on {DS} (N={N})\n\n")
    f.write(f"Never {never_acc:.3f} @ {e_lo:.0f} mWh; Always {metrics(np.ones(N,bool))[1]:.3f} @ {e_hi:.0f} mWh; "
            f"Oracle EAA {oracle_eaa:.4f}. Judge: {'gold' if DS=='gsm8k' else 'reference-free grade-both'}.\n\n")
    f.write(f"**Random-router EAA {random_eaa:.4f} ({random_pct:.0f}% of oracle gain)** — the floor any "
            f"real router must beat (when Always ≫ Never, random routing already captures gain).\n\n")
    f.write("|router (trained on OTB)|label-AUC vs our oracle|EAA on new data|% oracle gain|vs random|\n|--|--|--|--|--|\n")
    for name, (auc, e, pct) in rt.items():
        f.write(f"|{name}|{auc:.3f}|{e:.4f}|{pct:.0f}%|{pct-random_pct:+.0f}pp|\n")
    f.write(extra)
print(f"\nwrote {OUT}/eval_{DS}.md")
