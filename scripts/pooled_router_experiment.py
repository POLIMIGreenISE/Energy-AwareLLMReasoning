"""Experiment A -- pooled, IN-DISTRIBUTION router on OTB + Mohseni + GSM8K.

All three datasets carry OUR energy-aware Qwen3-8B oracle labels (think iff thinking flips
this model wrong->right) and the identical energy basis (completion_tokens x 3.7 J/tok). We run
5-fold stratified CV and, on the SAME held-out folds, compare:
  - POOLED router   : LogReg+MiniLM trained on all-3 train folds
  - OTB-ONLY router : LogReg+MiniLM trained on ONLY the OTB rows of the train fold
  - p_need probe    : fixed, non-trainable
  - random floor / oracle ceiling
The pooled-vs-OTB-only gap on the Mohseni/GSM8K slices is the in-distribution read on
"does adding diverse data to training help?" (Experiment B will test TRANSFER to an unseen set).

EAA is reported on the NATURAL distribution (energy savings depend on the real base rate);
a balanced 50/50 subset is reported only for the decision metric (AUC/accuracy). Free, local.
Out: results/pooled/experimentA.md
"""
import json, os, re, sys, hashlib
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache

OUT = "results/pooled"; os.makedirs(OUT, exist_ok=True)
CD = "results/crossdataset"

# ---------------------------------------------------------------- load 3 datasets
def numeric_correct(ans, gold):
    g = str(gold).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(ans).replace(",", "")[-200:])
    try:
        return any(abs(float(x) - float(g)) <= 1e-6 * max(1, abs(float(g))) for x in nums)
    except ValueError:
        return None

def rec(qid, question, cn, ct, en_wh, et_wh, ds, source, p_need):
    return dict(qid=f"{ds}:{qid}", question=question, cn=float(cn), ct=float(ct),
                en=en_wh * 1000, et=et_wh * 1000, dataset=ds, source=source,
                oracle=("think" if (ct and not cn) else "no_think"), p_need=p_need)

def load_otb():
    lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
    pn = {json.loads(l)["qid"]: json.loads(l)["p_need_think"]
          for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
    out = []
    for e in lab:
        if e["qid"] not in pn:
            continue
        out.append(rec(e["qid"], e["question"], e["correct_no_think"], e["correct_think"],
                       e["energy_wh_no_think"], e["energy_wh_think"], "otb",
                       "otb_" + e["subset"][:4], pn[e["qid"]]))
    return out

def load_crossds(ds):
    rows = [json.loads(l) for l in open(f"{CD}/gen_test_{ds}.jsonl", encoding="utf-8") if l.strip()]
    byq = defaultdict(dict)
    for r in rows:
        byq[r["qid"]][r["mode"]] = r
    pn = {json.loads(l)["qid"]: json.loads(l)["p_need_think"]
          for l in open(f"{CD}/pneed_test_{ds}.jsonl", encoding="utf-8")}
    cache = JudgeCache(f"{CD}/judge_cache_{ds}.jsonl") if ds == "mohseni" else None
    out = []
    for q, mm in byq.items():
        if "think" not in mm or "no_think" not in mm or q not in pn:
            continue
        nt, th = mm["no_think"], mm["think"]
        if ds == "gsm8k":
            cn, ct = numeric_correct(nt["answer_text"], nt["gold"]), numeric_correct(th["answer_text"], th["gold"])
            if cn is None or ct is None:
                continue
        else:
            v = cache.get(nt["answer_text"], th["answer_text"])
            if v == "__miss__":
                continue
            cn, ct = v
        out.append(rec(q, nt["question"], cn, ct, nt["energy_b1eq_wh"], th["energy_b1eq_wh"],
                       ds, nt.get("source") or ds, pn[q]))
    return out

DATA = load_otb() + load_crossds("mohseni") + load_crossds("gsm8k")
print(f"pooled N={len(DATA)} | by dataset:", dict(Counter(d['dataset'] for d in DATA)))
print("oracle by dataset:", {ds: dict(Counter(d['oracle'] for d in DATA if d['dataset'] == ds))
                             for ds in ["otb", "mohseni", "gsm8k"]})

# ---------------------------------------------------------------- embeddings (cached)
qids = [d["qid"] for d in DATA]
sig = hashlib.md5("|".join(qids).encode()).hexdigest()[:10]
cache_npz = f"{OUT}/emb_MiniLM_{sig}.npz"
if os.path.exists(cache_npz):
    X = np.load(cache_npz)["X"]; print(f"loaded cached embeddings {X.shape}")
else:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X = enc.encode([d["question"] for d in DATA], batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True).astype(np.float32)
    np.savez_compressed(cache_npz, X=X); print(f"embedded + cached {X.shape}")

cn = np.array([d["cn"] for d in DATA]); ct = np.array([d["ct"] for d in DATA])
en = np.array([d["en"] for d in DATA]); et = np.array([d["et"] for d in DATA])
pneed = np.array([d["p_need"] for d in DATA])
ds_arr = np.array([d["dataset"] for d in DATA])
y = np.array([d["oracle"] == "think" for d in DATA]).astype(int)

# ---------------------------------------------------------------- metric helpers
def analyze(idx, scores: dict, seeds=30):
    """EAA/AUC on the rows in idx for each named score array; + oracle, random floor."""
    c0, c1, e0, e1 = cn[idx], ct[idx], en[idx], et[idx]
    n = len(idx)
    def metrics(ch):
        ch = np.asarray(ch, bool)
        return float(np.where(ch, e1, e0).mean()), float(np.where(ch, c1, c0).mean())
    e_lo, never = metrics(np.zeros(n, bool)); e_hi, always = metrics(np.ones(n, bool))
    def eaa(pts):
        best, out = -1, []
        for e, a in sorted(pts, key=lambda p: (p[0], -p[1])):
            if a > best + 1e-12: out.append((e, a)); best = a
        es = np.linspace(e_lo, e_hi, 200); acc = []
        for e in es:
            f = [a for pe, a in out if pe <= e + 1e-9]; acc.append(max(f) if f else never)
        tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(tz(np.array(acc), es) / (e_hi - e_lo + 1e-12))
    dA, dE = c1 - c0, e1 - e0; pos = (dA > 0) & (dE > 0)
    lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist())) if pos.any() else [0.0]
    oracle_eaa = eaa([metrics((dA - l * dE) > 1e-12) for l in lams] +
                     [metrics(np.zeros(n, bool)), metrics(np.ones(n, bool))])
    rng = np.random.default_rng(0)
    rnd = []
    for _ in range(seeds):
        s = rng.random(n); ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
        rnd.append(eaa([metrics(s >= t) for t in ts]))
    random_eaa = float(np.mean(rnd))
    gain = lambda e: 100 * (e - never) / (oracle_eaa - never + 1e-12)
    from sklearn.metrics import roc_auc_score
    yy = y[idx]; res = {}
    for name, s in scores.items():
        s = s[idx]; ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
        e = eaa([metrics(s >= t) for t in ts])
        auc = roc_auc_score(yy, s) if len(set(yy)) > 1 else float("nan")
        res[name] = dict(auc=auc, eaa=e, gain=gain(e), vs_rand=gain(e) - gain(random_eaa))
    return dict(n=n, never=never, always=always, e_lo=e_lo, e_hi=e_hi,
                oracle_eaa=oracle_eaa, random_eaa=random_eaa, random_gain=gain(random_eaa),
                think_frac=float(yy.mean()), routers=res)

# ---------------------------------------------------------------- 5-fold CV: pooled vs OTB-only
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
strat = np.array([f"{d}_{o}" for d, o in zip(ds_arr, y)])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pool = np.zeros(len(DATA)); oof_otb = np.zeros(len(DATA))
for tr, te in skf.split(X, strat):
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    oof_pool[te] = lr.predict_proba(X[te])[:, 1]
    otb_tr = tr[ds_arr[tr] == "otb"]
    lo = LogisticRegression(max_iter=2000, C=1.0).fit(X[otb_tr], y[otb_tr])
    oof_otb[te] = lo.predict_proba(X[te])[:, 1]
SCORES = {"pooled-router": oof_pool, "OTB-only-router": oof_otb, "p_need": pneed}

# ---------------------------------------------------------------- report
def fmt(r):
    L = [f"N={r['n']} think={r['think_frac']*100:.0f}% | Never {r['never']:.3f}@{r['e_lo']:.0f} "
         f"Always {r['always']:.3f}@{r['e_hi']:.0f} | Oracle EAA {r['oracle_eaa']:.4f} | "
         f"Random {r['random_eaa']:.4f} ({r['random_gain']:.0f}%)"]
    for name, m in r["routers"].items():
        L.append(f"  {name:16} AUC {m['auc']:.3f} | EAA {m['eaa']:.4f} "
                 f"({m['gain']:.0f}% of gain, {m['vs_rand']:+.0f}pp vs random)")
    return "\n".join(L)

lines = ["# Experiment A -- pooled in-distribution router (OTB + Mohseni + GSM8K)\n",
         "5-fold stratified CV, OOF predictions. EAA on the natural distribution; "
         "energy basis identical across datasets (tok x 3.7 J). p_need is fixed (non-trainable).\n",
         "## Overall (pooled)", "```", fmt(analyze(np.arange(len(DATA)), SCORES)), "```\n",
         "## Per-dataset slice of the SAME pooled-CV predictions (in-distribution)"]
print("\n=== OVERALL (pooled) ==="); print(fmt(analyze(np.arange(len(DATA)), SCORES)))
per = {}
for ds in ["otb", "mohseni", "gsm8k"]:
    idx = np.where(ds_arr == ds)[0]
    r = analyze(idx, SCORES); per[ds] = r
    print(f"\n=== {ds} (in-dist) ==="); print(fmt(r))
    lines += [f"\n### {ds}", "```", fmt(r), "```"]

# balanced 50/50 decision metric (user's ask): AUC + accuracy@0.5 on a think/no_think-balanced pool
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
th_idx = np.where(y == 1)[0]; no_idx = np.where(y == 0)[0]
rng = np.random.default_rng(7); accs = defaultdict(list); aucs = defaultdict(list)
for _ in range(20):
    samp = np.concatenate([th_idx, rng.choice(no_idx, len(th_idx), replace=False)])
    for name, s in SCORES.items():
        aucs[name].append(roc_auc_score(y[samp], s[samp]))
        accs[name].append(balanced_accuracy_score(y[samp], (s[samp] >= 0.5).astype(int)))
lines += ["\n## Balanced 50/50 decision metric (think=no_think count; 20 subsamples)",
          "```", f"balanced N={2*len(th_idx)} (think {len(th_idx)} + equal no_think)"]
print(f"\n=== BALANCED 50/50 (N={2*len(th_idx)}) ===")
for name in SCORES:
    ln = f"  {name:16} AUC {np.mean(aucs[name]):.3f}  bal-acc@0.5 {np.mean(accs[name]):.3f}"
    print(ln); lines.append(ln)
lines.append("```")

# headline deltas
def delta(slice_r):
    p = slice_r["routers"]["pooled-router"]; o = slice_r["routers"]["OTB-only-router"]
    return p["eaa"] - o["eaa"], p["vs_rand"] - o["vs_rand"], p["auc"] - o["auc"]
lines += ["\n## Does adding diverse data to training help (in-distribution)?",
          "pooled-router minus OTB-only-router, same held-out folds:\n",
          "|slice|ΔEAA|Δ(pp vs random)|ΔAUC|\n|--|--|--|--|"]
print("\n=== POOLED minus OTB-ONLY (in-dist gain from adding data) ===")
for ds in ["otb", "mohseni", "gsm8k"]:
    de, dv, da = delta(per[ds])
    lines.append(f"|{ds}|{de:+.4f}|{dv:+.0f}pp|{da:+.3f}|")
    print(f"  {ds:8} dEAA {de:+.4f}  d(vs-random) {dv:+.0f}pp  dAUC {da:+.3f}")
de, dv, da = delta(analyze(np.arange(len(DATA)), SCORES))
lines.append(f"|**overall**|{de:+.4f}|{dv:+.0f}pp|{da:+.3f}|")
print(f"  {'overall':8} dEAA {de:+.4f}  d(vs-random) {dv:+.0f}pp  dAUC {da:+.3f}")

open(f"{OUT}/experimentA.md", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nwrote {OUT}/experimentA.md")
