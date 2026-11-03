"""Run Mohseni's OFFICIAL released routers on his own test split, scored with OUR metrics.

Checkpoints (both binary seq-classifiers, id2label {0:no_think, 1:think}):
  - AmirMohseni/reasoning-router-mmbert-small   (ModernBERT, 22L/384h)
  - AmirMohseni/reasoning-router-0.6b           (Qwen3-0.6B)

Evaluated three ways on the SAME 1,810 queries we generated Qwen3-8B answers for:
  (a) AUC vs HIS OWN labels   -> is his label learnable at all (he trained on 70k)?
  (b) AUC vs OUR energy-aware oracle -> does it predict whether thinking actually helps Qwen3-8B?
  (c) EAA + vs the random-routing floor -> does it deliver real energy-accuracy routing?
(a) is his home turf; (b)/(c) are the deployment question. Runs locally on GPU.
Out: results/crossdataset/official_router_eval.md
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CD = "results/crossdataset"
rows = [json.loads(l) for l in open(f"{CD}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows: byq[r["qid"]][r["mode"]] = r
cache = JudgeCache(f"{CD}/judge_cache_mohseni.jsonl")
D = []
for q, mm in byq.items():
    if "think" not in mm or "no_think" not in mm: continue
    nt, th = mm["no_think"], mm["think"]
    v = cache.get(nt["answer_text"], th["answer_text"])
    if v == "__miss__": continue
    cn, ct = int(v[0]), int(v[1])
    D.append(dict(qid=q, q=nt["question"], src=nt.get("source") or "?",
                  cn=float(cn), ct=float(ct),
                  en=nt["energy_b1eq_wh"] * 1000, et=th["energy_b1eq_wh"] * 1000,
                  y=1 if (ct and not cn) else 0,
                  his=1 if nt.get("his_label") == "think" else 0))
N = len(D)
y = np.array([d["y"] for d in D]); his = np.array([d["his"] for d in D])
cn = np.array([d["cn"] for d in D]); ct = np.array([d["ct"] for d in D])
en = np.array([d["en"] for d in D]); et = np.array([d["et"] for d in D])
src = np.array([d["src"] for d in D])
print(f"N={N} | our think {y.mean()*100:.0f}% | his labels think {his.mean()*100:.0f}%")

dev = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def score(repo, maxlen=1024, bs=16, dtype=torch.float32):
    # ModernBERT overflows to NaN in fp16 -> keep it fp32 (small, fast anyway).
    # Qwen3-0.6B in fp32 @ bs16 pins an 8GB laptop GPU -> bf16 + small batch (bf16 has fp32 range).
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo, dtype=dtype).to(dev).eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if getattr(mdl.config, "pad_token_id", None) is None: mdl.config.pad_token_id = tok.pad_token_id
    lab = {int(k): v for k, v in mdl.config.id2label.items()}
    think_idx = [i for i, v in lab.items() if str(v).lower() == "think"][0]
    out = []
    qs = [d["q"] for d in D]
    import time; t0 = time.time()
    for i in range(0, len(qs), bs):
        enc = tok(qs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
        p = mdl(**enc).logits.float().softmax(-1)[:, think_idx]
        out += p.cpu().tolist()
        if (i // bs) % 25 == 0:
            el = time.time() - t0
            print(f"    {i}/{len(qs)}  {el:.0f}s  eta {el/max(1,i+bs)*(len(qs)-i-bs):.0f}s", flush=True)
    del mdl; torch.cuda.empty_cache()
    return np.array(out)

def metrics(ch): ch = np.asarray(ch, bool); return float(np.where(ch, et, en).mean()), float(np.where(ch, ct, cn).mean())
e_lo, never = metrics(np.zeros(N, bool)); e_hi, always = metrics(np.ones(N, bool))
def eaa(pts):
    best, o = -1, []
    for e, a in sorted(pts):
        if a > best + 1e-12: o.append((e, a)); best = a
    es = np.linspace(e_lo, e_hi, 200)
    acc = [max([a for pe, a in o if pe <= e + 1e-9] or [never]) for e in es]
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(tz(np.array(acc), es) / (e_hi - e_lo + 1e-12))
dA, dE = ct - cn, et - en; pos = (dA > 0) & (dE > 0)
lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
oracle_eaa = eaa([metrics((dA - l * dE) > 1e-12) for l in lams] + [(e_lo, never), (e_hi, always)])
rng = np.random.default_rng(0)
rand_eaa = float(np.mean([eaa([metrics(rng.random(N) >= t) for t in np.linspace(0, 1, 60)]) for _ in range(20)]))
gain = lambda v: 100 * (v - never) / (oracle_eaa - never + 1e-12)
print(f"Never {never:.3f}@{e_lo:.0f}mWh  Always {always:.3f}@{e_hi:.0f}mWh  OracleEAA {oracle_eaa:.4f}  "
      f"RandomEAA {rand_eaa:.4f} ({gain(rand_eaa):.0f}% of gain)")

def sweep_eaa(s):
    ts = np.unique(np.concatenate([np.linspace(0, 1, 101), np.unique(s)]))
    return eaa([metrics(s >= t) for t in ts])

RES = {}
CFG = {"AmirMohseni/reasoning-router-mmbert-small": dict(bs=16, dtype=torch.float32),
       "AmirMohseni/reasoning-router-0.6b": dict(bs=4, dtype=torch.bfloat16)}
for repo in ["AmirMohseni/reasoning-router-mmbert-small", "AmirMohseni/reasoning-router-0.6b"]:
    name = repo.split("/")[-1]
    print(f"\nscoring {name} on {dev} ({CFG[repo]}) ...", flush=True)
    try:
        s = score(repo, **CFG[repo])
    except Exception as e:
        print(f"  FAILED: {repr(e)[:160]}"); continue
    e_ = sweep_eaa(s)
    RES[name] = dict(auc_his=roc_auc_score(his, s), auc_our=roc_auc_score(y, s), eaa=e_,
                     gain=gain(e_), vs_rand=gain(e_) - gain(rand_eaa),
                     think_rate=float((s >= 0.5).mean()), mean_p=float(s.mean()), s=s)
    r = RES[name]
    print(f"  AUC vs HIS labels {r['auc_his']:.3f} | AUC vs OUR oracle {r['auc_our']:.3f} | "
          f"EAA {r['eaa']:.4f} ({r['vs_rand']:+.0f}pp vs random) | think-rate@0.5 {r['think_rate']*100:.0f}%")
    for sname in ["Big-Math-RL", "Nectar", "WildChat-filtered"]:
        m = src == sname
        if m.sum() > 30 and len(set(y[m])) > 1:
            print(f"     {sname:20} AUC vs our oracle {roc_auc_score(y[m], s[m]):.3f} | vs his {roc_auc_score(his[m], s[m]):.3f}")

lines = [f"# Mohseni's OFFICIAL routers on his own test split (N={N})\n",
         f"Never {never:.3f} @ {e_lo:.0f} mWh; Always {always:.3f} @ {e_hi:.0f} mWh; Oracle EAA {oracle_eaa:.4f}; "
         f"**Random floor EAA {rand_eaa:.4f} ({gain(rand_eaa):.0f}% of oracle gain)**.\n",
         "|router|AUC vs HIS labels|AUC vs OUR oracle|EAA|vs random|think-rate@0.5|", "|--|--|--|--|--|--|"]
for n_, r in RES.items():
    lines.append(f"|{n_}|{r['auc_his']:.3f}|{r['auc_our']:.3f}|{r['eaa']:.4f}|{r['vs_rand']:+.0f}pp|{r['think_rate']*100:.0f}%|")
lines += ["|*our LogReg (OTB-trained)*|—|0.505|0.7304|+6pp|—|",
          "|*our ModernBERT (OTB-trained)*|—|0.521|0.7301|+6pp|—|",
          "|*p_need probe*|—|0.564|0.6572|−24pp|—|"]
os.makedirs(CD, exist_ok=True)
open(f"{CD}/official_router_eval.md", "w", encoding="utf-8").write("\n".join(lines))
np.savez(f"{CD}/official_router_scores.npz", **{n_: r["s"] for n_, r in RES.items()},
         qids=np.array([d["qid"] for d in D]))
print(f"\nwrote {CD}/official_router_eval.md")
