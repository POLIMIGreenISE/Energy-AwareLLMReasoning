"""Two overview figures: held-out router transfer, and the pooled energy-accuracy Pareto front.

Fig 1: held-out transfer -- each router's EAA advantage over the random-routing floor on
       GSM8K and Mohseni (trained routers positive, p_need negative).
Fig 2: pooled energy-accuracy Pareto (Experiment A, N=4839) -- pooled router vs OTB-only vs
       p_need vs oracle frontier vs random floor, on the natural distribution.
Palette: warm paper, green accent, amber, brick-red.
Out: results/pooled/fig_transfer.png, results/pooled/fig_pooled_pareto.png
"""
import json, os, re, sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache

INK="#1F2823"; MUT="#5C6660"; ACC="#1D6B4F"; ACC2="#3E9B76"; AMBER="#9A6A15"; RED="#8C2F1B"; RULE="#D8DCD2"
plt.rcParams.update({"font.family":"Segoe UI","font.size":11,"text.color":INK,
                     "axes.edgecolor":RULE,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
                     "axes.linewidth":0.9,"figure.dpi":150})

# ---------- Fig 1: transfer vs random floor (numbers from results/crossdataset/eval_*.md) ----------
groups=["GSM8K  (held-out)","Mohseni  (held-out)"]
series={"LogReg+MiniLM":[+6,+6], "ModernBERT":[+5,+6], "p_need probe":[-11,-24]}
colors={"LogReg+MiniLM":ACC2, "ModernBERT":ACC, "p_need probe":RED}
fig,ax=plt.subplots(figsize=(7.6,4.0)); x=np.arange(len(groups)); w=0.26
for i,(name,vals) in enumerate(series.items()):
    bars=ax.bar(x+(i-1)*w, vals, w, label=name, color=colors[name],
                edgecolor="white", linewidth=0.6, zorder=3)
    for b,v in zip(bars,vals):
        ax.annotate(f"{v:+d}", (b.get_x()+b.get_width()/2, v), ha="center",
                    va="bottom" if v>=0 else "top", fontsize=9.5, color=INK, weight="bold",
                    xytext=(0,3 if v>=0 else -3), textcoords="offset points")
ax.axhline(0, color=INK, lw=1.3, zorder=2)
ax.text(1.46, 0.6, "random-routing floor", fontsize=8.5, color=MUT, style="italic", ha="right")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
ax.set_ylabel("EAA advantage over random routing  (pp of oracle gain)", fontsize=10)
ax.set_ylim(-30,14); ax.set_yticks(range(-30,11,10))
ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=1)
ax.set_title("On unseen data, trained routers beat random routing; the free probe is worse than random",
             fontsize=10.8, color=INK, weight="bold", pad=10)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.6, zorder=0)
fig.tight_layout(); fig.savefig("results/pooled/fig_transfer.png", bbox_inches="tight", facecolor="white")
print("wrote results/pooled/fig_transfer.png")

# ---------- load pooled data (mirror of pooled_router_experiment) for the Pareto ----------
CD="results/crossdataset"
def numeric_correct(ans,gold):
    g=str(gold).replace(",","");nums=re.findall(r"-?\d+(?:\.\d+)?",str(ans).replace(",","")[-200:])
    try: return any(abs(float(x)-float(g))<=1e-6*max(1,abs(float(g))) for x in nums)
    except ValueError: return None
def rc(q,cn,ct,en,et,ds,pn): return dict(cn=float(cn),ct=float(ct),en=en*1000,et=et*1000,ds=ds,pn=pn,
                                          oracle=("think" if (ct and not cn) else "no_think"))
def load_otb():
    lab=json.load(open("results/paired/paired_oracle_final.json",encoding="utf-8"))
    pn={json.loads(l)["qid"]:json.loads(l)["p_need_think"] for l in open("results/paired/probes_v1.jsonl",encoding="utf-8")}
    return [(e["question"],rc(e["qid"],e["correct_no_think"],e["correct_think"],e["energy_wh_no_think"],e["energy_wh_think"],"otb",pn[e["qid"]])) for e in lab if e["qid"] in pn]
def load_cd(ds):
    rows=[json.loads(l) for l in open(f"{CD}/gen_test_{ds}.jsonl",encoding="utf-8") if l.strip()]
    byq=defaultdict(dict)
    for r in rows: byq[r["qid"]][r["mode"]]=r
    pn={json.loads(l)["qid"]:json.loads(l)["p_need_think"] for l in open(f"{CD}/pneed_test_{ds}.jsonl",encoding="utf-8")}
    cache=JudgeCache(f"{CD}/judge_cache_{ds}.jsonl") if ds=="mohseni" else None
    out=[]
    for q,mm in byq.items():
        if "think" not in mm or "no_think" not in mm or q not in pn: continue
        nt,th=mm["no_think"],mm["think"]
        if ds=="gsm8k":
            cn,ct=numeric_correct(nt["answer_text"],nt["gold"]),numeric_correct(th["answer_text"],th["gold"])
            if cn is None or ct is None: continue
        else:
            v=cache.get(nt["answer_text"],th["answer_text"])
            if v=="__miss__": continue
            cn,ct=v
        out.append((nt["question"],rc(q,cn,ct,nt["energy_b1eq_wh"],th["energy_b1eq_wh"],ds,pn[q])))
    return out
DATA=load_otb()+load_cd("mohseni")+load_cd("gsm8k")
qs=[d[0] for d in DATA]; D=[d[1] for d in DATA]
cn=np.array([d["cn"] for d in D]);ct=np.array([d["ct"] for d in D])
en=np.array([d["en"] for d in D]);et=np.array([d["et"] for d in D])
pneed=np.array([d["pn"] for d in D]); y=np.array([d["oracle"]=="think" for d in D]).astype(int)

import hashlib
sig=hashlib.md5("|".join(f"otb:{i}" if D[i]["ds"]=="otb" else f'{D[i]["ds"]}:{i}' for i in range(len(D))).encode()).hexdigest()[:10]
# reuse the experiment's embedding cache if present, else embed
cache_files=[f for f in os.listdir("results/pooled") if f.startswith("emb_MiniLM_")]
if cache_files:
    X=np.load(f"results/pooled/{cache_files[0]}")["X"]
    if X.shape[0]!=len(D): X=None
else: X=None
if X is None:
    from sentence_transformers import SentenceTransformer
    X=SentenceTransformer("all-MiniLM-L6-v2").encode(qs,batch_size=64,normalize_embeddings=True).astype(np.float32)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
ds_arr=np.array([d["ds"] for d in D])
strat=np.array([f"{a}_{b}" for a,b in zip(ds_arr,y)])
oof_pool=np.zeros(len(D)); oof_otb=np.zeros(len(D))
for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X,strat):
    oof_pool[te]=LogisticRegression(max_iter=2000).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    otr=tr[ds_arr[tr]=="otb"]
    oof_otb[te]=LogisticRegression(max_iter=2000).fit(X[otr],y[otr]).predict_proba(X[te])[:,1]

def metrics(ch): ch=np.asarray(ch,bool); return float(np.where(ch,et,en).mean()), float(np.where(ch,ct,cn).mean())
def sweep(s):
    ts=np.unique(np.concatenate([np.linspace(0,1,120),np.unique(s)]))
    pts=[metrics(s>=t) for t in ts]+[metrics(np.zeros(len(D),bool)),metrics(np.ones(len(D),bool))]
    best,out=-1,[]
    for e,a in sorted(pts):
        if a>best+1e-12: out.append((e,a)); best=a
    return np.array(out)
n=len(D); e_lo,never=metrics(np.zeros(n,bool)); e_hi,always=metrics(np.ones(n,bool))
dA,dE=ct-cn,et-en; pos=(dA>0)&(dE>0); lams=sorted(set([0.0]+(dA[pos]/dE[pos]).tolist()))
of=[metrics((dA-l*dE)>1e-12) for l in lams]+[(e_lo,never),(e_hi,always)]
best,orc=-1,[]
for e,a in sorted(of):
    if a>best+1e-12: orc.append((e,a)); best=a
orc=np.array(orc)
rng=np.random.default_rng(0); rp=[]
for _ in range(200): rp.append(metrics(rng.random(n)>=0.5))
rand_e=np.mean([p[0] for p in rp]); rand_a=np.mean([p[1] for p in rp])

fig,ax=plt.subplots(figsize=(7.6,4.6))
ax.plot(orc[:,0],orc[:,1],"-",color=INK,lw=1.4,alpha=0.55,zorder=2,label="Oracle frontier (ceiling)")
P=sweep(oof_pool); O=sweep(oof_otb); PN=sweep(pneed)
ax.plot(P[:,0],P[:,1],"-",color=ACC,lw=2.6,zorder=5,label="Pooled router (trained on all 3)")
ax.plot(O[:,0],O[:,1],"--",color=ACC2,lw=2.0,zorder=4,label="OTB-only router")
ax.plot(PN[:,0],PN[:,1],":",color=AMBER,lw=2.0,zorder=4,label="p_need probe (free)")
ax.scatter([e_lo],[never],color=MUT,s=42,zorder=6); ax.annotate("Never-Think",(e_lo,never),xytext=(8,-2),textcoords="offset points",fontsize=8.5,color=MUT)
ax.scatter([e_hi],[always],color=MUT,s=42,zorder=6); ax.annotate("Always-Think",(e_hi,always),xytext=(-6,6),textcoords="offset points",fontsize=8.5,color=MUT,ha="right")
ax.scatter([rand_e],[rand_a],marker="x",color=RED,s=55,lw=2,zorder=6); ax.annotate("random routing",(rand_e,rand_a),xytext=(8,-10),textcoords="offset points",fontsize=8.5,color=RED)
ax.set_xlabel("Energy  (mWh / query, batch-1-equivalent)",fontsize=10)
ax.set_ylabel("Accuracy",fontsize=10)
ax.set_title("Pooled energy–accuracy frontier on the realistic mixture (OTB+Mohseni+GSM8K, N=4,839)",
             fontsize=10.6,color=INK,weight="bold",pad=10)
ax.legend(frameon=False,fontsize=9,loc="lower right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(color=RULE,lw=0.6,alpha=0.5,zorder=0)
fig.tight_layout(); fig.savefig("results/pooled/fig_pooled_pareto.png",bbox_inches="tight",facecolor="white")
print("wrote results/pooled/fig_pooled_pareto.png")
print(f"pooled sweep pts={len(P)}  never={never:.3f}@{e_lo:.0f}  always={always:.3f}@{e_hi:.0f}  rand={rand_a:.3f}@{rand_e:.0f}")
