"""Figures for the 5 Aug Vitali update (palette-matched to the weekly-update artifact).

fig1_mechanism : why J/token collapses -- power is flat while throughput scales
fig2_budget    : E2, the flagship -- token proxy cannot hold an energy budget
fig3_frontier  : E1 + E3 -- accuracy under an energy cap, and the latency price
fig4_validation: E6 -- simulator vs real vLLM server
"""
import json, os, statistics as st, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INK="#1F2823"; MUT="#5C6660"; ACC="#1D6B4F"; ACC2="#3E9B76"; AMBER="#9A6A15"; RED="#8C2F1B"; RULE="#D8DCD2"
plt.rcParams.update({"font.family":"Segoe UI","font.size":11,"text.color":INK,
                     "axes.edgecolor":RULE,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
                     "axes.linewidth":0.9,"figure.dpi":150})
OUT="results/artifact"; os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- fig 1: the mechanism
conc=[json.loads(l) for l in open("results/batching/concurrency.jsonl",encoding="utf-8") if l.strip()]
agg=defaultdict(lambda: defaultdict(list))
for r in conc:
    agg[r["batch"]]["tput"].append(r["throughput_tok_s"]); agg[r["batch"]]["power"].append(r["power_w"])
    agg[r["batch"]]["jtok"].append(r["energy_per_out_token_j"])
B=sorted(agg); TP=[st.mean(agg[b]["tput"]) for b in B]
PW=[st.mean(agg[b]["power"]) for b in B]; JT=[st.mean(agg[b]["jtok"]) for b in B]

fig,ax=plt.subplots(1,2,figsize=(11.5,4.0))
a=ax[0]
a.plot(B,PW,"o-",color=RED,lw=2.2,ms=6,label=f"GPU power  (+{100*(PW[-1]/PW[0]-1):.0f}% over the range)")
a.set_xscale("log",base=2); a.set_xticks(B); a.set_xticklabels(B)
a.set_ylim(0,320); a.set_xlabel("concurrent requests (batch)"); a.set_ylabel("GPU power (W)", color=RED)
a.tick_params(axis="y", labelcolor=RED)
a2=a.twinx()
a2.plot(B,TP,"s-",color=ACC,lw=2.2,ms=6,label=f"throughput  (×{TP[-1]/TP[0]:.0f})")
a2.set_ylabel("throughput (tokens/s)", color=ACC); a2.tick_params(axis="y", labelcolor=ACC)
a.set_title(f"Power barely moves; throughput scales {TP[-1]/TP[0]:.0f}×", fontsize=10.8, weight="bold", pad=10)
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8.5,loc="upper left",frameon=False)
for s_ in ["top"]: a.spines[s_].set_visible(False); a2.spines[s_].set_visible(False)
a=ax[1]
a.plot(B,JT,"o-",color=ACC,lw=2.4,ms=6)
for b,j in zip(B,JT):
    a.annotate(f"{j:.2f}",(b,j),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8.5,color=INK)
a.set_xscale("log",base=2); a.set_yscale("log"); a.set_xticks(B); a.set_xticklabels(B)
a.set_xlabel("concurrent requests (batch)"); a.set_ylabel("energy per token (J)")
a.set_title(f"…so energy per token collapses {JT[0]/JT[-1]:.0f}×", fontsize=10.8, weight="bold", pad=10)
a.grid(alpha=.3, which="both")
for s_ in ["top","right"]: a.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_mechanism.png",bbox_inches="tight",facecolor="white")
print("fig1 ok")

# ---------------------------------------------------------------- fig 2: E2 budget adherence
labels=["token proxy\ncalibrated at\nbatch 1","token proxy\ncalibrated at\nbatch 8","token proxy\ncalibrated at\nbatch 32","measured\nload-aware\nJ(L)"]
v35=[-41.6,115.8,115.8,-0.8]; v60=[-43.8,51.5,51.5,-1.0]
x=np.arange(len(labels)); w=0.36
fig,a=plt.subplots(figsize=(8.6,4.4))
b1=a.bar(x-w/2,v35,w,color=[RED,RED,RED,ACC],label="budget = 35% of always-think")
b2=a.bar(x+w/2,v60,w,color=[RED,RED,RED,ACC],alpha=.55,label="budget = 60%")
for bars,vals in [(b1,v35),(b2,v60)]:
    for bb,vv in zip(bars,vals):
        a.annotate(f"{vv:+.0f}%",(bb.get_x()+bb.get_width()/2,vv),ha="center",
                   va="bottom" if vv>=0 else "top",xytext=(0,4 if vv>=0 else -4),
                   textcoords="offset points",fontsize=9,weight="bold",color=INK)
a.axhline(0,color=INK,lw=1.4)
a.axhspan(-5,5,color=ACC,alpha=.10)
a.annotate("on budget (±5%)",(3.42,7),fontsize=8.5,color=ACC,style="italic",ha="right")
a.set_xticks(x); a.set_xticklabels(labels,fontsize=9)
a.set_ylabel("actual energy vs the authorised budget")
a.set_title("The token proxy cannot hold an energy budget — the measured cost model can",
            fontsize=11,weight="bold",pad=10)
a.set_yticks([-50,0,50,100]); a.set_yticklabels(["−50%","0","+50%","+100%"])
a.legend(fontsize=8.5,frameon=False,loc="upper left")
a.grid(axis="y",alpha=.3)
for s_ in ["top","right"]: a.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_budget.png",bbox_inches="tight",facecolor="white")
print("fig2 ok")

# ---------------------------------------------------------------- fig 3: E3 at matched energy
# Actual E3 numbers (results/e3/e3_tradeoff.md), bimodal, matched 217 Wh cap.
pol=["fixed threshold\n(best of 174 settings)", "load-aware\n(think when BUSY)", "inverse control\n(think when IDLE)"]
acc=[0.8698,0.8852,0.8633]; think=[6.1,12.5,6.5]; lat=[44.8,79.0,37.6]
cols=[AMBER,ACC,RED]
fig,ax=plt.subplots(1,3,figsize=(12.6,4.0))
for a,vals,ttl,ylab,fmt in [
    (ax[0],acc,"Accuracy delivered","accuracy","{:.4f}"),
    (ax[1],think,"Thinking bought for the SAME energy","% of queries routed to think","{:.1f}%"),
    (ax[2],lat,"…and the price: tail latency","p95 latency (s)","{:.1f}s")]:
    bars=a.bar(range(3),vals,color=cols,width=.6)
    for bb,vv in zip(bars,vals):
        a.annotate(fmt.format(vv),(bb.get_x()+bb.get_width()/2,vv),xytext=(0,4),
                   textcoords="offset points",ha="center",fontsize=10,weight="bold",color=INK)
    a.set_xticks(range(3)); a.set_xticklabels(pol,fontsize=8.5)
    a.set_title(ttl,fontsize=10.5,weight="bold",pad=10); a.set_ylabel(ylab,fontsize=9.5)
    a.grid(axis="y",alpha=.3)
    for s_ in ["top","right"]: a.spines[s_].set_visible(False)
ax[0].set_ylim(0.84,0.90)
fig.suptitle("All three policies under the same 217 Wh energy budget",
             fontsize=11.5,weight="bold",y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_frontier.png",bbox_inches="tight",facecolor="white")
print("fig3 ok")

# ---------------------------------------------------------------- fig 4: validation
rows=[json.loads(l) for l in open("results/validation/sim_vs_real.jsonl",encoding="utf-8") if l.strip()]
fig,a=plt.subplots(figsize=(8.4,3.9))
names=["energy\n(static-τ)","energy\n(load-aware)","p95 latency\n(load-aware)","p95 latency\n(static-τ)"]
vals=[rows[0]["err_energy_pct"],rows[1]["err_energy_pct"],rows[1]["err_lat_p95_pct"],rows[0]["err_lat_p95_pct"]]
cols=[ACC,ACC,ACC,AMBER]
bars=a.barh(names[::-1],vals[::-1],color=cols[::-1],height=.6)
for bb,vv in zip(bars,vals[::-1]):
    a.annotate(f"{vv:+.1f}%",(vv,bb.get_y()+bb.get_height()/2),
               xytext=(6 if vv>=0 else -6,0),textcoords="offset points",
               va="center",ha="left" if vv>=0 else "right",fontsize=9.5,weight="bold",color=INK)
a.axvline(0,color=INK,lw=1.3); a.axvspan(-10,10,color=ACC,alpha=.10)
a.annotate("within ±10%",(10,3.35),fontsize=8.5,color=ACC,style="italic")
a.set_xlim(-40,25); a.set_xlabel("simulator prediction error vs the real vLLM server")
a.set_title("The simulator was checked against a real server — energy within 10%",
            fontsize=10.8,weight="bold",pad=10)
a.grid(axis="x",alpha=.3)
for s_ in ["top","right","left"]: a.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_validation.png",bbox_inches="tight",facecolor="white")
print("fig4 ok")
