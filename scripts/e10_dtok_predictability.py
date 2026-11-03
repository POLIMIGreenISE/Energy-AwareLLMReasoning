"""E10 -- can the router ESTIMATE the token cost it needs, and does the gain survive if it does?

E8's decomposition (results/e8/CONSOLIDATED.md S3) produced an uncomfortable result: the ingredient
that earns accuracy at a matched energy cap is not live load -- it is the PER-QUERY TOKEN COUNT.
Ranking by benefit-per-token (score/dtok) beats a plain score threshold by +1.8 to +5.0pp, while
load-awareness alone is worth roughly nothing outside bursty traffic.

That is a problem, because dtok(q) is not observable before generating. Reporting the gain and
stopping there would be reporting an oracle. So this asks the constructive question instead:

  1. IS dtok(q) PREDICTABLE from the query text alone?  (a regression, honestly cross-validated)
  2. DOES THE GAIN SURVIVE when the policy uses the PREDICTION instead of the truth?

Question 2 is the one that matters. A high correlation that does not convert into routing value
would repeat exactly the mistake RQ3 already documented -- ranking the label is not the same as
routing profitably (see results/crossdataset/official_router_eval.md).

Note the asymmetry that makes this plausible: predicting whether thinking will FLIP an answer is
predicting the model's own competence, and RQ3 showed that sits at chance within a topic. Predicting
HOW LONG a reasoning chain will be is predicting a property of the QUESTION. Those are different
difficulties, and the second one is the one this policy actually needs.

Out-of-fold predictions only -- a query's dtok estimate never comes from a model that saw it.

Free -- no GPU. Out: results/e10/{DTOK_PREDICTABILITY.md, fig_dtok.png, dtok_pred.json}
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate
from src.capfit import refine_to_bound

OUT = "results/e10"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
QTEXT = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}

QIDS = [q for q in wl.qids if q in QTEXT]
TXT = [QTEXT[q] for q in QIDS]
DTOK = np.array([wl.dtok(q) for q in QIDS], float)
DT_MEAN = float(DTOK.mean())
t0 = time.time()
print(f"{len(QIDS)} queries | dtok mean {DT_MEAN:.0f} median {np.median(DTOK):.0f} "
      f"min {DTOK.min():.0f} max {DTOK.max():.0f}")

# ============================================================ 1. predict dtok from text
# Deliberately cheap and deployable: character + word TF-IDF into a ridge on log(dtok). A router
# that needed a second LLM to price a query would have priced itself out of the problem.
Y = np.log(np.maximum(DTOK, 1.0))
FEATS = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                   max_features=60000, sublinear_tf=True))

def cv_predict(model_fn, folds=5, seed=0):
    """Out-of-fold predictions -- a query's estimate never comes from a model that saw it."""
    oof = np.zeros(len(QIDS))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(QIDS):
        m = model_fn()
        m.fit([TXT[i] for i in tr], Y[tr])
        oof[te] = m.predict([TXT[i] for i in te])
    return oof

MODELS = {
    "constant E[dtok]": lambda: None,                       # handled specially
    "prompt length only": lambda: None,                     # handled specially
    "TF-IDF + ridge": lambda: make_pipeline(FEATS, Ridge(alpha=1.0)),
}
preds, rows = {}, []
plen = np.array([len(t.split()) for t in TXT], float)

for name in MODELS:
    if name == "constant E[dtok]":
        yhat = np.full(len(QIDS), np.log(DT_MEAN))
    elif name == "prompt length only":
        yhat = np.zeros(len(QIDS))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(QIDS):
            A = np.c_[np.log1p(plen[tr]), np.ones(len(tr))]
            coef, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
            yhat[te] = np.c_[np.log1p(plen[te]), np.ones(len(te))] @ coef
    else:
        yhat = cv_predict(MODELS[name])
    dhat = np.exp(yhat)
    preds[name] = dhat
    rho = float("nan") if np.ptp(dhat) == 0 else spearmanr(dhat, DTOK).statistic   # constant -> undefined
    r2 = 1 - np.sum((Y - yhat) ** 2) / np.sum((Y - Y.mean()) ** 2)
    # ratio error: what matters for a PRICE is being right to within a factor, not to within a token
    ratio = np.exp(np.abs(yhat - Y))
    rows.append(dict(model=name, spearman=float(rho), r2_log=float(r2),
                     median_ratio_err=float(np.median(ratio)),
                     frac_within_2x=float(np.mean(ratio <= 2.0))))
    print(f"  {name:22s} rho {rho:+.3f} | R2(log) {r2:+.3f} | median ratio err {np.median(ratio):.2f}x "
          f"| within 2x {np.mean(ratio <= 2.0)*100:.0f}%")

BEST = "TF-IDF + ridge"
DHAT = {q: float(max(preds[BEST][i], 1.0)) for i, q in enumerate(QIDS)}
DTRUE = {q: float(max(DTOK[i], 1.0)) for i, q in enumerate(QIDS)}

# ============================================================ 2. does the gain survive?
# The same protocol as E8: matched energy, dense per-family grids, floor and ceiling visible.
N_TRACE, EVAL = 500, list(range(12))
CAPS = [0.25, 0.40]
REGIMES = {
    "near-idle": dict(hi_rate=0.20, lo_rate=0.20, hi_s=30.0, lo_s=30.0),
    "saturated": dict(hi_rate=3.00, lo_rate=3.00, hi_s=30.0, lo_s=30.0),
    "bursty":    dict(hi_rate=6.00, lo_rate=0.03, hi_s=6.0,  lo_s=110.0),
}
FLIP = sorted([q for q in wl.qids if wl.dacc(q) > 0], key=lambda q: wl.dtok(q))

def mk_static(t, _s):      return lambda L, q, c: SCORE[q] >= t
def mk_tok_true(t, _s):    return lambda L, q, c: SCORE[q] > t * DTRUE.get(q, DT_MEAN)
def mk_tok_pred(t, _s):    return lambda L, q, c: SCORE[q] > t * DHAT.get(q, DT_MEAN)
def mk_load_const(l, _s):  return lambda L, q, c: SCORE[q] > l * hw.jtok(L) * DT_MEAN
def mk_load_pred(l, _s):   return lambda L, q, c: SCORE[q] > l * hw.jtok(L) * DHAT.get(q, DT_MEAN)
def mk_load_true(l, _s):   return lambda L, q, c: SCORE[q] > l * hw.jtok(L) * DTRUE.get(q, DT_MEAN)
def mk_random(r, s):
    rng = np.random.default_rng(1000 + s)
    return lambda L, q, c: rng.random() < r
def mk_oracle(k, _s):
    chosen = set(FLIP[:int(k)])
    return lambda L, q, c: q in chosen

def score_grid(vals, n=180):
    u = np.unique(np.asarray(vals, float))
    return np.unique(np.concatenate([[0.0], u[:: max(1, len(u) // n)], u[-3:], [u[-1] * 1.01]]))

def price_grid(sgrid, scale, target=220):
    c = np.concatenate([np.asarray(sgrid, float) / max(scale(L), 1e-12) for L in (1, 4, 16, 48)])
    c = np.unique(c[np.isfinite(c) & (c > 0)])
    if len(c) > target: c = c[:: max(1, len(c) // target)]
    return np.unique(np.concatenate([c, np.logspace(-8, 4, 40)]))

TAU = score_grid([SCORE[q] for q in wl.qids])
TOK_T = score_grid([SCORE[q] / DTRUE.get(q, DT_MEAN) for q in wl.qids])
TOK_P = score_grid([SCORE[q] / DHAT.get(q, DT_MEAN) for q in wl.qids])
FAMILIES = {
    "static-tau":            (TAU, mk_static),
    "token (true dtok)":     (TOK_T, mk_tok_true),
    "token (PREDICTED)":     (TOK_P, mk_tok_pred),
    "load-aware (const)":    (price_grid(TAU, lambda L: hw.jtok(L) * DT_MEAN), mk_load_const),
    "load-aware (PREDICTED)": (price_grid(TOK_P, hw.jtok), mk_load_pred),
    "load-aware (true dtok)": (price_grid(TOK_T, hw.jtok), mk_load_true),
    "random@rate":           (np.unique(np.r_[np.linspace(0, .3, 31), np.linspace(.35, 1, 14)]), mk_random),
    "oracle":                (np.unique(np.linspace(0, len(FLIP), 41).astype(int)), mk_oracle),
}
KEYS = ("energy_wh", "acc", "lat_p95", "think_rate")
# How each family's parameter is refined at a cap, so none is scored on budget its own grid cannot
# reach (src/capfit.py, results/CORRECTIONS.md).
SCALE = {"static-tau": "linear", "random@rate": "linear", "oracle": "int"}

def run(rg, fac, v):
    rs = [simulate(make_trace(wl, N_TRACE, s, **rg), fac(v, s), wl, hw) for s in EVAL]
    return {k: np.array([r[k] for r in rs], float) for k in KEYS}

def boot(d, b=20000, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def best(pts, cap):
    f = [p for p in pts if p["energy_wh"].mean() <= cap + 1e-9]
    return max(f, key=lambda p: p["acc"].mean()) if f else None

print(f"\ngrids: " + "  ".join(f"{k} {len(g)}" for k, (g, _) in FAMILIES.items()))
sim_rows = []
for rname, rg in REGIMES.items():
    nev = run(rg, lambda p, s: (lambda L, q, c: False), None)
    alw = run(rg, lambda p, s: (lambda L, q, c: True), None)
    lo, hi = nev["energy_wh"].mean(), alw["energy_wh"].mean()
    SW = {f: [dict(param=float(v), **run(rg, fac, v)) for v in grid] for f, (grid, fac) in FAMILIES.items()}
    for frac in CAPS:
        cap = lo + frac * (hi - lo)
        for f, (_, fac) in FAMILIES.items():
            refine_to_bound(SW[f], cap, lambda v, fac=fac: run(rg, fac, v),
                            scale=SCALE.get(f, "log"))
        sel = {f: best(SW[f], cap) for f in FAMILIES}
        base = sel["static-tau"]
        for f, p in sel.items():
            if p is None or base is None: continue
            d = (p["acc"] - base["acc"]) * 100
            l_, h_ = (0.0, 0.0) if f == "static-tau" else boot(d)
            sim_rows.append(dict(regime=rname, frac=frac, policy=f, acc=p["acc"].mean(),
                                 wh=p["energy_wh"].mean(), think=p["think_rate"].mean(),
                                 p95=p["lat_p95"].mean(), gap=float(d.mean()), lo=l_, hi=h_))
    print(f"[{time.time()-t0:5.0f}s] {rname} done")

json.dump(dict(regression=rows, routing=sim_rows,
               dhat={q: DHAT[q] for q in QIDS}), open(f"{OUT}/dtok_pred.json", "w"), indent=1)

# ============================================================ figure
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_POS, C_NEG, C_TRUE = "#2a78d6", "#e34948", "#9a9a95"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.8))
ax = axes[0]
ax.scatter(preds[BEST], DTOK, s=5, alpha=.25, color=C_POS, edgecolors="none")
lim = [80, max(DTOK.max(), preds[BEST].max()) * 1.1]
ax.plot(lim, lim, color=MUTED, lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted Δtokens (out-of-fold)"); ax.set_ylabel("true Δtokens")
_best_row = [r for r in rows if r["model"] == BEST][0]
ax.set_title(f"1. Δtokens is predictable\nSpearman ρ = {_best_row['spearman']:.2f}, "
             f"{_best_row['frac_within_2x']*100:.0f}% within 2×",
             fontsize=9.5, color=INK, loc="left", pad=6)
ax.grid(True, color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)

SHOW = ["token (PREDICTED)", "load-aware (PREDICTED)", "token (true dtok)"]
for ax, rname in zip(axes[1:], REGIMES):
    g = [r for r in sim_rows if r["regime"] == rname and r["frac"] == 0.40 and r["policy"] in SHOW]
    g = sorted(g, key=lambda r: SHOW.index(r["policy"]))
    x = np.arange(len(g)); v = [r["gap"] for r in g]
    cols = [(C_TRUE if "true" in r["policy"] else (C_POS if r["gap"] >= 0 else C_NEG)) for r in g]
    ax.bar(x, v, width=.6, color=cols, zorder=3)
    ax.errorbar(x, v, yerr=[[r["gap"] - r["lo"] for r in g], [r["hi"] - r["gap"] for r in g]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["token\n(predicted)", "load-aware\n(predicted)", "token\n(true, oracle)"], fontsize=7.5)
    ax.set_title(f"{rname}", fontsize=9.5, color=INK, loc="left", pad=6)
    ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
axes[1].set_ylabel("accuracy gap (pp)\nvs static threshold")
fig.suptitle("Δtokens can be estimated from the question — and the routing gain survives the estimate  ·  "
             "40% energy cap  ·  grey = oracle upper bound", fontsize=10.5, color=INK, x=.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(f"{OUT}/fig_dtok.png", dpi=200); plt.close(fig)

# ============================================================ write-up
def tab(h, rs): return "\n".join(["|" + "|".join(h) + "|", "|" + "|".join("--" for _ in h) + "|"] + rs) + "\n"
L = ["# E10 — estimating the token cost, and whether the routing gain survives it\n"]
L.append("E8 §3 found that the ingredient earning accuracy at a matched energy cap is the **per-query "
         "token count**, not live load. But Δtokens is not observable before generating, so that gain "
         "was an oracle result. This asks whether it can be earned honestly.\n")
L.append("## 1. Is Δtokens predictable from the question text?\n")
L.append("Out-of-fold, 5-fold CV. A price only needs to be right to within a *factor*, so the error "
         "measure is the ratio, not the token count.\n")
L.append(tab(["model", "Spearman ρ", "R² (log)", "median ratio error", "within 2×"],
             [f"|{r['model']}|{r['spearman']:+.3f}|{r['r2_log']:+.3f}|{r['median_ratio_err']:.2f}×|"
              f"{r['frac_within_2x']*100:.0f}%|" for r in rows]))
L.append("**Contrast with RQ3.** Predicting whether thinking will *flip* an answer means predicting "
         "the model's own competence, and that sits at chance within a fixed topic "
         "(`results/pooled/WHY_ROUTING_FAILS.md`). Predicting *how long* the reasoning will run is a "
         "property of the question, not of the model's competence — a different and easier target, and "
         "the one this policy actually needs.\n")
L.append("## 2. Does the routing gain survive the estimate?\n")
L.append("Same protocol as E8: matched energy, dense per-family grids, 12 held-out traces × 500 "
         "queries, paired bootstrap CIs. `(true dtok)` rows are the oracle upper bound.\n")
for rname in REGIMES:
    rs = []
    for frac in CAPS:
        for r in [x for x in sim_rows if x["regime"] == rname and x["frac"] == frac]:
            g = "—" if r["policy"] == "static-tau" else f"{r['gap']:+.2f}pp [{r['lo']:+.2f}, {r['hi']:+.2f}]"
            rs.append(f"|{int(frac*100)}%|{r['policy']}|{r['acc']:.4f}|{r['think']*100:.0f}%|"
                      f"{r['wh']:.1f}|{r['p95']:.0f} s|{g}|")
    L.append(f"**{rname}**\n")
    L.append(tab(["cap", "policy", "accuracy", "think", "Wh", "p95", "gap vs static, 95% CI"], rs))
L.append("![dtok](fig_dtok.png)\n")
L.append("\n*Repro: `python scripts/e10_dtok_predictability.py` (CPU).*\n")
open(f"{OUT}/DTOK_PREDICTABILITY.md", "w", encoding="utf-8").write("\n".join(L))
print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/DTOK_PREDICTABILITY.md + fig")
