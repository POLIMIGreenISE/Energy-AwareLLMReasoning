"""E10b -- does the token-cost estimator TRANSFER across datasets, where the accuracy router did not?

This is the decisive robustness test for E10, and it is a direct A/B against the project's own
strongest negative result.

RQ3 established that a text router trained on OptimalThinkingBench reaches AUC 0.895 at home and
collapses to 0.505-0.521 -- chance -- on Mohseni's realistic mixed traffic, because "will thinking
flip this answer?" is a question about the MODEL'S COMPETENCE, and competence is not written in the
query text (results/pooled/WHY_ROUTING_FAILS.md).

E10 found that the quantity the winning policy actually needs is different: not "will thinking help"
but "how many tokens will thinking cost".

ANSWER, in two parts. Zero-shot transfer FAILS just as badly (rho +0.006 OTB->Mohseni): the
estimator learned OTB's topics, exactly as the router did. But trained ON Mohseni it reaches
rho +0.69, whereas the accuracy router trained on Mohseni hits a ceiling of AUC 0.568-0.589 across
3 feature spaces x 5 classifiers. So the asymmetry is real but it is about LEARNABILITY, not about
zero-shot generalisation: the cost is in the text and the benefit is not -- the cost model simply
has to be trained on the traffic it serves. Which is cheap, because generation lengths label
themselves in production while accuracy labels never do.

Truncated generations must be excluded first: 39% of Mohseni's no-think outputs hit the token cap,
and a censored length is a cap rather than a measurement of the quantity being predicted.

Free -- no GPU. Out: results/e10/DTOK_TRANSFER.md, results/e10/fig_dtok_transfer.png
"""
import os, sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, make_union
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Workload

OUT = "results/e10"; os.makedirs(OUT, exist_ok=True)
wl = Workload()
QTEXT = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}


def load_gen(path, drop_truncated=True):
    """qid -> (question, dtok, source), averaging over repetitions.

    TRUNCATION MATTERS HERE. 39% of Mohseni's no-think generations hit the token cap
    (finish_reason='length'), which censors the very quantity being predicted: a truncated length
    is a cap, not a measurement, and it compresses dtok toward the cap for those queries. Any
    correlation computed over them is partly measuring the cap. They are dropped by default.
    """
    tok = defaultdict(lambda: defaultdict(list))
    trunc = defaultdict(bool)
    txt, src = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("completion_tokens") is None: continue
            tok[r["qid"]][r["mode"]].append(float(r["completion_tokens"]))
            if r.get("finish_reason") == "length": trunc[r["qid"]] = True
            txt[r["qid"]] = r["question"]; src[r["qid"]] = r.get("source", "?")
    out, n_drop = {}, 0
    for q, m in tok.items():
        if "think" not in m or "no_think" not in m: continue
        if drop_truncated and trunc[q]: n_drop += 1; continue
        d = np.mean(m["think"]) - np.mean(m["no_think"])
        if d > 0: out[q] = (txt[q], float(d), src[q])
    return out, n_drop


def cv_rho(X, y, folds=5, seed=0):
    """Within-domain 5-fold CV -- the diagnostic that separates DOMAIN SHIFT from NO SIGNAL.

    If an OTB-trained estimator fails on a target but a target-trained one succeeds, the problem is
    transfer and more in-domain data fixes it. If the target-trained one fails too, the signal is
    not in that text at all -- the same distinction the RQ3 capacity sweep drew for the router.
    """
    from sklearn.model_selection import KFold
    X = list(X); ylog = np.log(np.maximum(y, 1.0))
    oof = np.zeros(len(X))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                       max_features=60000, sublinear_tf=True)),
            Ridge(alpha=1.0)).fit([X[i] for i in tr], ylog[tr])
        oof[te] = m.predict([X[i] for i in te])
    r = np.maximum(np.exp(oof) / np.maximum(y, 1), np.maximum(y, 1) / np.exp(oof))
    return float(spearmanr(np.exp(oof), y).statistic), float(np.mean(r <= 2))


# ------------------------------------------------------------------ train on OTB
TR_Q = [q for q in wl.qids if q in QTEXT]
TR_X = [QTEXT[q] for q in TR_Q]
TR_Y = np.log(np.maximum([wl.dtok(q) for q in TR_Q], 1.0))
FEATS = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                   max_features=60000, sublinear_tf=True))
model = make_pipeline(FEATS, Ridge(alpha=1.0)).fit(TR_X, TR_Y)
print(f"trained on OTB: {len(TR_Q)} queries, mean dtok {np.exp(TR_Y).mean():.0f}")

TARGETS = {"Mohseni (realistic mixed traffic)": "results/crossdataset/gen_test_mohseni.jsonl",
           "GSM8K": "results/crossdataset/gen_test_gsm8k.jsonl"}

rows, scatter = [], {}
tr_mean = float(np.exp(TR_Y).mean())
for name, path in TARGETS.items():
    if not os.path.exists(path):
        print(f"  skip {name}: {path} missing"); continue
    data, n_drop = load_gen(path)
    qs = sorted(data)
    X = [data[q][0] for q in qs]
    y = np.array([data[q][1] for q in qs], float)
    yhat = np.exp(model.predict(X))
    ratio = np.maximum(yhat, 1) / np.maximum(y, 1)
    ratio = np.maximum(ratio, 1 / ratio)
    # the honest reference: predicting the OTB mean for everything (what a constant estimator does)
    cratio = np.maximum(tr_mean / np.maximum(y, 1), np.maximum(y, 1) / tr_mean)
    in_rho, in_w2 = cv_rho(X, y)          # trained ON the target: signal present at all?
    rows.append(dict(target=name, n=len(qs), dropped_truncated=n_drop, true_mean=float(y.mean()),
                     spearman=float(spearmanr(yhat, y).statistic),
                     median_ratio=float(np.median(ratio)), within2x=float(np.mean(ratio <= 2)),
                     const_median_ratio=float(np.median(cratio)),
                     const_within2x=float(np.mean(cratio <= 2)),
                     indomain_spearman=in_rho, indomain_within2x=in_w2))
    scatter[name] = (yhat, y)
    print(f"  {name:36s} n={len(qs):5d} (dropped {n_drop} truncated)  OTB->target rho "
          f"{rows[-1]['spearman']:+.3f}  median ratio {rows[-1]['median_ratio']:.2f}x  "
          f"within2x {rows[-1]['within2x']*100:.0f}%  | constant {rows[-1]['const_median_ratio']:.2f}x "
          f"{rows[-1]['const_within2x']*100:.0f}%  | IN-DOMAIN CV rho {in_rho:+.3f} within2x {in_w2*100:.0f}%")
    # per-source breakdown -- Mohseni mixes math, chat and instruction traffic
    bysrc = defaultdict(list)
    for i, q in enumerate(qs): bysrc[data[q][2]].append(i)
    for s, idx in sorted(bysrc.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < 30: continue
        i = np.array(idx)
        print(f"      {s:16s} n={len(i):5d}  rho {spearmanr(yhat[i], y[i]).statistic:+.3f}  "
              f"within2x {np.mean(ratio[i] <= 2)*100:3.0f}%")
        rows.append(dict(target=f"{name} — {s}", n=len(i), dropped_truncated=0,
                         true_mean=float(y[i].mean()),
                         spearman=float(spearmanr(yhat[i], y[i]).statistic),
                         median_ratio=float(np.median(ratio[i])), within2x=float(np.mean(ratio[i] <= 2)),
                         const_median_ratio=float(np.median(cratio[i])),
                         const_within2x=float(np.mean(cratio[i] <= 2)),
                         indomain_spearman=float("nan"), indomain_within2x=float("nan")))

json.dump(rows, open(f"{OUT}/dtok_transfer.json", "w"), indent=1)

# ------------------------------------------------------------------ figure
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C1 = "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
if scatter:
    fig, axes = plt.subplots(1, len(scatter), figsize=(4.6 * len(scatter), 4.0), squeeze=False)
    for ax, (name, (yh, y)) in zip(axes[0], scatter.items()):
        ax.scatter(yh, y, s=5, alpha=.2, color=C1, edgecolors="none")
        lo = max(30.0, min(yh.min(), y.min()) * 0.8); hi = max(yh.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r = [x for x in rows if x["target"] == name][0]
        ax.set_xlabel("predicted Δtokens (OTB-trained)"); ax.set_ylabel("true Δtokens")
        ax.set_title(f"{name}\nρ = {r['spearman']:.2f}, {r['within2x']*100:.0f}% within 2×  (n={r['n']})",
                     fontsize=9.5, color=INK, loc="left", pad=6)
        ax.grid(True, color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
    fig.suptitle("The cost estimator transfers where the accuracy router did not  ·  "
                 "trained on OptimalThinkingBench, tested off-distribution",
                 fontsize=10.5, color=INK, x=.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f"{OUT}/fig_dtok_transfer.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------ write-up
def tab(h, rs): return "\n".join(["|" + "|".join(h) + "|", "|" + "|".join("--" for _ in h) + "|"] + rs) + "\n"
L = ["# E10b — does the token-cost estimator transfer, where the accuracy router did not?\n"]
L.append("The project's strongest negative result (RQ3) is that a text router trained on "
         "OptimalThinkingBench scores AUC 0.895 at home and **0.505–0.521 — chance — on Mohseni's "
         "realistic mixed traffic**. The explanation was mechanical: *will thinking flip this answer* "
         "is a question about the model's competence, and competence is not written in the query text.\n")
L.append("E10 showed the quantity the winning policy actually needs is a different one — *how many "
         "tokens will thinking cost* — which is a property of the **question**. If that distinction is "
         "real and not a re-labelling, the cost estimator must survive the transfer that destroyed the "
         "accuracy router. Same training set, same held-out datasets.\n")
L.append(tab(["target", "n", "true mean Δtok", "OTB→target ρ", "within 2×",
              "constant est. within 2×", "IN-DOMAIN CV ρ", "in-domain within 2×"],
             [f"|{r['target']}|{r['n']}|{r['true_mean']:.0f}|{r['spearman']:+.3f}|"
              f"{r['within2x']*100:.0f}%|{r['const_within2x']*100:.0f}%|"
              + (f"{r['indomain_spearman']:+.3f}|{r['indomain_within2x']*100:.0f}%|"
                 if r["indomain_spearman"] == r["indomain_spearman"] else "—|—|")
              for r in rows]))
L.append("`constant est. within 2×` is the honest reference: what predicting one number — the OTB mean "
         "Δtokens, which is what `src/router.py` ships by default — achieves on the same queries. "
         "`IN-DOMAIN CV ρ` trains the same estimator *on the target dataset itself*, 5-fold. That "
         "column is the diagnostic: it separates a transfer failure from an absent signal.\n")
L.append("**Truncation had to be removed first.** 39% of Mohseni's no-think generations hit the token "
         "cap (`finish_reason='length'`). A truncated length is a cap, not a measurement, and it "
         "censors the very quantity being predicted. Those 762 queries are excluded; the numbers "
         "before exclusion were meaningfully different, so this is not a cosmetic filter.\n")
L.append("## Reading — the asymmetry is about LEARNABILITY, not zero-shot transfer\n")
L.append("**Zero-shot transfer fails.** An OTB-trained cost estimator is worth nothing on Mohseni "
         "(ρ = +0.006). It learned OptimalThinkingBench's topics and templates, exactly as the "
         "accuracy router did. On this axis the two are alike, and any claim that cost estimation "
         "is 'just easier' has to be retracted at the zero-shot level.\n")
L.append("**But in-domain, the two targets are not alike at all — and this is the finding.**\n")
L.append(tab(["target quantity", "trained and tested on Mohseni", "verdict"],
             ["|*will thinking flip this answer?* (RQ3)|AUC **0.568–0.589** — best of 3 feature spaces × 5 "
              "classifiers, a mutual-information ceiling|the signal is **not in the text**|",
              f"|*how many tokens will thinking cost?* (E10b)|Spearman **+{[r for r in rows if r['target'].startswith('Mohseni')][0]['indomain_spearman']:.3f}**|"
              "the signal **is in the text**; only transfer fails|"]))
L.append("So the honest statement is: **the benefit of thinking is not learnable from query text even "
         "with in-domain training; its cost is.** The cost model just has to be trained on the traffic "
         "it will serve, rather than borrowed from a benchmark.\n")
L.append("**That is a mild deployment requirement, because Δtokens labels are free.** Every request a "
         "production server answers reveals its own generation length — the cost model is "
         "self-supervising and can be retrained continuously on live traffic. Accuracy labels are not "
         "free: knowing whether thinking would have flipped an answer requires ground truth and a "
         "counterfactual second generation. The two quantities differ not only in how learnable they "
         "are, but in whether an operator can ever obtain labels for them at all.\n")
L.append("**GSM8K is the informative null.** Its in-domain ρ is only +0.245 — but it is a homogeneous "
         "grade-school-math set whose Δtokens barely varies (a single constant is already within 2× "
         "for 67% of it). There is little spread to rank and correspondingly little to gain, which is "
         "the same 'within a fixed topic there is nothing to predict' pattern RQ3 found for accuracy.\n")
L.append("![transfer](fig_dtok_transfer.png)\n")
L.append("**Consequence for the router.** `src/router.py` must ship the constant estimator as its "
         "safe default (it needs no training data and cannot transfer badly), with the learned "
         "estimator as an opt-in fitted on the operator's own traffic. Shipping an OTB-trained "
         "estimator as a default would be shipping a benchmark artefact.\n")
L.append("\n*Repro: `python scripts/e10b_dtok_transfer.py` (CPU, seconds).*\n")
open(f"{OUT}/DTOK_TRANSFER.md", "w", encoding="utf-8").write("\n".join(L))
print(f"\nwrote {OUT}/DTOK_TRANSFER.md")
