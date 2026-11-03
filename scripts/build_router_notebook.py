"""Builds notebooks/03_router.ipynb (Phase-3 router + Pareto evaluation).

Consumes results/paired/paired_oracle.jsonl (from notebook 02): per query it has
both modes' correctness and measured energy, plus the energy-aware oracle label.
Trains a lightweight classifier on query embeddings, sweeps its threshold, and
plots the energy-accuracy Pareto vs always/never/random/oracle baselines.

Run:  python scripts/build_router_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Phase-3 — Energy-aware think/no-think router

Trains a lightweight external classifier (query embedding → think / no-think) on the
**energy-aware oracle labels** from notebook 02, then evaluates it on the
**energy–accuracy Pareto plane** against Always-Think, Never-Think, Random, and the
Oracle. Win condition: the router sits **up-and-left** of Always-Think (most of the
accuracy for a fraction of the energy) and approaches the Oracle.

> CPU-only (no GPU needed) — it reuses the measured energy/accuracy from 02.
> Requires `results/paired/paired_oracle.jsonl` (run notebook 02 first).''')

md(r'''## 0. Setup (clone for data + `src`)''')
code(r'''import os, subprocess, sys
REPO, BRANCH = "cemalgndzz/llm-routing-tesina", "main"
try:
    from google.colab import userdata
    TOKEN = userdata.get("GITHUB_TOKEN")
    repo_dir = "/content/" + REPO.split("/")[1]
    auth = f"https://{TOKEN}@github.com/{REPO}.git"
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "clone", "--branch", BRANCH, auth, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
    os.chdir(repo_dir)
except Exception as e:
    print("Not on Colab / no token — assuming local repo. (", repr(e)[:80], ")")
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())''')

code(r'''%pip install -q -U sentence-transformers scikit-learn pandas matplotlib datasets''')

md(r'''## 1. Load the paired dataset (per-query energy + correctness + oracle label)

Prefers the **Claude-judged corrected labels** (`paired_oracle_final.json`) and
falls back to notebook 02's raw output. If the labels lack `question` text (the
corrected file does), it reconstructs `qid → question` by replaying notebook 02's
deterministic OTB sample (`SEED=12345`, 40 per subset).''')
code(r'''import os, json, random
import pandas as pd, numpy as np

FINAL, RAW = "results/paired/paired_oracle_final.json", "results/paired/paired_oracle.jsonl"
if os.path.exists(FINAL):
    df = pd.DataFrame(json.load(open(FINAL))); print("Loaded corrected labels:", FINAL)
elif os.path.exists(RAW):
    df = pd.read_json(RAW, lines=True); print("Loaded:", RAW)
else:
    raise FileNotFoundError("Run notebook 02 first to produce the paired labels.")

# Merge question text if absent (replay notebook 02's deterministic OTB sample).
if "question" not in df.columns or df["question"].isna().any():
    from datasets import load_dataset
    SEED, N_PER_SUBSET = 12345, 40                     # must match notebook 02's full run
    random.seed(SEED)
    ds = load_dataset("facebook/optimal_thinking_bench")["train"]
    allrows = [{"q": r["question"], "s": r["subset"]} for r in ds]
    over  = [r for r in allrows if r["s"] == "overthinkingbench"]
    under = [r for r in allrows if r["s"] == "underthinkingbench"]
    random.shuffle(over); random.shuffle(under)        # shuffles list positions, seed-reproducible
    qmap = {}
    for i, r in enumerate(over[:N_PER_SUBSET]):  qmap[f"over_{i:03d}"]  = r["q"]
    for i, r in enumerate(under[:N_PER_SUBSET]): qmap[f"under_{i:03d}"] = r["q"]
    df["question"] = df["qid"].map(qmap)
    # Silent-failure guard: the reconstruction MUST match known questions from the
    # raw file (same seed/sample). If it doesn't, features would be mismatched.
    if os.path.exists(RAW):
        raw = pd.read_json(RAW, lines=True)
        if "question" in raw.columns:
            known = raw.dropna(subset=["question"]).set_index("qid")["question"].to_dict()
            ok = sum(1 for q, t in known.items() if qmap.get(q) == t)
            print(f"reconstruction check: {ok}/{len(known)} known qid->question match")
            assert ok == len(known), "qid->question mismatch — reconstruction is wrong, do not trust."

need = ["question","correct_think","correct_no_think","energy_wh_think","energy_wh_no_think","oracle"]
df = df.dropna(subset=need).reset_index(drop=True)
print("paired queries:", len(df), "| question merged:", df["question"].notna().all())
print("oracle balance:\n", df["oracle"].value_counts())
print("subsets:\n", df["subset"].value_counts())''')

md(r'''## 2. Features + energy-aware labels

Features: sentence embedding of the query (`all-MiniLM-L6-v2`, à la GreenServ) plus
a cheap length signal. Label: the energy-aware oracle (`think`=1).''')
code(r'''from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode(df["question"].tolist(), normalize_embeddings=True, show_progress_bar=False)
Xlen = (df["question"].str.len().values / 1000.0).reshape(-1, 1)
X = np.hstack([Xe, Xlen])
y = (df["oracle"] == "think").astype(int).values
print("X:", X.shape, "| positives (think):", int(y.sum()), "/", len(y))''')

md(r'''## 3. Train the router (out-of-fold probabilities to avoid leakage)

Data is small, so we use 5-fold `cross_val_predict` to get an honest P(think) for
every query without training on it. (Scale `N_PER_SUBSET` in notebook 02 for a
stronger router — this is the proof-of-concept pipeline.)''')
code(r'''from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
counts = Counter(y); min_class = min(counts.values()) if counts else 0
if len(counts) < 2 or min_class < 2:
    # Too few paired queries / one class only (e.g. smoke data) — CV impossible.
    clf.fit(X, y); p_think = clf.predict_proba(X)[:, 1]
    print(f"classes={dict(counts)} -> in-sample probs only. Raise N_PER_SUBSET in nb 02 "
          "for a meaningful router.")
else:
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    p_think = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    try:
        print(f"Router label AUC (out-of-fold, {n_splits}-fold):", round(roc_auc_score(y, p_think), 3))
    except ValueError:
        print("AUC undefined (single class in a fold).")''')

md(r'''## 4. Policy simulation on the energy–accuracy plane

Every policy assigns a mode per query; we read off that mode's **measured** accuracy
and energy. Baselines: Always/Never/Random and the **Oracle** — but the oracle is not
a single point. Any assignment of think/no-think per query is a valid oracle policy,
so the oracle's real power is its **Pareto frontier**: for every energy budget, the
best accuracy *some* assignment achieves. That frontier is the ceiling for any router;
the single energy-aware label point (`df["oracle"]`) is just one operating point on it.''')
code(r'''ct, cn = df["correct_think"].values, df["correct_no_think"].values
et, en = df["energy_wh_think"].values, df["energy_wh_no_think"].values

def metrics(choose_think):
    choose_think = np.asarray(choose_think, bool)
    acc    = np.where(choose_think, ct, cn).mean()
    energy = np.where(choose_think, et, en).mean() * 1000  # mWh/query
    return energy, acc

def pareto_ul(points):
    """Upper-left frontier: sort by energy asc, keep points with strictly rising acc."""
    pts = sorted(points, key=lambda p: (p[0], -p[1])); out = []
    best = -1.0
    for e, a in pts:
        if a > best + 1e-12:
            out.append((e, a)); best = a
    return out

def oracle_frontier():
    """Exact concave Pareto frontier over per-query think/no-think choices.
    For an energy price lambda, thinking on query i is optimal iff
    dAcc_i - lambda*dEnergy_i > 0. Sweeping lambda from 0 (energy free) up traces
    every Pareto-optimal assignment (cheaper-and-better flips are included at any
    lambda; costlier-and-worse are never taken)."""
    dA = ct - cn
    dE = (et - en) * 1000.0
    ratios = [dA[i] / dE[i] for i in range(len(dA)) if dE[i] > 0 and dA[i] > 0]
    lambdas = sorted(set([0.0] + ratios + [max(ratios) * 1.5 if ratios else 1.0]))
    pts = []
    for lam in lambdas:
        choose = (dA - lam * dE) > 1e-12
        pts.append(metrics(choose))
    pts += [metrics(np.zeros(len(dA), bool)), metrics(np.ones(len(dA), bool))]
    return pareto_ul(pts)

N = len(df); rng = np.random.default_rng(0)
base = {
    "Always-Think": metrics(np.ones(N, bool)),
    "Never-Think":  metrics(np.zeros(N, bool)),
    "Random":       metrics(rng.random(N) < 0.5),
    "Oracle-label":  metrics(df["oracle"].values == "think"),  # our energy-aware label point
}
for k, (e, a) in base.items():
    print(f"{k:13} energy={e:8.1f} mWh/q   acc={a:.3f}")

oracle_front = oracle_frontier()
print(f"Oracle frontier: {len(oracle_front)} Pareto points "
      f"({oracle_front[0][0]:.0f}..{oracle_front[-1][0]:.0f} mWh)")

taus = np.linspace(0, 1, 51)
router = np.array([metrics(p_think >= t) for t in taus])  # columns: energy, acc''')

md(r'''## 5. Pareto plot''')
code(r'''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(router[:, 0], router[:, 1], "-o", ms=3, lw=1.5, color="tab:green",
        label="Router (τ sweep)")
ofx = [e for e, a in oracle_front]; ofy = [a for e, a in oracle_front]
ax.plot(ofx, ofy, "--", lw=1.5, color="tab:red", label="Oracle frontier (ceiling)")
marks = {"Always-Think": "^", "Never-Think": "v", "Random": "x", "Oracle-label": "*"}
for k, (e, a) in base.items():
    ax.scatter([e], [a], marker=marks[k], s=140 if k == "Oracle-label" else 90,
               zorder=5, label=k)
ax.set_xlabel("energy per query (mWh)"); ax.set_ylabel("accuracy")
ax.set_title("Energy–accuracy Pareto: router vs baselines")
ax.legend(); plt.tight_layout()
os.makedirs("results/router", exist_ok=True)
plt.savefig("results/router/pareto.png", dpi=120); plt.show()''')

md(r'''## 6. Energy-Adjusted Accuracy (EAA) — accuracy under an energy budget

EAA(e) = best accuracy a policy reaches without exceeding mean energy `e` (the
upper-left envelope of its points). We integrate EAA over [Never-Think, Always-Think]
and normalize. The **Oracle frontier** is the ceiling; the router's gap to it is the
accuracy a perfect classifier on these labels would recover. (Comparing the router to
the *single* oracle-label point instead would be misleading — a 3-point step function
integrates to nearly the router's own AUC by construction. The frontier is the honest
baseline.)''')
code(r'''_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # trapz removed in newer numpy

def eaa_auc(points, e_lo, e_hi, grid=200):
    # Fair EAA: budgets below a policy's cheapest operating point fall back to Never-Think
    # accuracy (every policy can reject all queries), normalized over the fixed [e_lo,e_hi].
    never_acc = base["Never-Think"][1]
    pts = sorted(points, key=lambda p: p[0])
    es = np.linspace(e_lo, e_hi, grid); accs = []
    for e in es:
        feasible = [a for (pe, a) in pts if pe <= e + 1e-9]
        accs.append(max(feasible) if feasible else never_acc)
    return _trapz(np.array(accs), es) / (e_hi - e_lo)

e_lo, e_hi = base["Never-Think"][0], base["Always-Think"][0]
router_pts = [tuple(r) for r in router]
router_auc = eaa_auc(router_pts, e_lo, e_hi)
oracle_auc = eaa_auc(oracle_front, e_lo, e_hi)
print(f"Energy range: {e_lo:.1f} .. {e_hi:.1f} mWh/query")
print(f"Router          EAA-AUC: {router_auc:.4f}")
print(f"Oracle frontier EAA-AUC: {oracle_auc:.4f}  (ceiling)")
print(f"Router captures {100*(router_auc - base['Never-Think'][1]) / (oracle_auc - base['Never-Think'][1] + 1e-12):.1f}% "
      "of the oracle's achievable gain over Never-Think")
print("Always acc:", round(base["Always-Think"][1], 3),
      "| Never acc:", round(base["Never-Think"][1], 3))
# Headline operating point: router accuracy at ~half the Always-Think energy budget
half = e_hi * 0.5
r_half = [a for (e, a) in router_pts if e <= half]
o_half = [a for (e, a) in oracle_front if e <= half]
print(f"At <= {half:.0f} mWh/q (half of Always-Think):  "
      f"router acc={max(r_half):.3f}  vs  oracle acc={max(o_half):.3f}")''')

md(r'''## 7. Notes / next

- **Read the plot:** the router curve should bow up-and-left of the Always–Never chord
  and sit **below the red Oracle-frontier line**. The vertical gap to that line at a
  given energy is the accuracy a perfect classifier on these labels would recover — the
  room a better router (more data, fine-tuned encoder) has to grow.
- **Why the frontier, not the star:** the single `Oracle-label` point is one operating
  point; scoring it alone made Router and Oracle EAA-AUC coincide (a 3-point step
  integrates to ~the router's own AUC). The frontier is the true ceiling.
- **Data size:** ~250 paired queries (150/98 over/under) with a real 21% think-label
  balance — enough for a meaningful curve. Raise `N_PER_SUBSET` in notebook 02 for more
  power; optionally pretrain on Mohseni's ~70k labels.
- **Energy vs token labels (RQ4):** at batch=1, energy ∝ tokens so the two label sets
  nearly coincide — the interesting divergence is **load-dependent** (from the batching
  result: think is ~40× cheaper per token under high concurrency, so the energy-optimal
  policy thinks more than the token-optimal one). Re-deriving labels at a realistic
  batch is the natural follow-up that ties the router back to the batching finding.''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]; c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "03_router.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
