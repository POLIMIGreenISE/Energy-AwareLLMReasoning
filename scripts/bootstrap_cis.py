"""Tier-1 validity: bootstrap 95% CIs for every headline number (offline, no GPU).

Percentile bootstrap over QUERIES (resample qids with replacement), B=2000.
Covers:
  RQ1  mean energy per mode (b1eq mWh/query) and the think/no_think ratio
  RQ2  accuracy per mode x subset and the thinking gain (hard +42pp / easy +3pp)
  RQ3  router EAA-AUC, oracle-frontier EAA-AUC, % of oracle gain captured
  RQ4  load-adaptive tau(L) gain over static tau at batch 64 (lambda=1.0/Wh)
       and the 100 mWh/query budget gap

Router CIs are CONDITIONAL on the fitted classifier: P(think) comes from the
full-data 5-fold out-of-fold predictions (identical to notebook 03) and is held
fixed while queries are resampled. This quantifies evaluation uncertainty, not
retraining variance (refitting inside a bootstrap duplicates queries across CV
folds and leaks). Stated in the output.

Run:  python scripts/bootstrap_cis.py
Out:  results/audit/bootstrap_cis.md
"""
import json, os, statistics as st
from collections import defaultdict

import numpy as np

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(12345)
B = 2000
J_B1 = 3.7
WH = 1 / 3600.0

paired = json.load(open("results/paired/paired_oracle_final.json"))
N = len(paired)
cn = np.array([p["correct_no_think"] for p in paired])
ct = np.array([p["correct_think"] for p in paired])
en = np.array([p["energy_wh_no_think"] for p in paired]) * 1000  # mWh b1eq
et = np.array([p["energy_wh_think"] for p in paired]) * 1000
tok_nt = en / 1000 * 3600 / J_B1
tok_th = et / 1000 * 3600 / J_B1
subset = np.array([p["subset"] for p in paired])
y = np.array([p["oracle"] == "think" for p in paired]).astype(int)
easy, hard = subset == "overthinkingbench", subset == "underthinkingbench"
print(f"queries: {N} (easy {easy.sum()}, hard {hard.sum()}), bootstrap B={B}")

def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return lo, hi

rows = []  # (name, point, lo, hi)
def report(name, point, samples, fmt="{:.3f}"):
    lo, hi = ci(samples)
    rows.append((name, fmt.format(point), fmt.format(lo), fmt.format(hi)))
    print(f"  {name:55} {fmt.format(point):>8}  [{fmt.format(lo)}, {fmt.format(hi)}]")

IDX = RNG.integers(0, N, size=(B, N))  # shared resamples for all statistics

# ---------------------------------------------------------------- RQ1 energy
print("\nRQ1 — energy per mode (b1eq mWh/query):")
report("no_think mean energy (mWh/q)", en.mean(), en[IDX].mean(axis=1), "{:.0f}")
report("think mean energy (mWh/q)", et.mean(), et[IDX].mean(axis=1), "{:.0f}")
report("think / no_think energy ratio", et.mean() / en.mean(),
       et[IDX].mean(axis=1) / en[IDX].mean(axis=1), "{:.1f}")

# ---------------------------------------------------------------- RQ2 accuracy
print("\nRQ2 — accuracy per mode x subset (bootstrap within subset):")
for name, mask in [("easy (overthinking)", easy), ("hard (underthinking)", hard)]:
    ids = np.flatnonzero(mask)
    bs = ids[RNG.integers(0, len(ids), size=(B, len(ids)))]
    report(f"{name}: no_think acc", cn[ids].mean(), cn[bs].mean(axis=1))
    report(f"{name}: think acc", ct[ids].mean(), ct[bs].mean(axis=1))
    report(f"{name}: thinking gain (pp)", (ct[ids] - cn[ids]).mean() * 100,
           (ct[bs] - cn[bs]).mean(axis=1) * 100, "{:+.1f}")

# ---------------------------------------------------------------- RQ3 router / oracle EAA
# P(think): identical pipeline to notebook 03 (full data, 5-fold OOF), then FIXED.
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

emb = SentenceTransformer("all-MiniLM-L6-v2")
Xe = emb.encode([p["question"] for p in paired], normalize_embeddings=True,
                show_progress_bar=False)
qlen = np.array([len(p["question"]) for p in paired]).reshape(-1, 1) / 1000.0
X = np.hstack([Xe, qlen])
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p_think = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

TAUS_R = np.linspace(0, 1, 51)              # notebook 03's sweep
CHOOSE_R = p_think[None, :] >= TAUS_R[:, None]   # (51, N)

def pareto_ul(points):
    pts = sorted(points, key=lambda p: (p[0], -p[1])); out = []; best = -1.0
    for e, a in pts:
        if a > best + 1e-12:
            out.append((e, a)); best = a
    return out

def eaa_auc(points, e_lo, e_hi, never_acc, grid=200):
    # FAIR EAA (2026-07-22): budgets below the policy's cheapest point fall back to the
    # true Never-Think accuracy; normalize over the fixed [e_lo,e_hi]. (Old version
    # nan-masked and renormalized per-policy -- unfair to saturated-score policies.)
    pts = sorted(points, key=lambda p: p[0])
    es = np.linspace(e_lo, e_hi, grid); accs = []
    for e in es:
        feas = [a for (pe, a) in pts if pe <= e + 1e-9]
        accs.append(max(feas) if feas else never_acc)
    accs = np.array(accs)
    tr = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tr(accs, es) / (e_hi - e_lo)

def rq3_stats(idx):
    """(router_auc, oracle_auc, pct_gain) on the resample idx."""
    c_t, c_n, e_t, e_n = ct[idx], cn[idx], et[idx], en[idx]
    ch = CHOOSE_R[:, idx]                                       # (51, n)
    r_acc = np.where(ch, c_t, c_n).mean(axis=1)
    r_e = np.where(ch, e_t, e_n).mean(axis=1)
    router_pts = list(zip(r_e, r_acc))
    dA, dE = c_t - c_n, e_t - e_n
    pos = (dA > 0) & (dE > 0)
    lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
    if len(lams) > 1:
        lams.append(max(lams) * 1.5)
    pts = []
    for lam in lams:
        chl = (dA - lam * dE) > 1e-12
        pts.append((np.where(chl, e_t, e_n).mean(), np.where(chl, c_t, c_n).mean()))
    pts += [(e_n.mean(), c_n.mean()), (e_t.mean(), c_t.mean())]
    front = pareto_ul(pts)
    e_lo, e_hi = e_n.mean(), e_t.mean()
    never_acc = c_n.mean()
    r_auc = eaa_auc(router_pts, e_lo, e_hi, never_acc)
    o_auc = eaa_auc(front, e_lo, e_hi, never_acc)
    never = c_n.mean()
    pct = 100 * (r_auc - never) / (o_auc - never + 1e-12)
    return r_auc, o_auc, pct

print("\nRQ3 — router vs oracle frontier (conditional on fitted classifier):")
pt = rq3_stats(np.arange(N))
bs3 = np.array([rq3_stats(IDX[b]) for b in range(B)])
report("router EAA-AUC", pt[0], bs3[:, 0])
report("oracle frontier EAA-AUC", pt[1], bs3[:, 1])
report("% of oracle gain captured", pt[2], bs3[:, 2], "{:.1f}")

# ---------------------------------------------------------------- RQ4 load-adaptive router
conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for r in conc:
    jt[r["batch"]].append(r["energy_per_out_token_j"])
J_OF_L = {L: st.mean(v) for L, v in jt.items()}   # J/token measured curve (held fixed)

TAUS_L = np.linspace(0, 1, 201)                    # analyze_load_router's sweep
CHOOSE_L = p_think[None, :] >= TAUS_L[:, None]     # (201, N)
LAM, L_IDLE, L_BUSY, BUDGET = 1.0, 1, 64, 100.0

def rq4_stats(idx):
    """(adaptive-static acc gap at batch 64, budget gap at 100 mWh/q batch 64)."""
    c_t, c_n, t_t, t_n = ct[idx], cn[idx], tok_th[idx], tok_nt[idx]
    ch = CHOOSE_L[:, idx]
    acc = np.where(ch, c_t, c_n).mean(axis=1)                        # (201,)
    mean_tok = np.where(ch, t_t, t_n).mean(axis=1)
    e_idle = mean_tok * J_OF_L[L_IDLE] * WH * 1000                   # mWh/q
    e_busy = mean_tok * J_OF_L[L_BUSY] * WH * 1000
    i_static = int(np.argmax(acc - LAM * e_idle / 1000))             # frozen at idle
    i_adapt = int(np.argmax(acc - LAM * e_busy / 1000))              # re-solved at load
    gap = (acc[i_adapt] - acc[i_static]) * 100
    feas = acc[e_busy <= BUDGET + 1e-9]
    a_ad = feas.max() if feas.size else np.nan
    a_st = acc[i_static] if e_busy[i_static] <= BUDGET + 1e-9 else np.nan
    bgap = (a_ad - a_st) * 100 if (a_ad == a_ad and a_st == a_st) else np.nan
    return gap, bgap

print("\nRQ4 — load-adaptive tau(L) vs static tau (lambda=1.0/Wh, batch 64):")
pt4 = rq4_stats(np.arange(N))
bs4 = np.array([rq4_stats(IDX[b]) for b in range(B)])
report("acc gain of tau(load) (pp)", pt4[0], bs4[:, 0], "{:+.1f}")
bg = bs4[:, 1][~np.isnan(bs4[:, 1])]
if pt4[1] == pt4[1] and bg.size > B * 0.9:
    report(f"budget-{BUDGET:.0f}mWh acc gap (pp)", pt4[1], bg, "{:+.1f}")
else:
    print(f"  budget gap: point {pt4[1]:+.1f}pp but static infeasible in "
          f"{B - bg.size}/{B} resamples — reporting adaptive-only CI")
    feas_only = bs4[:, 1]  # NaN-aware handled above

# ---------------------------------------------------------------- write report
with open(f"{OUT}/bootstrap_cis.md", "w", encoding="utf-8") as f:
    f.write("# Bootstrap 95% CIs for headline numbers\n\n"
            f"Percentile bootstrap over queries, B={B}, N={N} "
            f"(easy {easy.sum()} / hard {hard.sum()}), seed 12345.\n\n"
            "Router/oracle CIs are **conditional on the fitted classifier**: P(think) is "
            "the full-data 5-fold out-of-fold prediction (identical to notebook 03), held "
            "fixed while queries are resampled — evaluation uncertainty, not retraining "
            "variance. The measured J/token load curve is likewise held fixed for RQ4.\n\n"
            "| headline | point | 2.5% | 97.5% |\n|---|---|---|---|\n")
    for name, p, lo, hi in rows:
        f.write(f"| {name} | {p} | {lo} | {hi} |\n")
print(f"\nwrote {OUT}/bootstrap_cis.md")
