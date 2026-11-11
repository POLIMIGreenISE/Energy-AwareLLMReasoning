"""E8 addendum -- why ranking by score per token beats ranking by score, shown on the corpus itself.

The decomposition in results/e8/CONSOLIDATED.md section 3 finds that per-query token knowledge is what
earns a static threshold its accuracy outside bursty traffic. That is a simulator result. This file
isolates the mechanism without the simulator: rank the paired corpus two ways, give each ranking the
same token budget, and compare what each buys.

Reported:
  1. the spread of reasoning's token cost across the corpus (why dividing by it matters at all);
  2. rank correlations between the router score, the token cost and the accuracy change
     (tie-corrected: the accuracy change takes only five values, most of them zero);
  3. the matched-budget comparison of ranking by score against ranking by score per token.

Correctness conventions: the repair rate uses the strict convention
(correct means a judge score of 1), the accuracy gain uses the graded judge scores.

Free -- no GPU, seconds. Out: results/e8/KNAPSACK.md
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Workload

OUT = "results/e8"; os.makedirs(OUT, exist_ok=True)
BUDGET_FRAC = 0.20          # share of the tokens that reasoning on everything would spend
N_RANDOM = 2000             # random-ranking draws for the floor row

wl = Workload()
S = wl.scores["ModernBERT"]
q = list(wl.qids)
s = np.array([S[x] for x in q])
dt = np.array([max(wl.dtok(x), 1.0) for x in q])            # >=1 so the ratio is defined
ct = np.array([wl.ct[x] for x in q]); cn = np.array([wl.cn[x] for x in q])
dacc = ct - cn
repair = (ct >= 1.0) & (cn < 1.0)                           # strict convention: correct means score 1

# 1. spread of the token cost
p10, p25, p50, p75, p90 = np.percentile(dt, [10, 25, 50, 75, 90])

# 2. rank correlations
rho_sd = spearmanr(s, dt).statistic
rho_sa = spearmanr(s, dacc).statistic
rho_da = spearmanr(dt, dacc).statistic

# 3. matched token budget
budget = BUDGET_FRAC * dt.sum()


def take(order):
    k = int(np.searchsorted(np.cumsum(dt[order]), budget))
    sel = order[:k]
    return dict(n=k, mean_tok=dt[sel].mean(), spent=dt[sel].sum(),
                repair=repair[sel].mean(), gain=dacc[sel].sum() / len(q))


rows = {"score alone": take(np.argsort(-s, kind="stable")),
        "score per token": take(np.argsort(-(s / dt), kind="stable"))}
rng = np.random.default_rng(0)
draws = [take(rng.permutation(len(q))) for _ in range(N_RANDOM)]
rows["uniformly at random"] = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}

L = ["# E8 addendum: score per token against score alone, on the paired corpus", "",
     f"Paired corpus, N = {len(q)}. Budget = {BUDGET_FRAC:.0%} of the {dt.sum():,.0f} extra tokens that "
     f"reasoning on every query would spend ({budget:,.0f} tokens). No simulator is involved.", "",
     "## 1. How unequal the cost of a reasoning decision is", "",
     "|percentile|10th|25th|50th|75th|90th|90th / 10th|", "|--|--|--|--|--|--|--|",
     f"|extra tokens|{p10:.0f}|{p25:.0f}|{p50:.0f}|{p75:.0f}|{p90:.0f}|{p90 / p10:.1f}x|", "",
     "## 2. Rank correlations (Spearman, tie-corrected)", "",
     "|pair|rho|", "|--|--|",
     f"|router score vs token cost|{rho_sd:+.3f}|",
     f"|router score vs accuracy change|{rho_sa:+.3f}|",
     f"|token cost vs accuracy change|{rho_da:+.3f}|", "",
     "## 3. The same token budget, ranked two ways", "",
     "|ranking|queries|mean extra tokens|tokens spent|reasoning repairs (strict)|accuracy gain (pp)|",
     "|--|--|--|--|--|--|"]
for name, r in rows.items():
    L.append(f"|{name}|{r['n']:.0f}|{r['mean_tok']:.0f}|{r['spent']:,.0f}|{100 * r['repair']:.1f}%|"
             f"{100 * r['gain']:+.2f}|")
sa, sp = rows["score alone"], rows["score per token"]
L += ["", "## Reading", "",
      f"Ranking by score alone spends the budget on {sa['n']:.0f} long generations, repairing "
      f"{100 * sa['repair']:.1f}% of them. Dividing by the token cost lowers the hit rate to "
      f"{100 * sp['repair']:.1f}% but buys {sp['n']:.0f} decisions at {sp['mean_tok'] / sa['mean_tok']:.0%} "
      f"of the cost apiece, and collects {100 * (sp['gain'] - sa['gain']):.2f} more percentage points. "
      "Score, cost and benefit are positively correlated on this corpus, so the queries a score is most "
      "confident about are also the dearest to reason on.", "",
      "*Repro: `python scripts/e8_knapsack.py` (CPU, seconds).*", ""]
open(os.path.join(OUT, "KNAPSACK.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
