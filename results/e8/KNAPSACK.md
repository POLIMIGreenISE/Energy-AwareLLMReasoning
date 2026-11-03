# E8 addendum: score per token against score alone, on the paired corpus

Paired corpus, N = 1710. Budget = 20% of the 4,052,642 extra tokens that reasoning on every query would spend (810,528 tokens). No simulator is involved.

## 1. How unequal the cost of a reasoning decision is

|percentile|10th|25th|50th|75th|90th|90th / 10th|
|--|--|--|--|--|--|--|
|extra tokens|469|626|895|1786|8028|17.1x|

## 2. Rank correlations (Spearman, tie-corrected)

|pair|rho|
|--|--|
|router score vs token cost|+0.444|
|router score vs accuracy change|+0.427|
|token cost vs accuracy change|+0.365|

## 3. The same token budget, ranked two ways

|ranking|queries|mean extra tokens|tokens spent|reasoning repairs (strict)|accuracy gain (pp)|
|--|--|--|--|--|--|
|score alone|110|7331|806,444|67.3%|+4.09|
|score per token|192|4187|803,840|52.6%|+5.20|
|uniformly at random|342|2370|806,890|11.7%|+1.86|

## Reading

Ranking by score alone spends the budget on 110 long generations, repairing 67.3% of them. Dividing by the token cost lowers the hit rate to 52.6% but buys 192 decisions at 57% of the cost apiece, and collects 1.11 more percentage points. Score, cost and benefit are positively correlated on this corpus, so the queries a score is most confident about are also the dearest to reason on.

*Repro: `python scripts/e8_knapsack.py` (CPU, seconds).*
