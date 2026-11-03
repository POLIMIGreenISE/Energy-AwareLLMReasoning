# The price the rule gets to choose between

J(L) as charged to each arriving query under a policy that never reasons, over 20 traces x 500 queries. **Cheap side** is median / p10: how much cheaper the cheap moments are than a typical one, which is the only direction a rule pricing at the live load can exploit. **Dear side** is p90 / median. Mean gap is the deployable rule minus a static threshold, averaged over the four energy caps (`e8_rows.json`).

|regime|p10 (J)|median (J)|p90 (J)|cheap side|dear side|mean gap (pp)|
|--|--|--|--|--|--|--|
|bursty|0.244|3.422|3.422|**14.04x**|1.00x|+1.17|
|near-idle|1.702|3.422|3.422|**2.01x**|1.00x|+1.17|
|moderate|0.659|0.865|3.422|**1.31x**|3.96x|-0.02|
|saturated|0.209|0.270|0.401|**1.29x**|1.49x|-0.53|

Ranked by the cheap side, the regimes rank as they do by mean gap, which load CV no longer does: near-idle and moderate differ by \~0.03 in load CV and by a factor of 1.5 in how much cheaper their cheap moments are.

## The production arrival processes

Here the mean arrival rate is held fixed at 0.8 req/s, so mean concurrency cannot explain the outcome: code completion serves 11.6 requests at a time against conversation's 4.5 and is the one that wins. 12 traces x 500 queries.

|arrival process|p10 (J)|median (J)|p90 (J)|cheap side|
|--|--|--|--|--|
|Azure code @0.8|0.127|0.322|0.762|**2.53x**|
|Azure code @0.8 (thinned)|0.161|0.349|1.283|**2.16x**|
|Azure conv @0.8|0.454|0.865|1.702|**1.91x**|
|Poisson @0.8 (control)|0.659|0.865|3.422|**1.31x**|

*Repro: `python scripts/e8_price_spread.py` (CPU, 1 s).*
