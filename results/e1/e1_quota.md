# E1 — allocating think/no-think under an energy quota

> **⚠ PARTIALLY SUPERSEDED (2026-08-05).** The fixed-threshold baseline here was swept over a
> coarse grid that reached only 26 of the router's 684 achievable operating points, which
> inflated the load-aware advantage. The methodology and the qualitative direction stand; the
> magnitudes do not. **Authoritative numbers: `results/audit/fair_headtohead.md`.** Full detail:
> `results/CORRECTIONS.md` §1.

Frontiers built on 8 held-out traces; offline tuning uses 3 separate historical traces. Energy = measured GPU power integrated over time; load emerges from arrivals+service. Prefill time omitted (decode-dominated).

## E1a — accuracy at MATCHED energy (fair comparison)

|regime|cap|policy|accuracy|vs random@rate|% of oracle headroom|
|--|--|--|--|--|--|
|bimodal|25%|random@rate|0.8416|+0.00pp|18%|
|bimodal|25%|static-tau|0.8308|-1.07pp|2%|
|bimodal|25%|load-aware|0.8691|+2.75pp|60%|
|bimodal|25%|hindsight oracle|0.8957|+5.42pp|100%|
|bimodal|25%|load-aware+feedback|0.8544|+1.28pp|37%|
|bimodal|40%|random@rate|0.8513|+0.00pp|25%|
|bimodal|40%|static-tau|0.8529|+0.17pp|27%|
|bimodal|40%|load-aware|0.8972|+4.59pp|80%|
|bimodal|40%|hindsight oracle|0.9145|+6.32pp|100%|
|bimodal|40%|load-aware+feedback|0.8684|+1.72pp|46%|
|bimodal|55%|random@rate|0.8602|+0.00pp|31%|
|bimodal|55%|static-tau|0.8783|+1.81pp|49%|
|bimodal|55%|load-aware|0.8972|+3.70pp|68%|
|bimodal|55%|hindsight oracle|0.9283|+6.81pp|100%|
|bimodal|55%|load-aware+feedback|0.8827|+2.25pp|54%|
|bimodal|70%|random@rate|0.8721|+0.00pp|39%|
|bimodal|70%|static-tau|0.9133|+4.13pp|77%|
|bimodal|70%|load-aware|0.9204|+4.83pp|83%|
|bimodal|70%|hindsight oracle|0.9384|+6.64pp|100%|
|bimodal|70%|load-aware+feedback|0.8976|+2.55pp|62%|
|stable|25%|random@rate|0.8321|+0.00pp|10%|
|stable|25%|static-tau|0.8308|-0.12pp|5%|
|stable|25%|load-aware|0.8312|-0.08pp|7%|
|stable|25%|hindsight oracle|0.8521|+2.00pp|100%|
|stable|25%|load-aware+feedback|0.8414|+0.93pp|52%|
|stable|40%|random@rate|0.8323|+0.00pp|6%|
|stable|40%|static-tau|0.8308|-0.15pp|2%|
|stable|40%|load-aware|0.8312|-0.10pp|3%|
|stable|40%|hindsight oracle|0.8720|+3.97pp|100%|
|stable|40%|load-aware+feedback|0.8471|+1.48pp|41%|
|stable|55%|random@rate|0.8381|+0.00pp|14%|
|stable|55%|static-tau|0.8308|-0.73pp|2%|
|stable|55%|load-aware|0.8581|+2.00pp|48%|
|stable|55%|hindsight oracle|0.8885|+5.04pp|100%|
|stable|55%|load-aware+feedback|0.8556|+1.75pp|44%|
|stable|70%|random@rate|0.8472|+0.00pp|19%|
|stable|70%|static-tau|0.8529|+0.57pp|26%|
|stable|70%|load-aware|0.8581|+1.09pp|31%|
|stable|70%|hindsight oracle|0.9204|+7.32pp|100%|
|stable|70%|load-aware+feedback|0.8642|+1.70pp|38%|

## E1b — cap adherence (tuned offline, deployed on held-out traces)

|regime|cap|policy|accuracy|energy (Wh)|vs cap|breach rate|think-rate|p95 lat (s)|
|--|--|--|--|--|--|--|--|--|
|bimodal|25%|random@rate|0.8368|172.1|-14.8%|0%|9%|20.9|
|bimodal|25%|static-tau|0.8308|104.4|-48.3%|0%|0%|14.3|
|bimodal|25%|load-aware|0.8691|191.5|-5.2%|12%|9%|57.3|
|bimodal|25%|hindsight oracle|0.8885|177.6|-12.1%|12%|8%|40.1|
|bimodal|25%|load-aware+feedback|0.8552|196.5|-2.7%|25%|15%|46.6|
|bimodal|40%|random@rate|0.8472|233.2|-10.9%|12%|19%|28.9|
|bimodal|40%|static-tau|0.8529|209.9|-19.8%|0%|4%|21.1|
|bimodal|40%|load-aware|0.8691|191.5|-26.8%|0%|9%|57.3|
|bimodal|40%|hindsight oracle|0.9058|222.8|-14.9%|0%|9%|57.7|
|bimodal|40%|load-aware+feedback|0.8772|276.6|+5.7%|75%|27%|109.7|
|bimodal|55%|random@rate|0.8513|255.1|-20.6%|0%|22%|39.1|
|bimodal|55%|static-tau|0.8783|288.3|-10.3%|25%|8%|85.9|
|bimodal|55%|load-aware|0.8972|259.9|-19.2%|0%|16%|102.4|
|bimodal|55%|hindsight oracle|0.9145|250.1|-22.2%|0%|10%|68.8|
|bimodal|55%|load-aware+feedback|0.8855|317.1|-1.3%|50%|38%|137.6|
|bimodal|70%|random@rate|0.8721|353.9|-7.1%|12%|42%|98.8|
|bimodal|70%|static-tau|0.8783|288.3|-24.4%|0%|8%|85.9|
|bimodal|70%|load-aware|0.8972|259.9|-31.8%|0%|16%|102.4|
|bimodal|70%|hindsight oracle|0.9384|334.2|-12.3%|25%|14%|128.2|
|bimodal|70%|load-aware+feedback|0.9030|368.7|-3.3%|25%|48%|162.0|
|stable|25%|random@rate|0.8321|41.1|-2.9%|25%|2%|16.1|
|stable|25%|static-tau|0.8308|35.1|-17.0%|0%|0%|14.8|
|stable|25%|load-aware|0.8312|35.5|-16.1%|0%|1%|15.1|
|stable|25%|hindsight oracle|0.8615|43.0|+1.6%|62%|4%|20.0|
|stable|25%|load-aware+feedback|0.8421|41.9|-0.9%|25%|3%|19.9|
|stable|40%|random@rate|0.8321|41.1|-12.3%|12%|2%|16.1|
|stable|40%|static-tau|0.8308|35.1|-25.1%|0%|0%|14.8|
|stable|40%|load-aware|0.8312|35.5|-24.2%|0%|1%|15.1|
|stable|40%|hindsight oracle|0.8720|45.1|-3.7%|50%|6%|23.3|
|stable|40%|load-aware+feedback|0.8562|50.8|+8.4%|75%|8%|48.7|
|stable|55%|random@rate|0.8323|44.9|-12.7%|12%|4%|18.2|
|stable|55%|static-tau|0.8308|35.1|-31.7%|0%|0%|14.8|
|stable|55%|load-aware|0.8581|48.6|-5.4%|12%|7%|39.2|
|stable|55%|hindsight oracle|0.8821|48.5|-5.7%|12%|7%|31.9|
|stable|55%|load-aware+feedback|0.8713|56.1|+9.2%|88%|16%|99.8|
|stable|70%|random@rate|0.8344|47.3|-15.5%|12%|7%|20.1|
|stable|70%|static-tau|0.8308|35.1|-37.2%|0%|0%|14.8|
|stable|70%|load-aware|0.8581|48.6|-13.1%|0%|7%|39.2|
|stable|70%|hindsight oracle|0.9058|53.4|-4.6%|12%|9%|61.0|
|stable|70%|load-aware+feedback|0.8884|59.4|+6.0%|88%|27%|130.0|
