# E2b — the constant a token-based cost model needs is a property of the TRAFFIC

> **SUPERSEDED (2026-09-15).** Not reproducible: the script was never committed and the "steady" /
> "heavy" regimes were not recorded. The budget below is mislabelled; the code defines it as
> never + 35% of (always − never). Use `results/e2/TRANSFER_TOKENS.md` (`scripts/e2_transfer_tokens.py`),
> which the thesis now cites. See `results/CORRECTIONS.md` §8a.

*(2026-08-05. This is the strongest form of the E2 result and supersedes the earlier
"−45% to +117%" framing, which invited the rebuttal "so just calibrate it properly".)*

## Setup

An operator caps a workload's energy. The controller must decide per query whether letting it think
still fits, which requires estimating what thinking costs:

- **token proxy (the field)** : `joules = C × extra_tokens`, C a constant chosen once
- **load-aware (ours)**       : `joules = J(L) × extra_tokens`, L = live `num_requests_running`

λ is bisected on a *historical* trace so the controller's own model predicts it will hit the budget;
it is then deployed on held-out traces and the **actual** energy is measured. Budget = 35% of
always-think. 8 eval traces, 500 queries each.

## Result — sweeping the constant across three traffic patterns

Actual energy vs the authorised budget:

|C (J/token)|3.42|2.40|**1.70**|1.20|0.865|**0.60**|0.454|0.244|**0.13**|load-aware|
|--|--|--|--|--|--|--|--|--|--|--|
|**bursty** (swings idle↔busy)|−46|−28|**−9**|+33|+76|+108|+111|+111|+111|**−8.3%**|
|**steady** (consistently loaded)|−25|−25|−25|−25|−25|**+1**|+13|+32|+43|**+2.7%**|
|**heavy** (saturated)|−62|−62|−62|−62|−62|−62|−57|−31|**+9**|**−4.4%**|

**The constant that hits the budget moves 13×: 1.70 → 0.60 → 0.13 J/token.** It is a property of the
*traffic*, not of the hardware. Using the bursty-tuned constant (1.70) elsewhere gives **−25% on
steady and −62% on heavy**. Choosing too low saturates the controller — it concludes it can afford to
think on ~96% of queries and **overshoots the cap by 111%**.

**The load-aware model lands within 8% on all three with no calibration**, because it reads the price
at the moment of each decision.

## Why the flat regions appear

Left flat region (large C): the controller believes thinking is so expensive it barely thinks — it
converges on never-think and undershoots. Right flat region (small C): it believes thinking is nearly
free, thinks on ~everything, and lands on the always-think energy. Only a narrow band in between
tracks the budget, and that band sits at a different place for each traffic pattern.

## Robustness note

This experiment does **not** involve the router threshold grid, so it is unaffected by the fairness
correction that revised E1/E3/E4 (see `results/CORRECTIONS.md`). The controller family and the sweep
are identical for both cost models; only the *belief about cost* differs.

*Repro: the sweep logic is in `scripts/e2_token_proxy_controller.py` (open-loop section); the
three-traffic transfer table was produced by the same bisect-then-deploy procedure over the regimes
in `src/serving_sim.py`.*
