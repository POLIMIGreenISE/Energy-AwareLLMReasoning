# System-aware router (finale): p_need + tau(load)

N=1710 (p_need router), N=1710 (DART re-verdict). Measured J/token curve: {1: 3.422, 2: 1.702, 4: 0.865, 8: 0.454, 16: 0.244, 32: 0.134, 64: 0.082}

## Part 1 — system-aware p_need router (probe cost FIXED, primary accounting)

### lambda=0.04/Wh (accuracy-first) (static tau*=0.025)

|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|
|--|--|--|--|--|--|--|--|
|1|0.025|26.7%|0.925|0.925|+0.0pp|1979.2|1979.2|
|2|0.005|27.8%|0.925|0.925|+0.0pp|990.7|985.4|
|4|0.002|29.6%|0.925|0.925|+0.1pp|509.0|501.3|
|8|0.002|29.6%|0.925|0.925|+0.1pp|267.7|263.7|
|16|0.000|100.0%|0.928|0.925|+0.4pp|184.3|142.3|
|32|0.000|100.0%|0.928|0.925|+0.4pp|102.0|78.9|
|64|0.000|100.0%|0.928|0.925|+0.4pp|63.2|49.0|

### lambda=1.0/Wh (energy-conscious) (static tau*=1.000)

|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|
|--|--|--|--|--|--|--|--|
|1|1.000|13.7%|0.878|0.878|+0.0pp|1401.4|1401.4|
|2|1.000|13.7%|0.878|0.878|+0.0pp|697.9|697.9|
|4|1.000|13.7%|0.878|0.878|+0.0pp|355.2|355.2|
|8|1.000|13.7%|0.878|0.878|+0.0pp|187.0|187.0|
|16|0.050|26.1%|0.924|0.878|+4.6pp|141.9|101.1|
|32|0.025|26.7%|0.925|0.878|+4.6pp|78.9|56.3|
|64|0.005|27.8%|0.925|0.878|+4.7pp|49.3|35.1|

**Sensitivity** (probe cost amortized with load instead of fixed, lambda=1.0/Wh): batch64 acc 0.925 vs fixed-probe 0.925 — conclusion unchanged (probe is negligible next to generation energy at every load).

Reference — learned-router tau(load) result at batch64 (results/load_labels/router_summary.md): `|64|0.70|16%|0.918|0.834|+8.4pp|35.9|7.6|`

## Part 2 — DART re-verdict under load

DART rule (think unless all 3 constrained drafts agree): accuracy 0.891 (load-invariant — same queries think regardless of batch).

**Constrained drafts (12-tok, ~24 mWh probe)** — already fixed to be cheap in step 3b; not a fair re-test of the reversal, included for completeness:

|load|DART honest, fixed probe (mWh)|DART honest, amortized probe (mWh)|Always-Think (mWh)|verdict (fixed)|verdict (amortized)|
|--|--|--|--|--|--|
|1|1596.7|1594.9|2567.5|DART CHEAPER|DART CHEAPER|
|2|806.2|793.4|1277.3|DART CHEAPER|DART CHEAPER|
|4|421.2|403.0|648.8|DART CHEAPER|DART CHEAPER|
|8|232.2|211.4|340.4|DART CHEAPER|DART CHEAPER|
|16|135.7|113.6|182.8|DART CHEAPER|DART CHEAPER|
|32|85.2|62.4|100.5|DART CHEAPER|DART CHEAPER|
|64|61.5|38.3|61.7|DART CHEAPER|DART CHEAPER|

### Part 2b — FREE-FORM drafts (858 mWh probe): the rule that actually reversed

N=1710, think-rate 93.8%. Both the probe (2 full no_think generations) and the thinking branch are GENERATED TOKENS through the same batched engine, so both scale by the identical measured J/token factor — under this physically correct accounting, the honest-cost/Always-Think ratio is a **constant, provably load-invariant**: 1.235 (DART costs more than Always-Think at every load).

|load|DART honest (mWh)|Always-Think (mWh)|ratio|
|--|--|--|--|
|1|3171.7|2567.5|1.235|
|2|1577.8|1277.3|1.235|
|4|801.5|648.8|1.235|
|8|420.5|340.4|1.235|
|16|225.8|182.8|1.235|
|32|124.1|100.5|1.235|
|64|76.2|61.7|1.235|

**Reading:** load does NOT rescue a probe that is overpriced relative to what it saves — if the probe's cost structure matches the thing it's trying to save (both are generation), the verdict is a fixed ratio. Load only changes a verdict when the two cost components scale DIFFERENTLY — which is exactly the p_need case above (Part 1): a single-forward-pass probe (compute-bound, doesn't shrink much with batching) versus a decode chain (memory-bandwidth-bound, shrinks ~43x) is the load-DEPENDENT regime; two generation-vs-generation costs (DART) are not.
