# RQ4-lite: load-dependent energy-aware labels

- N=1710 paired queries; j(L) measured on A100 (concurrency sweep); lambda*=0.001/Wh calibrated at batch=1 (agreement with existing oracle 0.984).
- token policy calibrated identically at batch=1 (mu*=1.00e-06/token), load-blind by construction.

|load|tail|think share|label flips vs idle|E-policy acc @ mWh|token policy acc @ mWh|acc gap|
|--|--|--|--|--|--|--|
|1||13.5%|0|0.941 @ 1157.1|0.941 @ 1157.1|+0.0pp|
|2||13.5%|0|0.941 @ 575.6|0.941 @ 575.6|+0.0pp|
|4||13.5%|0|0.941 @ 292.4|0.941 @ 292.4|+0.0pp|
|8||13.5%|0|0.941 @ 153.4|0.941 @ 153.4|+0.0pp|
|16||13.5%|0|0.941 @ 82.4|0.941 @ 82.4|+0.0pp|
|32||13.5%|0|0.941 @ 45.3|0.941 @ 45.3|+0.0pp|
|64||13.5%|0|0.941 @ 27.8|0.941 @ 27.8|+0.0pp|
|64|yes|13.5%|0|0.941 @ 165.2|0.941 @ 165.2|+0.0pp|

## Lambda sweep
Max label-flip fraction between idle and busy: **13.1%** of queries at λ=1.40/Wh (idle 0% think vs busy 13% think). At the accuracy-dominant calibrated λ*=0.001/Wh flips are minimal (0/1710) — note dAcc is quantized at 0.5 with REPS=2, which widens the dead zone; more reps would sharpen this.

## Budget framing (accuracy achievable under fixed mWh/query)

|budget (mWh/q)|batch 1|batch 8|batch 64|
|--|--|--|--|
|50|--|0.864|0.941|
|100|--|0.916|0.941|
|200|--|0.941|0.941|
|500|0.887|0.941|0.941|
|1000|0.935|0.941|0.941|
|2000|0.941|0.941|0.941|
|4000|0.941|0.941|0.941|
