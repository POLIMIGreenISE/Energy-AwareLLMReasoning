# Constrained-draft router (step 3b)

N=248; probe = 3 drafts, mean 25 mWh/query.

Agreement: easy 0.85 | hard-solvable 0.57 | hard-unsolvable 0.47; solvability AUC among hard: 0.434

|policy|label AUC|EAA-AUC (honest)|
|--|--|--|
|A text-only|0.829|0.7986|
|B text+drafts|0.841|0.8031|
|C drafts-only|0.616|0.7807|
|oracle frontier||0.8557|

DART all-3: acc 0.782 @ 2697 mWh; majority: acc 0.782 @ 2697 mWh.
