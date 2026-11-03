# Draft-agreement router (step 3)

N=248 queries with 2 no_think drafts. Agreement by group: easy 0.18, hard-solvable 0.11, hard-unsolvable 0.38

|policy|label AUC|EAA-AUC|
|--|--|--|
|A text-only|0.829|0.7986|
|B text+drafts|0.838|0.7851|
|C drafts-only|0.685|0.7638|
|oracle frontier||0.8557|

DART rule (think iff drafts disagree): acc 0.855 @ 4334 mWh honest (3536 naive). Draft policies pay 2*E_no_think always.
