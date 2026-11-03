# Ceiling decomposition: classification gap vs solvability wall

N=1710 (1327 easy, 383 hard).

|policy|EAA|% of oracle gain|
|--|--|--|
|Never-Think|0.8342|0%|
|Subset-oracle (perfect easy/hard, random within hard)|0.8913|60.3%|
|Learned text router (actual)|0.9002|69%|
|p_need rule (actual)|0.9103|81%|
|True oracle frontier (per-query optimal)|0.9290|100%|

**Decomposition of the total achievable gain (0.0948 over Never-Think):**

- Recoverable by PERFECT easy/hard classification alone (a training/capacity problem): 0.0571 (60.3%)
- Locked behind the solvability wall (which hard queries thinking rescues — no signal we tested predicts this, AUC~0.46): 0.0377 (39.7%)

**Reading:** p_need (81% of gain) is already close to the subset-oracle ceiling (60.3%) — most of its remaining gap to the true oracle is NOT a classification problem a bigger/better-trained model would fix; it is the solvability wall. A fine-tuned encoder could plausibly close some of the gap between the CURRENT learned router (69%) and p_need/the subset-oracle ceiling (since label AUC already jumped 0.829->0.900 purely from more data, no architecture change) — but is unlikely to crack the solvability wall specifically.
