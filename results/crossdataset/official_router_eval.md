# Mohseni's officially released routers on the dataset's own test split (N=1,810)

Both released checkpoints run locally (RTX 4060) on the same 1,810 queries we generated Qwen3-8B
think/no-think answers for. Scored three ways: against **the dataset's own labels** (the setting the routers
were trained for, on 70k examples), against **our energy-aware oracle** (does thinking actually flip Qwen3-8B?), and on the
**energy–accuracy plane vs the random-routing floor** (the deployment question).

Setup: `Never 0.611 @ 701 mWh · Always 0.823 @ 4237 mWh · Oracle EAA 0.8561 · Random floor 0.7161`.
(ModernBERT in fp32 — it emits NaN in fp16; Qwen3-0.6B in bf16 @ batch 4 to fit 8 GB.)

## Ranking quality

|router|AUC vs dataset labels|AUC vs OUR oracle|think-rate@0.5|
|--|--|--|--|
|Mohseni reasoning-router-mmbert-small (140M)|0.663|0.605|44%|
|Mohseni reasoning-router-0.6b (600M)|**0.673**|0.604|43%|
|our LogReg (OTB-trained)|—|0.505|—|
|our ModernBERT (OTB-trained)|—|0.521|—|
|our best-of-13 sweep trained *on Mohseni*|0.589|0.568|—|

By source (vs our oracle / vs dataset labels): Big-Math **0.625 / 0.720**, WildChat 0.561 / 0.582,
Nectar 0.517 / 0.627 — best on math, weakest on chat, matching the stated limitation in the routers' README that
*"for tasks labeled think, the training data is heavily skewed towards mathematical reasoning."*

## Energy–accuracy routing — bootstrap CIs (B=200)

|router|EAA|95% CI|EAA − random|95% CI of difference|
|--|--|--|--|--|
|Mohseni mmBERT|0.7090|[0.6912, 0.7284]|−0.0071|**[−0.0189, +0.0044]**|
|Mohseni 0.6B|0.7101|[0.6928, 0.7311]|−0.0060|**[−0.0175, +0.0061]**|
|our LogReg (OTB)|0.7289|[0.7119, 0.7475]|+0.0128|**[−0.0004, +0.0267]**|
|our ModernBERT (OTB)|0.7255|[0.7082, 0.7420]|+0.0094|**[−0.0046, +0.0236]**|
|*random floor*|0.7161|—|—|—|

**Every CI includes zero.** On realistic mixed traffic, no router — the published checkpoints or ours
— is statistically distinguishable from random routing on the energy–accuracy plane.

## What this establishes

1. **Correction to our earlier point estimates.** We previously quoted "+5–6pp vs random" for our
   trained routers on Mohseni. With CIs that advantage is **not significant** ([−0.2, +10.9]pp in
   oracle-gain units). Likewise the published routers' "−2pp" is not significantly below random. The honest
   statement is: **all routers sit at the random floor here.**
2. **Training scale genuinely helps RANKING — our small-data conclusion was too pessimistic.** The published
   checkpoints (70k training examples) rank the dataset labels at 0.663–0.673 vs our 0.563 from ~1.5k, and
   rank *our* oracle at ~0.605 vs our 0.505–0.568. More data buys real ranking signal.
3. **But ranking does not convert into energy–accuracy value.** AUC 0.60 still yields EAA at the
   random floor. Label-AUC weights every error equally; EAA weights each query by its actual ΔAccuracy
   and ΔWh. Ranking the label ≠ routing profitably. (Same pattern as p_need: best AUC, worst EAA.)
4. **Capacity saturates.** 600M (0.6B) ≈ 140M (mmBERT-small): 0.673 vs 0.663 on the dataset labels, 0.604 vs
   0.605 on ours. A 4× bigger router buys ~nothing — independent confirmation, on the published models and
   their 70k training data, of our own capacity sweep.
5. **The contrast with OTB stands.** On OTB the same style of router reaches AUC 0.90 and clears the
   random floor decisively. Benchmarks that align topic with difficulty make routing look solved;
   realistic traffic does not.

*Repro: `scripts/eval_official_mohseni_routers.py` (scores cached in `official_router_scores.npz`).*
