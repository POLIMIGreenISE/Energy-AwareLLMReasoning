# Experiment A -- pooled in-distribution router (OTB + Mohseni + GSM8K)

5-fold stratified CV, OOF predictions. EAA on the natural distribution; energy basis identical across datasets (tok x 3.7 J). p_need is fixed (non-trainable).

## Overall (pooled)
```
N=4839 think=14% | Never 0.782@456 Always 0.897@3234 | Oracle EAA 0.9170 | Random 0.8400 (43%)
  pooled-router    AUC 0.792 | EAA 0.8556 (55% of gain, +12pp vs random)
  OTB-only-router  AUC 0.650 | EAA 0.8427 (45% of gain, +2pp vs random)
  p_need           AUC 0.598 | EAA 0.8125 (23% of gain, -20pp vs random)
```

## Per-dataset slice of the SAME pooled-CV predictions (in-distribution)

### otb
```
N=1710 think=10% | Never 0.834@341 Always 0.928@2776 | Oracle EAA 0.9290 | Random 0.8814 (50%)
  pooled-router    AUC 0.900 | EAA 0.8982 (68% of gain, +18pp vs random)
  OTB-only-router  AUC 0.911 | EAA 0.9004 (70% of gain, +20pp vs random)
  p_need           AUC 0.842 | EAA 0.8737 (42% of gain, -8pp vs random)
```

### mohseni
```
N=1810 think=27% | Never 0.611@701 Always 0.823@4237 | Oracle EAA 0.8561 | Random 0.7164 (43%)
  pooled-router    AUC 0.559 | EAA 0.7203 (45% of gain, +2pp vs random)
  OTB-only-router  AUC 0.528 | EAA 0.7106 (41% of gain, -2pp vs random)
  p_need           AUC 0.564 | EAA 0.6572 (19% of gain, -24pp vs random)
```

### gsm8k
```
N=1319 think=3% | Never 0.948@270 Always 0.960@2452 | Oracle EAA 0.9746 | Random 0.9545 (25%)
  pooled-router    AUC 0.500 | EAA 0.9524 (18% of gain, -8pp vs random)
  OTB-only-router  AUC 0.464 | EAA 0.9528 (19% of gain, -6pp vs random)
  p_need           AUC 0.602 | EAA 0.9515 (14% of gain, -11pp vs random)
```

## Balanced 50/50 decision metric (think=no_think count; 20 subsamples)
```
balanced N=1380 (think 690 + equal no_think)
  pooled-router    AUC 0.792  bal-acc@0.5 0.556
  OTB-only-router  AUC 0.651  bal-acc@0.5 0.546
  p_need           AUC 0.598  bal-acc@0.5 0.569
```

## Does adding diverse data to training help (in-distribution)?
pooled-router minus OTB-only-router, same held-out folds:

|slice|ΔEAA|Δ(pp vs random)|ΔAUC|
|--|--|--|--|
|otb|-0.0022|-2pp|-0.011|
|mohseni|+0.0096|+4pp|+0.031|
|gsm8k|-0.0004|-2pp|+0.036|
|**overall**|+0.0128|+9pp|+0.142|
## What this says

1. **The pooled router handles the realistic mixture; the OTB-only router does not.**
   On the full pooled set (N=4839, 14% think) the pooled router hits **AUC 0.792, +12pp vs the
   random floor**, vs OTB-only **AUC 0.650, +2pp**, and p_need **−20pp**. Caveat: this overall gap
   is partly definitional — the OTB-only router is being tested on two datasets (Mohseni, GSM8K) it
   never saw, which drags its overall number down. The per-dataset rows below are the honest read.

2. **Pooling costs almost nothing on the home dataset.** On OTB the pooled router is AUC 0.900 vs
   OTB-only 0.911 (EAA −0.0022, −2pp vs random) — a tiny specialization cost. You can add diverse
   data without wrecking your best benchmark. This is the reassuring result.

3. **Adding a dataset to training helps that dataset (in-distribution).** Mohseni AUC 0.528→0.559
   (+0.031), crossing from below random to +2pp above; GSM8K AUC 0.464→0.500. These are
   in-distribution gains (the pooled router saw those rows) — they show "train on X → do better on
   X," which is expected and NOT yet a transfer claim. Whether diversity helps an *unseen* dataset
   is Experiment B.

4. **p_need stays worse-than-random on the energy axis everywhere** (−8pp OTB, −24pp Mohseni,
   −11pp GSM8K, −20pp overall) — the trained-router-over-probe finding now holds on the largest,
   most diverse set we have (N=4839).

5. **GSM8K has no routing signal to capture** (thinking buys +1.2pp; pooled AUC 0.500). Pooling
   can't manufacture signal where thinking barely helps — a natural-limit observation, not a
   failure.

**Bottom line for A:** if the deployment distribution is this kind of mixture, train the router on
the mixture — it's clearly the best option (AUC 0.792, +12pp vs random) and barely dents OTB. What
A does **not** show is better *transfer* to a genuinely new distribution; the per-dataset gains are
in-distribution. That is exactly what Experiment B (train on the pool, test on a fresh held-out
open-ended dataset) will decide.
