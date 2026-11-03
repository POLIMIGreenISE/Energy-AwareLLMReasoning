# Corrections log — read this before citing any number

Several results in this repository were corrected after the experiments that produced them. Each
entry says what was wrong, what changed, and **which file is now authoritative**. Anything not listed
here stands as written.

---

## 1. The fairness correction (2026-08-05) — affects E1, E3, E4, E5's C2

**What was wrong.** The fixed-threshold baseline was swept over a grid built from *quantiles of the
router's score values*. Because 81.5% of scores sit below 0.1, `np.unique` collapsed those quantiles
into a single threshold, leaving the baseline with **26 operating points out of 684 the router
actually admits** (683 distinct scores). The reachable think-rates had a large gap, so at tight energy
caps the baseline was stranded at never-think and could not spend its budget — an artefact of the
grid, not a property of the router.

**How it was found.** `scripts/audit_fairness.py`, written to check whether the comparison was fair on
five criteria (same traffic, same questions, best baseline setting, matched energy, load range).
Four passed; the "best baseline setting" check failed.

**What changed.** Re-run with a dense grid (sorted unique scores sampled by index → 174 distinct
think-rates) in `scripts/fair_headtohead.py`:

|claim|before|**after**|
|--|--|--|
|load-aware vs fixed threshold, bimodal|+0.7 … +4.4pp|**+2.18 / +1.53 / +0.48 / −0.08pp** at caps 25/40/55/70%|
|"fixed threshold cannot use its budget (−48%)"|stated|**withdrawn** — with a proper grid it spends 216.6 of 217 Wh|
|"saturated scores → only ~3 usable settings"|stated|**withdrawn** — 683 distinct scores, 684 operating points|
|E3 three-way at 217 Wh|static 0.8302 / load-aware 0.8906 / inverse 0.8488|**static 0.8698 / load-aware 0.8852 / inverse 0.8633**|
|mechanism multiplier (think bought per joule)|14% vs 4% → 3.5×|**12.5% vs 6.5% → 1.9×**|
|latency price at equal energy|13.5 s → 85.1 s|**44.8 s → 79.0 s**|
|under a p95 SLO|"advantage falls to +0.36pp"|**it reverses**: fixed threshold wins by 1.17pp at 20 s, 0.65pp at 41 s; load-aware only wins at ≥82 s|

**Authoritative files:** `results/audit/fairness.md` (the audit) and
`results/audit/fair_headtohead.md` (corrected numbers).

**Superseded — do not cite the headline numbers in:** `results/e1/e1_quota.md`,
`results/e3/e3_tradeoff.md`, `results/e4/e4_sweep.md`, and the C2 column of
`results/e5/sensitivity.md`. Their *methodology* and *qualitative direction* still hold; their
magnitudes were computed with the coarse grid. E4's r = 0.69 correlation with load variability is
directional and unaffected in sign, but its pp magnitudes are inflated for the same reason.

**Unaffected:** everything in E2 (it compares two *cost models* inside one controller family and never
touches the threshold grid), the measured hardware curves, the E6/E7 validation, and the both-fail
label result.

---

## 2. The EAA metric fix (earlier) — reversed a headline

The energy-adjusted-accuracy metric normalised per policy over its own non-NaN range, which flattered
policies whose scores saturate. Fixed by filling budgets below a policy's cheapest operating point
with the true never-think accuracy and normalising every policy over the same fixed [lo, hi].
**Consequence: the earlier claim that the p_need probe beat the trained routers was retracted** —
fine-tuned ModernBERT is the best router (EAA 0.901), p_need the worst (0.874).

---

## 3. Point estimates quoted without confidence intervals

"Trained routers beat the random floor by +5–6pp on Mohseni" was a point estimate. With bootstrap CIs
the difference is **not significant** (+0.0128 EAA, 95% CI [−0.0004, +0.0267]). See
`results/crossdataset/official_router_eval.md`: on realistic mixed traffic **no router — ours or the
published prior art — is statistically distinguishable from random routing** on the energy–accuracy
plane.

---

## 4. Smaller fixes worth knowing

- **Unpaired CIs in E2b.** Both controllers run on identical traces, so the CI belongs on the per-seed
  *paired* difference. The unpaired version was ~10× too wide and hid a real effect.
- **Actuation lag in the budget controller.** A thinking decision's energy arrives long after the
  decision, so spend-only feedback over-thought (76% vs 14% think-rate at the same cap). Fixed by
  counting energy already *committed* by in-flight requests (`ctx["inflight_j"]`).
- **Coarse parameter grids** made the stable regime degenerate in the first E1 run; grids must be
  dense at low think-rates and follow score quantiles.
- **Confounded comparison** in E1 v1: policies landing at different energies were compared on accuracy
  alone, producing a nonsensical "119% of oracle". Fixed by reading accuracy at matched energy.

---

## 5. The E8 consolidation (2026-08-29) — supersedes §1's numbers, and two new fairness faults

`results/e8/CONSOLIDATED.md` replaces the scattered E1/E3/E4 comparisons with one protocol on an
explicit concurrency axis. In building it, two further fairness faults were found — both in the
comparison's *favour* this time, which is why they matter.

### 5a. The dynamic policy was reading information the baseline could not have

**What was wrong.** The load-aware rule is `score(q) > λ·J(L)·Δtokens(q)`, and `Δtokens(q)` was read
from the paired dataset. The static baseline saw only `score(q)`. So the dynamic policy was given a
per-query quantity that (i) the baseline was denied and (ii) **no live router can observe**, since the
length of a reasoning chain is not known before generating it. Every load-aware number in E1–E5, and
in the 5 August supervisor update, carries this advantage.

**What changed.** E8 adds a 2×2 decomposition separating the two ingredients the rule uses — live load
and per-query token counts — and the headline now reports `load-aware-const`, which replaces
Δtokens(q) with the constant E[Δtokens] = 2370 tokens. That is the deployable rule, and it is what
`src/router.py` ships by default.

**Consequence — this is the substantive one.** The ingredient carrying the accuracy gain is the
**token count, not the load**. A load-*blind* policy that ranks by benefit-per-token beats the static
threshold by +1.8 to +5.0pp, while load-awareness alone is worth roughly nothing outside bursty
traffic. The earlier framing — that the win came from *when* thinking happens — holds only in the
bursty regime.

### 5b. The λ grid was starving the dynamic policy (the mirror image of §1)

**What was wrong.** §1 fixed a grid that handicapped the *static* baseline. The load-aware families
had the opposite problem and it went unnoticed: their grids were `np.logspace(-8, 4, 180)`, i.e. 1.16×
steps in λ. Because 81.5% of router scores sit below 0.1, one such step can sweep the effective
threshold across a large slice of the score mass. A dry run showed the family jumping from a 0.3% to a
>5% think-rate between adjacent grid points and being stranded at never-think under a cap it could
have used — which would have handed the win to the static baseline for entirely the wrong reason.

**What changed.** Every family's grid is now the image of its own dense score-space grid under its own
price map (`λ = score / (J(L)·E[Δtok])` for the constant-token rule, `λ = (score/Δtok) / J(L)` for the
per-query one), evaluated across the measured range of L. Verified: at L = 2/6/24 the load-aware
families now resolve 170–244 distinct think-rates against the static baseline's 231.

### 5c. What the consolidated result says

|claim|before (5 Aug artifact)|**after (E8)**|
|--|--|--|
|load-aware vs static, bimodal/bursty|+2.18 / +1.53 / +0.48 / −0.08pp|**+1.88 / +1.53 / +0.68 / +0.17pp** (deployable rule)|
|the same result on *steady* traffic|not reported by concurrency|**mostly negative** — −0.88 to −3.25pp at moderate and saturated load|
|"the win comes from *when* thinking happens"|stated|**bounded** — true in bursty traffic; elsewhere the win comes from per-query token pricing (§5a)|
|what decides whether it pays|load level implied|**load *variability*, not level** — E4's r = 0.69 shown directly on the concurrency axis|

**Authoritative file:** `results/e8/CONSOLIDATED.md`. **Superseded headline numbers:**
`results/audit/fair_headtohead.md` (still correct for what it measured, but it compares the
oracle-token variant), and everything already flagged in §1.

**Unaffected:** the measured hardware curves, E2 and the calibration-transfer result, the E6/E7
validation, and the RQ3 routing conclusions.

### 5d. Cost prediction — and a claim withdrawn within the same day

`results/e10/` follows from §5a: if per-query token cost is what earns the accuracy, can it be
*estimated* rather than read from an oracle? In-domain, yes — Spearman 0.76 on OTB, and 0.69 on
Mohseni, where the accuracy router hits a mutual-information ceiling of AUC 0.568–0.589.

**Withdrawn on the same day it was made:** the interim claim that predicting token cost is "far
easier" than predicting accuracy, stated before transfer had been tested. It is not, zero-shot: an
OTB-trained estimator scores **ρ = +0.006** on Mohseni — it learned OTB's topics exactly as the
accuracy router did. The surviving, narrower claim is about *learnability*: the cost is in the text
and the benefit is not, but the cost model must be trained on the traffic it serves.

**One measurement artefact removed en route.** 39% of Mohseni's no-think generations hit the token
cap (`finish_reason='length'`). A censored length is a cap, not a measurement of the quantity being
predicted, and including those queries changed the transfer numbers materially (ρ +0.097 → +0.006).

---

## 6. A prediction withdrawn, and a hypothesis confirmed (2026-09-05)

### 6a. The closed-loop controller was proposed as the fix. It is not.

**What was claimed.** `results/livetest/LIVE_SUMMARY.md` §4 called `BudgetController` *"the strongest
argument in the project for the closed-loop controller"* and predicted that closing the loop on a
real energy meter would repair offline tuning's budget misses (+66% / +69% / +79%).

**What happened when it was run.** Worse than what it replaced, on an A100 against an NVML counter
that agreed with Zeus to 0.0%:

|regime|offline static|offline load-aware|**closed loop**|
|--|--|--|--|
|bursty|+37.0%|−15.7%|**+36.7%**|
|high|+17.7%|+42.0%|**+134.4%**|

At saturation λ finished pinned at its *upper* clamp and the run still spent 2.3× the authorised
energy. Its own pacing trace shows λ driven five orders of magnitude to the *lower* clamp within 29
of 200 requests, producing an 82% think-rate in the second quartile of arrivals.

**The prediction is withdrawn.** What stands is narrower: *neither offline tuning nor naive
closed-loop feedback reliably holds an energy budget for a reasoning router under load.* Energy-
budgeted reasoning control is an open problem this work bounds rather than solves.

**Authoritative file:** `results/closedloop/CLOSED_LOOP.md`.

**A second, smaller withdrawal.** The proposed explanation — that pacing against completed requests
is biased because cheap requests finish first — was tested in simulation (`results/e15/`) and is
**not confirmed**: the simulator does not reproduce the failure at all, and pacing against a clock,
which would remove that bias entirely, does not help. An asymmetric gain is the most promising of
three candidate repairs but still breaches on 42% of traces at saturation. No fix is established.

### 6b. The saturation hypothesis WAS confirmed

E9 blamed its 43% energy error and 2.4× latency error at saturation on vLLM preempting under
KV-cache pressure — asserted, not measured. Measured now: **0 preemptions** at every low and bursty
load, and at saturation the KV cache at **100% occupancy**, a waiting queue reaching 224, and
preemptions scaling with think-rate (6 at 5.2%, **169** at 12.9%).

This makes the simulator's validity boundary mechanical rather than empirical: it holds while the KV
cache is not saturated, which an operator can check. It also means thinking has a **super-linear**
cost at saturation that no per-token model captures — the marginal cost of one more thinking request
includes the recomputation it forces on others.

### 6c. The energy curve generalises; the policy advantage does not

Measured on an **NVIDIA L4** (Ada, 72 W, quarter the throughput of the A100), identical protocol:
the normalised curve J(L)/J(1) matches the A100 to **4.6% mean error** (correlation 0.9995), and the
spread is *larger* — 48.9× against 41.6×. The L4's power is flat to within 1.00× across all batch
sizes, so the mechanism is cleaner there than on the reference card. **RQ1 is not an A100 result.**

But re-running E8 on the L4's curve, the load-aware advantage weakens or reverses at tight caps
(−0.70 / −0.25 / +1.22pp against the A100's +0.95 / +1.25 / +0.91pp), because a 4× slower card holds
the same traffic at more uniform concurrency — the never→always span compresses from 5.1× to 2.2×.
Consistent with E8 and E11: the advantage tracks load *variability*, and hardware speed is one more
thing that sets it. **Report both cards, not the favourable one.**

**Authoritative file:** `results/batching/CROSS_GPU.md`.

---

## 7. §6a partially reversed: the closed loop CAN be repaired below saturation (2026-09-05)

§6a withdrew the closed-loop proposal and stated *"no fix is established"*. E16
(`results/repair/CONTROLLER_REPAIR.md`) ran three controllers on hardware and split that verdict.

**Established — the repair works below saturation.** With the loosening gain quartered, a 1.1×
per-update step cap, λ confined to `[λ*/10, λ*·100]` and a 15% warmup:

|approach (bursty, same budget and traces)|miss|
|--|--|
|offline static τ|+37.0%|
|offline load-aware λ|−15.7%|
|as-shipped closed loop|+35.9% (reproduces E12's +36.7%)|
|**guarded closed loop**|**+6.6%**|

λ's excursion narrowed from 6.3 decades to 2.2, and the loop held λ inside a factor of three until
request 60 of 200 rather than request 11. **§6a's "no fix is established" is withdrawn for this
regime**; the guarded loop is the best budget-holder of anything tried, online or offline.

**Confirmed — at saturation the problem is not the controller.** 155% → 108% with guarding, still
double the budget. The pacing trace shows why and it is structural: at 58% of completions the
controller had spent 14.3 Wh of an eventual 71.6 Wh, and read `pace 0.87` — comfortably under
budget. **Four fifths of the energy arrived during the last 42% of completions**, in the drain, with
a 100%-full KV cache and 102 preemptions forcing recomputation (§6b's mechanism). No controller
paced on *completed requests* can see that coming, because the energy is not consumed in proportion
to completions at all.

**The operator rule this yields:** guarded closed loop below saturation; at saturation fall back to
a conservative static threshold (E12's was +17.7% there, six times closer than the best closed loop)
— not because it controls better, but because a 4.1% think-rate never enters the regime where
control becomes impossible.

**Also corrected:** the §6a explanation of the *original* failure — completion-order bias — is still
not confirmed, but it no longer matters much. What the repair actually needed was bounding how far
and how fast λ may move, which works regardless of why the pace signal was low.

**Authoritative file:** `results/repair/CONTROLLER_REPAIR.md`.

---

## 8. Thesis review experiments (2026-09-15): one table replaced, one claim corrected, one question closed

### 8a. The calibration-transfer table could not be reproduced, and its budget was mislabelled

**What was wrong.** `results/e2/calibration_transfer.md` (log `results/e2xfer.log`) was produced by a
script that was never committed, on "steady" and "heavy" regimes whose parameters were not recorded.
It also labels its budget "35% of always-think", but the E2 code defines it as
`never + 0.35 x (always - never)`; `e2spec.log` is only consistent with the span definition. Both of
its cost models were given each query's *true* token count, so it said nothing about the deployable
rule.

**What changed.** Rebuilt as `scripts/e2_transfer_tokens.py` -> `results/e2/TRANSFER_TOKENS.md`: E2's
plan-then-deploy procedure in the shared simulator on the four documented E8 regimes (guarded: the
never/always anchors reproduce CONSOLIDATED.md section 0), with true, constant and out-of-fold
estimated token counts. Same conclusion, now reproducible: the price that works moves by about an
order of magnitude with the traffic (best single price misses one regime by 51%), while the load-aware
model misses by at most 4.7% (true tokens), 8.1% (constant, the shipped rule) and 3.8% (estimated).
The thesis table and figure for section 5.2 now come from this file. **Superseded:**
`calibration_transfer.md` (-62% / +111% are replaced by -51% / +109%).

### 8b. "Wherever the two ingredients were separated, the accuracy came from token cost" was false

**What was wrong.** E8d decomposed the rule only at the 25% and 40% caps. `scripts/e8_decomp_caps.py`
-> `results/e8/DECOMP_CAPS.md` runs all four caps (guarded against CONSOLIDATED.md sections 1 and 3).
On a near-idle server at 55% and 70%, live load earns as much as or more than token knowledge
(+1.29 vs +1.18, +1.75 vs +1.20 pp).

**Corrected claim.** Token knowledge improves the static threshold in 15 of 16 regime x cap cells; live
load helps measurably in exactly the cells where the deployable rule wins (bursty at every cap,
near-idle at the loosest) and harms elsewhere. *Counts and cells superseded by section 9 below, which
re-ran this experiment with the grid fix: 7 help, 2 harm.*

### 8c. Bursty production traffic under a latency target (the question section 5.5 left open)

`scripts/e11_slo.py` -> `results/e11/SLO.md` (guarded: unconstrained rows reproduce REAL_TRACE.md).
Under a p95 target of 3x the never-reasoning p95 or tighter, the load-aware rule is ahead in 0 of 32
comparisons (18 behind, 14 indistinguishable). In the four cells where it wins unconstrained, the gain
survives only targets looser than the tail it creates (3.8x to 10.1x). *Re-run with the grid fix of
section 9: the headline is unchanged at 0 of 32, with 13 behind and 19 ties, over five winning cells
whose tails run 3.2x to 10.4x.*

## 9. The matched-energy comparison was scored on grids that could not reach the cap

**What was wrong.** Every matched-energy comparison reads each policy family at its best operating
point subject to `energy <= cap`. Each family is swept over a parameter grid built in score space and
mapped through that family's own price. Dense in score space is not dense in energy, and where a
family's grid had no point just below the cap it was scored at whatever point it did have -- charged
for budget its grid never offered rather than for how it allocates.

Only one side of the comparison was affected. The static threshold's grid *is* the score grid, so it
lands on almost any cap. The load-aware family's grid is that grid divided by a price that varies over
the trace, and it left holes: on the saturated regime at the 40% cap its points jumped from 23.2 Wh to
26.4 Wh across a 26.1 Wh cap, so 11% of the budget was unreachable. Across the 16 regime x cap cells,
every large published loss coincided with a large slice of unreachable budget.

**The fix.** `src/capfit.py` (`refine_to_bound`) bisects between the two operating points that straddle
a boundary and evaluates a few neighbours, so no family is scored on a hole in its own grid. It is
applied to every family, at every cap, and also at the p95 target in the SLO experiments, in
`e8_consolidated.py`, `e8_decomp_caps.py`, `e11_real_trace.py`, `e11_slo.py` and
`e10_dtok_predictability.py`. The E2 budget experiments are unaffected: they bisect lambda directly and
never select from a grid.

**What changed.** The direction of every conclusion survives; two claims do not.

| experiment | published | corrected |
|--|--|--|
| E8a bursty (4 caps) | +1.88 / +1.53 / +0.68 / +0.17 | +1.99 / +1.68 / +0.83 / +0.18, all significant |
| E8a near-idle | +0.22 / -0.34 / +1.28 / +1.75 | +0.30 / +1.16 / +1.44 / +1.77, three significant |
| E8a moderate | -0.88 / +0.08 / -0.41 / **-3.14** | +0.11 / +0.07 / +0.41 / -0.64, **none significant** |
| E8a saturated | -1.02 / **-3.25** / -0.56 / 0.00 | -0.11 / **-1.61** / -0.38 / 0.00 |
| E8d ingredients | tokens 15 help; load 6 help, 5 harm | tokens 15 help; load **7 help, 2 harm** |
| E11 chat | -0.73 / -2.19 / -3.58 / -2.05 | -0.11 / -0.95 / -0.92 / -1.98, **only the last significant** |
| E11 code | +0.68 / +2.22 / +2.13 / -0.08 | +1.03 / **+2.42** / +1.97 / -0.08 |
| E11 Poisson control | +0.03 / -0.23 / -0.59 / **-3.67** | -0.05 / +0.18 / +0.25 / -0.56, **none significant** |
| E11 under a 3x p95 target | ahead in 0 of 32 (18 behind) | ahead in 0 of 32 (13 behind, 19 ties) |
| E10 token estimate | improves the rule in 6 of 6, 37-79% | improves it in **4 of 6**, 39-77% |

**Corrected claims.** (i) The rule is *indistinguishable* from a static threshold under moderate load,
not behind it; it is behind only under saturation. (ii) The token estimate helps where the constant
fails and adds nothing on a near-idle server, where the constant is already enough. (iii) The
explanation changed with the numbers: load CV no longer orders the regimes (near-idle and moderate
differ by 0.03 in it and by more than a point in mean gap). `scripts/e8_price_spread.py` ->
`results/e8/PRICE_SPREAD.md` measures the price each arriving query is charged and reports how much
cheaper the cheap moments are than a typical one: 14.0x (bursty), 2.0x (near-idle), 1.3x (moderate),
1.3x (saturated), which does order them, and orders the production processes where mean concurrency
cannot.

**Superseded:** the pre-fix figures above, everywhere they appeared -- section 5.3, 5.4, 5.5, 5.6 and
the chapter summary, the English and Italian abstracts, and section 4.5's table of the token estimate.
