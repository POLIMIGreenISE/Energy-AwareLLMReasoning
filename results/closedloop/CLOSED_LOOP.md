# E12 + E13 — closing the budget loop on a real meter, and why the model breaks at saturation

Real A100, Qwen3-8B, vLLM, NVML energy counter. Two regimes × four policies, 8 runs, 1,720 requests.
Follow-up to E9 (`results/livetest/LIVE_SUMMARY.md`), which left both questions open.

**One experiment confirmed a hypothesis; the other refuted a proposal this project had been making.**

---

## Setup

The budget is **1.5× the measured never-think energy of that regime**, so it is feasible by
construction. E9's budgets were simulator-derived and its bursty budget was actually *below* what the
hardware spends doing no thinking at all, which made that comparison meaningless.

The offline method is also given a **one-point calibration**: the never-think run yields
`k = measured/simulated` on that same trace, and the simulator budget is divided by `k` before
tuning. So offline tuning is not being blamed for a constant scale error any operator would correct.

| regime | n | measured never | k | authorised budget | τ\* | λ\* |
|--|--|--|--|--|--|--|
| bursty | 200 | 24.75 Wh | 1.166 | 37.13 Wh | 0.9433 | 1.18e-03 |
| high | 660 | 22.93 Wh | 1.053 | 34.40 Wh | 0.9668 | 1.63e-03 |

**The meter is sound.** The NVML total-energy counter agreed with Zeus to **0.0% on all eight runs**.
Nothing below is a metering artefact.

---

## E12 — the closed loop failed, and failed worse than what it replaced

| regime | policy | energy | vs budget | accuracy | think-rate | p95 | final λ |
|--|--|--|--|--|--|--|--|
| bursty | static, offline | 50.86 Wh | **+37.0%** | 0.8925 | 11.0% | 164 s | — |
| bursty | load-aware, offline | 31.28 Wh | **−15.7%** | 0.8350 | 2.5% | 19 s | — |
| bursty | **load-aware, closed loop** | 50.75 Wh | **+36.7%** | 0.8975 | 39.0% | 298 s | 2.83e-02 |
| high | static, offline | 40.48 Wh | **+17.7%** | 0.8689 | 4.1% | 46 s | — |
| high | load-aware, offline | 48.84 Wh | **+42.0%** | 0.8689 | 5.2% | 90 s | — |
| high | **load-aware, closed loop** | 80.62 Wh | **+134.4%** | 0.8977 | 12.9% | 635 s | 1.00e+04 |

The controller was built to repair offline tuning's budget misses. It did the opposite: worse than
offline in both regimes, and catastrophically so at saturation, where λ ended pinned at its **upper
clamp** and the run still finished at 2.3× the authorised energy.

### How it failed — from its own pacing trace

```
n_done=10   spent 0.34 Wh   target 1.86 Wh   pace 0.305   ->  lambda 6.50e-04
n_done=12   spent 0.37 Wh   target 2.23 Wh   pace 0.270   ->  lambda 1.81e-04
n_done=29   spent 1.00 Wh   target 5.38 Wh   pace 0.301   ->  lambda 1.00e-08   <- the FLOOR
n_done=49   spent 8.47 Wh   target 9.10 Wh   pace 1.432   ->  lambda 3.86e-08
```

Within **29 of 200 requests** the controller drove λ five orders of magnitude down to its clamp —
think on everything. Think-rate by arrival quartile: **26% / 82% / 38% / 10%**. By the time the bill
for the second quartile arrived, pace had inverted and λ climbed back to 2.8e-02, far too late.

At saturation the same shape appears more slowly and ends worse: λ drifts from 1.63e-03 to 4.9e-04
over the first third, think-rate rises to 22% in the third quartile, energy explodes, and λ slams
into the ceiling.

### The best available explanation, and its limits

The pacing signal is `committed / (budget × n_done/N)` — it assumes energy accrues in proportion to
**completed requests**. Completion order is correlated with cost: short no-think requests finish
first, so early in any window realised spend is *structurally* below the linear target whether or not
the policy is on track. The controller reads that as headroom and loosens.

**This explanation is not confirmed.** `results/e15/CONTROLLER_FIX.md` tested it in simulation and
the simulator does not reproduce the failure at all (−3.7% on bursty where live was +36.7%), and
pacing against a clock instead of a counter — which should remove the bias entirely — does not help.
An asymmetric gain (loosen at a quarter the rate you tighten) is the most promising of three
candidate repairs, but at saturation it still breaches on 42% of traces. **The fix is not
established, and re-running E12 with it on hardware is the honest next step.**

### One thing the controller was good at

It produced the **highest accuracy in both regimes** (0.8975, 0.8977). It is an effective accuracy
maximiser and a bad budget controller — which is the expected failure mode of a loop whose feedback
arrives after the decisions it should have constrained.

### What this does to the project's claims

E9's write-up proposed the closed loop as *"the strongest argument in the project for the closed-loop
controller"* and predicted it would fix offline tuning's misses. **That prediction is withdrawn.**
What stands is narrower and still useful:

> Neither offline tuning nor naive closed-loop feedback reliably holds an energy budget for a
> reasoning router under load. Offline tuning misses in both directions (+37% and −16% in the same
> regime); feedback on metered spend overshoots because the energy of a thinking decision arrives
> long after the decision. Energy-budgeted reasoning control is an open control problem, and this
> work bounds it rather than solving it.

---

## E13 — preemption at saturation: hypothesis confirmed

E9 found energy under-predicted by 43% and p95 latency by 2.4× at saturation, and blamed vLLM
preempting and recomputing under KV-cache pressure. That was asserted, never measured. Every run here
recorded the server's own counters at 4 Hz.

| regime | policy | preemptions | KV mean | KV max | wait mean | wait max | L mean |
|--|--|--|--|--|--|--|--|
| bursty | never | 0 | 0.006 | 0.042 | 0.0 | 0 | 1.9 |
| bursty | static, offline | 0 | 0.164 | 0.363 | 0.0 | 0 | 5.7 |
| bursty | load-aware, offline | 0 | 0.033 | 0.087 | 0.0 | 0 | 2.8 |
| bursty | load-aware, closed loop | 0 | 0.401 | 0.776 | 0.0 | 6 | 12.2 |
| high | never | 0 | 0.030 | 0.084 | 0.0 | 0 | 10.5 |
| high | static, offline | 0 | 0.276 | 0.594 | 3.8 | 48 | 15.4 |
| high | load-aware, offline | **6** | 0.414 | **1.000** | 4.9 | 56 | 16.3 |
| high | load-aware, closed loop | **169** | 0.699 | **1.000** | 75.5 | 224 | 22.2 |

**Confirmed.** Zero preemptions everywhere at low and bursty load; at saturation the KV cache hits
**100% occupancy**, the waiting queue reaches 224, and preemptions appear and scale with think-rate —
6 at a 5.2% think-rate, **169** at 12.9%.

The mechanism is now explicit: thinking requests generate ~2,700 tokens each, their KV footprint
fills the cache, vLLM preempts and later recomputes the evicted prefixes, and that recomputation is
work the simulator's FIFO queue does not model. It costs both the 43% energy and the 2.4× latency
E9 measured.

**Consequences worth carrying into the write-up:**

1. *The simulator's validity boundary is now mechanical, not empirical.* It is valid while the KV
   cache is not saturated, and that is a condition an operator can check.
2. *Thinking has a super-linear cost at saturation that no per-token model captures.* Every energy
   model in this project, including the load-aware one, prices reasoning per token. Once preemption
   starts, the true marginal cost of one more thinking request includes the recomputation it forces
   on others — which strengthens the general argument for not thinking unnecessarily.
3. *It explains E12's severity.* The closed loop's overshoot was amplified by the very mechanism
   E13 measured: it over-thought, which saturated the cache, which triggered 169 preemptions, which
   burned energy that made the overshoot worse still.

---

*Repro: `notebooks/11_closed_loop_budget.ipynb`. Raw per-request logs in `requests_*.jsonl`,
per-completion pacing traces in `pacing_*.jsonl`, summaries in `summary.jsonl`.*
