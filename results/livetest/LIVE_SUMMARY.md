# E9 — the live matched-energy head-to-head: what it validated, and what it broke

Real A100, Qwen3-8B, vLLM, Zeus/NVML. Three arrival regimes × four policies, 12 runs, 980 requests.
The router under test is `src/router.py` — the deployable component, reading the server's real
`vllm:num_requests_running` over HTTP and pricing with the measured J(L) curve. Both policies were
tuned to the same energy budget **before** either ran, using the simulator on separate historical
traces.

The run validated three things and broke one. The broken one is the more useful result.

---

## 1. Validated: decision fidelity is exact

| level | policy | sim think-rate | measured think-rate |
|--|--|--|--|
| low | static | 0.033333 | 0.033333 |
| low | load-aware | 0.008333 | 0.008333 |
| bursty | static | 0.025000 | 0.025000 |
| bursty | load-aware | 0.005000 | 0.005000 |
| high | static | 0.136364 | 0.136364 |
| high | load-aware | 0.127273 | **0.128788** |

Eleven of twelve runs reproduce the simulated decision **request for request**. The twelfth differs
by a single request out of 660 — the load-aware policy at saturation, where one admission saw a
different live concurrency than the model predicted. Delivered accuracy is identical in every run
by construction, since correctness is looked up per query.

**The live probe was flawless: 0 HTTP errors in 25,783 reads.** The router never once priced a query
with a stale or missing concurrency reading.

## 2. Validated: energy prediction, *below saturation*

The simulator charges power only while the GPU is busy, so its prediction must fall **below** the
measured total and **above** the idle-subtracted figure. That bracket is the test.

| level | policy | sim (Wh) | measured (Wh) | idle-sub (Wh) | err vs total | err vs idle-sub |
|--|--|--|--|--|--|--|
| low | never | 19.18 | 21.42 | 17.45 | −10.5% | +9.9% |
| low | static | 21.21 | 23.56 | 19.40 | −10.0% | +9.3% |
| low | load-aware | 20.86 | 23.01 | 19.04 | −9.3% | +9.6% |
| low | random | 20.24 | 22.58 | 18.61 | −10.3% | +8.8% |
| bursty | never | 21.22 | 25.40 | 19.54 | −16.5% | +8.6% |
| bursty | static | 34.12 | 38.35 | 31.20 | −11.0% | +9.4% |
| bursty | load-aware | 24.63 | 28.48 | 22.62 | −13.5% | +8.9% |
| bursty | random | 23.92 | 27.75 | 21.85 | −13.8% | +9.5% |
| high | never | 21.77 | 22.94 | 19.17 | −5.1% | +13.6% |
| high | random | 29.11 | 33.59 | 28.06 | −13.3% | +3.7% |

**Ten of twelve runs sit inside the bracket**, with errors of 5–17% against the total — reproducing
the E6 result on two new arrival regimes and confirming that the residual is idle-power accounting
rather than luck.

## 3. BROKEN: the model fails at saturation

| level | policy | sim (Wh) | measured (Wh) | idle-sub (Wh) | err vs total | err vs idle-sub |
|--|--|--|--|--|--|--|
| high | static | 41.50 | **73.03** | 61.01 | **−43.2%** | **−32.0%** |
| high | load-aware | 43.95 | **77.66** | 64.89 | **−43.4%** | **−32.3%** |

These two fall **outside the bracket** — the prediction is below even the idle-subtracted figure, so
idle-power accounting cannot explain it. The latency tells the same story: predicted p95 190 s
against **459 s and 514 s measured**, a 2.4× under-prediction.

**The cause is queueing, not energy.** At this level the server sits pinned at its concurrency cap
(`L` median 32 = `--max-num-seqs`), serving 13% thinking requests of ~2,700 tokens each. Real vLLM
under KV-cache pressure preempts and recomputes; the simulator's FIFO queue does neither. Everything
simply takes far longer than modelled, and a GPU held at ~240 W for 2.4× longer burns 1.8× the
energy. E6 already found latency under-predicted at high load by 28%; at genuine saturation the
error compounds into the energy channel.

**Scope statement this forces.** *The simulator is validated below saturation and must not be
trusted at it.* E8's `saturated` regime rests on the same model and inherits the same error.

**What partially rescues E8:** the error is nearly identical for both policies (−43.17% vs −43.41%),
so it is close to a common scale factor and the *relative* comparison likely survives even though
absolute energy does not. That is an argument, not a proof, and it should be stated as such.

## 4. BROKEN: offline calibration does not hold the budget

| level | policy | authorised | realised | vs budget |
|--|--|--|--|--|
| low | static | 24.02 Wh | 23.56 | **−1.9%** |
| low | load-aware | 24.02 Wh | 23.01 | **−4.2%** |
| bursty | static | 23.05 Wh | 38.35 | **+66.4%** |
| bursty | load-aware | 23.05 Wh | 28.48 | **+23.5%** |
| high | static | 43.30 Wh | 73.03 | **+68.7%** |
| high | load-aware | 43.30 Wh | 77.66 | **+79.4%** |

Tuning λ and τ offline against the simulator held the budget only at the low level. At bursty and
high it overshot by 24–79% — because the parameter was chosen against a model that under-predicts
energy in exactly those regimes (§3), and because a single λ produces very different behaviour on
different traces (§5).

> **UPDATE (E12, `results/closedloop/CLOSED_LOOP.md`): this proposal was tested and it FAILED.**
> The paragraph below predicted that closing the loop on a real meter would fix these misses. It
> was run on an A100 against an NVML energy counter and did worse than the offline tuning it was
> meant to repair: **+36.7%** over budget on bursty and **+134.4%** at saturation, with λ pinned
> at its clamp. The prediction is withdrawn; see CLOSED_LOOP.md for the pacing trace and what
> now stands in its place.

~~**This is the strongest argument in the project for the closed-loop controller.**~~ `BudgetController`
in `src/router.py` adapts λ against a real energy meter and counts energy already committed by
in-flight requests. It was never run live. It should be: three runs, roughly 25 minutes, and it
directly addresses the one thing this experiment shows offline tuning cannot do.

## 5. Why the load-aware policy barely thought at low and bursty load

Not a simulator-vs-reality divergence — simulator and reality agree exactly (§1). The divergence is
between the **historical traces used for tuning** and the **deployment trace**:

| level | think-rate predicted on tuning traces | on the deployment trace (sim) | measured |
|--|--|--|--|
| low | 4.7% | 0.83% | 0.83% |
| bursty | 4.2% | 0.50% | 0.50% |
| high | 12.6% | 12.7% | 12.9% |

The mechanism is the steepness of J(L). The threshold is `λ · J(L) · E[Δtok]`, and J(L) **doubles
between L = 1 and L = 2**. Recorded concurrency at admission:

| level | L at admission | thinking actually occurred at |
|--|--|--|
| low | median 2, p90 4, max 5 | only L = 5 |
| bursty | median 5, p90 15, max 20 | only L = 20 |
| high | median 32, p90 32 | L = 13 … 32 |

At low concurrency the policy only thinks in the rare moments the queue happens to fill, so its
think-rate is governed by the tail of the load distribution — which varies sharply from trace to
trace. At saturation J(L) is nearly flat, the policy becomes insensitive to L, and the tuned
parameter transfers almost perfectly (12.6% → 12.9%).

**The finding:** *the load-aware policy is hardest to tune offline exactly where the price curve is
steepest.* That is the flip side of the mechanism that makes it work — the same steep J(L) that
creates the opportunity also makes the policy's behaviour unstable across traces. The static
threshold has no such sensitivity and reproduced its tuned think-rate at every level.

## 6. The head-to-head itself — and why it is not the headline

| level | static | load-aware | gap | energy match |
|--|--|--|--|--|
| low | 0.8125 @ 23.56 Wh | 0.8000 @ 23.01 Wh | −1.25pp | −2.4% ✔ |
| high | 0.9121 @ 73.03 Wh | 0.8992 @ 77.66 Wh | −1.29pp | +6.3% ✘ |
| bursty | 0.8350 @ 38.35 Wh | 0.8225 @ 28.48 Wh | −1.25pp | **−25.7%** ✘ |

**Only the low level is a matched-energy comparison**, and at n = 120 a 1.25pp gap is 1.5 queries —
far inside noise. At high the energies differ by 6.3% and load-aware is *dominated* (more energy,
less accuracy), which is a real if uncontrolled loss. At bursty the two policies sat 26% apart in
energy, so their accuracies are simply not comparable.

**The accuracy claim does not come from this experiment and should not.** It comes from E8 and E11 —
20 and 12 held-out traces of 500 queries with paired bootstrap CIs. E9's job is validating the
energy model, the decision fidelity, and the router as a working component. On those it delivered;
on the head-to-head the calibration failed first.

---

## What to change

1. **Run the closed-loop `BudgetController` live** (~25 min). It is the direct answer to §4 and the
   only untested part of the shipped artefact.
2. **Restate the simulator's validity as a bounded claim**: within ~10% below saturation, unreliable
   at it. Add the saturation caveat to E8's `saturated` rows.
3. **Report `L` consistently.** The notebook's `LoadRecorder` samples at 4 Hz across the whole run
   including idle gaps; the simulator's `load_mean` is time-weighted over busy periods only. The two
   `L` columns in `sim_vs_live.jsonl` are not the same statistic.
4. **Raise `--max-num-seqs` or state it as the binding constraint.** At the high level `L` is pinned
   at the 32-request cap, so that regime measures the admission limit as much as the arrival process.

*Repro: `notebooks/10_live_router_test.ipynb`. Raw per-request logs in `live_requests_*.jsonl`,
summaries in `live_summary.jsonl`, `sim_vs_live.jsonl`, `headtohead.jsonl`.*
