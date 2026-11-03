# E16 — repairing the closed-loop budget controller: it works below saturation and cannot work at it

E12 measured `BudgetController` failing on hardware (+36.7% over budget on bursty, +134.4% at
saturation). E15 proposed repairs but could not test them, because the simulator does not reproduce
the failure. This runs three controllers on real hardware, same budget, same traces.

**Result: the repair works on bursty traffic and does not work at saturation — and the reason it
cannot work at saturation is structural, not a tuning problem.**

---

## 1. The control reproduces the failure

| regime | E12 (as-shipped) | E16 (as-shipped) |
|--|--|--|
| bursty | +36.7% | **+35.9%** |
| high | +134.4% | **+154.6%** |

Different rental, different day, same collapse. The failure is real and repeatable, so any
improvement below is attributable to the controller rather than the box.

## 2. Below saturation the repair works

Authorised budget 37.25 Wh (1.5× the measured never-think energy of 24.83 Wh).

| controller | energy | vs budget | accuracy | think-rate | p95 | λ range | λ left ±3× band at |
|--|--|--|--|--|--|--|--|
| as-shipped | 50.60 Wh | **+35.9%** | 0.8975 | 40.0% | 297 s | 1.0e-08 … 2.2e-02 (**6.3 decades**) | n = 11 of 200 |
| asymmetric | 40.38 Wh | **+8.4%** | 0.8700 | 12.5% | 166 s | 1.6e-05 … 4.4e-02 (3.4 decades) | n = 60 |
| **guarded** | 39.69 Wh | **+6.6%** | 0.8625 | 9.5% | 113 s | 1.9e-04 … 2.7e-02 (**2.2 decades**) | n = 60 |

Think-rate by arrival quartile tells the story directly:

| controller | Q1 | Q2 | Q3 | Q4 |
|--|--|--|--|--|
| as-shipped | 26% | **82%** | 42% | 10% |
| guarded | 4% | 18% | 16% | 0% |

The as-shipped loop lost control by the **11th of 200 requests** and spent the second quartile
thinking on 82% of arrivals. The guarded loop held λ inside a factor of three until request 60 and
never exceeded an 18% think-rate.

**The guarded closed loop is also the best budget-holder of anything tried, offline or online.**
Against E12's offline results on the same regime and construction:

| approach | miss |
|--|--|
| offline static τ | +37.0% |
| offline load-aware λ | −15.7% |
| as-shipped closed loop | +35.9% |
| **guarded closed loop** | **+6.6%** |

That is the E12 negative result repaired: with the loosening direction damped and λ confined to a
bounded correction around the offline estimate, feedback on a real meter does what it was supposed
to do.

## 3. At saturation it does not — and cannot

Authorised budget 34.37 Wh (never-think measured 22.91 Wh, so the budget has ample headroom).

| controller | energy | vs budget | accuracy | think-rate | p95 | preemptions | KV max |
|--|--|--|--|--|--|--|--|
| as-shipped | 87.50 Wh | **+154.6%** | 0.8992 | 14.8% | 656 s | 181 | 1.00 |
| guarded | 71.57 Wh | **+108.2%** | 0.8932 | 10.8% | 433 s | 102 | 1.00 |

Guarding helped — 155% → 108% — and it did exactly what it was designed to do: λ stayed inside a
factor of three until request **364 of 660**, more than half the run. It still finished at more than
double its budget.

### Why: the spend is back-loaded, so the feedback signal is misleading, not merely late

The pacing trace shows the controller being told it was *under* budget for most of the run:

| controller | first update | midpoint | final |
|--|--|--|--|
| as-shipped | pace 0.84 (n=33) | pace 0.95 (n=347), spent 13.7 vs 18.1 Wh target | pace **2.55** |
| guarded | pace 0.78 (n=99) | pace 0.87 (n=380), spent 14.3 vs 19.8 Wh target | pace **2.08** |

At 58% of completions the guarded controller had spent **14.3 Wh of its eventual 71.6 Wh**. The
remaining **57 Wh — four fifths of the total — arrived during the last 42% of completions**, as the
queue drained with a 100%-full KV cache and 102 preemptions forcing recomputation (E13's mechanism).

This is not actuation lag that a faster loop would fix. **No controller paced on completed requests
can see this coming**, because the energy is not consumed in proportion to completions at all: it is
consumed in the drain, after the decisions that caused it have already been made and while nothing
new is completing. The signal reads "comfortably under budget" right up until the point where the
outcome is already determined.

## 4. What to do about it

**A rule an operator can follow:**

> Use the guarded closed loop below saturation, where it holds a budget to within ~7% and beats
> every offline alternative. Do not rely on any completion-paced controller at saturation. There,
> either keep the server out of saturation, or fall back to a conservative static threshold — E12's
> offline static τ came in at **+17.7%** at this load, six times closer than the best closed loop.

**Why the static threshold wins at saturation** is worth stating: its 4.1% think-rate kept the KV
cache below the preemption cliff. It is not better control; it is a policy conservative enough never
to enter the regime where control becomes impossible.

**What would be needed to control saturation properly** — none of it attempted here — is a signal
that leads rather than lags: queue depth and KV occupancy are both observable at admission
(E13 recorded them at 4 Hz), and both were saturated long before the energy arrived. A controller
pacing on *committed work in the queue* rather than *completed requests* is the obvious next design,
and it is future work.

## 5. Effect on the project's claims

CORRECTIONS §6a stated *"no fix is established"*. That now stands corrected in one direction and
confirmed in the other:

- **Established:** a guarded closed loop (asymmetric gain, 1.1× per-update step cap, λ bounded to
  [λ\*/10, λ\*·100], 15% warmup) holds an energy budget to +6.6% on bursty traffic, against +35.9%
  for the shipped loop and +37.0% / −15.7% for offline tuning.
- **Confirmed:** at saturation the problem is not the controller. Energy-budgeted reasoning control
  under KV-cache saturation remains open, and this work now bounds *why* rather than only *that*.

---

*Repro: `notebooks/13_controller_repair.ipynb`. Per-completion pacing traces in
`pacing_<level>_<controller>.jsonl`, per-request logs in `requests_*.jsonl`, summaries in
`summary.jsonl`.*
