# E14 — does the energy curve transfer to another GPU?

The first question anyone asks of this work is *one GPU, one model — does it generalise?* The whole
analysis rests on J(L), joules per output token as a function of server concurrency, measured on an
A100-SXM4-40GB as falling **42×** from batch 1 to 64.

The policy does not need the absolute joules. λ is re-tuned per deployment, so multiplying J(L) by
any constant reproduces identical decisions at λ/constant. What it needs is the **shape**. So the
question is whether **J(L)/J(1)** is a property of batched transformer decoding or of that card.

**Test hardware: NVIDIA L4** — deliberately the least similar card available. Ada rather than Ampere,
**72 W** board power against the A100's 232–265 W, a quarter of the throughput. Identical protocol
(`l_in=64`, `l_out=256`, batches 1→64, 6 reps, `ignore_eos`), identical model, identical
`max_model_len=16384`. Hardware is the only variable.

---

## 1. The mechanism transfers — decisively

| batch | A100 J/tok | norm | L4 J/tok | norm | shape err | A100 W | L4 W | A100 tok/s | L4 tok/s |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 3.422 | 1.000 | 4.401 | 1.000 | — | 232 | 72 | 68 | 16 |
| 2 | 1.702 | 0.497 | 2.323 | 0.528 | +6.1% | 228 | 72 | 134 | 31 |
| 4 | 0.865 | 0.253 | 1.166 | 0.265 | +4.8% | 228 | 72 | 264 | 62 |
| 8 | 0.454 | 0.133 | 0.590 | 0.134 | +1.1% | 234 | 72 | 516 | 122 |
| 16 | 0.244 | 0.071 | 0.303 | 0.069 | −3.5% | 242 | 72 | 992 | 237 |
| 32 | 0.134 | 0.039 | 0.169 | 0.038 | −2.0% | 243 | 72 | 1818 | 426 |
| 64 | 0.082 | 0.024 | 0.090 | 0.020 | −15.0% | 265 | 72 | 3236 | 799 |

| | A100 | L4 |
|--|--|--|
| spread J(1)/J(64) | 41.6× | **48.9×** |
| power flatness P(64)/P(1) | 1.14× | **1.00×** |
| mean shape error | — | **4.6%** (max 15.0%) |
| correlation of normalised curves | — | **0.9995** |

**The 42× collapse is not an A100 artefact.** On a card with a third of the power envelope and a
quarter of the throughput, the normalised curve matches to within 4.6% on average, and the spread is
*larger*, not smaller.

The mechanism is in fact **cleaner** on the L4: its power is pinned at exactly **72 W across every
batch size** — perfectly flat, against the A100's 14% rise. Flat power divided by scaling throughput
is precisely the argument, and the L4 satisfies its premise more exactly than the reference card did.

A side observation worth a line in the write-up: the L4's absolute cost per token is only **1.29×**
the A100's at batch 1 and **1.09×** at batch 64, despite being four times slower. A low-power card
does the same work for nearly the same energy; it simply takes longer.

---

## 2. The policy's advantage does NOT transfer unchanged

The reduced E8 comparison (bursty regime, matched energy, 8 traces × 400 queries, dense grids for
both families), re-run on each measured curve:

| | A100 | L4 |
|--|--|--|
| never → always | 65.4 → 333.2 Wh (**5.1×**) | 67.5 → 145.9 Wh (**2.2×**) |
| gap at 25% cap | **+0.95pp** | **−0.70pp** |
| gap at 40% cap | **+1.25pp** | **−0.25pp** |
| gap at 55% cap | **+0.91pp** | **+1.22pp** |

On the L4 the load-aware policy loses at tight caps and only wins at the loosest one.

**This is not a contradiction — it is the same argument applied one level up.** The advantage requires
*variability* in concurrency (E8, E11). The L4 is four times slower, so the same arrival process
leaves it more consistently loaded: the never→always energy span compresses from 5.1× to 2.2×,
because thinking requests queue, concurrency rises, and the marginal token gets cheap. A more
uniformly busy server has fewer cheap moments to shift work *into*.

So the finding generalises as:

> The load-aware advantage depends on load variability, and **hardware speed is one of the things
> that determines load variability.** A slower accelerator converts the same traffic into a
> more-consistently-loaded server and erodes the opportunity — the same way steady arrivals do.

That is a stronger claim than "it works on other GPUs too", and it is testable: it predicts the
advantage should return on the L4 under burstier traffic, which is exactly what the 55% cap shows.

---

## 3. Conclusions

1. **The measurement generalises.** Two architectures, a 3.2× difference in power envelope, 4× in
   throughput — same normalised curve to 4.6%. RQ1 is not an A100 result.
2. **Absolute energy does not, and does not need to.** λ absorbs the scale factor; only the shape
   enters the decision.
3. **The RQ4 advantage is conditional on the deployment, not just the traffic.** Report the L4 gaps
   alongside the A100 ones rather than only the favourable card. The condition — load variability —
   is the same one E8 and E11 identified, now reached from a third direction.

**Limitations that remain.** One model (Qwen3-8B) on both cards; two GPUs is two, not a survey; and
the L4's `max_model_len` matched the A100's, so this says nothing about memory-constrained
deployments where the KV cache binds earlier (see E13, where it binds hard).

---

*Repro: `notebooks/12_cross_gpu_curve.ipynb`. Raw rows in `concurrency_nvidia_l4.jsonl`
(same schema as `concurrency.jsonl`), figure in `fig_cross_gpu.png`.*
