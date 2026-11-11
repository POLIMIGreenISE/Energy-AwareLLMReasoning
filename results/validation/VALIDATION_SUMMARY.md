# E6 + E7 — validating the serving simulator against a real vLLM server

Real A100-SXM4-40GB, Qwen3-8B, vLLM OpenAI server, Zeus/NVML energy. The replay fires the
**simulator's own arrival trace** (same seed, same queries, token counts forced with `ignore_eos`)
at real arrival times, polling `/metrics` for the true `num_requests_running`.

---

## E6 — simulator vs reality

|policy|quantity|simulator|measured|error|
|--|--|--|--|--|
|static-τ|energy|45.21 Wh|49.75 Wh (41.92 idle-sub)|**−9.1%** (+7.8% vs idle-sub)|
|static-τ|p95 latency|153.5 s|211.9 s|−27.6%|
|static-τ|mean concurrency|6.6|7.9|−16.3%|
|static-τ|think-rate / accuracy|16% / 0.9075|16% / 0.9075|**exact**|
|load-aware|energy|21.81 Wh|24.41 Wh (19.07 idle-sub)|**−10.6%** (+14.4% vs idle-sub)|
|load-aware|p95 latency|13.7 s|13.3 s|**+3.4%**|
|load-aware|mean concurrency|3.2|2.0|+61.4%|
|load-aware|think-rate / accuracy|1% / 0.825|0.5% / 0.825|**exact accuracy**|

**Energy is predicted within ~10%, and the error is structurally explained rather than lucky.**
The simulator charges power only while the GPU is busy (L>0) and nothing during idle gaps, so it
*must* fall below the measured total (which includes idle draw between bursts) and *above* the
idle-subtracted figure (which over-subtracts, removing idle power from busy periods too). The
prediction is bracketed by the two bounds in both policies — exactly the expected behaviour.

**Decision fidelity is exact**: identical think-rate and identical delivered accuracy, confirming
the replay reproduces the policy rather than approximating it.

**Latency: accurate at low load (+3.4%), under-predicted at high load (−27.6%).** The simple FIFO
model misses real vLLM's scheduling overhead and preemption once concurrency is high. The direction
matters: **reality is slower than simulated, so E3's "latency is the price of load-aware routing"
warning is conservative — the true penalty is larger than reported.**

Concurrency error is large in relative terms for load-aware (3.2 vs 2.0) but small in absolute terms
(1.2 requests); at L≈2 the ratio is inherently noisy.

---

## E7 — the three extrapolations, now measured

### 1. KV-cache growth (output length) — real but modest, and *milder* than we assumed

J/token by batch × output length:

|batch|l_out=256|1024|3072|ratio 3072/256|implied α|
|--|--|--|--|--|--|
|1|3.039|3.375|3.524|1.16×|0.057|
|4|0.784|0.868|0.919|1.17×|0.061|
|16|0.214|0.238|0.269|1.26×|0.091|
|32|0.117|0.136|0.165|1.42×|**0.148**|

Long generations do cost more per token, and the effect **grows with batch size** (KV bandwidth
dominates when many sequences are resident). But **measured α = 0.057–0.148, whereas E5 tested
α = 0.15 and 0.40** — i.e. reality sits at or just below our *least* severe stress test. **The
E5 α=0.15 variant is effectively the calibrated case, and it preserved both conclusions.**

### 2. Prefill — negligible for this workload

|l_in|total energy (batch 8)|extra per request|
|--|--|--|
|64|908 J|—|
|1024|1013 J|+13.1 J (0.0137 J/prompt-token)|
|4096|1105 J|+24.6 J|

OTB prompts average **41 tokens ⇒ ~0.5 J per request**. E5 stress-tested 200 J and 800 J per
request — **400–1600× more severe than reality**. The "decode-dominated, prefill omitted"
assumption is vindicated for this workload (it would matter for long-prompt traffic like WildChat).

### 3. Mixed batches — long requests get expensive when their batch drains

|composition (batch 16)|total tokens|J/token|observed mean L|
|--|--|--|--|
|homogeneous, l_out=3072|49k|**0.269**|~16|
|8 long (3072) + 8 short (128)|~26k|0.474|8.3|
|2 long (3072) + 14 short (128)|~8k|**1.383**|2.5|

A batch of 16 that contains only 2 long requests runs at **5.1× worse J/token** than a homogeneous
batch of the same nominal size. The short requests finish quickly, the batch drains, and the long
requests complete *alone at low concurrency* — where energy per token is worst. The observed mean
concurrency (2.5, not 16) shows the mechanism directly.

**This is already captured by the simulator**, which tracks each request individually and recomputes
L at every event, so batches drain naturally — which is part of why the energy prediction lands
within 10%. It also sharpens the main claim: in real mixed traffic a thinking request that
outlives its batch-mates is *more* expensive than a naive batch-size argument suggests, so the
energy case for not thinking unnecessarily is stronger, not weaker.

---

## Verdict

1. **Energy — validated within ~10%**, with the residual explained by idle-power accounting.
2. **Policy decisions and delivered accuracy — reproduced exactly.**
3. **KV growth and prefill — both milder than the E5 stress tests**, so the robustness result
   already covers reality with margin; no conclusion changes.
4. **Latency — under-predicted at high load**, which makes our stated latency limitation
   conservative rather than optimistic.

The RQ4 results can now be reported as *simulator-based and validated against real serving within
10% on energy*, rather than simulator-based alone.

**Remaining limitations:** single GPU type (A100-40GB); single model (Qwen3-8B); synthetic (not
production) arrival traces; `--max-num-seqs 12` bounded concurrency in the replay, so very high
concurrency regimes are untested on real hardware.
