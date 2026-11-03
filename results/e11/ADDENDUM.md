# E11 addendum: burst timescales and idle-corrected energy

## 1. Timescale of the bursts (original traces, before rescaling)

Fano factor is the variance over the mean of arrival counts in fixed windows; a Poisson process gives 1 at every window size, as does its inter-arrival CV. `window` is how much original-trace time one 500-request evaluation window covers.

|trace|requests|span|mean rate|inter-arrival CV|Fano 1 s|Fano 10 s|Fano 60 s|window|dilation to 0.8 req/s|
|--|--|--|--|--|--|--|--|--|--|
|conversation|19366|58.4 min|5.53 req/s|1.09|1.39|4.00|20.55|90 s|x6.91|
|code completion|8819|57.3 min|2.57 req/s|13.15|13.18|71.30|160.54|195 s|x3.21|

Conversational traffic is close to Poisson arrival by arrival (inter-arrival CV and 1 s Fano both near 1) and bursty only as a slow drift in rate, visible at 60 s. One evaluation window covers about 90 s of it, so a minute-scale drift barely registers inside a window. Code-completion traffic is bursty at every window size.

## 2. Idle-corrected energy

The simulator charges no power while the GPU is idle. This adds the measured **37.2 W** A100 idle draw (weights resident, nothing in service; `results/validation/replay_summary.jsonl`) over each run's idle wall clock. Same 12 traces x 500 queries as `REAL_TRACE.md`; the busy-only columns reproduce that file exactly (checked in the script).

|arrival process|policy|idle fraction|busy-only (Wh)|with idle (Wh)|
|--|--|--|--|--|
|Poisson @0.8 (control)|never|1.9%|39.67|39.79|
|Poisson @0.8 (control)|always|0.2%|56.06|56.08|
|Azure conv @0.8|never|2.1%|37.59|37.75|
|Azure conv @0.8|always|0.0%|56.51|56.51|
|Azure code @0.8|never|47.1%|16.64|20.27|
|Azure code @0.8|always|4.4%|58.36|59.17|
|Azure code @0.8 (thinned)|never|46.4%|19.54|22.40|
|Azure code @0.8 (thinned)|always|0.8%|55.76|55.85|

### Ratios

|comparison|busy-only|with idle|
|--|--|--|
|never-reason, Poisson / Azure code|2.38x|1.96x|
|never-reason, Poisson / Azure code (thinned)|2.03x|1.78x|
|always / never, Poisson @0.8 (control)|1.41x|1.41x|
|always / never, Azure conv @0.8|1.50x|1.50x|
|always / never, Azure code @0.8|3.51x|2.92x|
|always / never, Azure code @0.8 (thinned)|2.85x|2.49x|

### Reading

The correction runs the opposite way to the one in the E8 concurrency axis. There the expensive, near-idle server was the one that idled, so charging idle power widened the gap. Here the cheap server idles: the code-completion server is idle for about half its wall clock and the Poisson control almost never. The burstiness discount therefore narrows once idle power is charged, but it survives at about a factor of two.

*Repro: `python scripts/e11_addendum.py` (CPU, under a minute).*
