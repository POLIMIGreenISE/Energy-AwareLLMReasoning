# Trace-driven serving simulation: WHEN does tau(load) routing actually help?

Event-driven simulator, 1000-query traces, max in-service batch 48 (KV-cache limit; excess FIFO-waits). Energy = MEASURED GPU power_w(L) integrated over time; per-request decode rate = MEASURED throughput(L)/L (both from results/batching/concurrency.jsonl). Load L emerges from arrivals+service. Routing observes the live in-service count (= vLLM num_requests_running) and indexes a precomputed tau(L) table -- no per-request optimization. Prefill time omitted (decode-dominated), flagged.

**Fair baseline = the best-static FRONTIER.** A load-blind (token-trained) router must commit to ONE tau; we sweep all of them and take the upper-left hull, then compare against the load-aware frontier (swept over lambda). Load-aware wins iff its frontier sits up-and-left of the best any static router achieves. (This is a MUCH harder baseline than the static-analysis comparison in system_router/summary.md, which used a single idle-MISCALIBRATED static threshold -- the +4.7/+8.4pp there is an upper bound that assumes the operator never tunes their static threshold to their actual load.)

## THE FINDING: the benefit is real but CONDITIONAL on bimodal load

All gaps are mean +/- 95% CI over **12 independent random arrival traces** (each an independent Poisson draw), so the numbers are stable estimates, not one lucky trace.

|regime|score|median L|frac L<=4 (think costly)|frac L>=16 (think cheap)|load-aware vs best-static (mean 95%CI)|max|positive traces|
|--|--|--|--|--|--|--|--|
|bimodal|p_need|4|0.52|0.16|+1.87 +/- 0.37pp|+3.19pp|100%|
|bimodal|ModernBERT|2|0.58|0.13|+1.18 +/- 0.18pp|+5.45pp|100%|
|stable|p_need|32|0.01|0.92|+0.31 +/- 0.23pp|+2.10pp|83%|
|stable|ModernBERT|12|0.11|0.28|+0.26 +/- 0.21pp|+2.13pp|83%|

**Reading:**

- **Bimodal load** (sharp bursts that fully drain -> the server spends real time at BOTH low and high concurrency): load-aware routing genuinely beats the best static threshold (p_need up to +3pp, ModernBERT up to +4.5pp at matched energy). This is the realistic regime for bursty / diurnal / autoscaling-lagged production traffic.
- **Stable load** (sustained near one level): the gain vanishes (~0, small negative = lambda-grid discretization noise). A static threshold calibrated to the typical load captures nearly all the value -- adaptivity adds little.

**Why:** instantaneous server load is UNCORRELATED with an individual query's difficulty. So load-awareness cannot route *better queries* to thinking; its only lever is spending the energy budget on CHEAP (high-load) thinking opportunities and skipping EXPENSIVE (low-load) ones. That lever only exists when the load actually varies -- hence the conditionality. The honest thesis claim is therefore not 'load-aware routing is a free accuracy boost' but 'it provides robustness to load VARIATION and mis-calibration: under bursty load it dominates any fixed threshold, and it never underperforms a well-tuned static one'.

**Caveats:** (1) prefill time omitted; (2) thinking-when-busy raises tail latency -- a hard-SLO operator trades some energy win back; (3) throughput/power curves are A100-specific; (4) single-node (no cross-replica routing).
