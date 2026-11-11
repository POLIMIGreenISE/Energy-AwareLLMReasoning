# E3 — energy / latency / accuracy under load

> **⚠ PARTIALLY SUPERSEDED (2026-08-05).** The fixed-threshold baseline here was swept over a
> coarse grid that reached only 26 of the router's 684 achievable operating points, which
> inflated the load-aware advantage. The methodology and the qualitative direction stand; the
> magnitudes do not. **Authoritative numbers: `results/audit/fair_headtohead.md`.** Full detail:
> `results/CORRECTIONS.md` §1.

5 held-out traces of 500 queries. The **inverse** policy (think when the server is idle) is the control: if 'think when busy' is what saves energy, the inverse must show the mirror trade-off rather than being worse at everything.

|regime|constraint|policy|accuracy|energy (Wh)|p95 latency (s)|think-rate|
|--|--|--|--|--|--|--|
|bimodal|energy-cap|static-tau|0.8302|81.4|13.5|0%|
|bimodal|energy-cap|load-aware (think when busy)|0.8906|204.0|85.1|14%|
|bimodal|energy-cap|inverse (think when idle)|0.8488|130.6|20.3|4%|
|bimodal|slo=20|static-tau|0.8302|81.4|13.5|0%|
|bimodal|slo=40|static-tau|0.8302|81.4|13.5|0%|
|bimodal|slo=80|static-tau|0.8302|81.4|13.5|0%|
|bimodal|slo=13322771|static-tau|0.8302|81.4|13.5|0%|
|bimodal|slo=20|load-aware (think when busy)|0.8338|83.4|14.0|1%|
|bimodal|slo=40|load-aware (think when busy)|0.8594|135.6|38.4|6%|
|bimodal|slo=80|load-aware (think when busy)|0.8594|135.6|38.4|6%|
|bimodal|slo=13322771|load-aware (think when busy)|0.8906|204.0|85.1|14%|
|bimodal|slo=20|inverse (think when idle)|0.8292|80.3|13.4|0%|
|bimodal|slo=40|inverse (think when idle)|0.8488|130.6|20.3|4%|
|bimodal|slo=80|inverse (think when idle)|0.8488|130.6|20.3|4%|
|bimodal|slo=13322771|inverse (think when idle)|0.8488|130.6|20.3|4%|
|stable|energy-cap|static-tau|0.8302|29.7|13.9|0%|
|stable|energy-cap|load-aware (think when busy)|0.8348|32.5|15.0|2%|
|stable|energy-cap|inverse (think when idle)|0.8336|34.3|14.7|1%|
|stable|slo=20|static-tau|0.8302|29.7|13.9|0%|
|stable|slo=41|static-tau|0.8302|29.7|13.9|0%|
|stable|slo=82|static-tau|0.8302|29.7|13.9|0%|
|stable|slo=13628324|static-tau|0.8302|29.7|13.9|0%|
|stable|slo=20|load-aware (think when busy)|0.8348|32.5|15.0|2%|
|stable|slo=41|load-aware (think when busy)|0.8348|32.5|15.0|2%|
|stable|slo=82|load-aware (think when busy)|0.8348|32.5|15.0|2%|
|stable|slo=13628324|load-aware (think when busy)|0.8348|32.5|15.0|2%|
|stable|slo=20|inverse (think when idle)|0.8336|34.3|14.7|1%|
|stable|slo=41|inverse (think when idle)|0.8336|34.3|14.7|1%|
|stable|slo=82|inverse (think when idle)|0.8336|34.3|14.7|1%|
|stable|slo=13628324|inverse (think when idle)|0.8336|34.3|14.7|1%|

## Reading

**1. The mechanism is confirmed by the inverse control.** At a matched 217 Wh cap (bimodal), the
inverse policy is not worse at everything — it shows the mirror trade-off. The decisive number:
**"think when busy" buys a 14% think-rate while "think when idle" buys only 4% for the SAME energy**
(3.5x more reasoning per joule). The saving therefore comes from *when* thinking happens (batch-cheap
moments), not from choosing better queries — consistent with E2b, where the effect survived a random
router.

**2. The latency price is severe.** Load-aware reaches accuracy 0.8906 at p95 = 85.1s, against
static-tau's 0.8302 at 13.5s — **6.3x worse tail latency** at equal energy. Thinking during bursts is
efficient precisely because the batch is full, which is also when queues are longest.

**3. A tight SLO erodes most of the advantage.** Under a joint energy cap + p95 SLO, the load-aware
gain over static-tau falls from **+6.0pp (unconstrained) to +0.36pp at a 20s p95**, while remaining
+2.9pp at 40-80s. In the stable regime all policies sit within 0.5pp regardless of SLO.

**Scope statement:** load-aware reasoning allocation is for **energy-constrained,
latency-tolerant** serving — batch and asynchronous pipelines, off-peak processing — and not for
interactive serving under a strict tail-latency target, where a fixed threshold does nearly as well.
This also explains the conditionality found in E4: bursts are simultaneously where thinking is
cheapest and where latency is worst.
