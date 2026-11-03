# E5 — sensitivity of the RQ4 conclusions to the simulator's assumptions

> **⚠ PARTIALLY SUPERSEDED (2026-08-05).** The fixed-threshold baseline here was swept over a
> coarse grid that reached only 26 of the router's 684 achievable operating points, which
> inflated the load-aware advantage. The methodology and the qualitative direction stand; the
> magnitudes do not. **Authoritative numbers: `results/audit/fair_headtohead.md`.** Full detail:
> `results/CORRECTIONS.md` §1.

Bimodal regime, 5 held-out traces of 500 queries, budget = 40% of always-think. **C1** = a token-blind controller misses its energy budget while the load-aware one hits it. **C2** = load-aware beats a fixed threshold at matched energy (mean gap across the overlapping frontier range).

|variant|never (Wh)|always (Wh)|C1 token-blind miss|C1 load-aware miss|C2 static|C2 load-aware|C2 gap|
|--|--|--|--|--|--|--|--|
|baseline|78.9|424.0|+94.8%|-9.6%|0.8651|0.8784|+1.33pp|
|KV growth a=0.15|80.1|643.9|+110.5%|-8.1%|0.8650|0.8770|+1.21pp|
|KV growth a=0.40|82.2|1010.4|+123.0%|-7.3%|0.8644|0.8770|+1.27pp|
|prefill 200 J/req|106.7|451.8|+84.0%|-8.5%|0.8651|0.8784|+1.33pp|
|prefill 800 J/req|190.0|535.2|+62.7%|-6.3%|0.8651|0.8784|+1.33pp|
|max_batch=24|78.9|424.3|+94.8%|-9.6%|0.8651|0.8784|+1.33pp|
|max_batch=96|78.9|424.0|+94.8%|-9.6%|0.8651|0.8784|+1.33pp|
|step interpolation|81.3|442.6|+95.5%|-7.5%|0.8651|0.8779|+1.28pp|
|KV .4 + prefill 800|193.3|1121.5|+98.8%|-5.9%|0.8644|0.8770|+1.27pp|

**C1: holds in every variant. C2: holds in every variant.**

## Reading

- **Both conclusions are assumption-robust.** C1 holds with the token-blind controller overshooting
  +63…+123% while load-aware stays within 10% in all nine variants; C2's gap is stable at
  +1.2…+1.3pp regardless of the modelling choice.
- **The KV bias runs in our favour.** A KV penalty inflates always-think energy sharply (424 → 1010 Wh
  at a=0.40) because thinking requests generate by far the most tokens. If the real KV effect is
  large, we are currently **under**-stating the cost of thinking, so our energy-saving estimates are
  conservative and correcting the model strengthens the case.
- **max_batch does not matter** in this regime (24/48/96 give identical results) — the bimodal trace
  rarely exceeds 24 in-service requests. This assumption can be dropped from the limitations.
- **What E6/E7 is therefore for:** not correctness (E5 covers that) but *calibration and credibility* —
  measuring the true KV and prefill coefficients so the absolute Wh figures are right, and showing the
  simulator predicts a real vLLM server to within a stated error.
