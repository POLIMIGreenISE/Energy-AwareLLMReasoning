# Cross-dataset generalization of the OTB-trained routers

**Question.** Our routers are trained on OptimalThinkingBench (OTB), which is bimodal by
construction (over-think half = simple, under-think half = hard math). Do they generalize to
*different* query distributions? We test on two held-out datasets, judged on the **same model
(Qwen3-8B, `enable_thinking` toggled)** so the label definition is *our* energy-aware oracle,
not the source dataset's:

- **GSM8K** (1,319 q) — grade-school math with **gold answers** (programmatic scoring, clean).
- **Mohseni reasoning-router-data-v2** (1,810 q) — Big-Math-RL + AIME + Nectar + WildChat, **no
  gold** → reference-free grade-both judging (gemini-2.5-flash, our certified judge, κ 0.824).

Routers evaluated: **LogReg+MiniLM** and **fine-tuned ModernBERT** (retrained on *all* OTB, then
predicted on the new prompts — no leakage, the test set is a different corpus) and the
**p_need probe** (ask Qwen3-8B "would you need to reason?", read the Yes/No logit).

The crucial control is the **random-router EAA floor**: when Always-Think is much better than
Never-Think, even *random* routing captures a large share of the oracle's energy-accuracy gain
just by partially sweeping toward thinking. A router only demonstrates transfer if it beats that
floor.

---

## Results

### GSM8K (thinking barely helps — 95% saturated)
Never 0.948 @ 270 mWh · Always 0.960 @ 2452 mWh (**+1.2pp for 9× energy**) · Oracle EAA 0.9746.
Oracle routes only **36/1319 (3%)** to think.

| router | label-AUC | EAA | % oracle gain | **vs random** |
|--|--|--|--|--|
| *random floor* | — | 0.9545 | 25% | — |
| ModernBERT | 0.623 | 0.9560 | 31% | **+5pp** |
| LogReg+MiniLM | 0.567 | 0.9562 | 31% | **+6pp** |
| p_need | 0.602 | 0.9515 | 14% | **−11pp** |

### Mohseni (thinking genuinely helps — real headroom)
Never 0.611 @ 701 mWh · Always 0.823 @ 4237 mWh (**+21.2pp for 6× energy**) · Oracle EAA 0.8561.
Oracle routes **490/1810 (27%)** to think.

| router | label-AUC | EAA | % oracle gain | **vs random** |
|--|--|--|--|--|
| *random floor* | — | 0.7164 | 43% | — |
| LogReg+MiniLM | 0.505 | 0.7304 | 49% | **+6pp** |
| ModernBERT | 0.521 | 0.7301 | 49% | **+6pp** |
| p_need | 0.564 | 0.6572 | 19% | **−24pp** |

---

## What this says (consistent across both datasets)

1. **Trained routers transfer weakly but positively: +5–6pp above the random floor on both
   datasets.** The signal is real but modest — a benchmark-specific router does *not* fully
   generalize zero-shot. This is the expected ML behavior and directly **motivates training on a
   diverse pool** (OTB + others) rather than a single bimodal benchmark.

2. **p_need does NOT transfer — it is *worse than random* off-distribution** (−11pp GSM8K,
   −24pp Mohseni). This extends and hardens our OTB headline (trained router > p_need): off its
   training distribution the probe is not merely weaker, it is actively harmful as an
   energy-accuracy router. Note p_need's label-AUC is the *highest* of the three on Mohseni
   (0.564) yet its EAA is the *worst* — its Yes/No scores rank marginally better than chance but
   are so poorly spread that threshold-sweeping lands all operating points in a bad energy region.
   Ranking ≠ deployable routing.

3. **Label definitions are genuinely model- and energy-specific.** Our energy-aware oracle labels
   only **27%** of Mohseni "think" vs Mohseni's own **50%** — agreement just **55.4%**, and only
   ~56–59% even on the math sources. This *disentangles* the earlier free-test ambiguity (AUC 0.50
   vs his labels): the low agreement is not purely router failure, it is partly that "needs
   thinking" for Qwen3-8B (does thinking flip *this model* from wrong→right?) ≠ Mohseni's
   definition. We can now quantify that split, which the label-only free test could not.

4. **The overthinking thesis holds on held-out data.** On GSM8K, thinking buys +1.2pp for 9× the
   energy and the oracle routes 97% to no-think; on Mohseni it buys a real +21.2pp. A router only
   pays off where the distribution actually contains think-needed queries — exactly why a mixed
   benchmark matters.

**Honest framing for the thesis:** this is a *valuable negative/limitation result*, not a fatal
flaw. Zero-shot cross-benchmark transfer is weak (trained routers barely beat random; p_need loses
to it), which is a known phenomenon and the standard motivation for diverse-distribution training
and/or light target-domain calibration. The clean, model-consistent, energy-aware labels + the
random-router floor are what let us state this rigorously instead of hand-waving.

*Repro: `scripts/eval_testset.py --dataset {gsm8k,mohseni}` (needs GEMINI_API_KEY; Mohseni reuses
`judge_cache_mohseni.jsonl`). Random floor / offline re-score: `scripts/eval_mohseni_offline.py`.
Per-dataset detail in `eval_gsm8k.md`, `eval_mohseni.md`.*
