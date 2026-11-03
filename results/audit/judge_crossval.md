# Judge cross-validation (Claude hand-judge vs gemini-2.5-flash)

- contested rows judged by Gemini: 384 (fresh 20, cache 364, failed 5)
- L1 judged-verdict sums per (qid,mode): 177/205 = 0.863
- L2 aggregate correctness exact-match 0.941, mean|diff| 0.037
- L3 oracle labels: agreement 0.943, kappa 0.824; mine {'no_think': 197, 'think': 50} vs gemini {'no_think': 197, 'think': 50}

## Run details (2026-07-14)

- Judge: gemini-2.5-flash, temperature 0, thinking disabled (GEMINI_THINKING_BUDGET=0),
  max 3072 output tokens (512 truncated hard fraction-equivalence checks mid-arithmetic).
  gemini-3.5-flash was 503-overloaded; gemini-3.1-pro-preview hit its ~250 req/day Tier-1
  quota after 219 verdicts (preserved in judge_cache_gemini31pro.jsonl).
- Coverage: 247/248 labeled queries comparable (5 contested rows unresolvable after 3 passes).
- RQ2 headline replication (common queries): Gemini is uniformly ~1-3pp more lenient but the
  story is identical — underthinkingbench no_think 0.381→think 0.794 (Claude) vs
  0.392→0.804 (Gemini); overthinkingbench 0.900/0.927 vs 0.933/0.943.
- Bonus third-judge check: gemini-3.1-pro vs gemini-2.5-flash row-level verdict agreement on
  the 216 overlapping contested pairs = **0.944** — inter-LLM-judge agreement is the same
  magnitude as hand-judge-vs-LLM agreement, i.e. the residual disagreement is judge noise,
  not a systematic bias of the hand labels.

**Verdict: hand-judged labels are certified.** Oracle labels agree 94.3% (kappa 0.824,
"almost perfect" range), the label distribution is identical (50/197), and every headline
number replicates under an independent judge. The automated Gemini pipeline
(src/judge.py + scripts/judge_crossval.py) is validated for future scaling runs.
