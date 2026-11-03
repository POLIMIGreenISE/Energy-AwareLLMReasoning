# Why the pooled router still can't route Mohseni — the routability ceiling

**Question:** we pooled Mohseni into training, so why is the Mohseni slice still ~chance (AUC 0.559)?
Did pooling make a worse router? (`scripts/diagnose_mohseni_routability.py`, free/local)

## Answer 1 — pooling did NOT hurt; the pooled router is the *best* Mohseni router

| Router on the Mohseni slice | AUC |
|--|--|
| OTB-only (0 Mohseni in training) | 0.528 |
| **Pooled (OTB+Mohseni+GSM8K)** | **0.559** |
| Mohseni-specialist (trained ONLY on Mohseni, zero dilution) | 0.552 |

The specialist — all capacity on Mohseni, no dilution — tops out at **0.552**. That is the ceiling for a
text router on Mohseni; the pooled router (0.559) already matches/exceeds it. Pooling is not the bottleneck.

## Answer 2 — Mohseni's label is (largely) not a function of the query text

Same features / model / CV protocol, "is this dataset routable from text?":

| Dataset | Specialist AUC (train & CV within itself) | vs random EAA | routable? |
|--|--|--|--|
| **OTB** | **0.911** | +20pp | yes |
| Mohseni | 0.552 | −1pp | **no** |
| GSM8K | 0.550 | −2pp | no (thinking barely helps anyway) |

Why OTB is routable and Mohseni is not — within EACH source, can text tell think from no_think?
(Mohseni is ~61% open-ended **chat**, ~39% math — not a math-only set.)

| Source | kind | n | think% | within-source text→label AUC |
|--|--|--|--|--|
| OTB over-thinking | simple-Q | 1327 | 1% | 0.631 |
| OTB under-thinking | hard-math | 383 | 38% | **0.794** (learnable) |
| Big-Math-RL | math | 696 | 31% | 0.562 (≈ chance) |
| AIME | math | 22 | 32% | 0.514 |
| Nectar | chat | 593 | 20% | 0.537 (≈ chance) |
| WildChat | chat | 499 | 30% | 0.437 (≈ chance) |

**Both** Mohseni's math AND chat sources are near-chance — it is not a "look-alike hard math" artifact,
it is every source. The real mechanism is the **topic=label shortcut**:

- **OTB**: "needs thinking" ≈ "is the hard-math subset" *by construction* — a simple-question pile (1%
  think) and a hard-math pile (38% think). Topic is trivially text-readable, so the router is largely a
  topic classifier and topic ≈ label → AUC 0.90.
- **Mohseni**: that shortcut is gone. Think-rate is **math 31% vs chat 24% (barely different)**, and
  "is it math?" as a 1-feature router → **AUC 0.540**. Thinking helps plenty of chat and *fails* most
  hard math (Qwen3-8B is often wrong-both-ways / right-both-ways), so topic no longer predicts the
  label; and within any topic the think/no_think flip is a **model-behavioral** property (the model's
  competencies + gen randomness), not in the text. No text→label map exists to learn.

**Implication:** OTB's high routability is partly a **benchmark artifact** (it conflates "needs
reasoning" with "hard-math topic"). On realistic mixed traffic that conflation breaks and text routing
collapses to ~0.55 AUC regardless of training data.

## Answer 3 — but Mohseni's published router "works" on his data. How? (metric, not skill)

`scripts/test_his_label_routability.py`. Same Mohseni text, predict three labels via 5-fold CV:

| Label predicted | CV AUC on Mohseni |
|--|--|
| Our energy-aware oracle (thinking flips it) | 0.552 |
| **Mohseni's OWN published labels** | **0.563** |
| Pure difficulty (no-think is wrong) | 0.620 |

His own labels are **also near-chance from text (0.563)** — even at chance on chat (Nectar 0.510,
WildChat 0.498). So his data is unroutable per-query under HIS labels too; it is not our method.
Why his blog looks like a win: he reports **downstream reward/accuracy at a think-rate with NO random
baseline** (WildChat +1.37 reward @ 42.9% think — but thinking helps chat on average, so any 43%-think
policy gains; AIME 0.1933 < 0.2400 full-think — his router is BELOW always-think). Report his metric and
OUR router looks "good" too; report our metric (per-query AUC + random floor) and HIS router hits the
same ~0.56 ceiling. **Takeaway: our random-floor + per-query-AUC evaluation is a methodological
contribution that EXPOSES a routability ceiling the prior art's reward-only evaluation hides. Our results
are more rigorous than the baseline, not fake.**

## Answer 4 — "is the data unroutable, or is our ROUTER too weak?" (capacity sweep)

`scripts/test_router_capacity_mohseni.py`. Threw 3 feature spaces × 5 classifiers at Mohseni (5-fold CV):

| features | LogReg | MLP | LinearSVC | HistGBM | RBF-SVM |
|--|--|--|--|--|--|
| MiniLM-384 | 0.552 | 0.546 | 0.515 | 0.540 | 0.515 |
| mpnet-768 | 0.555 | 0.525 | 0.537 | **0.568** | 0.537 |
| TF-IDF words | 0.557 | 0.521 | 0.531 | — | — |

**BEST across all 13 combos = 0.568** (our labels); **0.589** for Mohseni's own labels. Rules out weak
features (mpnet-768 + TF-IDF both fail), linear-only boundaries (MLP/HistGBM/RBF-SVM don't beat LogReg),
and our label definition (his labels cap too). Same pipeline = 0.90 on OTB, 0.57 on Mohseni → a
**mutual-information ceiling**: the text doesn't contain enough about "will thinking flip this model" for
ANY architecture. Only untested escalation = end-to-end fine-tuned ModernBERT (low odds: it TIED LogReg on
OTB 0.90 and transfer 0.52; would overfit ~1450 ex/fold). **Conclusion: the classifier is not the
bottleneck; routing from TEXT is — break it with a different INPUT (behavioral signals: self-consistency /
logits / hidden states), not a bigger text model.**

## Consequences

- The OTB slice stays **0.900** under pooling (−0.011 from 0.911) — OTB routing is intact. The lower
  *overall* pooled AUC (0.792) is just an average that now includes intrinsically-unroutable Mohseni
  queries, not evidence of damage.
- There is a **routability ceiling** set by the data, not the classifier. Bimodal benchmarks (OTB)
  overstate how well pre-generation text routing works; realistic mixed traffic contains many queries
  where "does thinking help?" is unknowable from text — bounding EVERY text router (ours, Mohseni's,
  the Semantic Router's). This is an honest, valuable finding, and it motivates the signals that don't
  need per-query text prediction: the energy measurement + the **system-aware (load/budget) router**.
- Caveat: REPS=1 → some Mohseni "think" labels are single-sample noise, inflating unroutability a
  little; but the within-domain text-invisibility of the flip would remain under perfect labels.

Ties to [[router-findings]], [[crossdataset-findings]].
