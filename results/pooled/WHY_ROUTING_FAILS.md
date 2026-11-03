# Why text routing works on OTB and fails on Mohseni — the mechanism, decomposed

Answers three questions directly: (1) is OTB's 0.90 AUC real difficulty signal or topic routing?
(2) does the router send all math to "think"? (3) did we make a mistake?
Scripts: `decompose_otb_routability.py`, `diagnose_transfer_inversion.py`, `test_router_capacity_mohseni.py`.

---

## 1. OTB's composition — and a real DATA GAP we should fix

| subset | n | oracle think% | content |
|--|--|--|--|
| overthinkingbench | 1327 | **2.0%** | 72 domains (MMLU-Pro-like), types: mcq 298 / open-ended 355 / **numeric 312** / open-ended-long 362 |
| underthinkingbench | 383 | **45.7%** | **one homogeneous block** — Reasoning-Gym puzzles, no domain metadata, 56 templates |

**The HF release of OTB EXCLUDES the math subsets** (the full set needs the repo's `otbench prepare`).
So our think-half contains **zero hard math** — it is puzzles only. Our easy math (312 `numeric`
items) lives entirely in the *no-think* half. **This is a genuine data limitation, and it is fixable.**

## 2. OTB's routability is topic/template detection — at EVERY level

| router | AUC |
|--|--|
| **"is it the under-half?" — ONE bit** | **0.864** |
| "which domain?" lookup table | 0.857 |
| "which answer-type?" lookup table | 0.858 |
| **FULL text router (MiniLM+LogReg)** | **0.895** |

The full text router adds only **+0.03** over knowing which pile the question came from. And once
topic is held constant, the signal disappears:

| slice | text→label AUC |
|--|--|
| within over-half (n=1327) | **0.512** (chance) |
| within under-half (n=383) | 0.759 |
| ↳ but "which puzzle TEMPLATE?" lookup alone | 0.724 |
| ↳ **within a single puzzle template** (n=29–50 each) | **mean 0.491** (chance) |

*Caveat: the per-template slices are small (n=29–50, AUC SE ≈ 0.08), so each is individually
underpowered — but all four land at 0.41–0.61 with none above chance meaningfully.*

**Unified finding (holds on BOTH datasets): text routing = topic/template detection. Within any fixed
topic, text→think/no-think is at chance — OTB 0.49, Mohseni 0.51–0.52.**

## 3. "Is math all routed to think?" — NO, the opposite

| OTB slice | n | oracle think% | router p(think) |
|--|--|--|--|
| over-half **numeric (easy math)** | 312 | 3.2% | **0.057** |
| over-half non-math easy | 1015 | 1.7% | 0.029 |
| under-half (reasoning puzzles) | 383 | 45.7% | 0.404 |

At τ=0.3 and τ=0.5, **0% of OTB's easy math is routed to think** (vs 61% / 33% of hard puzzles).
So easy math is routed *correctly*. But because every math item in our OTB is easy, the router
learns **"math/numeric → NO think"** — it never sees hard math at all.

## 4. Why Mohseni fails (mechanism)

OTB-trained router applied to Mohseni:

| source | n | ORACLE think% | router p(think) | AUC |
|--|--|--|--|--|
| AIME | 22 | 31.8% | 0.173 | 0.638 |
| Big-Math-RL | 696 | 31.0% | 0.148 | 0.516 |
| Nectar (chat) | 593 | 20.1% | 0.129 | 0.522 |
| WildChat (chat) | 499 | 29.7% | 0.116 | 0.512 |

The router learned **"is it a Reasoning-Gym puzzle?"**. Mohseni contains no puzzles, so *everything*
gets a low score (0.12–0.17) and within-source discrimination is nil (0.51–0.52).

**Hypothesis tested and REJECTED:** I predicted an "easy-math trap" — that OTB would teach
math→no-think and the router would *invert* on hard math. It does not invert: it ranks math (0.149)
above chat (0.123), matching the oracle's direction (31% vs 24%). The failure is not inversion, it is
**near-absence of signal off the trained topics**, with the correct direction but far too weak a tilt.

## 5. Is the router just too weak? No — capacity sweep says the data is the limit

3 feature spaces × 5 classifiers on Mohseni (5-fold CV): **best AUC 0.568** (our labels), **0.589**
(Mohseni's own labels). MiniLM/mpnet-768/TF-IDF × LogReg/MLP/LinearSVC/HistGBM/RBF-SVM — none clear
0.60. Same pipeline scores 0.90 on OTB. It is a **mutual-information ceiling**, not a model choice.

## 6. Did we make a mistake?

- **Method: no.** Labels, energy basis, CV protocol, and the random-floor metric all hold up.
- **Data: yes, one real gap.** We trained on the HF release, whose think-half has **no math**. Fixing
  it (full OTB via `otbench prepare`) is cheap and is also the cleanest *constructive* test of the
  topic mechanism: if text routing is topic-matching, adding hard math to training should lift
  Big-Math transfer specifically.
- **Interpretation: sharpened.** "Our router is good" (OTB 0.90) was really "OTB aligns topic with
  difficulty". That is true of the benchmarks the whole field uses.
