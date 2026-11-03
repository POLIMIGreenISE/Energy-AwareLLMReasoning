# Beyond Token Counts: Energy-Aware Reasoning Allocation in Batched LLM Serving

Source code, measurements and figure provenance for the MSc thesis of the same title.

Politecnico di Milano, School of Industrial and Information Engineering.
MSc in Computer Science and Engineering. Author: Cemal Gunduz. Advisor: Prof. Monica Vitali.

This repository is the reference target for every path cited in the thesis. **Appendix B,
"Reproducing the Results", is the authoritative map**; this file is its index.

## What the work measures

The energy cost of one output token falls by a factor of 42 between one request in service and
sixty-four, while board power rises only 14 percent. Accounting for reasoning in tokens is
therefore load-blind. The thesis measures that curve, tests whether the queries worth reasoning on
can be identified in advance, and evaluates an allocation rule that costs each decision at the
batch size the server is actually running.

## Layout

| Directory | Contents |
|---|---|
| `src/` | The router component and the serving simulator |
| `tests/` | Unit tests, including the simulator-equivalence regression test |
| `scripts/` | Every experiment and every figure-drawing script |
| `notebooks/` | The GPU measurement notebooks, in the order they were run |
| `results/` | All committed measurements. Every number in the thesis comes from here |
| `data/` | Third-party corpora and traces. Not redistributed; see `data/README.md` |

## Reproducing without a GPU

Everything except the generation and serving measurements runs on CPU in minutes, because the
measurements themselves are committed under `results/`.

```bash
python -m pip install -r requirements.txt      # or: numpy scipy scikit-learn matplotlib pyarrow
python scripts/verify_pipeline.py              # invariant suite, seconds, no GPU
python -m pytest tests/                        # includes the simulator-equivalence test
```

`scripts/verify_pipeline.py` asserts four groups of invariants: that the measured hardware curves
are internally consistent (energy per token equals power divided by throughput), that the corpus
token counts round-trip, that the simulator agrees with the corpus at both extremes, and that
single-run outputs are physically sane. Each check prints the numbers behind its verdict.

The regression test in `tests/test_router.py` asserts that the shipped router reaches
bit-identical decisions to the simulator over 300 queries at six batch sizes. That is what makes
the live comparison in Chapter 8 a test of the cost model rather than a test of whether two
implementations of one formula agree.

## Figure provenance

Every figure is redrawn by a committed script from a committed results file, so a figure and the
numbers quoted beside it in the text cannot drift apart. Paths are relative to `results/`.

| Figure | Drawn by | Reads |
|---|---|---|
| 4.1 the experimental apparatus | `scripts/fig_apparatus.py` | (schematic) |
| 4.2 energy-adjusted accuracy | `scripts/fig_eaa.py` | `paired/paired_oracle_final.json`, `paired/router_fte_modernbert.jsonl` |
| 5.1 the energy curve | `scripts/fig_ch3.py` | `batching/concurrency.jsonl` |
| 5.2 cross-GPU | `scripts/fig_ch3.py` | `batching/concurrency_nvidia_l4.jsonl` |
| 6.1 benefit against cost | `scripts/fig_ch6_ch8.py` | `crossdataset/official_router_eval.md`, `e10/dtok_pred.json` |
| 7.1 budget adherence | `scripts/fig_ch5.py` | `e2/e2_transfer_tokens.json` |
| 7.2 rule against threshold | `scripts/fig_ch5.py` | `e8/e8_rows.json` |
| 7.3 ingredients separated | `scripts/fig_ch5.py` | `e8/e8_decomp_caps.json` |
| 7.4 under a latency target | `scripts/fig_ch5.py` | `e8/e8_rows.json` |
| 7.5 production traces | `scripts/fig_ch5.py` | `e11/e11_rows.json` |
| 8.1 simulated against measured | `scripts/fig_ch6_ch8.py` | `livetest/sim_vs_live.jsonl` |
| 8.2 the budget controller | `scripts/fig_ch6_ch8.py` | `repair/pacing_*.jsonl` |

Each table in the thesis carries a comment in the LaTeX source naming the file it was read from,
in the same form as the paths above.

## Results by chapter

| Chapter | Claim | Reads |
|---|---|---|
| 5. The Energy Cost of Reasoning | C1 | `results/batching/` |
| 6. The Limits of Per-Query Routing | C2, C3 | `results/paired/`, `results/crossdataset/`, `results/pooled/`, `results/e10/` |
| 7. Allocating Reasoning Under an Energy Budget | C4 | `results/e2/`, `results/e8/`, `results/e11/` |
| 8. Implementation and Live Validation | C5 | `results/livetest/`, `results/closedloop/`, `results/repair/` |

`results/CORRECTIONS.md` is the corrections log: every figure superseded during the work, with the
reason. Appendix A of the thesis is drawn from it.

## Measurement environment

The serving measurements and the live validation ran `Qwen/Qwen3-8B` in `bfloat16` through
vLLM 0.10.2, with a context limit of 16384 tokens, metered with the Zeus toolkit, on an
A100-SXM4-40GB and on an NVIDIA L4. The launch arguments and the vLLM version of every reported
run are recorded in the committed server logs under `results/*/server*.log`.

Everything else runs with Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and
Matplotlib 3.11. The router component in `src/router.py` depends on the standard library and NumPy
alone, so it imports and runs its tests with no server attached.
