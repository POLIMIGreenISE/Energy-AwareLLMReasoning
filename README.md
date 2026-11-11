# Energy-Aware Reasoning Allocation in Batched LLM Serving

Code, measurements and analyses for deciding when a hybrid-reasoning language model should reason,
when requests are served in batches and the service has an energy budget.

Hybrid-reasoning models such as Qwen3 can answer a request directly or first generate a long chain of
intermediate steps. Methods that make this choice usually count the cost of reasoning in generated
tokens. On a batched inference server the energy of a token is not constant: it depends on how many
requests share the GPU when the token is produced. This repository measures that dependence, tests
whether the requests that benefit from reasoning can be identified in advance, and evaluates an
allocation rule that costs each decision at the batch size the server is running.

## Main results

- **The energy of a token depends on the batch size.** With Qwen3-8B served by vLLM on an A100, the
  energy of one output token falls from 3.42 J at batch size 1 to 0.082 J at batch size 64, a factor
  of 42, while board power rises only from 232 W to 265 W. An NVIDIA L4 gives the same normalized
  curve, with a mean difference of 4.6%.
- **The benefit of reasoning is not predictable from the prompt, but its token cost is.** On realistic
  mixed traffic, no router tested, including two published ones, is statistically distinguishable
  from a random-selection control that spends the same energy. The number of extra tokens reasoning
  will generate is predictable: rank correlation 0.761, and within a factor of two for 92% of queries.
- **Costing each decision at the live batch size holds an energy budget.** In simulation, the rule
  `reason(q)  iff  s(q) > λ · J(L) · Δtok(q)` stays within 4.7% of its budget across four traffic
  regimes without recalibration, which no fixed energy per token achieves. Its accuracy gain over a
  static threshold appears only when the load varies for reasons outside the policy, as in bursty
  traffic, and disappears under a tight tail-latency target.
- **On a live server** the router reproduces the simulated decisions, and a guarded closed-loop
  controller holds the budget to within 6.6% until the server saturates. At saturation vLLM preempts
  and recomputes requests, and the energy model under-predicts by 43%.

## Research questions

The notes under `results/` refer to four research questions:

| | Question |
|---|---|
| **RQ1** | Does the energy of a generated token depend on the load of the server, and is the token count a good proxy for energy under batched serving? |
| **RQ2** | Does reasoning always improve answer quality, and how does its energy cost compare with the quality it adds? |
| **RQ3** | Can the benefit of reasoning on a request be predicted from its prompt? |
| **RQ4** | With an energy budget fixed in advance and a changing workload, how should reasoning be allocated across requests? |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Reusable components: the deployable router (`router.py`), the trace-driven serving simulator (`serving_sim.py`), answer extraction and verification (`scoring.py`), the LLM judge (`judge.py`), and grid refinement at a budget boundary (`capfit.py`) |
| `tests/` | Unit tests, including a regression test that the router and the simulator reach identical decisions |
| `scripts/` | Experiments (`e*_*.py`), analyses (`analyze_*`, `audit_*`, `diagnose_*`, `eval_*`), figures (`fig_*`, `plot_*`, `make_ceiling_plot.py`), notebook generators (`build_*_notebook.py`), data retrieval (`fetch_data.py`) and invariant checks (`verify_pipeline.py`) |
| `notebooks/` | The GPU notebooks that produced the measurements, numbered in the order they were run |
| `results/` | Every committed measurement, with the notes written while analysing it |
| `figures/` | The main figures, redrawn from `results/` by the `fig_*` scripts |
| `data/` | Third-party corpora and traces, fetched on demand and not redistributed (see `data/README.md`) |

## Quick start (no GPU)

The measurements are committed, so everything except generation and serving runs on a CPU in minutes.

```bash
python -m pip install -r requirements.txt
python scripts/verify_pipeline.py     # invariant checks, a few seconds
python -m pytest tests/               # unit and regression tests
```

`scripts/verify_pipeline.py` checks that the measured hardware curves are internally consistent
(energy per token equals power divided by throughput), that corpus token counts round-trip, that the
simulator matches the corpus at the never-reason and always-reason extremes, and that single runs are
physically sane. Each check prints the numbers behind its verdict.

`tests/test_router.py` checks, among other things, that the deployable router in `src/router.py`
reaches bit-identical decisions to the simulator over 300 queries at six batch sizes, so a live run
tests the cost model and not two implementations of the same formula.

## Figures

Each figure script reads committed results and writes PNG files to `figures/`. Set `FIG_DIR` to write
them elsewhere.

```bash
python scripts/fig_energy_curves.py
python scripts/fig_routing_and_deployment.py
python scripts/fig_budget_allocation.py
python scripts/fig_eaa.py
python scripts/fig_apparatus.py
```

| Figure | Script | Reads (under `results/`) |
|---|---|---|
| `apparatus.png`: what is measured and what is simulated | `fig_apparatus.py` | (schematic) |
| `eaa_metric.png`: the energy-adjusted accuracy metric | `fig_eaa.py` | `paired/paired_oracle_final.json`, `paired/router_fte_modernbert.jsonl` |
| `jl_curve.png`: energy, power and throughput against batch size | `fig_energy_curves.py` | `batching/concurrency.jsonl` |
| `cross_gpu.png`: the same curve on a second GPU | `fig_energy_curves.py` | `batching/concurrency_nvidia_l4.jsonl` |
| `asymmetry.png`: benefit not predictable, token cost predictable | `fig_routing_and_deployment.py` | `crossdataset/official_router_eval.md`, `e10/dtok_pred.json` |
| `e2_sweep.png`: holding an energy budget | `fig_budget_allocation.py` | `e2/e2_transfer_tokens.json` |
| `e8_gap.png`: load-aware rule against a static threshold | `fig_budget_allocation.py` | `e8/e8_rows.json` |
| `e8_decomp.png`: which ingredient the gain comes from | `fig_budget_allocation.py` | `e8/e8_decomp_caps.json` |
| `e8_slo.png`: under a tail-latency target | `fig_budget_allocation.py` | `e8/e8_rows.json` |
| `e11_real.png`: production arrival traces | `fig_budget_allocation.py` | `e11/e11_rows.json` |
| `live_energy.png`: simulated against measured energy on a live server | `fig_routing_and_deployment.py` | `livetest/sim_vs_live.jsonl` |
| `controller.png`: the closed-loop budget controller | `fig_routing_and_deployment.py` | `repair/pacing_*.jsonl` |

## Experiments

| ID | Question | Code | Results |
|---|---|---|---|
| | Energy of reasoning and of batching on one GPU | `notebooks/00_pilot_energy_measurement.ipynb`, `notebooks/01_batching_divergence.ipynb` | `results/pilot/`, `results/batching/` |
| | Paired generation: every query answered directly and with reasoning, then graded | `notebooks/02_paired_generation_scored.ipynb` | `results/paired/` |
| | Text routers, probes and published routers, on benchmark and realistic traffic | `notebooks/03`–`08`, `scripts/analyze_*.py`, `scripts/eval_*.py` | `results/router/`, `results/crossdataset/`, `results/pooled/`, `results/system_router/`, `results/draft_router/` |
| | Validity checks: bootstrap intervals, judge agreement, leakage, fairness | `scripts/bootstrap_cis.py`, `scripts/judge_crossval.py`, `scripts/audit_*.py` | `results/audit/` |
| E1 | Allocation policies under an energy quota | `scripts/e1_energy_quota_controller.py` | `results/e1/` |
| E2 | Whether a token-count cost model can hold an energy budget | `scripts/e2_token_proxy_controller.py`, `scripts/e2_transfer_tokens.py` | `results/e2/` |
| E3 | The energy, latency and accuracy trade-off | `scripts/e3_energy_latency_accuracy.py` | `results/e3/` |
| E4 | Arrival rate and burstiness | `scripts/e4_arrival_sweep.py` | `results/e4/` |
| E5 | Sensitivity to the simulator's assumptions | `scripts/e5_sensitivity.py` | `results/e5/` |
| E6, E7 | The simulator against a real vLLM server | `notebooks/09_serving_validation.ipynb` | `results/validation/` |
| E8 | Load-aware rule against a densely swept static threshold, at matched energy | `scripts/e8_*.py` | `results/e8/` |
| E9 | Live matched-energy comparison at three load levels | `notebooks/10_live_router_test.ipynb` | `results/livetest/` |
| E10 | Predicting the token cost of reasoning, and its transfer across datasets | `scripts/e10_dtok_predictability.py`, `scripts/e10b_dtok_transfer.py` | `results/e10/` |
| E11 | Production arrival traces from Azure | `scripts/e11_real_trace.py`, `scripts/e11_addendum.py`, `scripts/e11_slo.py` | `results/e11/` |
| E12, E13 | Closing the budget loop on a real energy meter, and behavior at saturation | `notebooks/11_closed_loop_budget.ipynb` | `results/closedloop/` |
| E14 | The energy curve on a second GPU | `notebooks/12_cross_gpu_curve.ipynb` | `results/batching/` |
| E15, E16 | Repairing the closed-loop controller | `scripts/e15_controller_fix.py`, `notebooks/13_controller_repair.ipynb` | `results/e15/`, `results/repair/` |

The notes call the random-selection control the *random floor*. `results/CORRECTIONS.md` records
every result that was revised during the work, what changed, and which file is authoritative
afterwards.

## Measurement environment

The serving measurements and the live validation ran `Qwen/Qwen3-8B` in `bfloat16` through
vLLM 0.10.2, with a context limit of 16384 tokens, metered with the Zeus toolkit, on an
A100-SXM4-40GB and on an NVIDIA L4. The launch arguments and the vLLM version of every reported run
are recorded in the committed server logs under `results/*/server*.log`.

Everything else runs with Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and Matplotlib 3.11.
The router in `src/router.py` depends on the standard library and NumPy alone, so it imports and runs
its tests with no server attached.

The notebooks ran on Colab and on rented GPU hosts. Their first cell clones this repository and
needs a GitHub token only to push results back. Set your own git identity in that cell before
committing from a notebook.

## Credits

Developed by Cemal Gunduz at Politecnico di Milano, under the supervision of Monica Vitali.
