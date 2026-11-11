# Data

**Nothing in this directory is redistributed with this repository.** Every corpus and trace below
belongs to a third party and is published under its own licence. `scripts/fetch_data.py` retrieves
each one from its own release:

```bash
python scripts/fetch_data.py --list    # what is expected, and what is already here
python scripts/fetch_data.py           # fetch everything missing
python scripts/fetch_data.py otb       # fetch one source
```

`data/` is gitignored apart from this file.

## You probably do not need any of it

The generation step is the only part of the work that reads these corpora, and its output is
committed. Model generations, judge decisions and oracle labels live under `results/paired/` and
`results/crossdataset/`, so the routers, the serving simulations and all twelve figures rebuild
from `results/` alone, with no GPU and no downloads.

Fetch these only to repeat the generation step itself, or to re-run the two experiments that read
the production arrival traces (`scripts/e11_real_trace.py`, `scripts/e11_addendum.py`).

## Sources

| Key | File | Source | Used for |
|---|---|---|---|
| `otb` | `otb/train.parquet` | [`facebook/optimal_thinking_bench`](https://huggingface.co/datasets/facebook/optimal_thinking_bench) | the paired corpus; every serving simulation |
| `mohseni` | `mohseni/test.parquet` | [`AmirMohseni/reasoning-router-data-v2`](https://huggingface.co/datasets/AmirMohseni/reasoning-router-data-v2) | realistic mixed traffic; the router evaluation |
| `gsm8k` | `gsm8k/test.parquet` | [`gsm8k`](https://huggingface.co/datasets/gsm8k), `main` config | transfer check where scoring is exact |
| `azure-conv` | `azure/AzureLLMInferenceTrace_conv.csv` | [Azure public dataset](https://github.com/Azure/AzurePublicDataset) | production arrival times, conversation service |
| `azure-code` | `azure/AzureLLMInferenceTrace_code.csv` | [Azure public dataset](https://github.com/Azure/AzurePublicDataset) | production arrival times, code completion |

Check each licence before redistributing any of these files. That is the reason they are fetched
rather than committed.

## A note on the Azure traces

Only the **arrival timestamps** are read. The queries, their token counts and their correctness all
come from the paired corpus. The traces supply
nothing but the shape of the arrival process, so a reproduction that cannot obtain them can still
run every experiment that uses the synthetic arrival regimes.

Scripts resolve the trace directory through `AZURE_TRACE_DIR`, defaulting to `data/azure`:

```bash
AZURE_TRACE_DIR=/somewhere/else python scripts/e11_real_trace.py
```
