"""Builds notebooks/02_paired_generation_scored.ipynb (Phase-1, SCALED).

v3 — FULL OptimalThinkingBench (2026-07-15, post-Vitali scale-up):
  - N_PER_SUBSET = None -> the whole of each OTB half (HF release: 1,327 over /
    550 under = 1,877 rows; the paper's full 2,050 was never fully released).
    Same SEED/shuffle as v2, so the first 150 qids per subset are IDENTICAL to
    the existing run and their generations on disk are reused via resume.
  - Provider-agnostic setup (Colab or vast.ai): env-token flow + non-destructive
    install, as in notebook 05.
  - Judging is NO LONGER done in the notebook: the certified Gemini pipeline
    (kappa 0.824 vs hand labels) runs locally via scripts/build_labels_v3.py.
    The GPU session does generation only, then pushes gen_v2.jsonl.
  - Energy per query still reported two ways: amortized batch share, and the
    batch-1-equivalent 3.7 J/output-token line (pilot R^2=0.9999).

Run:  python scripts/build_paired_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Phase-1 v3 — Paired think/no-think generation, FULL OptimalThinkingBench

Generates the router's training data at full scale: **every** OTB query
(~1,460 overthinking + ~610 underthinking) answered by Qwen3-8B in **both**
modes, energy recorded per query. Same seed as the 283-query run, so all
existing generations are **reused** — only new (qid, mode, rep) triples run.

**Judging happens locally, not here** — the certified Gemini pipeline
(kappa 0.824 agreement with hand labels) runs on the laptop from the pushed
`gen_v2.jsonl`. This GPU session = generation + push only.

**Speed trick (from our own batching result):** generation is **batched** through
vLLM. Per-query energy is reported as (a) the amortized batch share and (b) a
**batch-1-equivalent** estimate `completion_tokens × 3.7 J/token` — valid because
the pilot measured R²=0.9999 between tokens and energy at batch=1.

> Expected full-run cost: ~7,200 new generations ≈ **1.5–2.5 A100-hours**.
> Resumable — a crash never loses completed work.''')

# ---------------------------------------------------------------- section 0
md(r'''## 0. Repo setup (Colab or vast.ai)

Token lookup order: Colab secret `GITHUB_TOKEN` → env var `GITHUB_TOKEN` → prompt.''')
code(r'''import os, subprocess, sys
REPO, BRANCH = "cemalgndzz/llm-routing-tesina", "main"
TOKEN = None
try:
    from google.colab import userdata; TOKEN = userdata.get("GITHUB_TOKEN")
except Exception:
    TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    from getpass import getpass
    TOKEN = getpass("GitHub personal access token (repo scope): ").strip()

repo_dir = os.path.abspath(REPO.split("/")[1])
auth = f"https://{TOKEN}@github.com/{REPO}.git"
if not os.path.isdir(os.path.join(repo_dir, ".git")):
    subprocess.run(["git", "clone", "--branch", BRANCH, auth, repo_dir], check=True)
else:
    subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
os.chdir(repo_dir)
subprocess.run(["git", "remote", "set-url", "origin", auth], check=True)
subprocess.run(["git", "config", "user.name", "cemalgndzz"], check=True)
subprocess.run(["git", "config", "user.email", "cgogit@gmail.com"], check=True)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd(), "| src importable:", os.path.isdir("src"))''')

md(r'''## 1. Install (non-destructive — never downgrades a working stack)

- vLLM importable → leave everything alone.
- vLLM broken (mixed-version `ImportError`) → set `REPAIR_VLLM = True`, run once,
  then **restart the kernel** and re-run from Section 0.
- No vLLM → install latest unpinned.

**Known-good vast.ai recipe (2026-07-19, CUDA-12.6 driver, verified working):**
if the preinstalled torch was built for CUDA 13 (`driver too old (found 12060)`),
or Ray crashes on `libamd_smi.so`, or the tokenizer lacks
`all_special_tokens_extended` — run this ONCE, then restart the kernel:
```
%pip uninstall -y amdsmi vllm vllm-flash-attn xformers flashinfer-python torch torchvision torchaudio
%pip install "vllm>=0.10,<0.11" "transformers>=4.51,<5"
%pip uninstall -y amdsmi ray torchcodec   # vllm re-pulls broken amdsmi; removing ray kills the AMD-probe crash path (single-GPU needs no ray)
```''')
code(r'''import importlib, subprocess, sys
REPAIR_VLLM = False   # set True ONCE if vllm import errors, then restart kernel

def pipq(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    print(("ok  : " if r.returncode == 0 else "FAIL: ") + " ".join(args))
    if r.returncode != 0: print(r.stderr[-400:])

pipq("-U", "zeus-ml", "datasets", "pynvml", "accelerate")

def importable(mod):
    try:
        importlib.import_module(mod); return True
    except Exception as e:
        print(f"{mod}: broken/absent ({repr(e)[:120]})"); return False

vllm_ok = importable("vllm")
if REPAIR_VLLM or not vllm_ok:
    if REPAIR_VLLM:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                        "vllm", "vllm-flash-attn", "xformers"], capture_output=True)
    pipq("-U", "--no-cache-dir", "vllm")
    print(">>> vLLM (re)installed: Kernel > Restart, then re-run from Section 0.")
if not importable("transformers"):
    pipq("transformers>=4.51")
print("install done")''')

md(r'''## 2. Config''')
code(r'''import os
SMOKE_TEST = True
MODEL = "Qwen/Qwen3-8B"
N_PER_SUBSET = 5 if SMOKE_TEST else None     # None = the WHOLE subset (v3 full-OTB run)
REPS = 1 if SMOKE_TEST else 2                 # repeats per (query, mode)
THINK_MAX_TOKENS   = 2048 if SMOKE_TEST else 16384
NOTHINK_MAX_TOKENS = 512  if SMOKE_TEST else 2048
MAX_MODEL_LEN = 20480
GEN_BATCH = 64                                # concurrency per vLLM batch
J_PER_OUT_TOKEN = 3.7                         # measured batch=1 J/token (pilot, A100)
SEED = 12345                                  # MUST stay 12345: qid stability + resume

OUTDIR = "results/paired"; os.makedirs(OUTDIR, exist_ok=True)
GEN_JSONL = os.path.join(OUTDIR, "gen_v2.jsonl")          # raw generations (append)
SAMPLING = {"think":    dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=THINK_MAX_TOKENS),
            "no_think": dict(temperature=0.7, top_p=0.80, top_k=20, max_tokens=NOTHINK_MAX_TOKENS)}
print(f"SMOKE_TEST={SMOKE_TEST} | N_PER_SUBSET={N_PER_SUBSET} (None=all) | REPS={REPS}")''')

md(r'''## 3. Load OTB sample (seed-stable superset — first 150/subset match the v2 run)

`N_PER_SUBSET=None` takes the whole shuffled subset. Because the seed and shuffle
are unchanged, every previously-run qid keeps its number and its rows on disk
are skipped by the resume logic — the run only pays for NEW queries.''')
code(r'''import json, random
from datasets import load_dataset
random.seed(SEED)

ds = load_dataset("facebook/optimal_thinking_bench")["train"]
def meta(r):
    m = r.get("metadata")
    if isinstance(m, dict): return m
    try: return json.loads(m)
    except Exception: return {}
rows = [{"question": r["question"], "answer": r.get("answer",""), "subset": r["subset"],
         "answer_type": str(meta(r).get("answer_type","unknown"))} for r in ds]
over  = [r for r in rows if r["subset"]=="overthinkingbench"]
under = [r for r in rows if r["subset"]=="underthinkingbench"]
random.shuffle(over); random.shuffle(under)   # same seed => first 40 match the old run

sample = []
for i, r in enumerate(over[:N_PER_SUBSET]):
    sample.append({**r, "qid": f"over_{i:03d}",  "route_label": "no_think"})
for i, r in enumerate(under[:N_PER_SUBSET]):
    sample.append({**r, "qid": f"under_{i:03d}", "route_label": "think"})
sample = [s for s in sample if str(s["answer"]).strip() not in ("", "None", "nan")]
from collections import Counter
print("usable queries:", len(sample), dict(Counter(s["subset"] for s in sample)))''')

md(r'''## 4. Serving + energy''')
code(r'''import time, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=MAX_MODEL_LEN, enforce_eager=False)

zeus_ok = False
try:
    from zeus.monitor import ZeusMonitor
    zmon = ZeusMonitor(gpu_indices=[0]); zeus_ok = True
except Exception as e:
    print("Zeus unavailable:", repr(e)[:120])

def measure(fn):
    if zeus_ok:
        zmon.begin_window("b"); r = fn(); m = zmon.end_window("b")
        ej = getattr(m, "total_energy", None) or sum(m.gpu_energy.values())
        return r, ej, m.time
    t0 = time.time(); r = fn(); return r, float("nan"), time.time() - t0

def build_prompt(q, think):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=think)

def split_think(text):
    if "</think>" in text:
        t, _, a = text.partition("</think>")
        return t.replace("<think>", "").strip(), a.strip()
    return "", text.strip()

def generate_batch(queries, mode):
    """Batched generation for a list of query dicts; returns list of row dicts."""
    sp = SAMPLING[mode]
    prompts = [build_prompt(q["question"], mode == "think") for q in queries]
    params = SamplingParams(temperature=sp["temperature"], top_p=sp["top_p"],
                            top_k=sp["top_k"], max_tokens=sp["max_tokens"])
    outs, energy_j, secs = measure(lambda: llm.generate(prompts, params, use_tqdm=True))
    total_tok = sum(len(o.outputs[0].token_ids) for o in outs) or 1
    rows = []
    for q, o in zip(queries, outs):
        text = o.outputs[0].text
        think_text, answer_text = split_think(text)
        ct = len(o.outputs[0].token_ids)
        rows.append({
            "qid": q["qid"], "subset": q["subset"], "route_label": q["route_label"],
            "answer_type": q["answer_type"], "question": q["question"],
            "gold": str(q["answer"])[:800], "mode": mode,
            "completion_tokens": ct,
            "think_tokens": len(tok(think_text, add_special_tokens=False)["input_ids"]) if think_text else 0,
            "finish_reason": o.outputs[0].finish_reason,
            "answer_text": answer_text[:3000],
            "energy_share_wh": (energy_j * ct / total_tok) / 3600.0,   # amortized batch share
            "energy_b1eq_wh": ct * J_PER_OUT_TOKEN / 3600.0,           # batch-1 equivalent
        })
    print(f"  [{mode}] {len(rows)} gens | {secs:.0f}s | batch {energy_j/3600:.1f} Wh")
    return rows

_ = generate_batch(sample[:2], "no_think")  # warm-up (discarded)
print("Warm-up done.")''')

md(r'''## 5. Generate (batched, resumable at (qid, mode, rep) level)''')
code(r'''done = set()
if os.path.exists(GEN_JSONL):
    for l in open(GEN_JSONL):
        try:
            d = json.loads(l); done.add((d["qid"], d["mode"], d["rep"]))
        except Exception: pass
print("already on disk:", len(done))

with open(GEN_JSONL, "a") as f:
    for mode in ["no_think", "think"]:
        for rep in range(REPS):
            pending = [q for q in sample if (q["qid"], mode, rep) not in done]
            print(f"mode={mode} rep={rep}: {len(pending)} pending")
            for i in range(0, len(pending), GEN_BATCH):
                chunk = pending[i:i+GEN_BATCH]
                for row in generate_batch(chunk, mode):
                    row["rep"] = rep
                    f.write(json.dumps(row) + "\n")
                f.flush()
print("Generation complete.")''')

md(r'''## 6. Quick stats + push `gen_v2.jsonl`

Labeling is NOT done here — after the push, the certified Gemini judge pipeline
runs locally: `python scripts/build_labels_v3.py` (fast-path + Gemini 2.5-flash,
kappa 0.824 vs hand labels, dedup cache so re-runs are nearly free).''')
code(r'''import statistics as st
from collections import Counter
gens = [json.loads(l) for l in open(GEN_JSONL) if l.strip()]
qids = {r["qid"] for r in gens}
print(f"generations on disk: {len(gens)} | unique qids: {len(qids)}")
for mode in ["no_think", "think"]:
    g = [r for r in gens if r["mode"] == mode]
    if g:
        trunc = sum(1 for r in g if str(r.get("finish_reason")).lower() == "length")
        print(f"  {mode:9} n={len(g):5}  mean tokens {st.mean(r['completion_tokens'] for r in g):7.0f}"
              f"  truncated {trunc/len(g)*100:4.1f}%")
print("subset coverage:", dict(Counter(r["subset"] for r in gens)))''')
code(r'''import subprocess
# *.jsonl is gitignored -> force-add the results we need locally
subprocess.run(["git", "add", "-f", GEN_JSONL], check=True)
r = subprocess.run(["git", "commit", "-m", "gen_v2 full-OTB scale-up (v3 run)"],
                   capture_output=True, text=True)
print(r.stdout[-300:] or r.stderr[-300:])
r = subprocess.run(["git", "push"], capture_output=True, text=True)
print("push:", (r.stdout + r.stderr)[-300:])''')

md(r'''## 7. Sanity anchors (check before ending the session)

- think-mode truncation should stay **< ~15%** at the 16k budget;
- no_think mean tokens ≈ **500–600**, think ≈ **2,000–3,000**;
- unique qids ≈ **~1,850** (HF release = 1,877 rows: 1,327 over + 550 under,
  minus empty-gold rows);
- after the local labeling run, subset accuracies should land near the certified
  248-q anchors: overthinking no_think ≈0.90 / think ≈0.93, underthinking
  no_think ≈0.38 / think ≈0.80. Large deviations → spot-check verdicts.''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]; c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "02_paired_generation_scored.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
