"""Builds notebooks/01_batching_divergence.ipynb (Phase-1 core experiment).

Tests whether the token->energy proxy (R^2=1.0 at batch=1 in the pilot) breaks
under realistic batched serving and prefill-heavy prompts.

Run:  python scripts/build_batching_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

# ------------------------------------------------------------------ title
md(r'''# Phase-1 — Does the token→energy proxy break under batched serving?

**Why.** The pilot found energy ∝ output tokens with **R² = 0.9999 at batch = 1**
(GPU pinned at ~297 W, ~81 tok/s → a flat ~3.7 J/token). That means the
"tokens ≠ energy" divergence this work is about **cannot** come from single-query
decode — it must come from the regimes that break the constant-power /
constant-throughput assumption. This notebook measures two of them, on a fixed
model (Qwen3-8B), holding everything else constant:

1. **Concurrency / batch size** — submit *B* identical fixed-length requests and
   measure batch energy. As *B* grows, the GPU is better utilized and per-query
   energy should fall. If it does, the batch=1 token proxy **overestimates** real
   per-query energy in production.
2. **Prefill vs decode** — vary input length with output length fixed. Output-token
   count (the usual proxy) ignores prefill, so long prompts should make it
   mispredict total energy.

**Controlled design:** every request uses `ignore_eos` + fixed `min=max` output
tokens, so all requests do *identical* work and differences are pure
batching/prefill effects. Energy is reported at **batch granularity** (no
per-query-under-batch attribution guesswork); "per query" = batch_energy / B is
a clearly-labelled amortization.

> Requires **vLLM** (batched serving is the whole point) on a CUDA GPU (A100).''')

# ------------------------------------------------------------------ colab + github
md(r'''## 0. Colab + GitHub setup (run first when on Colab)

Same as the pilot: open from the private repo, **File → Save a copy in GitHub** to
save the notebook, and the cell below clones the repo (needs the `GITHUB_TOKEN`
Colab secret) so results land in `results/`.''')
code(r'''import os, subprocess
REPO, BRANCH = "POLIMIGreenISE/Energy-AwareLLMReasoning", "main"
TOKEN = None
try:
    from google.colab import userdata
    TOKEN = userdata.get("GITHUB_TOKEN")
except Exception as e:
    print("Not on Colab or no GITHUB_TOKEN secret:", repr(e)[:120])
if TOKEN:
    repo_dir = "/content/" + REPO.split("/")[1]
    auth = f"https://{TOKEN}@github.com/{REPO}.git"
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "clone", "--branch", BRANCH, auth, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
    os.chdir(repo_dir)
    subprocess.run(["git", "remote", "set-url", "origin", auth], check=True)
    subprocess.run(["git", "config", "user.name", "your-name"], check=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], check=True)
    print("Repo ready:", os.getcwd())
else:
    print("No token -> running standalone.")''')

# ------------------------------------------------------------------ install
md(r'''## 1. Install (vLLM required)''')
code(r'''# transformers pinned <5 (v5.0 breaks vLLM's Qwen tokenizer). vLLM is required.
%pip install -q -U zeus-ml pynvml "transformers>=4.51,<5"
%pip install -q "vllm>=0.10,<0.11"
print(">>> If vLLM was (re)installed: Runtime > Restart session, then re-run from Section 0.")''')

# ------------------------------------------------------------------ config
md(r'''## 2. Config''')
code(r'''import os
SMOKE_TEST = True                    # tiny sanity run first
MODEL = "Qwen/Qwen3-8B"

# Experiment A: concurrency sweep (decode-dominated, short fixed input)
BATCH_SIZES = [1, 4] if SMOKE_TEST else [1, 2, 4, 8, 16, 32, 64]
L_OUT_LIST  = [256] if SMOKE_TEST else [256, 1024]
L_IN_FIXED  = 64                     # short prompt so decode dominates

# Experiment B: prefill sweep (fixed small output, growing input)
PREFILL_BATCH = 4 if SMOKE_TEST else 16
L_IN_LIST     = [64, 512] if SMOKE_TEST else [64, 512, 2048, 8192]
L_OUT_PREFILL = 64

REPS = 1 if SMOKE_TEST else 3
MAX_MODEL_LEN = 16384

OUTDIR = "results/batching"; os.makedirs(OUTDIR, exist_ok=True)
CONC_JSONL    = os.path.join(OUTDIR, "concurrency.jsonl")
PREFILL_JSONL = os.path.join(OUTDIR, "prefill.jsonl")
print("SMOKE_TEST =", SMOKE_TEST, "| batches:", BATCH_SIZES, "| L_out:", L_OUT_LIST)''')

# ------------------------------------------------------------------ load
md(r'''## 3. Load vLLM + Zeus, build fixed-length prompts, warm up''')
code(r'''import time, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=MAX_MODEL_LEN, enforce_eager=False,
          enable_prefix_caching=False)  # each request pays its own prefill (clean energy)

# Zeus energy monitor
zeus_ok = False
try:
    from zeus.monitor import ZeusMonitor
    zmon = ZeusMonitor(gpu_indices=[0]); zeus_ok = True
    print("Zeus ready.")
except Exception as e:
    print("Zeus unavailable:", repr(e)[:150], "-> pynvml power fallback.")
import pynvml; pynvml.nvmlInit(); _NVH = pynvml.nvmlDeviceGetHandleByIndex(0)

# Filler to build prompts of a target INPUT length (in tokens).
_FILLER_IDS = tok("The quick brown fox jumps over the lazy dog. " * 1500,
                  add_special_tokens=False)["input_ids"]
def make_prompt(target_in_tokens):
    return tok.decode(_FILLER_IDS[:max(1, target_in_tokens)])

META = {"model": MODEL, "gpu": torch.cuda.get_device_name(0), "max_model_len": MAX_MODEL_LEN}
print("Loaded. GPU:", META["gpu"])''')

# ------------------------------------------------------------------ measure helper
md(r'''## 4. Measurement helper

`run_batch(n, l_in, l_out)` submits **n identical** requests, each forced to emit
exactly `l_out` tokens (`ignore_eos`, greedy), and wraps the whole batched
`generate` in a Zeus energy window. Returns batch-level energy plus derived
per-query / per-token figures.''')
code(r'''def _measure(fn):
    if zeus_ok:
        zmon.begin_window("b"); r = fn(); m = zmon.end_window("b")
        ej = getattr(m, "total_energy", None)
        if ej is None: ej = sum(m.gpu_energy.values())
        return r, ej, m.time
    import threading
    samples, stop = [], threading.Event()
    def s():
        while not stop.is_set():
            samples.append(pynvml.nvmlDeviceGetPowerUsage(_NVH)/1000.0); time.sleep(0.05)
    t0=time.time(); th=threading.Thread(target=s); th.start()
    r=fn(); stop.set(); th.join(); dt=time.time()-t0
    aw = sum(samples)/len(samples) if samples else float("nan")
    return r, aw*dt, dt

def run_batch(n, l_in, l_out):
    prompts = [make_prompt(l_in)] * n
    sp = SamplingParams(temperature=0.0, min_tokens=l_out, max_tokens=l_out, ignore_eos=True)
    outs, energy_j, dt = _measure(lambda: llm.generate(prompts, sp, use_tqdm=False))
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tok  = sum(len(o.prompt_token_ids) for o in outs)
    return {
        "n": n, "energy_j": energy_j, "time_s": dt,
        "out_tokens": out_tok, "in_tokens": in_tok,
        "energy_wh": energy_j/3600.0,
        "energy_per_query_wh": energy_j/3600.0/n,
        "energy_per_out_token_j": energy_j/out_tok if out_tok else None,
        "throughput_tok_s": out_tok/dt if dt else None,
        "power_w": energy_j/dt if dt else None,
    }

# warm-up (not recorded) to avoid lazy-init skew
_ = run_batch(2, L_IN_FIXED, 32)
print("Warm-up done. Zeus used:", zeus_ok)''')

# ------------------------------------------------------------------ experiment A
md(r'''## 5. Experiment A — concurrency sweep (the money plot)

For each output length, sweep batch size. Watch **J/token** and **mWh/query**:
if they fall as batch grows, the batch=1 token proxy overestimates real serving.''')
code(r'''import json, datetime
done = set()
if os.path.exists(CONC_JSONL):
    for l in open(CONC_JSONL):
        try: d=json.loads(l); done.add((d["l_out"], d["batch"], d["rep"]))
        except Exception: pass
with open(CONC_JSONL, "a") as f:
    for l_out in L_OUT_LIST:
        for B in BATCH_SIZES:
            for rep in range(REPS):
                if (l_out, B, rep) in done: continue
                r = run_batch(B, L_IN_FIXED, l_out)
                row = {"experiment": "concurrency", "l_in": L_IN_FIXED, "l_out": l_out,
                       "batch": B, "rep": rep, **r, **META,
                       "ts": datetime.datetime.utcnow().isoformat()}
                f.write(json.dumps(row)+"\n"); f.flush()
                print(f"L_out={l_out:>4} B={B:>3} rep{rep}: "
                      f"{r['energy_per_query_wh']*1000:8.2f} mWh/query | "
                      f"{r['energy_per_out_token_j']:.3f} J/tok | "
                      f"{r['throughput_tok_s']:7.0f} tok/s | {r['power_w']:.0f} W")
print("Experiment A done.")''')

# ------------------------------------------------------------------ experiment B
md(r'''## 6. Experiment B — prefill sweep (fixed output, growing input)

Output tokens are held fixed, so the usual output-token proxy predicts *constant*
energy. If energy rises with input length, the proxy is blind to prefill.''')
code(r'''done = set()
if os.path.exists(PREFILL_JSONL):
    for l in open(PREFILL_JSONL):
        try: d=json.loads(l); done.add((d["l_in"], d["rep"]))
        except Exception: pass
with open(PREFILL_JSONL, "a") as f:
    for l_in in L_IN_LIST:
        for rep in range(REPS):
            if (l_in, rep) in done: continue
            r = run_batch(PREFILL_BATCH, l_in, L_OUT_PREFILL)
            row = {"experiment": "prefill", "l_in": l_in, "l_out": L_OUT_PREFILL,
                   "batch": PREFILL_BATCH, "rep": rep, **r, **META,
                   "ts": datetime.datetime.utcnow().isoformat()}
            f.write(json.dumps(row)+"\n"); f.flush()
            # actual mean input tokens per request (target may differ by a few)
            mean_in = r["in_tokens"]/r["n"]
            print(f"L_in~{l_in:>5} (actual {mean_in:6.0f}) rep{rep}: "
                  f"{r['energy_wh']*1000:8.2f} mWh/batch | "
                  f"{r['energy_per_out_token_j']:.3f} J/out-tok | {r['power_w']:.0f} W")
print("Experiment B done.")''')

# ------------------------------------------------------------------ analysis
md(r'''## 7. Analysis''')
code(r'''import pandas as pd, numpy as np, matplotlib.pyplot as plt

ca = pd.read_json(CONC_JSONL, lines=True)
g = ca.groupby(["l_out", "batch"]).agg(
    j_per_tok=("energy_per_out_token_j", "mean"),
    mwh_per_query=("energy_per_query_wh", lambda s: s.mean()*1000),
    tok_s=("throughput_tok_s", "mean"), power_w=("power_w", "mean")).reset_index()
print("Concurrency sweep (means):\n", g.to_string(index=False))

# batch=1 vs largest batch: how much does the batch=1 proxy overestimate?
for l_out in sorted(ca["l_out"].unique()):
    sub = g[g["l_out"]==l_out].sort_values("batch")
    e1 = sub[sub["batch"]==sub["batch"].min()]["mwh_per_query"].iloc[0]
    eN = sub[sub["batch"]==sub["batch"].max()]["mwh_per_query"].iloc[0]
    print(f"L_out={l_out}: mWh/query batch={sub['batch'].min()} -> {e1:.1f} ; "
          f"batch={sub['batch'].max()} -> {eN:.1f}  ({e1/eN:.1f}x overestimate at batch=1)")''')

md(r'''### 7a. Money plot — energy per output token vs batch size''')
code(r'''fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for l_out in sorted(ca["l_out"].unique()):
    s = g[g["l_out"]==l_out].sort_values("batch")
    ax[0].plot(s["batch"], s["j_per_tok"], "o-", label=f"L_out={l_out}")
    ax[1].plot(s["batch"], s["mwh_per_query"], "o-", label=f"L_out={l_out}")
ax[0].axhline(3.7, ls="--", c="gray", label="pilot batch=1 (3.7 J/tok)")
ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
ax[0].set_xlabel("batch size"); ax[0].set_ylabel("J / output token"); ax[0].set_title("Energy per token vs batch"); ax[0].legend()
ax[1].set_xlabel("batch size"); ax[1].set_ylabel("mWh / query (amortized)"); ax[1].set_title("Per-query energy vs batch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "concurrency_energy.png"), dpi=120); plt.show()''')

md(r'''### 7b. Throughput & power vs batch size (the utilization story)''')
code(r'''fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for l_out in sorted(ca["l_out"].unique()):
    s = g[g["l_out"]==l_out].sort_values("batch")
    ax[0].plot(s["batch"], s["tok_s"], "o-", label=f"L_out={l_out}")
    ax[1].plot(s["batch"], s["power_w"], "o-", label=f"L_out={l_out}")
ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
ax[0].set_xlabel("batch size"); ax[0].set_ylabel("tokens / s"); ax[0].set_title("Throughput"); ax[0].legend()
ax[1].set_xlabel("batch size"); ax[1].set_ylabel("GPU power (W)"); ax[1].set_title("Power draw"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "concurrency_throughput.png"), dpi=120); plt.show()''')

md(r'''### 7c. Prefill — energy rises while output tokens stay fixed''')
code(r'''pf = pd.read_json(PREFILL_JSONL, lines=True)
pf["mean_in"] = pf["in_tokens"] / pf["n"]
pg = pf.groupby("l_in").agg(mean_in=("mean_in","mean"),
                            energy_wh=("energy_wh","mean"),
                            j_per_out_tok=("energy_per_out_token_j","mean")).reset_index()
print(pg.to_string(index=False))
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(pg["mean_in"], pg["j_per_out_tok"], "o-")
ax.set_xlabel(f"input tokens (output fixed at {L_OUT_PREFILL})")
ax.set_ylabel("J / output token")
ax.set_title("Output-token proxy is blind to prefill")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "prefill_energy.png"), dpi=120); plt.show()''')

# ------------------------------------------------------------------ notes
md(r'''## 8. Reading the result

- **Sanity:** at `batch=1, L_in=64`, J/token should land near the pilot's **3.7 J/tok**.
- **Divergence (Exp A):** if J/token and mWh/query drop steeply as batch grows, then
  the batch=1 token proxy **overestimates** production per-query energy — a routing
  policy calibrated on batch=1 tokens would misjudge the real trade-off. Report the
  batch=1 ÷ batch=max ratio as the headline "how wrong is the proxy" number.
- **No divergence:** if the curves are flat, that's still a finding — the token proxy
  is robust to batching on this GPU. (Either way it answers the question.)
- **Prefill (Exp B):** rising J/output-token with input length shows output-token
  counting misses prefill energy — long-context queries break the proxy differently.

**Caveats:** energy is at batch granularity (per-query is an
amortization); vLLM continuous batching means "batch size" = offered concurrency;
numbers are A100-specific. Next: repeat the A-sweep with *mixed* think/no-think
workloads (realistic serving) and feed measured per-query energy back into the
router's labels.''')

# ------------------------------------------------------------------ write
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]
    c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "01_batching_divergence.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
