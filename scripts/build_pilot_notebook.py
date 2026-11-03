"""Builds notebooks/00_pilot_energy_measurement.ipynb (Phase-0 pilot).

Run:  python scripts/build_pilot_notebook.py
Regenerate the notebook any time you edit the cells below. Keeping the
notebook's source here (instead of hand-editing JSON) avoids escaping bugs
(e.g. the \\boxed regex) and makes diffs reviewable.
"""
import json, os

cells = []

def _cid():
    return f"cell-{len(cells):02d}"

def md(text):
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": text})

def code(src):
    cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})

# ---------------------------------------------------------------- title
md(r'''# Phase-0 Pilot — Energy of Think vs. No-Think (Qwen3-8B)

**Goal (de-risk the real pipeline before spending GPU credits).** Run a small
paired study on Qwen3-8B over OptimalThinkingBench, toggling only
`enable_thinking`, and measure **real GPU energy (Wh)** per query with **Zeus**.

This pilot answers three plumbing questions and produces three numbers:

**Does it work?**
1. Does **Zeus read NVML energy** on this Colab GPU?
2. Does **vLLM toggle `enable_thinking`** and emit parseable `<think>...</think>`?
3. Can we get **clean per-query attribution** at low concurrency?

**What do we learn?**
- The **Wh-vs-output-tokens scatter** (the RQ1 spine: does token count predict energy?).
- Mean **Wh/query** for think and no-think → used to **size the Phase-1 GPU budget**.

> Runtime: set **Runtime → Change runtime type → A100 GPU** before running.
> Everything is **resumable** — results append to a JSONL, so a crash or a
> Colab restart (vLLM install often forces one) never loses completed work.''')

# ---------------------------------------------------------------- colab + github
md(r'''## 0. Colab + GitHub setup (run first when on Colab)

Open this straight from the private repo:
`https://colab.research.google.com/github/cemalgndzz/llm-routing-tesina/blob/main/notebooks/00_pilot_energy_measurement.ipynb`
(Colab will ask you to authorize GitHub and to *include private repos*.)

**Save notebook edits back to GitHub:** use **File → Save a copy in GitHub** in
the Colab menu — it commits this `.ipynb` to the repo (pick branch `main`). No
code needed; it uses Colab's GitHub authorization.

**Work with the rest of the repo** (`scripts/`, `src/`, configs) and push other
files: run the cell below. It clones the repo into the runtime using a GitHub
token. Add the token once in Colab → **key icon (left sidebar) → + Add new
secret**, name `GITHUB_TOKEN`, value = a fine-grained PAT with *Contents: Read
and write* on this repo, and toggle **Notebook access** on.''')
code(r'''import os, subprocess

REPO   = "cemalgndzz/llm-routing-tesina"
BRANCH = "main"

TOKEN = None
try:
    from google.colab import userdata            # present only on Colab
    TOKEN = userdata.get("GITHUB_TOKEN")
except Exception as e:
    print("Not on Colab or no GITHUB_TOKEN secret:", repr(e)[:120])

if TOKEN:
    repo_dir = "/content/" + REPO.split("/")[1]
    auth_url = f"https://{TOKEN}@github.com/{REPO}.git"      # token stays in this runtime only
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "clone", "--branch", BRANCH, auth_url, repo_dir], check=True)
    else:
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
    os.chdir(repo_dir)
    subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
    subprocess.run(["git", "config", "user.name", "cemalgndzz"], check=True)
    subprocess.run(["git", "config", "user.email", "cgogit@gmail.com"], check=True)
    print("Repo ready. Working dir:", os.getcwd())
else:
    print("No token -> running standalone (notebook still works; it pulls data/model from HF).")''')
code(r'''# Push files you changed in the CLONED repo back to GitHub.
# (For the notebook itself, prefer File > Save a copy in GitHub.)
# Note: data/, results/ and *.jsonl are gitignored -> pilot outputs are NOT pushed.
import subprocess, datetime

BRANCH = "main"
COMMIT_MESSAGE = "Update from Colab " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

subprocess.run(["git", "add", "-A"], check=True)
changed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
if changed:
    subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=True)
    subprocess.run(["git", "push", "origin", BRANCH], check=True)
    print("Pushed to GitHub.")
else:
    print("Nothing to commit.")''')

# ---------------------------------------------------------------- GPU check
md(r'''## 1. Environment check''')
code(r'''import subprocess, torch
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True)
print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))''')

# ---------------------------------------------------------------- installs
md(r'''## 2. Install dependencies

`zeus-ml` (energy), `datasets` (OTB), `pynvml` (fallback power), and
`transformers` **pinned `<5`** — transformers 5.0 changed the `Qwen2Tokenizer`
API (`all_special_tokens_extended`) that vLLM depends on, so an unpinned
`-U transformers` breaks vLLM. vLLM is optional and pinned to a **CUDA-12** build
(the latest wheel targets CUDA 13, which Colab's driver can't load).

If vLLM installs, do **Runtime → Restart session**, then re-run from Section 0.
If it still errors, the serving cell falls back to **HF Transformers** — fine for
the pilot (only the *absolute* Wh differ; run those on a dedicated A100 VM).''')
code(r'''# Colab + vLLM is version-fragile (CUDA 13 wheels, transformers 5.0, amdsmi
# probe bugs). Leave TRY_VLLM=False to run the pilot cleanly on HF Transformers;
# flip to True only to experiment with vLLM (then Runtime > Restart session).
TRY_VLLM = False

# Core deps. transformers pinned <5 (v5.0 breaks vLLM's Qwen tokenizer).
%pip install -q -U zeus-ml datasets pynvml accelerate "transformers>=4.51,<5"

if TRY_VLLM:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm>=0.10,<0.11"])
    print(">>> vLLM installed. Do Runtime > Restart session, then re-run from Section 0.")''')

# ---------------------------------------------------------------- config
md(r'''## 3. Config

Start with `SMOKE_TEST = True` (a 6-generation sanity run, ~minutes). Once the
whole notebook runs clean end-to-end, set `SMOKE_TEST = False` and re-run the
loop for the real ~50-query pilot.''')
code(r'''import os, random, numpy as np

SMOKE_TEST = True                 # True = tiny sanity run; False = full pilot
MODEL = "Qwen/Qwen3-8B"
N_PER_SUBSET = 3 if SMOKE_TEST else 25     # queries from each of over/under-thinking
REPS = 1 if SMOKE_TEST else 3              # repeats per (query, mode) for variance/CIs
THINK_MAX_TOKENS = 1024 if SMOKE_TEST else 8192
NOTHINK_MAX_TOKENS = 512 if SMOKE_TEST else 1024
SEED = 12345

OUTDIR = "results/pilot"
os.makedirs(OUTDIR, exist_ok=True)
RESULTS_JSONL = os.path.join(OUTDIR, "pilot_results.jsonl")

random.seed(SEED); np.random.seed(SEED)

# Qwen3 recommended sampling (thinking vs non-thinking presets)
SAMPLING = {
    "think":    dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=THINK_MAX_TOKENS),
    "no_think": dict(temperature=0.7, top_p=0.80, top_k=20, max_tokens=NOTHINK_MAX_TOKENS),
}
print("SMOKE_TEST =", SMOKE_TEST, "| N_PER_SUBSET =", N_PER_SUBSET, "| REPS =", REPS)''')

# ---------------------------------------------------------------- data
md(r'''## 4. Load OptimalThinkingBench and sample queries

`subset == "overthinkingbench"` → think is wasted → **ground-truth label `no_think`**.
`subset == "underthinkingbench"` → think is needed → **ground-truth label `think`**.
(The HF release excludes the math subsets, which is fine for the pilot.) For the
overthinking side we prefer auto-scorable answer types so correctness is
checkable without an LLM judge in the pilot.''')
code(r'''import json
from collections import Counter
from datasets import load_dataset

ds = load_dataset("facebook/optimal_thinking_bench")["train"]

def parse_meta(m):
    if isinstance(m, dict):
        return m
    try:
        return json.loads(m)
    except Exception:
        return {}

rows = []
for r in ds:
    meta = parse_meta(r.get("metadata"))
    rows.append({
        "question": r["question"],
        "answer": r.get("answer", ""),
        "subset": r["subset"],
        "answer_type": str(meta.get("answer_type", "unknown")),
        "domain": str(meta.get("domain", "unknown")),
    })

over  = [r for r in rows if r["subset"] == "overthinkingbench"]
under = [r for r in rows if r["subset"] == "underthinkingbench"]

scorable = {"numeric", "multiple-choice", "mcq", "short", "short-answer", "one-word", "short-phrase"}
over_pref = [r for r in over if r["answer_type"].lower() in scorable] or over
random.shuffle(over_pref); random.shuffle(under)

sample = []
for i, r in enumerate(over_pref[:N_PER_SUBSET]):
    r = dict(r); r["qid"] = f"over_{i:03d}"; r["route_label"] = "no_think"; sample.append(r)
for i, r in enumerate(under[:N_PER_SUBSET]):
    r = dict(r); r["qid"] = f"under_{i:03d}"; r["route_label"] = "think"; sample.append(r)

print(f"OTB rows={len(ds)} | over={len(over)} under={len(under)} | sampled={len(sample)}")
print("answer types in sample:", dict(Counter(r["answer_type"] for r in sample)))
print("\nExample:", sample[0]["qid"], "|", sample[0]["question"][:120])''')

# ---------------------------------------------------------------- serving
md(r'''## 5. Serving layer — vLLM first, HF Transformers fallback

A unified `generate(question, mode)` builds the Qwen3 chat prompt with
`enable_thinking` on/off, generates, and reports prompt / completion / **thinking**
token counts (parsed at the `</think>` delimiter).''')
code(r'''from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

BACKEND, llm, hf_model = None, None, None
if globals().get("TRY_VLLM", False):
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
                  max_model_len=16384, enforce_eager=False)
        BACKEND = "vllm"
    except Exception as e:
        print("vLLM unavailable -> HF Transformers fallback. Reason:", repr(e)[:300])
if BACKEND is None:
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    BACKEND = "hf"
print("BACKEND =", BACKEND)

def build_prompt(question, think):
    messages = [{"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=think)

def split_think(text):
    # Qwen3 emits <think>...</think> then the final answer.
    if "</think>" in text:
        think_part, _, answer_part = text.partition("</think>")
        return think_part.replace("<think>", "").strip(), answer_part.strip()
    return "", text.strip()

def n_tokens(s):
    return len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0

def generate(question, mode):
    think = (mode == "think")
    prompt = build_prompt(question, think)
    sp = SAMPLING[mode]
    if BACKEND == "vllm":
        from vllm import SamplingParams
        params = SamplingParams(temperature=sp["temperature"], top_p=sp["top_p"],
                                top_k=sp["top_k"], max_tokens=sp["max_tokens"])
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        text = out.outputs[0].text
        completion_tokens = len(out.outputs[0].token_ids)
        prompt_tokens = len(out.prompt_token_ids)
    else:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen = hf_model.generate(**inputs, do_sample=True, temperature=sp["temperature"],
                                top_p=sp["top_p"], top_k=sp["top_k"], max_new_tokens=sp["max_tokens"])
        gen_ids = gen[0][prompt_tokens:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        completion_tokens = int(gen_ids.shape[0])
    think_text, answer_text = split_think(text)
    return {"text": text, "think_text": think_text, "answer_text": answer_text,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "think_tokens": n_tokens(think_text)}''')

# ---------------------------------------------------------------- energy
md(r'''## 6. Energy measurement (Zeus) + idle baseline + warm-up

`measure(fn)` wraps a single generation in a Zeus energy window for clean
per-query attribution at low concurrency (one query at a time). If Zeus can't
init, it falls back to integrating NVML power over the call. We also record an
idle-power baseline (context) and do one **un-recorded warm-up** per mode to
avoid lazy-init skew.

> **Attribution caveat:** this is the controlled low-concurrency setting. Energy
> attribution under real *batched* serving is a separate (harder) question we
> handle in a later phase — see CLAUDE.md §4.5.''')
code(r'''import time, threading

zeus_ok = False
try:
    from zeus.monitor import ZeusMonitor
    zmon = ZeusMonitor(gpu_indices=[0])
    zeus_ok = True
    print("Zeus monitor ready.")
except Exception as e:
    print("Zeus unavailable:", repr(e)[:200], "-> NVML power-sampling fallback.")

import pynvml
pynvml.nvmlInit()
_NVH = pynvml.nvmlDeviceGetHandleByIndex(0)

def measure(fn):
    """Run fn(); return (result, energy_wh, seconds)."""
    if zeus_ok:
        zmon.begin_window("q")
        r = fn()
        m = zmon.end_window("q")
        energy_j = getattr(m, "total_energy", None)
        if energy_j is None:
            energy_j = sum(m.gpu_energy.values())
        return r, energy_j / 3600.0, m.time
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append(pynvml.nvmlDeviceGetPowerUsage(_NVH) / 1000.0)  # W
            time.sleep(0.05)
    t0 = time.time(); th = threading.Thread(target=sampler); th.start()
    r = fn()
    stop.set(); th.join(); dt = time.time() - t0
    avg_w = sum(samples) / len(samples) if samples else float("nan")
    return r, avg_w * dt / 3600.0, dt

def idle_power_w(seconds=3):
    s, t = [], time.time()
    while time.time() - t < seconds:
        s.append(pynvml.nvmlDeviceGetPowerUsage(_NVH) / 1000.0); time.sleep(0.1)
    return sum(s) / len(s) if s else float("nan")

IDLE_W = idle_power_w(3)
print("Idle GPU power (W):", round(IDLE_W, 1))

_ = generate(sample[0]["question"], "no_think")   # warm-up (not recorded)
_ = generate(sample[0]["question"], "think")
print("Warm-up done.")''')

# ---------------------------------------------------------------- scoring
md(r'''## 7. Lightweight correctness scoring

Pilot-grade automatic scoring: pull the final answer (last `\boxed{}`, else
`Answer:`, else last line), then numeric- or normalized-string-compare to gold.
Open-ended-long answers return `None` (they need an LLM judge — that's Phase 1).''')
code(r'''import re

def extract_answer(answer_text):
    m = re.findall(r"\\boxed\{([^}]*)\}", answer_text)
    if m:
        return m[-1].strip()
    m = re.search(r"(?i)answer\s*[:=]\s*(.+)", answer_text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [l.strip() for l in answer_text.splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _norm(s):
    return re.sub(r"[^a-z0-9.]+", "", str(s).lower())

def _try_float(s):
    m = re.search(r"-?\d+\.?\d*", str(s).replace(",", ""))
    return float(m.group()) if m else None

def is_correct(answer_text, gold, answer_type):
    if answer_type and "long" in answer_type.lower():
        return None  # needs LLM judge (Phase 1)
    pred = extract_answer(answer_text)
    a, b = _try_float(pred), _try_float(gold)
    if a is not None and b is not None:
        return bool(abs(a - b) <= 1e-4 * max(1.0, abs(b)))
    np_, ng = _norm(pred), _norm(gold)
    if not ng:
        return None
    return bool(np_ == ng or ng in np_ or np_ in ng)''')

# ---------------------------------------------------------------- loop
md(r'''## 8. Main measurement loop (resumable)

Writes one JSONL row per `(query, mode, rep)`. Re-running skips rows already on
disk, so you can stop/restart freely.''')
code(r'''import datetime

done = set()
if os.path.exists(RESULTS_JSONL):
    for line in open(RESULTS_JSONL):
        try:
            d = json.loads(line); done.add((d["qid"], d["mode"], d["rep"]))
        except Exception:
            pass
print("Already on disk:", len(done))

META = {"backend": BACKEND, "model": MODEL, "gpu": torch.cuda.get_device_name(0)}
modes = ["no_think", "think"]
total = len(sample) * len(modes) * REPS
cnt = 0
with open(RESULTS_JSONL, "a") as fout:
    for q in sample:
        for mode in modes:
            for rep in range(REPS):
                cnt += 1
                if (q["qid"], mode, rep) in done:
                    continue
                res, wh, secs = measure(lambda: generate(q["question"], mode))
                corr = is_correct(res["answer_text"], q["answer"], q["answer_type"])
                row = {
                    "qid": q["qid"], "subset": q["subset"], "route_label": q["route_label"],
                    "answer_type": q["answer_type"], "domain": q["domain"],
                    "mode": mode, "rep": rep,
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "think_tokens": res["think_tokens"],
                    "energy_wh": wh, "latency_s": secs, "correct": corr,
                    "idle_w": IDLE_W,
                    "answer_text": res["answer_text"][:2000],
                    "gold": str(q["answer"])[:500],
                    **META,
                    "ts": datetime.datetime.utcnow().isoformat(),
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"[{cnt}/{total}] {q['qid']:>10} {mode:<8} rep{rep}: "
                      f"{wh*1000:7.2f} mWh  {secs:6.1f}s  think={res['think_tokens']:>5}  correct={corr}")
print("Loop complete.")''')

# ---------------------------------------------------------------- analysis
md(r'''## 9. Analysis''')
code(r'''import pandas as pd
df = pd.read_json(RESULTS_JSONL, lines=True)
print("rows:", len(df), "| backend:", df["backend"].iloc[0])

agg = df.groupby("mode").agg(
    energy_wh=("energy_wh", "mean"),
    latency_s=("latency_s", "mean"),
    completion_tokens=("completion_tokens", "mean"),
    think_tokens=("think_tokens", "mean"),
).round(4)
print("\nPer-mode means:\n", agg)

sc = df[df["correct"].notna()]
if len(sc):
    print("\nAccuracy (scorable only) by subset/mode:\n",
          sc.groupby(["subset", "mode"])["correct"].mean().round(3))''')

# Plot 1
md(r'''### 9a. RQ1 spine — does output-token count predict energy?''')
code(r'''import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
for mode, c in [("no_think", "tab:blue"), ("think", "tab:red")]:
    d = df[df["mode"] == mode]
    ax.scatter(d["completion_tokens"], d["energy_wh"] * 1000, s=20, alpha=0.6, label=mode, color=c)

x = df["completion_tokens"].to_numpy(float)
y = (df["energy_wh"] * 1000).to_numpy(float)
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() > 2:
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 50)
    ax.plot(xs, intercept + slope * xs, "k--", lw=1, label="linear fit")
    resid = y[mask] - (intercept + slope * x[mask])
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    ax.set_title(f"Energy vs output tokens  (R^2 = {r2:.3f}, {int(mask.sum())} pts)")
ax.set_xlabel("completion tokens"); ax.set_ylabel("energy per query (mWh)"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "scatter_energy_vs_tokens.png"), dpi=120); plt.show()''')

# Plot 2
md(r'''### 9b. Energy distribution: think vs no-think''')
code(r'''fig, ax = plt.subplots(figsize=(7, 5))
data = [(df[df["mode"] == m]["energy_wh"] * 1000).dropna() for m in ["no_think", "think"]]
ax.boxplot(data, labels=["no_think", "think"])
ax.set_ylabel("energy per query (mWh)"); ax.set_title("Per-query energy by mode")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "energy_box.png"), dpi=120); plt.show()''')

# Budget
md(r'''## 10. Budget extrapolation for Phase 1

Uses the measured mean seconds/query and Wh/query to estimate the cost of the
full paired study (both modes on all OptimalThinkingBench queries). Use this to
right-size the GPU-credit request **before** launching it.''')
code(r'''mean_s = df.groupby("mode")["latency_s"].mean().to_dict()
mean_wh = df.groupby("mode")["energy_wh"].mean().to_dict()

N_FULL = 1460 + 610      # OverthinkingBench + UnderthinkingBench (incl. math)
REPS_FULL = 3
per_query_s = sum(mean_s.get(m, 0) for m in ["think", "no_think"])
per_query_wh = sum(mean_wh.get(m, 0) for m in ["think", "no_think"])
gpu_hours = per_query_s * N_FULL * REPS_FULL / 3600
kwh = per_query_wh * N_FULL * REPS_FULL / 1000

print(f"Per query (both modes): {per_query_s:.1f} s, {per_query_wh*1000:.1f} mWh")
print(f"Full paired study ({N_FULL} queries x {REPS_FULL} reps, both modes):")
print(f"  ~{gpu_hours:.1f} GPU-hours   ~{kwh:.2f} kWh on this GPU")''')

# Notes
md(r'''## 11. Notes, caveats & next steps

**Sanity checks before trusting numbers**
- Confirm `BACKEND == "vllm"` (else the pilot ran on the HF fallback; Wh won't match Phase 1).
- Confirm `think_tokens` is ~0 for `no_think` rows and hundreds–thousands for `think` rows.
- Confirm Zeus was used (not the power-sampling fallback) — check the cell-6 print.

**Known pilot limitations (handled in later phases)**
- Open-ended-long answers are unscored → add an LLM/reward-model judge (Phase 1).
- Math subsets are excluded from the HF release → add via the OTB `prepare` workflow.
- Per-query energy here is **low-concurrency**; batched-serving attribution is a separate study (CLAUDE.md §4.5).
- Optional cross-check: wrap the loop in CodeCarbon (`pip install codecarbon`) and compare total kWh to `df["energy_wh"].sum()`.

**What to report back from this pilot**
1. The cell-9a `R^2` and the scatter shape (proportional? or do think-heavy points peel off?).
2. The per-mode mean Wh/query and the cell-10 GPU-hour estimate.
3. Whether all three plumbing checks passed (Zeus, vLLM thinking toggle, parsing).''')

# ------------------------------------------------------------------ write
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# normalize sources to list-of-lines (canonical nbformat)
for c in nb["cells"]:
    s = c["source"]
    c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s

out_path = os.path.join("notebooks", "00_pilot_energy_measurement.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} with {len(cells)} cells.")
