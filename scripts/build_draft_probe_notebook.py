"""Builds notebooks/05_draft_probe.ipynb — format-constrained draft generation (GPU).

v2 — provider-agnostic (Colab OR vast.ai/any Jupyter):
  - GitHub token: Colab secret -> env GITHUB_TOKEN -> interactive prompt.
  - Serving: tries vLLM, falls back to HF Transformers automatically (this run is
    tiny — 849 drafts x <=64 tokens — so the HF backend is perfectly adequate).

Why this run: step-3 showed DART-style draft agreement fails on open-ended prose
because free-form answers never string-match. Fix: K short drafts per query with
an explicit "final answer only" instruction, no_think mode, tiny max_tokens.
Agreement becomes meaningful AND the probe's energy cost shrinks ~10x.

Run:  python scripts/build_draft_probe_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Draft probe — format-constrained short drafts (for router confidence features)

Generates **K=3 short no-think drafts** per OTB query with a *"final answer only"*
instruction. Draft **agreement** becomes a meaningful confidence signal, and the
probe's energy cost drops ~10× vs full no-think answers.

Runs on **Colab or vast.ai** (any Jupyter with an NVIDIA GPU). Uses vLLM when it
works, HF Transformers otherwise — for ~849 short generations both are fine.''')

md(r'''## 0. Repo setup (Colab or vast.ai)

Token lookup order: Colab secret `GITHUB_TOKEN` → env var `GITHUB_TOKEN` → prompt.
On vast.ai you can also `export GITHUB_TOKEN=...` in the template's env settings.''')
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
print("cwd:", os.getcwd())''')

md(r'''## 1. Install (non-destructive)

Never downgrades an existing vLLM/transformers/torch — version pins on top of a
provider's preinstalled stack are how environments end up half-upgraded (e.g.
`ImportError: MultiModalDataDict`). Rules:
- vLLM already importable → leave the whole stack alone.
- vLLM present but **broken** → clean-reinstall latest (`REPAIR_VLLM = True`), then
  **restart the kernel** and re-run from Section 0.
- No vLLM at all → install latest unpinned (it brings matching deps).
- vLLM keeps failing? `TRY_VLLM = False` in Section 2 uses HF Transformers instead.

**Known-good vast.ai recipe (2026-07-19, CUDA-12.6 driver, verified working):**
```
%pip uninstall -y amdsmi vllm vllm-flash-attn xformers flashinfer-python torch torchvision torchaudio
%pip install "vllm>=0.10,<0.11" "transformers>=4.51,<5"
%pip uninstall -y amdsmi ray torchcodec   # vllm re-pulls broken amdsmi; removing ray kills the AMD-probe crash path (single-GPU needs no ray)
```
then restart the kernel and re-run from Section 0.''')
code(r'''import importlib, subprocess, sys
REPAIR_VLLM = False   # set True ONCE if vllm import errors (mixed versions), then restart kernel

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
TRY_VLLM = True                              # False -> go straight to HF Transformers
MODEL = "Qwen/Qwen3-8B"
N_PER_SUBSET = 5 if SMOKE_TEST else None     # None = whole subset; MUST match notebook 02 v3
K_DRAFTS = 3
DRAFT_MAX_TOKENS = 64
HF_BATCH = 32                                # HF-backend batch size
J_PER_OUT_TOKEN = 3.7                        # measured batch=1 J/token (A100 pilot)
SEED = 12345                                 # MUST stay 12345 (same sample as nb 02)
OUTDIR = "results/paired"; os.makedirs(OUTDIR, exist_ok=True)
DRAFTS_JSONL = os.path.join(OUTDIR, "drafts_v1.jsonl")
DRAFT_INSTRUCTION = ("\n\nIMPORTANT: Answer with ONLY the final answer - a number, a word, "
                     "or one short phrase. No explanation, no steps, no reasoning.")
print(f"N_PER_SUBSET={N_PER_SUBSET} (None=all) x {K_DRAFTS} drafts, <= {DRAFT_MAX_TOKENS} tokens each")''')

md(r'''## 3. Load the SAME OTB sample as notebook 02 (seed-locked)''')
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
random.shuffle(over); random.shuffle(under)
sample = []
for i, r in enumerate(over[:N_PER_SUBSET]):  sample.append({**r, "qid": f"over_{i:03d}"})
for i, r in enumerate(under[:N_PER_SUBSET]): sample.append({**r, "qid": f"under_{i:03d}"})
sample = [s for s in sample if str(s["answer"]).strip() not in ("", "None", "nan")]
print("queries:", len(sample))
# self-check against the paired labels file (same qids expected)
if os.path.exists("results/paired/paired_oracle_final.json"):
    lbl = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}
    ok = sum(1 for s in sample if s["qid"] in lbl and lbl[s["qid"]] == s["question"])
    print(f"qid/question match with labels file: {ok}/{len([s for s in sample if s['qid'] in lbl])}")''')

md(r'''## 4. Serving backend (vLLM → HF Transformers fallback) + energy''')
code(r'''import time, torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

BACKEND = None
if TRY_VLLM:
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
                  max_model_len=8192, enforce_eager=False)
        BACKEND = "vllm"
    except Exception as e:
        print("vLLM unavailable -> HF fallback. Reason:", repr(e)[:200])
if BACKEND is None:
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                    device_map="cuda")
    hf_model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    BACKEND = "hf"
print("backend:", BACKEND)

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

def build_prompt(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": q + DRAFT_INSTRUCTION}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)

def generate_batch(prompts):
    """Returns list of (text, completion_tokens, finish_reason)."""
    if BACKEND == "vllm":
        params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20,
                                max_tokens=DRAFT_MAX_TOKENS)
        outs = llm.generate(prompts, params, use_tqdm=True)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason)
                for o in outs]
    results = []
    for i in range(0, len(prompts), HF_BATCH):
        chunk = prompts[i:i+HF_BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            gen = hf_model.generate(**enc, max_new_tokens=DRAFT_MAX_TOKENS,
                                    do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                    pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id)]
                fin = "stop"
            else:
                fin = "stop" if len(ids) < DRAFT_MAX_TOKENS else "length"
            results.append((tok.decode(ids, skip_special_tokens=True), len(ids), fin))
        print(f"  hf batch {i//HF_BATCH + 1}/{(len(prompts)+HF_BATCH-1)//HF_BATCH}", end="\r")
    print()
    return results

_ = generate_batch([build_prompt("What is 2+2?")])  # warm-up (discarded)
print("warm-up done")''')

md(r'''## 5. Generate drafts (batched, resumable)''')
code(r'''done = set()
if os.path.exists(DRAFTS_JSONL):
    for l in open(DRAFTS_JSONL):
        try:
            d = json.loads(l); done.add((d["qid"], d["rep"]))
        except Exception: pass
print("already on disk:", len(done))

with open(DRAFTS_JSONL, "a") as f:
    for rep in range(K_DRAFTS):
        pending = [q for q in sample if (q["qid"], rep) not in done]
        print(f"draft rep={rep}: {len(pending)} pending")
        if not pending: continue
        prompts = [build_prompt(q["question"]) for q in pending]
        outs, energy_j, secs = measure(lambda: generate_batch(prompts))
        total_tok = sum(ct for _, ct, _ in outs) or 1
        for q, (text, ctok, fin) in zip(pending, outs):
            f.write(json.dumps({
                "qid": q["qid"], "subset": q["subset"], "answer_type": q["answer_type"],
                "rep": rep, "draft_text": text.strip()[:400],
                "completion_tokens": ctok, "finish_reason": fin, "backend": BACKEND,
                "energy_share_wh": (energy_j * ctok / total_tok) / 3600.0,
                "energy_b1eq_wh": ctok * J_PER_OUT_TOKEN / 3600.0,
            }) + "\n")
        f.flush()
        print(f"  rep {rep}: {len(pending)} drafts | {secs:.0f}s | {energy_j/3600:.2f} Wh")
print("done ->", DRAFTS_JSONL)''')

md(r'''## 6. Quick agreement preview + push''')
code(r'''import re, statistics as st
from collections import defaultdict
drafts = defaultdict(dict)
for l in open(DRAFTS_JSONL):
    d = json.loads(l); drafts[d["qid"]][d["rep"]] = d
def norm(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", str(s))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())
n_pair = agree2 = agree_all = 0
tok_mean = []
for qid, reps in drafts.items():
    if len(reps) < 2: continue
    texts = [norm(reps[i]["draft_text"]) for i in sorted(reps)]
    n_pair += 1
    agree2 += texts[0] == texts[1]
    agree_all += len(set(texts)) == 1
    tok_mean.append(st.mean(reps[i]["completion_tokens"] for i in reps))
print(f"queries: {n_pair} | pairwise(0,1) exact agree: {agree2/n_pair:.2f} | "
      f"all-{K_DRAFTS} agree: {agree_all/n_pair:.2f}")
print(f"mean draft length: {st.mean(tok_mean):.0f} tokens "
      f"(vs ~540 free-form -> probe ~{540/max(1,st.mean(tok_mean)):.0f}x cheaper)")''')
code(r'''# push results (drafts_v1.jsonl is gitignored by *.jsonl? -> force-add)
import subprocess
subprocess.run(["git", "add", "-f", DRAFTS_JSONL], check=True)
subprocess.run(["git", "commit", "-m", "draft probe: format-constrained short drafts"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed (check output above for errors)")''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]; c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "05_draft_probe.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
