"""Builds notebooks/06_confidence_probe.ipynb — logit-based confidence probes (GPU).

Why: constrained-draft agreement (nb 05) measures difficulty but NOT solvability —
the missing signal for the router's gap to the oracle. This run reads the model's
own beliefs from first-token logits (Kadavath-style P(True); cheap cousin of
Internal Confidence, arXiv:2506.09669): a single batched forward pass per probe,
no generation, so the energy cost is prefill-only (~nothing).

Per query:
  p_true_k   (k=0..2): P("Yes") for "is draft k correct?"     [3 forward passes]
  p_need_think      : P("Yes") for "do you need careful reasoning here?" [1 pass]

HF-backend only (we need logits; the run is tiny). vast.ai/Colab compatible.
Output: results/paired/probes_v1.jsonl (resumable).

Run:  python scripts/build_probe_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Confidence probe — first-token P(Yes/No) self-evaluation

Reads the model's **own confidence** from logits, before any generation:
- **verify probes**: "is this draft answer correct? Yes/No" → P(Yes), for each of the
  3 short drafts from notebook 05;
- **difficulty probe**: "do you need careful step-by-step reasoning? Yes/No" → P(Yes).

Each probe is one batched **forward pass** (no decoding) → the energy cost is
prefill-only. This targets *solvability* — the signal draft agreement missed.

Needs `results/paired/drafts_v1.jsonl` in the repo (notebook 05 output).''')

md(r'''## 0. Repo setup (Colab or vast.ai)''')
code(r'''import os, subprocess, sys
REPO, BRANCH = "POLIMIGreenISE/Energy-AwareLLMReasoning", "main"
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
subprocess.run(["git", "config", "user.name", "your-name"], check=True)
subprocess.run(["git", "config", "user.email", "you@example.com"], check=True)
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd(), "| drafts present:", os.path.exists("results/paired/drafts_v1.jsonl"))''')

md(r'''## 1. Install (non-destructive — HF only, no vLLM needed)''')
code(r'''import importlib, subprocess, sys
def pipq(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    print(("ok  : " if r.returncode == 0 else "FAIL: ") + " ".join(args))
    if r.returncode != 0: print(r.stderr[-400:])
pipq("-U", "zeus-ml", "datasets", "pynvml", "accelerate")
try:
    importlib.import_module("transformers"); print("transformers: present")
except Exception:
    pipq("transformers>=4.51")
print("install done")''')

md(r'''## 2. Config''')
code(r'''import os
SMOKE_TEST = True
MODEL = "Qwen/Qwen3-8B"
FWD_BATCH = 16
SEED = 12345
OUTDIR = "results/paired"; os.makedirs(OUTDIR, exist_ok=True)
DRAFTS_JSONL = os.path.join(OUTDIR, "drafts_v1.jsonl")
PROBES_JSONL = os.path.join(OUTDIR, "probes_v1.jsonl")
VERIFY_PROMPT = ("{question}\n\nProposed answer: {draft}\n\n"
                 "Is the proposed answer correct? Reply with only Yes or No.")
NEED_PROMPT = ("{question}\n\nWould you need to reason carefully step by step to answer this "
               "correctly, or can you answer it directly and reliably? "
               "Reply with only Yes if you need careful reasoning, or No if you can answer directly.")
print("config ok | smoke:", SMOKE_TEST)''')

md(r'''## 3. Load queries + drafts

Queries come from `gen_v2.jsonl` (qid + question live in every generation row),
NOT from the labels file — so this notebook can run in the same GPU session as
notebooks 02/05, before the local labeling pass exists.''')
code(r'''import json
from collections import defaultdict
seen = {}
for l in open("results/paired/gen_v2.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    if d["qid"] not in seen:
        seen[d["qid"]] = {"qid": d["qid"], "subset": d["subset"], "question": d["question"]}
drafts = defaultdict(dict)
for l in open(DRAFTS_JSONL):
    d = json.loads(l); drafts[d["qid"]][d["rep"]] = d
queries = [p for p in seen.values() if len(drafts.get(p["qid"], {})) >= 3]
if SMOKE_TEST: queries = queries[:8]
print(f"gen_v2 qids: {len(seen)} | with >=3 drafts: {len(queries)}")''')

md(r'''## 4. Model + probe machinery (forward passes only)''')
code(r'''import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

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

def yes_no_ids():
    ys, ns = [], []
    for v in ["Yes", " Yes", "yes", " yes", "YES"]:
        ids = tok.encode(v, add_special_tokens=False)
        if len(ids) == 1: ys.append(ids[0])
    for v in ["No", " No", "no", " no", "NO"]:
        ids = tok.encode(v, add_special_tokens=False)
        if len(ids) == 1: ns.append(ids[0])
    return sorted(set(ys)), sorted(set(ns))
YES_IDS, NO_IDS = yes_no_ids()
print("yes ids:", YES_IDS, "| no ids:", NO_IDS)

def chatize(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def p_yes_batch(texts):
    """P(Yes | prompt) from the first generated token's logits."""
    out = []
    for i in range(0, len(texts), FWD_BATCH):
        chunk = [chatize(t) for t in texts[i:i+FWD_BATCH]]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        logp = logits.log_softmax(-1)
        py = torch.logsumexp(logp[:, YES_IDS], dim=-1)
        pn = torch.logsumexp(logp[:, NO_IDS], dim=-1)
        out += (py - torch.logaddexp(py, pn)).exp().cpu().tolist()
        print(f"  fwd {min(i+FWD_BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return out

_ = p_yes_batch(["Is 2+2 equal to 4? Reply with only Yes or No."])   # warm-up
print("warm-up p_yes(2+2=4):", round(_[0], 3))''')

md(r'''## 5. Run probes (resumable)''')
code(r'''done = set()
if os.path.exists(PROBES_JSONL):
    for l in open(PROBES_JSONL):
        try: done.add(json.loads(l)["qid"])
        except Exception: pass
pending = [p for p in queries if p["qid"] not in done]
print(f"pending: {len(pending)} (done {len(done)})")

if pending:
    ver_prompts, need_prompts = [], []
    for p in pending:
        q = p["question"]
        for k in range(3):
            ver_prompts.append(VERIFY_PROMPT.format(question=q, draft=drafts[p["qid"]][k]["draft_text"]))
        need_prompts.append(NEED_PROMPT.format(question=q))
    (ver, need), energy_j, secs = measure(lambda: (p_yes_batch(ver_prompts), p_yes_batch(need_prompts)))
    per_q_mwh = (energy_j / 3600.0) * 1000 / max(1, len(pending))
    with open(PROBES_JSONL, "a") as f:
        for i, p in enumerate(pending):
            f.write(json.dumps({
                "qid": p["qid"], "subset": p["subset"],
                "p_true": [round(ver[3*i + k], 5) for k in range(3)],
                "p_need_think": round(need[i], 5),
                "probe_energy_mwh": round(per_q_mwh, 3),
            }) + "\n")
    print(f"\n{len(pending)} queries | {secs:.0f}s | total {energy_j/3600:.2f} Wh "
          f"| {per_q_mwh:.1f} mWh per query (all 4 probes)")
print("done ->", PROBES_JSONL)''')

md(r'''## 6. Instant diagnostic — does P(True) know what agreement doesn't?''')
code(r'''import statistics as st
import numpy as np
probes = {json.loads(l)["qid"]: json.loads(l) for l in open(PROBES_JSONL)}
rows = [(p, probes[p["qid"]]) for p in paired if p["qid"] in probes]
if len(rows) >= 30:
    from sklearn.metrics import roc_auc_score
    cn = np.array([p["correct_no_think"] for p, _ in rows])
    ct = np.array([p["correct_think"] for p, _ in rows])
    sub = np.array([p["subset"] for p, _ in rows])
    pt = np.array([st.mean(pr["p_true"]) for _, pr in rows])
    pneed = np.array([pr["p_need_think"] for _, pr in rows])
    print("calibration: does mean P(true) predict no_think correctness?  AUC",
          round(roc_auc_score((cn > 0.5).astype(int), pt), 3))
    hard = sub == "underthinkingbench"
    solv = ((ct > 0.5) & (cn <= 0.5))[hard]
    print("solvability among HARD (1-p_true):  AUC",
          round(roc_auc_score(solv.astype(int), -pt[hard]), 3), " (drafts got 0.434)")
    print("p_need_think vs subset (easy/hard): AUC",
          round(roc_auc_score(hard.astype(int), pneed), 3))
else:
    print("smoke run - too few rows for AUCs; flip SMOKE_TEST=False and re-run from Section 3")''')

md(r'''## 7. Push results''')
code(r'''import subprocess
subprocess.run(["git", "add", "-f", PROBES_JSONL], check=True)
subprocess.run(["git", "commit", "-m", "confidence probes: first-token P(yes) verify + difficulty"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed (check output above)")''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]; c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "06_confidence_probe.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
