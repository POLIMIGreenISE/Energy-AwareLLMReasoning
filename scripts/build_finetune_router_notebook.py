"""Builds notebooks/07_finetune_router.ipynb — fine-tune ModernBERT-base and
mmBERT-small as the router encoder, replacing the frozen-MiniLM+LogReg setup.

Motivation (2026-07-20 finding): the learned router's embedding-based P(think)
score predicts solvability-among-hard at AUC 0.739 -- real signal invisible to
every confidence probe we tried (P(true) 0.46, p_need 0.41). That signal was
only partly exploited by a frozen embedding + linear head. A fine-tuned encoder
can extract more of it, AND (unlike p_need, which needs a forward pass through
the full 8B serving model, ~1.5 mWh/query measured) a small encoder's own
inference cost should be ~2 orders of magnitude cheaper -- this notebook
measures that directly with Zeus instead of estimating it.

Same 5-fold out-of-fold protocol as the existing router (StratifiedKFold, seed 0)
so results are directly comparable to results/draft_router/summary3.md.

Primary run: FRESH fine-tune of both base architectures on our own 1,710
energy-aware labels (verified label semantics). Mohseni's own fine-tuned
mmBERT-small checkpoint (AmirMohseni/reasoning-router-mmbert-small) is offered
as an optional warm-start toggle, OFF by default -- its model card does not
document the label index mapping, so trusting it blindly risks silently
inverting think/no_think.

Run:  python scripts/build_finetune_router_notebook.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Phase-3 extension — fine-tuned encoder router (3-way comparison)

Replaces the frozen-MiniLM-embedding + logistic-regression router with a
**fine-tuned encoder**, fully trained end-to-end on our 1,710 energy-aware
labels. Motivated by a finding this week: the current router's score already
predicts hard-query solvability at AUC 0.739 (vs ~0.46 for every confidence
probe we tried) — a signal a frozen linear head only partly exploits.

Trains **three** variants for comparison:
- `answerdotai/ModernBERT-base` (150M, fresh) — the vLLM Semantic Router's choice.
- `jhu-clsp/mmBERT-small` (140M, fresh) — the architecture Mohseni used.
- `AmirMohseni/reasoning-router-mmbert-small` (140M, **warm start**) — his own
  checkpoint, already fine-tuned on his released 70k-label dataset. Verified
  safe to warm-start from: its config confirms `id2label={0:"no_think",
  1:"think"}` — an exact match to our own convention, `ModernBertForSequence
  Classification` architecture, loads cleanly via `AutoModelForSequenceClassification`
  with both the encoder AND the classification head restored. Tests whether
  pretraining on his larger (if token-labeled, not energy-aware) dataset gives
  a head start over training from scratch on just our 1,710 examples.

Same 5-fold out-of-fold protocol as the existing router (StratifiedKFold,
seed=0) — directly comparable to `results/draft_router/summary3.md`.

Also **measures real inference energy with Zeus** for each fine-tuned model —
the point being that p_need needs a forward pass through the full 8B serving
model (~1.5 mWh/query, measured) while a small encoder should cost far less;
this notebook gets the real number instead of an estimate.

> GPU needed for training; ~20-40 min total for all three variants on an A100.
> Needs: repo cloned, `GEMINI_API_KEY` NOT needed here (no judging).''')

md(r'''## 0. Repo setup (Colab or vast.ai)''')
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

md(r'''## 1. Install

No vLLM needed here (classification only, no generation). Pins torch to a
**cu126 wheel explicitly** (the verified vast.ai fix: this fleet's driver
reports CUDA 12.6 — `pip install torch` alone can resolve a cu13 build that
fails with `driver too old (found 12060)`). Also removes the leftover broken
`amdsmi` package seen on the same fleet (Ray probes for AMD GPUs on import and
crashes on a stale ROCm shim otherwise).''')
code(r'''import subprocess, sys
def pipq(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    print(("ok  : " if r.returncode == 0 else "FAIL: ") + " ".join(args))
    if r.returncode != 0: print(r.stderr[-400:])

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "amdsmi"], capture_output=True)
pipq("--index-url", "https://download.pytorch.org/whl/cu126", "torch")
pipq("-U", "transformers>=4.51,<5", "accelerate", "scikit-learn", "zeus-ml", "pynvml")
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())''')

md(r'''## 2. Config''')
code(r'''SMOKE_TEST = True
MODELS = {
    "modernbert": "answerdotai/ModernBERT-base",
    "mmbert_small": "jhu-clsp/mmBERT-small",
    # Mohseni's own checkpoint, warm start -- verified safe: config.json confirms
    # id2label={0:"no_think",1:"think"}, exact match to our convention, so
    # AutoModelForSequenceClassification(num_labels=2) restores BOTH the encoder
    # and the classification head, not just the body.
    "mmbert_mohseni_warmstart": "AmirMohseni/reasoning-router-mmbert-small",
}

MAX_LEN = 256
N_FOLDS = 2 if SMOKE_TEST else 5
EPOCHS = 1 if SMOKE_TEST else 6
BATCH_SIZE = 16
LR = 2e-5
SEED = 0
OUTDIR = "results/paired"; os.makedirs(OUTDIR, exist_ok=True)
print(f"SMOKE_TEST={SMOKE_TEST} | folds={N_FOLDS} | epochs={EPOCHS}")''')

md(r'''## 3. Load labels''')
code(r'''import json
import numpy as np
paired = json.load(open("results/paired/paired_oracle_final.json"))
if SMOKE_TEST:
    # keep a few of each class so stratified splitting works
    think = [p for p in paired if p["oracle"] == "think"][:15]
    nothink = [p for p in paired if p["oracle"] == "no_think"][:25]
    paired = think + nothink
qids = [p["qid"] for p in paired]
texts = [p["question"] for p in paired]
y = np.array([1 if p["oracle"] == "think" else 0 for p in paired])
print(f"N={len(paired)} | positives (think): {y.sum()} ({y.mean()*100:.1f}%)")''')

md(r'''## 4. Zeus energy measurement helper (same pattern as notebooks 02/05/06)''')
code(r'''import time
try:
    from zeus.monitor import ZeusMonitor
    zmon = ZeusMonitor(gpu_indices=[0]); zeus_ok = True
except Exception as e:
    zeus_ok = False; print("Zeus unavailable:", repr(e)[:150])

def measure(fn):
    if zeus_ok:
        zmon.begin_window("w"); r = fn(); m = zmon.end_window("w")
        ej = getattr(m, "total_energy", None) or sum(m.gpu_energy.values())
        return r, ej, m.time
    t0 = time.time(); r = fn(); return r, float("nan"), time.time() - t0
print("zeus_ok:", zeus_ok)''')

md(r'''## 5. Fine-tuning: 5-fold out-of-fold training

Fresh model per fold (no leakage across folds). Class-weighted cross-entropy
(mirrors the LogReg router's `class_weight="balanced"`). Collects an honest
out-of-fold P(think) for every query -- comparable to the existing router's
`cross_val_predict` probabilities.''')
code(r'''import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class QDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "label": int(self.labels[i])}

def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); mask[i, :n] = torch.tensor(b["attention_mask"])
        labels[i] = b["label"]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def train_one_fold(model_id, train_texts, train_y, val_texts, epochs):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    n_pos, n_neg = int(train_y.sum()), int((1 - train_y).sum())
    w = torch.tensor([len(train_y) / (2 * n_neg), len(train_y) / (2 * n_pos)],
                     dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    ds = QDataset(train_texts, train_y, tok, MAX_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps),
                                            num_training_steps=total_steps)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        print(f"    epoch {ep+1}/{epochs}  loss {tot/len(dl):.4f}")

    model.eval()
    val_ds = QDataset(val_texts, [0] * len(val_texts), tok, MAX_LEN)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    probs = []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            probs += torch.softmax(logits, -1)[:, 1].cpu().tolist()
    return probs, model, tok

def run_cv(model_id, texts, y, qids, name):
    print(f"\n=== {name} ({model_id}) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    last_model = last_tok = None
    for fi, (tr, va) in enumerate(skf.split(texts, y)):
        t0 = time.time()
        probs, last_model, last_tok = train_one_fold(
            model_id, [texts[i] for i in tr], y[tr], [texts[i] for i in va], EPOCHS)
        oof[va] = probs
        print(f"  fold {fi+1}/{N_FOLDS}: {len(tr)} train / {len(va)} val | {time.time()-t0:.0f}s")
    try:
        auc = roc_auc_score(y, oof)
        print(f"  {name} out-of-fold label AUC: {auc:.4f}")
    except ValueError:
        auc = float("nan")
    out_path = f"{OUTDIR}/router_fte_{name}.jsonl"
    with open(out_path, "w") as f:
        for q, p in zip(qids, oof):
            f.write(json.dumps({"qid": q, "p_think": round(float(p), 5)}) + "\n")
    print(f"  wrote {out_path}")
    return oof, auc, last_model, last_tok

results = {}
for name, model_id in MODELS.items():
    oof, auc, model, tok = run_cv(model_id, texts, y, qids, name)
    results[name] = dict(auc=auc, model=model, tok=tok)''')

md(r'''## 5b. Spot-check the warm-started model

Quick confirmation (not a gate — the config already verified the label mapping
matches ours) that the fine-tuned Mohseni checkpoint gives sane predictions.''')
code(r'''m, t = results["mmbert_mohseni_warmstart"]["model"], results["mmbert_mohseni_warmstart"]["tok"]
probes = ["What is 2 + 2?", "Prove that the square root of 2 is irrational, showing every step."]
m.eval()
with torch.no_grad():
    enc = t(probes, return_tensors="pt", padding=True, truncation=True).to(device)
    p = torch.softmax(m(**enc).logits, -1)[:, 1].cpu().tolist()
print("P(think) [easy, hard]:", [round(x, 3) for x in p], "(expect LOW then HIGH)")''')

md(r'''## 6. Measure REAL inference energy per query (Zeus) -- the other half of the question

p_need needs a forward pass through the full 8B serving model (~1.5 mWh/query,
measured in notebook 06). These classifiers are 50-300x fewer parameters --
this cell gets the real per-query cost instead of a FLOPs-based guess.''')
code(r'''def measure_inference_energy(model, tok, texts, batch_size=32, warmup=2):
    model.eval()
    ds = QDataset(texts, [0] * len(texts), tok, MAX_LEN)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    batches = list(dl)
    with torch.no_grad():
        for b in batches[:warmup]:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            model(**b)
    def run_all():
        with torch.no_grad():
            for b in batches:
                b = {k: v.to(device) for k, v in b.items() if k != "labels"}
                model(**b)
    _, energy_j, secs = measure(run_all)
    n = len(texts)
    mwh_per_q = (energy_j / 3600.0) * 1000 / n if energy_j == energy_j else float("nan")
    print(f"  {n} queries | {secs:.1f}s | {energy_j:.2f} J total | {mwh_per_q:.4f} mWh/query")
    return mwh_per_q

energy_summary = {}
for name in results:
    print(f"\nmeasuring inference energy: {name}")
    energy_summary[name] = measure_inference_energy(results[name]["model"], results[name]["tok"], texts)
print("\n--- for comparison ---")
print("p_need probe (8B forward pass, measured notebook 06): ~1.5 mWh/query")
print("fine-tuned encoders (measured above):", {k: round(v, 4) for k, v in energy_summary.items()}, "mWh/query")

with open(f"{OUTDIR}/router_fte_energy.json", "w") as f:
    json.dump({name: energy_summary[name] for name in results}, f, indent=1)
print(f"wrote {OUTDIR}/router_fte_energy.json (needed by the local EAA comparison script)")''')

md(r'''## 7. Push results (force-add — `*.jsonl`/`*.json` outputs are gitignored)''')
code(r'''import subprocess
files = [f"{OUTDIR}/router_fte_{name}.jsonl" for name in results] + [f"{OUTDIR}/router_fte_energy.json"]
subprocess.run(["git", "add", "-f", *files], check=True)
msg = f"fine-tuned encoder router OOF predictions ({', '.join(results)})"
r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
print(r.stdout[-300:] or r.stderr[-300:])
r = subprocess.run(["git", "push"], capture_output=True, text=True)
print("push:", (r.stdout + r.stderr)[-300:])''')

md(r'''## 8. Summary

|architecture|out-of-fold label AUC|measured inference energy (mWh/query)|
|--|--|--|
(3 rows: modernbert, mmbert_small fresh, mmbert_mohseni_warmstart)

run the cell below to fill this in, then compare against:
- text-only LogReg+MiniLM router: label AUC 0.900
- p_need rule: label AUC 0.782 (but EAA 0.9103, its ranking quality is not the
  full story -- see results/draft_router/summary3.md for why)

Full EAA/Pareto comparison happens **locally, offline**, in a follow-up script
that joins `router_fte_*.jsonl` with `paired_oracle_final.json` — the same
honest-energy-accounting machinery already used for every other router variant.''')
code(r'''print(f"{'architecture':15} | {'label AUC':>10} | {'mWh/query':>10}")
for name in results:
    print(f"{name:15} | {results[name]['auc']:10.4f} | {energy_summary[name]:10.4f}")''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    s = c["source"]; c["source"] = s.splitlines(keepends=True) if isinstance(s, str) else s
os.makedirs("notebooks", exist_ok=True)
out = os.path.join("notebooks", "07_finetune_router.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
