"""Builds notebooks/08_generalization_testset.ipynb (GPU) — cross-dataset test.

Generates Qwen3-8B think+no_think answers on a HELD-OUT test dataset (never seen
in training), runs the p_need probe, and scores our OTB-trained routers on the
new prompts -- so we can test whether the router generalizes off OptimalThinkingBench.

Two datasets via one flag:
  DATASET="gsm8k"   -> clean gold answers (difficulty continuum; judged vs gold locally)
  DATASET="mohseni" -> AmirMohseni/reasoning-router-data-v2 test split, NO gold
                       (judged reference-free locally: a strong judge grades both
                        Qwen3 answers, generating its own ground truth)

Routers trained on ALL of OTB (no leakage -- test set is a different dataset),
predicted on the new prompts:
  - LogReg + MiniLM   (frozen embeddings)
  - fine-tuned ModernBERT   (retrained here; weights weren't saved from nb07)
  - p_need probe            (Qwen3-8B single forward pass per query)

Run:  python scripts/build_testset_notebook.py
Then: run the notebook on vast.ai, push, and locally run scripts/eval_testset.py
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# Phase-3 generalization — Qwen3-8B on a held-out test dataset

Tests whether our OTB-trained router transfers to a DIFFERENT distribution.
Generates think+no_think answers, the p_need probe, and OTB-trained router
scores on the new prompts. Labels + evaluation happen locally afterward
(`scripts/eval_testset.py`): GSM8K judged against gold, Mohseni judged
reference-free (a strong judge grades both answers, generating its own truth).

> Set `DATASET` in Section 2. Expected: ~1 GPU-hour for ~1,800 queries × 2 modes.''')

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
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())''')

md(r'''## 1. Install (verified vast.ai recipe — CUDA 12.6 driver)

If the preinstalled torch is CUDA-13 (`driver too old (found 12060)`), Ray crashes
on `libamd_smi.so`, or the tokenizer lacks `all_special_tokens_extended`, run once
then restart the kernel:
```
%pip uninstall -y amdsmi vllm vllm-flash-attn xformers flashinfer-python torch torchvision torchaudio
%pip install "vllm>=0.10,<0.11" "transformers>=4.51,<5"
%pip uninstall -y amdsmi ray torchcodec   # vllm re-pulls broken amdsmi; removing ray kills the AMD-probe crash path (single-GPU needs no ray)
```''')
code(r'''import subprocess, sys
def pipq(*a):
    r = subprocess.run([sys.executable,"-m","pip","install","-q",*a], capture_output=True, text=True)
    print(("ok  : " if r.returncode==0 else "FAIL: ")+" ".join(a))
    if r.returncode: print(r.stderr[-300:])
# Ray's GPU probe crashes on the broken preinstalled amdsmi (KeyError libamd_smi.so).
# Removing ray kills the crash path (single-GPU vLLM never uses ray); also drop amdsmi.
subprocess.run([sys.executable,"-m","pip","uninstall","-y","amdsmi","ray"], capture_output=True)
pipq("-U","zeus-ml","datasets","pyarrow","accelerate","scikit-learn","sentence-transformers")
# newer sentence-transformers pulls torchcodec (audio/video); on this box it fails
# to load its FFmpeg libs and raises RuntimeError (which s-t doesn't catch) -> the
# `from sentence_transformers import ...` in section 7 crashes. Remove torchcodec so
# s-t's `except (ImportError, OSError)` catches the clean ModuleNotFoundError instead.
subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchcodec"], capture_output=True)
import importlib
for m in ("vllm","transformers"):
    try: importlib.import_module(m); print(m,"ok")
    except Exception as e: print(m,"BROKEN:",repr(e)[:100])
try: importlib.import_module("sentence_transformers"); print("sentence_transformers ok")
except Exception as e: print("sentence_transformers BROKEN:",repr(e)[:100])''')

md(r'''## 2. Config''')
code(r'''DATASET = "mohseni"          # "mohseni" (no gold) or "gsm8k" (gold)
SMOKE_TEST = True
MODEL = "Qwen/Qwen3-8B"
N_TEST = 8 if SMOKE_TEST else 1812          # mohseni test = 1812; gsm8k up to 1319
REPS = 1                                     # test set: 1 rep per mode is enough
THINK_MAX, NOTHINK_MAX = 16384, 2048
GEN_BATCH = 64; J_PER_OUT_TOKEN = 3.7; SEED = 12345
OUT = "results/crossdataset"; os.makedirs(OUT, exist_ok=True)
GEN_JSONL = f"{OUT}/gen_test_{DATASET}.jsonl"
NEED_PROMPT = ("{q}\n\nWould you need to reason carefully step by step to answer this correctly, "
               "or can you answer it directly and reliably? Reply with only Yes if you need careful "
               "reasoning, or No if you can answer directly.")
print(f"DATASET={DATASET} N_TEST={N_TEST} REPS={REPS} SMOKE={SMOKE_TEST}")''')

md(r'''## 3. Load the held-out test dataset''')
code(r'''import json
items = []
if DATASET == "mohseni":
    import pyarrow.parquet as pq, urllib.request
    url = "https://huggingface.co/datasets/AmirMohseni/reasoning-router-data-v2/resolve/main/data/test-00000-of-00001.parquet"
    urllib.request.urlretrieve(url, "/tmp/moh_test.parquet")
    d = pq.read_table("/tmp/moh_test.parquet").to_pydict()
    for i in range(len(d["prompt"])):
        items.append({"qid": f"moh_{i:04d}", "question": d["prompt"][i], "gold": None,
                      "his_label": d["label"][i], "source": d["source"][i]})
elif DATASET == "gsm8k":
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip()
        items.append({"qid": f"gsm_{i:04d}", "question": r["question"], "gold": gold,
                      "his_label": None, "source": "gsm8k"})
items = items[:N_TEST]
print(f"loaded {len(items)} {DATASET} queries")
print("example:", items[0]["question"][:100], "| gold:", items[0]["gold"], "| his_label:", items[0]["his_label"])''')

md(r'''## 4. Serving + energy (vLLM + Zeus)''')
code(r'''import time, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=20480)
try:
    from zeus.monitor import ZeusMonitor
    zmon = ZeusMonitor(gpu_indices=[0]); zeus_ok = True
except Exception as e:
    zeus_ok = False; print("Zeus off:", repr(e)[:80])
def measure(fn):
    if zeus_ok:
        zmon.begin_window("w"); r = fn(); m = zmon.end_window("w")
        return r, (getattr(m,"total_energy",None) or sum(m.gpu_energy.values())), m.time
    t0=time.time(); r=fn(); return r, float("nan"), time.time()-t0
def build_prompt(q, think):
    return tok.apply_chat_template([{"role":"user","content":q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=think)
def split_think(t):
    if "</think>" in t:
        a,_,b = t.partition("</think>"); return a.replace("<think>","").strip(), b.strip()
    return "", t.strip()
SAMP = {"think":dict(temperature=0.6,top_p=0.95,top_k=20,max_tokens=THINK_MAX),
        "no_think":dict(temperature=0.7,top_p=0.80,top_k=20,max_tokens=NOTHINK_MAX)}
def gen_batch(qs, mode):
    sp=SAMP[mode]; prompts=[build_prompt(q["question"],mode=="think") for q in qs]
    params=SamplingParams(**{k:sp[k] for k in ("temperature","top_p","top_k","max_tokens")})
    outs,ej,secs = measure(lambda: llm.generate(prompts,params,use_tqdm=False))
    tot=sum(len(o.outputs[0].token_ids) for o in outs) or 1; rows=[]
    for q,o in zip(qs,outs):
        _,ans = split_think(o.outputs[0].text); ct=len(o.outputs[0].token_ids)
        rows.append({**q,"mode":mode,"answer_text":ans[:3000],"completion_tokens":ct,
                     "finish_reason":o.outputs[0].finish_reason,
                     "energy_share_wh":(ej*ct/tot)/3600.0,"energy_b1eq_wh":ct*J_PER_OUT_TOKEN/3600.0})
    print(f"  [{mode}] {len(rows)} gens {secs:.0f}s"); return rows
_=gen_batch(items[:2],"no_think"); print("warm-up done")''')

md(r'''## 5. Generate (resumable)''')
code(r'''done=set()
if os.path.exists(GEN_JSONL):
    for l in open(GEN_JSONL):
        try: d=json.loads(l); done.add((d["qid"],d["mode"],d["rep"]))
        except: pass
with open(GEN_JSONL,"a") as f:
    for mode in ["no_think","think"]:
        for rep in range(REPS):
            pend=[q for q in items if (q["qid"],mode,rep) not in done]
            print(f"{mode} rep{rep}: {len(pend)} pending")
            for i in range(0,len(pend),GEN_BATCH):
                for row in gen_batch(pend[i:i+GEN_BATCH],mode):
                    row["rep"]=rep; f.write(json.dumps(row)+"\n")
                f.flush()
print("generation done ->",GEN_JSONL)''')

md(r'''## 6. p_need probe on the test prompts (one forward pass/query)

**RECOMMENDED: restart the kernel after generation, then run sections 0 → 2 → 3 → 6.**
The p_need probe needs raw logits (HF model), and in-process vLLM memory can't be
reliably freed (its tensors stay referenced), so two copies of Qwen3-8B OOM. A fresh
kernel gives the whole GPU to the HF model; generations are already on disk, so skip
sections 4-5. This cell attempts a vLLM teardown for the same-kernel path and loads
the tokenizer itself if section 4 was skipped.''')
code(r'''import gc, torch
# best-effort free of any vLLM engine still resident (unreliable in-process; restart is safer)
try: del llm
except NameError: pass
try:
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    destroy_model_parallel(); destroy_distributed_environment()
except Exception: pass
gc.collect(); torch.cuda.empty_cache()
free_gb = torch.cuda.mem_get_info()[0]/1e9
print(f"free GPU: {free_gb:.1f} GB (need ~16 for the 8B HF model)")
assert free_gb > 17, ("Not enough free GPU — vLLM didn't release. RESTART the kernel, then run "
                      "sections 0/2/3 and re-run this cell. Generations are safe on disk.")

from transformers import AutoTokenizer, AutoModelForCausalLM
try: tok            # section 4 defines it; may be absent after a kernel restart
except NameError: tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
tok.truncation_side = "left"   # keep the end (the Yes/No instruction) on long chat prompts
def yes_no_ids():
    ys,ns=[],[]
    for v in ["Yes"," Yes","yes"," yes"]:
        e=tok.encode(v,add_special_tokens=False);  ys+= [e[0]] if len(e)==1 else []
    for v in ["No"," No","no"," no"]:
        e=tok.encode(v,add_special_tokens=False);  ns+= [e[0]] if len(e)==1 else []
    return sorted(set(ys)), sorted(set(ns))
YES,NO = yes_no_ids()
def chatize(t): return tok.apply_chat_template([{"role":"user","content":t}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
@torch.no_grad()
def p_need_batch(texts, B=8):
    out=[]
    for i in range(0,len(texts),B):
        enc=tok([chatize(t) for t in texts[i:i+B]], return_tensors="pt", padding=True,
                padding_side="left", truncation=True, max_length=4096).to("cuda")
        try:
            logits = mdl(**enc, logits_to_keep=1).logits   # [B,1,vocab]: only the last token, avoids [B,seq,vocab] OOM
        except TypeError:
            logits = mdl(**enc).logits                     # older transformers fallback (truncation bounds it)
        lp = logits[:,-1,:].float().log_softmax(-1)
        py=torch.logsumexp(lp[:,YES],-1); pn=torch.logsumexp(lp[:,NO],-1)
        out+=(py-torch.logaddexp(py,pn)).exp().cpu().tolist()
        del enc, logits, lp; torch.cuda.empty_cache()
    return out
pn_prompts=[NEED_PROMPT.format(q=q["question"]) for q in items]
(pneed,), ej, secs = measure(lambda:(p_need_batch(pn_prompts),))
per_q = (ej/3600.0)*1000/max(1,len(items))
with open(f"{OUT}/pneed_test_{DATASET}.jsonl","w") as f:
    for q,p in zip(items,pneed):
        f.write(json.dumps({"qid":q["qid"],"p_need_think":round(p,5),"probe_energy_mwh":round(per_q,3)})+"\n")
print(f"p_need probe: {len(items)} queries {secs:.0f}s -> {OUT}/pneed_test_{DATASET}.jsonl")''')

md(r'''## 7. Train OTB routers, predict on the NEW prompts (held-out generalization)

Trains on ALL of OTB (no leakage — the test set is a different dataset), predicts
`P(think)` on the test prompts. LogReg+MiniLM and fine-tuned ModernBERT.''')
code(r'''import numpy as np, json
otb = json.load(open("results/paired/paired_oracle_final.json"))
y_otb = np.array([p["oracle"]=="think" for p in otb]).astype(int)
test_q = [q["question"] for q in items]

# --- LogReg + MiniLM
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
emb = SentenceTransformer("all-MiniLM-L6-v2")
def feats(qs):
    Xe=emb.encode(qs,normalize_embeddings=True,show_progress_bar=False)
    ql=np.array([len(x) for x in qs]).reshape(-1,1)/1000.0
    return np.hstack([Xe,ql])
lr=LogisticRegression(max_iter=2000,C=1.0,class_weight="balanced").fit(feats([p["question"] for p in otb]),y_otb)
p_lr = lr.predict_proba(feats(test_q))[:,1]

# --- fine-tuned ModernBERT (retrain on OTB, predict on test)
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
mid="answerdotai/ModernBERT-base"; mtok=AutoTokenizer.from_pretrained(mid)
mm=AutoModelForSequenceClassification.from_pretrained(mid,num_labels=2).to("cuda")
def enc_batch(texts):
    e=mtok(texts,truncation=True,max_length=256,padding=True,return_tensors="pt"); return {k:v.to("cuda") for k,v in e.items()}
otb_texts=[p["question"] for p in otb]
w=torch.tensor([len(y_otb)/(2*(y_otb==0).sum()),len(y_otb)/(2*(y_otb==1).sum())],dtype=torch.float,device="cuda")
lossf=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(mm.parameters(),lr=2e-5,weight_decay=0.01)
EP=1 if SMOKE_TEST else 6; idx=np.arange(len(otb_texts)); BS=16
steps=(len(idx)//BS+1)*EP; sch=get_linear_schedule_with_warmup(opt,int(0.1*steps),steps)
mm.train()
for ep in range(EP):
    np.random.shuffle(idx); tot=0
    for i in range(0,len(idx),BS):
        b=idx[i:i+BS]; enc=enc_batch([otb_texts[j] for j in b]); yb=torch.tensor(y_otb[b],device="cuda")
        opt.zero_grad(); out=mm(**enc); l=lossf(out.logits,yb); l.backward()
        torch.nn.utils.clip_grad_norm_(mm.parameters(),1.0); opt.step(); sch.step(); tot+=l.item()
    print(f"  ModernBERT epoch {ep+1}/{EP} loss {tot/(len(idx)//BS+1):.4f}")
mm.eval(); p_mb=[]
with torch.no_grad():
    for i in range(0,len(test_q),32):
        enc=enc_batch(test_q[i:i+32]); p_mb+=torch.softmax(mm(**enc).logits,-1)[:,1].cpu().tolist()
p_mb=np.array(p_mb)

with open(f"{OUT}/routerscores_test_{DATASET}.jsonl","w") as f:
    for q,a,b in zip(items,p_lr,p_mb):
        f.write(json.dumps({"qid":q["qid"],"p_logreg":round(float(a),5),"p_modernbert":round(float(b),5)})+"\n")
print(f"router scores -> {OUT}/routerscores_test_{DATASET}.jsonl")''')

md(r'''## 8. Push''')
code(r'''import subprocess
fs=[GEN_JSONL,f"{OUT}/pneed_test_{DATASET}.jsonl",f"{OUT}/routerscores_test_{DATASET}.jsonl"]
subprocess.run(["git","add","-f",*fs],check=True)
r=subprocess.run(["git","commit","-m",f"generalization test set: {DATASET}"],capture_output=True,text=True)
print(r.stdout[-200:] or r.stderr[-200:])
r=subprocess.run(["git","push"],capture_output=True,text=True); print("push:",(r.stdout+r.stderr)[-200:])''')

md(r'''## 9. Next (local)

`python scripts/eval_testset.py --dataset mohseni` (or gsm8k):
- GSM8K: judge answers vs gold (numeric).
- Mohseni: reference-free — a strong judge (`GEMINI_MODEL=gemini-2.5-pro` or Sonnet)
  grades both Qwen3 answers, generating its own ground truth -> energy-aware oracle.
- Builds oracle labels, evaluates the OTB-trained routers (agreement with Mohseni's
  labels; energy-accuracy Pareto on the new distribution).''')

nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[],"gpuType":"A100"},
    "kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
for c in nb["cells"]:
    s=c["source"]; c["source"]=s.splitlines(keepends=True) if isinstance(s,str) else s
os.makedirs("notebooks",exist_ok=True)
out=os.path.join("notebooks","08_generalization_testset.ipynb")
json.dump(nb,open(out,"w",encoding="utf-8"),indent=1)
print(f"Wrote {out} with {len(cells)} cells.")
