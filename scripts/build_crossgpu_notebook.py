"""Builds notebooks/12_cross_gpu_curve.ipynb (GPU) — E14.

Answers the first question any examiner asks: **one GPU, one model — does any of this generalise?**

The thesis rests on J(L), the measured joules-per-output-token as a function of server concurrency,
which falls 42x from batch 1 to 64 on an A100-SXM4-40GB. Every downstream claim uses that curve. If
it is an A100 artefact, RQ4 is an A100 result.

The key observation is that **the routing policy does not need the absolute joules — it needs the
SHAPE.** The rule is

    think(q)  <=>  score(q) > lambda * J(L) * E[dtok]

and lambda is a free parameter re-tuned per deployment. Multiply J(L) by any constant and the same
decisions come back at lambda/constant. So the question is not "is J(L) the same on other hardware"
(it certainly is not) but "is J(L)/J(1) the same" — whether the *relative* price of a token at low
versus high concurrency is a property of batched transformer decoding or a property of this card.

This notebook re-runs the ORIGINAL concurrency protocol unchanged (l_in=64, l_out=256, batches
1..64) on whatever GPU it is placed on, then reports three things:

  1. the raw curve, tagged by GPU, appended alongside the A100 reference;
  2. whether the SHAPE transfers -- J(L)/J(1) against the A100, and the spread J(1)/J(64);
  3. whether the CONCLUSION transfers -- a reduced E8 comparison re-run on the new curve, to see
     whether load-aware still beats a static threshold on bursty traffic.

Point 3 is the one that matters for the thesis. A shape that differs slightly but leaves the policy
ranking intact is a much weaker threat than one that flips it.

**Best GPU to run this on: an L4.** It is the sharpest available test precisely because it is the
least like an A100 -- Ada rather than Ampere, and a 72 W board power envelope against 400 W. If the
normalised shape survives *that*, it is not a property of the card. An H100 is the useful other
direction (newer, faster, 80 GB). Either is cheap for ~45 minutes.

Run:  python scripts/build_crossgpu_notebook.py
Then: run notebooks/12_cross_gpu_curve.ipynb on a NON-A100 box and push the outputs.
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# E14 — does the energy curve transfer to another GPU?

The thesis rests on **J(L)**, joules per output token as a function of concurrency, measured on an
A100-SXM4-40GB as falling **42×** from batch 1 to batch 64. Everything downstream uses it.

The policy does **not** need the absolute joules — only the shape:

```
think(q)  ⟺  score(q) > λ · J(L) · E[Δtokens]
```

λ is re-tuned per deployment, so multiplying J(L) by any constant reproduces the same decisions at
λ/constant. The real question is therefore whether **J(L)/J(1)** transfers — whether the *relative*
price of a token at low versus high concurrency is a property of batched transformer decoding or a
property of this particular card.

**Run this on an L4 if you can.** It is the sharpest test available: Ada rather than Ampere, 72 W
board power against the A100's 400 W. If the normalised shape survives that, it is not a card
artefact. An H100 is the useful opposite direction.

**Runtime ~45 min**, most of it the model download.''')

md(r'''## 0. Repo setup''')
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
os.makedirs("results/batching", exist_ok=True)
print("cwd:", os.getcwd())''')

md(r'''## 1. Install

Same vast.ai recipe as the other notebooks. **vLLM re-pulls a broken `amdsmi` and Ray's GPU probe
then dies with `KeyError: 'libamd_smi.so'`** — uninstall `ray` too, restart the kernel, re-run from 0.''')
code(r'''import importlib, subprocess, sys
def have(m):
    try: importlib.import_module(m); return True
    except Exception: return False
if not have("vllm"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
for pkg in ["zeus-ml", "aiohttp", "nest_asyncio"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "amdsmi", "ray", "torchcodec"], check=False)
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("If anything was installed/removed: Kernel -> Restart, then re-run from 0.")''')

md(r'''## 2. Launch the server

`max_model_len` is stepped down until the server fits. The sweep only ever uses 64 + 256 = 320
tokens, so a smaller window changes nothing about the measurement — but it is **recorded in every
row**, because it is a deviation from the A100 run (16384) and the reader should see it.''')
code(r'''MODEL = "Qwen/Qwen3-8B"          # identical to the A100 run — the GPU is the only variable
PORT, GPU_UTIL = 8000, 0.90
OUT = "results/batching"

import subprocess, sys, time, os, json, urllib.request, torch
GPU_NAME = torch.cuda.get_device_name(0)
SLUG = "".join(c if c.isalnum() else "_" for c in GPU_NAME).strip("_").lower()
print("GPU:", GPU_NAME, "| slug:", SLUG)
if "A100" in GPU_NAME:
    print("\n*** WARNING: this is an A100 — the point of E14 is to run on DIFFERENT hardware. ***")

def server_up(port=PORT, timeout=1.0):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout); return True
    except Exception: return False

MAX_LEN = None
if not server_up():
    for candidate in (16384, 8192, 4096, 2048):
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL, "--port", str(PORT), "--max-model-len", str(candidate),
               "--gpu-memory-utilization", str(GPU_UTIL), "--disable-log-requests",
               "--max-num-seqs", "64"]
        print(f"trying max_model_len={candidate} ...")
        logf = open(f"{OUT}/server_{SLUG}.log", "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        ok = False
        for i in range(420):
            if server_up(): ok = True; break
            if proc.poll() is not None: break
            time.sleep(2)
            if i % 30 == 0 and i: print(f"   ... {i*2}s")
        if ok:
            MAX_LEN = candidate; print(f"server up with max_model_len={candidate}"); break
        try: proc.kill()
        except Exception: pass
        print(f"   did not start at {candidate}; stepping down")
    if MAX_LEN is None:
        print("SERVER WOULD NOT START — last 40 log lines:")
        print("".join(open(f"{OUT}/server_{SLUG}.log").readlines()[-40:])); raise SystemExit(1)
else:
    MAX_LEN = 16384
print("server up:", server_up(), "| max_model_len:", MAX_LEN)''')

md(r'''## 3. The sweep — the A100 protocol, unchanged

`l_in=64`, `l_out=256`, batches 1→64, 6 repetitions, `ignore_eos` so every request generates exactly
256 tokens. Identical to `results/batching/concurrency.jsonl`; only the hardware differs. Rows append
after every batch size, so a crash costs one batch rather than the run.''')
code(r'''import asyncio, aiohttp, numpy as np
import nest_asyncio; nest_asyncio.apply()
from zeus.monitor import ZeusMonitor
mon = ZeusMonitor(gpu_indices=[0])

L_IN, L_OUT, BATCHES, REPS = 64, 256, [1, 2, 4, 8, 16, 32, 64], 6
PAD = "The quick brown fox jumps over the lazy dog. "          # ~10 tokens
PROMPT = (PAD * ((L_IN // 10) + 2))[:L_IN * 6]
NO_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
PATH = f"{OUT}/concurrency_{SLUG}.jsonl"

def measure_idle(seconds=20):
    mon.begin_window("idle"); time.sleep(seconds); m = mon.end_window("idle")
    return m.gpu_energy[0] / seconds
IDLE_W = measure_idle(20)
print(f"idle power on this GPU: {IDLE_W:.1f} W")

async def fire(session):
    body = dict(model=MODEL, messages=[{"role": "user", "content": PROMPT}],
                max_tokens=L_OUT, temperature=0, ignore_eos=True,
                chat_template_kwargs={"enable_thinking": False})
    async with session.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
        d = await r.json()
    return (d.get("usage", {}) or {})

async def run_batch(n, rep):
    conn = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=conn, timeout=NO_TIMEOUT) as s:
        await fire(s)                                    # warm up, never measured
        mon.begin_window(f"b{n}_{rep}"); t0 = time.time()
        us = await asyncio.gather(*[fire(s) for _ in range(n)])
        el = time.time() - t0; m = mon.end_window(f"b{n}_{rep}")
    e_j = m.gpu_energy[0]
    out_tok = sum(u.get("completion_tokens", 0) for u in us)
    in_tok = sum(u.get("prompt_tokens", 0) for u in us)
    return dict(experiment="concurrency", l_in=L_IN, l_out=L_OUT, batch=n, rep=rep, n=n,
                energy_j=e_j, time_s=el, out_tokens=out_tok, in_tokens=in_tok,
                energy_wh=e_j / 3600.0, energy_per_query_wh=e_j / 3600.0 / n,
                energy_per_out_token_j=e_j / max(out_tok, 1),
                throughput_tok_s=out_tok / el, power_w=e_j / el,
                model=MODEL, gpu=GPU_NAME, max_model_len=MAX_LEN, idle_w=IDLE_W,
                ts=__import__("datetime").datetime.now().isoformat())

rows = []
for n in BATCHES:
    got = [await run_batch(n, r) for r in range(REPS)]
    rows += got
    with open(PATH, "a", encoding="utf-8") as f:
        for g in got: f.write(json.dumps(g) + "\n")
    jt = np.mean([g["energy_per_out_token_j"] for g in got])
    pw = np.mean([g["power_w"] for g in got])
    tp = np.mean([g["throughput_tok_s"] for g in got])
    print(f"  batch {n:3d}: {jt:6.3f} J/token | {pw:6.1f} W | {tp:7.1f} tok/s", flush=True)
print("wrote", PATH)''')

md(r'''## 4. Does the SHAPE transfer?

Absolute joules will differ — different silicon, different power envelope. What matters for the
policy is the normalised curve **J(L)/J(1)**, because λ absorbs any constant factor.''')
code(r'''import statistics as st
from collections import defaultdict

def curve_from(path):
    agg = defaultdict(list); meta = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        agg[r["batch"]].append(r); meta = r
    B = sorted(agg)
    return (B,
            {b: st.mean(x["energy_per_out_token_j"] for x in agg[b]) for b in B},
            {b: st.mean(x["power_w"] for x in agg[b]) for b in B},
            {b: st.mean(x["throughput_tok_s"] for x in agg[b]) for b in B},
            meta.get("gpu", "?"))

B_new, J_new, P_new, T_new, gpu_new = curve_from(PATH)
B_ref, J_ref, P_ref, T_ref, gpu_ref = curve_from(f"{OUT}/concurrency.jsonl")
common = [b for b in B_new if b in J_ref]

print(f"{'batch':>6} | {'A100 J/tok':>11} {'norm':>6} | {gpu_new[:22]:>22} {'norm':>6} | {'shape err':>9}")
errs = []
for b in common:
    n_ref, n_new = J_ref[b] / J_ref[1], J_new[b] / J_new[1]
    e = 100 * (n_new - n_ref) / n_ref; errs.append(abs(e))
    print(f"{b:>6} | {J_ref[b]:>11.3f} {n_ref:>6.3f} | {J_new[b]:>22.3f} {n_new:>6.3f} | {e:>+8.1f}%")

bmax = max(common)
sp_ref, sp_new = J_ref[1] / J_ref[bmax], J_new[1] / J_new[bmax]
print(f"\nspread J(1)/J({bmax}):  A100 {sp_ref:.1f}x   {gpu_new} {sp_new:.1f}x")
print(f"power flatness P({bmax})/P(1):  A100 {P_ref[bmax]/P_ref[1]:.2f}x   "
      f"{gpu_new} {P_new[bmax]/P_new[1]:.2f}x")
print(f"absolute J/token ratio at batch 1: {J_new[1]/J_ref[1]:.2f}x")
print(f"\nmean |shape error| across batch sizes: {np.mean(errs):.1f}%  (max {np.max(errs):.1f}%)")
print("SHAPE TRANSFERS — lambda absorbs the rest." if np.mean(errs) < 25 else
      "SHAPE DIFFERS MATERIALLY — the policy needs per-GPU calibration, which is itself a finding.")''')

md(r'''## 5. Does the CONCLUSION transfer?

The shape is a proxy; this is the question that matters. The E8 comparison is re-run on the **new
GPU's curve** — bursty regime, matched energy, the same densely-swept static baseline and the same
deployable load-aware rule. If load-aware still wins on bursty traffic, RQ4's result is a property
of batched serving rather than of the A100.

Reduced for runtime (6 traces rather than 20), so read the sign and rough magnitude, not the third
decimal.''')
code(r'''from src.serving_sim import Hardware, Workload, make_trace, simulate

wl = Workload(); SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
hw_ref = Hardware()
hw_new = Hardware(curves=dict(BATCHES=B_new, TPUT=T_new, POWER=P_new, JTOK=J_new))
REG, N, SEEDS = dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0), 400, range(6)

_sv = np.unique([SCORE[q] for q in wl.qids])
TAU = np.unique(np.concatenate([[0.0], _sv[::max(1, len(_sv)//120)], _sv[-3:], [1.01]]))

def sweep(hw):
    cand = np.concatenate([TAU / max(hw.jtok(L) * DT_MEAN, 1e-12) for L in (1, 4, 16, 48)])
    cand = np.unique(cand[np.isfinite(cand) & (cand > 0)])
    LAM = cand[::max(1, len(cand) // 160)]
    def run(dec):
        rs = [simulate(make_trace(wl, N, s, **REG), dec, wl, hw) for s in SEEDS]
        return (float(np.mean([r["energy_wh"] for r in rs])), float(np.mean([r["acc"] for r in rs])))
    nev = run(lambda L, q, c: False)[0]; alw = run(lambda L, q, c: True)[0]
    st_pts = [run((lambda t: (lambda L, q, c: SCORE[q] >= t))(t)) for t in TAU]
    la_pts = [run((lambda l: (lambda L, q, c: SCORE[q] > l * hw.jtok(L) * DT_MEAN))(l)) for l in LAM]
    out = {}
    for frac in (0.25, 0.40):
        cap = nev + frac * (alw - nev)
        best = lambda pts: max([a for e, a in pts if e <= cap + 1e-9], default=float("nan"))
        out[frac] = (best(st_pts), best(la_pts))
    return nev, alw, out

for name, hw in (("A100 (reference)", hw_ref), (gpu_new, hw_new)):
    nev, alw, out = sweep(hw)
    print(f"\n{name}: never {nev:.1f} Wh -> always {alw:.1f} Wh (spread {hw.jtok(1)/hw.jtok(max(hw.B)):.0f}x)")
    for frac, (s, l) in out.items():
        print(f"   cap {int(frac*100)}%: static {s:.4f}  load-aware {l:.4f}   gap {100*(l-s):+.2f}pp")''')

md(r'''## 6. Figure and push''')
code(r'''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_REF, C_NEW = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
for ax, (title, yref, ynew, ylab) in zip(axes, [
        ("Absolute — differs, as expected", [J_ref[b] for b in common], [J_new[b] for b in common],
         "J per output token"),
        ("Normalised J(L)/J(1) — what the policy needs",
         [J_ref[b]/J_ref[1] for b in common], [J_new[b]/J_new[1] for b in common], "J(L) / J(1)"),
        ("Power", [P_ref[b] for b in common], [P_new[b] for b in common], "GPU power (W)")]):
    ax.plot(common, yref, "o-", color=C_REF, lw=2, ms=5, label="A100-SXM4-40GB")
    ax.plot(common, ynew, "s-", color=C_NEW, lw=2, ms=5, label=gpu_new)
    ax.set_xscale("log", base=2); ax.set_xticks(common); ax.set_xticklabels(common)
    if "Power" not in title: ax.set_yscale("log")
    ax.set_xlabel("concurrency (batch)"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.grid(True, color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("E14 — does the energy curve transfer across GPUs?  ·  identical protocol, "
             "identical model, hardware is the only variable", fontsize=10.5, color=INK, x=.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(f"{OUT}/fig_cross_gpu.png", dpi=200)
print("wrote", f"{OUT}/fig_cross_gpu.png")''')
code(r'''import subprocess
subprocess.run(["git", "add", "results/batching"], check=False)
subprocess.run(["git", "commit", "-m", f"E14: energy curve measured on {GPU_NAME}"], check=False)
subprocess.run(["git", "pull", "--rebase", "-q"], check=False)   # notebook 11 may have pushed
subprocess.run(["git", "push"], check=False)
print("pushed")''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                     name="python3"),
                                     language_info=dict(name="python", version="3.11")),
          nbformat=4, nbformat_minor=5)
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/12_cross_gpu_curve.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {path} ({len(cells)} cells)")
