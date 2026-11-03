"""Builds notebooks/09_serving_validation.ipynb (GPU) — E6 + E7.

Closes the two biggest limitations of the simulator-based RQ4 results:

  E7  EXTENDED HARDWARE CURVE. The existing J(L) curve was measured at l_in=64, l_out=256 with
      homogeneous batches, but our thinking requests average ~2,700 tokens (max 16k). This
      re-measures J/token across batch x OUTPUT LENGTH, adds a prefill (l_in) sweep, and adds
      MIXED batches (long-think + short-no-think together) -- the three extrapolations the
      simulator currently makes.

  E6  REAL SERVING REPLAY. Launches a real vLLM OpenAI server and replays the SAME arrival trace
      the simulator uses (src/serving_sim.make_trace, same seed), firing async requests at their
      arrival times, polling /metrics for the true num_requests_running, and metering GPU energy
      with Zeus. Output: measured energy / latency / concurrency vs the simulator's prediction.

Token counts are forced (ignore_eos + max_tokens = the recorded count) so the replay isolates the
SERVING physics from generation variance -- we are validating the energy+queueing model, not
re-generating answers.

Run:  python scripts/build_validation_notebook.py
Then: run notebooks/09_serving_validation.ipynb on vast.ai (A100 preferred), push the outputs.
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# E6 + E7 — validating the serving simulator against a real vLLM server

Our RQ4 results (E1/E2/E4) compose **measured** hardware curves with a **simulated** queue.
This notebook tests both halves against reality:

| | what it checks |
|--|--|
| **E7** | the hardware curve beyond where it was measured — long outputs (KV growth), long prompts (prefill), and **mixed** batches |
| **E6** | the whole pipeline — replay the simulator's own arrival trace on a real vLLM server and compare measured energy / latency / concurrency to the prediction |

**Expected runtime ~60–90 min on an A100.** Everything appends to disk, so a crash never loses
completed work.''')

# ------------------------------------------------------------------ 0. repo
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
os.makedirs("results/validation", exist_ok=True)
print("cwd:", os.getcwd())''')

# ------------------------------------------------------------------ 1. install
md(r'''## 1. Install

Verified vast.ai recipe. **vLLM re-pulls a broken `amdsmi`, and Ray's GPU probe then dies with
`KeyError: 'libamd_smi.so'`** — uninstalling `ray` too is the reliable fix (single-GPU vLLM never
uses Ray, and its `is_ray_initialized()` catches `ImportError`).

- **CUDA-12.6 driver box** (`driver too old (found 12060)`): pin
  `pip install "vllm>=0.10,<0.11" "transformers>=4.51,<5"`.
- **CUDA-13 box**: `pip install -U vllm` (leave transformers unpinned).

Then **always**: `pip uninstall -y amdsmi ray torchcodec` and restart the kernel.''')
code(r'''import importlib, subprocess, sys
def have(m):
    try: importlib.import_module(m); return True
    except Exception: return False
if not have("vllm"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
for pkg in ["zeus-ml", "aiohttp", "nest_asyncio"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
# vLLM drags in a broken amdsmi; ray's probe then raises KeyError instead of ImportError
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "amdsmi", "ray", "torchcodec"], check=False)
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("If vLLM was just installed or amdsmi/ray were removed: Kernel -> Restart, then re-run from 0.")''')

# ------------------------------------------------------------------ 2. config + server
md(r'''## 2. Config and launch the vLLM server

We use the real OpenAI-compatible server (not the offline `LLM` class) so that requests queue and
batch exactly as they would in production, and so `/metrics` exposes the true
`num_requests_running` — the very signal our load-aware router reads.''')
code(r'''MODEL = "Qwen/Qwen3-8B"
PORT = 8000
MAX_LEN = 16384
GPU_UTIL = 0.90
# KV cache is ~2.1 GB per request at 14k tokens for Qwen3-8B, and a 40GB A100 has only ~20 GB
# left for KV after weights. Bounding concurrent sequences keeps bursts from exhausting it (vLLM
# then queues rather than thrashing). This is also the real analogue of the simulator's MAX_BATCH.
MAX_NUM_SEQS = 12
OUT = "results/validation"

import subprocess, sys, time, os, json, urllib.request
def server_up(port=PORT, timeout=1.0):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout); return True
    except Exception: return False

proc = None
if not server_up():
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL, "--port", str(PORT), "--max-model-len", str(MAX_LEN),
           "--gpu-memory-utilization", str(GPU_UTIL), "--disable-log-requests",
           "--max-num-seqs", str(MAX_NUM_SEQS)]
    print("launching:", " ".join(cmd))
    logf = open(f"{OUT}/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    for i in range(600):
        if server_up(): break
        if proc.poll() is not None:
            print("SERVER DIED — last 40 log lines:")
            print("".join(open(f"{OUT}/server.log").readlines()[-40:])); raise SystemExit(1)
        time.sleep(2)
        if i % 15 == 0: print(f"  waiting for server... {i*2}s")
print("server up:", server_up())''')

# ------------------------------------------------------------------ 3. helpers
md(r'''## 3. Energy metering + concurrency polling helpers

Zeus reads whole-GPU energy over a window via NVML, so it captures the server process even though
it runs separately. We also record the **idle baseline** so the write-up can report both total and
idle-subtracted energy.''')
code(r'''import asyncio, aiohttp, time, json, re, numpy as np, threading
import nest_asyncio; nest_asyncio.apply()
from zeus.monitor import ZeusMonitor
mon = ZeusMonitor(gpu_indices=[0])

def measure_idle(seconds=20):
    mon.begin_window("idle"); time.sleep(seconds); m = mon.end_window("idle")
    return m.gpu_energy[0] / seconds        # watts
IDLE_W = measure_idle(20)
print(f"idle GPU power: {IDLE_W:.1f} W")

# aiohttp defaults to a 300 s TOTAL timeout. A 16k-token generation at ~30 tok/s (normal under
# concurrency) needs ~545 s, so the default silently kills the longest -- and most interesting --
# requests. Disable it; the server, not the client, bounds these runs.
NO_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)

METRIC_RE = re.compile(r'^vllm:num_requests_running\{[^}]*\}\s+([0-9.]+)', re.M)
def poll_running():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as r:
            txt = r.read().decode()
        m = METRIC_RE.search(txt)
        return float(m.group(1)) if m else float("nan")
    except Exception:
        return float("nan")

class LoadRecorder(threading.Thread):
    """Samples the server's true num_requests_running while an experiment runs.

    NB: the halt flag must NOT be called `_stop` -- threading.Thread already defines an internal
    `_stop()` method, and shadowing it with an Event makes join() raise
    `TypeError: 'Event' object is not callable`.
    """
    def __init__(self, hz=4.0):
        super().__init__(daemon=True); self.hz = hz; self.samples = []; self._halt = threading.Event()
    def run(self):
        t0 = time.time()
        while not self._halt.is_set():
            self.samples.append((time.time() - t0, poll_running()))
            time.sleep(1.0 / self.hz)
    def stop(self): self._halt.set(); self.join(timeout=3)

async def fire(session, prompt, max_tokens, think=False, ignore_eos=True):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens), "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": bool(think)},
            "ignore_eos": bool(ignore_eos)}
    t0 = time.time()
    async with session.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
        d = await r.json()
    lat = time.time() - t0
    usage = d.get("usage", {}) or {}
    return dict(latency=lat, completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0))

def append(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
print("helpers ready")''')

# ------------------------------------------------------------------ 4. E7
md(r'''## 4. E7 — extended hardware curve (the three extrapolations)

The simulator prices every request at `J(L)` measured with **l_in=64, l_out=256, homogeneous
batches**. Here we measure the three regimes it extrapolates into:

1. **output length** (KV-cache growth): batch × `l_out` ∈ {256, 1024, 3072}
2. **prefill**: `l_in` ∈ {64, 1024, 4096} at fixed batch/output
3. **mixed batches**: long-think + short-no-think served together

Each configuration fires exactly `batch` concurrent requests with `ignore_eos`, so the server is
held at that concurrency for the whole window.''')
code(r'''PAD = "The quick brown fox jumps over the lazy dog. "     # ~10 tokens
def prompt_of(approx_tokens):
    return ("Answer briefly.\n" + PAD * max(1, approx_tokens // 10))[:approx_tokens * 6]

async def run_batch(n, out_len, in_len=64, mixed=None, reps=2):
    """Hold the server at concurrency n; return measured J/token etc."""
    rows = []
    for rep in range(reps):
        lens = mixed if mixed else [out_len] * n
        prompts = [prompt_of(in_len) for _ in lens]
        conn = aiohttp.TCPConnector(limit=max(n * 2, 16))
        async with aiohttp.ClientSession(connector=conn, timeout=NO_TIMEOUT) as s:
            rec = LoadRecorder(); rec.start()
            mon.begin_window(f"b{n}_{out_len}_{in_len}_{rep}")
            t0 = time.time()
            res = await asyncio.gather(*[fire(s, p, L) for p, L in zip(prompts, lens)])
            el = time.time() - t0
            m = mon.end_window(f"b{n}_{out_len}_{in_len}_{rep}")
            rec.stop()
        e_j = m.gpu_energy[0]
        toks = sum(r["completion_tokens"] for r in res)
        Ls = [x[1] for x in rec.samples if x[1] == x[1]]
        rows.append(dict(experiment="e7", batch=n, l_out=out_len, l_in=in_len,
                         mixed=bool(mixed), rep=rep, n=len(res), out_tokens=toks,
                         time_s=el, energy_j=e_j, power_w=e_j / max(el, 1e-9),
                         energy_per_out_token_j=e_j / max(toks, 1),
                         idle_w=IDLE_W,
                         energy_per_out_token_j_idlesub=max(e_j - IDLE_W * el, 0) / max(toks, 1),
                         throughput_tok_s=toks / max(el, 1e-9),
                         observed_L_mean=float(np.mean(Ls)) if Ls else float("nan"),
                         model=MODEL, gpu=__import__("torch").cuda.get_device_name(0)))
    append(f"{OUT}/curve_v2.jsonl", rows)
    r = rows[-1]
    print(f"  batch {n:3d} l_out {out_len:5d} l_in {in_len:5d} mixed={bool(mixed)} -> "
          f"{r['energy_per_out_token_j']:.3f} J/tok  ({r['throughput_tok_s']:.0f} tok/s, "
          f"obsL {r['observed_L_mean']:.1f})")
    return rows

# --- 4a. output-length sweep (KV growth)  ---------------------------------
print("4a. batch x output length")
for out_len in [256, 1024, 3072]:
    for n in [1, 4, 16, 32]:
        await run_batch(n, out_len, in_len=64)

# --- 4b. prefill sweep ----------------------------------------------------
print("\n4b. prefill (input length) at batch 8, l_out 256")
for in_len in [64, 1024, 4096]:
    await run_batch(8, 256, in_len=in_len)

# --- 4c. mixed batches ----------------------------------------------------
print("\n4c. mixed batches (long-think + short-no-think together), batch 16")
await run_batch(16, 0, in_len=64, mixed=[3072] * 8 + [128] * 8)
await run_batch(16, 0, in_len=64, mixed=[3072] * 2 + [128] * 14)
print(f"\nwrote {OUT}/curve_v2.jsonl")''')

# ------------------------------------------------------------------ 5. E6
md(r'''## 5. E6 — replay the simulator's own trace on the real server

Same trace generator, same seed, same queries and token counts as the simulation, so the numbers
are directly comparable. Requests are fired **at their arrival times** and token counts are forced
(`ignore_eos`), which isolates the serving physics from generation randomness.

Two policies are replayed — a fixed threshold and the load-aware rule — at a matched think-rate.''')
code(r'''from src.serving_sim import Hardware, Workload, make_trace, simulate
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
# Wall-clock is set by the ARRIVAL SPAN (the replay sleeps until each arrival), so a long idle
# phase is pure waiting: lo_rate=0.03/lo_s=110 needs 83 min for 250 queries AND idles at mean
# L=1.8 -- i.e. mostly validating batch~2. This regime keeps a 40x burst ratio, runs in ~12 min,
# and holds mean L=6.6, which actually exercises the batching regime we care about.
N_REPLAY = 200
SEED = 0
REGIME = dict(hi_rate=6.0, lo_rate=0.15, hi_s=6.0, lo_s=30.0)   # bursty, ~12 min/policy

# max_model_len must hold prompt AND output; the longest recorded generation is 16,354 tokens
# against a 16,384 limit. Cap the workload (affects ~3% of think requests) and apply it to BOTH
# the replay and the section-6 simulator comparison, so they stay exactly comparable.
TOK_CAP = MAX_LEN - 2384
for _q in wl.qids:
    wl.tok_th[_q] = min(wl.tok_th[_q], TOK_CAP)
    wl.tok_nt[_q] = min(wl.tok_nt[_q], TOK_CAP)
print(f"token cap {TOK_CAP}: {sum(1 for q in wl.qids if wl.tok_th[q] >= TOK_CAP)} think requests clipped")

trace = make_trace(wl, N_REPLAY, SEED, **REGIME)
qtext = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}

def decide_static(tau): return lambda L, q: SCORE[q] >= tau
def decide_load(lam):   return lambda L, q: SCORE[q] > lam * hw.jtok(max(L, 1)) * wl.dtok(q)

async def replay(decide, tag):
    rec = LoadRecorder(); rec.start()
    fails = []
    span = max(t for t, _ in trace)
    print(f"  [{tag}] {len(trace)} queries, arrivals span {span/60:.1f} min — progress every 25:")
    prog = {"n": 0}
    conn = aiohttp.TCPConnector(limit=256)
    results, t_start = [], time.time()
    async with aiohttp.ClientSession(connector=conn, timeout=NO_TIMEOUT) as s:
        mon.begin_window(tag)
        async def one(arr_t, qid):
            await asyncio.sleep(max(0.0, arr_t - (time.time() - t_start)))
            L = poll_running()                       # the live signal a real router would read
            think = decide(0 if L != L else L, qid)
            toks = int(round(wl.tok_th[qid] if think else wl.tok_nt[qid]))
            r = None
            for attempt in range(3):        # a dropped connection must not kill a 12-min run
                try:
                    r = await fire(s, qtext[qid], max(toks, 1), think=think); break
                except Exception as e:
                    if attempt == 2:
                        fails.append((qid, repr(e)[:120]))
                        return None
                    await asyncio.sleep(2.0 * (attempt + 1))
            r.update(qid=qid, think=bool(think), arrival=arr_t, observed_L_at_admit=L,
                     correct=(wl.ct[qid] if think else wl.cn[qid]))
            prog["n"] += 1
            if prog["n"] % 25 == 0:
                print(f"    {prog['n']}/{len(trace)} done  ({time.time()-t_start:.0f}s elapsed, "
                      f"live L={poll_running():.0f})", flush=True)
            return r
        raw = await asyncio.gather(*[one(a, q) for a, q in trace], return_exceptions=True)
        m = mon.end_window(tag)
        results = [r for r in raw if isinstance(r, dict)]
        if len(results) < len(trace):
            print(f"    WARNING: {len(trace)-len(results)}/{len(trace)} requests failed; "
                  f"first: {fails[:2]}")
        if not results:
            raise RuntimeError("every request failed — check results/validation/server.log")
    rec.stop()
    el = time.time() - t_start
    e_j = m.gpu_energy[0]
    Ls = [x[1] for x in rec.samples if x[1] == x[1]]
    out = dict(experiment="e6", tag=tag, n=len(results), makespan_s=el,
               energy_j=e_j, energy_wh=e_j / 3600.0,
               energy_wh_idlesub=max(e_j - IDLE_W * el, 0) / 3600.0, idle_w=IDLE_W,
               acc=float(np.mean([r["correct"] for r in results])),
               think_rate=float(np.mean([r["think"] for r in results])),
               lat_mean=float(np.mean([r["latency"] for r in results])),
               lat_p95=float(np.percentile([r["latency"] for r in results], 95)),
               observed_L_mean=float(np.mean(Ls)) if Ls else float("nan"),
               observed_L_p95=float(np.percentile(Ls, 95)) if Ls else float("nan"),
               total_completion_tokens=int(sum(r["completion_tokens"] for r in results)),
               seed=SEED, regime=REGIME, model=MODEL)
    append(f"{OUT}/replay_summary.jsonl", [out])
    append(f"{OUT}/replay_requests_{tag}.jsonl", results)
    append(f"{OUT}/replay_load_{tag}.jsonl", [{"t": t, "L": L} for t, L in rec.samples])
    print(f"  {tag}: {out['energy_wh']:.2f} Wh ({out['energy_wh_idlesub']:.2f} idle-sub) | "
          f"acc {out['acc']:.3f} | think {out['think_rate']:.0%} | p95 lat {out['lat_p95']:.1f}s | "
          f"obsL mean {out['observed_L_mean']:.1f} | makespan {el:.0f}s")
    return out

print("replaying static-tau ...")
r_static = await replay(decide_static(0.5), "static")
print("replaying load-aware ...")
r_load = await replay(decide_load(1e-3), "loadaware")''')

# ------------------------------------------------------------------ 6. compare
md(r'''## 6. Simulator vs reality

The headline validation number: **relative error of the simulator's predicted energy, latency and
concurrency** against the measured replay on the identical trace and policy.''')
code(r'''rows = []
for tag, dec, meas in [("static", decide_static(0.5), r_static),
                       ("loadaware", decide_load(1e-3), r_load)]:
    sim = simulate(trace, (lambda d: (lambda L, q, c: d(L, q)))(dec), wl, hw)
    def rel(a, b): return 100 * (a - b) / b if b else float("nan")
    row = dict(tag=tag,
               sim_energy_wh=sim["energy_wh"], real_energy_wh=meas["energy_wh"],
               real_energy_wh_idlesub=meas["energy_wh_idlesub"],
               err_energy_pct=rel(sim["energy_wh"], meas["energy_wh"]),
               err_energy_idlesub_pct=rel(sim["energy_wh"], meas["energy_wh_idlesub"]),
               sim_lat_p95=sim["lat_p95"], real_lat_p95=meas["lat_p95"],
               err_lat_p95_pct=rel(sim["lat_p95"], meas["lat_p95"]),
               sim_load_mean=sim["load_mean"], real_load_mean=meas["observed_L_mean"],
               err_load_pct=rel(sim["load_mean"], meas["observed_L_mean"]),
               sim_think=sim["think_rate"], real_think=meas["think_rate"],
               sim_acc=sim["acc"], real_acc=meas["acc"])
    rows.append(row)
    print(f"\n{tag}")
    print(f"  energy   sim {row['sim_energy_wh']:.2f} vs real {row['real_energy_wh']:.2f} Wh "
          f"({row['err_energy_pct']:+.1f}%; idle-subtracted {row['err_energy_idlesub_pct']:+.1f}%)")
    print(f"  p95 lat  sim {row['sim_lat_p95']:.1f} vs real {row['real_lat_p95']:.1f} s "
          f"({row['err_lat_p95_pct']:+.1f}%)")
    print(f"  mean L   sim {row['sim_load_mean']:.1f} vs real {row['real_load_mean']:.1f} "
          f"({row['err_load_pct']:+.1f}%)")
    print(f"  think    sim {row['sim_think']:.0%} vs real {row['real_think']:.0%}")
append(f"{OUT}/sim_vs_real.jsonl", rows)
print(f"\nwrote {OUT}/sim_vs_real.jsonl")''')

# ------------------------------------------------------------------ 7. push
md(r'''## 7. Push results''')
code(r'''import subprocess
subprocess.run(["git", "add", "-f", "results/validation"], check=False)
subprocess.run(["git", "commit", "-m", "E6/E7: real vLLM serving validation + extended hardware curve"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed. Files:")
for f in sorted(os.listdir(OUT)): print("  ", f)
try:
    if proc is not None: proc.terminate(); print("server stopped")
except Exception: pass''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/09_serving_validation.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {path} with {len(cells)} cells.")
