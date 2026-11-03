"""Builds notebooks/10_live_router_test.ipynb (GPU) — E9.

The last open item from the 5 Aug supervisor direction: run the ACTUAL router against a real vLLM
server, at several concurrency levels, with both policies tuned to the SAME energy budget.

Why this run exists, precisely:

  * E6 (notebook 09) validated the *simulator* per policy, but the two runs sat at different
    operating points (16% vs 0.5% think-rate), so it never was a head-to-head. Here the two
    policies are matched to one budget BY CONSTRUCTION before either is run.
  * Every simulated result so far comes from one arrival regime at a time. Here the same protocol
    is repeated at THREE concurrency levels, which is the axis the supervisor asked to see.
  * The router under test is `src/router.py` — the deployable component, reading the server's real
    `vllm:num_requests_running` through its own HTTP probe, pricing with the measured J(L) curve.
    Not a simulator function with a different name. Its default dtok estimator is the CONSTANT one,
    so nothing here uses per-query token knowledge a live system could not have.

Token counts are forced (ignore_eos + max_tokens = the recorded count) so the run isolates serving
physics from generation variance, exactly as E6 did — we are testing the routing policy and its
energy, not re-generating answers.

Run:  python scripts/build_livetest_notebook.py
Then: run notebooks/10_live_router_test.ipynb on vast.ai (A100 preferred) and push the outputs.
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# E9 — the live matched-energy head-to-head, across concurrency levels

Runs `src/router.py` — the real component — against a real vLLM server.

| | |
|--|--|
| **what** | static threshold vs load-aware threshold, **tuned to the same energy budget before the run** |
| **where** | three arrival regimes — low and high constant load, plus bursty — same protocol at each |
| **against** | never-think (floor) and random-at-rate (the honest floor at the same think-rate) |
| **measured** | Zeus/NVML energy, request latency, true `num_requests_running`, delivered accuracy |

This closes the two limitations stated at the end of `results/validation/VALIDATION_SUMMARY.md`:
the policy comparison had been validated only in simulation, and only at one operating point.

**Expected runtime ~90 min on an A100** (12 runs). Everything appends to disk after each run, so a
crash never loses completed work.''')

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
os.makedirs("results/livetest", exist_ok=True)
print("cwd:", os.getcwd())''')

# ------------------------------------------------------------------ 1. install
md(r'''## 1. Install

Same verified vast.ai recipe as notebook 09. **vLLM re-pulls a broken `amdsmi`, and Ray's GPU probe
then dies with `KeyError: 'libamd_smi.so'`** — uninstall `ray` too (single-GPU vLLM never uses it).

- CUDA-12.6 driver box: `pip install "vllm>=0.10,<0.11" "transformers>=4.51,<5"`.
- CUDA-13 box: `pip install -U vllm`.

Then **always** `pip uninstall -y amdsmi ray torchcodec` and restart the kernel.''')
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
print("If vLLM was just installed or amdsmi/ray were removed: Kernel -> Restart, then re-run from 0.")''')

# ------------------------------------------------------------------ 2. server
md(r'''## 2. Launch the vLLM server

`--max-num-seqs 32` (notebook 09 used 12). The high-concurrency level needs headroom above 12 or
the server, not the arrival process, would be setting the concurrency — and then "high load" would
be untestable. 32 x ~3k tokens of KV still fits the ~20 GB an A100-40GB has left after weights.''')
code(r'''MODEL = "Qwen/Qwen3-8B"
PORT, MAX_LEN, GPU_UTIL, MAX_NUM_SEQS = 8000, 16384, 0.90, 32
OUT = "results/livetest"

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
md(r'''## 3. Metering, load recording, and the request path

Zeus reads whole-GPU energy through NVML, so it captures the server process even though it runs
separately. The idle baseline is recorded so both total and idle-subtracted energy can be reported
— the E6 write-up showed the simulator's prediction is *bracketed* by those two figures, and that
bracket is how the ~10% error was explained rather than excused.''')
code(r'''import asyncio, aiohttp, time, json, re, numpy as np, threading, urllib.request
import nest_asyncio; nest_asyncio.apply()
from zeus.monitor import ZeusMonitor
mon = ZeusMonitor(gpu_indices=[0])

def measure_idle(seconds=20):
    mon.begin_window("idle"); time.sleep(seconds); m = mon.end_window("idle")
    return m.gpu_energy[0] / seconds
IDLE_W = measure_idle(20)
print(f"idle GPU power: {IDLE_W:.1f} W")

# aiohttp's default 300 s TOTAL timeout silently kills the longest (most interesting) generations.
NO_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)

class LoadRecorder(threading.Thread):
    """Samples true num_requests_running during a run.
    NB: the flag must NOT be named `_stop` — Thread already defines `_stop()` and shadowing it
    makes join() raise `TypeError: 'Event' object is not callable`."""
    def __init__(self, probe, hz=4.0):
        super().__init__(daemon=True); self.probe, self.hz = probe, hz
        self.samples = []; self._halt = threading.Event()
    def run(self):
        t0 = time.time()
        while not self._halt.is_set():
            self.samples.append((time.time() - t0, self.probe.read())); time.sleep(1.0 / self.hz)
    def stop(self): self._halt.set(); self.join(timeout=3)

async def fire(session, prompt, max_tokens, think=False):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=int(max_tokens), temperature=0,
                chat_template_kwargs={"enable_thinking": bool(think)}, ignore_eos=True)
    t0 = time.time()
    async with session.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
        d = await r.json()
    u = d.get("usage", {}) or {}
    return dict(latency=time.time() - t0, completion_tokens=u.get("completion_tokens", 0))

def append(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
print("helpers ready")''')

# ------------------------------------------------------------------ 4. the router
md(r'''## 4. The router under test

This is `src/router.py` as it would ship. Its load probe polls the real `/metrics` endpoint over
HTTP with a 0.25 s cache; its price curve is the measured `J(L)`; its Δtokens estimator is the
**constant** one, so no per-query token knowledge — the thing a live system cannot have — enters
any decision. The static baseline runs through the same class hierarchy, so the two policies differ
only in their decision rule.''')
code(r'''from src.router import (EnergyAwareRouter, StaticThresholdRouter, VLLMLoadProbe,
                        ConstantDeltaTokens, EnergyCurve)
from src.serving_sim import Hardware, Workload, make_trace, simulate

hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
probe = VLLMLoadProbe(base_url=f"http://127.0.0.1:{PORT}", ttl_s=0.25)

curve = EnergyCurve.from_jsonl()
print(f"measured curve: J(1)={curve.jtok(1):.2f}  J(8)={curve.jtok(8):.3f}  J(32)={curve.jtok(32):.3f} "
      f"J/token  (spread {curve.spread:.0f}x)")
print(f"live probe reads num_requests_running = {probe.read():.0f}")
print(f"E[dtok] = {DT_MEAN:.0f} tokens (the constant the deployable router prices with)")

# max_model_len must hold prompt AND output; longest recorded generation is 16,354 vs a 16,384
# limit. Cap the WORKLOAD (not just the request) so the simulator comparison stays matched.
TOK_CAP = MAX_LEN - 2384
for _q in wl.qids:
    wl.tok_th[_q] = min(wl.tok_th[_q], TOK_CAP); wl.tok_nt[_q] = min(wl.tok_nt[_q], TOK_CAP)
qtext = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}
print(f"token cap {TOK_CAP}: {sum(1 for q in wl.qids if wl.tok_th[q] >= TOK_CAP)} think requests clipped")''')

# ------------------------------------------------------------------ 5. calibration
md(r'''## 5. Calibrate both policies to the SAME budget — before either runs

This is the step E6 lacked. For each concurrency level the simulator (validated to ~10% on energy)
is used to find:

* the never-think and always-think energy, which set the scale;
* **τ\*** — the static threshold whose energy lands on the target budget;
* **λ\*** — the load-aware multiplier whose energy lands on the same target;
* **r\*** — the random rate matching the static policy's think-rate, i.e. the honest floor.

The budget is 40% of the never→always span. Tuning happens on *historical* seeds; the live run then
replays a different, held-out trace — so the calibration is never fitted to the run it is judged on.''')
code(r'''HIST_SEEDS, LIVE_SEED = [100, 101, 102], 0
BUDGET_FRAC = 0.40
SPAN_S = 300.0        # target arrival span, held EQUAL across levels -- see below

# Trace length is set per level so every level has the SAME arrival span. This is not cosmetic.
# A single thinking request generates ~2,700 tokens and takes ~55 s. If the arrival span is not
# comfortably longer than that, the last thinking request finishes alone after everything else has
# drained, and the whole run's energy is dominated by that tail (the E7 mixed-batch effect, turned
# into a measurement artefact). A dry run at 150 queries / 68 s span showed exactly this: total
# energy DOUBLED between a 0.4% and a 1.6% think-rate, leaving barely three usable operating points
# under the cap. Span-matching removes the artefact and makes the levels comparable to each other.
LEVELS = {
    "low":    dict(hi_rate=0.4, lo_rate=0.4,  hi_s=30.0, lo_s=30.0),    # constant, near-idle
    "high":   dict(hi_rate=2.2, lo_rate=2.2,  hi_s=30.0, lo_s=30.0),    # constant, saturated
    "bursty": dict(hi_rate=6.0, lo_rate=0.15, hi_s=6.0,  lo_s=30.0),    # swings idle <-> busy
}
N_AT = {"low": 120, "high": 660, "bursty": 200}   # ~SPAN_S of arrivals at each level's mean rate
MAXB = MAX_NUM_SEQS   # simulate the SAME concurrency bound the live server enforces

# The bursty level is not optional decoration: E4 found the load-aware advantage tracks load
# VARIABILITY (r = 0.69), not load level. A constant-rate server has no exogenous variability to
# exploit -- whatever load swing exists there was caused by the router's own decisions. Testing only
# constant rates would test only the regimes where the mechanism has nothing to work with.

def sim_all(regime, decide, n, seeds=HIST_SEEDS):
    """Mean of every metric over the tuning traces, in ONE pass (energy and accuracy together)."""
    rs = [simulate(make_trace(wl, n, s, **regime), decide, wl, hw, max_batch=MAXB) for s in seeds]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

def sim_mean(regime, decide, n, key, seeds=HIST_SEEDS):
    return sim_all(regime, decide, n, seeds)[key]

def decider(family, v):
    return {"static": lambda L, q, c: SCORE[q] >= v,
            "load":   lambda L, q, c: SCORE[q] > v * hw.jtok(L) * DT_MEAN,
            "random": (lambda rng: (lambda L, q, c: rng.random() < v))(np.random.default_rng(7)),
            }[family]

def tune(regime, family, budget_wh, grid, n):
    """Best feasible point under the budget — the fairness lesson applies to calibration too:
    the grid must resolve the family's operating points or the policy is handicapped before it runs."""
    best = None
    for v in grid:
        m = sim_all(regime, decider(family, v), n)
        if m["energy_wh"] <= budget_wh + 1e-9 and (best is None or m["acc"] > best[2]):
            best = (float(v), m["energy_wh"], m["acc"], m["think_rate"])
    return best

_sv = np.unique([SCORE[q] for q in wl.qids])
TAU_GRID = np.unique(np.concatenate([[0.0], _sv[::max(1, len(_sv)//180)], _sv[-3:], [1.01]]))
# The lambda grid must be the IMAGE of the score grid under the price map, not a log sweep.
# A log grid takes 1.16x steps in lambda, but 81.5% of router scores sit below 0.1, so one step can
# sweep the effective threshold across a large slice of the score mass -- stranding the load-aware
# policy at never-think under a budget it could have used, and handing the head-to-head to the
# static baseline for entirely the wrong reason. (results/CORRECTIONS.md S5b)
_cand = np.concatenate([TAU_GRID / max(hw.jtok(L) * DT_MEAN, 1e-12) for L in (1, 4, 16, 48)])
_cand = np.unique(_cand[np.isfinite(_cand) & (_cand > 0)])
LAM_GRID = np.unique(np.concatenate([_cand[::max(1, len(_cand)//240)], np.logspace(-8, 4, 40)]))
print(f"grids: tau {len(TAU_GRID)}  lambda {len(LAM_GRID)} (price-mapped)")

PLAN = {}
for lv, rg in LEVELS.items():
    N = N_AT[lv]
    nev = sim_all(rg, lambda L, q, c: False, N)
    alw = sim_all(rg, lambda L, q, c: True, N)
    budget = nev["energy_wh"] + BUDGET_FRAC * (alw["energy_wh"] - nev["energy_wh"])
    tau, e_tau, a_tau, tr_static = tune(rg, "static", budget, TAU_GRID, N)
    lam, e_lam, a_lam, tr_load = tune(rg, "load", budget, LAM_GRID, N)
    st = sim_all(rg, decider("static", tau), N)
    PLAN[lv] = dict(regime=rg, n=N, budget_wh=budget, tau=tau, lam=lam, rate=tr_static,
                    sim_e_static=e_tau, sim_e_load=e_lam, sim_acc_static=a_tau, sim_acc_load=a_lam,
                    sim_think_static=tr_static, sim_think_load=tr_load,
                    e_never=nev["energy_wh"], e_always=alw["energy_wh"],
                    sim_L=st["load_mean"], est_wall_min=st["makespan"] / 60.0)
    print(f"{lv:7s} N={N:4d} budget {budget:5.2f} Wh (never {nev['energy_wh']:.2f} / always {alw['energy_wh']:.2f}) | "
          f"tau*={tau:.4f} -> {e_tau:.2f} Wh, think {tr_static:.1%} | "
          f"lam*={lam:.2e} -> {e_lam:.2f} Wh, think {tr_load:.1%} | "
          f"sim L {st['load_mean']:.1f} | ~{st['makespan']/60:.0f} min/run")
tot = sum(p["est_wall_min"] for p in PLAN.values()) * 4
print(f"\nestimated total run time: ~{tot:.0f} min for {len(LEVELS)*4} runs (+ server start)")
json.dump(PLAN, open(f"{OUT}/plan.json", "w"), indent=1, default=float)''')

# ------------------------------------------------------------------ 6. the runs
md(r'''## 6. Run it — 3 concurrency levels x 4 policies

Each run replays the **same held-out trace** (same arrival times, same questions in the same order)
so only the routing decisions differ. The load-aware router calls its own HTTP probe at every
admission — the concurrency it prices with is the server's real one, not a simulated one.''')
code(r'''def make_policies(lv):
    p = PLAN[lv]
    load_router = EnergyAwareRouter(curve, probe, lam=p["lam"], dtok_fn=ConstantDeltaTokens(DT_MEAN))
    static_router = StaticThresholdRouter(tau=p["tau"], probe=probe)
    rng = np.random.default_rng(7)
    return {
        "never":      lambda qid: False,
        "static":     lambda qid: static_router.decide(SCORE[qid], qid=qid).think,
        "load-aware": lambda qid: load_router.decide(SCORE[qid], qid=qid).think,
        "random":     lambda qid: bool(rng.random() < p["rate"]),
    }, load_router, static_router

async def run_one(lv, pname, decide, trace):
    rec = LoadRecorder(probe); rec.start()
    fails, t_start = [], time.time()
    span = max(t for t, _ in trace)
    print(f"  [{lv}/{pname}] {len(trace)} queries, arrivals span {span/60:.1f} min", flush=True)
    conn = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=conn, timeout=NO_TIMEOUT) as s:
        mon.begin_window(f"{lv}-{pname}")
        async def one(arr_t, qid):
            await asyncio.sleep(max(0.0, arr_t - (time.time() - t_start)))
            L_at_admit = probe.read()          # the live signal, recorded for every request
            think = decide(qid)                # <- the router decides here, on the real server
            toks = int(round(wl.tok_th[qid] if think else wl.tok_nt[qid]))
            for attempt in range(3):           # a dropped connection must not kill a 10-min run
                try:
                    r = await fire(s, qtext[qid], max(toks, 1), think=think); break
                except Exception as e:
                    if attempt == 2:
                        fails.append((qid, repr(e)[:120])); return None
                    await asyncio.sleep(2.0 * (attempt + 1))
            r.update(qid=qid, think=bool(think), arrival=arr_t, L_at_admit=L_at_admit,
                     correct=(wl.ct[qid] if think else wl.cn[qid]))
            return r
        raw = await asyncio.gather(*[one(a, q) for a, q in trace], return_exceptions=True)
        m = mon.end_window(f"{lv}-{pname}")
    rec.stop()
    res = [r for r in raw if isinstance(r, dict)]
    if len(res) < len(trace):
        print(f"    WARNING {len(trace)-len(res)}/{len(trace)} failed; first: {fails[:2]}")
    lat = np.array([r["latency"] for r in res]); Ls = np.array([x for _, x in rec.samples])
    Ls = Ls[np.isfinite(Ls)]
    wall = time.time() - t_start          # our own clock, as in notebook 09 (Zeus's field name varies)
    row = dict(level=lv, policy=pname, n=len(res), energy_wh=m.gpu_energy[0] / 3600.0,
               energy_wh_idlesub=(m.gpu_energy[0] - IDLE_W * wall) / 3600.0, wall_s=wall,
               acc=float(np.mean([r["correct"] for r in res])),
               think_rate=float(np.mean([r["think"] for r in res])),
               lat_mean=float(lat.mean()), lat_p95=float(np.percentile(lat, 95)),
               L_mean=float(Ls.mean()) if len(Ls) else float("nan"),
               L_p95=float(np.percentile(Ls, 95)) if len(Ls) else float("nan"),
               probe_errors=probe.n_errors)   # cumulative over the session
    append(f"{OUT}/live_requests_{lv}_{pname}.jsonl", res)
    append(f"{OUT}/live_summary.jsonl", [row])
    print(f"    -> {row['energy_wh']:.2f} Wh | acc {row['acc']:.4f} | think {row['think_rate']:.1%} "
          f"| p95 {row['lat_p95']:.0f}s | L {row['L_mean']:.1f}", flush=True)
    return row

ROWS = []
for lv in LEVELS:
    trace = make_trace(wl, N_AT[lv], LIVE_SEED, **LEVELS[lv])    # held-out: seed 0, tuned on 100-102
    pols, lr, sr = make_policies(lv)
    print(f"\n=== {lv}: budget {PLAN[lv]['budget_wh']:.2f} Wh, tau*={PLAN[lv]['tau']:.4f}, "
          f"lam*={PLAN[lv]['lam']:.2e} ===")
    for pname, dec in pols.items():
        ROWS.append(await run_one(lv, pname, dec, trace))
    print(f"  router think-rates — load-aware {lr.think_rate:.1%}, static {sr.think_rate:.1%}")''')

# ------------------------------------------------------------------ 7. compare
md(r'''## 7. The head-to-head, and simulator vs reality

Two questions, kept separate:

1. **Did the two policies actually land on the same energy?** If they did, the accuracy difference
   is the answer to the thesis question, measured rather than simulated. If they did not, say so
   and read accuracy against realised energy instead — a matched-energy claim is only as good as
   the match.
2. **Does the simulator still predict reality?** Same check as E6, now at three load levels.''')
code(r'''import pandas as pd
df = pd.DataFrame(ROWS)

print("\n### measured, per level\n")
print(df[["level", "policy", "energy_wh", "energy_wh_idlesub", "acc", "think_rate",
          "lat_p95", "L_mean", "n"]].to_string(index=False))

print("\n### the head-to-head: static vs load-aware at a matched budget\n")
h2h = []
for lv in LEVELS:
    d = {r["policy"]: r for r in ROWS if r["level"] == lv}
    if not {"static", "load-aware"} <= set(d): continue
    s, l = d["static"], d["load-aware"]
    h2h.append(dict(level=lv, budget_wh=PLAN[lv]["budget_wh"],
                    static_wh=s["energy_wh"], load_wh=l["energy_wh"],
                    energy_match_pct=100 * (l["energy_wh"] - s["energy_wh"]) / s["energy_wh"],
                    static_acc=s["acc"], load_acc=l["acc"], gap_pp=100 * (l["acc"] - s["acc"]),
                    static_p95=s["lat_p95"], load_p95=l["lat_p95"],
                    static_think=s["think_rate"], load_think=l["think_rate"],
                    floor_acc=d.get("random", {}).get("acc", float("nan")),
                    never_acc=d.get("never", {}).get("acc", float("nan"))))
h = pd.DataFrame(h2h); print(h.to_string(index=False))
h.to_json(f"{OUT}/headtohead.jsonl", orient="records", lines=True)

print("\n### simulator vs measured (same trace, same policy, same concurrency bound)\n")
cmp_rows = []
for r in ROWS:
    lv, rg = r["level"], LEVELS[r["level"]]
    p = PLAN[lv]
    dec = {"never": lambda L, q, c: False, "static": decider("static", p["tau"]),
           "load-aware": decider("load", p["lam"]), "random": decider("random", p["rate"]),
           }[r["policy"]]
    sim = simulate(make_trace(wl, N_AT[lv], LIVE_SEED, **rg), dec, wl, hw, max_batch=MAXB)
    rel = lambda a, b: 100 * (a - b) / b if b else float("nan")
    cmp_rows.append(dict(level=lv, policy=r["policy"], sim_wh=sim["energy_wh"],
                         meas_wh=r["energy_wh"], meas_wh_idlesub=r["energy_wh_idlesub"],
                         err_pct=rel(sim["energy_wh"], r["energy_wh"]),
                         err_pct_idlesub=rel(sim["energy_wh"], r["energy_wh_idlesub"]),
                         sim_p95=sim["lat_p95"], meas_p95=r["lat_p95"],
                         sim_think=sim["think_rate"], meas_think=r["think_rate"],
                         sim_L=sim["load_mean"], meas_L=r["L_mean"],
                         sim_acc=sim["acc"], meas_acc=r["acc"]))
c = pd.DataFrame(cmp_rows); print(c.to_string(index=False))
c.to_json(f"{OUT}/sim_vs_live.jsonl", orient="records", lines=True)
print(f"\nprobe HTTP errors over the whole session: {probe.n_errors}/{probe.n_reads}")''')

# ------------------------------------------------------------------ 8. figure + push
md(r'''## 8. Figure and push''')
code(r'''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_STATIC, C_LOAD, C_RAND = "#2a78d6", "#eb6834", "#9a9a95"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
lv_order = [l for l in LEVELS if any(r["level"] == l for r in ROWS)]
x = np.arange(len(lv_order)); w = 0.26
for ax, (key, ylab, fmt) in zip(axes, [("energy_wh", "energy (Wh)", "{:.1f}"),
                                       ("acc", "accuracy", "{:.3f}"),
                                       ("lat_p95", "p95 latency (s)", "{:.0f}")]):
    for i, (pol, col) in enumerate([("random", C_RAND), ("static", C_STATIC), ("load-aware", C_LOAD)]):
        vals = [next((r[key] for r in ROWS if r["level"] == lv and r["policy"] == pol), np.nan)
                for lv in lv_order]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=col,
               label={"random": "random at rate (floor)", "static": "static threshold",
                      "load-aware": "load-aware (dynamic)"}[pol], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(lv_order); ax.set_ylabel(ylab)
    ax.set_xlabel("offered concurrency")
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("E9 — measured on a live vLLM server: both policies tuned to the same energy budget",
             fontsize=10.5, color=INK, x=0.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig_live_headtohead.png", dpi=200)
print("wrote", f"{OUT}/fig_live_headtohead.png")''')
code(r'''import subprocess
subprocess.run(["git", "add", "results/livetest"], check=False)
subprocess.run(["git", "commit", "-m",
                "E9: live matched-energy head-to-head across three concurrency levels"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed")''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                     name="python3"),
                                     language_info=dict(name="python", version="3.11")),
          nbformat=4, nbformat_minor=5)
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/10_live_router_test.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {path} ({len(cells)} cells)")
