"""Builds notebooks/11_closed_loop_budget.ipynb (GPU) — E12 + E13.

Two experiments on one server, both answering questions E9 raised and could not settle.

E12 — CLOSING THE LOOP ON A REAL METER.
E9's sharpest negative: parameters tuned offline against the simulator missed the authorised energy
budget by +66% (bursty) and +69/+79% (high), because the simulator under-predicts energy at load and
because one lambda behaves differently on different traces. `BudgetController` was built for exactly
this and never run live. Here it is, against an NVML energy counter read at every completion.

The comparison is deliberately generous to the offline method:

  * the budget is set from a MEASURED never-think run (1.5x it), so it is feasible by construction
    -- E9's budgets were simulator-derived and bursty's was actually below what the hardware spends
    doing no thinking at all;
  * the offline tuning is given a ONE-POINT CALIBRATION: the never-think run yields
    k = measured/simulated, and the simulator budget is divided by k before tuning. So the offline
    method is not being blamed for a constant scale error, only for what remains.

If offline tuning still misses and the closed loop lands, the result is clean.

E13 — WHY THE MODEL BREAKS AT SATURATION.
E9 found energy under-predicted by 43% at saturation, outside the idle-power bracket, and latency
under-predicted 2.4x. The stated hypothesis was vLLM preempting and recomputing under KV-cache
pressure -- a hypothesis, never measured. Every run here records `vllm:num_preemptions_total`,
`vllm:num_requests_waiting` and `vllm:gpu_cache_usage_perc`, which confirms or kills it.

Run:  python scripts/build_closedloop_notebook.py
Then: run notebooks/11_closed_loop_budget.ipynb on vast.ai (A100) and push the outputs.
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# E12 + E13 — closing the budget loop on a real meter, and why the model breaks at saturation

Follow-up to E9 (`results/livetest/LIVE_SUMMARY.md`), which validated the router and the energy
model below saturation and found two things it could not fix:

| | E9 found | this notebook |
|--|--|--|
| **E12** | offline-tuned parameters missed the budget by **+66% / +69% / +79%** | run `BudgetController` against an NVML meter and see whether it lands |
| **E13** | energy under-predicted **43%** at saturation, outside the idle bracket | record preemptions and KV-cache pressure to test the stated cause |

**Runtime ~40 min for the bursty regime alone, ~100 min including saturated.** The bursty regime
runs first and is self-contained — you can stop after it and still have E12's answer.''')

# ------------------------------------------------------------------ 0/1/2 setup (as notebook 10)
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
os.makedirs("results/closedloop", exist_ok=True)
print("cwd:", os.getcwd())''')

md(r'''## 1. Install

Same vast.ai recipe as notebooks 09/10, plus `pynvml` for the energy counter. **vLLM re-pulls a
broken `amdsmi` and Ray's GPU probe then dies with `KeyError: 'libamd_smi.so'`** — uninstall `ray`
too. Then restart the kernel and re-run from 0.''')
code(r'''import importlib, subprocess, sys
def have(m):
    try: importlib.import_module(m); return True
    except Exception: return False
if not have("vllm"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
for pkg in ["zeus-ml", "aiohttp", "nest_asyncio", "pynvml"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "amdsmi", "ray", "torchcodec"], check=False)
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("If anything was installed/removed: Kernel -> Restart, then re-run from 0.")''')

md(r'''## 2. Launch the vLLM server''')
code(r'''MODEL = "Qwen/Qwen3-8B"
PORT, MAX_LEN, GPU_UTIL, MAX_NUM_SEQS = 8000, 16384, 0.90, 32
OUT = "results/closedloop"

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

# ------------------------------------------------------------------ 3. meter + metrics
md(r'''## 3. The energy meter and the saturation instruments

**The meter (E12).** `NVMLEnergyMeter` prefers `nvmlDeviceGetTotalEnergyConsumption`, a monotonic
millijoule counter the driver maintains — exact, and readable at any instant, which is what a
controller needs. Zeus measures *named windows*, so it cannot serve as a running total; it is kept
here only as an independent cross-check on each run's total.

**The instruments (E13).** Every run records, at 4 Hz: `num_requests_running`,
`num_requests_waiting`, `num_preemptions_total`, and `gpu_cache_usage_perc`. Preemption and KV
pressure are the stated explanation for E9's 43% energy error at saturation; these make it
measurable rather than asserted.''')
code(r'''import asyncio, aiohttp, time, json, re, numpy as np, threading, urllib.request
import nest_asyncio; nest_asyncio.apply()
from zeus.monitor import ZeusMonitor
from src.router import (EnergyAwareRouter, StaticThresholdRouter, VLLMLoadProbe,
                        ConstantDeltaTokens, EnergyCurve, BudgetController, NVMLEnergyMeter)
from src.serving_sim import Hardware, Workload, make_trace, simulate

mon = ZeusMonitor(gpu_indices=[0])
meter = NVMLEnergyMeter(index=0).start()
print("energy meter:", meter.mechanism)

def measure_idle(seconds=20):
    mon.begin_window("idle"); time.sleep(seconds); m = mon.end_window("idle")
    return m.gpu_energy[0] / seconds
IDLE_W = measure_idle(20)
print(f"idle GPU power: {IDLE_W:.1f} W")

NO_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=None)
_PATS = {k: re.compile(rf'^vllm:{k}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)', re.M) for k in
         ("num_requests_running", "num_requests_waiting", "num_preemptions_total",
          "gpu_cache_usage_perc")}

def scrape():
    """All four instruments in one HTTP call. Missing metrics come back as NaN, not an exception."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as r:
            txt = r.read().decode()
    except Exception:
        return {k: float("nan") for k in _PATS}
    out = {}
    for k, pat in _PATS.items():
        m = pat.search(txt)
        out[k] = float(m.group(1)) if m else float("nan")
    return out

class Instruments(threading.Thread):
    """Samples the server's own counters throughout a run. (`_halt`, never `_stop` — Thread
    already defines `_stop()` and shadowing it makes join() raise.)"""
    def __init__(self, hz=4.0):
        super().__init__(daemon=True); self.hz = hz; self.rows = []; self._halt = threading.Event()
    def run(self):
        t0 = time.time()
        while not self._halt.is_set():
            self.rows.append(dict(t=time.time() - t0, **scrape())); time.sleep(1.0 / self.hz)
    def stop(self): self._halt.set(); self.join(timeout=3)
    def summary(self):
        if not self.rows: return {}
        g = lambda k: np.array([r[k] for r in self.rows], float)
        run_, wait, pre, kv = g("num_requests_running"), g("num_requests_waiting"), \
                              g("num_preemptions_total"), g("gpu_cache_usage_perc")
        fin = lambda a: a[np.isfinite(a)]
        pre_f = fin(pre)
        return dict(L_mean=float(np.nanmean(run_)), L_p95=float(np.nanpercentile(fin(run_), 95)) if len(fin(run_)) else float("nan"),
                    wait_mean=float(np.nanmean(wait)), wait_max=float(np.nanmax(wait)) if len(fin(wait)) else float("nan"),
                    preemptions=float(pre_f[-1] - pre_f[0]) if len(pre_f) > 1 else float("nan"),
                    kv_mean=float(np.nanmean(kv)), kv_max=float(np.nanmax(kv)) if len(fin(kv)) else float("nan"))

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
print("instruments ready:", list(_PATS))''')

# ------------------------------------------------------------------ 4. workload
md(r'''## 4. Workload and regimes

Same workload and arrival regimes as E9, so the numbers are directly comparable. The **bursty**
regime runs first: it is where offline tuning overshot by 66% and it finishes in about 40 minutes.
**high** is the saturated regime that produced E9's 43% energy error — it is what E13 needs.''')
code(r'''hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
probe = VLLMLoadProbe(base_url=f"http://127.0.0.1:{PORT}", ttl_s=0.25)
curve = EnergyCurve.from_jsonl()

TOK_CAP = MAX_LEN - 2384
for _q in wl.qids:
    wl.tok_th[_q] = min(wl.tok_th[_q], TOK_CAP); wl.tok_nt[_q] = min(wl.tok_nt[_q], TOK_CAP)
qtext = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}

LIVE_SEED = 0
LEVELS = {                                    # bursty first: fastest, and where E12's answer lives
    "bursty": dict(regime=dict(hi_rate=6.0, lo_rate=0.15, hi_s=6.0, lo_s=30.0), n=200),
    "high":   dict(regime=dict(hi_rate=2.2, lo_rate=2.2,  hi_s=30.0, lo_s=30.0), n=660),
}
BUDGET_MULT = 1.5     # authorised budget = 1.5x the MEASURED never-think energy of that regime
MAXB = MAX_NUM_SEQS

_sv = np.unique([SCORE[q] for q in wl.qids])
TAU_GRID = np.unique(np.concatenate([[0.0], _sv[::max(1, len(_sv)//180)], _sv[-3:], [1.01]]))
_cand = np.concatenate([TAU_GRID / max(hw.jtok(L) * DT_MEAN, 1e-12) for L in (1, 4, 16, 48)])
_cand = np.unique(_cand[np.isfinite(_cand) & (_cand > 0)])
LAM_GRID = np.unique(np.concatenate([_cand[::max(1, len(_cand)//240)], np.logspace(-8, 4, 40)]))
print(f"grids: tau {len(TAU_GRID)}  lambda {len(LAM_GRID)} (price-mapped, see CORRECTIONS S5b)")

def sim_all(reg, decide, n, seeds=(100, 101, 102)):
    rs = [simulate(make_trace(wl, n, s, **reg), decide, wl, hw, max_batch=MAXB) for s in seeds]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

def decider(family, v):
    return {"static": lambda L, q, c: SCORE[q] >= v,
            "load":   lambda L, q, c: SCORE[q] > v * hw.jtok(L) * DT_MEAN}[family]

def tune(reg, family, budget_wh, grid, n):
    best = None
    for v in grid:
        m = sim_all(reg, decider(family, v), n)
        if m["energy_wh"] <= budget_wh + 1e-9 and (best is None or m["acc"] > best[2]):
            best = (float(v), m["energy_wh"], m["acc"], m["think_rate"])
    return best
print("workload ready")''')

# ------------------------------------------------------------------ 5. the runner
md(r'''## 5. One runner, three policies

`static` and `load-aware` take a parameter fixed before the run — the offline method that failed in
E9. `closed-loop` takes no parameter: it starts from the same λ and adapts it against the meter at
every completion, counting energy already committed by in-flight requests.''')
code(r'''async def run_one(level, pname, trace, decide=None, ctl=None, budget_j=None):
    """decide(qid)->bool for open-loop policies; ctl=BudgetController for the closed loop."""
    inst = Instruments(); inst.start()
    meter.reset(); mon.begin_window(f"{level}-{pname}")
    fails, t_start, res = [], time.time(), []
    conn = aiohttp.TCPConnector(limit=256)
    print(f"  [{level}/{pname}] {len(trace)} queries, span {max(t for t,_ in trace)/60:.1f} min", flush=True)
    async with aiohttp.ClientSession(connector=conn, timeout=NO_TIMEOUT) as s:
        async def one(arr_t, qid):
            await asyncio.sleep(max(0.0, arr_t - (time.time() - t_start)))
            L = probe.read()
            if ctl is not None:
                d, h = ctl.decide(SCORE[qid], qid=qid); think = d.think
            else:
                think, h = decide(qid), None
            toks = int(round(wl.tok_th[qid] if think else wl.tok_nt[qid]))
            r = None
            for attempt in range(3):
                try:
                    r = await fire(s, qtext[qid], max(toks, 1), think=think); break
                except Exception as e:
                    if attempt == 2:
                        fails.append((qid, repr(e)[:120]))
                        if ctl is not None: ctl.complete(h)
                        return None
                    await asyncio.sleep(2.0 * (attempt + 1))
            if ctl is not None: ctl.complete(h)      # closes the loop on the METER
            r.update(qid=qid, think=bool(think), arrival=arr_t, L_at_admit=L,
                     lam=(ctl.r.lam if ctl is not None else float("nan")),
                     correct=(wl.ct[qid] if think else wl.cn[qid]))
            return r
        raw = await asyncio.gather(*[one(a, q) for a, q in trace], return_exceptions=True)
    m = mon.end_window(f"{level}-{pname}"); metered_j = meter(); inst.stop()
    res = [r for r in raw if isinstance(r, dict)]
    if len(res) < len(trace):
        print(f"    WARNING {len(trace)-len(res)}/{len(trace)} failed; first: {fails[:2]}")
    wall = time.time() - t_start
    lat = np.array([r["latency"] for r in res])
    row = dict(level=level, policy=pname, n=len(res), wall_s=wall,
               energy_wh=metered_j / 3600.0,                       # the meter: what E12 controls on
               energy_wh_zeus=m.gpu_energy[0] / 3600.0,            # independent cross-check
               energy_wh_idlesub=(metered_j - IDLE_W * wall) / 3600.0,
               budget_wh=(budget_j / 3600.0 if budget_j else float("nan")),
               acc=float(np.mean([r["correct"] for r in res])),
               think_rate=float(np.mean([r["think"] for r in res])),
               lat_mean=float(lat.mean()), lat_p95=float(np.percentile(lat, 95)),
               lam_final=(ctl.r.lam if ctl is not None else float("nan")),
               **inst.summary())
    if budget_j: row["vs_budget_pct"] = 100 * (metered_j - budget_j) / budget_j
    append(f"{OUT}/requests_{level}_{pname}.jsonl", res)
    append(f"{OUT}/summary.jsonl", [row])
    if ctl is not None: append(f"{OUT}/pacing_{level}.jsonl", ctl.trace)
    print(f"    -> {row['energy_wh']:.2f} Wh" + (f" ({row['vs_budget_pct']:+.1f}% vs budget)" if budget_j else "")
          + f" | acc {row['acc']:.4f} | think {row['think_rate']:.1%} | p95 {row['lat_p95']:.0f}s"
          + f" | L {row['L_mean']:.1f} | preempt {row['preemptions']:.0f} | KV {row['kv_max']:.2f}", flush=True)
    return row
print("runner ready")''')

# ------------------------------------------------------------------ 6. the experiment
md(r'''## 6. Run it

Per regime: measure never-think, set the budget from it, calibrate the offline method with that
same measurement, then run the three policies.

The one-point calibration `k = measured/simulated` is what makes this a fair test. Without it the
offline method would be failing on a constant scale error that any operator would trivially correct;
with it, whatever remains is the part that actually matters.''')
code(r'''ROWS, PLAN = [], {}
for level, cfg in LEVELS.items():
    reg, N = cfg["regime"], cfg["n"]
    trace = make_trace(wl, N, LIVE_SEED, **reg)
    print(f"\n=== {level}: {N} queries ===")

    # 1. measure never-think -> the budget, and the calibration constant
    r_never = await run_one(level, "never", trace, decide=lambda q: False)
    ROWS.append(r_never)
    # k must be measured on the SAME trace that was just run, or it conflates model error with
    # trace-to-trace variance. (Bursty's simulated never-think energy differs by 44% between the
    # tuning traces and the deployment trace -- calibrating across that would be meaningless.)
    sim_never = simulate(trace, lambda L, q, c: False, wl, hw, max_batch=MAXB)["energy_wh"]
    k = r_never["energy_wh"] / sim_never
    budget_wh = BUDGET_MULT * r_never["energy_wh"]
    sim_budget = budget_wh / k                      # the offline method gets the calibration too
    print(f"  measured never {r_never['energy_wh']:.2f} Wh | simulated {sim_never:.2f} Wh -> k={k:.3f}")
    print(f"  authorised budget = {BUDGET_MULT}x never = {budget_wh:.2f} Wh (simulator terms {sim_budget:.2f} Wh)")

    # 2. tune the open-loop policies against the calibrated budget
    tau, e_tau, a_tau, th_tau = tune(reg, "static", sim_budget, TAU_GRID, N)
    lam, e_lam, a_lam, th_lam = tune(reg, "load", sim_budget, LAM_GRID, N)
    PLAN[level] = dict(n=N, k=k, budget_wh=budget_wh, sim_budget_wh=sim_budget, tau=tau, lam=lam,
                       sim_e_static=e_tau, sim_e_load=e_lam, measured_never_wh=r_never["energy_wh"])
    print(f"  tau*={tau:.4f} (sim {e_tau:.2f} Wh, think {th_tau:.1%}) | "
          f"lam*={lam:.2e} (sim {e_lam:.2f} Wh, think {th_lam:.1%})")
    json.dump(PLAN, open(f"{OUT}/plan.json", "w"), indent=1, default=float)

    B = budget_wh * 3600.0
    # 3. open loop: parameters frozen before the run
    ROWS.append(await run_one(level, "static-offline", trace,
                              decide=(lambda t: (lambda q: SCORE[q] >= t))(tau), budget_j=B))
    ROWS.append(await run_one(level, "loadaware-offline", trace,
                              decide=(lambda l: (lambda q: SCORE[q] > l * curve.jtok(max(probe.read(), 1))
                                                 * DT_MEAN))(lam), budget_j=B))
    # 4. closed loop: same starting lambda, but paced on the meter
    router = EnergyAwareRouter(curve, probe, lam=lam, dtok_fn=ConstantDeltaTokens(DT_MEAN))
    ctl = BudgetController(router, budget_j=B, n_total=N, meter=meter)
    ROWS.append(await run_one(level, "loadaware-closedloop", trace, ctl=ctl, budget_j=B))
    print(f"  lambda: {lam:.2e} -> {ctl.r.lam:.2e} over the run")''')

# ------------------------------------------------------------------ 7. results
md(r'''## 7. E12 — did the loop hold the budget?''')
code(r'''import pandas as pd
df = pd.DataFrame(ROWS)
cols = ["level", "policy", "energy_wh", "budget_wh", "vs_budget_pct", "acc", "think_rate",
        "lat_p95", "L_mean", "preemptions", "kv_max", "wait_max", "n"]
print(df[[c for c in cols if c in df]].to_string(index=False))
df.to_json(f"{OUT}/summary_table.jsonl", orient="records", lines=True)

print("\n### budget adherence — the E12 answer\n")
for lv in PLAN:
    sub = {r["policy"]: r for r in ROWS if r["level"] == lv}
    print(f"{lv}: authorised {PLAN[lv]['budget_wh']:.2f} Wh")
    for p in ("static-offline", "loadaware-offline", "loadaware-closedloop"):
        if p in sub:
            print(f"   {p:22s} {sub[p]['energy_wh']:6.2f} Wh  {sub[p]['vs_budget_pct']:+7.1f}%   "
                  f"acc {sub[p]['acc']:.4f}  think {sub[p]['think_rate']:.1%}")

print("\n### meter vs Zeus (independent cross-check of the meter itself)\n")
print(df[["level", "policy", "energy_wh", "energy_wh_zeus"]].assign(
    diff_pct=lambda d: 100 * (d.energy_wh - d.energy_wh_zeus) / d.energy_wh_zeus).to_string(index=False))''')

md(r'''## 8. E13 — is preemption the reason the model breaks at saturation?

E9 predicted 41.5 Wh and measured 73.0 Wh at saturation (−43%), with p95 latency under-predicted
2.4×. The stated cause was vLLM preempting and recomputing under KV pressure, which the simulator
does not model. If that is right, the saturated runs should show **non-zero preemptions, a non-empty
waiting queue, and KV-cache usage near 1.0** — and the low-load runs should show none of it.''')
code(r'''print(df[["level", "policy", "preemptions", "kv_mean", "kv_max", "wait_mean", "wait_max",
          "L_mean", "lat_p95"]].to_string(index=False))

sat = df[df.level == "high"]
if len(sat):
    pre = sat.preemptions.max()
    print(f"\nsaturated regime: max preemptions in a run = {pre:.0f}, "
          f"peak KV usage = {sat.kv_max.max():.2f}, peak waiting queue = {sat.wait_max.max():.0f}")
    print("HYPOTHESIS CONFIRMED — preemption/recompute is a real mechanism here." if pre > 0 else
          "HYPOTHESIS REJECTED — no preemptions; the 43% error has another cause (look at the\\n"
          "  waiting queue and at whether the arrival replay itself fell behind).")
else:
    print("\nsaturated regime not run — E13 needs the 'high' level.")''')

md(r'''## 9. Figure and push''')
code(r'''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#ffffff"
C_POS, C_NEG = "#2a78d6", "#e34948"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
lvs = [l for l in PLAN]
fig, axes = plt.subplots(1, max(len(lvs), 1), figsize=(4.6 * max(len(lvs), 1), 3.8), squeeze=False)
POL = [("static-offline", "static\n(offline)"), ("loadaware-offline", "load-aware\n(offline)"),
       ("loadaware-closedloop", "load-aware\n(closed loop)")]
for ax, lv in zip(axes[0], lvs):
    sub = {r["policy"]: r for r in ROWS if r["level"] == lv}
    g = [(lab, sub[p]["vs_budget_pct"]) for p, lab in POL if p in sub]
    x = np.arange(len(g)); v = [b for _, b in g]
    ax.bar(x, v, width=.6, color=[C_NEG if t > 0 else C_POS for t in v], zorder=3)
    ax.axhline(0, color=MUTED, lw=1.2, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in g], fontsize=8)
    ax.set_title(f"{lv} — authorised {PLAN[lv]['budget_wh']:.1f} Wh", fontsize=9.5, color=INK,
                 loc="left", pad=6)
    ax.set_ylabel("energy vs authorised budget (%)")
    ax.grid(True, axis="y", color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
fig.suptitle("E12 — offline tuning vs a closed loop on a real energy meter  ·  "
             "red = over budget, blue = under", fontsize=10.5, color=INK, x=.008, ha="left")
fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(f"{OUT}/fig_closedloop.png", dpi=200)
print("wrote", f"{OUT}/fig_closedloop.png")''')
code(r'''meter.stop()
import subprocess
subprocess.run(["git", "add", "results/closedloop"], check=False)
subprocess.run(["git", "commit", "-m",
                "E12/E13: closed-loop budget control on a real meter; saturation diagnostics"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed")''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                     name="python3"),
                                     language_info=dict(name="python", version="3.11")),
          nbformat=4, nbformat_minor=5)
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/11_closed_loop_budget.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {path} ({len(cells)} cells)")
