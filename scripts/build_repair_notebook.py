"""Builds notebooks/13_controller_repair.ipynb (GPU) — E16.

E12 measured the closed-loop budget controller failing on real hardware: +36.7% over budget on
bursty traffic, +134.4% at saturation, worse than the offline tuning it was built to repair. Its
pacing trace shows lambda driven five orders of magnitude to its floor within 29 of 200 requests.

This runs three controllers against the SAME budget on the SAME traces, so the question stops being
"does the fix work" (binary, and one run either way proves little) and becomes "how much guarding
does this loop actually need" -- which is answerable in one session and informative whichever way it
lands:

  as-shipped   the E12 controller, unchanged. Re-run because this is a different rental on a
               different day; without a same-session control, any improvement could be the box.
  asymmetric   kp_loosen = kp/4. Loosening is the dangerous direction -- overspending breaches a
               cap, underspending merely forfeits accuracy. E15 found this the most promising of
               three candidates IN SIMULATION, and also found that the simulator cannot reproduce
               the live failure, so it is a candidate and not a known fix.
  guarded      asymmetric, plus a 1.1x cap on the per-update step, plus lambda confined to
               [lambda*/10, lambda*x100] around the offline-tuned value, plus a 15% warmup. The
               premise is that offline tuning gets lambda right to within about a decade and
               feedback should correct it by a bounded amount rather than by five orders of
               magnitude.

A note on step_limit sizing, learned the hard way: a loose cap does almost nothing when it compounds.
1.5 per update reaches 10^5 in 29 updates, which is exactly the collapse E12 suffered. 1.1 reaches
15x. Choose it against the number of updates a run will make.

EXPECT THIS TO BE GENUINELY UNCERTAIN. E15's simulation could not reproduce the live failure, and at
saturation even its best variant breached 42% of traces. If all three overspend, that is a real
result about energy-budgeted reasoning control being hard, and it is better evidence than one
attempt was.

Run:  python scripts/build_repair_notebook.py
Then: run notebooks/13_controller_repair.ipynb on an A100 and push the outputs.
"""
import json, os

cells = []
def _cid(): return f"cell-{len(cells):02d}"
def md(t):   cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(r'''# E16 — how much guarding does the budget loop actually need?

E12 measured `BudgetController` failing on real hardware:

| regime | offline static | offline load-aware | **closed loop** |
|--|--|--|--|
| bursty | +37.0% | −15.7% | **+36.7%** |
| high | +17.7% | +42.0% | **+134.4%** |

Its pacing trace shows λ driven five orders of magnitude to its floor within **29 of 200 requests**,
producing an 82% think-rate in the second quartile of arrivals.

Three controllers, same budget, same traces:

| | change | rationale |
|--|--|--|
| **as-shipped** | none | control — different rental, different day |
| **asymmetric** | `kp_loosen = kp/4` | loosening breaches caps; tightening only forfeits accuracy |
| **guarded** | + 1.1× step cap, λ ∈ [λ\*/10, λ\*·100], 15% warmup | trust offline tuning to a decade; let feedback correct within that |

**This is genuinely uncertain.** E15 could not reproduce the live failure in simulation, and at
saturation its best variant still breached 42% of traces. If all three overspend, that is a real
finding about energy-budgeted reasoning control, better evidenced than one attempt was.

**~90 min for both regimes.** Bursty runs first and answers the question on its own — you can stop
after it.''')

# ---------------- setup cells (identical to notebook 11) ----------------
md(r'''## 0. Repo setup''')
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
os.makedirs("results/repair", exist_ok=True)
print("cwd:", os.getcwd())''')

md(r'''## 1. Install''')
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
OUT = "results/repair"

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

md(r'''## 3. Meter, instruments, request path

Identical to notebook 11, so E16's numbers are directly comparable to E12's. The NVML counter agreed
with Zeus to 0.0% across all eight E12 runs; it is not the suspect here.''')
code(r'''import asyncio, aiohttp, time, json, re, numpy as np, threading, urllib.request
import nest_asyncio; nest_asyncio.apply()
from zeus.monitor import ZeusMonitor
from src.router import (EnergyAwareRouter, VLLMLoadProbe, ConstantDeltaTokens, EnergyCurve,
                        BudgetController, NVMLEnergyMeter)
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
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as r:
            txt = r.read().decode()
    except Exception:
        return {k: float("nan") for k in _PATS}
    out = {}
    for k, pat in _PATS.items():
        m = pat.search(txt); out[k] = float(m.group(1)) if m else float("nan")
    return out

class Instruments(threading.Thread):
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
        fin = lambda a: a[np.isfinite(a)]
        pre = fin(g("num_preemptions_total")); run_ = g("num_requests_running")
        return dict(L_mean=float(np.nanmean(run_)),
                    preemptions=float(pre[-1] - pre[0]) if len(pre) > 1 else float("nan"),
                    kv_max=float(np.nanmax(fin(g("gpu_cache_usage_perc"))) or 0)
                            if len(fin(g("gpu_cache_usage_perc"))) else float("nan"),
                    wait_max=float(np.nanmax(fin(g("num_requests_waiting"))))
                            if len(fin(g("num_requests_waiting"))) else float("nan"))

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
print("ready")''')

md(r'''## 4. Workload, regimes, and the three controllers

Budget = 1.5× the **measured** never-think energy, exactly as E12 constructed it, so the two
experiments are directly comparable.''')
code(r'''hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
DT_MEAN = float(np.mean([wl.dtok(q) for q in wl.qids]))
probe = VLLMLoadProbe(base_url=f"http://127.0.0.1:{PORT}", ttl_s=0.25)
curve = EnergyCurve.from_jsonl()

TOK_CAP = MAX_LEN - 2384
for _q in wl.qids:
    wl.tok_th[_q] = min(wl.tok_th[_q], TOK_CAP); wl.tok_nt[_q] = min(wl.tok_nt[_q], TOK_CAP)
qtext = {p["qid"]: p["question"] for p in json.load(open("results/paired/paired_oracle_final.json"))}

LIVE_SEED, BUDGET_MULT, MAXB = 0, 1.5, MAX_NUM_SEQS
LEVELS = {"bursty": dict(regime=dict(hi_rate=6.0, lo_rate=0.15, hi_s=6.0, lo_s=30.0), n=200,
                         variants=("as-shipped", "asymmetric", "guarded")),
          "high":   dict(regime=dict(hi_rate=2.2, lo_rate=2.2, hi_s=30.0, lo_s=30.0), n=660,
                         variants=("as-shipped", "guarded"))}

_sv = np.unique([SCORE[q] for q in wl.qids])
TAU_GRID = np.unique(np.concatenate([[0.0], _sv[::max(1, len(_sv)//180)], _sv[-3:], [1.01]]))
_cand = np.concatenate([TAU_GRID / max(hw.jtok(L) * DT_MEAN, 1e-12) for L in (1, 4, 16, 48)])
_cand = np.unique(_cand[np.isfinite(_cand) & (_cand > 0)])
LAM_GRID = np.unique(np.concatenate([_cand[::max(1, len(_cand)//240)], np.logspace(-8, 4, 40)]))

def sim_all(reg, decide, n, seeds=(100, 101, 102)):
    rs = [simulate(make_trace(wl, n, s, **reg), decide, wl, hw, max_batch=MAXB) for s in seeds]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}

def tune_lambda(reg, budget_wh, n):
    best = None
    for v in LAM_GRID:
        m = sim_all(reg, (lambda l: (lambda L, q, c: SCORE[q] > l * hw.jtok(L) * DT_MEAN))(v), n)
        if m["energy_wh"] <= budget_wh + 1e-9 and (best is None or m["acc"] > best[2]):
            best = (float(v), m["energy_wh"], m["acc"], m["think_rate"])
    return best

def controller_kw(variant, lam):
    """The three configurations. `guarded` bounds lambda around the offline-tuned value."""
    if variant == "as-shipped":  return dict()
    if variant == "asymmetric":  return dict(kp_loosen=0.125)
    if variant == "guarded":     return dict(kp_loosen=0.125, step_limit=1.1, warmup_frac=0.15,
                                             lam_bounds=(lam / 10.0, lam * 100.0))
    raise ValueError(variant)
print("ready")''')

md(r'''## 5. Run''')
code(r'''async def run_one(level, pname, trace, ctl=None, decide=None, budget_j=None):
    inst = Instruments(); inst.start()
    meter.reset(); mon.begin_window(f"{level}-{pname}")
    fails, t_start = [], time.time()
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
            if ctl is not None: ctl.complete(h)
            r.update(qid=qid, think=bool(think), arrival=arr_t, L_at_admit=L,
                     lam=(ctl.r.lam if ctl is not None else float("nan")),
                     correct=(wl.ct[qid] if think else wl.cn[qid]))
            return r
        raw = await asyncio.gather(*[one(a, q) for a, q in trace], return_exceptions=True)
    m = mon.end_window(f"{level}-{pname}"); metered_j = meter(); inst.stop()
    res = [r for r in raw if isinstance(r, dict)]
    if len(res) < len(trace): print(f"    WARNING {len(trace)-len(res)} failed; {fails[:2]}")
    wall = time.time() - t_start
    lat = np.array([r["latency"] for r in res])
    lams = np.array([r["lam"] for r in res], float)
    row = dict(level=level, policy=pname, n=len(res), wall_s=wall, energy_wh=metered_j / 3600.0,
               budget_wh=(budget_j / 3600.0 if budget_j else float("nan")),
               acc=float(np.mean([r["correct"] for r in res])),
               think_rate=float(np.mean([r["think"] for r in res])),
               lat_p95=float(np.percentile(lat, 95)),
               lam_min=float(np.nanmin(lams)) if np.isfinite(lams).any() else float("nan"),
               lam_max=float(np.nanmax(lams)) if np.isfinite(lams).any() else float("nan"),
               lam_final=(ctl.r.lam if ctl is not None else float("nan")), **inst.summary())
    if budget_j: row["vs_budget_pct"] = 100 * (metered_j - budget_j) / budget_j
    append(f"{OUT}/requests_{level}_{pname}.jsonl", res)
    append(f"{OUT}/summary.jsonl", [row])
    if ctl is not None: append(f"{OUT}/pacing_{level}_{pname}.jsonl", ctl.trace)
    print(f"    -> {row['energy_wh']:.2f} Wh" + (f" ({row['vs_budget_pct']:+.1f}%)" if budget_j else "")
          + f" | acc {row['acc']:.4f} | think {row['think_rate']:.1%} | p95 {row['lat_p95']:.0f}s"
          + f" | lam {row['lam_min']:.1e}..{row['lam_max']:.1e} | preempt {row['preemptions']:.0f}", flush=True)
    return row

ROWS, PLAN = [], {}
for level, cfg in LEVELS.items():
    reg, N = cfg["regime"], cfg["n"]
    trace = make_trace(wl, N, LIVE_SEED, **reg)
    print(f"\n=== {level}: {N} queries ===")
    r_never = await run_one(level, "never", trace, decide=lambda q: False)
    ROWS.append(r_never)
    budget_wh = BUDGET_MULT * r_never["energy_wh"]
    sim_never = simulate(trace, lambda L, q, c: False, wl, hw, max_batch=MAXB)["energy_wh"]
    k = r_never["energy_wh"] / sim_never
    lam, e_lam, a_lam, th_lam = tune_lambda(reg, budget_wh / k, N)
    PLAN[level] = dict(n=N, k=k, budget_wh=budget_wh, lam=lam, measured_never_wh=r_never["energy_wh"])
    print(f"  budget {budget_wh:.2f} Wh (k={k:.3f}) | lam*={lam:.2e} (sim {e_lam:.2f} Wh, think {th_lam:.1%})")
    json.dump(PLAN, open(f"{OUT}/plan.json", "w"), indent=1, default=float)
    B = budget_wh * 3600.0
    for variant in cfg["variants"]:
        router = EnergyAwareRouter(curve, probe, lam=lam, dtok_fn=ConstantDeltaTokens(DT_MEAN))
        ctl = BudgetController(router, budget_j=B, n_total=N, meter=meter,
                               **controller_kw(variant, lam))
        ROWS.append(await run_one(level, variant, trace, ctl=ctl, budget_j=B))''')

md(r'''## 6. Results''')
code(r'''import pandas as pd
df = pd.DataFrame(ROWS)
print(df[[c for c in ["level","policy","energy_wh","budget_wh","vs_budget_pct","acc","think_rate",
                      "lat_p95","lam_min","lam_max","lam_final","preemptions","kv_max","n"]
          if c in df]].to_string(index=False))
df.to_json(f"{OUT}/summary_table.jsonl", orient="records", lines=True)

E12 = {("bursty","closed loop"): 36.7, ("high","closed loop"): 134.4}
print("\n### vs E12 (same budget construction, same traces)\n")
for lv in PLAN:
    print(f"{lv}: authorised {PLAN[lv]['budget_wh']:.2f} Wh   [E12 as-shipped was "
          f"{E12.get((lv,'closed loop'),float('nan')):+.1f}%]")
    for r in ROWS:
        if r["level"] == lv and r["policy"] != "never":
            print(f"   {r['policy']:12s} {r['energy_wh']:6.2f} Wh  {r['vs_budget_pct']:+7.1f}%   "
                  f"acc {r['acc']:.4f}  think {r['think_rate']:5.1%}  "
                  f"lam {r['lam_min']:.1e}..{r['lam_max']:.1e}")
    print()
print("Read the lam range: if it still spans many decades, the loop is still losing control,")
print("whatever the final energy happened to be.")''')

md(r'''## 7. Push''')
code(r'''meter.stop()
import subprocess
subprocess.run(["git", "add", "results/repair"], check=False)
subprocess.run(["git", "commit", "-m", "E16: guarded closed-loop budget controllers on hardware"], check=False)
subprocess.run(["git", "pull", "--rebase", "-q"], check=False)
subprocess.run(["git", "push"], check=False)
print("pushed")''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                     name="python3"),
                                     language_info=dict(name="python", version="3.11")),
          nbformat=4, nbformat_minor=5)
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/13_controller_repair.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {path} ({len(cells)} cells)")
