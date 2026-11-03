"""E11 addendum -- two things REAL_TRACE.md asserts but does not measure.

1. TIMESCALE OF THE BURSTS. REAL_TRACE.md says conversational burstiness "lives on a slower timescale
   than a 500-request evaluation window resolves", quoting one Fano factor. This measures the raw
   traces at several window sizes, their inter-arrival CV, and how much original-trace time one
   500-request window actually covers.

2. IDLE POWER. The simulator charges power only while at least one request is in service. That is
   harmless when the servers compared are equally busy, but the code-completion server is idle for
   almost half the wall clock and its Poisson control almost never is, so the "2.4x less energy"
   reading needs the idle term before it can be quoted. This reproduces scripts/e11_real_trace.py's
   trace construction verbatim, re-runs only the never- and always-reasoning policies on the same 12
   evaluation traces, checks the busy-only energies against REAL_TRACE.md, and adds the measured A100
   idle draw (results/validation/replay_summary.jsonl) over each run's idle wall clock.

Free -- no GPU, under a minute. Needs data/azure/AzureLLMInferenceTrace_{conv,code}.csv.
Out: results/e11/ADDENDUM.md
"""
import os, sys, csv, json, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate

OUT = "results/e11"; os.makedirs(OUT, exist_ok=True)
TRACE_DIR = os.environ.get("AZURE_TRACE_DIR", "data/azure")
hw, wl = Hardware(), Workload()
N_TRACE, EVAL = 500, list(range(12))
IDLE_W = json.loads(open("results/validation/replay_summary.jsonl").readline())["idle_w"]


# ---------------------------------------------------------------- verbatim from e11_real_trace.py
def load_azure(name):
    path = os.path.join(TRACE_DIR, f"AzureLLMInferenceTrace_{name}.csv")
    ts = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts.append(dt.datetime.fromisoformat(r["TIMESTAMP"][:26]).timestamp())
    t = np.sort(np.asarray(ts, float))
    return t - t[0]


def dilate_to(t, target_rate):
    return t * (len(t) / (t[-1] - t[0]) / target_rate)


def thin_to(t, target_rate, rng):
    p = min(1.0, target_rate * (t[-1] - t[0]) / len(t))
    return t[rng.random(len(t)) < p]


def real_trace(t_all, n, seed, target_rate, mode="dilate"):
    rng = np.random.default_rng(seed)
    kept = dilate_to(t_all, target_rate) if mode == "dilate" else thin_to(t_all, target_rate, rng)
    if len(kept) <= n:
        raise RuntimeError(f"trace too short after {mode}: {len(kept)} <= {n}")
    i = int(rng.integers(0, len(kept) - n))
    arr = kept[i:i + n] - kept[i]
    qs = [wl.qids[j] for j in rng.integers(0, len(wl.qids), n)]
    return list(zip(arr.tolist(), qs))


CONV, CODE = load_azure("conv"), load_azure("code")
REGIMES = {
    "Poisson @0.8 (control)":    dict(kind="poisson", rate=0.8),
    "Azure conv @0.8":           dict(kind="real", t=CONV, rate=0.8, mode="dilate"),
    "Azure code @0.8":           dict(kind="real", t=CODE, rate=0.8, mode="dilate"),
    "Azure code @0.8 (thinned)": dict(kind="real", t=CODE, rate=0.8, mode="thin"),
}


def trace_for(rp, seed):
    if rp["kind"] == "poisson":
        return make_trace(wl, N_TRACE, seed, hi_rate=rp["rate"], lo_rate=rp["rate"],
                          hi_s=30.0, lo_s=30.0)
    return real_trace(rp["t"], N_TRACE, seed, rp["rate"], rp.get("mode", "dilate"))


# ---------------------------------------------------------------- 1. timescale of the bursts
# Measured on the ORIGINAL cluster-level traces, before any rescaling. Fano factor = variance / mean of
# arrival counts in fixed windows; a Poisson process gives 1 at every window size.
WINDOWS = (1, 10, 60)
stats = {}
for name, t in (("conversation", CONV), ("code completion", CODE)):
    span = t[-1] - t[0]
    ia = np.diff(t)
    fano = {}
    for w in WINDOWS:
        c = np.histogram(t, bins=np.arange(0.0, span + w, w))[0]
        fano[w] = float(c.var() / c.mean())
    stats[name] = dict(n=len(t), span_min=span / 60.0, rate=len(t) / span,
                       ia_cv=float(ia.std() / ia.mean()), fano=fano,
                       window_s=N_TRACE / (len(t) / span), dilation=(len(t) / span) / 0.8)

# ---------------------------------------------------------------- 2. the idle correction
shape = json.load(open(os.path.join(OUT, "e11_rows.json")))["shape"]
POLICIES = {"never": lambda L, q, c: False, "always": lambda L, q, c: True}

res = {}
for name, rp in REGIMES.items():
    res[name] = {}
    for pol, decide in POLICIES.items():
        busy, idle_c, idle_frac = [], [], []
        for s in EVAL:
            r = simulate(trace_for(rp, s), decide, wl, hw)
            idle = max(0.0, 1.0 - r["util"])          # util can exceed 1 by float error when never idle
            busy.append(r["energy_wh"])
            idle_c.append(r["energy_wh"] + IDLE_W * r["makespan"] * idle / 3600.0)
            idle_frac.append(idle)
        res[name][pol] = dict(busy=float(np.mean(busy)), idle=float(np.mean(idle_c)),
                              idle_frac=float(np.mean(idle_frac)))
    # guard: this must be the same experiment REAL_TRACE.md reports, or the correction means nothing
    for pol, key in (("never", "never_wh"), ("always", "always_wh")):
        if abs(res[name][pol]["busy"] - shape[name][key]) > 1e-6:
            raise SystemExit(f"{name} {pol}: {res[name][pol]['busy']:.4f} Wh does not reproduce "
                             f"REAL_TRACE.md's {shape[name][key]:.4f} Wh")

ctl, code, thin = (res["Poisson @0.8 (control)"]["never"], res["Azure code @0.8"]["never"],
                   res["Azure code @0.8 (thinned)"]["never"])

L = ["# E11 addendum: burst timescales and idle-corrected energy", "",
     "## 1. Timescale of the bursts (original traces, before rescaling)", "",
     "Fano factor is the variance over the mean of arrival counts in fixed windows; a Poisson process "
     "gives 1 at every window size, as does its inter-arrival CV. `window` is how much original-trace "
     f"time one {N_TRACE}-request evaluation window covers.", "",
     "|trace|requests|span|mean rate|inter-arrival CV|Fano 1 s|Fano 10 s|Fano 60 s|window|dilation to 0.8 req/s|",
     "|--|--|--|--|--|--|--|--|--|--|"]
for name, x in stats.items():
    L.append(f"|{name}|{x['n']}|{x['span_min']:.1f} min|{x['rate']:.2f} req/s|{x['ia_cv']:.2f}|"
             f"{x['fano'][1]:.2f}|{x['fano'][10]:.2f}|{x['fano'][60]:.2f}|{x['window_s']:.0f} s|"
             f"x{x['dilation']:.2f}|")
L += ["",
      "Conversational traffic is close to Poisson arrival by arrival (inter-arrival CV and 1 s Fano "
      "both near 1) and bursty only as a slow drift in rate, visible at 60 s. One evaluation window covers "
      "about 90 s of it, so a minute-scale drift barely registers inside a window. Code-completion traffic "
      "is bursty at every window size.", "",
      "## 2. Idle-corrected energy", "",
     f"The simulator charges no power while the GPU is idle. This adds the measured **{IDLE_W:.1f} W** "
     "A100 idle draw (weights resident, nothing in service; `results/validation/replay_summary.jsonl`) "
     "over each run's idle wall clock. Same 12 traces x 500 queries as `REAL_TRACE.md`; the busy-only "
     "columns reproduce that file exactly (checked in the script).", "",
     "|arrival process|policy|idle fraction|busy-only (Wh)|with idle (Wh)|",
     "|--|--|--|--|--|"]
for name in REGIMES:
    for pol in POLICIES:
        x = res[name][pol]
        L.append(f"|{name}|{pol}|{100 * x['idle_frac']:.1f}%|{x['busy']:.2f}|{x['idle']:.2f}|")
L += ["", "### Ratios", "",
      "|comparison|busy-only|with idle|", "|--|--|--|",
      f"|never-reason, Poisson / Azure code|{ctl['busy'] / code['busy']:.2f}x|{ctl['idle'] / code['idle']:.2f}x|",
      f"|never-reason, Poisson / Azure code (thinned)|{ctl['busy'] / thin['busy']:.2f}x|{ctl['idle'] / thin['idle']:.2f}x|"]
for name in REGIMES:
    x = res[name]
    L.append(f"|always / never, {name}|{x['always']['busy'] / x['never']['busy']:.2f}x|"
             f"{x['always']['idle'] / x['never']['idle']:.2f}x|")
L += ["", "### Reading", "",
      "The correction runs the opposite way to the one in the E8 concurrency axis. There the expensive, "
      "near-idle server was the one that idled, so charging idle power widened the gap. Here the cheap "
      "server idles: the code-completion server is idle for about half its wall clock and the Poisson "
      "control almost never. The burstiness discount therefore narrows once idle power is charged, but "
      "it survives at about a factor of two.", "",
      "*Repro: `python scripts/e11_addendum.py` (CPU, under a minute).*", ""]
open(os.path.join(OUT, "ADDENDUM.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
