"""Shared trace-driven serving simulator for the energy-budget experiments (E1, E2, E4...).

Physics is entirely MEASURED (results/batching/concurrency.jsonl):
  - per-request decode rate  = throughput_tok_s(L) / L
  - GPU power                = power_w(L)          -> energy = integral power_w(L(t)) dt
  - per-token energy         = energy_per_out_token_j(L)   (~42x spread over L=1..64)
Event-driven: between events the in-flight set (hence L, rate, power) is constant, so the next
completion time is exact -- no time-step discretisation error. Prefill time omitted
(decode-dominated at these lengths); flagged in every write-up that uses this.
"""
import json, statistics as st
from collections import defaultdict
import numpy as np

J_B1 = 3.7            # J/token at batch 1, used to recover decode token counts from stored Wh
WH = 1 / 3600.0
MAX_BATCH = 48        # KV-cache-limited in-service concurrency; excess requests FIFO-wait


def load_curves(path="results/batching/concurrency.jsonl"):
    conc = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    agg = defaultdict(lambda: defaultdict(list))
    for r in conc:
        agg[r["batch"]]["tput"].append(r["throughput_tok_s"])
        agg[r["batch"]]["power"].append(r["power_w"])
        agg[r["batch"]]["jtok"].append(r["energy_per_out_token_j"])
    B = sorted(agg)
    return dict(BATCHES=B,
                TPUT={b: st.mean(agg[b]["tput"]) for b in B},
                POWER={b: st.mean(agg[b]["power"]) for b in B},
                JTOK={b: st.mean(agg[b]["jtok"]) for b in B})


class Hardware:
    """Measured curves, with knobs for the extrapolations the simulator makes (see E7).

    kv_alpha : models KV-cache growth -- per-token cost scaled by (1 + kv_alpha * tokens/1000).
               The curves were measured at l_out=256, but thinking requests average ~2.7k tokens,
               so kv_alpha>0 tests whether under-pricing long generations changes any conclusion.
    interp   : "linear" (default) or "step" (nearest measured batch at or below L) -- tests
               sensitivity to how we fill the gaps between measured batch sizes.
    """
    def __init__(self, curves=None, kv_alpha=0.0, interp="linear"):
        c = curves or load_curves()
        self.B = c["BATCHES"]; self.TPUT = c["TPUT"]; self.POWER = c["POWER"]; self.JTOK = c["JTOK"]
        self.kv_alpha = kv_alpha; self.interp = interp

    def _i(self, table, L):
        if L <= self.B[0]: return table[self.B[0]]
        if L >= self.B[-1]: return table[self.B[-1]]
        if self.interp == "step":
            return table[max(b for b in self.B if b <= L)]
        return float(np.interp(L, self.B, [table[b] for b in self.B]))

    def req_rate(self, L): return self._i(self.TPUT, L) / L if L > 0 else 0.0
    def power(self, L): return self._i(self.POWER, L) if L > 0 else 0.0
    def jtok(self, L, tokens=0.0):
        j = self._i(self.JTOK, L)
        return j * (1.0 + self.kv_alpha * tokens / 1000.0) if self.kv_alpha else j


class Workload:
    """Per-query outcomes + recovered decode token counts from the paired OTB run."""
    def __init__(self, paired="results/paired/paired_oracle_final.json",
                 probes="results/paired/probes_v1.jsonl",
                 fte="results/paired/router_fte_modernbert.jsonl", seed=11):
        pd = json.load(open(paired))
        pr = {json.loads(l)["qid"]: json.loads(l) for l in open(probes, encoding="utf-8")}
        ft = {json.loads(l)["qid"]: json.loads(l)["p_think"] for l in open(fte, encoding="utf-8")}
        by = {p["qid"]: p for p in pd}
        self.qids = sorted(set(by) & set(pr) & set(ft))
        self.cn = {q: by[q]["correct_no_think"] for q in self.qids}
        self.ct = {q: by[q]["correct_think"] for q in self.qids}
        self.tok_nt = {q: by[q]["energy_wh_no_think"] * 3600 / J_B1 for q in self.qids}
        self.tok_th = {q: by[q]["energy_wh_think"] * 3600 / J_B1 for q in self.qids}
        rng = np.random.default_rng(seed)
        self.scores = {"ModernBERT": {q: ft[q] for q in self.qids},
                       "p_need": {q: pr[q]["p_need_think"] for q in self.qids},
                       "random": {q: float(rng.random()) for q in self.qids}}

    def dtok(self, q): return max(self.tok_th[q] - self.tok_nt[q], 0.0)
    def dacc(self, q): return self.ct[q] - self.cn[q]


def make_trace(wl, n, seed, hi_rate, lo_rate, hi_s, lo_s):
    """Poisson arrivals alternating hi_rate (hi_s s) and lo_rate (lo_s s); queries sampled iid."""
    rng = np.random.default_rng(seed)
    times, t = [], 0.0
    while len(times) < n:
        phase = t % (hi_s + lo_s)
        t += rng.exponential(1.0 / (hi_rate if phase < hi_s else lo_rate))
        times.append(t)
    qs = [wl.qids[i] for i in rng.integers(0, len(wl.qids), n)]
    return list(zip(times, qs))


def simulate(trace, decide, wl, hw, on_complete=None, max_batch=MAX_BATCH, prefill_j=0.0):
    """decide(L, qid, ctx) -> think(bool).  L = in-service count (= vLLM num_requests_running).

    ctx exposes what a real controller can read from the server:
      energy_j   -- energy metered so far
      n_done     -- completed requests
      inflight_j -- energy ALREADY COMMITTED by in-service requests (their remaining tokens priced
                    at the current J(L)). A budget controller must count this: a think-decision's
                    energy arrives long after the decision, so spend-only feedback overshoots.
    """
    arrivals = sorted(trace); ai, t = 0, 0.0
    inservice, waitq, done = [], [], []
    energy_j = 0.0
    load_t, load_L = [], []          # time-weighted load samples (for E4's load statistics)
    nonlocal_energy = [0.0]          # prefill energy charged at admission (closure-writable)

    def ctx(L):
        return dict(energy_j=energy_j, n_done=len(done), t=t,
                    inflight_j=sum(r["rem"] for r in inservice) * hw.jtok(max(L, 1)))

    def start(q, arr_t):
        L = len(inservice)
        think = decide(L, q, ctx(L))
        tot = wl.tok_th[q] if think else wl.tok_nt[q]
        inservice.append(dict(rem=tot, total=tot,
                              correct=(wl.ct[q] if think else wl.cn[q]),
                              think=think, arr=arr_t))
        if prefill_j:
            nonlocal_energy[0] += prefill_j          # one-off prefill cost on admission
    while ai < len(arrivals) or inservice or waitq:
        L = len(inservice); rate = hw.req_rate(L)
        t_comp = min((r["rem"] / rate for r in inservice), default=float("inf")) if rate > 0 else float("inf")
        t_arr = (arrivals[ai][0] - t) if ai < len(arrivals) else float("inf")
        dt = min(t_comp, t_arr)
        if dt == float("inf"): break
        if L > 0:
            pw = hw.power(L)
            if getattr(hw, "kv_alpha", 0.0):
                # KV growth: price the batch up by the mean generated-so-far length of its members
                gen = np.mean([r["total"] - r["rem"] for r in inservice]) if inservice else 0.0
                pw *= 1.0 + hw.kv_alpha * gen / 1000.0
            energy_j += pw * dt
            for r in inservice: r["rem"] -= rate * dt
        load_t.append(dt); load_L.append(L)      # weight each observed L by its duration
        t += dt
        if t_arr <= t_comp:
            _, q = arrivals[ai]; ai += 1
            if len(inservice) < max_batch: start(q, t)
            else: waitq.append((t, q))
        else:
            for r in [r for r in inservice if r["rem"] <= 1e-6]:
                r["latency"] = t - r["arr"]; done.append(r)
                if on_complete: on_complete(ctx(len(inservice)))
            inservice = [r for r in inservice if r["rem"] > 1e-6]
            while waitq and len(inservice) < max_batch:
                arr_t, q = waitq.pop(0); start(q, arr_t)
    energy_j += nonlocal_energy[0]
    lat = np.array([r["latency"] for r in done])
    w = np.array(load_t); Ls = np.array(load_L, dtype=float)
    busy = w[Ls > 0]; Lb = Ls[Ls > 0]
    # time-weighted load stats over BUSY time (idle gaps would swamp the mean under low arrival rates)
    mL = float((Lb * busy).sum() / busy.sum()) if busy.sum() > 0 else 0.0
    varL = float(((Lb - mL) ** 2 * busy).sum() / busy.sum()) if busy.sum() > 0 else 0.0
    return dict(acc=float(np.mean([r["correct"] for r in done])), energy_wh=energy_j * WH,
                think_rate=float(np.mean([r["think"] for r in done])), n=len(done),
                lat_mean=float(lat.mean()), lat_p95=float(np.percentile(lat, 95)), makespan=t,
                load_mean=mL, load_cv=float(varL ** 0.5 / mL) if mL > 0 else 0.0,
                util=float(busy.sum() / w.sum()) if w.sum() > 0 else 0.0)


def bisect(fn, target, lo=1e-10, hi=1e6, steps=26, increasing=False):
    """Find x with fn(x) ~= target. `increasing` = fn rises with x."""
    for _ in range(steps):
        mid = (lo * hi) ** 0.5
        v = fn(mid)
        if (v > target) == increasing: hi = mid
        else: lo = mid
    return (lo * hi) ** 0.5
