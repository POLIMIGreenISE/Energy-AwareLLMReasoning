r"""Energy-aware reasoning router for a live vLLM server.

This is the deployable form of the rule the simulation studies (RQ4). It reads the server's real
concurrency, prices the query in JOULES at that concurrency using the measured J(L) curve, and
decides whether reasoning is worth buying right now:

    think(q)  <=>  score(q)  >  lambda * J(L) * dtok(q)
                               \_________ _________/
                                 the extra joules this query
                                 costs AT THIS MOMENT

  score(q) : router's estimate that thinking flips this query (0..1)
  J(L)     : measured joules per output token at in-service concurrency L -- spans 42x (3.42 -> 0.08)
  dtok(q)  : extra output tokens thinking would generate
  lambda   : the price of accuracy in joules; the Lagrange multiplier of
             "maximise accuracy s.t. energy <= B". Fixed offline, or adapted online (BudgetController).

Deliberately dependency-light (stdlib + numpy): importable and testable with no GPU and no server.

--------------------------------------------------------------------------------------------------
THE ONE HONEST GAP, MADE EXPLICIT
--------------------------------------------------------------------------------------------------
`dtok(q)` is not observable before generating. The simulator can read it from the paired dataset; a
live router cannot. So it is a pluggable ESTIMATOR here, and the default is a CONSTANT (the mean
measured dtok), which collapses the rule to

    think(q)  <=>  score(q) > lambda * J(L) * E[dtok]        -- load-aware, token-blind per query

i.e. a threshold that still moves with load but not with the individual query. That is the honest
deployable policy; `OracleDeltaTokens` exists only to reproduce simulator/replay runs exactly and
to measure what per-query token knowledge is actually worth. Do not ship it.
"""
from __future__ import annotations

import json
import re
import statistics as st
import threading
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

WH = 1 / 3600.0
DEFAULT_CURVES = "results/batching/concurrency.jsonl"


# ==================================================================== measured hardware
class EnergyCurve:
    """Joules per output token as a function of in-service concurrency L, from measurement.

    Built from the concurrency sweep (A100-SXM4-40GB, Qwen3-8B, vLLM, Zeus/NVML). Linear between
    measured batch sizes, clamped outside the measured range -- extrapolating past batch 64 would
    invent physics, and clamping is the conservative choice (it never prices a token cheaper than
    anything actually observed).
    """

    def __init__(self, batches, jtok, power=None, tput=None):
        self.B = list(batches)
        self.JTOK = dict(jtok)
        self.POWER = dict(power or {})
        self.TPUT = dict(tput or {})

    @classmethod
    def from_jsonl(cls, path: str = DEFAULT_CURVES) -> "EnergyCurve":
        agg = defaultdict(lambda: defaultdict(list))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                agg[r["batch"]]["jtok"].append(r["energy_per_out_token_j"])
                agg[r["batch"]]["power"].append(r["power_w"])
                agg[r["batch"]]["tput"].append(r["throughput_tok_s"])
        B = sorted(agg)
        return cls(B,
                   {b: st.mean(agg[b]["jtok"]) for b in B},
                   {b: st.mean(agg[b]["power"]) for b in B},
                   {b: st.mean(agg[b]["tput"]) for b in B})

    def _interp(self, table, L: float) -> float:
        if L <= self.B[0]:
            return table[self.B[0]]
        if L >= self.B[-1]:
            return table[self.B[-1]]
        return float(np.interp(L, self.B, [table[b] for b in self.B]))

    def jtok(self, L: float) -> float:
        """Joules per output token at concurrency L (L<1 is treated as 1: someone is being served)."""
        return self._interp(self.JTOK, max(float(L), 1.0))

    def power_w(self, L: float) -> float:
        return self._interp(self.POWER, max(float(L), 1.0))

    @property
    def spread(self) -> float:
        return max(self.JTOK.values()) / min(self.JTOK.values())


# ==================================================================== the live load signal
class LoadProbe:
    """Interface: return the server's current in-service request count."""

    def read(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class StaticLoadProbe(LoadProbe):
    """Fixed L. For tests, and for the load-blind ablation (pin L to pretend load never changes)."""

    def __init__(self, L: float = 1.0):
        self.L = float(L)

    def read(self) -> float:
        return self.L


_METRIC_RE = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.]+)", re.M)


class VLLMLoadProbe(LoadProbe):
    """Polls a vLLM server's Prometheus endpoint for `vllm:num_requests_running`.

    Cached for `ttl_s` so a burst of routing decisions does not become a burst of HTTP calls, and
    it FAILS SOFT: on any error the last good reading is reused (or `fallback` if there is none).
    A router that crashes because the metrics endpoint hiccuped is worse than one that prices a
    query with a slightly stale L.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", ttl_s: float = 0.25,
                 timeout_s: float = 2.0, fallback: float = 1.0):
        self.url = base_url.rstrip("/") + "/metrics"
        self.ttl_s, self.timeout_s, self.fallback = ttl_s, timeout_s, float(fallback)
        self._last: Optional[float] = None
        self._at = 0.0
        self._lock = threading.Lock()
        self.n_reads = self.n_errors = 0

    def read(self) -> float:
        now = time.monotonic()
        with self._lock:
            if self._last is not None and now - self._at < self.ttl_s:
                return self._last
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as r:
                m = _METRIC_RE.search(r.read().decode())
            val = float(m.group(1)) if m else None
        except Exception:
            val = None
        with self._lock:
            self.n_reads += 1
            if val is None:
                self.n_errors += 1
                return self._last if self._last is not None else self.fallback
            self._last, self._at = val, now
            return val


# ==================================================================== dtok estimators
class ConstantDeltaTokens:
    """The deployable default: one number, measured offline. See the module docstring."""

    def __init__(self, mean_dtok: float = 2370.0):
        self.mean = float(mean_dtok)

    def __call__(self, query: str, qid=None) -> float:
        return self.mean


class OracleDeltaTokens:
    """Per-query ground truth, keyed by qid. REPLAY/ABLATION ONLY -- not deployable."""

    def __init__(self, table: dict, default: float = 2370.0):
        self.table, self.default = dict(table), float(default)

    def __call__(self, query: str, qid=None) -> float:
        return float(self.table.get(qid, self.default))


# ==================================================================== the router
@dataclass
class Decision:
    think: bool
    score: float
    L: float
    j_per_token: float
    dtok: float
    tau: float          # the threshold score(q) had to clear: lambda * J(L) * dtok
    joules: float       # priced cost of letting this query think, right now
    lam: float

    def as_dict(self) -> dict:
        return dict(think=self.think, score=self.score, L=self.L, j_per_token=self.j_per_token,
                    dtok=self.dtok, tau=self.tau, joules=self.joules, lam=self.lam)


class EnergyAwareRouter:
    """think(q) iff score(q) > lambda * J(L) * dtok(q), with L read live from the server.

        router = EnergyAwareRouter.from_measurements(lam=1e-3, base_url="http://127.0.0.1:8000")
        d = router.decide(score=0.31, query="...")
        body = router.request_body("Qwen/Qwen3-8B", "...", d.think, max_tokens=1024)

    `lam=0` thinks on everything with a positive score; `lam=inf` never thinks.
    """

    def __init__(self, curve: EnergyCurve, probe: LoadProbe, lam: float,
                 dtok_fn: Callable[..., float] = None):
        self.curve, self.probe = curve, probe
        self.lam = float(lam)
        self.dtok_fn = dtok_fn or ConstantDeltaTokens()
        self.log: list[Decision] = []

    @classmethod
    def from_measurements(cls, lam: float, base_url: str = "http://127.0.0.1:8000",
                          curves_path: str = DEFAULT_CURVES, dtok_fn=None,
                          probe: LoadProbe = None) -> "EnergyAwareRouter":
        return cls(EnergyCurve.from_jsonl(curves_path),
                   probe or VLLMLoadProbe(base_url), lam, dtok_fn)

    def decide(self, score: float, query: str = "", qid=None, L: float = None,
               record: bool = True) -> Decision:
        L = self.probe.read() if L is None else float(L)
        if L != L:                      # NaN: the probe never got a reading
            L = 1.0
        j = self.curve.jtok(L)
        dtok = float(self.dtok_fn(query, qid))
        tau = self.lam * j * dtok
        d = Decision(think=bool(score > tau), score=float(score), L=L, j_per_token=j, dtok=dtok,
                     tau=tau, joules=j * dtok, lam=self.lam)
        if record:
            self.log.append(d)
        return d

    def threshold(self, L: float, dtok: float = None) -> float:
        """tau(L, q) -- the score a query must beat at concurrency L. Useful for plots and tests."""
        return self.lam * self.curve.jtok(L) * (self.dtok_fn("", None) if dtok is None else dtok)

    @staticmethod
    def request_body(model: str, prompt: str, think: bool, max_tokens: int = 2048,
                     temperature: float = 0.0, **extra) -> dict:
        """vLLM OpenAI-compatible body. Qwen3's thinking switch is a chat-template kwarg."""
        return dict(model=model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(max_tokens), temperature=temperature,
                    chat_template_kwargs={"enable_thinking": bool(think)}, **extra)

    @property
    def think_rate(self) -> float:
        return float(np.mean([d.think for d in self.log])) if self.log else 0.0


# ==================================================================== the live energy meter
class NVMLEnergyMeter:
    """Joules consumed since `start()`, read live from the GPU. Calling it returns that total.

    Two mechanisms, in order of preference:

      1. `nvmlDeviceGetTotalEnergyConsumption` -- a monotonic millijoule counter maintained by the
         driver since it loaded. Exact, free to read, no sampling error. Available on Volta and
         later (the A100 has it).
      2. Integrating `nvmlDeviceGetPowerUsage` on a background thread at `hz`. Used only where the
         counter is missing; a trapezoid integral, so it carries the usual sampling error.

    This is what lets BudgetController close its loop on MEASURED rather than modelled energy.
    Zeus is not used here because Zeus measures named windows, and a controller needs a running
    total it can read at any instant.
    """

    def __init__(self, index: int = 0, hz: float = 10.0):
        self.index, self.hz = int(index), float(hz)
        self._h = None
        self._base = None
        self._counter = True
        self._acc_j = 0.0
        self._halt = threading.Event()
        self._thread = None

    def start(self) -> "NVMLEnergyMeter":
        import pynvml
        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        try:
            self._base = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)   # mJ
        except Exception:
            self._counter = False                                             # fall back to power
            self._thread = threading.Thread(target=self._integrate, daemon=True)
            self._thread.start()
        return self

    def _integrate(self):
        last_t, last_w = time.monotonic(), self._power_w()
        while not self._halt.is_set():
            time.sleep(1.0 / self.hz)
            now, w = time.monotonic(), self._power_w()
            self._acc_j += 0.5 * (last_w + w) * (now - last_t)                # trapezoid
            last_t, last_w = now, w

    def _power_w(self) -> float:
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0       # mW -> W
        except Exception:
            return 0.0

    def __call__(self) -> float:
        if self._counter:
            return (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h) - self._base) / 1000.0
        return self._acc_j

    def reset(self):
        """Zero the meter -- call between runs so each run's budget starts from zero."""
        if self._counter:
            self._base = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
        else:
            self._acc_j = 0.0

    def stop(self):
        self._halt.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def mechanism(self) -> str:
        return "nvml total-energy counter" if self._counter else "integrated power sampling"


# ==================================================================== closed loop
class BudgetController:
    """Adapts lambda against a real energy meter so the run lands on an authorised budget.

    The subtlety that cost us a whole experiment (CORRECTIONS.md S4): a think-decision's energy
    arrives LONG after the decision. Feeding back on metered spend alone, the controller sees a low
    bill, loosens lambda, and over-thinks badly (76% vs 14% think-rate at the same cap). So the
    pacing signal counts energy already COMMITTED by in-flight requests as well as energy spent.

        ctl = BudgetController(router, budget_j=..., n_total=..., meter=NVMLEnergyMeter().start())
        d = ctl.decide(score, query, qid)          # admits, and books the commitment
        ...                                        # generate
        ctl.complete(handle)                       # release it; spend is read from the meter

    `meter` is a zero-argument callable returning total joules consumed since the run began. Given
    one, the loop closes on MEASURED energy and the controller is immune to the model being wrong --
    which matters, because E9 showed the model is wrong by 43% at saturation and offline tuning
    consequently overshot its budget by 66-79% there. Without a meter the loop still runs, but on
    the router's own cost model, so it corrects trace-to-trace variation and nothing else.
    """

    def __init__(self, router: EnergyAwareRouter, budget_j: float, n_total: int,
                 kp: float = 0.5, warmup_frac: float = 0.05,
                 lam_bounds: tuple = (1e-8, 1e4), meter: Callable[[], float] = None,
                 kp_loosen: float = None, step_limit: float = None):
        """`kp_loosen` and `step_limit` exist because the plain loop FAILED on hardware (E12).

        Measured on an A100: +36.7% over budget on bursty traffic and +134.4% at saturation, worse
        than the offline tuning it was meant to repair. Its pacing trace shows why -- lambda driven
        five orders of magnitude to its floor within 29 of 200 requests, producing an 82% think-rate
        in the second quartile of arrivals, long before the bill for those decisions arrived.

        Both knobs attack the same asymmetry: LOOSENING is the dangerous direction, because
        overspending breaches a cap while underspending merely leaves accuracy unclaimed.

          kp_loosen  : gain used only when pace < 1 (spending under target). Default = kp, i.e.
                       symmetric, i.e. the shipped behaviour that failed. E15 found kp/4 the most
                       promising of three candidate repairs -- in simulation, which could not
                       reproduce the live failure, so this is a candidate and not a known fix.
          step_limit : cap on the multiplicative move per update, e.g. 1.1. Note that a loose cap
                       does almost nothing over many updates: 1.5 compounds to 10^5 in 29 updates,
                       which is exactly the collapse observed. Choose it against the number of
                       updates a run will make, not for its own sake.
        """
        self.r, self.budget_j, self.n_total = router, float(budget_j), int(n_total)
        self.kp, self.warmup_frac, self.lam_bounds = kp, warmup_frac, lam_bounds
        self.kp_loosen = kp if kp_loosen is None else float(kp_loosen)
        self.step_limit = step_limit
        self.meter = meter
        self._booked_j = 0.0
        self.n_done = 0
        self._inflight: dict[int, float] = {}
        self._next = 0
        self._lock = threading.Lock()
        self.trace: list[dict] = []          # per-completion pacing record, for the write-up

    @property
    def spent_j(self) -> float:
        """Measured joules when a meter is attached; otherwise the router's own booked estimate."""
        return float(self.meter()) if self.meter is not None else self._booked_j

    @property
    def committed_j(self) -> float:
        return self.spent_j + sum(self._inflight.values())

    def decide(self, score: float, query: str = "", qid=None):
        d = self.r.decide(score, query, qid)
        with self._lock:
            h = self._next
            self._next += 1
            self._inflight[h] = d.joules if d.think else 0.0   # what we just promised to spend
        return d, h

    def complete(self, handle: int, metered_j: float = None):
        """Release a request. `metered_j` is the true energy if you have a meter; without one the
        booked estimate is used, which keeps the pacing loop running on the router's own model."""
        with self._lock:
            booked = self._inflight.pop(handle, 0.0)
            self._booked_j += booked if metered_j is None else float(metered_j)
            self.n_done += 1
            frac = self.n_done / max(self.n_total, 1)
            if frac < self.warmup_frac:
                return
            pace = self.committed_j / max(self.budget_j * frac, 1e-9)   # >1 = spending too fast
            lam0 = self.r.lam
            k = self.kp if pace >= 1.0 else self.kp_loosen              # loosening is the risky way
            step = pace ** k
            if self.step_limit:
                step = float(np.clip(step, 1.0 / self.step_limit, self.step_limit))
            self.r.lam = float(np.clip(lam0 * step, *self.lam_bounds))
            self.trace.append(dict(n_done=self.n_done, frac=frac, spent_j=self.spent_j,
                                   committed_j=self.committed_j, pace=pace,
                                   lam_before=lam0, lam_after=self.r.lam))

    def report(self) -> dict:
        return dict(spent_wh=self.spent_j * WH, budget_wh=self.budget_j * WH,
                    vs_budget_pct=100 * (self.spent_j - self.budget_j) / self.budget_j,
                    n_done=self.n_done, lam=self.r.lam, think_rate=self.r.think_rate)


# ==================================================================== baseline, for head-to-heads
class StaticThresholdRouter:
    """The incumbent: think(q) iff score(q) >= tau. Load-blind by construction.

    Present so a live head-to-head runs the SAME code path as the load-aware router -- same probe,
    same logging, same request body -- with only the decision rule differing.
    """

    def __init__(self, tau: float, probe: LoadProbe = None):
        self.tau = float(tau)
        self.probe = probe
        self.log: list[Decision] = []

    def decide(self, score: float, query: str = "", qid=None, L: float = None,
               record: bool = True) -> Decision:
        if L is None:
            L = self.probe.read() if self.probe else float("nan")   # observed, never used
        d = Decision(think=bool(score >= self.tau), score=float(score), L=float(L),
                     j_per_token=float("nan"), dtok=float("nan"), tau=self.tau,
                     joules=float("nan"), lam=float("nan"))
        if record:
            self.log.append(d)
        return d

    request_body = staticmethod(EnergyAwareRouter.request_body)

    @property
    def think_rate(self) -> float:
        return float(np.mean([d.think for d in self.log])) if self.log else 0.0
