"""Tests for src/router.py -- the deployable energy-aware router. Run: python tests/test_router.py

These check the properties the thesis claims, not just that the code runs:
  * the price curve the router reads is the MEASURED one (42x spread, J = power / throughput);
  * the threshold MOVES with load, in the right direction and by the right factor;
  * the same query gets opposite answers on an idle vs a busy server (the whole point);
  * the router's rule is bit-identical to the simulator's, so the artefact and the experiments
    cannot silently diverge;
  * the budget controller counts in-flight commitments (the bug that cost us an experiment);
  * the live probe fails soft rather than taking the serving path down with it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.router import (EnergyCurve, StaticLoadProbe, VLLMLoadProbe, ConstantDeltaTokens,
                        OracleDeltaTokens, EnergyAwareRouter, StaticThresholdRouter,
                        BudgetController)

CURVES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results", "batching", "concurrency.jsonl")


def test_curve_is_the_measured_one():
    c = EnergyCurve.from_jsonl(CURVES)
    assert c.B == [1, 2, 4, 8, 16, 32, 64]
    assert abs(c.jtok(1) - 3.42) < 0.05, c.jtok(1)
    assert abs(c.jtok(64) - 0.082) < 0.01, c.jtok(64)
    assert 40 < c.spread < 45, c.spread          # the 42x headline
    js = [c.jtok(b) for b in c.B]
    assert all(js[i] > js[i + 1] for i in range(len(js) - 1)), js


def test_curve_internal_consistency():
    """J/token must equal power / aggregate throughput at every measured batch size."""
    c = EnergyCurve.from_jsonl(CURVES)
    for b in c.B:
        assert abs(c.POWER[b] / c.TPUT[b] - c.JTOK[b]) / c.JTOK[b] < 0.02, b


def test_curve_clamps_outside_measured_range():
    """Never invent physics past batch 64, and never price below anything observed."""
    c = EnergyCurve.from_jsonl(CURVES)
    assert c.jtok(200) == c.jtok(64)
    assert c.jtok(0) == c.jtok(1) and c.jtok(-5) == c.jtok(1)   # L<1 -> someone is being served


def test_threshold_falls_as_the_server_fills():
    """tau(L) = lam*J(L)*dtok. Busier server -> cheaper tokens -> lower bar to think."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES, probe=StaticLoadProbe(1))
    taus = [r.threshold(L) for L in (1, 2, 4, 8, 16, 32, 64)]
    assert all(taus[i] > taus[i + 1] for i in range(len(taus) - 1)), taus
    assert taus[0] / taus[-1] > 40, taus[0] / taus[-1]


def test_same_query_opposite_answers_by_load():
    """The claim in one assertion: identical query, identical tokens, load decides."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(1),
                                            dtok_fn=ConstantDeltaTokens(2000))
    assert r.decide(score=0.30, L=1).think is False      # idle: 3.42*2000*1e-3 = 6.84 > 0.30
    assert r.decide(score=0.30, L=32).think is True      # busy: 0.134*2000*1e-3 = 0.27 < 0.30


def test_priced_joules_match_the_curve():
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(1),
                                            dtok_fn=ConstantDeltaTokens(2000))
    d1, d32 = r.decide(0.5, L=1), r.decide(0.5, L=32)
    assert abs(d1.joules - 3.42 * 2000) / d1.joules < 0.02
    assert d1.joules / d32.joules > 20        # the range one constant would have to stand in for


def test_lambda_extremes():
    r = EnergyAwareRouter.from_measurements(lam=0.0, curves_path=CURVES, probe=StaticLoadProbe(8))
    assert r.decide(1e-9, L=8).think is True              # lam=0 -> anything positive thinks
    r.lam = float("inf")
    assert r.decide(0.999, L=8).think is False            # lam=inf -> never


def test_rule_matches_the_simulator_exactly():
    """The artefact and the experiments must implement the SAME rule."""
    from src.serving_sim import Hardware, Workload
    hw, wl = Hardware(), Workload()
    score = wl.scores["ModernBERT"]
    lam = 1e-3
    r = EnergyAwareRouter.from_measurements(
        lam=lam, curves_path=CURVES, probe=StaticLoadProbe(1),
        dtok_fn=OracleDeltaTokens({q: wl.dtok(q) for q in wl.qids}))
    rng = np.random.default_rng(0)
    qs = [wl.qids[i] for i in rng.integers(0, len(wl.qids), 300)]
    for q in qs:
        for L in (1, 3, 7, 16, 33, 64):
            sim = score[q] > lam * hw.jtok(L) * wl.dtok(q)      # scripts/e1..e8 pol_load
            assert r.decide(score[q], qid=q, L=L, record=False).think == sim, (q, L)


def test_static_router_is_load_blind():
    s = StaticThresholdRouter(tau=0.5, probe=StaticLoadProbe(1))
    assert {s.decide(0.6, L=L).think for L in (1, 8, 64)} == {True}
    assert {s.decide(0.4, L=L).think for L in (1, 8, 64)} == {False}
    assert s.decide(0.5, L=1).think is True               # >= tau, matches the sweep code


def test_probe_fails_soft():
    """A dead metrics endpoint must degrade to a fallback, never raise into the serving path."""
    p = VLLMLoadProbe(base_url="http://127.0.0.1:9", ttl_s=0.0, timeout_s=0.25, fallback=4.0)
    assert p.read() == 4.0 and p.n_errors == 1
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES, probe=p)
    assert isinstance(r.decide(0.5).think, bool)


def test_probe_caches_within_ttl():
    p = VLLMLoadProbe(base_url="http://127.0.0.1:9", ttl_s=60.0, timeout_s=0.25, fallback=2.0)
    p._last, p._at = 7.0, __import__("time").monotonic()
    assert p.read() == 7.0 and p.n_reads == 0             # served from cache, no HTTP at all


def test_budget_controller_counts_inflight_commitments():
    """The lag correction: energy promised but not yet metered must still pace the controller."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(32),
                                            dtok_fn=ConstantDeltaTokens(2000))
    c = BudgetController(r, budget_j=10_000.0, n_total=100)
    ds = [c.decide(0.9) for _ in range(5)]
    assert all(d.think for d, _ in ds)
    assert c.spent_j == 0.0                               # nothing metered yet...
    assert c.committed_j > 0.0                            # ...but the bill is already promised
    assert abs(c.committed_j - 5 * ds[0][0].joules) < 1e-6


def test_budget_controller_tightens_when_overspending():
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(32),
                                            dtok_fn=ConstantDeltaTokens(2000))
    c = BudgetController(r, budget_j=1_000.0, n_total=100)   # tiny budget: must clamp down
    lam0 = r.lam
    for _ in range(40):
        _, h = c.decide(0.9)
        c.complete(h)
    assert r.lam > lam0, (lam0, r.lam)                    # higher lam = higher bar = less thinking
    r2 = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                             probe=StaticLoadProbe(32),
                                             dtok_fn=ConstantDeltaTokens(2000))
    c2 = BudgetController(r2, budget_j=10_000_000.0, n_total=100)   # huge budget: must loosen
    for _ in range(40):
        _, h = c2.decide(0.9)
        c2.complete(h)
    assert r2.lam < 1e-3, r2.lam


def test_budget_controller_reads_a_meter_when_given_one():
    """With a meter attached, spend is MEASURED, not accumulated from the router's own estimates.

    This is the whole point of the closed loop: E9 showed the cost model is 43% wrong at saturation,
    so a controller that paces on its own model inherits that error, while one reading a meter does
    not.
    """
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(32),
                                            dtok_fn=ConstantDeltaTokens(2000))
    box = {"j": 0.0}
    c = BudgetController(r, budget_j=10_000.0, n_total=100, meter=lambda: box["j"])
    _, h = c.decide(0.9)
    assert c.spent_j == 0.0
    box["j"] = 4321.0                       # the GPU burned energy the model never predicted
    assert c.spent_j == 4321.0
    assert c.committed_j > 4321.0           # plus what is still in flight
    c.complete(h)
    assert c.spent_j == 4321.0              # still the meter, not the booked estimate


def test_budget_controller_tightens_on_metered_overspend_the_model_cannot_see():
    """The failure mode E9 found: the model says the budget is fine, the meter says it is not."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(32),
                                            dtok_fn=ConstantDeltaTokens(10))   # model thinks ~free
    box = {"j": 0.0}
    c = BudgetController(r, budget_j=1000.0, n_total=100, meter=lambda: box["j"])
    lam0 = r.lam
    for i in range(40):
        _, h = c.decide(0.9)
        box["j"] += 500.0                   # reality: wildly over budget
        c.complete(h)
    assert r.lam > lam0, (lam0, r.lam)
    assert c.trace and c.trace[-1]["pace"] > 1.0


def test_asymmetric_gain_loosens_slower_than_it_tightens():
    """The E12 repair: overspending breaches a cap, underspending only forfeits accuracy."""
    def ctl(**kw):
        r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                                probe=StaticLoadProbe(8),
                                                dtok_fn=ConstantDeltaTokens(2000))
        return r, BudgetController(r, budget_j=1e6, n_total=10, warmup_frac=0.0, **kw)
    # pace ~= 0.3, the value seen throughout E12's live collapse
    box = {"j": 3e4}                  # target after 1 of 10 completions is 1e5
    r_sym, c_sym = ctl(meter=lambda: box["j"])
    r_asy, c_asy = ctl(meter=lambda: box["j"], kp_loosen=0.125)
    for c in (c_sym, c_asy):
        _, h = c.decide(0.0)          # score 0 -> never thinks -> no in-flight commitment
        c.complete(h)
    assert r_sym.lam < 1e-3 and r_asy.lam < 1e-3          # both loosened
    assert r_asy.lam > r_sym.lam, (r_sym.lam, r_asy.lam)  # asymmetric loosened LESS


def test_zero_spend_collapses_lambda_in_one_update_unless_step_limited():
    """An edge case found while testing the repair, and a real robustness bug.

    If the meter reads zero -- a genuinely idle start, or a meter that has not warmed up -- then
    pace = 0 and lambda *= 0**k = 0 for ANY gain, so the controller slams to its floor in a single
    update and thinks on everything. Asymmetry does not help; only a step limit does. This is the
    E12 collapse in miniature, and it is why step_limit is not merely a smoothing nicety.
    """
    def build(**kw):
        r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                                probe=StaticLoadProbe(8))
        return r, BudgetController(r, budget_j=1e6, n_total=10, warmup_frac=0.0,
                                   meter=lambda: 0.0, **kw)
    r_bare, c_bare = build()
    _, h = c_bare.decide(0.0); c_bare.complete(h)
    assert r_bare.lam == 1e-8, r_bare.lam                 # floor, in ONE update

    r_lim, c_lim = build(step_limit=1.1)
    _, h = c_lim.decide(0.0); c_lim.complete(h)
    assert r_lim.lam > 9e-4, r_lim.lam                    # barely moved


def test_step_limit_caps_how_far_lambda_moves_per_update():
    """E12 collapsed lambda 10^5 in 29 updates. A step cap has to be tight enough to matter."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(8),
                                            dtok_fn=ConstantDeltaTokens(2000))
    c = BudgetController(r, budget_j=1e9, n_total=10, warmup_frac=0.0, step_limit=1.1,
                         meter=lambda: 0.0)
    lam0 = r.lam
    _, h = c.decide(0.0); c.complete(h)
    assert r.lam >= lam0 / 1.1 - 1e-18, (lam0, r.lam)     # one update cannot exceed the cap
    for _ in range(9):
        _, h = c.decide(0.0); c.complete(h)
    assert r.lam >= lam0 / (1.1 ** 10) - 1e-18            # nor can ten, compounding


def test_default_controller_is_still_the_shipped_symmetric_one():
    """The fix is opt-in. E12's behaviour must remain reproducible by default."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(8))
    c = BudgetController(r, budget_j=1000.0, n_total=10)
    assert c.kp_loosen == c.kp and c.step_limit is None


def test_budget_controller_records_a_pacing_trace():
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES,
                                            probe=StaticLoadProbe(8),
                                            dtok_fn=ConstantDeltaTokens(2000))
    c = BudgetController(r, budget_j=50_000.0, n_total=50)
    for _ in range(20):
        _, h = c.decide(0.5)
        c.complete(h)
    assert len(c.trace) > 0
    row = c.trace[-1]
    assert set(row) >= {"n_done", "frac", "spent_j", "committed_j", "pace", "lam_before", "lam_after"}
    assert row["n_done"] == 20


def test_request_body_toggles_qwen_thinking():
    b = EnergyAwareRouter.request_body("Qwen/Qwen3-8B", "hi", think=True, max_tokens=64)
    assert b["chat_template_kwargs"] == {"enable_thinking": True}
    assert b["messages"][0]["content"] == "hi" and b["max_tokens"] == 64
    assert StaticThresholdRouter.request_body("m", "p", False)["chat_template_kwargs"] == \
        {"enable_thinking": False}


def test_constant_dtok_is_the_default_not_the_oracle():
    """Guard the honest default: a fresh router must NOT be using per-query token knowledge."""
    r = EnergyAwareRouter.from_measurements(lam=1e-3, curves_path=CURVES, probe=StaticLoadProbe(8))
    assert isinstance(r.dtok_fn, ConstantDeltaTokens)
    assert r.decide(0.5, qid="a", L=8).dtok == r.decide(0.5, qid="b", L=8).dtok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
        except Exception as e:
            failed += 1; print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
