"""Sanity/invariant checks on the simulation pipeline used by E1/E2/E4.

Not a unit test of style -- these are the checks that would CATCH the classes of bug we already
hit (silent unit mismatches, token round-trip errors, simulator drifting from the offline data,
non-monotone curves). Every check prints PASS/FAIL with the numbers, so failures are diagnosable.
Free, seconds to run.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate, WH, J_B1

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")

hw, wl = Hardware(), Workload()
print("=== 1. measured hardware curves ===")
jt = [hw.JTOK[b] for b in hw.B]
check("J/token strictly decreases with batch", all(jt[i] > jt[i+1] for i in range(len(jt)-1)),
      f"{ {b: round(hw.JTOK[b],3) for b in hw.B} }")
tp = [hw.TPUT[b] for b in hw.B]
check("aggregate throughput increases with batch", all(tp[i] < tp[i+1] for i in range(len(tp)-1)),
      f"{ {b: round(hw.TPUT[b]) for b in hw.B} }")
pw = [hw.POWER[b] for b in hw.B]
# NOT strictly monotone: measured power dips ~1.7% from batch 1->2 (noise). The physically
# meaningful invariant is that power is NEARLY FLAT while throughput scales -- that is precisely
# why J/token collapses: an almost-fixed power draw amortised over far more tokens.
check("GPU power is near-flat across batch (the reason J/token falls)",
      pw[-1] > pw[0] and max(pw) / min(pw) < 1.5,
      f"{ {b: round(hw.POWER[b]) for b in hw.B} } -> power +{100*(pw[-1]/pw[0]-1):.0f}% "
      f"while throughput +{100*(tp[-1]/tp[0]-1):.0f}%")
check("per-request decode rate falls with batch", hw.req_rate(1) > hw.req_rate(64),
      f"{hw.req_rate(1):.0f} -> {hw.req_rate(64):.1f} tok/s")
# energy identity: J/token should equal power / aggregate throughput
err = max(abs(hw.POWER[b]/hw.TPUT[b] - hw.JTOK[b]) / hw.JTOK[b] for b in hw.B)
check("J/token == power / throughput (internal consistency)", err < 0.02, f"max rel err {err*100:.1f}%")

print("\n=== 2. token round-trip from stored Wh ===")
tn = np.array([wl.tok_nt[q] for q in wl.qids]); tt = np.array([wl.tok_th[q] for q in wl.qids])
check("no-think token counts in a sane range", 1 <= tn.mean() <= 2000,
      f"mean {tn.mean():.0f}, median {np.median(tn):.0f}, max {tn.max():.0f}")
check("think token counts in a sane range", 100 <= tt.mean() <= 8000,
      f"mean {tt.mean():.0f}, median {np.median(tt):.0f}, max {tt.max():.0f}")
check("thinking generates more tokens on average", tt.mean() > tn.mean(),
      f"{tt.mean():.0f} vs {tn.mean():.0f} ({tt.mean()/tn.mean():.1f}x)")
raw = json.load(open("results/paired/paired_oracle_final.json"))
by = {p["qid"]: p for p in raw}
rt = max(abs(by[q]["energy_wh_think"] - wl.tok_th[q]*J_B1*WH) for q in wl.qids[:200])
check("Wh -> tokens -> Wh round-trip is exact", rt < 1e-9, f"max abs err {rt:.2e} Wh")

print("\n=== 3. simulator agrees with the offline dataset ===")
p = dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0)
accs_n, accs_t = [], []
for s in range(6):
    tr = make_trace(wl, 800, s, **p)
    accs_n.append(simulate(tr, lambda L, q, c: False, wl, hw)["acc"])
    accs_t.append(simulate(tr, lambda L, q, c: True, wl, hw)["acc"])
mcn = np.mean([wl.cn[q] for q in wl.qids]); mct = np.mean([wl.ct[q] for q in wl.qids])
check("sim never-think accuracy == dataset mean(correct_no_think)", abs(np.mean(accs_n) - mcn) < 0.02,
      f"sim {np.mean(accs_n):.4f} vs data {mcn:.4f}")
check("sim always-think accuracy == dataset mean(correct_think)", abs(np.mean(accs_t) - mct) < 0.02,
      f"sim {np.mean(accs_t):.4f} vs data {mct:.4f}")
check("always-think costs more energy than never-think",
      simulate(make_trace(wl, 800, 0, **p), lambda L, q, c: True, wl, hw)["energy_wh"] >
      simulate(make_trace(wl, 800, 0, **p), lambda L, q, c: False, wl, hw)["energy_wh"])

print("\n=== 4. simulator conservation / physics ===")
tr = make_trace(wl, 500, 3, **p)
r = simulate(tr, lambda L, q, c: wl.scores["ModernBERT"][q] > 0.5, wl, hw)
check("every request completes", r["n"] == 500, f"{r['n']}/500")
check("energy is positive and finite", 0 < r["energy_wh"] < 1e6, f"{r['energy_wh']:.1f} Wh")
check("p95 latency >= mean latency", r["lat_p95"] >= r["lat_mean"],
      f"p95 {r['lat_p95']:.1f}s, mean {r['lat_mean']:.1f}s")
# energy must lie between the all-at-max-efficiency and all-at-min-efficiency bounds
toks = sum(wl.tok_th[q] if wl.scores["ModernBERT"][q] > 0.5 else wl.tok_nt[q] for _, q in tr)
lo_b, hi_b = toks*min(hw.JTOK.values())*WH, toks*max(hw.JTOK.values())*WH
check("energy within [min-J, max-J] token bounds", lo_b <= r["energy_wh"] <= hi_b,
      f"{lo_b:.1f} <= {r['energy_wh']:.1f} <= {hi_b:.1f} Wh")

print("\n=== 5. monotonicity: more thinking => more energy AND more accuracy ===")
E, A = [], []
for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rng = np.random.default_rng(0)
    rr = simulate(make_trace(wl, 600, 5, **p), (lambda x: (lambda L, q, c: rng.random() < x))(rate), wl, hw)
    E.append(rr["energy_wh"]); A.append(rr["acc"])
check("energy increases with think-rate", all(E[i] <= E[i+1]+1e-9 for i in range(len(E)-1)),
      " -> ".join(f"{e:.1f}" for e in E))
check("accuracy increases with think-rate", all(A[i] <= A[i+1]+0.02 for i in range(len(A)-1)),
      " -> ".join(f"{a:.4f}" for a in A))

print("\n=== 6. the committed-energy (actuation lag) signal is live ===")
seen = {"inflight": 0.0}
def probe(L, q, c):
    seen["inflight"] = max(seen["inflight"], c["inflight_j"]); return True
simulate(make_trace(wl, 300, 1, **p), probe, wl, hw)
check("ctx exposes non-zero committed in-flight energy", seen["inflight"] > 0,
      f"max observed {seen['inflight']*WH:.2f} Wh committed")

print("\n" + ("ALL CHECKS PASSED" if ok_all else "*** SOME CHECKS FAILED ***"))
sys.exit(0 if ok_all else 1)
