"""E3 -- the energy/latency/accuracy tension: what does load-aware routing cost in latency?

E1 exposed the trade-off but did not quantify it: the load-aware policy deliberately thinks DURING
bursts (when J/token is cheapest), which is exactly when queues are longest -- so it buys energy
efficiency with tail latency.

Three policies span the tension. The INVERSE policy is the control that tests the mechanism: if
"think when busy" is really what saves energy, then "think when idle" must show the mirror image
(worse energy, better latency) rather than being simply worse at everything.

  static-tau     load-blind: one fixed threshold                (baseline)
  load-aware     think MORE at high L  (energy-first)           score > lam * J(L) * dtok
  inverse        think MORE at low  L  (latency-first)          score > lam * dtok / J(L)

Then the operator question: given BOTH an energy cap AND a p95 latency SLO, which policy delivers
the most accuracy, and is the load-aware win preserved once latency is constrained?

Free (no GPU). Out: results/e3/{e3_tradeoff.md, e3_tradeoff.png}
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.serving_sim import Hardware, Workload, make_trace, simulate

OUT = "results/e3"; os.makedirs(OUT, exist_ok=True)
hw, wl = Hardware(), Workload()
SCORE = wl.scores["ModernBERT"]
REGIMES = {"bimodal": dict(hi_rate=6.0, lo_rate=0.03, hi_s=6.0, lo_s=110.0),
           "stable":  dict(hi_rate=2.5, lo_rate=0.15, hi_s=25.0, lo_s=45.0)}
N_TRACE, N_SEEDS = 500, 5
_sv = np.array(sorted(SCORE.values()))
TAU_GRID = np.unique(np.concatenate([[0.0], np.quantile(_sv, np.linspace(0.002, 0.998, 16)), [1.01]]))
LAM_GRID = np.logspace(-7, 3, 16)
INV_GRID = np.logspace(-4, 6, 16)

def pol_static(t):  return lambda L, q, c: SCORE[q] >= t
def pol_load(lam):  return lambda L, q, c: SCORE[q] > lam * hw.jtok(L) * wl.dtok(q)
def pol_inv(lam):   return lambda L, q, c: SCORE[q] > lam * wl.dtok(q) / max(hw.jtok(L), 1e-9)

FAMILIES = {"static-tau": (TAU_GRID, pol_static),
            "load-aware (think when busy)": (LAM_GRID, pol_load),
            "inverse (think when idle)": (INV_GRID, pol_inv)}

def sweep(regime, grid, fn):
    """(energy, accuracy, p95 latency, think-rate) averaged over held-out traces, per parameter."""
    out = []
    for v in grid:
        E, A, P, T = [], [], [], []
        for s in range(N_SEEDS):
            r = simulate(make_trace(wl, N_TRACE, s, **regime), fn(v), wl, hw)
            E.append(r["energy_wh"]); A.append(r["acc"]); P.append(r["lat_p95"]); T.append(r["think_rate"])
        out.append(dict(param=float(v), energy=float(np.mean(E)), acc=float(np.mean(A)),
                        p95=float(np.mean(P)), think=float(np.mean(T))))
    return out

def best_under(points, e_cap, slo):
    """Highest accuracy among operating points satisfying BOTH constraints."""
    feas = [p for p in points if p["energy"] <= e_cap + 1e-9 and p["p95"] <= slo + 1e-9]
    return max(feas, key=lambda p: p["acc"]) if feas else None

DATA, rows = {}, []
for reg, p in REGIMES.items():
    nev = np.mean([simulate(make_trace(wl, N_TRACE, s, **p), lambda L, q, c: False, wl, hw)["energy_wh"]
                   for s in range(N_SEEDS)])
    alw_runs = [simulate(make_trace(wl, N_TRACE, s, **p), lambda L, q, c: True, wl, hw) for s in range(N_SEEDS)]
    alw = np.mean([r["energy_wh"] for r in alw_runs])
    lat_never = np.mean([simulate(make_trace(wl, N_TRACE, s, **p), lambda L, q, c: False, wl, hw)["lat_p95"]
                         for s in range(N_SEEDS)])
    lat_always = np.mean([r["lat_p95"] for r in alw_runs])
    DATA[reg] = {n: sweep(p, g, f) for n, (g, f) in FAMILIES.items()}
    print(f"\n### {reg}: never {nev:.0f} Wh / p95 {lat_never:.0f}s | always {alw:.0f} Wh / p95 {lat_always:.0f}s")

    # --- the tension, at a matched energy cap: what does each policy cost in latency?
    B = nev + 0.40 * (alw - nev)
    print(f"  at a {B:.0f} Wh cap (40% of always-think):")
    print(f"    {'policy':32} {'acc':>7} {'p95 lat':>9} {'think':>7}")
    for n, pts in DATA[reg].items():
        feas = [x for x in pts if x["energy"] <= B + 1e-9]
        b = max(feas, key=lambda x: x["acc"]) if feas else None
        if b:
            print(f"    {n:32} {b['acc']:.4f} {b['p95']:8.1f}s {b['think']*100:6.0f}%")
            rows.append((reg, "energy-cap", n, b["acc"], b["energy"], b["p95"], b["think"]))

    # --- the operator question: accuracy under BOTH an energy cap and a latency SLO
    print(f"  accuracy under BOTH a {B:.0f} Wh cap and a p95 SLO:")
    slos = [lat_never * k for k in (1.5, 3.0, 6.0, 1e6)]
    hdr = "    " + f"{'policy':32}" + "".join(
        f"{('SLO ' + (f'{s:.0f}s' if s < 1e5 else 'none')):>12}" for s in slos)
    print(hdr)
    for n, pts in DATA[reg].items():
        cells = []
        for s in slos:
            b = best_under(pts, B, s)
            cells.append(f"{b['acc']:.4f}" if b else "infeas.")
            rows.append((reg, f"slo={s:.0f}", n, b["acc"] if b else np.nan,
                         b["energy"] if b else np.nan, b["p95"] if b else np.nan,
                         b["think"] if b else np.nan))
        print("    " + f"{n:32}" + "".join(f"{c:>12}" for c in cells))

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
COL = {"static-tau": "tab:orange", "load-aware (think when busy)": "tab:blue",
       "inverse (think when idle)": "tab:red"}
reg = "bimodal"
ax = axes[0]
for n, pts in DATA[reg].items():
    o = sorted(pts, key=lambda x: x["energy"])
    ax.plot([x["energy"] for x in o], [x["acc"] for x in o], "o-", ms=4, color=COL[n], label=n)
ax.set_xlabel("energy (Wh)"); ax.set_ylabel("accuracy"); ax.set_title("Energy vs accuracy (bimodal)")
ax.grid(alpha=.3); ax.legend(fontsize=7)
ax = axes[1]
for n, pts in DATA[reg].items():
    o = sorted(pts, key=lambda x: x["p95"])
    ax.plot([x["p95"] for x in o], [x["acc"] for x in o], "o-", ms=4, color=COL[n], label=n)
ax.set_xlabel("p95 latency (s)"); ax.set_ylabel("accuracy"); ax.set_title("Latency vs accuracy (bimodal)")
ax.grid(alpha=.3); ax.legend(fontsize=7)
ax = axes[2]
for n, pts in DATA[reg].items():
    o = sorted(pts, key=lambda x: x["energy"])
    ax.plot([x["energy"] for x in o], [x["p95"] for x in o], "o-", ms=4, color=COL[n], label=n)
ax.set_xlabel("energy (Wh)"); ax.set_ylabel("p95 latency (s)")
ax.set_title("The tension: energy vs latency"); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/e3_tradeoff.png", dpi=130)

with open(f"{OUT}/e3_tradeoff.md", "w", encoding="utf-8") as f:
    f.write("# E3 — energy / latency / accuracy under load\n\n")
    f.write(f"{N_SEEDS} held-out traces of {N_TRACE} queries. The **inverse** policy (think when the "
            "server is idle) is the control: if 'think when busy' is what saves energy, the inverse "
            "must show the mirror trade-off rather than being worse at everything.\n\n")
    f.write("|regime|constraint|policy|accuracy|energy (Wh)|p95 latency (s)|think-rate|\n")
    f.write("|--|--|--|--|--|--|--|\n")
    for reg, con, n, a, e, l, t in rows:
        aa = f"{a:.4f}" if a == a else "infeasible"
        ee = f"{e:.1f}" if e == e else "—"; ll = f"{l:.1f}" if l == l else "—"
        tt = f"{t*100:.0f}%" if t == t else "—"
        f.write(f"|{reg}|{con}|{n}|{aa}|{ee}|{ll}|{tt}|\n")
print(f"\nwrote {OUT}/e3_tradeoff.md, e3_tradeoff.png")
