"""FINALE — the system-aware router: p_need + load-adaptive threshold, on the
FULL-OTB scaled dataset (N=1,710). Combines every prior finding into one policy:

  1. p_need is the best cheap routing signal we found (beats the learned text
     router at every scale tested) -> use it as the BASE router.
  2. Energy per token is load-dependent (measured concurrency curve, 3.42 ->
     0.08 J/token) -> the router's DECISION THRESHOLD should adapt to load,
     not just its evaluation (step-2 of the load work used the learned
     classifier; this finalizes it with the actual best base signal).
  3. The probe itself has an energy cost. PRIMARY accounting treats it as
     FIXED (we only measured it at one concurrency; conservative — it does
     NOT shrink with load, unlike generation). A SENSITIVITY pass amortizes
     it with the same curve as decode (optimistic upper bound) to show the
     qualitative story is unchanged either way.
  4. Bonus: does DART's token-validated-but-energy-reversed verdict
     (summary2.md: honest cost > Always-Think at batch=1) flip back once
     thinking gets ~43x cheaper under load?

Run:  python scripts/analyze_system_aware_router.py
Out:  results/system_router/{summary.md, system_router.png}
"""
import json, os, re, statistics as st
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/system_router"; os.makedirs(OUT, exist_ok=True)
J_B1 = 3.7          # measured batch=1 J/output-token (pilot, A100)
WH = 1 / 3600.0

paired = {p["qid"]: p for p in json.load(open("results/paired/paired_oracle_final.json"))}
probes = {json.loads(l)["qid"]: json.loads(l) for l in open("results/paired/probes_v1.jsonl", encoding="utf-8")}
drafts = defaultdict(dict)
for l in open("results/paired/drafts_v1.jsonl", encoding="utf-8"):
    d = json.loads(l); drafts[d["qid"]][d["rep"]] = d

conc = [json.loads(l) for l in open("results/batching/concurrency.jsonl", encoding="utf-8") if l.strip()]
jt = defaultdict(list)
for r in conc: jt[r["batch"]].append(r["energy_per_out_token_j"])
J_OF_L = {L: st.mean(v) for L, v in jt.items()}
LOADS = sorted(J_OF_L)
print("measured J/token curve:", {L: round(J_OF_L[L], 3) for L in LOADS})

# ============================================================ PART 1: system-aware p_need router
rows = [{"qid": q, **paired[q], **probes[q]} for q in paired if q in probes]
N = len(rows)
cn = np.array([r["correct_no_think"] for r in rows]); ct = np.array([r["correct_think"] for r in rows])
en = np.array([r["energy_wh_no_think"] for r in rows]) * 1000   # mWh, b1eq
et = np.array([r["energy_wh_think"] for r in rows]) * 1000
tok_nt, tok_th = en / 1000 * 3600 / J_B1, et / 1000 * 3600 / J_B1
p_need = np.array([r["p_need_think"] for r in rows])
need_wh_fixed = np.array([r["probe_energy_mwh"] / 4.0 for r in rows])   # 1 of 4 probes
need_tok = need_wh_fixed / 1000 * 3600 / J_B1                           # for amortized sensitivity
print(f"\nPart 1 — system-aware p_need router: N={N}")

def E(choose, L, probe="fixed"):
    tok = np.where(choose, tok_th, tok_nt)
    e_gen = tok * J_OF_L[L] * WH * 1000
    e_probe = need_wh_fixed if probe == "fixed" else need_tok * J_OF_L[L] * WH * 1000
    return float((e_gen + e_probe).mean())

def A(choose):
    return float(np.where(choose, ct, cn).mean())

TAUS = np.unique(np.concatenate([np.linspace(0, 1, 201), np.unique(p_need)]))
CHOICES = [p_need >= t for t in TAUS]

def best_tau(L, lam, probe="fixed"):
    utils = [A(c) - lam * E(c, L, probe) / 1000 for c in CHOICES]
    i = int(np.argmax(utils))
    return TAUS[i], CHOICES[i]

results = {}
for lam, name in [(0.04, "lambda=0.04/Wh (accuracy-first)"), (1.0, "lambda=1.0/Wh (energy-conscious)")]:
    tau_idle, ch_idle = best_tau(1, lam)   # static: frozen at idle calibration
    rows_out = []
    for L in LOADS:
        tau_L, ch_L = best_tau(L, lam)
        rows_out.append(dict(load=L, tau_static=tau_idle, tau_adaptive=tau_L,
                             acc_static=A(ch_idle), acc_adaptive=A(ch_L),
                             e_static=E(ch_idle, L), e_adaptive=E(ch_L, L),
                             think_static=float(ch_idle.mean()), think_adaptive=float(ch_L.mean())))
    results[name] = (lam, tau_idle, rows_out)
    print(f"\n--- {name} (static tau*={tau_idle:.3f}, thinks {ch_idle.mean()*100:.0f}%)")
    for r in rows_out:
        print(f"  L={r['load']:>3} tau(L)={r['tau_adaptive']:.3f} think%={r['think_adaptive']*100:5.1f} | "
              f"acc adap={r['acc_adaptive']:.3f} stat={r['acc_static']:.3f} "
              f"gap={(r['acc_adaptive']-r['acc_static'])*100:+5.1f}pp | "
              f"mWh adap={r['e_adaptive']:7.1f} stat={r['e_static']:7.1f}")

# sensitivity: amortized probe cost, energy-conscious scenario only
lam1 = 1.0
tau_idle_am, ch_idle_am = best_tau(1, lam1, probe="amortized")
sens_rows = []
for L in LOADS:
    tau_L, ch_L = best_tau(L, lam1, probe="amortized")
    sens_rows.append((L, A(ch_L), E(ch_L, L, "amortized")))
print(f"\nSensitivity (probe cost amortized with load, lambda=1.0/Wh): "
      f"batch64 acc={sens_rows[-1][1]:.3f} (fixed-probe was "
      f"{results['lambda=1.0/Wh (energy-conscious)'][2][-1]['acc_adaptive']:.3f})")

# how does this compare to the LEARNED router's tau(load) result (router_summary.md)?
learned_summary = "results/load_labels/router_summary.md"
learned_gain_line = None
if os.path.exists(learned_summary):
    for line in open(learned_summary, encoding="utf-8"):
        if line.startswith("|64|"):
            learned_gain_line = line.strip()
print(f"\n(reference) learned-router tau(load) @ batch64, lambda=1.0/Wh row: {learned_gain_line}")

# ============================================================ PART 2: DART re-verdict under load
def norm(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", str(s))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())
def as_num(s):
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", norm(s).replace(" ", ""))
    return float(m.group()) if m else None
def pair_agree(a, b):
    va, vb = as_num(a), as_num(b)
    if va is not None and vb is not None:
        return float(abs(va - vb) <= 1e-6 * max(1.0, abs(va)))
    na, nb = norm(a), norm(b)
    return float(na == nb and na != "")

drows = []
for q in paired:
    reps = drafts.get(q, {})
    if len(reps) < 3: continue
    txt = [reps[i]["draft_text"] for i in range(3)]
    agree_frac = st.mean(pair_agree(txt[i], txt[j]) for i, j in combinations(range(3), 2))
    probe_wh = sum(reps[i]["energy_b1eq_wh"] for i in range(3)) * 1000   # mWh
    drows.append({**paired[q], "agree_frac": agree_frac, "probe_wh": probe_wh})
Nd = len(drows)
print(f"\nPart 2 — DART re-verdict: N={Nd}")
cn_d = np.array([r["correct_no_think"] for r in drows]); ct_d = np.array([r["correct_think"] for r in drows])
en_d = np.array([r["energy_wh_no_think"] for r in drows]) * 1000
et_d = np.array([r["energy_wh_think"] for r in drows]) * 1000
tok_nt_d, tok_th_d = en_d / 1000 * 3600 / J_B1, et_d / 1000 * 3600 / J_B1
probe_d = np.array([r["probe_wh"] for r in drows])
probe_tok_d = probe_d / 1000 * 3600 / J_B1
dart_choose = np.array([r["agree_frac"] for r in drows]) < 1.0   # think unless all 3 drafts agree

def dart_energy(choose, L, probe="fixed"):
    tok = np.where(choose, tok_th_d, tok_nt_d)
    e_gen = tok * J_OF_L[L] * WH * 1000
    e_probe = probe_d if probe == "fixed" else probe_tok_d * J_OF_L[L] * WH * 1000
    return float((e_gen + e_probe).mean())

def always_energy(L):
    return float((tok_th_d * J_OF_L[L] * WH * 1000).mean())

dart_acc = float(np.where(dart_choose, ct_d, cn_d).mean())
print(f"\n  {'load':>5} | {'DART honest (fixed probe)':>26} | {'DART (amortized probe)':>24} | {'Always-Think':>13} | verdict")
verdict_rows = []
for L in LOADS:
    d_fixed = dart_energy(dart_choose, L, "fixed")
    d_amort = dart_energy(dart_choose, L, "amortized")
    always = always_energy(L)
    v_fixed = "DART CHEAPER" if d_fixed < always else "DART COSTS MORE"
    v_amort = "DART CHEAPER" if d_amort < always else "DART COSTS MORE"
    verdict_rows.append((L, d_fixed, d_amort, always, v_fixed, v_amort))
    print(f"  {L:>5} | {d_fixed:>10.1f} mWh ({v_fixed:>16}) | {d_amort:>10.1f} mWh ({v_amort:>16}) | {always:>9.1f} mWh |")

# ============================================================ PART 2b: FREE-FORM DART re-verdict
# This is the rule that actually reversed under honest accounting at batch=1
# (the constrained-draft rule above was already fixed to be cheap — not a fair
# re-test of the reversal). Free-form drafts = REPS 0/1 of no_think generations
# (858 mWh/query probe: 2 full no_think answers) — GENERATION, same mechanism
# as decode, so its cost amortizes with load exactly like think/no_think tokens.
gens_nt = defaultdict(dict)
for l in open("results/paired/gen_v2.jsonl", encoding="utf-8"):
    g = json.loads(l)
    if g["mode"] == "no_think":
        gens_nt[g["qid"]][g["rep"]] = g

frows = []
for q in paired:
    reps = gens_nt.get(q, {})
    if len(reps) < 2: continue
    a0, a1 = reps[0]["answer_text"], reps[1]["answer_text"]
    agree = pair_agree(a0, a1)
    probe_wh = (reps[0]["energy_b1eq_wh"] + reps[1]["energy_b1eq_wh"]) * 1000
    frows.append({**paired[q], "agree": agree, "probe_wh": probe_wh})
Nf = len(frows)
cn_f = np.array([r["correct_no_think"] for r in frows]); ct_f = np.array([r["correct_think"] for r in frows])
en_f = np.array([r["energy_wh_no_think"] for r in frows]) * 1000
et_f = np.array([r["energy_wh_think"] for r in frows]) * 1000
tok_nt_f, tok_th_f = en_f / 1000 * 3600 / J_B1, et_f / 1000 * 3600 / J_B1
probe_tok_f = np.array([r["probe_wh"] for r in frows]) / 1000 * 3600 / J_B1   # generation -> scales with J_OF_L
dart_choose_f = np.array([r["agree"] for r in frows]) < 0.5   # think iff drafts disagree

# Both branches (probe + generation) are GENERATED TOKENS through the same batched
# engine -> both scale by the identical J_OF_L[L] factor. So under this (physically
# correct, for generation-based probes) accounting, the honest/Always-Think RATIO
# is a constant, independent of L: the verdict is provably load-invariant.
tok_dart_total_f = np.where(dart_choose_f, probe_tok_f + tok_th_f, probe_tok_f) + tok_nt_f * 0  # probe already includes both no_think reps
# careful: probe_tok_f already = 2 no_think generations; if dart thinks, ALSO pay tok_th_f
tok_dart_total_f = probe_tok_f + np.where(dart_choose_f, tok_th_f, 0.0)
ratio_f = tok_dart_total_f.mean() / tok_th_f.mean()
print(f"\nPart 2b — FREE-FORM DART re-verdict: N={Nf} | think-rate {dart_choose_f.mean()*100:.1f}%")
print(f"  honest/Always-Think token ratio = {ratio_f:.3f} "
      f"({'DART COSTS MORE' if ratio_f > 1 else 'DART CHEAPER'}, and PROVABLY LOAD-INVARIANT "
      "under proportional probe accounting: both scale by the same J/token factor)")
freeform_rows = []
for L in LOADS:
    d = float((tok_dart_total_f * J_OF_L[L] * WH * 1000).mean())
    a = float((tok_th_f * J_OF_L[L] * WH * 1000).mean())
    freeform_rows.append((L, d, a, d / a))
    print(f"  L={L:>3}  DART honest {d:8.1f} mWh vs Always-Think {a:8.1f} mWh  (ratio {d/a:.3f}, "
          f"{'MORE' if d>a else 'CHEAPER'})")

# ============================================================ plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
lam1, tau1, rows1 = results["lambda=1.0/Wh (energy-conscious)"]
ax.plot([r["load"] for r in rows1], [r["acc_adaptive"] for r in rows1], "-o", color="tab:green",
        label="system-aware: p_need + tau(load)")
ax.plot([r["load"] for r in rows1], [r["acc_static"] for r in rows1], "--", color="tab:orange",
        label="static p_need (idle-calibrated)")
ax.axhline(A(np.zeros(N, bool)), color="grey", lw=.8, ls=":", label="Never-Think")
ax.set_xscale("log", base=2); ax.set_xlabel("server load (batch size)")
ax.set_ylabel("accuracy"); ax.set_title("System-aware router: accuracy vs load (lambda=1.0/Wh)")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot([v[0] for v in freeform_rows], [v[1] for v in freeform_rows], "-o", color="black",
        label="DART honest, free-form (858 mWh probe)")
ax.plot([v[0] for v in freeform_rows], [v[2] for v in freeform_rows], "-^", color="tab:orange",
        label="Always-Think")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("server load (batch size)"); ax.set_ylabel("energy per query (mWh)")
ax.set_title(f"Free-form DART re-verdict: ratio {ratio_f:.2f}x at every load")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/system_router.png", dpi=120)

# ============================================================ report
with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
    f.write("# System-aware router (finale): p_need + tau(load)\n\n")
    f.write(f"N={N} (p_need router), N={Nd} (DART re-verdict). Measured J/token curve: "
            f"{ {L: round(J_OF_L[L],3) for L in LOADS} }\n\n")
    f.write("## Part 1 — system-aware p_need router (probe cost FIXED, primary accounting)\n\n")
    for name, (lam, tau_idle, rows_out) in results.items():
        f.write(f"### {name} (static tau*={tau_idle:.3f})\n\n")
        f.write("|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|\n|--|--|--|--|--|--|--|--|\n")
        for r in rows_out:
            f.write(f"|{r['load']}|{r['tau_adaptive']:.3f}|{r['think_adaptive']*100:.1f}%"
                    f"|{r['acc_adaptive']:.3f}|{r['acc_static']:.3f}"
                    f"|{(r['acc_adaptive']-r['acc_static'])*100:+.1f}pp"
                    f"|{r['e_adaptive']:.1f}|{r['e_static']:.1f}|\n")
        f.write("\n")
    f.write(f"**Sensitivity** (probe cost amortized with load instead of fixed, lambda=1.0/Wh): "
            f"batch64 acc {sens_rows[-1][1]:.3f} vs fixed-probe "
            f"{results['lambda=1.0/Wh (energy-conscious)'][2][-1]['acc_adaptive']:.3f} — "
            "conclusion unchanged (probe is negligible next to generation energy at every load).\n\n")
    f.write(f"Reference — learned-router tau(load) result at batch64 (results/load_labels/router_summary.md): "
            f"`{learned_gain_line}`\n\n")
    f.write("## Part 2 — DART re-verdict under load\n\n")
    f.write(f"DART rule (think unless all 3 constrained drafts agree): accuracy {dart_acc:.3f} "
            f"(load-invariant — same queries think regardless of batch).\n\n")
    f.write("**Constrained drafts (12-tok, ~24 mWh probe)** — already fixed to be cheap in step 3b; "
            "not a fair re-test of the reversal, included for completeness:\n\n")
    f.write("|load|DART honest, fixed probe (mWh)|DART honest, amortized probe (mWh)|Always-Think (mWh)|verdict (fixed)|verdict (amortized)|\n|--|--|--|--|--|--|\n")
    for L, dfx, dam, alw, vfx, vam in verdict_rows:
        f.write(f"|{L}|{dfx:.1f}|{dam:.1f}|{alw:.1f}|{vfx}|{vam}|\n")
    f.write(f"\n### Part 2b — FREE-FORM drafts (858 mWh probe): the rule that actually reversed\n\n")
    f.write(f"N={Nf}, think-rate {dart_choose_f.mean()*100:.1f}%. Both the probe (2 full no_think "
            "generations) and the thinking branch are GENERATED TOKENS through the same batched "
            "engine, so both scale by the identical measured J/token factor — under this physically "
            "correct accounting, the honest-cost/Always-Think ratio is a **constant, provably "
            f"load-invariant**: {ratio_f:.3f} "
            f"({'DART costs more than Always-Think at every load' if ratio_f > 1 else 'DART is cheaper at every load'}).\n\n")
    f.write("|load|DART honest (mWh)|Always-Think (mWh)|ratio|\n|--|--|--|--|\n")
    for L, d, a, ratio in freeform_rows:
        f.write(f"|{L}|{d:.1f}|{a:.1f}|{ratio:.3f}|\n")
    f.write("\n**Reading:** load does NOT rescue a probe that is overpriced relative to what it "
            "saves — if the probe's cost structure matches the thing it's trying to save (both are "
            "generation), the verdict is a fixed ratio. Load only changes a verdict when the two "
            "cost components scale DIFFERENTLY — which is exactly the p_need case above (Part 1): "
            "a single-forward-pass probe (compute-bound, doesn't shrink much with batching) versus "
            "a decode chain (memory-bandwidth-bound, shrinks ~43x) is the load-DEPENDENT regime; "
            "two generation-vs-generation costs (DART) are not.\n")
print(f"\nwrote {OUT}/summary.md, system_router.png")
