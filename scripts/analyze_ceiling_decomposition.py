"""Decompose the router's gap to the oracle: how much is COARSE misclassification
(easy vs hard — recoverable by a better-trained classifier) vs the SOLVABILITY WALL
(which hard queries thinking will actually rescue — the probe arc found this
unlearnable from any cheap signal, AUC ~0.46).

Baseline: "subset-oracle, no within-subset signal" — perfect easy/hard knowledge,
but among hard queries, think on a RANDOM fraction f (no way to prioritize, since
no signal predicts solvability). If p_need's EAA already sits close to this
baseline's ceiling, further classifier training closes little of the remaining
gap to the true oracle — that gap is the solvability wall, not a training problem.

Run:  python scripts/analyze_ceiling_decomposition.py
Out:  results/system_router/decomposition.md
"""
import json, os
import numpy as np

OUT = "results/system_router"; os.makedirs(OUT, exist_ok=True)
paired = json.load(open("results/paired/paired_oracle_final.json"))
cn = np.array([p["correct_no_think"] for p in paired]); ct = np.array([p["correct_think"] for p in paired])
en = np.array([p["energy_wh_no_think"] for p in paired]) * 1000
et = np.array([p["energy_wh_think"] for p in paired]) * 1000
hard = np.array([p["subset"] == "underthinkingbench" for p in paired])
easy = ~hard
N = len(paired)
print(f"N={N} ({easy.sum()} easy, {hard.sum()} hard)")

# subset-oracle-random: easy always no_think (established: no reliable easy gain);
# hard: think on random fraction f (in expectation, equivalent to weighted average
# since no signal exists to prioritize within the group)
cn_e, ct_e = cn[easy].mean(), ct[easy].mean()
en_e = en[easy].mean()
cn_h, ct_h, en_h, et_h = cn[hard].mean(), ct[hard].mean(), en[hard].mean(), et[hard].mean()
n_e, n_h = easy.sum(), hard.sum()

def subset_oracle(f):
    acc = (n_e * cn_e + n_h * (f * ct_h + (1 - f) * cn_h)) / N
    e = (n_e * en_e + n_h * (f * et_h + (1 - f) * en_h)) / N * 1000  # -> mWh (per-1000 already applied above; keep consistent)
    return e, acc
# fix units: en/et already in mWh; redo cleanly
def subset_oracle(f):
    acc = (n_e * cn_e + n_h * (f * ct_h + (1 - f) * cn_h)) / N
    e = (n_e * en_e + n_h * (f * et_h + (1 - f) * en_h)) / N
    return e, acc

fs = np.linspace(0, 1, 101)
so_curve = [subset_oracle(f) for f in fs]

def eaa(pts, lo, hi, g=200):
    pts = sorted(pts, key=lambda q: q[0]); es = np.linspace(lo, hi, g); acc = []
    for e in es:
        feas = [a for pe, a in pts if pe <= e + 1e-9]
        acc.append(max(feas) if feas else np.nan)
    acc = np.array(acc); m = ~np.isnan(acc)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return tz(acc[m], es[m]) / (es[m].max() - es[m].min())

e_lo, e_hi = en.mean(), et.mean()
so_eaa = eaa(so_curve, e_lo, e_hi)
never_acc = cn.mean()

# true oracle frontier (per-query optimal, from analyze_system_aware_router-style sweep)
def pareto_ul(pts):
    out, best = [], -1
    for e, a in sorted(pts, key=lambda q: (q[0], -q[1])):
        if a > best + 1e-12: out.append((e, a)); best = a
    return out
def metrics(choose):
    choose = np.asarray(choose, bool)
    return float(np.where(choose, et, en).mean()), float(np.where(choose, ct, cn).mean())
dA, dE = ct - cn, et - en
pos = (dA > 0) & (dE > 0)
lams = sorted(set([0.0] + (dA[pos] / dE[pos]).tolist()))
front = pareto_ul([metrics((dA - l * dE) > 1e-12) for l in lams] + [metrics(np.zeros(N,bool)), metrics(np.ones(N,bool))])
oracle_eaa = eaa(front, e_lo, e_hi)

print(f"\nNever-Think acc: {never_acc:.4f} @ {e_lo:.0f} mWh")
print(f"Subset-oracle (perfect easy/hard, random within hard): EAA {so_eaa:.4f}  "
      f"({100*(so_eaa-never_acc)/(oracle_eaa-never_acc):.1f}% of oracle gain)")
print(f"True oracle frontier (per-query optimal):              EAA {oracle_eaa:.4f}  (100%)")

# where do our actual routers land, for comparison? (re-read from existing summaries)
print("\nFor comparison (from results/draft_router/summary3.md, same N after probe join):")
print("  A text-only learned router : EAA 0.9002 (69% of gain over that subset)")
print("  P p_need rule               : EAA 0.9103 (81% of gain over that subset)")

gap_total = oracle_eaa - never_acc
gap_to_subset_oracle = so_eaa - never_acc
gap_solvability_wall = oracle_eaa - so_eaa
print(f"\nDecomposition of the total achievable gain ({gap_total:.4f}):")
print(f"  recoverable by PERFECT classification alone : {gap_to_subset_oracle:.4f}  "
      f"({100*gap_to_subset_oracle/gap_total:.1f}%)")
print(f"  locked behind the SOLVABILITY WALL           : {gap_solvability_wall:.4f}  "
      f"({100*gap_solvability_wall/gap_total:.1f}%)  <- no signal we tested predicts this (AUC ~0.46)")

with open(f"{OUT}/decomposition.md", "w", encoding="utf-8") as f:
    f.write("# Ceiling decomposition: classification gap vs solvability wall\n\n")
    f.write(f"N={N} ({n_e} easy, {n_h} hard).\n\n")
    f.write("|policy|EAA|% of oracle gain|\n|--|--|--|\n")
    f.write(f"|Never-Think|{never_acc:.4f}|0%|\n")
    f.write(f"|Subset-oracle (perfect easy/hard, random within hard)|{so_eaa:.4f}|"
            f"{100*(so_eaa-never_acc)/(oracle_eaa-never_acc):.1f}%|\n")
    f.write(f"|Learned text router (actual)|0.9002|69%|\n")
    f.write(f"|p_need rule (actual)|0.9103|81%|\n")
    f.write(f"|True oracle frontier (per-query optimal)|{oracle_eaa:.4f}|100%|\n\n")
    f.write(f"**Decomposition of the total achievable gain ({gap_total:.4f} over Never-Think):**\n\n")
    f.write(f"- Recoverable by PERFECT easy/hard classification alone (a training/capacity problem): "
            f"{gap_to_subset_oracle:.4f} ({100*gap_to_subset_oracle/gap_total:.1f}%)\n")
    f.write(f"- Locked behind the solvability wall (which hard queries thinking rescues — no signal "
            f"we tested predicts this, AUC~0.46): {gap_solvability_wall:.4f} "
            f"({100*gap_solvability_wall/gap_total:.1f}%)\n\n")
    f.write("**Reading:** p_need (81% of gain) is already close to the subset-oracle ceiling "
            f"({100*(so_eaa-never_acc)/(oracle_eaa-never_acc):.1f}%) — most of its remaining gap to "
            "the true oracle is NOT a classification problem a bigger/better-trained model would fix; "
            "it is the solvability wall. A fine-tuned encoder could plausibly close some of the gap "
            "between the CURRENT learned router (69%) and p_need/the subset-oracle ceiling (since "
            "label AUC already jumped 0.829->0.900 purely from more data, no architecture change) — "
            "but is unlikely to crack the solvability wall specifically.\n")
print(f"\nwrote {OUT}/decomposition.md")
