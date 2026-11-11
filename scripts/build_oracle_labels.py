"""Build energy-aware router labels from the Claude-judged paired runs.

The per-(query,mode) correctness came from Claude judging each answer against gold
(programmatic scoring fails on OTB's prose/exact formats — see scoring-findings).
This script joins those judgments with the measured energy in paired_scored.jsonl
and applies the ENERGY-AWARE oracle rule.

Energy-aware rule (differs from a token/accuracy heuristic):
  - think strictly helps (ct>0.5, cn<=0.5)  -> think
  - no-think helps                          -> no_think
  - both correct                            -> no_think   (same accuracy, 10x cheaper)
  - both WRONG                              -> no_think   (don't pay to think when it
                                                           won't help — this is the fix;
                                                           §4.3's "both wrong -> think" is
                                                           energy-wasteful)

Run:  python scripts/build_oracle_labels.py
Out:  results/paired/paired_oracle_final.json
"""
import json, os, statistics as st
from collections import defaultdict, Counter

# --- Claude judgments: qid -> (no_think_correct/3, think_correct/3) ---
OVER = {"over_000":(3,3),"over_001":(3,3),"over_002":(3,3),"over_003":(3,3),"over_004":(3,3),
"over_005":(3,3),"over_006":(3,3),"over_007":(3,3),"over_008":(3,3),"over_009":(3,2),
"over_010":(3,3),"over_011":(3,3),"over_012":(3,3),"over_013":(1,0),"over_014":(3,3),
"over_015":(3,3),"over_016":(0,0),"over_017":(3,3),"over_018":(3,3),"over_019":(3,3),
"over_020":(3,3),"over_021":(3,3),"over_022":(3,3),"over_023":(3,3),"over_024":(3,3),
"over_025":(3,3),"over_026":(3,3),"over_027":(3,3),"over_028":(3,3),"over_029":(3,3),
"over_030":(3,3),"over_031":(3,3),"over_032":(3,3),"over_033":(3,3),"over_034":(3,3),
"over_035":(3,3),"over_036":(3,3),"over_037":(3,3),"over_038":(3,2),"over_039":(3,3)}
UNDER = {"under_001":(0,0),"under_003":(2,3),"under_004":(1,3),"under_005":(3,3),"under_006":(2,3),
"under_007":(0,0),"under_008":(2,0),"under_009":(1,2),"under_010":(0,1),"under_012":(0,0),
"under_013":(2,3),"under_014":(0,0),"under_015":(0,0),"under_016":(0,0),"under_017":(2,3),
"under_018":(0,0),"under_019":(3,3),"under_020":(0,0),"under_021":(0,2),"under_022":(0,0),
"under_023":(2,2),"under_025":(3,1),"under_026":(3,3),"under_027":(0,2),"under_028":(3,3),
"under_029":(3,3),"under_030":(0,0),"under_031":(0,1),"under_032":(0,0),"under_034":(0,2),
"under_035":(0,3),"under_036":(1,2),"under_037":(0,0),"under_038":(0,0),"under_039":(0,0)}
J = {**OVER, **UNDER}

rows = [json.loads(l) for l in open("results/paired/paired_scored.jsonl", encoding="utf-8") if l.strip()]
energy = defaultdict(lambda: defaultdict(list))
for r in rows:
    energy[r["qid"]][r["mode"]].append(r["energy_wh"])
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def oracle(cn, ct):
    if ct > 0.5 and cn <= 0.5:  return "think"        # thinking strictly helps
    if cn > 0.5 and ct <= 0.5:  return "no_think"      # no-think strictly helps
    if ct > 0.5 and cn > 0.5:   return "no_think"      # both right -> cheaper
    return "no_think"                                   # both wrong -> don't waste energy


paired = []
for q in sorted(J):
    cn, ct = J[q][0] / 3, J[q][1] / 3
    en, et = mean(energy[q]["no_think"]), mean(energy[q]["think"])
    paired.append({
        "qid": q,
        "subset": "overthinkingbench" if q.startswith("over") else "underthinkingbench",
        "route_label": "no_think" if q.startswith("over") else "think",
        "correct_no_think": round(cn, 3), "correct_think": round(ct, 3),
        "energy_wh_no_think": round(en, 4), "energy_wh_think": round(et, 4),
        "oracle": oracle(cn, ct),
    })

os.makedirs("results/paired", exist_ok=True)
json.dump(paired, open("results/paired/paired_oracle_final.json", "w"), indent=1)

# --- report ---
o_acc = mean([p["correct_think"] if p["oracle"] == "think" else p["correct_no_think"] for p in paired])
o_en = mean([p["energy_wh_think"] if p["oracle"] == "think" else p["energy_wh_no_think"] for p in paired]) * 1000
print(f"wrote results/paired/paired_oracle_final.json  (N={len(paired)})")
print("oracle label distribution:", dict(Counter(p["oracle"] for p in paired)))
print(f"Oracle now: acc={o_acc:.3f}  energy={o_en:.1f} mWh/q "
      f"(vs Always-Think 0.711 / 5726 mWh, Never-Think 0.658 / 585 mWh)")
