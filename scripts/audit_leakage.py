"""Tier-1 validity check: does reasoning LEAK into our no_think outputs?

Motivated by "Demystifying Hybrid Thinking" (2510.12680): hybrid LLMs achieve only
partial mode separation — reasoning behaviors leak into no-think mode. If our
no_think generations secretly reason, their energy is inflated and the measured
think/no-think gap is understated. This audit quantifies leakage and its energy
impact on gen_v2.jsonl (no GPU).

Markers (case-insensitive), tiered:
  strong: unambiguous deliberation ("wait,", "hmm", "let me think/check/calculate",
          "step 1"/"### step", "step-by-step" as *performed* structure)
  soft  : common in helpful prose too ("let's", "first,", "we need to") — reported
          separately, NOT counted as leakage on their own.

Run:  python scripts/audit_leakage.py
Out:  results/audit/{leakage.md, leakage_lengths.png}
"""
import json, os, re, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open("results/paired/gen_v2.jsonl", encoding="utf-8") if l.strip()]
nt = [r for r in rows if r["mode"] == "no_think"]
th = [r for r in rows if r["mode"] == "think"]

STRONG = [r"\bwait\b", r"\bhmm\b", r"\blet me (think|check|verify|calculate|compute|count|re-?read)\b",
          r"(?m)^#{0,4}\s*step\s*1\b", r"###\s*step", r"\bdouble-?check\b", r"\bre-?examine\b"]
SOFT = [r"\blet'?s\b", r"(?m)^first,", r"\bwe need to\b", r"\bstep by step\b"]
THINK_TAG = re.compile(r"</?think>", re.I)

def flags(text):
    t = text.lower()
    return (any(re.search(p, t) for p in STRONG),
            any(re.search(p, t) for p in SOFT),
            bool(THINK_TAG.search(text)))

for r in nt:
    s, so, tag = flags(r["answer_text"])
    r["leak_strong"], r["leak_soft"], r["think_tag"] = s, so, tag

def frac(xs): return sum(xs) / len(xs) if xs else float("nan")

print(f"no_think generations audited: {len(nt)}")
print(f"  contains <think> tag        : {frac([r['think_tag'] for r in nt])*100:5.1f}%")
print(f"  strong deliberation markers : {frac([r['leak_strong'] for r in nt])*100:5.1f}%")
print(f"  soft markers (prose-compat) : {frac([r['leak_soft'] for r in nt])*100:5.1f}%")
print("\nby subset:")
stats = {}
for ss in ["overthinkingbench", "underthinkingbench"]:
    g = [r for r in nt if r["subset"] == ss]
    strong = frac([r["leak_strong"] for r in g])
    toks_leak = [r["completion_tokens"] for r in g if r["leak_strong"]]
    toks_clean = [r["completion_tokens"] for r in g if not r["leak_strong"]]
    stats[ss] = (g, strong, toks_leak, toks_clean)
    print(f"  {ss:20} n={len(g):3}  strong={strong*100:4.1f}%  "
          f"tokens leaked/clean = {st.mean(toks_leak) if toks_leak else 0:.0f} / "
          f"{st.mean(toks_clean) if toks_clean else 0:.0f}")

# think-mode baseline for the same markers (sanity: they SHOULD fire there)
th_strong = frac([flags(r["answer_text"])[0] for r in th])
print(f"\n(sanity) strong markers in think-mode FINAL answers: {th_strong*100:.1f}%")

# does leakage pay? accuracy of leaked vs clean no_think rows (needs correctness — approximate
# with per-query correct_no_think from labels joined on qid)
lbl = {p["qid"]: p for p in json.load(open("results/paired/paired_oracle_final.json"))}
acc_leak = [lbl[r["qid"]]["correct_no_think"] for r in nt if r["leak_strong"] and r["qid"] in lbl]
acc_clean = [lbl[r["qid"]]["correct_no_think"] for r in nt if not r["leak_strong"] and r["qid"] in lbl]
print(f"\nper-query no_think accuracy: leaked {st.mean(acc_leak):.3f} (n={len(acc_leak)}) "
      f"vs clean {st.mean(acc_clean):.3f} (n={len(acc_clean)})")

# energy inflation: counterfactual — trim leaked rows to their subset's clean median length
J = 3.7 / 3600  # Wh per token
e_actual = st.mean(r["completion_tokens"] for r in nt) * J * 1000
cf = []
for r in nt:
    if r["leak_strong"]:
        _, _, _, toks_clean = stats[r["subset"]]
        cf.append(st.median(toks_clean))
    else:
        cf.append(r["completion_tokens"])
e_cf = st.mean(cf) * J * 1000
think_e = st.mean(r["completion_tokens"] for r in th) * J * 1000
print(f"\nenergy impact (b1eq): no_think actual {e_actual:.0f} mWh vs leakage-trimmed {e_cf:.0f} mWh "
      f"({(e_actual-e_cf)/e_actual*100:.1f}% inflation)")
print(f"think/no_think ratio: actual {think_e/e_actual:.1f}x -> without leakage {think_e/e_cf:.1f}x")

# length distributions
fig, ax = plt.subplots(figsize=(8, 4.6))
bins = np.geomspace(5, 2100, 40)
for ss, c in [("overthinkingbench", "tab:green"), ("underthinkingbench", "tab:red")]:
    g, *_ = stats[ss]
    ax.hist([r["completion_tokens"] for r in g], bins=bins, alpha=.55, color=c,
            label=f"no_think {ss.replace('bench','')}")
ax.set_xscale("log"); ax.set_xlabel("no_think completion tokens (log)")
ax.set_ylabel("generations"); ax.legend()
ax.set_title("no_think output lengths — long tail = candidate leakage")
plt.tight_layout(); plt.savefig(f"{OUT}/leakage_lengths.png", dpi=120)

with open(f"{OUT}/leakage.md", "w", encoding="utf-8") as f:
    f.write("# no_think reasoning-leakage audit\n\n")
    f.write(f"- generations: {len(nt)}; `<think>` tags: {frac([r['think_tag'] for r in nt])*100:.1f}%; "
            f"strong markers: {frac([r['leak_strong'] for r in nt])*100:.1f}%; "
            f"soft: {frac([r['leak_soft'] for r in nt])*100:.1f}%\n")
    for ss in stats:
        g, strong, tl, tc = stats[ss]
        f.write(f"- {ss}: strong {strong*100:.1f}%, tokens leaked/clean "
                f"{st.mean(tl) if tl else 0:.0f}/{st.mean(tc) if tc else 0:.0f}\n")
    f.write(f"- think-mode final answers with same markers (sanity): {th_strong*100:.1f}%\n")
    f.write(f"- no_think acc leaked {st.mean(acc_leak):.3f} vs clean {st.mean(acc_clean):.3f}\n")
    f.write(f"- energy inflation from leakage: {(e_actual-e_cf)/e_actual*100:.1f}% "
            f"(no_think {e_actual:.0f}->{e_cf:.0f} mWh; think/no_think ratio "
            f"{think_e/e_actual:.1f}x->{think_e/e_cf:.1f}x)\n")
# top examples for qualitative inspection
ex = sorted([r for r in nt if r["leak_strong"]], key=lambda r: -r["completion_tokens"])[:5]
with open(f"{OUT}/leakage.md", "a", encoding="utf-8") as f:
    f.write("\n## Longest leaked examples (first 300 chars)\n\n")
    for r in ex:
        f.write(f"- **{r['qid']}** ({r['completion_tokens']} tok): "
                f"{' '.join(r['answer_text'][:300].split())}...\n")
print(f"\nwrote {OUT}/leakage.md, leakage_lengths.png")
