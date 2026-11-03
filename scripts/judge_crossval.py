"""Tier-1 validity: cross-validate Claude's hand-judged labels with a Gemini judge.

Replicates the EXACT deterministic pipeline used for the final labels
(truncation rule + fast-path passes), sends only the contested rows to the
Gemini judge (same OTB verification prompt via src/judge.py), rebuilds
per-query aggregates + energy-aware oracle labels, and reports agreement:

  L1  judged-verdict sums per (qid, mode)   [reconstructed from the final labels]
  L2  per-(qid, mode) correctness aggregates
  L3  oracle think/no_think labels (+ Cohen's kappa) and RQ2 headline numbers

Needs: GEMINI_API_KEY env (Tier 1), optional GEMINI_MODEL / JUDGE_MIN_INTERVAL.
Run:  python scripts/judge_crossval.py
Out:  results/audit/judge_crossval.md, results/paired/paired_oracle_gemini.json
"""
import json, os, re, sys, statistics as st
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scoring
from src.judge import judge_answer, JudgeCache

OUT = "results/audit"; os.makedirs(OUT, exist_ok=True)
CACHE = JudgeCache("results/paired/judge_cache_gemini.jsonl")

# ---------------------------------------------------------------- fast-path (EXACT replica)
def delatex(s):
    return re.sub(r"\\[a-zA-Z]+", "", str(s))
def normsp(s):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", delatex(s).lower()).split())
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", delatex(s).lower())
STOP = set("the a an of to in for and or is are was were be been by on with as at from that this it its".split())
def contentwords(s):
    return [w for w in normsp(s).split() if w not in STOP and len(w) > 2]

def fastpath(r):
    """Returns True/False/None(unscorable)/'JUDGE' — identical to the final labeling run."""
    trunc = str(r.get("finish_reason")).lower() == "length"
    ans = r["answer_text"]
    if trunc and not scoring._strong_answer(ans, r["answer_type"]):
        return None
    if scoring.verify(ans, r["gold"], r["answer_type"]) is True:
        return True
    g = str(r["gold"]).strip()
    gn, an = normsp(g), normsp(ans)
    if not re.fullmatch(r"[\d\.\,\s\-]+", g) and len(norm(g)) >= 4 and gn and gn in an:
        return True
    if re.fullmatch(r"-?[\d\.\,]+", g):
        gval = g.replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", delatex(ans)[-150:].replace(",", ""))
        try:
            if any(abs(float(x) - float(gval)) <= 1e-6 * max(1, abs(float(gval))) for x in nums):
                return True
        except ValueError:
            pass
    if "long" in r["answer_type"]:
        cw = contentwords(g)
        if cw:
            rec = sum(1 for w in cw if w in an) / len(cw)
            if rec >= 0.45: return True
            if rec < 0.12: return False
    return "JUDGE"

# ---------------------------------------------------------------- run pipeline with Gemini
rows = [json.loads(l) for l in open("results/paired/gen_v2.jsonl", encoding="utf-8") if l.strip()]
n_fast = n_gem = n_cache = n_fail = 0
for r in rows:
    v = fastpath(r)
    r["fp"] = v
    if v == "JUDGE":
        hit = CACHE.get(r["gold"], r["answer_text"])
        if hit != "__miss__":
            r["correct"] = hit; n_cache += 1; continue
        g = judge_answer(r["question"], r["gold"], r["answer_text"], provider="gemini")
        CACHE.put(r["gold"], r["answer_text"], g)
        r["correct"] = g; n_gem += 1
        if g is None: n_fail += 1
        if n_gem % 50 == 0:
            print(f"  gemini calls {n_gem} (cache {n_cache}, fastpath {n_fast})")
    else:
        r["correct"] = v; n_fast += 1
print(f"fastpath {n_fast} | gemini {n_gem} | cache {n_cache} | failed {n_fail}")

# ---------------------------------------------------------------- rebuild labels
byq = defaultdict(lambda: defaultdict(list)); qmeta = {}
for r in rows:
    byq[r["qid"]][r["mode"]].append(r); qmeta[r["qid"]] = r
gem = {}
for qid, modes in byq.items():
    rec = {}
    for mode in ["no_think", "think"]:
        rs = [x for x in modes.get(mode, []) if x["correct"] is not None]
        if not rs: rec = None; break
        rec[mode] = st.mean(1.0 if x["correct"] else 0.0 for x in rs)
    if rec is None: continue
    cn, ct = rec["no_think"], rec["think"]
    gem[qid] = {"cn": cn, "ct": ct, "oracle": "think" if (ct > 0.5 and cn <= 0.5) else "no_think"}

mine = {p["qid"]: p for p in json.load(open("results/paired/paired_oracle_final.json"))}
common = sorted(set(gem) & set(mine))
print(f"queries comparable: {len(common)} (mine {len(mine)}, gemini {len(gem)})")

# L1: reconstructed judged-verdict sums per (qid, mode)
l1_tot = l1_agree = 0
for qid in common:
    for mode, key in [("no_think", "correct_no_think"), ("think", "correct_think")]:
        rs = [x for x in byq[qid][mode] if x["fp"] is not None]
        n_sc = len(rs)
        if n_sc == 0: continue
        fp_true = sum(1 for x in rs if x["fp"] is True)
        judged = [x for x in rs if x["fp"] == "JUDGE"]
        if not judged: continue
        my_sum = round(mine[qid][key] * n_sc) - fp_true
        gm_sum = sum(1 for x in judged if x["correct"])
        l1_tot += 1; l1_agree += int(my_sum == gm_sum)
print(f"\nL1 judged-sum agreement per (qid,mode): {l1_agree}/{l1_tot} = {l1_agree/l1_tot:.3f}")

# L2: aggregate correctness agreement
diffs = []
for qid in common:
    diffs.append(abs(gem[qid]["cn"] - mine[qid]["correct_no_think"]))
    diffs.append(abs(gem[qid]["ct"] - mine[qid]["correct_think"]))
exact = sum(1 for d in diffs if d < 1e-9) / len(diffs)
print(f"L2 aggregate correctness: exact match {exact:.3f}, mean |diff| {st.mean(diffs):.3f}")

# L3: oracle labels + kappa + headlines
a = [mine[q]["oracle"] for q in common]; b = [gem[q]["oracle"] for q in common]
agree = sum(x == y for x, y in zip(a, b)) / len(a)
pa = agree
py = (a.count("think") / len(a)) * (b.count("think") / len(b)) + \
     (a.count("no_think") / len(a)) * (b.count("no_think") / len(b))
kappa = (pa - py) / (1 - py)
print(f"L3 oracle label agreement: {agree:.3f}  (Cohen's kappa {kappa:.3f})")
print("   label counts mine:", Counter(a), " gemini:", Counter(b))
print("\nRQ2 headline numbers under each judge:")
for name, src in [("claude", lambda q, m: mine[q][f"correct_{m}"]),
                  ("gemini", lambda q, m: gem[q]["cn" if m == "no_think" else "ct"])]:
    for ss in ["overthinkingbench", "underthinkingbench"]:
        qs = [q for q in common if mine[q]["subset"] == ss]
        print(f"  {name}  {ss:20} no_think={st.mean(src(q,'no_think') for q in qs):.3f} "
              f"think={st.mean(src(q,'think') for q in qs):.3f}")

json.dump([{"qid": q, **gem[q]} for q in common],
          open("results/paired/paired_oracle_gemini.json", "w"), indent=1)
with open(f"{OUT}/judge_crossval.md", "w", encoding="utf-8") as f:
    f.write(f"# Judge cross-validation (Claude hand-judge vs {os.environ.get('GEMINI_MODEL','gemini-2.5-flash')})\n\n"
            f"- contested rows judged by Gemini: {n_gem + n_cache} (fresh {n_gem}, cache {n_cache}, failed {n_fail})\n"
            f"- L1 judged-verdict sums per (qid,mode): {l1_agree}/{l1_tot} = {l1_agree/l1_tot:.3f}\n"
            f"- L2 aggregate correctness exact-match {exact:.3f}, mean|diff| {st.mean(diffs):.3f}\n"
            f"- L3 oracle labels: agreement {agree:.3f}, kappa {kappa:.3f}; "
            f"mine {dict(Counter(a))} vs gemini {dict(Counter(b))}\n")
print(f"\nwrote {OUT}/judge_crossval.md, results/paired/paired_oracle_gemini.json")
