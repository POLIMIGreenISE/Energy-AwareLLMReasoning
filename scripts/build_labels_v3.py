"""Build energy-aware oracle labels from gen_v2.jsonl with the CERTIFIED judge pipeline.

Pipeline (identical to the one cross-validated against hand labels, kappa 0.824,
results/audit/judge_crossval.md): programmatic fast-path resolves the easy rows;
contested rows go to Gemini 2.5-flash with thinking disabled and a 3072-token cap;
verdicts are cached in judge_cache_gemini.jsonl so re-runs and the certification
run's ~430 verdicts are free.

Needs: GEMINI_API_KEY in the environment.
Run:   python scripts/build_labels_v3.py
Out:   results/paired/paired_oracle_final.json (+ prints subset anchors)
"""
import json, os, re, sys, statistics as st
from collections import defaultdict, Counter

# Certified judge settings — set BEFORE importing src.judge (module reads env at import).
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("GEMINI_THINKING_BUDGET", "0")   # flash's own thinking eats the reply budget
os.environ.setdefault("GEMINI_MAX_TOKENS", "3072")     # room for fraction-equivalence arithmetic
os.environ.setdefault("JUDGE_MIN_INTERVAL", "0.5")     # Tier-1 pacing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scoring
from src.judge import judge_answer, JudgeCache

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set — the certified pipeline needs it.")

OUTDIR = "results/paired"
GEN_JSONL = f"{OUTDIR}/gen_v2.jsonl"
LABELS_JSON = f"{OUTDIR}/paired_oracle_final.json"
CACHE = JudgeCache(f"{OUTDIR}/judge_cache_gemini.jsonl")

# ---------------------------------------------------------------- fast-path (certified replica)
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
    """True/False/None(unscorable)/'JUDGE' — identical to the certified labeling run."""
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

# ---------------------------------------------------------------- score all generations
rows = [json.loads(l) for l in open(GEN_JSONL, encoding="utf-8") if l.strip()]
print(f"generations: {len(rows)} | unique qids: {len({r['qid'] for r in rows})}")
n_fast = n_gem = n_cache = n_fail = 0
for r in rows:
    v = fastpath(r)
    if v == "JUDGE":
        hit = CACHE.get(r["gold"], r["answer_text"])
        if hit != "__miss__":
            r["correct"] = hit; n_cache += 1; continue
        g = judge_answer(r["question"], r["gold"], r["answer_text"], provider="gemini")
        CACHE.put(r["gold"], r["answer_text"], g)
        r["correct"] = g; n_gem += 1
        if g is None: n_fail += 1
        if n_gem % 100 == 0:
            print(f"  gemini {n_gem} (cache {n_cache}, fastpath {n_fast}, failed {n_fail})")
    else:
        r["correct"] = v; n_fast += 1
print(f"scored: fastpath {n_fast} | gemini fresh {n_gem} | cache {n_cache} | failed {n_fail}")

# ---------------------------------------------------------------- energy-aware oracle labels
byq = defaultdict(lambda: defaultdict(list)); qmeta = {}
for r in rows:
    byq[r["qid"]][r["mode"]].append(r); qmeta[r["qid"]] = r

paired = []
for qid, modes in byq.items():
    rec = {}
    for mode in ["no_think", "think"]:
        rs = [x for x in modes.get(mode, []) if x.get("correct") is not None]
        if not rs: rec = None; break
        rec[mode] = {"acc": st.mean(1.0 if x["correct"] else 0.0 for x in rs),
                     "wh": st.mean(x["energy_b1eq_wh"] for x in rs)}
    if rec is None: continue
    cn, ct = rec["no_think"]["acc"], rec["think"]["acc"]
    m = qmeta[qid]
    paired.append({
        "qid": qid, "subset": m["subset"], "route_label": m.get("route_label"),
        "question": m["question"],
        "correct_no_think": round(cn, 3), "correct_think": round(ct, 3),
        "energy_wh_no_think": round(rec["no_think"]["wh"], 4),
        "energy_wh_think": round(rec["think"]["wh"], 4),
        "oracle": "think" if (ct > 0.5 and cn <= 0.5) else "no_think",
    })

json.dump(paired, open(LABELS_JSON, "w"), indent=1)
print(f"\nlabeled queries: {len(paired)} | oracle: {dict(Counter(p['oracle'] for p in paired))}")
print("anchors (certified 248-q run: over 0.900/0.927, under 0.378/0.796):")
for ss in ["overthinkingbench", "underthinkingbench"]:
    sub = [p for p in paired if p["subset"] == ss]
    if sub:
        print(f"  {ss:20} n={len(sub):4}  "
              f"no_think={st.mean(p['correct_no_think'] for p in sub):.3f}  "
              f"think={st.mean(p['correct_think'] for p in sub):.3f}")
unsc = sum(1 for r in rows if r.get("correct") is None)
print(f"unscorable generations: {unsc}/{len(rows)}")
print(f"wrote {LABELS_JSON}")
