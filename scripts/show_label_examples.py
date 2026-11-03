"""Print real examples: what "think" looks like in OTB vs in Mohseni, with the OTB router's score.

Shows (a) OTB's think-half (Reasoning-Gym puzzles) = what the router learned "think" means,
(b) Mohseni think examples per source, (c) the killer illustration: NEAR-IDENTICAL Mohseni queries
where one is think and one is no_think -- why no text router can separate them.
Free/local.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge import JudgeCache
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

def clip(s, n=230):
    s = " ".join(str(s).split())
    return s[:n] + ("…" if len(s) > n else "")

# ---- OTB
lab = json.load(open("results/paired/paired_oracle_final.json", encoding="utf-8"))
otb_q = [e["question"] for e in lab]
otb_y = np.array([1 if e["oracle"] == "think" else 0 for e in lab])
otb_sub = np.array([e["subset"] for e in lab])

# ---- Mohseni
CD = "results/crossdataset"
rows = [json.loads(l) for l in open(f"{CD}/gen_test_mohseni.jsonl", encoding="utf-8") if l.strip()]
byq = defaultdict(dict)
for r in rows: byq[r["qid"]][r["mode"]] = r
cache = JudgeCache(f"{CD}/judge_cache_mohseni.jsonl")
M = []
for q, mm in byq.items():
    if "think" not in mm or "no_think" not in mm: continue
    nt, th = mm["no_think"], mm["think"]
    v = cache.get(nt["answer_text"], th["answer_text"])
    if v == "__miss__": continue
    cn, ct = int(v[0]), int(v[1])
    M.append(dict(q=nt["question"], src=nt.get("source") or "?", cn=cn, ct=ct,
                  y=1 if (ct and not cn) else 0, his=nt.get("his_label")))

enc = SentenceTransformer("all-MiniLM-L6-v2")
Xo = enc.encode(otb_q, batch_size=64, normalize_embeddings=True).astype(np.float32)
router = LogisticRegression(max_iter=2000).fit(Xo, otb_y)
Xm = enc.encode([m["q"] for m in M], batch_size=64, normalize_embeddings=True).astype(np.float32)
pm = router.predict_proba(Xm)[:, 1]
for m, p in zip(M, pm): m["p"] = p

print("=" * 100)
print('WHAT "THINK" LOOKS LIKE IN OTB  (the router\'s training data) — under-half = Reasoning-Gym puzzles')
print("=" * 100)
un = [i for i in range(len(otb_q)) if otb_sub[i] == "underthinkingbench" and otb_y[i] == 1]
for i in un[:4]:
    print(f"\n  [THINK] {clip(otb_q[i], 260)}")
print("\n" + "-" * 100)
print('WHAT "NO-THINK" LOOKS LIKE IN OTB — over-half (easy, 72 domains incl. easy math)')
print("-" * 100)
ov = [i for i in range(len(otb_q)) if otb_sub[i] == "overthinkingbench" and otb_y[i] == 0]
for i in ov[:4]:
    print(f"\n  [NO-THINK] {clip(otb_q[i], 200)}")

print("\n\n" + "=" * 100)
print('WHAT "THINK" LOOKS LIKE IN MOHSENI — nothing like a Reasoning-Gym puzzle')
print("=" * 100)
for src in ["Big-Math-RL", "Nectar", "WildChat-filtered"]:
    ex = [m for m in M if m["src"] == src and m["y"] == 1]
    print(f"\n--- {src}  (our label = THINK: no-think was WRONG, thinking was RIGHT) ---")
    for m in ex[:3]:
        print(f"\n  [THINK, router says p={m['p']:.2f}] {clip(m['q'], 240)}")

print("\n\n" + "=" * 100)
print("THE CORE PROBLEM: near-identical queries, OPPOSITE labels, nearly the same router score")
print("=" * 100)
for src in ["Big-Math-RL", "Nectar", "WildChat-filtered"]:
    sub = [m for m in M if m["src"] == src]
    th = sorted([m for m in sub if m["y"] == 1], key=lambda m: len(m["q"]))
    nt = sorted([m for m in sub if m["y"] == 0], key=lambda m: len(m["q"]))
    print(f"\n--- {src} ---")
    for k in range(2):
        if k < len(th) and k < len(nt):
            a, b = th[k], nt[k]
            print(f"\n  THINK    (p={a['p']:.2f}): {clip(a['q'], 200)}")
            print(f"  NO-THINK (p={b['p']:.2f}): {clip(b['q'], 200)}")

print("\n\n" + "=" * 100)
print("SCORE DISTRIBUTIONS — the router cannot separate them")
print("=" * 100)
print(f"  {'source':22} {'think n':>8} {'mean p(think)':>14} | {'no-think n':>10} {'mean p':>8}")
for src in ["Big-Math-RL", "Nectar", "WildChat-filtered", "AIME-1983-2024"]:
    t = [m["p"] for m in M if m["src"] == src and m["y"] == 1]
    n = [m["p"] for m in M if m["src"] == src and m["y"] == 0]
    if t and n:
        print(f"  {src:22} {len(t):>8} {np.mean(t):>14.3f} | {len(n):>10} {np.mean(n):>8.3f}")
print(f"\n  For reference, OTB: think mean p = {router.predict_proba(Xo)[:,1][otb_y==1].mean():.3f} | "
      f"no-think mean p = {router.predict_proba(Xo)[:,1][otb_y==0].mean():.3f}  <- well separated")
