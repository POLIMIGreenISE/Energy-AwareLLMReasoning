"""LLM-as-judge for OTB answer verification (mirrors the OTB paper's method).

Provider chain (first available key wins, override with provider=...):
  1. "gemini"    — Google AI Studio free tier   (GEMINI_API_KEY)
  2. "groq"      — Groq free tier, Llama 3.3    (GROQ_API_KEY)
  3. "anthropic" — Claude Sonnet, paid API      (ANTHROPIC_API_KEY)

Keys are read from env vars or Colab secrets (same names). The judge prompt is
OTB's Figure-8 verification prompt: question + gold + student answer ->
"Final Decision: Yes/No". Returns True/False, or None on hard failure.
"""
from __future__ import annotations
import os, time, json

JUDGE_PROMPT = """### Question: {question}
### Ground Truth Answer: {gold}
### Student Answer: {answer}

For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output "Final Decision: Yes". If the student's answer is incorrect, output "Final Decision: No"."""


def _get_key(name: str):
    v = os.environ.get(name)
    if v:
        return v
    try:  # Colab secrets
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def available_provider():
    if _get_key("GEMINI_API_KEY"):
        return "gemini"
    if _get_key("GROQ_API_KEY"):
        return "groq"
    if _get_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def parse_decision(text: str):
    t = (text or "").lower()
    if "final decision: yes" in t:
        return True
    if "final decision: no" in t:
        return False
    # lenient fallback: last yes/no token
    for tok in reversed(t.replace(".", " ").split()):
        if tok == "yes":
            return True
        if tok == "no":
            return False
    return None


# ------------------------------------------------------------------ backends
def _call_gemini(prompt: str) -> str:
    import requests
    key = _get_key("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    gen_cfg = {"maxOutputTokens": int(os.environ.get("GEMINI_MAX_TOKENS", 512)),
               "temperature": 0}
    # Gemini 2.5+ models think by default; the thinking can eat the whole token
    # budget and truncate the reply before "Final Decision:". GEMINI_THINKING_BUDGET=0
    # disables thinking (a yes/no equivalence check doesn't need it).
    if os.environ.get("GEMINI_THINKING_BUDGET") is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": int(os.environ["GEMINI_THINKING_BUDGET"])}
    r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}],
                                 "generationConfig": gen_cfg},
                      timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str) -> str:
    import requests
    key = _get_key("GROQ_API_KEY")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 64, "temperature": 0},
                      timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=16,
        thinking={"type": "disabled"},   # yes/no check needs no reasoning spend
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


_BACKENDS = {"gemini": _call_gemini, "groq": _call_groq, "anthropic": _call_anthropic}


# Proactive throttling: free tiers are RPM-limited (Gemini free ~10 RPM), so
# space the calls out instead of slamming into 429s. Paid tiers can override via
# env JUDGE_MIN_INTERVAL (seconds between calls, e.g. 0.5 for Gemini Tier 1).
_MIN_INTERVAL = {"gemini": 6.5, "groq": 2.2, "anthropic": 0.0}
if os.environ.get("JUDGE_MIN_INTERVAL"):
    _MIN_INTERVAL = {k: float(os.environ["JUDGE_MIN_INTERVAL"]) for k in _MIN_INTERVAL}
_last_call = {}


def _throttle(provider: str):
    wait = _MIN_INTERVAL.get(provider, 0) - (time.time() - _last_call.get(provider, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call[provider] = time.time()


def _retry_after_seconds(exc) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return None


def judge_answer(question: str, gold: str, answer: str,
                 provider: str | None = None, retries: int = 6):
    """True/False equivalence verdict via LLM judge; None if judging failed."""
    provider = provider or available_provider()
    if provider is None:
        raise RuntimeError("No judge API key found (GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY).")
    call = _BACKENDS[provider]
    prompt = JUDGE_PROMPT.format(question=str(question)[:4000],
                                 gold=str(gold)[:1500], answer=str(answer)[:2500])
    delay = 15.0
    for attempt in range(retries):
        _throttle(provider)
        try:
            return parse_decision(call(prompt))
        except Exception as e:
            msg = str(e)
            if attempt == retries - 1:
                print(f"[judge] giving up ({provider}): {msg[:150]}")
                return None
            # Rate limit / transient error: honor Retry-After if given, else back off.
            wait = _retry_after_seconds(e) or delay
            if "429" in msg:
                print(f"[judge] rate-limited, waiting {wait:.0f}s "
                      f"(attempt {attempt+1}/{retries})")
            time.sleep(wait)
            delay = min(delay * 2, 120)
    return None


# Reference-free grading: for datasets WITHOUT gold answers (e.g. Mohseni's
# reasoning-router-data-v2), a strong judge determines the correct/ideal answer
# itself, then grades BOTH candidate answers. A=no_think, B=think (fixed order;
# minor order-bias, could swap-and-average later). Maps directly onto our
# energy-aware oracle rule (think iff B correct AND A wrong).
GRADE_BOTH_PROMPT = """### Question: {question}
### Answer A: {ans_a}
### Answer B: {ans_b}

First work out the correct or ideal answer to the question yourself. Then judge each candidate:
- For factual or math questions: is it essentially correct?
- For open-ended questions: is it adequately accurate and helpful?
Respond in EXACTLY this format, nothing else:
A: Yes or No
B: Yes or No"""


def parse_grade_both(text: str):
    """(a_correct, b_correct) as bools, or None on parse failure."""
    t = (text or "").lower()
    import re as _re
    ma = _re.search(r"a:\s*(yes|no)", t)
    mb = _re.search(r"b:\s*(yes|no)", t)
    if not (ma and mb):
        return None
    return (ma.group(1) == "yes", mb.group(1) == "yes")


def judge_grade_both(question: str, ans_no_think: str, ans_think: str,
                     provider: str | None = None, retries: int = 6):
    """Reference-free: judge grades both answers. Returns (nt_correct, th_correct)
    as bools, or None on hard failure. A=no_think, B=think."""
    provider = provider or available_provider()
    if provider is None:
        raise RuntimeError("No judge API key found.")
    call = _BACKENDS[provider]
    prompt = GRADE_BOTH_PROMPT.format(question=str(question)[:4000],
                                      ans_a=str(ans_no_think)[:2500],
                                      ans_b=str(ans_think)[:2500])
    delay = 15.0
    for attempt in range(retries):
        _throttle(provider)
        try:
            return parse_grade_both(call(prompt))
        except Exception as e:
            msg = str(e)
            if attempt == retries - 1:
                print(f"[grade_both] giving up ({provider}): {msg[:150]}")
                return None
            wait = _retry_after_seconds(e) or delay
            if "429" in msg:
                print(f"[grade_both] rate-limited, waiting {wait:.0f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
            delay = min(delay * 2, 120)
    return None


class JudgeCache:
    """Disk-backed dedup cache: identical (gold, answer) pairs are judged once."""

    def __init__(self, path: str):
        self.path = path
        self.cache = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    if d["v"] is not None:      # never resurrect failed verdicts
                        self.cache[d["k"]] = d["v"]
                except Exception:
                    pass

    @staticmethod
    def key(gold: str, answer: str) -> str:
        norm = lambda s: " ".join(str(s).lower().split())[:800]
        return norm(gold) + "\x1f" + norm(answer)

    def get(self, gold, answer):
        return self.cache.get(self.key(gold, answer), "__miss__")

    def put(self, gold, answer, verdict):
        if verdict is None:                      # don't persist failures
            return
        k = self.key(gold, answer)
        self.cache[k] = verdict
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": verdict}) + "\n")
