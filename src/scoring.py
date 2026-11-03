"""Hardened answer extraction + verification for OptimalThinkingBench outputs.

Fixes the two pilot problems:
  1. Truncation: `finish_reason == "length"` generations were silently scored
     wrong. `score()` now separates *truncated* from *incorrect*.
  2. Crude scoring: type-aware extraction + math-verify (when available) +
     numeric tolerance + normalized string match, with an optional LLM judge
     for open-ended answers.

Pure-stdlib except the optional `math_verify` (graceful fallback if absent).
"""
from __future__ import annotations
import re
from typing import Callable, Optional

try:  # optional, big win for math/numeric equivalence (e.g. 1/2 == 0.5)
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - depends on env
    _HAS_MATH_VERIFY = False

# A verdict is True (correct), False (incorrect), or None (couldn't score).
Verdict = Optional[bool]


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
# Requires an explicit ':' or '=' so prose like "the answer is C" is NOT matched.
_ANSWER_LINE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:=]\s*(.+?)\s*(?:\n|$)")
_MCQ = re.compile(r"\b([A-J])\b")


def _strong_answer(text: str, answer_type: Optional[str] = None) -> str:
    """An *explicit* final-answer signal (boxed / 'Answer:' / MCQ letter), or ''.

    Used to tell a real answer from a truncated ramble's trailing line.
    """
    if not text:
        return ""
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = _ANSWER_LINE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    at = str(answer_type or "").lower()
    if "choice" in at or "mcq" in at:
        letters = _MCQ.findall(text)
        if letters:
            return letters[-1]
    return ""


def extract_answer(text: str, answer_type: Optional[str] = None) -> str:
    """Best-effort final answer: a strong signal if present, else the last line."""
    if not text:
        return ""
    strong = _strong_answer(text, answer_type)
    if strong:
        return strong
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# --------------------------------------------------------------------------- #
# Normalization / numeric helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _last_number(s: str) -> Optional[float]:
    nums = _NUM.findall(str(s).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _numeric_equal(pred: str, gold: str, rtol: float = 1e-3) -> Verdict:
    p, g = _last_number(pred), _last_number(gold)
    if p is None or g is None:
        return None
    return abs(p - g) <= max(1e-6, rtol * abs(g))


def _math_verify(pred: str, gold: str) -> Verdict:
    if not _HAS_MATH_VERIFY:
        return None
    try:  # pragma: no cover - depends on optional dep
        return bool(_mv_verify(_mv_parse(str(gold)), _mv_parse(str(pred))))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(
    answer_text: str,
    gold: str,
    answer_type: Optional[str] = None,
    judge: Optional[Callable[[str, str], bool]] = None,
) -> Verdict:
    """Type-aware correctness. Returns True/False, or None if unscorable.

    `judge(answer_text, gold) -> bool` is used only for open-ended/long answers.
    """
    at = str(answer_type or "").lower()
    if gold is None or str(gold).strip() == "":
        return None

    # Open-ended: only an LLM judge can score it fairly.
    if "long" in at or "open" in at:
        return bool(judge(answer_text, gold)) if judge is not None else None

    pred = extract_answer(answer_text, answer_type)
    if pred == "":
        return None

    # Math / numeric first (handles 1/2 == 0.5, 0x.. etc. when math_verify present).
    if any(k in at for k in ("numeric", "math", "number")) or at == "":
        mv = _math_verify(pred, gold)
        if mv is not None:
            return mv
        ne = _numeric_equal(pred, gold)
        if ne is not None:
            return ne

    # Fallback: normalized string match / gold-contained-in-prediction.
    p, g = _norm(pred), _norm(gold)
    if not g:
        return None
    if p == g:
        return True
    # containment only when gold is non-trivial (avoid "4" in "1234" false hits)
    if len(g) >= 2 and g in p:
        return True
    return False


def is_truncated(finish_reason: Optional[str]) -> bool:
    """vLLM/HF report 'length' when the token cap cut generation short."""
    return str(finish_reason).lower() == "length"


def score(
    answer_text: str,
    gold: str,
    answer_type: Optional[str] = None,
    finish_reason: Optional[str] = None,
    has_closing_think: bool = True,
    judge: Optional[Callable[[str, str], bool]] = None,
) -> dict:
    """Full scoring for one generation.

    Returns dict(correct, truncated, scorable, extracted). A truncated generation
    with no parseable answer is flagged `truncated=True, correct=None` instead of
    being silently marked wrong (the pilot's bug).
    """
    # Truncation is finish_reason == "length". Do NOT use "no </think> tag" as a
    # truncation signal: no-think outputs legitimately have no think block, and a
    # cut-off think generation already reports finish_reason == "length".
    truncated = is_truncated(finish_reason)
    # A cut-off generation with no *explicit* final answer can't be fairly scored.
    if truncated and not _strong_answer(answer_text, answer_type):
        return {"correct": None, "truncated": True, "scorable": False,
                "extracted": extract_answer(answer_text, answer_type)}
    verdict = verify(answer_text, gold, answer_type, judge=judge)
    return {
        "correct": verdict,
        "truncated": bool(truncated),
        "scorable": verdict is not None,
        "extracted": extract_answer(answer_text, answer_type),
    }
