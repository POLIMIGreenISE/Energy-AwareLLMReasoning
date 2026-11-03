"""Tests for src/scoring.py. Run:  python tests/test_scoring.py  (or pytest)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scoring as S


def test_extract_boxed():
    assert S.extract_answer(r"work... so \boxed{42} done") == "42"
    assert S.extract_answer(r"\boxed{x+1} then \boxed{7}") == "7"  # last one


def test_extract_answer_line():
    assert S.extract_answer("blah\nAnswer: Paris\n") == "Paris"
    assert S.extract_answer("Final answer = 3.14") == "3.14"


def test_extract_mcq_and_lastline():
    assert S.extract_answer("I think the answer is C.", "multiple-choice") == "C"
    assert S.extract_answer("line1\nline2\n42") == "42"


def test_numeric():
    assert S.verify("The answer is 42", "42", "numeric") is True
    assert S.verify(r"\boxed{42.0}", "42", "numeric") is True
    assert S.verify("about 41", "42", "numeric") is False
    # false-positive guard: gold 4 must not match "1234"
    assert S.verify("1234", "4", "numeric") is False


def test_string_containment():
    assert S.verify("The ozone layer.", "Ozone Layer", "short") is True
    assert S.verify("stratosphere", "Ozone Layer", "short") is False


def test_open_ended_needs_judge():
    assert S.verify("some essay", "gold essay", "open-ended-long") is None
    assert S.verify("x", "y", "open-ended-long", judge=lambda a, g: True) is True


def test_empty_gold_unscorable():
    assert S.verify("anything", "", "short") is None


def test_score_truncated_not_wrong():
    r = S.score("...still reasoning with no final", gold="42", answer_type="numeric",
                finish_reason="length")
    assert r["truncated"] is True and r["correct"] is None and r["scorable"] is False


def test_no_think_output_not_truncated():
    # no-think outputs have no </think> tag; that must NOT be flagged truncated.
    r = S.score("Answer: 42", gold="42", answer_type="numeric",
                finish_reason="stop", has_closing_think=False)
    assert r["truncated"] is False and r["correct"] is True


def test_score_completed_correct():
    r = S.score("Answer: 42", gold="42", answer_type="numeric", finish_reason="stop")
    assert r["correct"] is True and r["truncated"] is False and r["scorable"] is True


def test_score_completed_wrong_stays_wrong():
    r = S.score("Answer: 41", gold="42", answer_type="numeric", finish_reason="stop")
    assert r["correct"] is False and r["truncated"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
        except Exception as e:
            failed += 1; print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
