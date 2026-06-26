"""Regression: a malformed LLM recommendation object must not crash the tick.

2026-06-25 16:20: deepseek-v4-flash-thinking returned a 312-object array where
one object omitted "rationale"; the parser did ``r["rationale"]`` and the
resulting KeyError propagated out of generate_recommendations and killed the
scheduler tick. ``_recommendations_from_data`` now skips a bad object (its ticker
is back-filled by the rule-based pass) and defaults the non-essential rationale.
"""

from datetime import datetime

from src.analysis.claude_analyst import _recommendations_from_data


def _now():
    return datetime(2026, 6, 25, 16, 20)


def test_missing_rationale_is_defaulted_not_fatal():
    data = [
        {"ticker": "AAA", "type": "STOCK", "direction": "BULLISH", "action": "BUY",
         "confidence": 0.9, "rationale": "clear setup"},
        {"ticker": "BBB", "type": "STOCK", "direction": "BULLISH", "action": "BUY",
         "confidence": 0.95},   # NO rationale → kept with a blank rationale
    ]
    recs = _recommendations_from_data(data, "deepseek-v4-flash-thinking", _now())
    by = {r.ticker: r for r in recs}
    assert set(by) == {"AAA", "BBB"}
    assert by["AAA"].rationale == "clear setup"
    assert by["BBB"].rationale == ""             # defaulted, object preserved
    assert by["BBB"].confidence == 0.95


def test_objects_missing_essential_fields_are_skipped_not_fatal():
    data = [
        {"ticker": "AAA", "direction": "BULLISH", "action": "BUY", "confidence": 0.9, "rationale": "ok"},
        {"direction": "BULLISH", "action": "BUY", "confidence": 0.8, "rationale": "no ticker"},   # skip
        {"ticker": "CCC", "action": "BUY", "confidence": 0.8, "rationale": "no direction"},        # skip
        {"ticker": "DDD", "direction": "BULLISH", "action": "BUY",
         "confidence": "abc", "rationale": "bad conf"},                                            # skip
        "not-a-dict-at-all",                                                                       # skip
    ]
    recs = _recommendations_from_data(data, "test", _now())
    assert [r.ticker for r in recs] == ["AAA"]   # only the fully-valid object survives


def test_empty_data_returns_empty():
    assert _recommendations_from_data([], "test", _now()) == []
