"""Exit-method scoring — the signed hold-conviction convention (analysis.exit_methods).

+ = the position should keep running (hold), − = it should reverse (exit); every
score is position-oriented (× the position's direction) so the exit panel can
correlate it directly against the position's direction-oriented forward return.
"""

from types import SimpleNamespace as NS

import pytest

from src.models import TickerSignal
from src.analysis.exit_methods import (
    build_exit_scores, exit_category_for,
    EXIT_CATEGORY_DECISION, EXIT_CATEGORY_SIGNAL,
)


def _rec(action, conf):
    return NS(action=action, confidence=conf, direction=None)


def _long(**extra):
    t = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
         "recommendation_id": "p1", "current_price": 100.0}
    t.update(extra)
    return t


def _short(**extra):
    t = {"ticker": "BBB", "action": "SELL", "direction": "BEARISH",
         "recommendation_id": "p2", "current_price": 50.0}
    t.update(extra)
    return t


def _sig(combined=0.0, **scores):
    s = TickerSignal(ticker="AAA", direction="BULLISH", confidence=0.8,
                     sentiment_score=scores.get("sentiment_score", 0.0),
                     technical_score=scores.get("technical_score", 0.0), rationale="t")
    s.combined_score = combined
    return s


# ── llm_review: the synthesized decider ────────────────────────────────────

def test_llm_review_reaffirm_is_positive():
    sc = build_exit_scores(_long(), _rec("BUY", 0.75), {}, NS(regime="NEUTRAL"))
    assert sc["llm_review"] == pytest.approx(0.75)     # reaffirms the long → hold


def test_llm_review_flip_is_negative():
    sc = build_exit_scores(_long(), _rec("SELL", 0.6), {}, NS(regime="NEUTRAL"))
    assert sc["llm_review"] == pytest.approx(-0.6)     # flips against the long → exit


def test_llm_review_flip_against_short_is_negative():
    # A BUY review is a flip against a SHORT position → exit signal (negative).
    sc = build_exit_scores(_short(), _rec("BUY", 0.6), {}, NS(regime="NEUTRAL"))
    assert sc["llm_review"] == pytest.approx(-0.6)


def test_hold_review_omits_llm_review():
    sc = build_exit_scores(_long(), _rec("HOLD", 0.9), {}, NS(regime="NEUTRAL"))
    assert "llm_review" not in sc                      # no directional exit view


def test_no_review_omits_llm_review():
    sc = build_exit_scores(_long(), None, {}, NS(regime="NEUTRAL"))
    assert "llm_review" not in sc


# ── macro regime overlay (position-oriented) ───────────────────────────────

def test_macro_regime_long_panic_is_exit():
    sc = build_exit_scores(_long(), None, {}, NS(regime="PANIC"))
    assert sc["macro_regime"] == pytest.approx(-1.0)   # long in PANIC → exit


def test_macro_regime_short_panic_is_hold():
    sc = build_exit_scores(_short(), None, {}, NS(regime="PANIC"))
    assert sc["macro_regime"] == pytest.approx(1.0)    # short in PANIC → keep


def test_macro_regime_risk_on_long_is_hold():
    sc = build_exit_scores(_long(), None, {}, NS(regime="RISK_ON"))
    assert sc["macro_regime"] == pytest.approx(0.5)


def test_macro_regime_neutral_omitted():
    sc = build_exit_scores(_long(), None, {}, NS(regime="NEUTRAL"))
    assert "macro_regime" not in sc                    # 0 = no view → omitted


# ── horizon time-stop (one-sided: 0 within window, negative past it) ───────

def test_horizon_within_window_omitted():
    # entered "now" with a long (1-month) horizon → well within window → no pressure.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    sc = build_exit_scores(_long(target_horizon="1m", entry_datetime=now),
                           None, {}, NS(regime="NEUTRAL"))
    assert "horizon" not in sc


def test_horizon_past_window_is_negative():
    # entered long ago with a 30-minute horizon → far past window → exit pressure.
    sc = build_exit_scores(_long(target_horizon="30m",
                                 entry_datetime="2020-01-01T09:30:00-05:00"),
                           None, {}, NS(regime="NEUTRAL"))
    assert sc["horizon"] < 0
    assert sc["horizon"] >= -1.0                        # saturates at −1


def test_horizon_no_target_omitted():
    sc = build_exit_scores(_long(), None, {}, NS(regime="NEUTRAL"))
    assert "horizon" not in sc


# ── signal methods re-scored as exit signals (oriented) ────────────────────

def test_signal_methods_oriented_long():
    sig = _sig(combined=0.4, sentiment_score=0.5)
    sc = build_exit_scores(_long(), None, {"AAA": sig}, NS(regime="NEUTRAL"))
    assert sc["aggregator"] == pytest.approx(0.4)       # combined × +1
    assert sc["news"] == pytest.approx(0.5)             # bullish news = hold a long


def test_signal_methods_oriented_short():
    sig = TickerSignal(ticker="BBB", direction="BEARISH", confidence=0.8,
                       sentiment_score=0.5, technical_score=0.0, rationale="t")
    sig.combined_score = -0.2
    sc = build_exit_scores(_short(), None, {"BBB": sig}, NS(regime="NEUTRAL"))
    assert sc["aggregator"] == pytest.approx(0.2)       # bearish combined × −1 = hold the short
    assert sc["news"] == pytest.approx(-0.5)            # bullish news is BAD for a short → exit


def test_zero_scores_are_omitted():
    sig = _sig(combined=0.0, sentiment_score=0.0)       # everything flat
    sc = build_exit_scores(_long(), None, {"AAA": sig}, NS(regime="NEUTRAL"))
    assert sc == {}                                     # nothing non-zero → empty


# ── category mapping ────────────────────────────────────────────────────────

def test_exit_category_for():
    for m in ("llm_review", "aggregator", "macro_regime", "horizon"):
        assert exit_category_for(m) == EXIT_CATEGORY_DECISION
    for m in ("news", "tech", "momentum", "pead"):
        assert exit_category_for(m) == EXIT_CATEGORY_SIGNAL
