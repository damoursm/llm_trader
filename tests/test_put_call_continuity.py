"""put_call continuity (2026-08-16, epoch "put_call"): the step map's four
values collapsed each day's cross-section into rank-tie blocks — the score is
now continuous in the raw ratio, anchored to the old extremes."""

from datetime import date

import pytest

from src.models import PutCallContext, PutCallSignal
from src.signals.aggregator import _pc_ratio_score, _put_call_score_for


def _ctx(ratio, signal="PUTS_HEAVY", with_ratio=True):
    sig = PutCallSignal(ticker="TST", put_volume=1000, call_volume=500,
                        put_call_ratio=(ratio if with_ratio else 0.0),
                        signal=signal, direction="BEARISH", summary="s")
    return PutCallContext(ticker_signals=[sig], report_date=date.today(), summary="s")


def test_anchors_match_the_old_extremes():
    assert _pc_ratio_score(2.0) == pytest.approx(0.70, abs=0.005)   # old EXTREME_PUTS
    assert _pc_ratio_score(0.3) == pytest.approx(-0.70, abs=0.005)  # old EXTREME_CALLS
    assert _pc_ratio_score(1.0) == 0.0


def test_continuous_and_monotone_in_the_ratio():
    ratios = [0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 6.0]
    scores = [_pc_ratio_score(r) for r in ratios]
    assert scores == sorted(scores)                      # monotone
    assert len(set(scores)) == len(scores)               # all distinct — no steps
    # two puts-heavy names no longer share a value (the tie-block defect)
    assert _pc_ratio_score(1.6) != _pc_ratio_score(3.0)
    assert -1.0 < scores[0] and scores[-1] < 1.0
    # extremes clamp instead of exploding
    assert _pc_ratio_score(1000.0) == _pc_ratio_score(20.0)
    assert _pc_ratio_score(0.0001) == _pc_ratio_score(0.05)


def test_score_for_uses_the_ratio_first():
    s = _put_call_score_for("TST", _ctx(3.0, signal="EXTREME_PUTS"))
    assert s == pytest.approx(_pc_ratio_score(3.0))
    assert s > 0.70                                       # deeper than the old cap


def test_score_for_falls_back_to_the_label_without_a_ratio():
    # ratio 0 (unusable) → the legacy label map keeps old cached contexts scored
    assert _put_call_score_for("TST", _ctx(0.0, signal="PUTS_HEAVY",
                                           with_ratio=False)) == pytest.approx(0.35)
    assert _put_call_score_for("TST", None) == 0.0
    assert _put_call_score_for("OTHER", _ctx(2.0)) == 0.0  # not in the surfaced set


def test_f_short_squeeze_excluded_from_the_additive_overlay(monkeypatch):
    """2026-08-16 shape audit: the crowding magnitude's payoff is a replicated
    U (middle bearish), so it must not be consumed as linear-positive — it
    keeps persisting for panel IC but adds NOTHING to combined_score."""
    from config.settings import settings
    import src.signals.aggregator as agg
    from tests.test_news_events import _OFF
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", False)
    monkeypatch.setattr(settings, "enable_news_bear_fresh", False)
    monkeypatch.setattr(settings, "enable_catalyst_tilt", False)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.4, "n"))

    def _build(factors):
        s = agg.build_signals(["TST"], articles=[], snapshots=[],
                              fundamental_factors={"TST": factors} if factors else None)[0]
        return float(s.combined_score)

    base = _build(None)
    with_sq = _build({"f_short_squeeze": 0.9})
    with_val = _build({"f_value": 0.9})
    assert with_sq == pytest.approx(base)                 # excluded from the sum
    assert with_val != pytest.approx(base)                # the others still add
    # (persistence is the pipeline's fundamental_scores merge — untouched here
    # and pinned by tests/test_db_signals.py's fundamentals bridge test.)
