"""Sentiment velocity (`src/signals/sentiment_velocity.py`) — a weighted method.

The RATE OF CHANGE of news tone, not its level: a stock improving from very
negative toward neutral often rallies while still net-negative. The `news`
method supplies the level; this one supplies the derivative, and the pair is
only informative if they stay genuinely different measurements.

What makes it safe to use a crude lexicon: the score is a DIFFERENCE of the same
measure across two windows, so a systematic bias in the lexicon cancels. That
property is worth pinning, because "improve the lexicon" is the obvious thing to
do to this module and the differencing is what makes its calibration not matter.

The other load-bearing rule is that an empty window scores 0, not "no change".
An unmeasurable derivative and a measured-zero derivative look identical in the
output but mean opposite things about the evidence.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.models import NewsArticle
from src.signals.sentiment_velocity import (_count_scale, _lexical_polarity,
                                            compute_sentiment_velocity)


def _art(hours_ago: float, title: str = "", summary: str = "") -> NewsArticle:
    return NewsArticle(
        title=title, summary=summary,
        url=f"http://x/{title}/{summary}/{hours_ago}", source="rss",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago))


_POSITIVE = "beat surge rally upgrade record"
_NEGATIVE = "miss plunge downgrade lawsuit warning"


# ── per-article polarity ────────────────────────────────────────────────────

def test_polarity_is_the_normalised_keyword_balance():
    assert _lexical_polarity(_art(1, _POSITIVE)) == pytest.approx(1.0)
    assert _lexical_polarity(_art(1, _NEGATIVE)) == pytest.approx(-1.0)
    assert _lexical_polarity(_art(1, "beat miss")) == pytest.approx(0.0)
    assert _lexical_polarity(_art(1, "beat beat miss")) == pytest.approx(1 / 3)


def test_toneless_text_scores_zero_not_an_error():
    for text in ("", "the company issued a statement today", "12345 !!!"):
        assert _lexical_polarity(_art(1, text)) == 0.0


def test_polarity_reads_title_and_summary_together():
    assert _lexical_polarity(_art(1, title="beat", summary="surge")) == pytest.approx(1.0)
    assert _lexical_polarity(_art(1, title="beat", summary="miss")) == pytest.approx(0.0)


def test_polarity_is_case_insensitive():
    assert _lexical_polarity(_art(1, "BEAT SURGE")) == \
        _lexical_polarity(_art(1, "beat surge"))


def test_the_lexicons_do_not_overlap():
    """A word in both sets cancels itself and silently contributes nothing —
    the kind of edit that looks like tuning and is actually a no-op."""
    from src.signals.sentiment_velocity import _NEG, _POS
    assert not (_POS & _NEG), f"words in both lexicons: {sorted(_POS & _NEG)}"


# ── the windows ─────────────────────────────────────────────────────────────

def test_improving_tone_scores_positive():
    """The whole thesis: still-negative news that is LESS negative than before."""
    arts = [_art(2, _NEGATIVE + " beat surge rally upgrade"),   # recent, less bad
            _art(50, _NEGATIVE), _art(60, _NEGATIVE)]           # prior, very bad
    score, recent, prior, n = compute_sentiment_velocity("AAA", arts)
    assert score > 0
    assert recent > prior
    assert n == 3


def test_deteriorating_tone_scores_negative():
    arts = [_art(2, _NEGATIVE), _art(50, _POSITIVE), _art(60, _POSITIVE)]
    score, recent, prior, _n = compute_sentiment_velocity("AAA", arts)
    assert score < 0 and recent < prior


def test_unchanged_tone_scores_zero_even_when_the_level_is_extreme():
    """The separation from the `news` method: uniformly terrible news has a
    strongly negative LEVEL and zero VELOCITY."""
    arts = [_art(2, _NEGATIVE), _art(3, _NEGATIVE),
            _art(50, _NEGATIVE), _art(60, _NEGATIVE)]
    score, recent, prior, _n = compute_sentiment_velocity("AAA", arts)
    assert score == pytest.approx(0.0)
    assert recent == prior == pytest.approx(-1.0)


def test_articles_are_bucketed_by_age_at_the_window_boundaries():
    """`age <= recent_hours` is recent; `recent < age <= prior_hours` is prior;
    anything older is ignored entirely."""
    arts = [_art(1, _POSITIVE),      # recent
            _art(30, _NEGATIVE),     # prior
            _art(500, _POSITIVE)]    # too old — must not enter either window
    score, recent, prior, n = compute_sentiment_velocity(
        "AAA", arts, recent_hours=24, prior_hours=96)
    assert n == 2, "a stale article leaked into a window"
    assert recent == pytest.approx(1.0) and prior == pytest.approx(-1.0)
    assert score > 0


def test_an_empty_window_yields_no_measurement():
    """0.0 here means 'cannot measure', which is why an absent window must not
    be treated as a zero-tone window — that would fabricate a derivative."""
    only_recent = [_art(1, _POSITIVE), _art(2, _POSITIVE)]
    assert compute_sentiment_velocity("AAA", only_recent)[0] == 0.0
    only_prior = [_art(50, _POSITIVE), _art(60, _POSITIVE)]
    assert compute_sentiment_velocity("AAA", only_prior)[0] == 0.0


def test_no_articles_returns_the_neutral_tuple():
    assert compute_sentiment_velocity("AAA", []) == (0.0, 0.0, 0.0, 0)
    assert compute_sentiment_velocity("AAA", None) == (0.0, 0.0, 0.0, 0)


def test_an_inverted_window_configuration_is_refused():
    """`prior_hours <= recent_hours` makes the prior window empty by
    construction; scoring it anyway would emit a permanent 0 that reads as a
    measured no-change."""
    arts = [_art(1, _POSITIVE), _art(50, _NEGATIVE)]
    assert compute_sentiment_velocity("AAA", arts, recent_hours=96,
                                      prior_hours=24) == (0.0, 0.0, 0.0, 0)


def test_future_timestamps_are_clamped_not_dropped():
    """Feeds occasionally publish a few minutes into the future; treating that
    as a negative age must not push the article out of the recent window."""
    arts = [_art(-2, _POSITIVE), _art(50, _NEGATIVE), _art(60, _NEGATIVE)]
    score, _r, _p, n = compute_sentiment_velocity("AAA", arts)
    assert n == 3 and score > 0


def test_articles_without_a_usable_timestamp_are_skipped():
    from types import SimpleNamespace
    good = _art(1, _POSITIVE)
    old = _art(50, _NEGATIVE)
    broken = SimpleNamespace(title="x", summary="y", published_at=None)
    score, _r, _p, n = compute_sentiment_velocity("AAA", [good, old, broken])
    assert n == 2 and score > 0


# ── the confidence damping ──────────────────────────────────────────────────

def test_thin_windows_are_damped():
    """One article per side is a real but weak measurement; the same tone gap
    over many articles must score harder."""
    thin = [_art(1, _POSITIVE), _art(50, _NEGATIVE)]
    thick = ([_art(1 + i * 0.1, _POSITIVE) for i in range(12)]
             + [_art(50 + i * 0.1, _NEGATIVE) for i in range(12)])
    assert 0 < compute_sentiment_velocity("AAA", thin)[0] < \
        compute_sentiment_velocity("AAA", thick)[0]


def test_count_scale_is_monotone_and_bounded():
    assert _count_scale(0) == 0.0
    vals = [_count_scale(n) for n in (1, 2, 3, 7, 12, 50, 1000)]
    assert vals == sorted(vals)
    assert all(0.0 < v <= 1.0 for v in vals)
    assert _count_scale(1) == pytest.approx(0.45, abs=0.01)


def test_score_is_bounded_to_the_unit_interval():
    """The combine assumes every method emits [-1, +1]."""
    extreme = ([_art(1, _POSITIVE) for _ in range(50)]
               + [_art(50, _NEGATIVE) for _ in range(50)])
    score, _r, _p, _n = compute_sentiment_velocity("AAA", extreme)
    assert -1.0 <= score <= 1.0
    assert score == pytest.approx(math.tanh(2.0 / 0.6), abs=0.01)


def test_the_sign_convention_matches_the_stock_not_the_news():
    """Project-wide invariant: + = the STOCK is expected to go up. Improving
    tone is bullish, so the derivative's sign passes straight through."""
    improving = [_art(1, _POSITIVE), _art(50, _NEGATIVE), _art(55, _NEGATIVE)]
    worsening = [_art(1, _NEGATIVE), _art(50, _POSITIVE), _art(55, _POSITIVE)]
    assert compute_sentiment_velocity("AAA", improving)[0] > 0
    assert compute_sentiment_velocity("AAA", worsening)[0] < 0
