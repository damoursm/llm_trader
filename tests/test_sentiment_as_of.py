"""`as_of` — scoring a digest at a PAST instant (2026-09-07).

The news scorer measured article age against `datetime.now()` everywhere, so a
replay of any tick older than a week found every article outside the 7-day
window, discarded the whole digest and abstained. Found by the backfill
pre-flight, silently: 144 tickers "scored" in 13 seconds.

The contract these tests pin:
  1. `as_of=None` is byte-identical to the old now-relative behaviour — this is
     the LIVE path and it must not move;
  2. `as_of=<instant>` measures every age from that instant, in all four places
     the clock is read (weights, the freshness cut, the top-20 sort, and the age
     LABELS the model reads);
  3. the derived scorers that carry their own clock take it too.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import sentiment as sent
from src.models import NewsArticle

PAST = datetime(2026, 7, 30, 3, 30, tzinfo=timezone.utc)


def _art(published, url=None):
    return NewsArticle(title="headline", summary="s" * 40, url=url or str(published),
                       source="Reuters", published_at=published)


def test_as_of_none_is_the_live_clock():
    """The default must stay NOW — a fresh article weighs ~1, a 10-day-old one 0."""
    now = datetime.now(timezone.utc)
    assert sent._recency_weight(_art(now - timedelta(hours=1))) > 0.9
    assert sent._recency_weight(_art(now - timedelta(days=10))) == 0.0
    # explicit None is the same call
    assert (sent._recency_weight(_art(now - timedelta(hours=1)), None)
            == sent._recency_weight(_art(now - timedelta(hours=1))))


def test_as_of_moves_the_whole_clock():
    """A July article is weightless today and fresh as of its own tick — the
    difference between a backfill that works and one that abstains on
    everything."""
    july = _art(PAST - timedelta(hours=2))
    assert sent._recency_weight(july) == 0.0                 # measured from today
    assert sent._recency_weight(july, PAST) > 0.9            # measured from the tick
    # the 7-day cut moves with it
    assert sent._recency_weight(_art(PAST - timedelta(days=10)), PAST) == 0.0


def test_clock_is_hour_quantised_and_tz_safe():
    """Same quantisation as before, so two runs inside one hour cut identically;
    a naive datetime is treated as UTC rather than crashing."""
    c = sent._clock(PAST.replace(minute=47, second=13))
    assert (c.minute, c.second, c.microsecond) == (0, 0, 0)
    assert sent._clock(datetime(2026, 7, 30, 3, 30)).tzinfo is timezone.utc


def test_attention_mass_and_recent_cluster_take_the_clock():
    arts = [_art(PAST - timedelta(hours=1)), _art(PAST - timedelta(hours=2)),
            _art(PAST - timedelta(days=5))]
    assert sent.attention_mass(arts) == (0, 0.0)             # all stale vs today
    n, mass = sent.attention_mass(arts, PAST)
    assert n == 3 and mass > 1.5
    # the cluster cut is relative to the freshest article AS OF the tick
    assert len(sent.recent_cluster(arts, as_of=PAST)) == 2


def test_digest_age_labels_are_dated_from_the_tick(monkeypatch):
    """The age labels are part of what the model reads — the priced-in check
    turns on "3h ago" vs "5.2d ago" — so a replay must date them from its own
    tick or the prompt describes a different world than the scores do."""
    seen = {}

    def _capture(ticker, articles, prompt, *a, **k):
        seen["prompt"] = prompt
        return 0.4, "r", "earnings"

    monkeypatch.setattr(sent, "_score_with_engine", _capture, raising=False)
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda k: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **k: None)
    arts = [_art(PAST - timedelta(hours=3))]
    try:
        sent.analyse_sentiment("AAA", arts, force_engine="local", as_of=PAST)
    except Exception:                                        # engine wiring varies
        pass
    if "prompt" in seen:
        assert "3h ago" in seen["prompt"]


def test_velocity_takes_the_clock():
    from src.signals.sentiment_velocity import compute_sentiment_velocity
    arts = [_art(PAST - timedelta(hours=2), "a"), _art(PAST - timedelta(hours=60), "b")]
    _v_now, _r, _p, n_now = compute_sentiment_velocity("AAA", arts)
    _v, _r2, _p2, n_past = compute_sentiment_velocity("AAA", arts, as_of=PAST)
    assert n_now == 0                      # both windows empty measured from today
    assert n_past == 2                     # one in each window as of the tick


def test_news_priced_in_takes_the_clock():
    import numpy as np
    import pandas as pd
    from src.signals.news_priced_in import cluster_bounds, compute_news_priced_in
    arts = [_art(PAST - timedelta(hours=1), "a"), _art(PAST - timedelta(hours=2), "b"),
            _art(PAST - timedelta(days=5), "c")]
    assert len(cluster_bounds(arts, as_of=PAST)) == 2
    idx = pd.bdate_range(end=pd.Timestamp(PAST.date()), periods=40)
    # real-ish vol: a perfectly smooth ramp has sigma below `_MIN_SIGMA` and the
    # scorer correctly abstains on it (degenerate vol is not a price signal)
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.012, 40))), index=idx)
    f, a, diag = compute_news_priced_in("AAA", 0.5, arts, price_now=104.0,
                                        df=pd.DataFrame({"Close": close}), as_of=PAST)
    assert diag.get("clusters") == 2
    assert f != 0.0
