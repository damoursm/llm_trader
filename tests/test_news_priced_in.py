"""news_unpriced / news_unpriced_all (2026-09-07, panel-first weight 0).

Pins the contract that makes the pair meaningful: the anchor is the news
CLUSTER's own start (not a fixed look-back), the score is the news read net of
the move already made, the two features genuinely differ (the aggregate sees the
older stories the fresh one cannot), and every degenerate input abstains rather
than inventing a view.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.models import NewsArticle
from src.signals.news_priced_in import (cluster_bounds, compute_news_priced_in,
                                        _unpriced)

# PINNED, not `datetime.now()`. `_frame` builds a business-day index ending
# "yesterday" and the article ages are measured from here, so a floating NOW
# makes the cluster->session mapping depend on which day the suite runs —
# `test_the_trailing_window_can_collapse_the_two_features` passed all day on
# 2026-09-10 and failed after the UTC date rollover, with no code change.
NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _art(hours, url=None):
    return NewsArticle(title="headline", summary="s" * 40, url=url or f"u{hours}",
                       source="Reuters", published_at=NOW - timedelta(hours=hours))


def _frame(closes):
    """A daily frame ending on the last completed session before today."""
    idx = pd.bdate_range(end=pd.Timestamp(NOW.date()) - pd.Timedelta(days=1),
                         periods=len(closes))
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=idx)


def _flat_then(moves, base=100.0, n=40, sigma=0.01):
    """`n` bars of ±sigma noise at `base`, then the given closing path."""
    rng = np.random.default_rng(3)
    noise = base * (1 + rng.normal(0, sigma, n)).cumprod() / (1 + rng.normal(0, sigma, n)).cumprod()[0]
    return list(noise) + [base * m for m in moves]


# ── clustering ───────────────────────────────────────────────────────────────

def test_cluster_bounds_splits_the_digest_and_anchors_on_each_story_start():
    """`as_of=NOW` is load-bearing, not decoration. Without it the partition runs
    on the WALL CLOCK while the fixtures are pinned to `NOW`, so the test passed
    only while real time stayed near 2026-09-09: two days of drift aged the
    "1h-old" article to 49h, the relative-gap cut moved with it, and the split
    silently became 3/2. A test whose verdict depends on the day it runs is worse
    than no test."""
    arts = [_art(1), _art(3), _art(96), _art(100), _art(101)]
    clusters = cluster_bounds(arts, as_of=NOW)
    assert len(clusters) == 2
    (fresh_start, fresh_mass, fresh_n), (old_start, old_mass, old_n) = clusters
    # the START of a cluster is its OLDEST article — when the story began
    assert fresh_n == 2 and old_n == 3
    assert abs((NOW - fresh_start).total_seconds() / 3600 - 3) < 0.1
    assert abs((NOW - old_start).total_seconds() / 3600 - 101) < 0.1
    assert fresh_mass > old_mass                      # recency mass, for the diag


def test_cluster_bounds_is_empty_safe_and_terminates():
    assert cluster_bounds([], as_of=NOW) == []
    assert len(cluster_bounds([_art(2)], as_of=NOW)) == 1
    # 30 separated stories must not loop forever (the guard caps the partition)
    many = [_art(h) for h in (1, 30, 200, 400, 900, 2000)]
    assert 1 <= len(cluster_bounds(many, as_of=NOW)) <= 20


# ── the unpriced ramp ────────────────────────────────────────────────────────

def test_unpriced_ramp_is_full_at_zero_zero_at_two_sigma_and_negative_past_it():
    # sigma 1%/day, one session: a +2% aligned move is 2 sigma.
    assert _unpriced(1.0, 100.0, 100.0, 0.01, 1) == pytest.approx(1.0)
    assert _unpriced(1.0, 102.0, 100.0, 0.01, 1) == pytest.approx(0.0, abs=1e-9)
    assert _unpriced(1.0, 104.0, 100.0, 0.01, 1) == pytest.approx(-1.0)
    # the horizon scales with sqrt(sessions): the same 2% over 4 sessions is 1 sigma
    assert _unpriced(1.0, 102.0, 100.0, 0.01, 4) == pytest.approx(0.5)
    # bearish news is symmetric — an un-fallen name keeps the full read
    assert _unpriced(-1.0, 100.0, 100.0, 0.01, 1) == pytest.approx(1.0)
    assert _unpriced(-1.0, 98.0, 100.0, 0.01, 1) == pytest.approx(0.0, abs=1e-9)


# ── the two features ─────────────────────────────────────────────────────────

def test_good_news_unmoved_keeps_the_read_and_already_moved_flips_it():
    df = _frame(_flat_then([1.0]))
    last = float(df["Close"].iloc[-1])
    arts = [_art(1), _art(3)]
    flat, _all_flat, _ = compute_news_priced_in("T", 0.6, arts, price_now=last, df=df, as_of=NOW)
    moved, _all_moved, _ = compute_news_priced_in("T", 0.6, arts, price_now=last * 1.10, df=df, as_of=NOW)
    assert flat > 0.5                              # the move is still ahead
    assert moved < 0                               # the tape overshot the story
    # bearish news: an un-fallen name keeps the bearish read
    bear, _b_all, _ = compute_news_priced_in("T", -0.6, arts, price_now=last, df=df, as_of=NOW)
    assert bear < -0.5


def test_the_aggregate_sees_the_older_story_the_fresh_feature_cannot(monkeypatch):
    """The whole point of the second feature: two clusters whose prices moved
    differently must not produce the same number. Recency-mass weighting was
    rejected for exactly this reason (the fresh cluster carries ~97% of it).

    Pins the CLUSTER-ANCHOR mechanism, so the trailing window is disabled here —
    with it on, this fixture's run-up makes the fixed reading bind for BOTH
    clusters and the two features legitimately collapse (see the test below)."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", False, raising=False)
    # 40 quiet bars, then a run-up over the last 4 sessions.
    closes = _flat_then([1.00, 1.02, 1.04, 1.06])
    df = _frame(closes)
    last = float(df["Close"].iloc[-1])
    # one story from ~5 days ago (anchored before the run-up) and one from today
    arts = [_art(1), _art(2), _art(120), _art(124), _art(126)]
    fresh, agg, diag = compute_news_priced_in("T", 0.5, arts, price_now=last, df=df, as_of=NOW)
    assert diag["clusters"] == 2
    assert fresh != agg
    # the old story has had the whole run-up priced against it, so the aggregate
    # sits BELOW the fresh-cluster read
    assert agg < fresh


def test_every_degenerate_input_abstains():
    df = _frame(_flat_then([1.0]))
    last = float(df["Close"].iloc[-1])
    arts = [_art(1)]
    assert compute_news_priced_in("T", 0.0, arts, price_now=last, df=df, as_of=NOW)[:2] == (0.0, 0.0)
    assert compute_news_priced_in("T", None, arts, price_now=last, df=df, as_of=NOW)[:2] == (0.0, 0.0)
    assert compute_news_priced_in("T", 0.5, [], price_now=last, df=df, as_of=NOW)[:2] == (0.0, 0.0)
    assert compute_news_priced_in("T", 0.5, arts, price_now=0.0,
                                  df=df, as_of=NOW)[:2] != (float("nan"), float("nan"))
    short = _frame([100.0] * 10)                    # below _MIN_BARS
    assert compute_news_priced_in("T", 0.5, arts, price_now=100.0, df=short, as_of=NOW)[:2] == (0.0, 0.0)
    flat = _frame([100.0] * 40)                     # zero vol -> no z to compute
    assert compute_news_priced_in("T", 0.5, arts, price_now=100.0, df=flat, as_of=NOW)[:2] == (0.0, 0.0)


def test_scores_are_bounded_and_signed_in_the_stock_direction():
    df = _frame(_flat_then([1.0]))
    last = float(df["Close"].iloc[-1])
    arts = [_art(1)]
    for news in (-1.0, -0.3, 0.3, 1.0):
        for px in (last * 0.5, last, last * 1.5):
            f, a, _ = compute_news_priced_in("T", news, arts, price_now=px, df=df, as_of=NOW)
            assert -1.0 <= f <= 1.0 and -1.0 <= a <= 1.0


# ── registration (the add-a-method checklist) ────────────────────────────────

def test_method_is_registered_everywhere_but_the_combine():
    from src.analysis.code_version import METHOD_SOURCES
    from src.analysis.ml_stacker import STACKER_SIGNED_FEATURES
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    from src.models import TickerSignal
    from src.performance.tracker import _ALL_METHODS, METHOD_LABELS
    from src.signals.aggregator import _BASE_WEIGHTS
    from src.signals.agreement import METHOD_FAMILIES
    for m in ("news_unpriced", "news_unpriced_all"):
        assert m in _ALL_METHODS
        assert m in SIGNAL_BASE_METHOD_COLUMNS
        assert m in METHOD_LABELS
        assert m in METHOD_SOURCES
        assert m in STACKER_SIGNED_FEATURES
        assert f"{m}_score" in TickerSignal.model_fields
        # panel-first: NO weight, NO family vote, so it cannot move the combine
        assert m not in _BASE_WEIGHTS
        assert not any(m in fam for fam in METHOD_FAMILIES.values())


# ── two-sided priced-in check (2026-09-09) ──────────────────────────────────

def _frame_from(path):
    import numpy as np
    idx = pd.bdate_range(end=pd.Timestamp(NOW.date()) - pd.Timedelta(days=1), periods=len(path))
    return pd.DataFrame({"Close": np.asarray(path, dtype=float)}, index=idx)


def _quiet(n=36, seed=4):
    import numpy as np
    return list(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.012, n))))


def test_fixed_window_catches_a_move_that_preceded_the_article(monkeypatch):
    """The cluster anchor only sees a reaction that came AFTER the story broke.
    A recap ("Is Up 40.6% After Record Q2") is written once the move is done, so
    its anchor is already post-move and the divergence reads ~0. The fixed
    trailing window does not care which came first."""
    from config.settings import settings
    from src.signals.news_priced_in import compute_news_priced_in
    q = _quiet()
    df = _frame_from(q + [q[-1] * 1.06, q[-1] * 1.12, q[-1] * 1.18, q[-1] * 1.18])
    px = float(df.Close.iloc[-1])
    arts = [_art(2, "a"), _art(3, "b")]                 # written after the move
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", False, raising=False)
    off, _a, _d = compute_news_priced_in("T", 0.6, arts, price_now=px, df=df, as_of=NOW)
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", True, raising=False)
    on, _a2, _d2 = compute_news_priced_in("T", 0.6, arts, price_now=px, df=df, as_of=NOW)
    assert off > 0.4                                      # old: reads un-priced
    assert on < 0                                         # new: reads over-priced -> fade


def test_two_sided_does_not_disturb_the_article_then_move_case(monkeypatch):
    """The case the cluster anchor already handled must be unchanged — taking
    the MORE priced-in of the two readings can only tighten, never loosen."""
    from config.settings import settings
    from src.signals.news_priced_in import compute_news_priced_in
    q = _quiet()
    df = _frame_from(q + [q[-1] * 1.001, q[-1] * 1.002, q[-1] * 1.001, q[-1]])
    px = float(df.Close.iloc[-1]) * 1.14                  # the surge is only in the live mark
    arts = [_art(2, "a"), _art(3, "b")]
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", False, raising=False)
    off, _a, _d = compute_news_priced_in("T", 0.6, arts, price_now=px, df=df, as_of=NOW)
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", True, raising=False)
    on, _a2, _d2 = compute_news_priced_in("T", 0.6, arts, price_now=px, df=df, as_of=NOW)
    assert on <= off                                      # never looser
    assert on < 0


def test_two_sided_cannot_fabricate_a_fade_on_a_quiet_tape(monkeypatch):
    """Both readings must be low for the score to pass through — a max() over
    two near-zero z values is still near zero."""
    from config.settings import settings
    from src.signals.news_priced_in import compute_news_priced_in
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", True, raising=False)
    q = _quiet(40, seed=9)
    df = _frame_from(q)
    f, _a, _d = compute_news_priced_in("T", 0.6, [_art(2, "a"), _art(3, "b")],
                                       price_now=float(df.Close.iloc[-1]), df=df)
    assert f > 0.3                                        # flat tape -> the move is still ahead


def test_the_trailing_window_can_collapse_the_two_features(monkeypatch):
    """Documented interaction, not a bug: when the fixed trailing reading is the
    binding one, every cluster inherits the same z and `news_unpriced_all`
    carries nothing beyond `news_unpriced`. That is semantically right — a tape
    that has already moved 3 sigma has priced every story in the digest — but it
    means the second feature only adds information when the trailing window is
    NOT binding, which is what its panel evaluation has to be read against."""
    from config.settings import settings
    from src.signals.news_priced_in import compute_news_priced_in
    monkeypatch.setattr(settings, "enable_news_unpriced_two_sided", True, raising=False)
    closes = _flat_then([1.00, 1.02, 1.04, 1.06])          # run-up inside the 3d window
    df = _frame(closes)
    arts = [_art(1), _art(2), _art(120), _art(124), _art(126)]
    fresh, agg, _d = compute_news_priced_in("T", 0.5, arts,
                                            price_now=float(df.Close.iloc[-1]), df=df)
    assert fresh == agg
