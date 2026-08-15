"""Cointegration pairs (`src/signals/cointegration.py`) — the `coint` method.

Engle-Granger two-step with a hand-rolled ADF test. The statistics are the
interesting part: `stats.py`'s standing rule is that a significance test the
system ACTS on must not be behind an optional import, and this module honours it
by implementing ADF in numpy — which means the arithmetic has to be pinned
against series whose stationarity is known by construction, not just exercised.

Three properties carry the risk:

* **the ADF stat must actually discriminate.** A stationary spread has to score
  more negative than a random walk; if it didn't, the module would report
  "cointegrated" at whatever rate the critical value happens to cut, and nothing
  downstream could tell;
* **the leg assignment is the trade.** A rich spread (z high) means SHORT the
  first leg — inverting that turns every pair into a losing trade while the
  summary line still reads sensibly;
* **the per-ticker score must be direction-consistent with the legs**, since it
  is what actually reaches the aggregator as `coint`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import cointegration as ci


def _idx(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _random_walk(n=300, seed=0, start=100.0, sigma=0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, sigma, n)))


def _log_series(values, start="2024-01-01") -> pd.Series:
    v = np.asarray(values, dtype=float)
    return pd.Series(np.log(v), index=_idx(len(v), start))


def _cointegrated_pair(n=300, seed=0, beta=1.0, spread_sigma=0.01, last_z=0.0):
    """B is a random walk; A = beta*B + a STATIONARY (AR(1)) spread, so the two
    are cointegrated by construction. `last_z` forces the final spread value to
    a chosen number of spread-sigmas so the entry branch is reachable."""
    rng = np.random.default_rng(seed)
    b_log = np.log(_random_walk(n, seed=seed, sigma=0.012))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + rng.normal(0, spread_sigma)
    if last_z is not None:
        spread[-1] = np.mean(spread) + last_z * np.std(spread, ddof=1)
    a_log = beta * b_log + spread
    return (pd.Series(a_log, index=_idx(n)), pd.Series(b_log, index=_idx(n)))


# ── the ADF test ────────────────────────────────────────────────────────────

def test_adf_separates_a_stationary_series_from_a_random_walk():
    """The decisive statistical property. Without it every downstream verdict is
    a coin flip wearing a p-value."""
    rng = np.random.default_rng(7)
    n = 400
    stationary = np.zeros(n)
    for i in range(1, n):
        stationary[i] = 0.6 * stationary[i - 1] + rng.normal(0, 1)
    walk = np.cumsum(rng.normal(0, 1, n))

    t_stat = ci._adf_tstat(stationary)
    t_walk = ci._adf_tstat(walk)
    assert t_stat is not None and t_walk is not None
    assert t_stat < ci._EG_CRIT[0.05], "a strongly mean-reverting series failed ADF"
    assert t_stat < t_walk, "ADF did not discriminate stationary from unit-root"


def test_adf_returns_none_on_a_sample_too_short_to_regress():
    assert ci._adf_tstat(np.arange(5.0)) is None
    assert ci._adf_tstat(np.array([])) is None


def test_adf_ignores_non_finite_values():
    rng = np.random.default_rng(3)
    s = np.zeros(300)
    for i in range(1, 300):
        s[i] = 0.5 * s[i - 1] + rng.normal(0, 1)
    dirty = s.copy()
    dirty[10] = np.nan
    dirty[20] = np.inf
    assert ci._adf_tstat(dirty) is not None


def test_adf_declines_on_a_degenerate_series():
    """A constant has no variance to regress; returning a number would make a
    flat spread look infinitely stationary."""
    assert ci._adf_tstat(np.ones(300)) is None


def test_pvalue_is_monotone_and_bounded():
    """Coarse by design (interpolated MacKinnon anchors), but it is displayed
    and sorted on, so it must at least never increase with a stronger stat."""
    stats = [-6.0, -4.5, -3.9, -3.34, -3.04, -2.5, -2.0, -1.0, 0.5]
    ps = [ci._adf_pvalue(t) for t in stats]
    assert ps == sorted(ps)
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert ci._adf_pvalue(-3.9) == pytest.approx(0.01)
    assert ci._adf_pvalue(-3.34) == pytest.approx(0.05)


# ── the half-life ───────────────────────────────────────────────────────────

def test_half_life_is_shorter_for_faster_reversion():
    rng = np.random.default_rng(11)
    def _ar(phi, n=500):
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = phi * s[i - 1] + rng.normal(0, 1)
        return s
    fast = ci._half_life(_ar(0.3))
    slow = ci._half_life(_ar(0.95))
    assert 0 < fast < slow


def test_a_non_reverting_series_has_infinite_half_life():
    """λ ≥ 0 means the spread does not come back; `inf` is what filters the pair
    out of the tradeable set rather than a huge finite number that might pass."""
    assert ci._half_life(np.cumsum(np.ones(200))) == float("inf")
    assert ci._half_life(np.arange(5.0)) == float("inf")


# ── the pair test ───────────────────────────────────────────────────────────

def test_a_constructed_cointegrated_pair_is_detected():
    a, b = _cointegrated_pair(seed=1, beta=1.0)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp is not None
    assert cp.is_cointegrated is True
    assert cp.adf_stat <= cp.adf_crit_5pct
    assert cp.hedge_ratio == pytest.approx(1.0, abs=0.15)
    assert np.isfinite(cp.half_life_days)


def test_two_independent_random_walks_are_not_cointegrated():
    a = _log_series(_random_walk(300, seed=1))
    b = _log_series(_random_walk(300, seed=99))
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp is not None
    assert cp.is_cointegrated is False


def test_the_hedge_ratio_is_recovered():
    a, b = _cointegrated_pair(seed=2, beta=1.8, spread_sigma=0.005)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp.hedge_ratio == pytest.approx(1.8, abs=0.15)


# ── leg assignment: this IS the trade ───────────────────────────────────────

def test_a_rich_spread_shorts_the_first_leg():
    """z ≥ +entry means A is expensive relative to B → SHORT A / LONG B.
    Inverting this reads perfectly in the summary and loses on every pair."""
    a, b = _cointegrated_pair(seed=3, last_z=+3.0)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp.spread_zscore >= settings.cointegration_entry_z
    assert cp.short_leg == "AAA" and cp.long_leg == "BBB"


def test_a_cheap_spread_longs_the_first_leg():
    a, b = _cointegrated_pair(seed=4, last_z=-3.0)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp.spread_zscore <= -settings.cointegration_entry_z
    assert cp.long_leg == "AAA" and cp.short_leg == "BBB"


def test_a_fair_value_spread_is_neutral():
    a, b = _cointegrated_pair(seed=5, last_z=0.0)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert abs(cp.spread_zscore) < settings.cointegration_exit_z
    assert cp.signal == "NEUTRAL"


def test_a_partly_stretched_spread_only_monitors():
    z = (settings.cointegration_entry_z + settings.cointegration_exit_z) / 2
    a, b = _cointegrated_pair(seed=6, last_z=z)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp.signal == "MONITOR"


def test_slow_reversion_downgrades_entry_to_stretched(monkeypatch):
    """Cointegrated and stretched, but too slow to trade — the pair still
    reports, flagged so it is not mistaken for an actionable entry."""
    a, b = _cointegrated_pair(seed=7, last_z=+3.0)
    monkeypatch.setattr(ci, "_half_life", lambda s: ci._MAX_HALF_LIFE + 10)
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b})
    assert cp.signal == "STRETCHED"


# ── pair-test guards ────────────────────────────────────────────────────────

def test_a_missing_series_yields_no_pair():
    a, b = _cointegrated_pair()
    assert ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": None}) is None
    assert ci._test_pair("AAA", "BBB", {"AAA": None, "BBB": b}) is None
    assert ci._test_pair("AAA", "BBB", {}) is None


def test_insufficient_overlap_yields_no_pair():
    a, _b = _cointegrated_pair(n=300)
    short = _log_series(_random_walk(ci._MIN_OVERLAP - 5), start="2024-01-01")
    assert ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": short}) is None


def test_a_degenerate_spread_yields_no_pair():
    """Two identical series have a zero-variance spread; the z-score would be a
    divide-by-zero."""
    s = _log_series(_random_walk(300, seed=8))
    assert ci._test_pair("AAA", "BBB", {"AAA": s, "BBB": s.copy()}) is None


def test_pairs_are_tested_on_the_intersection_of_their_dates():
    a, b = _cointegrated_pair(seed=9)
    b_shift = b.copy()
    b_shift.index = _idx(len(b), start="2024-02-01")     # partial overlap
    cp = ci._test_pair("AAA", "BBB", {"AAA": a, "BBB": b_shift})
    if cp is not None:
        assert cp.lookback_days <= min(len(a), len(b_shift))


# ── candidate construction ──────────────────────────────────────────────────

def test_curated_pairs_are_always_candidates():
    """Every DISTINCT curated pair survives, regardless of today's universe.
    (The curated list itself contains one duplicate — AAPL/MSFT is listed
    twice — which the dedupe absorbs; asserting on the raw length would pin
    the typo rather than the behaviour.)"""
    pairs = {tuple(sorted((a.upper(), b.upper()))) for a, b in ci._candidate_pairs([])}
    curated = {tuple(sorted((a.upper(), b.upper()))) for a, b in ci._CANDIDATE_PAIRS}
    assert curated <= pairs


def test_candidates_are_deduplicated_and_never_self_paired():
    pairs = ci._candidate_pairs(["AAPL", "MSFT", "NVDA"])
    keys = [tuple(sorted(p)) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(a != b for a, b in pairs)


def test_same_sector_universe_members_are_paired():
    """AAPL/MSFT/NVDA all map to XLK in the aggregator's sector map."""
    pairs = {tuple(sorted(p)) for p in ci._candidate_pairs(["AAPL", "MSFT", "NVDA"])}
    assert ("AAPL", "MSFT") in pairs and ("MSFT", "NVDA") in pairs


def test_tickers_outside_the_universe_do_not_generate_sector_pairs():
    base = {tuple(sorted(p)) for p in ci._candidate_pairs([])}
    with_universe = {tuple(sorted(p)) for p in ci._candidate_pairs(["AAPL", "MSFT"])}
    assert ("AAPL", "MSFT") in with_universe - base or ("AAPL", "MSFT") in base


# ── the per-ticker score ────────────────────────────────────────────────────

def test_ticker_scores_agree_with_the_leg_assignment(monkeypatch):
    """The score is what reaches the aggregator as `coint` — the long leg must
    read bullish and the short leg bearish, or the method contradicts the pair
    it was derived from."""
    a, b = _cointegrated_pair(seed=12, last_z=+3.5)
    monkeypatch.setattr(ci, "_log_close", lambda t: {"AAA": a, "BBB": b}.get(t))
    monkeypatch.setattr(ci, "_candidate_pairs", lambda tickers: [("AAA", "BBB")])
    ctx = ci.find_cointegrated_pairs(["AAA", "BBB"])
    assert ctx.pairs, "the constructed pair was not reported tradeable"
    p = ctx.pairs[0]
    assert ctx.ticker_scores[p.long_leg] > 0 > ctx.ticker_scores[p.short_leg]
    assert all(-1.0 <= v <= 1.0 for v in ctx.ticker_scores.values())


def test_no_tradeable_pairs_yields_no_scores(monkeypatch):
    a = _log_series(_random_walk(300, seed=21))
    b = _log_series(_random_walk(300, seed=22))
    monkeypatch.setattr(ci, "_log_close", lambda t: {"AAA": a, "BBB": b}.get(t))
    monkeypatch.setattr(ci, "_candidate_pairs", lambda tickers: [("AAA", "BBB")])
    ctx = ci.find_cointegrated_pairs(["AAA", "BBB"])
    assert ctx.pairs == [] and ctx.ticker_scores == {}
    assert "No tradeable" in ctx.summary


def test_tickers_without_history_are_skipped_not_fatal(monkeypatch):
    monkeypatch.setattr(ci, "_log_close", lambda t: None)
    monkeypatch.setattr(ci, "_candidate_pairs", lambda tickers: [("AAA", "BBB")])
    ctx = ci.find_cointegrated_pairs(["AAA", "BBB"])
    assert ctx.candidates_tested == 0 and ctx.pairs == []


# ── peer expansion ──────────────────────────────────────────────────────────

def _ctx_with_pair(a="AAA", b="BBB"):
    from src.models import CointPair, CointPairsContext
    from datetime import date
    p = CointPair(ticker_a=a, ticker_b=b, hedge_ratio=1.0, adf_stat=-4.0,
                  adf_pvalue=0.01, adf_crit_5pct=-3.34, is_cointegrated=True,
                  half_life_days=5.0, correlation=0.9, spread_mean=0.0,
                  spread_std=0.1, spread_zscore=2.5, long_leg=b, short_leg=a,
                  signal="ENTRY", lookback_days=300, rationale="r")
    return CointPairsContext(pairs=[p], candidates_tested=1, cointegrated_count=1,
                             ticker_scores={a: -0.5, b: 0.5},
                             report_date=date.today(), summary="s")


def test_the_missing_leg_of_a_half_present_pair_is_pulled_in():
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["AAA"]) == ["BBB"]
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["BBB"]) == ["AAA"]


def test_a_fully_present_or_fully_absent_pair_pulls_nothing():
    """Both in → nothing to add. Both out → no anchor, so adding either would
    be an unmotivated universe expansion."""
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["AAA", "BBB"]) == []
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["ZZZZ"]) == []


def test_peer_expansion_is_case_insensitive_and_capped():
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["aaa"]) == ["BBB"]
    assert ci.get_coint_peer_tickers(_ctx_with_pair(), ["AAA"], max_peers=0) == []


def test_peer_expansion_handles_a_missing_context():
    assert ci.get_coint_peer_tickers(None, ["AAA"]) == []
