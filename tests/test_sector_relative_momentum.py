"""Relative momentum (`src/signals/sector_relative_momentum.py`).

Two weighted methods off one engine: `sector_momentum` (vs the sector ETF) and
`market_momentum` (vs SPY). Both measure ticker-minus-benchmark excess return,
z-normalised against the ticker's OWN historical excess distribution.

What actually goes wrong here is alignment, not arithmetic. The two legs come
from different feeds — the daily cache is written by Polygon (naive UTC) and
yfinance (tz-aware) — so a mismatched or duplicated index silently produces an
excess return computed against the wrong bar. The module normalises and dedupes
for exactly that reason, and the dedupe is load-bearing: `.loc[common]` on a
duplicated index EXPANDS the series, which corrupts every downstream std.

The self-comparison guards matter too: benchmarking SPY against SPY, or a
ticker against itself, is a structural zero that must be returned as "no view"
rather than computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals import sector_relative_momentum as srm


def _series(values, start="2024-01-01", tz=None) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz=tz)
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _closes(n=400, start_px=100.0, drift=0.0, noise=0.0, seed=0) -> list:
    """A drifting price path.

    `noise` defaults to 0 (a pure exponential) for the degenerate-input tests,
    but any test that expects a real SCORE must pass some: the normaliser
    divides by the std of the 21-bar excess return, and two noise-free
    exponentials have a CONSTANT excess, so σ≈0 and the method correctly
    abstains. That is right behaviour and a trap for fixtures."""
    rng = np.random.default_rng(seed)
    out, px = [], start_px
    for _ in range(n):
        px *= (1.0 + drift) * (1.0 + rng.normal(0.0, noise) if noise else 1.0)
        out.append(px)
    return out


@pytest.fixture
def feed(monkeypatch):
    """Serve canned Close series by ticker; anything unstubbed returns None."""
    box: dict = {}
    monkeypatch.setattr(srm, "_get_close",
                        lambda tk, interval="1d": box.get((tk, interval), box.get(tk)))
    return box


# ── the sign convention ─────────────────────────────────────────────────────

def test_outperforming_the_benchmark_scores_positive(feed):
    feed["AAA"] = _series(_closes(drift=0.002, noise=0.004, seed=1))
    feed["XLK"] = _series(_closes(drift=0.0005, noise=0.004, seed=2))
    score, r1, r3, bench = srm._compute_relative_momentum("AAA", "XLK")
    assert score > 0 and r1 > 0 and r3 > 0
    assert bench == "XLK"


def test_underperforming_the_benchmark_scores_negative(feed):
    feed["AAA"] = _series(_closes(drift=0.0005, noise=0.004, seed=1))
    feed["XLK"] = _series(_closes(drift=0.002, noise=0.004, seed=2))
    score, r1, r3, _b = srm._compute_relative_momentum("AAA", "XLK")
    assert score < 0 and r1 < 0 and r3 < 0


def test_a_rising_stock_in_a_faster_sector_still_scores_negative(feed):
    """The whole point of stripping beta: absolute momentum would call this
    bullish, relative momentum correctly calls it a laggard."""
    feed["AAA"] = _series(_closes(drift=0.001, noise=0.004, seed=1))   # rising
    feed["XLK"] = _series(_closes(drift=0.003, noise=0.004, seed=2))   # faster
    score, r1, _r3, _b = srm._compute_relative_momentum("AAA", "XLK")
    assert r1 < 0 and score < 0


def test_score_is_bounded_and_a_matched_pair_scores_zero(feed):
    feed["AAA"] = _series(_closes(drift=0.002))
    feed["XLK"] = _series(_closes(drift=0.002))       # identical path
    score, r1, r3, _b = srm._compute_relative_momentum("AAA", "XLK")
    assert score == 0.0 and r1 == 0.0 and r3 == 0.0

    feed["AAA"] = _series(_closes(drift=0.05, noise=0.004, seed=3))   # violent
    score, _r1, _r3, _b = srm._compute_relative_momentum("AAA", "XLK")
    assert -1.0 <= score <= 1.0


def test_volatile_names_are_normalised_by_their_own_distribution(feed):
    """The same raw excess return must score LOWER for a name whose excess
    swings that much routinely — otherwise the method just ranks volatility."""
    n = 400
    feed["BENCH"] = _series(_closes(n=n, drift=0.0005, noise=0.002, seed=9))
    feed["CALM"] = _series(_closes(n=n, drift=0.0012, noise=0.002, seed=1))
    feed["NOISY"] = _series(_closes(n=n, drift=0.0012, noise=0.030, seed=2))
    calm_score = abs(srm._compute_relative_momentum("CALM", "BENCH")[0])
    noisy_score = abs(srm._compute_relative_momentum("NOISY", "BENCH")[0])
    assert calm_score > noisy_score


# ── self-comparison guards ──────────────────────────────────────────────────

def test_a_ticker_is_never_benchmarked_against_itself(feed):
    feed["XLK"] = _series(_closes(drift=0.002))
    assert srm._compute_relative_momentum("XLK", "XLK") == (0.0, 0.0, 0.0, "")
    assert srm._compute_relative_momentum("xlk", "XLK") == (0.0, 0.0, 0.0, "")


def test_an_empty_benchmark_is_no_view(feed):
    assert srm._compute_relative_momentum("AAA", "") == (0.0, 0.0, 0.0, "")
    assert srm._compute_relative_momentum("AAA", None) == (0.0, 0.0, 0.0, "")


def test_market_momentum_declines_to_score_spy(feed):
    feed["SPY"] = _series(_closes(drift=0.002))
    assert srm.compute_market_relative_momentum_score("SPY") == (0.0, 0.0, 0.0, "")
    assert srm.compute_market_relative_momentum_score("spy") == (0.0, 0.0, 0.0, "")


def test_sector_momentum_abstains_when_no_sector_applies(monkeypatch, feed):
    """Commodities have no equity sector — the resolver returns None and the
    method must return an empty benchmark, not compare against ''."""
    monkeypatch.setattr(srm, "get_sector_benchmark", lambda t, asset_type=None: None)
    assert srm.compute_sector_relative_momentum_score("GLD") == (0.0, 0.0, 0.0, "")


def test_sector_momentum_uses_the_resolved_benchmark(monkeypatch, feed):
    monkeypatch.setattr(srm, "get_sector_benchmark", lambda t, asset_type=None: "XLV")
    feed["AAA"] = _series(_closes(drift=0.002, noise=0.004, seed=1))
    feed["XLV"] = _series(_closes(drift=0.0005, noise=0.004, seed=2))
    score, _r1, _r3, bench = srm.compute_sector_relative_momentum_score("AAA")
    assert bench == "XLV" and score > 0


def test_market_momentum_always_reports_spy(feed):
    feed["AAA"] = _series(_closes(drift=0.002, noise=0.004, seed=1))
    feed["SPY"] = _series(_closes(drift=0.0005, noise=0.004, seed=2))
    assert srm.compute_market_relative_momentum_score("AAA")[3] == "SPY"


# ── index alignment ─────────────────────────────────────────────────────────

def test_align_intersects_on_normalised_dates():
    a = _series([1.0, 2.0, 3.0], start="2026-01-01")
    b = _series([10.0, 20.0], start="2026-01-02")
    ta, tb = srm._align(a, b)
    assert len(ta) == len(tb) == 2
    assert list(ta.index) == list(tb.index)


def test_align_reconciles_tz_aware_and_naive_indexes():
    """Polygon writes naive UTC, yfinance writes tz-aware. Without
    normalisation the intersection is EMPTY and the method silently abstains on
    every ticker whose two legs came from different feeds."""
    naive = _series([1.0, 2.0, 3.0], start="2026-01-01")
    aware = _series([10.0, 20.0, 30.0], start="2026-01-01", tz="America/New_York")
    ta, tb = srm._align(naive, aware)
    assert len(ta) == 3 and len(tb) == 3


def test_align_dedupes_so_the_intersection_cannot_expand():
    """`.loc[common]` on a duplicated index multiplies rows — a 3-row series can
    come back with 5 rows, which corrupts every std and pct_change after it."""
    idx = pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-02"])
    dupe = pd.Series([1.0, 99.0, 2.0], index=idx)
    clean = _series([10.0, 20.0], start="2026-01-01")
    ta, tb = srm._align(dupe, clean)
    assert len(ta) == len(tb) == 2
    assert ta.iloc[0] == 99.0, "dedupe must keep the LAST row for the date"


def test_no_overlap_yields_no_view(feed):
    feed["AAA"] = _series(_closes(noise=0.004, seed=1), start="2020-01-01")
    feed["XLK"] = _series(_closes(noise=0.004, seed=2), start="2024-01-01")
    assert srm._compute_relative_momentum("AAA", "XLK")[0] == 0.0


# ── evidence floors ─────────────────────────────────────────────────────────

def test_missing_history_on_either_leg_is_no_view(feed):
    feed["AAA"] = _series(_closes())
    assert srm._compute_relative_momentum("AAA", "XLK") == (0.0, 0.0, 0.0, "XLK")
    feed.clear()
    feed["XLK"] = _series(_closes())
    assert srm._compute_relative_momentum("AAA", "XLK") == (0.0, 0.0, 0.0, "XLK")


def test_too_little_overlap_for_the_medium_lookback_is_no_view(feed):
    n = srm._MOM_MED + 1
    feed["AAA"] = _series(_closes(n=n, drift=0.002, noise=0.004, seed=1))
    feed["XLK"] = _series(_closes(n=n, drift=0.0005, noise=0.004, seed=2))
    assert srm._compute_relative_momentum("AAA", "XLK")[0] == 0.0


def test_a_thin_normalisation_distribution_is_no_view(feed):
    """Fewer than 30 excess observations cannot give a usable σ; dividing by it
    manufactures a large z from noise."""
    # Past the _MOM_MED+2 overlap gate, but `excess_21` loses its first 21 rows
    # to the pct_change window, so fewer than 30 observations survive.
    n = srm._MOM_MED + 6
    assert srm._MOM_MED + 2 <= n and (n - srm._MOM_SHORT) < 30
    feed["AAA"] = _series(_closes(n=n, drift=0.002, noise=0.004, seed=1))
    feed["XLK"] = _series(_closes(n=n, drift=0.0005, noise=0.004, seed=2))
    assert srm._compute_relative_momentum("AAA", "XLK")[0] == 0.0


def test_a_degenerate_distribution_is_no_view(feed):
    """Two perfectly parallel series have zero excess variance — σ≈0 would make
    every z infinite."""
    feed["AAA"] = _series(_closes(drift=0.002))
    feed["XLK"] = _series(_closes(drift=0.002))
    assert srm._compute_relative_momentum("AAA", "XLK")[0] == 0.0


# ── the loader ──────────────────────────────────────────────────────────────

def test_get_close_drops_non_positive_prices(monkeypatch):
    """A zero or negative close makes pct_change meaningless; such rows are
    dropped rather than allowed to poison the distribution."""
    df = pd.DataFrame({"Close": [10.0, 0.0, -5.0] + [10.0] * 60},
                      index=pd.date_range("2026-01-01", periods=63, freq="D"))
    monkeypatch.setattr(srm, "load_ohlcv", lambda tk: df)
    s = srm._get_close("AAA")
    assert s is not None and (s > 0).all() and len(s) == 61


def test_get_close_returns_none_below_the_row_floor(monkeypatch):
    df = pd.DataFrame({"Close": [10.0] * (srm._MIN_ROWS - 1)},
                      index=pd.date_range("2026-01-01", periods=srm._MIN_ROWS - 1))
    monkeypatch.setattr(srm, "load_ohlcv", lambda tk: df)
    monkeypatch.setattr(srm, "get_history", lambda *a, **k: df)
    assert srm._get_close("AAA") is None


def test_get_close_is_fail_soft(monkeypatch):
    monkeypatch.setattr(srm, "load_ohlcv",
                        lambda tk: (_ for _ in ()).throw(RuntimeError("cache gone")))
    assert srm._get_close("AAA") is None


def test_get_close_rejects_an_unknown_interval():
    assert srm._get_close("AAA", interval="7y") is None


def test_both_legs_are_pulled_at_the_same_interval(monkeypatch):
    """An apples-to-oranges residual (daily ticker vs weekly benchmark) would
    still produce a plausible number."""
    seen = []
    monkeypatch.setattr(srm, "_get_close",
                        lambda tk, interval="1d": seen.append((tk, interval)) or None)
    srm._compute_relative_momentum("AAA", "XLK", interval="1w")
    assert seen == [("AAA", "1w"), ("XLK", "1w")]
