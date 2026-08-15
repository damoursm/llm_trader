"""Opportunity screener (`src/data/screener.py`).

A discovery source: everything it emits enters the universe, gets scored and
lands in the signals panel. So the two things worth pinning are the LIQUIDITY
gate (it runs before any screen, and a leak here puts illiquid names into the
run) and the `net` direction arithmetic — a hit whose `direction` disagrees with
its own screens is a coherent-looking recommendation pointing the wrong way.

All frames are synthetic. The real path reads `cache/ohlcv/` and falls back to a
network fetch, neither of which belongs in a unit test; `_screen_one` is pure
given a frame, which is where the logic actually lives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.data import screener as sc


def _frame(n: int = 300, price: float = 50.0, vol: float = 2_000_000.0,
           trend: float = 0.0, closes=None, hl_spread: float = 0.001) -> pd.DataFrame:
    """A flat-by-default OHLCV frame; `trend` is a per-bar drift in price units.

    `hl_spread=0` collapses High/Low onto Close. That matters for the 52-week
    screens: with an intrabar premium, a rising CLOSE never exceeds the previous
    bar's HIGH, so a breakout can't be constructed — the screen compares today's
    close against the prior window's highs.
    """
    if closes is None:
        closes = [price + trend * i for i in range(n)]
    close = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(end="2026-08-14", periods=len(close))
    return pd.DataFrame(
        {"Open": close, "High": close * (1 + hl_spread), "Low": close * (1 - hl_spread),
         "Close": close, "Volume": np.full(len(close), vol, dtype=float)},
        index=idx)


def _midrange_frame(n: int = 300) -> pd.DataFrame:
    """Liquid, quiet, and sitting in the MIDDLE of its range — the genuine
    no-setup case. A flat line does not qualify: its last close is within 2% of
    its own 52-week high, so `NEAR_52W_HIGH` correctly fires. One old spike puts
    a real ceiling overhead without disturbing the moving averages enough to
    cross."""
    closes = [50.0] * n
    closes[n // 3] = 60.0
    return _frame(closes=closes)


# ── the liquidity gate runs first ───────────────────────────────────────────

def test_penny_prices_are_gated_out():
    df = _frame(price=settings.screen_min_price - 0.01, vol=1e9)
    assert sc._screen_one("ZZZZ", df, spy_ret=0.0) is None


def test_thin_dollar_volume_is_gated_out():
    """Below the dollar-volume floor even a textbook setup must not surface —
    the screener is a discovery source, and an illiquid hit is a cost the rest
    of the pipeline pays."""
    price = 50.0
    thin = (settings.screen_min_dollar_volume / price) * 0.5
    df = _frame(price=price, vol=thin)
    df.iloc[-1, df.columns.get_loc("Volume")] = thin * 10   # a volume spike
    assert sc._screen_one("ZZZZ", df, spy_ret=0.0) is None


def test_short_history_is_skipped():
    assert sc._screen_one("ZZZZ", _frame(n=sc._MIN_BARS - 1), spy_ret=0.0) is None


def test_a_liquid_but_featureless_name_produces_no_hit():
    """No screen fired -> None, not an empty hit. A zero-screen ScreenHit would
    rank alongside real setups and dilute the top-N."""
    assert sc._screen_one("ZZZZ", _midrange_frame(), spy_ret=0.0) is None


# ── individual screens ──────────────────────────────────────────────────────

def test_unusual_volume_fires_and_takes_the_day_direction():
    """The volume surge itself is direction-less; the day's close decides."""
    df = _midrange_frame()
    n = len(df)
    df.iloc[n - 1, df.columns.get_loc("Volume")] *= settings.screen_volume_ratio + 1
    df.iloc[n - 1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-2]) + 1.0
    up = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert up is not None and "UNUSUAL_VOLUME" in up.screens
    assert up.direction == "BULLISH"
    assert up.vol_ratio >= settings.screen_volume_ratio

    df.iloc[n - 1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-2]) - 1.0
    down = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert down.direction == "BEARISH"


def test_volume_baseline_excludes_the_current_bar():
    """The ratio compares today against the PRIOR 20 days. Including today in
    its own baseline would damp exactly the spike the screen looks for."""
    df = _midrange_frame()
    n = len(df)
    base = float(df["Volume"].iloc[-2])
    df.iloc[n - 1, df.columns.get_loc("Volume")] = base * 3.0
    hit = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert hit.vol_ratio == pytest.approx(3.0, abs=0.01)


def test_new_52w_high_and_near_high_are_mutually_exclusive():
    """A close ABOVE the prior high is a breakout; just below it is a setup.
    Emitting both would double-count one observation in `net`."""
    df = _frame(trend=0.05, hl_spread=0.0)        # steadily rising -> at its high
    hit = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert hit is not None
    assert "NEW_52W_HIGH" in hit.screens
    assert "NEAR_52W_HIGH" not in hit.screens
    assert hit.direction == "BULLISH"


def test_new_52w_low_is_flagged_bearish():
    df = _frame(price=200.0, trend=-0.4, hl_spread=0.0)   # falling -> at its low
    hit = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert hit is not None and "NEW_52W_LOW" in hit.screens
    assert hit.direction == "BEARISH"


def test_relative_strength_needs_a_benchmark_return():
    """`spy_ret=None` (benchmark unavailable) must SKIP the RS screen, not treat
    the benchmark as flat — that would score every rising name as strong RS."""
    df = _frame(trend=0.30)
    hit = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert hit is not None
    assert "STRONG_RS" not in hit.screens and "WEAK_RS" not in hit.screens
    assert hit.rs_excess_pct == 0.0


def test_strong_and_weak_relative_strength():
    df = _frame(trend=0.30)                        # a large positive move
    lb = settings.screen_rs_lookback_days
    own = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-(lb + 1)]) - 1) * 100

    strong = sc._screen_one("ZZZZ", df, spy_ret=own - settings.screen_rs_threshold_pct - 1)
    assert "STRONG_RS" in strong.screens and strong.rs_excess_pct > 0

    weak = sc._screen_one("ZZZZ", df, spy_ret=own + settings.screen_rs_threshold_pct + 1)
    assert "WEAK_RS" in weak.screens and weak.rs_excess_pct < 0


def test_golden_cross_requires_a_fresh_crossing():
    """The screen is a CROSS, not a state: a name that has been above its 200d
    for a year must not report a golden cross every single day."""
    long_uptrend = _frame(n=300, trend=0.20, hl_spread=0.0)
    hit = sc._screen_one("ZZZZ", long_uptrend, spy_ret=None)
    assert hit is not None and "GOLDEN_CROSS" not in hit.screens


def test_direction_follows_the_net_vote():
    """`net` sums the screens' signs; a tie resolves BULLISH by construction.
    Pinned because a silent flip here mislabels every mixed setup."""
    # A falling name at new lows AND on a volume spike closing down -> net -2.
    df = _frame(price=200.0, trend=-0.4, hl_spread=0.0)
    n = len(df)
    df.iloc[n - 1, df.columns.get_loc("Volume")] *= settings.screen_volume_ratio + 1
    hit = sc._screen_one("ZZZZ", df, spy_ret=None)
    assert hit.direction == "BEARISH"
    assert {"NEW_52W_LOW", "UNUSUAL_VOLUME"} <= set(hit.screens)


# ── candidate pool hygiene ──────────────────────────────────────────────────

def test_pool_excludes_the_benchmark_and_non_equity_symbols(monkeypatch, tmp_path):
    """Indices (^VIX), futures (GC=F) and crypto (BTC-USD) have no meaningful
    52-week-high or dollar-volume reading, and SPY is the benchmark itself."""
    for name in ("^VIX", "GC=F", "BTC-USD", "SPY", "TOOLONG", "AAPL", "MSFT"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sc, "OHLCV_DIR", tmp_path)
    pool = sc._candidate_pool()
    assert "AAPL" in pool and "MSFT" in pool
    for bad in ("^VIX", "GC=F", "BTC-USD", "SPY", "TOOLONG"):
        assert bad not in pool


def test_pool_is_sorted_and_deduplicated(monkeypatch, tmp_path):
    (tmp_path / "AAPL.json").write_text("{}", encoding="utf-8")   # also in the curated list
    monkeypatch.setattr(sc, "OHLCV_DIR", tmp_path)
    pool = sc._candidate_pool()
    assert pool == sorted(set(pool))


def test_pool_survives_an_unreadable_cache_dir(monkeypatch):
    """The curated universe must still screen when the cache scan fails."""
    monkeypatch.setattr(sc, "OHLCV_DIR", pytest.importorskip("pathlib").Path("\0bad"))
    assert sc._candidate_pool(), "curated universe lost when the cache scan failed"


# ── the public entry point ──────────────────────────────────────────────────

def test_disabled_returns_an_empty_context(monkeypatch):
    monkeypatch.setattr(settings, "enable_opportunity_screener", False)
    ctx = sc.run_screener()
    assert ctx.hits == [] and "disabled" in ctx.summary


def test_run_screener_is_fail_graceful(monkeypatch):
    """It runs inside the universe-construction step; raising would take the
    whole tick down over a discovery nicety."""
    monkeypatch.setattr(settings, "enable_opportunity_screener", True)
    monkeypatch.setattr(sc, "_candidate_pool",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    ctx = sc.run_screener()
    assert ctx.hits == [] and "error" in ctx.summary.lower()


def test_run_screener_ranks_and_caps(monkeypatch):
    """Most screens first, then RS magnitude, then volume surge — and never more
    than `screen_max_results` names."""
    monkeypatch.setattr(settings, "enable_opportunity_screener", True)
    monkeypatch.setattr(settings, "screen_max_fetch_per_run", 0)

    strong = _frame(trend=0.30, hl_spread=0.0)        # breakout + strong RS
    n = len(strong)
    strong.iloc[n - 1, strong.columns.get_loc("Volume")] *= 5
    plain = _frame(trend=0.05, hl_spread=0.0)         # breakout only

    tickers = [f"T{i:02d}" for i in range(settings.screen_max_results + 5)]
    monkeypatch.setattr(sc, "_candidate_pool", lambda: tickers)
    monkeypatch.setattr(sc, "_load_for_screen",
                        lambda tk, budget: strong if tk == "T00" else plain)

    ctx = sc.run_screener()
    assert len(ctx.hits) <= settings.screen_max_results
    assert ctx.hits[0].ticker == "T00", "the richest setup did not rank first"
    counts = [len(h.screens) for h in ctx.hits]
    assert counts == sorted(counts, reverse=True)
    assert ctx.universe_size == len(tickers)


def test_run_screener_skips_frames_missing_required_columns(monkeypatch):
    monkeypatch.setattr(settings, "enable_opportunity_screener", True)
    monkeypatch.setattr(sc, "_candidate_pool", lambda: ["ZZZZ"])
    monkeypatch.setattr(sc, "_load_for_screen",
                        lambda tk, budget: _frame().drop(columns=["Volume"]))
    ctx = sc.run_screener()
    assert ctx.hits == [] and ctx.universe_size == 0
