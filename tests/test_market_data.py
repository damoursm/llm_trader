"""Tests for src.data.market_data._merge_ohlcv — data-loss prevention on refresh."""

import pandas as pd
import pytest

from src.data.market_data import _merge_ohlcv


def _bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes,
         "Close": closes, "Volume": [1] * len(closes)},
        index=pd.to_datetime(dates),
    )


def test_merge_keeps_history_outside_fresh_window():
    """A force-refresh that returns only the last 3mo must not erase pre-3mo bars."""
    old = _bars(
        ["2025-11-01", "2025-12-15", "2026-01-15", "2026-02-15", "2026-03-15"],
        [10.0, 11.0, 12.0, 13.0, 14.0],
    )
    fresh = _bars(
        ["2026-03-15", "2026-04-15", "2026-05-15"],
        [15.0, 16.0, 17.0],   # different close on overlap day = split rescale
    )
    merged = _merge_ohlcv(old, fresh)

    # 4 retained old bars (Nov, Dec, Jan, Feb) + 3 fresh = 7 rows.
    assert len(merged) == 7
    # Fresh wins on the overlapping date — Mar 15 close is 15.0, not 14.0.
    assert merged.loc[pd.to_datetime("2026-03-15"), "Close"] == 15.0
    # Pre-fresh-window history retained.
    assert merged.loc[pd.to_datetime("2025-11-01"), "Close"] == 10.0


def test_merge_empty_cache_returns_fresh():
    fresh = _bars(["2026-05-01"], [100.0])
    assert _merge_ohlcv(None, fresh).equals(fresh)
    assert _merge_ohlcv(pd.DataFrame(), fresh).equals(fresh)


def test_merge_empty_fresh_returns_cached():
    """When the fetch yields nothing, the existing cache must survive."""
    old = _bars(["2026-04-01", "2026-05-01"], [100.0, 110.0])
    result = _merge_ohlcv(old, pd.DataFrame())
    assert result.equals(old)


def test_merge_no_overlap_just_concatenates_and_sorts():
    old   = _bars(["2026-01-15", "2026-02-15"], [10.0, 11.0])
    fresh = _bars(["2026-04-15", "2026-03-15"], [13.0, 12.0])   # unsorted fresh
    merged = _merge_ohlcv(old, fresh)
    assert list(merged.index) == sorted(merged.index)
    assert len(merged) == 4
