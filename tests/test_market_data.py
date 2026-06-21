"""Tests for src.data.market_data — refresh merge, multi-timeframe resample,
forming-bar drops, and the fail-soft 30-minute fetch."""

import pandas as pd
import pytest

import src.data.market_data as md
from src.data.market_data import _merge_ohlcv


def _bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes,
         "Close": closes, "Volume": [1] * len(closes)},
        index=pd.to_datetime(dates),
    )


def _ohlcv(dates, o, h, l, c, v) -> pd.DataFrame:
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v},
                        index=pd.to_datetime(dates))


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


def test_get_snapshots_grouped_close_fallback(monkeypatch):
    """When the live snapshot (Polygon 403) + yfinance leave tickers uncovered,
    the deterministic grouped-daily close fills them, marked prev_close. Class
    shares are normalized (BRK-B <-> BRK.B)."""
    import src.data.market_data as md
    monkeypatch.setattr(md.settings, "enable_fetch_data", True)
    monkeypatch.setattr(md.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(md.polygon_client, "get_snapshots_batch", lambda tickers: {})   # 403 -> empty
    monkeypatch.setattr(md, "_fetch_ticker_yf", lambda t: (None, None))                 # yfinance covers nothing
    monkeypatch.setattr(md.polygon_client, "get_grouped_daily_closes",
                        lambda: {"AAA": 100.0, "BRK.B": 50.0})

    snaps = {s.ticker: s for s in md.get_snapshots(["AAA", "BRK-B", "ZZZ"])}
    assert snaps["AAA"].price == 100.0 and snaps["AAA"].price_source == "prev_close"
    assert snaps["BRK-B"].price == 50.0                     # dot/dash class-share normalization
    assert "ZZZ" not in snaps                               # absent from grouped -> still no price


# ── Weekly resample (daily → W-FRI) ────────────────────────────────────────

def test_resample_weekly_aggregates_ohlcv():
    dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",   # → Fri 06-05
             "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"]   # → Fri 06-12
    o = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    h = [15, 16, 17, 18, 19, 25, 26, 27, 28, 29]
    l = [5, 4, 3, 2, 1, 9, 8, 7, 6, 5]
    c = [10, 11, 12, 13, 18, 20, 21, 22, 23, 28]
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    w = md._resample_weekly(_ohlcv(dates, o, h, l, c, v))
    assert len(w) == 2
    wk1 = w.loc[pd.to_datetime("2026-06-05")]
    assert (wk1["Open"], wk1["High"], wk1["Low"], wk1["Close"], wk1["Volume"]) == (10, 19, 1, 18, 15)
    wk2 = w.loc[pd.to_datetime("2026-06-12")]
    assert (wk2["Open"], wk2["High"], wk2["Low"], wk2["Close"], wk2["Volume"]) == (20, 29, 5, 28, 40)


# ── Forming-bar drops ───────────────────────────────────────────────────────

def test_drop_forming_weekly_keeps_past_drops_future():
    # Weekly bars are labelled by their Friday; a future-Friday week is still forming.
    idx = pd.to_datetime(["2020-01-03", "2020-01-10", "2099-01-02"])
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    out = md._drop_forming_bar(df, interval="1w")
    assert list(out.index) == list(pd.to_datetime(["2020-01-03", "2020-01-10"]))


def test_drop_forming_30m_drops_unelapsed_window():
    now = md._datetime.now(md._timezone.utc).replace(tzinfo=None)   # tz-naive UTC (cache convention)
    idx = pd.to_datetime([now - pd.Timedelta(hours=2),       # completed
                          now - pd.Timedelta(minutes=45),    # completed (window elapsed)
                          now - pd.Timedelta(minutes=10)])   # window ends in the future → forming
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    out = md._drop_forming_bar(df, interval="30m")
    assert len(out) == 2
    assert 3.0 not in out["Close"].tolist()


# ── Intraday merge (dedupe by exact timestamp) ──────────────────────────────

def test_merge_intraday_dedupes_by_timestamp_fresh_wins():
    cached = pd.DataFrame({"Close": [1.0, 2.0]},
                          index=pd.to_datetime(["2026-06-01 13:30", "2026-06-01 14:00"]))
    fresh = pd.DataFrame({"Close": [20.0, 30.0]},
                         index=pd.to_datetime(["2026-06-01 14:00", "2026-06-01 14:30"]))
    merged = md._merge_intraday(cached, fresh)
    assert len(merged) == 3                                          # 13:30, 14:00, 14:30
    assert merged.loc[pd.to_datetime("2026-06-01 14:00"), "Close"] == 20.0   # fresh wins on overlap


# ── 30-minute fetch is fail-soft (never aborts the tick) ────────────────────

def test_fetch_intraday_yf_failsoft_on_rate_limit(monkeypatch):
    class _T:
        def history(self, *a, **k):
            raise Exception("YFRateLimitError: 429 Too Many Requests")
    monkeypatch.setattr(md.yf, "Ticker", lambda t: _T())
    out = md._fetch_intraday_yf("AAA", "30m")
    assert out.empty                                                # returns empty, does not raise


def test_get_intraday_history_failsoft_returns_empty(monkeypatch):
    monkeypatch.setattr(md.settings, "enable_fetch_data", True)
    monkeypatch.setattr(md, "_fetch_intraday_yf", lambda t, i="30m": pd.DataFrame())
    out = md.get_history("ZXCV", interval="30m")                    # valid format, no cache file
    assert out is not None and out.empty
