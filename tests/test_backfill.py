"""Universe OHLCV backfill utility (Massive/Polygon cache warm)."""

import pandas as pd
import pytest

import src.data.backfill as bf


def test_period_for_days():
    assert bf._period_for_days(2000) == "5y"
    assert bf._period_for_days(730) == "2y"
    assert bf._period_for_days(100) == "3mo"
    assert bf._period_for_days(5) == "1mo"


def test_gather_universe_merges_validates_sorts(monkeypatch):
    # stocks_list / sectors_list / … are read-only properties → set the CSV fields.
    monkeypatch.setattr(bf.settings, "stock_watchlist", "AAPL,MSFT")
    monkeypatch.setattr(bf.settings, "sector_etfs", "XLK")
    monkeypatch.setattr(bf.settings, "commodity_etfs", "GLD")
    monkeypatch.setattr(bf.settings, "enable_factor_etfs", False)          # factor_list → []
    monkeypatch.setattr(bf.settings, "enable_hypothetical_trades", False)  # hypothetical → []
    monkeypatch.setattr("src.performance.tracker.get_open_trade_tickers", lambda: ["NVDA"])
    monkeypatch.setattr(bf, "_recent_signal_tickers", lambda days: ["AAPL", "AMD", "N/A"])  # dup + junk

    out = bf.gather_universe(signal_days=30)
    assert {"AAPL", "MSFT", "XLK", "GLD", "NVDA", "AMD"} <= set(out)
    assert "N/A" not in out               # junk dropped by is_valid_ticker
    assert out == sorted(out)             # sorted
    assert len(out) == len(set(out))      # deduped


def test_backfill_counts_and_force_refresh(monkeypatch):
    monkeypatch.setattr(bf.settings, "enable_fetch_data", True)
    monkeypatch.setattr(bf, "gather_universe", lambda signal_days: ["AAPL", "MSFT", "BADX"])
    calls = []

    def fake_hist(tk, period=None, force_refresh=False, interval="1d"):
        calls.append((tk, interval, force_refresh))
        return pd.DataFrame() if tk == "BADX" else pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(bf, "get_history", fake_hist)
    res = bf.backfill(days=730, with_30m=False)
    assert res == {"total": 3, "daily": 2, "intraday": 0}   # BADX empty → not counted
    assert all(fr is True for _, _, fr in calls)            # force_refresh on
    assert all(iv == "1d" for _, iv, _ in calls)


def test_backfill_with_30m(monkeypatch):
    monkeypatch.setattr(bf.settings, "enable_fetch_data", True)
    monkeypatch.setattr(bf, "gather_universe", lambda signal_days: ["AAPL"])
    monkeypatch.setattr(bf, "get_history",
                        lambda tk, period=None, force_refresh=False, interval="1d": pd.DataFrame({"Close": [1.0]}))
    assert bf.backfill(with_30m=True) == {"total": 1, "daily": 1, "intraday": 1}


def test_backfill_skip_daily(monkeypatch):
    monkeypatch.setattr(bf.settings, "enable_fetch_data", True)
    monkeypatch.setattr(bf, "gather_universe", lambda signal_days: ["AAPL", "MSFT"])
    intervals = []

    def fake_hist(tk, period=None, force_refresh=False, interval="1d"):
        intervals.append(interval)
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(bf, "get_history", fake_hist)
    res = bf.backfill(with_30m=True, daily=False)
    assert res == {"total": 2, "daily": 0, "intraday": 2}
    assert intervals == ["30m", "30m"]        # daily pass skipped


def test_backfill_disabled(monkeypatch):
    monkeypatch.setattr(bf.settings, "enable_fetch_data", False)
    monkeypatch.setattr(bf, "get_history", lambda *a, **k: pytest.fail("must not fetch when disabled"))
    assert bf.backfill() == {"total": 0, "daily": 0, "intraday": 0}
