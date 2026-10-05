"""The 30-minute TICK cache's freshness (runtime review 2026-09-28).

The cache stores COMPLETED bars labelled by their start, so in regular hours its
newest bar is always >= 30 minutes old: judged against the 25-minute TTL it was
never fresh, and every call refetched 120 days (1,205 fetches a tick for ~400
names). What must hold: a cache that already holds the latest completed bar is
served without a fetch; one that lacks it is refetched.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data import market_data as md


class _Fixed(datetime):
    """10:05 ET on a session day: the latest completed bar is 09:30-10:00."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 28, 14, 5, tzinfo=timezone.utc)


def _cache(last_start_utc: str) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(last_start_utc, tz="UTC"), periods=5, freq="30min")
    return pd.DataFrame({"Open": 10.0, "High": 10.1, "Low": 9.9, "Close": 10.0, "Volume": 1e4}, index=idx)


def _env(monkeypatch, cached):
    import src.data.cache as cache
    fetched = []
    monkeypatch.setattr(md, "_datetime", _Fixed)
    monkeypatch.setattr(md, "current_session", lambda now=None: "rth")
    monkeypatch.setattr(md, "is_valid_ticker", lambda t: True)
    monkeypatch.setattr(cache, "load_ohlcv", lambda t, interval="1d": cached)
    monkeypatch.setattr(cache, "save_ohlcv", lambda t, df, interval="1d": None)
    monkeypatch.setattr(md, "_drop_forming_bar", lambda df, interval=None: df)
    monkeypatch.setattr(md, "_fetch_intraday", lambda t, interval: fetched.append(t) or _cache("2026-09-28 13:30"))
    return fetched


def test_the_latest_completed_bar_start():
    at = lambda t: md._latest_completed_30m_start(pd.Timestamp(t, tz="America/New_York").tz_convert("UTC"))  # noqa: E731
    assert at("2026-09-28 09:45") is None                                   # no bar complete yet
    assert at("2026-09-28 10:05").tz_convert("America/New_York").strftime("%H:%M") == "09:30"
    assert at("2026-09-28 16:20").tz_convert("America/New_York").strftime("%H:%M") == "15:30"


def test_a_cache_holding_the_latest_completed_bar_is_served_without_a_fetch(monkeypatch):
    fetched = _env(monkeypatch, _cache("2026-09-28 13:30"))                  # 09:30 ET bar, 35 min old
    out = md._get_intraday_history("XLK", "30m")
    assert fetched == [] and len(out) == 5


def test_a_cache_without_the_latest_completed_bar_is_refetched(monkeypatch):
    fetched = _env(monkeypatch, _cache("2026-09-25 19:30"))                  # Friday's 15:30 ET bar
    md._get_intraday_history("XLK", "30m")
    assert fetched == ["XLK"]
