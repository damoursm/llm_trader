"""Ticker-events watch — recent renames/delistings surfaced as material NewsArticles."""

from datetime import date, timedelta

import src.data.ticker_events as te


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(te, "_cache_path", lambda: tmp_path / "te.json")
    monkeypatch.setattr(te.settings, "enable_ticker_events", True)
    monkeypatch.setattr(te.settings, "enable_fetch_data", True)
    monkeypatch.setattr(te.settings, "ticker_events_lookback_days", 45)
    monkeypatch.setattr(te.polygon_client, "is_available", lambda: True)


def test_recent_rename_surfaced_old_ignored(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    recent = (date.today() - timedelta(days=5)).isoformat()
    old = (date.today() - timedelta(days=400)).isoformat()
    graph = {"FOO": [{"type": "ticker_change", "date": recent}],
             "BAR": [{"type": "ticker_change", "date": old}]}
    monkeypatch.setattr(te.polygon_client, "get_ticker_events", lambda tk: graph.get(tk, []))

    arts = te.fetch_ticker_events(["FOO", "BAR", "AAPL"])
    assert len(arts) == 1                              # BAR's old event + AAPL (none) excluded
    assert arts[0].tickers == ["FOO"]
    assert arts[0].provider_insights == {"FOO": "neutral"}   # rename = neutral (skips LLM scorer)
    assert (tmp_path / "te.json").exists()


def test_delisting_is_negative(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    recent = (date.today() - timedelta(days=3)).isoformat()
    monkeypatch.setattr(te.polygon_client, "get_ticker_events",
                        lambda tk: [{"type": "delisted", "date": recent}])
    arts = te.fetch_ticker_events(["ZZ"])
    assert len(arts) == 1 and arts[0].provider_insights == {"ZZ": "negative"}


def test_disabled_returns_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(te.settings, "enable_ticker_events", False)
    assert te.fetch_ticker_events(["FOO"]) == []
