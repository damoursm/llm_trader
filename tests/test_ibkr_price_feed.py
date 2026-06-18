"""Tracker live-price path prefers IBKR real-time when enabled (A/B flag)."""

import types

import src.broker as broker_pkg
from src.performance import tracker


class _FakeBroker:
    def __init__(self, price):
        self._price = price
        self.connected = True

    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True
        return True

    def get_market_price(self, ticker):
        return self._price


def _force_rth(monkeypatch):
    import src.performance.market_calendar as mc
    monkeypatch.setattr(mc, "current_session", lambda: "rth")


def _stub_yf(monkeypatch, last_price):
    class _FI:
        def __init__(self):
            self.last_price = last_price

    class _T:
        def __init__(self, _t):
            self.fast_info = _FI()

    monkeypatch.setattr(tracker, "yf", types.SimpleNamespace(Ticker=_T))


def test_prefers_ibkr_when_enabled(monkeypatch):
    monkeypatch.setattr(tracker.settings, "enable_fetch_data", True)
    monkeypatch.setattr(tracker.settings, "enable_ibkr_price_feed", True)
    monkeypatch.setattr(broker_pkg, "get_broker", lambda *a, **k: _FakeBroker(123.45))
    _force_rth(monkeypatch)
    tracker.reset_price_health()

    assert tracker._fetch_price("AAPL") == 123.45
    assert tracker.get_price_health()["ibkr"] == 1
    assert tracker.get_price_health()["yfinance"] == 0


def test_falls_back_to_yfinance_when_ibkr_has_no_quote(monkeypatch):
    monkeypatch.setattr(tracker.settings, "enable_fetch_data", True)
    monkeypatch.setattr(tracker.settings, "enable_ibkr_price_feed", True)
    monkeypatch.setattr(broker_pkg, "get_broker", lambda *a, **k: _FakeBroker(None))
    _force_rth(monkeypatch)
    _stub_yf(monkeypatch, 99.0)
    tracker.reset_price_health()

    assert tracker._fetch_price("AAPL") == 99.0
    assert tracker.get_price_health()["ibkr"] == 0
    assert tracker.get_price_health()["yfinance"] == 1


def test_flag_off_never_touches_broker(monkeypatch):
    monkeypatch.setattr(tracker.settings, "enable_fetch_data", True)
    monkeypatch.setattr(tracker.settings, "enable_ibkr_price_feed", False)

    def _boom(*a, **k):
        raise AssertionError("get_broker must not be called when the flag is off")

    monkeypatch.setattr(broker_pkg, "get_broker", _boom)
    _force_rth(monkeypatch)
    _stub_yf(monkeypatch, 50.0)
    tracker.reset_price_health()

    assert tracker._fetch_price("AAPL") == 50.0
    assert tracker.get_price_health()["ibkr"] == 0


def test_base_and_dryrun_market_price_is_none():
    # Default broker surface doesn't support market price (None → caller falls back).
    from src.broker.dryrun import DryRunBroker
    assert DryRunBroker().get_market_price("AAPL") is None
