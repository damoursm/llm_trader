"""Massive/Polygon server-side technical-indicator method (`massive`)."""

import src.data.polygon_client as pc
import src.signals.massive_tech as mt


# ── polygon_client indicator fetchers ──────────────────────────────────────────

def test_get_rsi(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    monkeypatch.setattr(pc, "_get",
                        lambda path, params=None: {"results": {"values": [{"timestamp": 1, "value": 55.5}]}})
    assert pc.get_rsi("AAPL") == 55.5


def test_get_macd(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    monkeypatch.setattr(pc, "_get",
                        lambda path, params=None: {"results": {"values": [{"value": 1.1, "signal": 2.0, "histogram": -0.9}]}})
    assert pc.get_macd("AAPL") == {"value": 1.1, "signal": 2.0, "histogram": -0.9}


def test_get_rsi_no_key(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "")
    assert pc.get_rsi("AAPL") is None


# ── compute_massive_tech_score ─────────────────────────────────────────────────

def _enable(monkeypatch):
    monkeypatch.setattr(mt.settings, "enable_massive_tech", True)
    monkeypatch.setattr(mt.polygon_client, "is_available", lambda: True)


def test_score_bullish(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(mt.polygon_client, "get_rsi", lambda t, window=14: 80.0)
    monkeypatch.setattr(mt.polygon_client, "get_macd",
                        lambda t: {"value": 1.0, "signal": 0.0, "histogram": 1.0})
    s = mt.compute_massive_tech_score("AAPL")
    assert 0.95 <= s <= 1.0          # rsi 1.0 + macd tanh(2)=0.96 → ~0.98


def test_score_bearish(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(mt.polygon_client, "get_rsi", lambda t, window=14: 20.0)
    monkeypatch.setattr(mt.polygon_client, "get_macd",
                        lambda t: {"value": 0.0, "signal": 0.0, "histogram": -1.0})
    assert mt.compute_massive_tech_score("AAPL") == -1.0


def test_score_rsi_only(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(mt.polygon_client, "get_rsi", lambda t, window=14: 65.0)
    monkeypatch.setattr(mt.polygon_client, "get_macd", lambda t: None)
    assert mt.compute_massive_tech_score("AAPL") == round((65 - 50) / 30, 3)  # 0.5


def test_disabled_returns_zero(monkeypatch):
    monkeypatch.setattr(mt.settings, "enable_massive_tech", False)
    monkeypatch.setattr(mt.polygon_client, "get_rsi", lambda t, window=14: 1 / 0)  # must not be called
    assert mt.compute_massive_tech_score("AAPL") == 0.0


def test_unavailable_returns_zero(monkeypatch):
    monkeypatch.setattr(mt.settings, "enable_massive_tech", True)
    monkeypatch.setattr(mt.polygon_client, "is_available", lambda: False)
    assert mt.compute_massive_tech_score("AAPL") == 0.0


def test_both_indicators_missing(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(mt.polygon_client, "get_rsi", lambda t, window=14: None)
    monkeypatch.setattr(mt.polygon_client, "get_macd", lambda t: None)
    assert mt.compute_massive_tech_score("AAPL") == 0.0


def test_invalid_ticker(monkeypatch):
    _enable(monkeypatch)
    assert mt.compute_massive_tech_score("N/A") == 0.0
