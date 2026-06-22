"""Alpaca market-data client: SIP intraday bars + the 30-min preference wiring.

Verifies key-gated availability, symbol mapping (BRK-B → BRK.B), timeframe mapping,
RTH-only filtering of 30-min bars, pagination, the warn-once 403, and that
market_data prefers Alpaca over yfinance when configured (falling back when empty).
"""

import io

import httpx
import pandas as pd
import pytest
from loguru import logger

import src.data.alpaca_client as ac
import src.data.market_data as md


class _FakeResp:
    """An httpx-like response: raise_for_status() honours `status`, json() returns `payload`."""

    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            req = httpx.Request("GET", "https://data.alpaca.markets/x")
            resp = httpx.Response(self._status, request=req)
            raise httpx.HTTPStatusError(f"HTTP {self._status}", request=req, response=resp)

    def json(self):
        return self._payload


def _enable(monkeypatch, feed="sip"):
    monkeypatch.setattr(ac.settings, "alpaca_api_key", "k")
    monkeypatch.setattr(ac.settings, "alpaca_api_secret", "s")
    monkeypatch.setattr(ac.settings, "enable_alpaca_intraday", True)
    monkeypatch.setattr(ac.settings, "alpaca_data_feed", feed)


def _bar(t, c=1.0):
    return {"t": t, "o": c, "h": c, "l": c, "c": c, "v": 100}


# ── availability gating ───────────────────────────────────────────────────────

def test_is_available_requires_keys_and_flag(monkeypatch):
    _enable(monkeypatch)
    assert ac.is_available() is True
    monkeypatch.setattr(ac.settings, "alpaca_api_secret", "")
    assert ac.is_available() is False          # missing secret
    monkeypatch.setattr(ac.settings, "alpaca_api_secret", "s")
    monkeypatch.setattr(ac.settings, "enable_alpaca_intraday", False)
    assert ac.is_available() is False          # disabled by flag


def test_get_bars_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(ac.settings, "alpaca_api_key", "")
    assert ac.get_bars("AAPL").empty


def test_unknown_interval_returns_empty(monkeypatch):
    _enable(monkeypatch)
    assert ac.get_bars("AAPL", interval="5m").empty


# ── request shape: symbol + timeframe mapping ─────────────────────────────────

def test_symbol_and_timeframe_mapping(monkeypatch):
    _enable(monkeypatch)
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        captured["headers"] = headers
        return _FakeResp({"bars": {}, "next_page_token": None})

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    ac.get_bars("BRK-B", interval="30m")

    assert captured["params"]["symbols"] == "BRK.B"      # hyphen → dot
    assert captured["params"]["timeframe"] == "30Min"
    assert captured["params"]["feed"] == "sip"
    assert captured["headers"]["APCA-API-KEY-ID"] == "k"


def test_to_alpaca_symbol():
    assert ac._to_alpaca_symbol("BRK-B") == "BRK.B"
    assert ac._to_alpaca_symbol("AAPL") == "AAPL"


# ── parsing + RTH filter (30-min) ─────────────────────────────────────────────

def test_30m_bars_parsed_and_rth_filtered(monkeypatch):
    _enable(monkeypatch)
    # 2024-01-02 is EST (UTC-5): 14:30Z=09:30 ET (keep), 20:30Z=15:30 ET (keep),
    # 14:00Z=09:00 ET pre-market (drop), 21:00Z=16:00 ET close boundary (drop).
    payload = {"bars": {"AAPL": [
        _bar("2024-01-02T14:00:00Z", 10),   # pre-market → dropped
        _bar("2024-01-02T14:30:00Z", 11),   # 09:30 ET  → kept
        _bar("2024-01-02T20:30:00Z", 12),   # 15:30 ET  → kept (last RTH bar)
        _bar("2024-01-02T21:00:00Z", 13),   # 16:00 ET  → dropped
    ]}, "next_page_token": None}
    monkeypatch.setattr(ac.httpx, "get", lambda *a, **k: _FakeResp(payload))

    df = ac.get_bars("AAPL", interval="30m")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2
    assert df["Close"].tolist() == [11.0, 12.0]          # pre-market + close-boundary stripped
    assert str(df.index.tz) == "UTC"


def test_pagination_concatenates_pages(monkeypatch):
    _enable(monkeypatch)
    pages = [
        _FakeResp({"bars": {"AAPL": [_bar("2024-01-02T14:30:00Z", 11)]},
                   "next_page_token": "p2"}),
        _FakeResp({"bars": {"AAPL": [_bar("2024-01-02T15:00:00Z", 12)]},
                   "next_page_token": None}),
    ]
    calls = {"n": 0}

    def fake_get(*a, **k):
        resp = pages[calls["n"]]
        calls["n"] += 1
        return resp

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    df = ac.get_bars("AAPL", interval="30m")
    assert len(df) == 2 and calls["n"] == 2               # both pages fetched + merged


# ── error handling ────────────────────────────────────────────────────────────

def _captured_warnings(fn):
    buf = io.StringIO()
    sid = logger.add(buf, level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sid)
    return buf.getvalue()


def test_403_warns_once_then_silent(monkeypatch):
    _enable(monkeypatch)
    ac._WARNED_FORBIDDEN.clear()
    monkeypatch.setattr(ac.httpx, "get", lambda *a, **k: _FakeResp(status=403))
    out = _captured_warnings(lambda: [ac.get_bars("AAPL"), ac.get_bars("MSFT")])
    assert out.count("HTTP 403") == 1                      # entitlement 403 deduped
    assert "yfinance fallback" in out


def test_429_returns_empty(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ac.httpx, "get", lambda *a, **k: _FakeResp(status=429))
    assert ac.get_bars("AAPL").empty


# ── market_data dispatcher: Alpaca preferred, yfinance fallback ───────────────

def test_dispatcher_prefers_alpaca(monkeypatch):
    df_alpaca = pd.DataFrame({"Close": [1.0]},
                             index=pd.to_datetime(["2024-01-02T14:30:00Z"]))
    monkeypatch.setattr(md.alpaca_client, "is_available", lambda: True)
    monkeypatch.setattr(md, "_fetch_intraday_alpaca", lambda t, interval="30m": df_alpaca)
    monkeypatch.setattr(md, "_fetch_intraday_yf",
                        lambda t, interval="30m": pytest.fail("yfinance should not be called"))
    out = md._fetch_intraday("AAPL", "30m")
    assert out is df_alpaca


def test_dispatcher_falls_back_to_yf_when_alpaca_empty(monkeypatch):
    sentinel = pd.DataFrame({"Close": [9.0]},
                            index=pd.to_datetime(["2024-01-02T14:30:00Z"]))
    monkeypatch.setattr(md.alpaca_client, "is_available", lambda: True)
    monkeypatch.setattr(md, "_fetch_intraday_alpaca", lambda t, interval="30m": pd.DataFrame())
    monkeypatch.setattr(md, "_fetch_intraday_yf", lambda t, interval="30m": sentinel)
    out = md._fetch_intraday("AAPL", "30m")
    assert out is sentinel


def test_dispatcher_uses_yf_when_alpaca_unavailable(monkeypatch):
    sentinel = pd.DataFrame({"Close": [7.0]},
                            index=pd.to_datetime(["2024-01-02T14:30:00Z"]))
    monkeypatch.setattr(md.alpaca_client, "is_available", lambda: False)
    monkeypatch.setattr(md, "_fetch_intraday_alpaca",
                        lambda t, interval="30m": pytest.fail("alpaca should not be called"))
    monkeypatch.setattr(md, "_fetch_intraday_yf", lambda t, interval="30m": sentinel)
    out = md._fetch_intraday("AAPL", "30m")
    assert out is sentinel


# ── multi-symbol batch ────────────────────────────────────────────────────────

def test_get_bars_batch_parses_per_symbol(monkeypatch):
    _enable(monkeypatch)
    payload = {"bars": {
        "AAPL": [_bar("2024-01-02T14:00:00Z", 9),    # 09:00 ET pre-market → dropped
                 _bar("2024-01-02T14:30:00Z", 11)],  # 09:30 ET → kept
        "MSFT": [_bar("2024-01-02T20:30:00Z", 12)],  # 15:30 ET → kept
    }, "next_page_token": None}
    monkeypatch.setattr(ac.httpx, "get", lambda *a, **k: _FakeResp(payload))

    out = ac.get_bars_batch(["AAPL", "MSFT"], interval="30m")
    assert set(out) == {"AAPL", "MSFT"}
    assert out["AAPL"]["Close"].tolist() == [11.0]    # pre-market bar RTH-filtered
    assert out["MSFT"]["Close"].tolist() == [12.0]


def test_get_bars_batch_chunks_over_limit(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ac, "_MAX_SYMBOLS_PER_REQ", 2)
    calls = []

    def fake_req(symbols, timeframe, lookback_days):
        calls.append(list(symbols))
        return {s: [_bar("2024-01-02T14:30:00Z", 1)] for s in symbols}

    monkeypatch.setattr(ac, "_request_bars", fake_req)
    out = ac.get_bars_batch(["AAPL", "MSFT", "NVDA"], interval="30m")
    assert calls == [["AAPL", "MSFT"], ["NVDA"]]       # chunked at the cap
    assert set(out) == {"AAPL", "MSFT", "NVDA"}


def test_get_bars_batch_restores_class_share(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ac, "_request_bars",
                        lambda symbols, timeframe, lookback_days: {"BRK.B": [_bar("2024-01-02T14:30:00Z", 5)]})
    out = ac.get_bars_batch(["BRK-B"], interval="30m")
    assert "BRK-B" in out and out["BRK-B"]["Close"].tolist() == [5.0]


def test_get_bars_batch_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(ac.settings, "alpaca_api_key", "")
    assert ac.get_bars_batch(["AAPL"]) == {}


# ── warm_intraday_cache: batch prefetch staged for the cycle ──────────────────

def test_warm_intraday_cache_stages_prefetch(monkeypatch):
    import src.data.cache as cache_mod
    df = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
        index=pd.to_datetime(["2024-01-02T14:30:00Z"]),
    )
    monkeypatch.setattr(md.alpaca_client, "is_available", lambda: True)
    monkeypatch.setattr(md.settings, "enable_fetch_data", True)
    monkeypatch.setattr(md, "_fetch_intraday_alpaca_batch",
                        lambda tickers, interval="30m": {"AAPL": df})
    saved = {}
    monkeypatch.setattr(cache_mod, "load_ohlcv", lambda t, i: None)
    monkeypatch.setattr(cache_mod, "save_ohlcv", lambda t, d, i: saved.__setitem__(t, d))

    n = md.warm_intraday_cache(["AAPL"])
    assert n == 1
    assert "AAPL" in md._INTRADAY_PREFETCH and "AAPL" in saved

    # get_history now reads the staged frame with NO per-ticker fetch.
    monkeypatch.setattr(md, "_fetch_intraday",
                        lambda t, interval="30m": pytest.fail("per-ticker fetch should be skipped"))
    out = md.get_history("AAPL", interval="30m")
    assert not out.empty


def test_warm_intraday_cache_noop_without_alpaca(monkeypatch):
    monkeypatch.setattr(md.alpaca_client, "is_available", lambda: False)
    monkeypatch.setattr(md, "_fetch_intraday_alpaca_batch",
                        lambda tickers, interval="30m": pytest.fail("should not fetch"))
    assert md.warm_intraday_cache(["AAPL", "MSFT"]) == 0
    assert md._INTRADAY_PREFETCH == {}
