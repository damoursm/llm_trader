"""Polygon REST helper: endpoint-family normalization + the 403 warn-once.

A free-tier 403 (entitlement limit) on the snapshot endpoint must NOT spam the log
every tick (observed 54×/day) — it's not transient and the yfinance fallback covers
it — so it warns once per endpoint family, then drops to debug. 429 (rate limit)
and other statuses are unaffected.
"""

import io

import httpx
import pytest
from loguru import logger

import src.data.polygon_client as pc


def test_endpoint_family_collapses_trailing_symbol():
    base = "/v2/snapshot/locale/us/markets/stocks/tickers"
    assert pc._endpoint_family(base + "/ARQQW") == base   # per-ticker → family
    assert pc._endpoint_family(base + "/BRK.B") == base
    assert pc._endpoint_family(base) == base              # batch already a family
    assert pc._endpoint_family("/v3/reference/tickers/AAPL") == "/v3/reference/tickers"


def _raiser(status: int):
    """An httpx-like response whose raise_for_status() raises `status`."""
    class _Resp:
        def raise_for_status(self):
            req = httpx.Request("GET", "https://api.polygon.io/x")
            resp = httpx.Response(status, request=req)
            raise httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)

        def json(self):  # never reached on an error status
            return {}
    return _Resp()


def _captured_warnings(fn):
    buf = io.StringIO()
    sid = logger.add(buf, level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sid)
    return buf.getvalue()


def test_403_warns_once_per_endpoint_family(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    pc._WARNED_FORBIDDEN.clear()
    monkeypatch.setattr(pc.httpx, "get", lambda *a, **k: _raiser(403))

    base = "/v2/snapshot/locale/us/markets/stocks/tickers"
    out = _captured_warnings(lambda: [pc._get(f"{base}/{t}") for t in ("AAPL", "MSFT", "NVDA")])

    assert out.count("HTTP 403") == 1                       # collapsed to one warning
    assert "not available on this API plan" in out


def test_429_warns_every_call(monkeypatch):
    # Rate-limit is transient — it must NOT be deduped like the 403 entitlement case.
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    monkeypatch.setattr(pc.httpx, "get", lambda *a, **k: _raiser(429))
    out = _captured_warnings(lambda: [pc._get(f"/v2/aggs/ticker/{t}/range") for t in ("A", "B")])
    assert out.count("Rate limited") == 2


def test_404_is_silent(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    monkeypatch.setattr(pc.httpx, "get", lambda *a, **k: _raiser(404))
    out = _captured_warnings(lambda: pc._get("/v2/aggs/ticker/ZZZZ/range"))
    assert out == ""                                        # missing ticker → no noise


# ── grouped-daily close fallback (the deterministic bulk price source) ────────

def test_grouped_daily_closes_parses_first_nonempty(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    calls = []

    def fake_get(path, params=None):
        calls.append(path)
        if len(calls) == 1:
            return {"results": []}                          # today/holiday empty → try prior day
        return {"results": [{"T": "AAPL", "c": 150.0}, {"T": "BRK.B", "c": 50.0},
                            {"T": "NOCLOSE"}]}              # missing 'c' → dropped
    monkeypatch.setattr(pc, "_get", fake_get)
    out = pc.get_grouped_daily_closes()
    assert out == {"AAPL": 150.0, "BRK.B": 50.0}
    assert len(calls) == 2                                  # fell through to the first non-empty


def test_grouped_daily_closes_no_key(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "")
    assert pc.get_grouped_daily_closes() == {}
