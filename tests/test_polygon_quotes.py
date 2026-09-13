"""Polygon real-time NBBO layer (2026-08-31): the last-NBBO helper, the batch
snapshot's lastQuote capture, the reconcile quote fallback that revives the
spread-aware caps + bid/ask persistence, and the liquidity forecast's live
layer. All network mocked; freshness gates are the load-bearing behaviour —
an off-hours "last NBBO" is the prior close's book and must never price an
order or a forecast."""

from __future__ import annotations

import time

import pytest

from config.settings import settings


# ── polygon_client.get_last_nbbo ───────────────────────────────────────────

def _nbbo_json(bid, ask, age_s):
    return {"results": {"p": bid, "P": ask,
                        "t": (time.time() - age_s) * 1e9}}


def test_get_last_nbbo_parses_and_ages(monkeypatch):
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc, "_get", lambda path, params=None: _nbbo_json(99.98, 100.02, 3.0))
    monkeypatch.setattr(pc.settings, "polygon_api_key", "test-key", raising=False)
    n = pc.get_last_nbbo("AAPL")
    assert n["bid"] == pytest.approx(99.98)
    assert n["ask"] == pytest.approx(100.02)
    assert 2.0 <= n["age_s"] <= 10.0


def test_get_last_nbbo_rejects_crossed_or_empty(monkeypatch):
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc.settings, "polygon_api_key", "test-key", raising=False)
    monkeypatch.setattr(pc, "_get", lambda *a, **k: _nbbo_json(100.02, 99.98, 1.0))
    assert pc.get_last_nbbo("AAPL") is None            # crossed
    monkeypatch.setattr(pc, "_get", lambda *a, **k: {"results": {}})
    assert pc.get_last_nbbo("AAPL") is None            # empty
    monkeypatch.setattr(pc, "_get", lambda *a, **k: None)
    assert pc.get_last_nbbo("AAPL") is None            # 403/None


# ── batch snapshot lastQuote capture ───────────────────────────────────────

def test_snapshot_batch_captures_last_quote(monkeypatch):
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc.settings, "polygon_api_key", "test-key", raising=False)
    now_ns = time.time() * 1e9

    def fake_get(path, params=None):
        if "snapshot" in path:
            return {"status": "OK", "tickers": [
                {"ticker": "AAPL", "day": {"c": 100.0, "v": 1e6},
                 "prevDay": {"c": 99.0, "v": 2e6}, "lastTrade": {"p": 100.5},
                 "lastQuote": {"p": 100.4, "P": 100.6, "t": now_ns - 2e9}},
                {"ticker": "BAD", "day": {"c": 10.0, "v": 1e6},
                 "prevDay": {"c": 9.0, "v": 1e6}, "lastTrade": {"p": 10.0},
                 "lastQuote": {"p": 10.2, "P": 10.1, "t": now_ns}},   # crossed
            ]}
        return {"results": []}      # the grouped-daily 5d call

    monkeypatch.setattr(pc, "_get", fake_get)
    out = pc.get_snapshots_batch(["AAPL", "BAD"])
    assert out["AAPL"]["bid"] == pytest.approx(100.4)
    assert out["AAPL"]["ask"] == pytest.approx(100.6)
    assert 1.0 <= out["AAPL"]["quote_age_s"] <= 10.0
    assert "bid" not in out["BAD"], "a crossed book must not be captured"


def test_ticker_snapshot_model_roundtrips_without_quote_fields():
    """Old cached snapshot JSON (pre-NBBO shape) must still validate — the
    date-keyed cache freezes shape, so absence is the normal case."""
    from src.models import TickerSnapshot
    s = TickerSnapshot.model_validate({
        "ticker": "AAPL", "price": 100.0, "pct_change_1d": 1.0,
        "pct_change_5d": 2.0, "volume": 1000})
    assert s.bid is None and s.ask is None and s.quote_age_s is None
    s2 = TickerSnapshot.model_validate(
        dict(s.model_dump(), bid=99.9, ask=100.1, quote_age_s=1.5))
    assert s2.ask == pytest.approx(100.1)


# ── symbology + transient-retry (2026-08-31 reliability work) ──────────────

def test_class_share_symbology_maps_to_polygon_form():
    """BRK-B is BRK.B at Polygon; the hyphen form returns NO data, which read
    as 'not covered' and burned the yfinance fallback (238 of 976 measured
    snapshot misses in 8 days). Futures/indices pass through untouched."""
    from src.data.polygon_client import to_polygon_symbol
    assert to_polygon_symbol("BRK-B") == "BRK.B"
    assert to_polygon_symbol("LEN-B") == "LEN.B"
    assert to_polygon_symbol("brk-b") == "BRK.B"
    assert to_polygon_symbol("AAPL") == "AAPL"
    assert to_polygon_symbol("CL=F") == "CL=F"       # futures: not on Polygon
    assert to_polygon_symbol("^VIX") == "^VIX"       # indices: not on Polygon
    # a hyphen that is NOT a trailing class letter stays put
    assert to_polygon_symbol("ABC-DEF") == "ABC-DEF"
    assert to_polygon_symbol("") == ""


def test_snapshot_batch_rekeys_class_shares(monkeypatch):
    """Requests go out in Polygon symbology and results come back under it —
    the caller must still get its own internal ticker as the key."""
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc.settings, "polygon_api_key", "k", raising=False)
    seen = {}

    def fake_get(path, params=None):
        if "snapshot" in path:
            seen["tickers"] = (params or {}).get("tickers", "")
            return {"status": "OK", "tickers": [
                {"ticker": "BRK.B", "day": {"c": 500.0, "v": 1},
                 "prevDay": {"c": 495.0, "v": 1}, "lastTrade": {"p": 504.0}}]}
        return {"results": []}

    monkeypatch.setattr(pc, "_get", fake_get)
    out = pc.get_snapshots_batch(["BRK-B"])
    assert "BRK.B" in seen["tickers"]
    assert "BRK-B" in out, "result must be re-keyed to the internal ticker"
    assert out["BRK-B"]["price"] == pytest.approx(504.0)


def test_get_retries_transient_failure_once(monkeypatch):
    """107 of 109 measured Polygon failures were socket timeouts — one retry
    recovers them. 403/429 must NOT retry."""
    from src.data import polygon_client as pc
    import httpx
    monkeypatch.setattr(pc.settings, "polygon_api_key", "k", raising=False)

    calls = {"n": 0}

    def flaky(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("handshake timed out")

        class R:
            def raise_for_status(self): pass
            def json(self): return {"status": "OK"}
        return R()

    monkeypatch.setattr(pc.httpx, "get", flaky)
    assert pc._get("/v2/x") == {"status": "OK"}
    assert calls["n"] == 2, "one retry expected"

    # 403 is an entitlement verdict — retrying is pointless traffic
    calls["n"] = 0

    def forbidden(url, params=None, timeout=None):
        calls["n"] += 1
        req = httpx.Request("GET", "http://x")
        raise httpx.HTTPStatusError(
            "403", request=req, response=httpx.Response(403, request=req))

    monkeypatch.setattr(pc.httpx, "get", forbidden)
    assert pc._get("/v2/y") is None
    assert calls["n"] == 1, "403 must not retry"


def test_price_chain_prefers_polygon_over_yfinance_in_rth():
    """Polygon is measured 99.876% reliable, ~2.7x faster and consolidated —
    the same feed as the snapshot and the NBBO that prices orders."""
    import inspect
    from src.performance import tracker
    src = inspect.getsource(tracker._fetch_price)
    rth = src.split("else:")[-1]
    assert "_polygon() or _yf_fast()" in rth.replace("\n", " ").replace("  ", " ")


# ── reconcile._quote_for fallback ──────────────────────────────────────────

class _NoQuoteBroker:
    def get_quote(self, tk):
        return None


def test_quote_for_falls_back_to_polygon(monkeypatch):
    from src.broker import reconcile as rc
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)
    monkeypatch.setattr(settings, "enable_polygon_quotes", True)
    monkeypatch.setattr("src.data.polygon_client.get_last_nbbo",
                        lambda tk: {"bid": 49.99, "ask": 50.01, "age_s": 2.0})
    rc._LAST_QUOTE.clear()
    q = rc._quote_for(_NoQuoteBroker(), "MSFT")
    assert q is not None and q.bid == pytest.approx(49.99)
    assert "MSFT" in rc._LAST_QUOTE            # stashed for _record_order
    rc._LAST_QUOTE.clear()


def test_quote_for_refuses_stale_nbbo(monkeypatch):
    from src.broker import reconcile as rc
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)
    monkeypatch.setattr(settings, "enable_polygon_quotes", True)
    monkeypatch.setattr("src.data.polygon_client.get_last_nbbo",
                        lambda tk: {"bid": 49.99, "ask": 50.01,
                                    "age_s": rc._NBBO_FRESH_SECONDS + 60})
    rc._LAST_QUOTE.clear()
    assert rc._quote_for(_NoQuoteBroker(), "MSFT") is None
    assert "MSFT" not in rc._LAST_QUOTE


def test_quote_for_respects_polygon_flag(monkeypatch):
    from src.broker import reconcile as rc
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)
    monkeypatch.setattr(settings, "enable_polygon_quotes", False)
    called = {"n": 0}

    def boom(tk):
        called["n"] += 1
        return {"bid": 1.0, "ask": 1.1, "age_s": 1.0}

    monkeypatch.setattr("src.data.polygon_client.get_last_nbbo", boom)
    assert rc._quote_for(_NoQuoteBroker(), "MSFT") is None
    assert called["n"] == 0


def test_broker_quote_wins_over_polygon(monkeypatch):
    """Venue-true broker book first; Polygon only fills the gap."""
    from src.broker import reconcile as rc
    from src.broker.base import Quote
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)
    monkeypatch.setattr(settings, "enable_polygon_quotes", True)
    monkeypatch.setattr("src.data.polygon_client.get_last_nbbo",
                        lambda tk: {"bid": 1.0, "ask": 2.0, "age_s": 1.0})

    class _B:
        def get_quote(self, tk):
            return Quote(ticker=tk, bid=99.0, ask=99.1)

    q = rc._quote_for(_B(), "MSFT")
    assert q.bid == pytest.approx(99.0)
    rc._LAST_QUOTE.clear()


# ── liquidity forecast live layer ──────────────────────────────────────────

def test_live_spreads_drive_the_forecast(monkeypatch):
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (2.295, 0.647))
    n = lf.set_live_spreads({"WIDE": 40.0, "TIGHT": 0.5, "bad": None})
    assert n == 2
    f = lf.forecast_for("WIDE")
    assert f["estimator"] == "nbbo"
    assert f["exp_halfspread_bps"] == pytest.approx(2.295 * 40.0 ** 0.647, abs=0.05)
    assert (lf.forecast_for("TIGHT")["exp_halfspread_bps"]
            < f["exp_halfspread_bps"])
    lf.reset_cache()


def test_live_nbbo_passes_through_unconverted(monkeypatch):
    """POINT-IN-TIME is the canonical basis the mapping is fitted on
    (2026-09-01), so a live book must reach it UNSCALED. Before the refit the
    conversion ran the other way — rebasing NBBO UP onto the day-average sweep
    basis — and getting that direction wrong misprices every live-covered
    name."""
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (2.0, 0.7))
    lf.set_live_spreads({"WIDE": 40.0})
    assert lf.forecast_for("WIDE")["exp_halfspread_bps"] == pytest.approx(
        2.0 * 40.0 ** 0.7, abs=0.05)
    lf.reset_cache()


def test_day_average_sweep_is_converted_down_to_point_in_time(monkeypatch):
    """The EOD sweep is a whole-RTH time-average running ~1.88x a
    point-in-time book, so it must be divided DOWN before the mapping."""
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (2.0, 1.0))  # identity-ish
    monkeypatch.setattr(lf, "_ibkr_spread_map", lambda: {"SWEPT": 40.0})
    monkeypatch.setattr(lf, "_memo_epoch", lambda: ("t", 0.0))
    lf.set_live_spreads({})                       # no overlap → prior scale
    assert lf._sweep_to_pit_scale() == pytest.approx(lf._SWEEP_TO_PIT_PRIOR)
    f = lf.forecast_for("SWEPT")
    assert f["estimator"] == "ibkr"
    assert f["exp_halfspread_bps"] == pytest.approx(
        2.0 * (40.0 * lf._SWEEP_TO_PIT_PRIOR), abs=0.05)
    lf.reset_cache()


def test_sweep_scale_self_calibrates_from_overlap(monkeypatch):
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    # every overlapping name shows the live book at a THIRD of the sweep value
    monkeypatch.setattr(lf, "_ibkr_spread_map",
                        lambda: {f"T{i}": 30.0 for i in range(40)})
    lf.set_live_spreads({f"T{i}": 10.0 for i in range(40)})
    scale = lf._sweep_to_pit_scale()
    # prior is 1/1.88 = 0.532; the measurement is 0.333, so the shrunk value
    # must land BETWEEN them — pulled down toward the evidence, not past it.
    assert 1.0 / 3.0 <= scale < lf._SWEEP_TO_PIT_PRIOR, \
        "a measured 1/3 must pull the 1/1.88 prior DOWN toward it"
    lf.reset_cache()


def test_live_spreads_expire_to_structural(monkeypatch):
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_ibkr_spread_map", lambda: {"FAKE": 5.0})
    monkeypatch.setattr(lf, "_memo_epoch", lambda: ("t", 0.0))
    lf.set_live_spreads({"FAKE": 300.0})
    assert lf.forecast_for("FAKE")["estimator"] == "nbbo"
    lf._LIVE_SPREADS["ts"] = time.time() - lf._LIVE_SPREAD_TTL - 1   # expire
    lf._FORECAST_MEMO.update(day=None, map={})
    assert lf.forecast_for("FAKE")["estimator"] == "ibkr"
    lf.reset_cache()


def test_prime_results_back_cached_bps_with_live_layer(monkeypatch):
    """The persistence read must reflect prime's output even when every name
    resolved from the live map (which bypasses the structural memo)."""
    from src.performance import liquidity_forecast as lf
    lf.reset_cache()
    monkeypatch.setattr(lf, "_refresh_realized", lambda: None)
    monkeypatch.setattr(lf, "_level_factor", lambda: 0.27)
    monkeypatch.setattr(lf, "_spread_mapping", lambda: (2.295, 0.647))
    lf.set_live_spreads({"AAA": 10.0})
    s = lf.prime_liquidity_forecast(["AAA"])
    assert s["live_nbbo"] == 1
    assert lf.cached_bps("AAA") == pytest.approx(2.295 * 10.0 ** 0.647, abs=0.05)
    assert lf.cached_bps("NEVER") is None
    lf.reset_cache()
