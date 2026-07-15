"""Asset-type (instrument) filter for the Performance & Returns tabs (2026-07-13).

The dashboard's Type toggle (Stocks / ETFs / Commodities / all) threads an
``asset_type`` through get_performance_for_email and the pseudo-trade streams,
exactly like the existing session/direction filters. Verifies the pure matcher
and that _build_pseudo_trades honours it (the two shared mechanisms; the
end-to-end ledger split is verified live).
"""

import pandas as pd

import src.data.cache as cache_mod
import src.performance.tracker as tr
from src.performance.tracker import _match_asset_type


# ── pure matcher ────────────────────────────────────────────────────────────

def test_match_asset_type():
    stock = {"type": "STOCK"}
    etf = {"type": "ETF"}
    comm = {"type": "COMMODITY"}
    # None / '' = all types (no filtering)
    assert all(_match_asset_type(t, None) for t in (stock, etf, comm))
    assert _match_asset_type(stock, "")
    # exact type match, case-insensitive
    assert _match_asset_type(stock, "stock") and not _match_asset_type(stock, "etf")
    assert _match_asset_type(etf, "ETF") and not _match_asset_type(etf, "commodity")
    assert _match_asset_type(comm, "commodity")
    # a call/trade with no stored type defaults to STOCK (the record-time default)
    assert _match_asset_type({}, "stock")
    assert not _match_asset_type({}, "etf")


# ── _build_pseudo_trades honours asset_type ─────────────────────────────────

def _bars():
    idx = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"])
    return pd.DataFrame({"Close": [100.0, 105.0, 110.0]}, index=idx)


def _call(ticker, atype):
    return {"ticker": ticker, "type": atype, "action": "BUY",
            "entry_date": "2026-07-01", "entry_datetime": "2026-07-01T14:00:00+00:00",
            "snap_price": 100.0}


def test_build_pseudo_trades_filters_by_asset_type(monkeypatch):
    bars = {t: _bars() for t in ("AAA", "SPY", "GLD")}
    monkeypatch.setattr(cache_mod, "load_ohlcv", lambda tk, **kw: bars.get(tk))
    calls = [_call("AAA", "STOCK"), _call("SPY", "ETF"), _call("GLD", "COMMODITY")]

    allt = tr._build_pseudo_trades(calls)
    assert {t["ticker"] for t in allt} == {"AAA", "SPY", "GLD"}

    only_stock = tr._build_pseudo_trades(calls, asset_type="stock")
    assert {t["ticker"] for t in only_stock} == {"AAA"}

    only_etf = tr._build_pseudo_trades(calls, asset_type="etf")
    assert {t["ticker"] for t in only_etf} == {"SPY"}

    only_comm = tr._build_pseudo_trades(calls, asset_type="commodity")
    assert {t["ticker"] for t in only_comm} == {"GLD"}


def test_solo_method_perf_asset_filter(monkeypatch):
    """compute_solo_method_performance applies the asset_type trade filter."""
    def _fake_trade(ticker, atype, score):
        return {"ticker": ticker, "type": atype, "status": "CLOSED", "action": "BUY",
                "direction": "BULLISH", "return_pct": 2.0, "entry_date": "2026-07-01",
                "position_size_multiplier": 1.0, "method_scores": {"news": score}}

    trades = [_fake_trade("AAA", "STOCK", 0.5), _fake_trade("SPY", "ETF", 0.5)]
    monkeypatch.setattr(tr, "_load_trades", lambda: trades)
    # daily-NAV compound needs closes; make it a no-op so the win-rate path is what we assert
    monkeypatch.setattr(tr, "_compute_nav_compound", lambda ts: 0.0)

    allp = tr.compute_solo_method_performance()
    assert allp.get("news", {}).get("overall", {}).get("trades") == 2
    etfp = tr.compute_solo_method_performance(asset_type="etf")
    assert etfp.get("news", {}).get("overall", {}).get("trades") == 1
    stockp = tr.compute_solo_method_performance(asset_type="stock")
    assert stockp.get("news", {}).get("overall", {}).get("trades") == 1
