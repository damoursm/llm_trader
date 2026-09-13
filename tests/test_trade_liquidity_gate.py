"""Trade-liquidity gate (actionable-filter Gate 4, 2026-07-08).

Penny / thin-volume names (< trade_min_price $5 or < trade_min_dollar_volume $5M)
are OBSERVE-ONLY — still scored + persisted to the signals panel (so penny-stock
performance keeps accruing) but never actionable. They enter the universe at the
LOWER discovery/observation floor. Fail-closed via is_liquid.
"""

import pytest

import src.data.liquidity as liq
from config.settings import settings
from src.pipeline import _is_tradeable


def test_discovery_floor_sits_below_the_trade_floor():
    # The whole design needs the observation floor BELOW the trade floor — else
    # sub-threshold names never enter the universe and the trade gate is a no-op.
    assert settings.discovery_min_price <= settings.trade_min_price
    assert settings.discovery_min_dollar_volume <= settings.trade_min_dollar_volume


def test_gate_off_is_always_tradeable(monkeypatch):
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", False)
    monkeypatch.setattr(liq, "is_liquid", lambda *a, **k: False)   # would gate — ignored
    assert _is_tradeable("ANY", {"n": 0}) is True


def test_gate_uses_the_trade_thresholds_not_discovery(monkeypatch):
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", True)
    monkeypatch.setattr(settings, "trade_min_price", 5.0)
    monkeypatch.setattr(settings, "trade_min_dollar_volume", 5_000_000)
    seen = {}

    def fake(ticker, budget, mp, mdv, price=None):
        seen["mp"], seen["mdv"] = mp, mdv
        return True

    monkeypatch.setattr(liq, "is_liquid", fake)
    assert _is_tradeable("AAPL", {"n": 5}) is True
    assert seen == {"mp": 5.0, "mdv": 5_000_000}   # the TRADE floor, not the discovery floor


def test_penny_or_thin_name_is_not_tradeable(monkeypatch):
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", True)
    monkeypatch.setattr(liq, "is_liquid", lambda *a, **k: False)   # below the trade floor
    assert _is_tradeable("PENNY", {"n": 5}) is False


# ── 2026-09-03 audit fixes: one dollar-volume formula, live price, exotics ──

def _frame(closes, volumes):
    import pandas as pd
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": volumes}, index=idx)


def test_dollar_volume_is_the_mean_of_the_per_bar_product():
    """THE definition every gate shares: mean(Close×Volume) over the last 20
    valid bars — NOT mean(Volume[-20:]) × Close[-1], which multiplied a 20-day
    volume average by a single price (chase-biased: names admitted only by
    that formula had a median 20d return of +36.5%; HYMCW read 961× too high)."""
    closes = [10.0] * 19 + [100.0]         # one 10× last close
    vols = [1_000_000.0] * 20
    df = _frame(closes, vols)
    assert liq.dollar_volume(df) == pytest.approx((19 * 1e7 + 1e8) / 20)
    # The old live formula would have said 1e6 × 100 = 1e8 — ten times higher.
    assert liq.dollar_volume(df) < 1e8 / 5
    # Window + min_bars semantics; unusable frames are None (fail-closed).
    assert liq.dollar_volume(df, window=5) == pytest.approx((4 * 1e7 + 1e8) / 5)
    assert liq.dollar_volume(df.iloc[:3], min_bars=5) is None
    assert liq.dollar_volume(None) is None
    assert liq.dollar_volume(df.drop(columns=["Volume"])) is None


def test_is_liquid_judges_the_live_price_and_the_cached_close(monkeypatch):
    """The floor is tested on BOTH: a live price below it defers the name even
    when the cached close clears (the dangerous direction — 10 of 16 measured
    straddles), and a cached close below it still fails when the live price
    clears (a wrong snapshot can defer, never admit). No live price → cached
    close alone, exactly as before."""
    monkeypatch.setattr(settings, "enable_fetch_data", False)
    ok = _frame([20.0] * 30, [1e6] * 30)            # $20 × 1M = $20M/day
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6, df=ok) is True
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6, price=4.99, df=ok) is False
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6, price=5.0, df=ok) is True
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6, price="junk", df=ok) is True   # unparseable = none
    low = _frame([4.0] * 30, [1e7] * 30)            # cached close below the floor
    assert liq.is_liquid("BBB", {"n": 0}, 5.0, 5e6, price=50.0, df=low) is False


def test_is_liquid_df_argument_skips_the_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(liq, "_load", lambda t, b: calls.append(t) or None)
    ok = _frame([20.0] * 30, [1e6] * 30)
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6, df=ok) is True
    assert calls == []
    assert liq.is_liquid("AAA", {"n": 0}, 5.0, 5e6) is False        # cache path, nothing there
    assert calls == ["AAA"]


def test_gate4_forwards_the_live_price(monkeypatch):
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", True)
    seen = {}

    def fake(ticker, budget, mp, mdv, price=None):
        seen["price"] = price
        return True

    monkeypatch.setattr(liq, "is_liquid", fake)
    assert _is_tradeable("AAPL", {"n": 0}, price=123.4) is True
    assert seen == {"price": 123.4}


def test_gate4_refuses_exotic_security_types(monkeypatch):
    """The exotic filter used to live ONLY at the discovery gate, so a warrant
    reaching the universe by another path was tradeable (ARQQW, 2026-06-17).
    Refused before is_liquid is even consulted; a class share is not exotic."""
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", True)
    monkeypatch.setattr(settings, "enable_security_type_filter", True)
    calls = []
    monkeypatch.setattr(liq, "is_liquid", lambda t, b, mp, mdv, price=None: calls.append(t) or True)
    assert _is_tradeable("ARQQW", {"n": 0}) is False
    assert _is_tradeable("ALL-PJ", {"n": 0}) is False
    assert calls == []
    assert _is_tradeable("BRK-B", {"n": 0}) is True
    monkeypatch.setattr(settings, "enable_security_type_filter", False)
    assert _is_tradeable("ARQQW", {"n": 0}) is True                # filter off → liquidity only
