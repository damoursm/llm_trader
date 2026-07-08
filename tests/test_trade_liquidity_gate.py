"""Trade-liquidity gate (actionable-filter Gate 4, 2026-07-08).

Penny / thin-volume names (< trade_min_price $5 or < trade_min_dollar_volume $5M)
are OBSERVE-ONLY — still scored + persisted to the signals panel (so penny-stock
performance keeps accruing) but never actionable. They enter the universe at the
LOWER discovery/observation floor. Fail-closed via is_liquid.
"""

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

    def fake(ticker, budget, mp, mdv):
        seen["mp"], seen["mdv"] = mp, mdv
        return True

    monkeypatch.setattr(liq, "is_liquid", fake)
    assert _is_tradeable("AAPL", {"n": 5}) is True
    assert seen == {"mp": 5.0, "mdv": 5_000_000}   # the TRADE floor, not the discovery floor


def test_penny_or_thin_name_is_not_tradeable(monkeypatch):
    monkeypatch.setattr(settings, "enable_trade_liquidity_gate", True)
    monkeypatch.setattr(liq, "is_liquid", lambda *a, **k: False)   # below the trade floor
    assert _is_tradeable("PENNY", {"n": 5}) is False
