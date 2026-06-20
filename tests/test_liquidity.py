"""Discovery liquidity gate — fail-closed on no data, robust to a NaN last bar."""

import pandas as pd

from src.data import liquidity


def _df(close: float, vol: float, n: int = 25, nan_last: bool = False) -> pd.DataFrame:
    df = pd.DataFrame({"Close": [close] * n, "Volume": [vol] * n})
    if nan_last:  # simulate the still-forming intraday bar / trailing yfinance NaN
        df = pd.concat([df, pd.DataFrame({"Close": [float("nan")], "Volume": [float("nan")]})],
                       ignore_index=True)
    return df


def test_no_data_is_dropped(monkeypatch):
    # FGRS case: OTC/new ticker with no OHLCV from any provider → fail-closed.
    monkeypatch.setattr(liquidity, "_load", lambda t, b: None)
    assert liquidity.is_liquid("FGRS", {"n": 1}, 5.0, 20e6) is False


def test_liquid_large_cap_is_kept(monkeypatch):
    monkeypatch.setattr(liquidity, "_load", lambda t, b: _df(200.0, 10_000_000))
    assert liquidity.is_liquid("AAPL", {"n": 1}, 5.0, 20e6) is True


def test_nan_last_bar_does_not_drop_a_liquid_name(monkeypatch):
    # Regression: a NaN last bar (forming intraday) must NOT make a liquid name
    # fail the gate — the old code read iloc[-1] raw → NaN*NaN>=floor → False.
    monkeypatch.setattr(liquidity, "_load", lambda t, b: _df(200.0, 10_000_000, nan_last=True))
    assert liquidity.is_liquid("AAPL", {"n": 1}, 5.0, 20e6) is True


def test_illiquid_low_dollar_volume_dropped(monkeypatch):
    monkeypatch.setattr(liquidity, "_load", lambda t, b: _df(200.0, 1_000))   # ~$200k ADV < $20M
    assert liquidity.is_liquid("THINLY", {"n": 1}, 5.0, 20e6) is False


def test_below_min_price_dropped(monkeypatch):
    monkeypatch.setattr(liquidity, "_load", lambda t, b: _df(2.0, 50_000_000))  # $2 < $5 floor
    assert liquidity.is_liquid("PENNY", {"n": 1}, 5.0, 20e6) is False


def test_gate_dedupes_and_uppercases(monkeypatch):
    monkeypatch.setattr(liquidity.settings, "enable_discovery_liquidity_gate", False)
    # no-op mode still cleans the input (dedupe + uppercase)
    assert liquidity.apply_liquidity_gate(["aapl", "AAPL", "nvda"]) == ["AAPL", "NVDA"]


def test_gate_drops_junk_tickers(monkeypatch):
    # "N/A" and friends leaking from a discovery source must never survive the
    # gate's cleaning pass — they otherwise reach yfinance and spam the log.
    monkeypatch.setattr(liquidity.settings, "enable_discovery_liquidity_gate", False)
    assert liquidity.apply_liquidity_gate(["AAPL", "N/A", "--", "", "nvda"]) == ["AAPL", "NVDA"]
