"""Ticker validation guard — rejects junk ("N/A" etc.) without dropping real
symbols (single letters, class shares, futures, indices, the DXY).

Root cause: a "N/A" leaking from a discovery source reached yfinance, which then
raised an opaque "'Response' object is not subscriptable" deep in its parser —
268× in one day's log, plus 30× "Not enough history for N/A" downstream.
"""

import pandas as pd

from src.data.market_data import is_valid_ticker, sanitize_tickers, get_history


def test_real_tickers_are_valid():
    for t in ["AAPL", "F", "V", "U", "X", "BRK-B", "CL=F", "GC=F", "HG=F",
              "^VIX", "^NYAD", "^VIX3M", "DX-Y.NYB", "XLK"]:
        assert is_valid_ticker(t), t


def test_junk_tickers_are_invalid():
    for t in ["N/A", "NA", "NAN", "NONE", "NULL", "--", "-", "", "  ",
              "AB/CD", "FOO BAR", "TOOLONGTICKER123", None, 123]:
        assert not is_valid_ticker(t), repr(t)


def test_sanitize_dedupes_uppercases_and_drops_junk():
    assert sanitize_tickers(
        ["aapl", "N/A", "MSFT", "msft", "BRK-B", None, "--", "GC=F"]
    ) == ["AAPL", "MSFT", "BRK-B", "GC=F"]


def test_get_history_fails_fast_on_junk(monkeypatch):
    # Must short-circuit BEFORE touching polygon/yfinance — no fetch, empty frame.
    import src.data.market_data as md

    def _boom(*a, **k):
        raise AssertionError("network should not be called for a junk ticker")

    monkeypatch.setattr(md.polygon_client, "get_bars", _boom)
    out = get_history("N/A")
    assert isinstance(out, pd.DataFrame) and out.empty
