"""Tests for src.broker.ibkr symbol mapping (pure functions — no ib_async needed).

The IBKRBroker itself imports ib_async lazily, so this module imports cleanly
without the dependency installed.
"""

import pytest

from src.broker.ibkr import from_ib_symbol, to_ib_symbol


@pytest.mark.parametrize("yf,ib", [
    ("BRK-B", "BRK B"),   # class shares: yfinance hyphen → IBKR space
    ("BF.B",  "BF B"),    # some feeds use a dot separator
    ("brk-b", "BRK B"),   # case-normalized
    ("AAPL",  "AAPL"),    # plain tickers pass through
    (" SPY ", "SPY"),     # whitespace stripped
])
def test_to_ib_symbol(yf, ib):
    assert to_ib_symbol(yf) == ib


@pytest.mark.parametrize("ib,internal", [
    ("BRK B", "BRK-B"),
    ("AAPL",  "AAPL"),
])
def test_from_ib_symbol(ib, internal):
    assert from_ib_symbol(ib) == internal


def test_class_share_roundtrip():
    assert from_ib_symbol(to_ib_symbol("BRK-B")) == "BRK-B"


# ── get_account: non-USD base-currency notice fires once per run ───────────────
# A CAD paper account triggers an FX-skew warning; before the fix it logged on
# EVERY get_account() (29×/day). get_account needs only a stubbed _ib (no ib_async
# on this path), so we can drive it directly.
def test_cad_base_currency_warns_once(monkeypatch):
    import io
    from types import SimpleNamespace as NS

    from loguru import logger

    import src.broker.ibkr as ibkr

    ibkr._WARNED_NON_USD_CCY.clear()
    broker = ibkr.IBKRBroker(account="DU123")
    rows = [
        NS(tag="NetLiquidation", value="100000", currency="CAD"),
        NS(tag="TotalCashValue", value="40000", currency="CAD"),
        NS(tag="BuyingPower",    value="80000", currency="CAD"),
    ]
    broker._ib = NS(isConnected=lambda: True, accountValues=lambda acct: rows)

    buf = io.StringIO()
    sid = logger.add(buf, level="WARNING")
    try:
        snaps = [broker.get_account() for _ in range(3)]
    finally:
        logger.remove(sid)

    # Still returns a correct snapshot every call — only the LOG is throttled.
    assert all(s is not None and s.currency == "CAD" and s.equity == 100000 for s in snaps)
    assert buf.getvalue().count("base currency is CAD") == 1

