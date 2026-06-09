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
