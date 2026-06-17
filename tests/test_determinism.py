"""Determinism regression for the daily-NAV compound engine (item #7).

The compound is documented as "deterministic given the DB + OHLCV cache". These
tests lock that contract so accidental nondeterminism (dict ordering, time-of-day
leakage, in-place mutation) fails loudly — and they pin the #1 reference-close fix
to an exact value so it cannot silently regress.
"""

from datetime import date

import pytest

import src.performance.daily_nav as dn
from src.performance.daily_nav import compute_compound_return, compute_trade_compound


@pytest.fixture
def fake_closes(monkeypatch):
    """Install a synthetic OHLCV close series (mirrors test_daily_nav)."""
    store: dict = {}

    def install(ticker, dates, closes):
        store[ticker] = dict(zip(dates, closes))

    monkeypatch.setattr(dn, "_load_close_series", lambda tk: store.get(tk, {}))
    return install


def _stk():
    return {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-11", "exit_date": "2026-05-13",
        "entry_price": 100.0, "exit_price": 110.0,
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }


def test_snapshot_locks_known_value(fake_closes):
    """Exact value lock (spread-only costs per conftest): the path-faithful daily
    walk over closes 105→108→110 with eff_entry 100.04, eff_exit 109.956 compounds
    to +9.93%. Pinned exactly so the #1 reference-close fix can't silently drift."""
    fake_closes("STK", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)],
                [105.0, 108.0, 110.0])
    assert compute_trade_compound(_stk()) == 9.93


def test_repeated_calls_bit_identical(fake_closes):
    fake_closes("STK", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)],
                [105.0, 108.0, 110.0])
    a = compute_compound_return([_stk()], today=date(2026, 5, 13))
    b = compute_compound_return([_stk()], today=date(2026, 5, 13))
    assert a == b


def test_order_independent(fake_closes):
    """Portfolio aggregation must not depend on trade list order."""
    fake_closes("STK", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)],
                [105.0, 108.0, 110.0])
    fake_closes("STK2", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)],
                [99.0, 98.0, 95.0])
    a = dict(_stk())
    b = {**_stk(), "ticker": "STK2", "action": "SELL", "exit_price": 95.0,
         "entry_price": 100.0, "position_size_multiplier": 2.0}
    forward = compute_compound_return([a, b], today=date(2026, 5, 13))
    backward = compute_compound_return([b, a], today=date(2026, 5, 13))
    assert forward == backward


def test_input_not_mutated(fake_closes):
    """The engine must not mutate the trade dicts it is handed."""
    fake_closes("STK", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)],
                [105.0, 108.0, 110.0])
    t = _stk()
    before = dict(t)
    compute_compound_return([t], today=date(2026, 5, 13))
    assert t == before
