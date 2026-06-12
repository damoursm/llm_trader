"""broker_view — the IBKR-fills projection behind the dashboard's
Simulated ⇄ IBKR toggle. Real fill prices, real commissions, no modeled costs.
"""

import pytest

from src.performance.broker_view import (build_broker_trades, summarize_broker_trades)


def _trade(**over):
    t = {
        "ticker": "XLE", "type": "ETF", "action": "BUY", "status": "CLOSED",
        "entry_price": 57.12, "exit_price": 57.50, "current_price": 57.50,
        "position_size_multiplier": 0.75,
        "broker_fill_qty": 9, "broker_fill_price": 57.15, "broker_commission": 1.0,
        "broker_submitted_at": "2026-06-11T20:33:56+00:00",
        "broker_exit_fill_qty": 9, "broker_exit_fill_price": 57.40,
        "broker_exit_commission": 1.0,
        "broker_exit_submitted_at": "2026-06-11T21:04:03+00:00",
    }
    t.update(over)
    return t


def test_closed_long_uses_fill_prices_and_real_commissions():
    [b] = build_broker_trades([_trade()])
    assert b["status"] == "CLOSED"
    assert b["entry_price"] == pytest.approx(57.15)   # fill, not the sim's 57.12
    assert b["exit_price"] == pytest.approx(57.40)
    notional = 9 * 57.15
    expected = (57.40 - 57.15) / 57.15 * 100 - 2.0 / notional * 100
    assert b["return_pct"] == pytest.approx(expected, abs=1e-3)
    assert b["position_size_multiplier"] == pytest.approx(notional)
    assert b["entry_datetime"] == "2026-06-11T20:33:56+00:00"


def test_closed_short_signs_the_move():
    t = _trade(action="SELL", broker_fill_price=50.0, broker_exit_fill_price=48.0,
               broker_fill_qty=10, broker_commission=0.5, broker_exit_commission=0.5)
    [b] = build_broker_trades([t])
    expected = (50.0 - 48.0) / 50.0 * 100 - 1.0 / 500.0 * 100   # +4% gross − 0.2% comm
    assert b["return_pct"] == pytest.approx(expected, abs=1e-6)


def test_ledger_closed_but_exit_unfilled_is_open_in_real_view():
    """The sim closed the trade, but the shares are genuinely still held —
    the real view must show an OPEN position marked at current_price."""
    t = _trade(broker_exit_fill_qty=0, broker_exit_fill_price=None,
               current_price=58.0)
    [b] = build_broker_trades([t])
    assert b["status"] == "OPEN"
    assert b["exit_price"] is None
    gross = (58.0 - 57.15) / 57.15 * 100
    expected = gross - 1.0 / (9 * 57.15) * 100   # entry commission only
    assert b["return_pct"] == pytest.approx(expected, abs=1e-3)


def test_unfilled_entries_do_not_exist_in_real_view():
    assert build_broker_trades([_trade(broker_fill_qty=0)]) == []
    assert build_broker_trades([_trade(broker_fill_price=None)]) == []
    assert build_broker_trades([{"ticker": "GLD", "action": "BUY",
                                 "status": "OPEN", "entry_price": 100.0}]) == []


def test_summary_dollar_pnl_and_win_rate():
    win = _trade()                                     # +0.25/share × 9 − $2
    loss = _trade(ticker="TRUP", broker_fill_price=23.30, broker_exit_fill_price=23.20,
                  broker_fill_qty=32, broker_commission=1.0, broker_exit_commission=1.0)
    open_t = _trade(ticker="RDW", broker_exit_fill_qty=0, broker_exit_fill_price=None,
                    broker_fill_price=10.0, broker_fill_qty=70, broker_commission=1.0,
                    current_price=10.50)
    s = summarize_broker_trades(build_broker_trades([win, loss, open_t]))
    assert s["trades"] == 3 and s["closed"] == 2 and s["open"] == 1
    assert s["win_rate"] == pytest.approx(50.0)
    assert s["realized_pnl_usd"] == pytest.approx((0.25 * 9 - 2.0) + (-0.10 * 32 - 2.0), abs=0.01)
    assert s["unrealized_pnl_usd"] == pytest.approx(0.50 * 70 - 1.0, abs=0.01)
    assert s["commissions_usd"] == pytest.approx(2.0 + 2.0 + 1.0)
