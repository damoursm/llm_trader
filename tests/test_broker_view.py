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


def _leg(side="BUY", filled_qty=10, model_price=100.0, fill_price=100.0, commission=1.0):
    return {"side": side, "filled_qty": filled_qty, "model_price": model_price,
            "fill_price": fill_price, "commission": commission}


def test_leg_cost_is_commission_plus_execution_by_side():
    """Per-leg all-in = commission% + execution vs decision price; the order's
    own side sets the adverse direction (BUY adverse filling higher, SELL
    adverse filling lower)."""
    from src.performance.broker_view import one_way_cost_pcts_from_legs
    buy = _leg(side="BUY", model_price=100.0, fill_price=100.10, commission=1.0)
    sell = _leg(side="SELL", model_price=102.0, fill_price=101.80, commission=1.0)
    [c_buy, c_sell] = one_way_cost_pcts_from_legs([buy, sell])
    assert c_buy == pytest.approx(1.0 / (10 * 100.10) * 100 + (100.10 - 100.0) / 100.0 * 100, abs=1e-4)
    assert c_sell == pytest.approx(1.0 / (10 * 101.80) * 100 + (102.0 - 101.80) / 102.0 * 100, abs=1e-4)


def test_favorable_fill_lowers_the_cost():
    """A SELL filled HIGHER than the decision price → negative execution term;
    the all-in leg cost drops below the commission alone (signed)."""
    from src.performance.broker_view import avg_one_way_cost_pct_from_legs
    leg = _leg(side="SELL", model_price=50.0, fill_price=50.25, commission=1.0)
    cost = avg_one_way_cost_pct_from_legs([leg])
    comm = 1.0 / (10 * 50.25) * 100               # ~0.199%
    slip = (50.0 - 50.25) / 50.0 * 100            # −0.50% (favorable sell)
    assert cost == pytest.approx(comm + slip, abs=1e-3)
    assert cost < 0


def test_leg_cost_degrades_to_commission_without_model_price():
    """No decision price → execution term 0, leg cost = commission only."""
    from src.performance.broker_view import avg_one_way_cost_pct_from_legs
    leg = _leg(side="BUY", model_price=None, fill_price=100.0, commission=1.0)
    assert avg_one_way_cost_pct_from_legs([leg]) == pytest.approx(0.1, abs=1e-4)


def test_real_fraction_floor_and_average():
    from src.performance.broker_view import real_one_way_cost_fraction
    legs = [_leg(commission=1.0) for _ in range(4)]          # each 0.1%, 4 legs
    assert real_one_way_cost_fraction(legs, min_legs=10) is None     # below floor
    assert real_one_way_cost_fraction(legs, min_legs=4) == pytest.approx(0.001, abs=1e-5)


def test_no_legs_is_none():
    from src.performance.broker_view import (avg_one_way_cost_pct_from_legs,
                                             real_one_way_cost_fraction)
    assert avg_one_way_cost_pct_from_legs([]) is None
    assert real_one_way_cost_fraction([], min_legs=10) is None
