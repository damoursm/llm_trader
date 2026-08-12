"""The win-rate convention: GROSS everywhere, returns stay cost-adjusted.

Standing directive (2026-08-06): a win rate answers "did the position move the
way it was supposed to", on raw prices. Spread and commission are judged by the
RETURN metrics, never by folding them into the win test.

The failure this guards is quiet. A cost-adjusted win rate looks perfectly
reasonable — it just answers a different question than the label claims, drifts
whenever the cost model is recalibrated, and penalises a signal for being traded
in wide markets. So the tests below use the decisive case: a trade whose
direction was RIGHT but whose move was SMALLER than the round trip. Under the old
convention that was a loss; under this one it is a win with a negative return,
and both facts must show up in the same stats dict.
"""

from __future__ import annotations

import pytest

from src.performance import tracker


def _trade(entry: float, exit_: float, action: str = "BUY", **kw) -> dict:
    """A closed trade whose `return_pct` is deliberately cost-adjusted."""
    sign = 1.0 if action == "BUY" else -1.0
    gross = sign * (exit_ - entry) / entry * 100.0
    t = {
        "ticker": kw.pop("ticker", "AAA"),
        "action": action,
        "status": "CLOSED",
        "entry_price": entry,
        "exit_price": exit_,
        "entry_date": "2026-01-02",
        "exit_date": "2026-01-05",
        "return_pct": round(gross - 0.60, 4),      # charge a 0.60% round trip
        "position_size_multiplier": 1.0,
    }
    t.update(kw)
    return t


# ── the primitives ───────────────────────────────────────────────────────────

def test_gross_return_is_direction_aware():
    assert tracker.gross_return_pct(_trade(100.0, 101.0, "BUY")) == pytest.approx(1.0)
    # A short that fell is a WIN and a positive gross return.
    assert tracker.gross_return_pct(_trade(100.0, 99.0, "SELL")) == pytest.approx(1.0)
    assert tracker.gross_return_pct(_trade(100.0, 101.0, "SELL")) == pytest.approx(-1.0)


def test_a_right_direction_smaller_than_costs_is_a_WIN():
    # +0.20% move against a 0.60% round trip: right call, losing trade.
    t = _trade(100.0, 100.20, "BUY")
    assert t["return_pct"] < 0, "fixture must be a net loser for this test to mean anything"
    assert tracker.is_gross_win(t) is True
    assert tracker.gross_win_rate([t]) == 100.0


def test_flat_round_trip_is_not_a_win():
    # The direction did not pay, so it is not a win — but it is not charged the
    # spread either, which is the whole difference from the old convention.
    assert tracker.is_gross_win(_trade(100.0, 100.0, "BUY")) is False


def test_unusable_prices_are_excluded_not_counted_as_losses():
    bad = {"action": "BUY", "status": "CLOSED", "entry_price": None, "exit_price": 10.0}
    assert tracker.is_gross_win(bad) is None
    # One good win + one unusable => 100%, not 50%.
    assert tracker.gross_win_rate([_trade(100.0, 101.0), bad]) == 100.0
    assert tracker.gross_win_rate([bad]) is None


def test_open_trade_uses_the_live_mark():
    t = _trade(100.0, 101.0)
    t.update(status="OPEN", exit_price=None, current_price=103.0)
    assert tracker.gross_return_pct(t) == pytest.approx(3.0)


# ── the convention, end to end ───────────────────────────────────────────────

def test_segment_stats_win_rate_is_gross_while_returns_stay_net():
    """The load-bearing assertion: both numbers in one dict, disagreeing."""
    trades = [_trade(100.0, 100.20, "BUY"),      # right call, net loser
              _trade(100.0, 100.10, "BUY")]      # right call, net loser
    stats = tracker._compute_segment_stats(trades)
    assert stats["win_rate"] == 100.0, "direction was right on both — gross win rate is 100%"
    assert stats["avg_return"] < 0, "and both lost money after costs — returns stay net"


def test_win_rate_does_not_move_when_the_cost_model_does():
    """A cost recalibration must not rewrite the win rate of unchanged history."""
    a = _trade(100.0, 100.20, "BUY")
    b = dict(a, return_pct=a["return_pct"] - 5.0)   # same trade, harsher cost model
    assert tracker._compute_segment_stats([a])["win_rate"] == \
           tracker._compute_segment_stats([b])["win_rate"]
    assert tracker._compute_segment_stats([a])["avg_return"] != \
           tracker._compute_segment_stats([b])["avg_return"]


def test_short_trades_are_scored_on_their_own_direction():
    winners = [_trade(100.0, 99.0, "SELL"), _trade(100.0, 98.0, "SELL")]
    losers = [_trade(100.0, 101.0, "SELL")]
    assert tracker._compute_segment_stats(winners)["win_rate"] == 100.0
    assert tracker._compute_segment_stats(losers)["win_rate"] == 0.0
