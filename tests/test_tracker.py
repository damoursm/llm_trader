"""Tests for src.performance.tracker — flip-trade, normalize, save-noop, holding."""

import json
from datetime import date
from pathlib import Path

import pytest

from src.performance import tracker
from src.performance.tracker import (
    _flip_trade,
    _normalize_closed_returns,
    _save_trades,
    _trading_days_held,
)


# ── _flip_trade: action / direction / return_pct all inverted ────────────

def test_flip_trade_inverts_action_and_direction():
    t = {
        "ticker": "ABC", "type": "STOCK",
        "action": "BUY", "direction": "BULLISH",
        "entry_price": 100.0, "exit_price": 110.0,
        "current_price": 110.0, "status": "CLOSED",
        "position_size_multiplier": 1.0,
    }
    flipped = _flip_trade(t)
    assert flipped["action"] == "SELL"
    assert flipped["direction"] == "BEARISH"


def test_flip_trade_rederives_return_pct():
    """A long that earned +10 % must, when flipped, return roughly -10 % minus
    round-trip spread cost — *not* simply the negation of the long's return_pct."""
    long_trade = {
        "ticker": "ABC", "type": "STOCK",
        "action": "BUY", "direction": "BULLISH",
        "entry_price": 100.0, "exit_price": 110.0,
        "current_price": 110.0, "status": "CLOSED",
        "position_size_multiplier": 1.0,
        "return_pct": 9.91,
    }
    flipped = _flip_trade(long_trade)
    # Hypothetical short on 100→110: bad outcome, ≈ -10 % minus spread
    assert flipped["return_pct"] < -9.9
    assert flipped["return_pct"] > -10.5


def test_flip_trade_preserves_dates_and_ticker():
    """Daily-NAV walk on flipped trades MUST hit the same OHLCV rows so it
    walks real prices in the opposite direction (not just negated returns)."""
    t = {
        "ticker": "ABC", "type": "STOCK", "action": "BUY", "direction": "BULLISH",
        "entry_date": "2026-05-01", "exit_date": "2026-05-05",
        "entry_price": 100.0, "exit_price": 110.0, "current_price": 110.0,
        "status": "CLOSED", "position_size_multiplier": 1.0,
    }
    flipped = _flip_trade(t)
    for field in ("ticker", "entry_date", "exit_date", "entry_price", "exit_price"):
        assert flipped[field] == t[field], f"{field} must survive the flip"


# ── _normalize_closed_returns: idempotency ───────────────────────────────

def test_normalize_closed_returns_idempotent():
    """Running normalisation twice must not change anything after the first run."""
    trades = [{
        "ticker": "ABC", "type": "STOCK", "action": "BUY",
        "entry_price": 100.0, "exit_price": 110.0,
        "position_size_multiplier": 1.0, "status": "CLOSED",
        "return_pct": 0.0,   # stale value
    }]
    n1 = _normalize_closed_returns(trades)
    after_first = trades[0]["return_pct"]
    n2 = _normalize_closed_returns(trades)
    after_second = trades[0]["return_pct"]
    assert n1 == 1, "first run should refresh the stale return"
    assert n2 == 0, "second run should be a no-op"
    assert after_first == after_second


def test_normalize_closed_returns_skips_open_trades():
    """Open trades carry live M2M return_pct — must not be normalized away."""
    trades = [{
        "ticker": "ABC", "type": "STOCK", "action": "BUY",
        "entry_price": 100.0, "current_price": 105.0,
        "position_size_multiplier": 1.0, "status": "OPEN",
        "return_pct": 4.92,
    }]
    n = _normalize_closed_returns(trades)
    assert n == 0


# ── _save_trades: skip write when content unchanged (Fix #20) ────────────

def test_save_trades_skips_write_when_unchanged(tmp_path, monkeypatch):
    """Writing the same content twice should only touch the file once."""
    target = tmp_path / "trades.json"
    monkeypatch.setattr(tracker, "TRADES_FILE", target)

    trades = [{"ticker": "ABC", "status": "OPEN", "return_pct": 1.0}]
    _save_trades(trades)
    assert target.exists()
    first_mtime = target.stat().st_mtime_ns

    # Save again with identical content — file must NOT be touched.
    _save_trades(trades)
    second_mtime = target.stat().st_mtime_ns
    assert first_mtime == second_mtime, "no-op save must not update mtime"


def test_save_trades_writes_when_content_differs(tmp_path, monkeypatch):
    target = tmp_path / "trades.json"
    monkeypatch.setattr(tracker, "TRADES_FILE", target)

    _save_trades([{"ticker": "ABC", "status": "OPEN", "return_pct": 1.0}])
    original = target.read_text(encoding="utf-8")

    _save_trades([{"ticker": "ABC", "status": "OPEN", "return_pct": 2.0}])
    updated = target.read_text(encoding="utf-8")
    assert original != updated
    assert "2.0" in updated


# ── _trading_days_held: uses real market calendar ────────────────────────

def test_trading_days_held_excludes_memorial_day(monkeypatch):
    """_trading_days_held delegates to the NYSE calendar — Memorial Day must NOT count."""
    import src.performance.tracker as trk

    class _FixedTodayDate(date):
        @classmethod
        def today(cls):
            return date(2026, 6, 2)

    monkeypatch.setattr(trk, "date", _FixedTodayDate)

    # Trade entered Fri 2026-05-22, "today" = Tue 2026-06-02.
    # Memorial Day 2026-05-25 falls inside the holding window — must NOT count.
    assert _trading_days_held("2026-05-22") == 6   # 7 under the old weekday-only counter
