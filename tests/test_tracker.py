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


# ── _save_trades / _load_trades: DuckDB round-trip ────────────────────────
# (These replace two legacy tests that monkeypatched the obsolete TRADES_FILE
# JSON path while _save_trades had moved to DuckDB — so they wrote their
# fixture rows into the PRODUCTION trades table via full-replace, wiping the
# real ledger. The conftest _isolated_db fixture now makes that impossible;
# tracker.TRADES_FILE is also pointed away so _load_trades' legacy self-seed
# can't read the real cache/trades.json.)

def test_save_trades_roundtrips_through_duckdb(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    trades = [{"ticker": "XYZ", "status": "OPEN", "return_pct": 1.0,
               "entry_date": "2026-06-01", "entry_price": 10.0, "action": "BUY"}]
    _save_trades(trades)
    assert tracker._load_trades() == trades   # byte-identical dict round-trip


def test_save_trades_is_full_replace(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    base = {"ticker": "XYZ", "status": "OPEN", "entry_date": "2026-06-01",
            "entry_price": 10.0, "action": "BUY"}
    _save_trades([dict(base, return_pct=1.0)])
    _save_trades([dict(base, return_pct=2.0)])
    loaded = tracker._load_trades()
    assert len(loaded) == 1
    assert loaded[0]["return_pct"] == 2.0


# ── Direction (Long / Short) filter — dashboard Model Performance + Returns ──

def test_direction_filter_splits_long_short(tmp_path, monkeypatch):
    """get_performance_for_email(direction=...) restricts to BUY (long) / SELL
    (short) entries; long + short == all, with no leakage. Backs the dashboard's
    Long/Short/both toggle on the Model Performance and Returns tabs."""
    from config.settings import settings
    from src.db import repo
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(settings, "enable_fetch_data", False)   # no network

    longs = [{"ticker": f"LNG{i}", "type": "STOCK", "action": "BUY", "direction": "BULLISH",
              "status": "CLOSED", "confidence": 0.8, "position_size_multiplier": 1.0,
              "entry_date": "2026-06-10", "entry_price": 10.0, "return_pct": 5.0,
              "exit_date": "2026-06-11", "exit_price": 10.5} for i in range(3)]
    shorts = [{"ticker": f"SHT{i}", "type": "STOCK", "action": "SELL", "direction": "BEARISH",
               "status": "CLOSED", "confidence": 0.8, "position_size_multiplier": 1.0,
               "entry_date": "2026-06-10", "entry_price": 20.0, "return_pct": -2.0,
               "exit_date": "2026-06-11", "exit_price": 20.4} for i in range(2)]
    repo.save_trades(longs + shorts)

    all_p   = tracker.get_performance_for_email()
    long_p  = tracker.get_performance_for_email(direction="long")
    short_p = tracker.get_performance_for_email(direction="short")

    assert all_p["stats"]["total_all"] == 5
    assert long_p["stats"]["total_all"] == 3
    assert short_p["stats"]["total_all"] == 2
    assert all(t["action"] == "BUY" for t in long_p["closed_trades"])
    assert all(t["action"] == "SELL" for t in short_p["closed_trades"])


# ── Ledger-corruption regression (2026-06-10 production incident) ─────────

def _poisoned_ledger():
    """The literal malformed row that crashed production, plus realistic
    open/closed trades (fake tickers so no OHLCV cache is consulted)."""
    fake = {"ticker": "ABC", "status": "OPEN", "return_pct": 1.0,
            "broker_status": "SKIPPED_ZERO_QTY"}
    open_t = {"ticker": "ZZTA", "type": "STOCK", "action": "SELL", "direction": "BEARISH",
              "status": "OPEN", "confidence": 0.8, "position_size_multiplier": 1.0,
              "entry_date": "2026-06-10", "entry_price": 50.0,
              "current_price": 49.0, "return_pct": 2.0, "days_held": 0,
              "exit_date": None, "exit_price": None}
    closed_t = {"ticker": "ZZTB", "type": "STOCK", "action": "BUY", "direction": "BULLISH",
                "status": "CLOSED", "confidence": 0.8, "position_size_multiplier": 1.0,
                "entry_date": "2026-06-10", "entry_price": 10.0,
                "current_price": 11.0, "return_pct": 9.8, "days_held": 0,
                "exit_date": "2026-06-10", "exit_price": 11.0}
    return [fake, open_t, closed_t]


def test_poisoned_ledger_does_not_crash_pipeline_paths(tmp_path, monkeypatch):
    """Regression for 2026-06-10: a structurally-invalid row in the trades
    table (a stray test fixture) crashed every pipeline tick with
    KeyError('entry_date') — first in log_performance_summary, then in
    get_performance_for_email. The loader's sanitize boundary must exclude it
    from every consumer, and the next ledger write must purge it from the DB.
    """
    from config.settings import settings
    from src.db import repo

    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(settings, "enable_fetch_data", False)   # no network

    repo.save_trades(_poisoned_ledger())   # corruption written raw, as the old test did

    loaded = tracker._load_trades()
    assert {t["ticker"] for t in loaded} == {"ZZTA", "ZZTB"}    # ABC excluded

    tracker.log_performance_summary()                  # crash site #1 — must not raise

    perf = tracker.get_performance_for_email()         # crash site #2 — must not raise
    assert [t["ticker"] for t in perf["open_trades"]] == ["ZZTA"]
    assert [t["ticker"] for t in perf["closed_trades"]] == ["ZZTB"]
    assert perf["stats"]["total_all"] == 2

    # The next ledger write (a normal pipeline step) self-purges the bad row.
    tracker.update_open_trades()
    raw = repo.load_trades()                           # raw read — no sanitize
    assert {t["ticker"] for t in raw} == {"ZZTA", "ZZTB"}


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
