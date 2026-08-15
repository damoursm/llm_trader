"""One-time time-cap unwind migration (`src/db/unwind_time_cap.py`).

Run by hand (`python -m src.db.unwind_time_cap`) to undo the removed 5-session
holding-period auto-close: each `holding_period` exit is REOPENED, and the
duplicate re-entry the old code opened moments later is DELETED.

It is the most destructive script in the repo — it rewrites the trade ledger
through a full-replace `save_trades` — so what needs pinning is the blast
radius, not the happy path:

* it must be a genuine NO-OP when there is nothing to undo (it is documented as
  idempotent, and someone will run it twice);
* it must delete ONLY the matching roll re-entry. The roll match is
  ticker + action + a 10-minute window; loosening any leg would drop a real,
  independent position and its P&L with it.

The repo is stubbed throughout — the conftest already points the DB elsewhere,
but this module's failure mode is "wrote the wrong thing", so the test asserts
on what it tried to save rather than on a round trip.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.db import unwind_time_cap as u


@pytest.fixture
def ledger(monkeypatch):
    """Captures what `unwind()` would persist; `saved` stays None if it never
    calls save_trades (the no-op path)."""
    box: dict = {"trades": [], "saved": None}
    monkeypatch.setattr(u.repo, "load_trades", lambda: box["trades"])
    monkeypatch.setattr(u.repo, "save_trades",
                        lambda t: box.__setitem__("saved", t))
    return box


_T0 = datetime(2026, 5, 1, 15, 0, 0)


def _capped(ticker="AAPL", action="BUY", exit_at=_T0, entry_price=100.0) -> dict:
    return {"ticker": ticker, "action": action, "status": "CLOSED",
            "exit_reason": "holding_period", "entry_price": entry_price,
            "entry_date": "2026-04-24", "entry_datetime": "2026-04-24T15:00:00",
            "exit_date": exit_at.date().isoformat(),
            "exit_datetime": exit_at.isoformat(),
            "exit_price": 110.0, "exit_ref_close": 109.0,
            "exit_ref_close_date": "2026-05-01",
            "exit_decision_datetime": exit_at.isoformat()}


def _roll(ticker="AAPL", action="BUY", after_seconds=5, base=_T0, **kw) -> dict:
    t = {"ticker": ticker, "action": action, "status": "OPEN",
         "entry_price": 110.0,
         "entry_datetime": (base + timedelta(seconds=after_seconds)).isoformat(),
         "entry_date": (base + timedelta(seconds=after_seconds)).date().isoformat()}
    t.update(kw)
    return t


# ── the no-op path ──────────────────────────────────────────────────────────

def test_nothing_to_undo_does_not_touch_the_ledger(ledger):
    """Documented as idempotent. Writing an unchanged ledger back would still be
    a full-replace of the trades table for no reason."""
    ledger["trades"] = [{"ticker": "AAPL", "status": "OPEN"},
                        {"ticker": "MSFT", "status": "CLOSED",
                         "exit_reason": "trailing_stop"}]
    u.unwind()
    assert ledger["saved"] is None


def test_running_twice_is_a_no_op(ledger):
    ledger["trades"] = [_capped(), _roll()]
    u.unwind()
    first = ledger["saved"]
    assert first is not None

    ledger["trades"], ledger["saved"] = first, None
    u.unwind()
    assert ledger["saved"] is None, "second run rewrote an already-unwound ledger"


# ── reopening ───────────────────────────────────────────────────────────────

def test_a_capped_close_is_reopened_with_its_original_cost_basis(ledger):
    ledger["trades"] = [_capped(entry_price=100.0)]
    u.unwind()
    t = ledger["saved"][0]
    assert t["status"] == "OPEN"
    assert t["entry_price"] == 100.0, "the original entry is the true cost basis"
    assert t["entry_datetime"] == "2026-04-24T15:00:00"


@pytest.mark.parametrize("field", u._EXIT_FIELDS)
def test_every_exit_field_is_cleared(ledger, field):
    """A leftover exit_price or exit_reason on an OPEN row is a contradiction the
    return engines read differently — `gross_return_pct` prefers `exit_price`
    over the live mark, so a stale one would freeze the position's return."""
    ledger["trades"] = [_capped()]
    u.unwind()
    assert ledger["saved"][0][field] is None


def test_exit_fields_covers_what_a_close_actually_writes():
    """Drift guard: a close field added later but missing from `_EXIT_FIELDS`
    would survive the reopen and silently contradict the OPEN status."""
    assert set(u._EXIT_FIELDS) >= {
        "exit_date", "exit_datetime", "exit_price", "exit_reason"}


# ── roll-re-entry removal ───────────────────────────────────────────────────

def test_the_roll_re_entry_is_dropped(ledger):
    ledger["trades"] = [_capped(), _roll(after_seconds=5)]
    u.unwind()
    saved = ledger["saved"]
    assert len(saved) == 1, "the duplicate re-entry survived"
    assert saved[0]["entry_price"] == 100.0     # the ORIGINAL, not the roll


def test_a_re_entry_outside_the_window_is_kept(ledger):
    """Beyond 10 minutes it is a genuine new decision, not the old code's
    instant roll — deleting it would erase a real position."""
    ledger["trades"] = [_capped(), _roll(after_seconds=u._ROLL_WINDOW_SECONDS + 1)]
    u.unwind()
    assert len(ledger["saved"]) == 2


def test_a_re_entry_BEFORE_the_cap_exit_is_kept(ledger):
    """The window is one-sided: a position opened before the cap close cannot be
    its roll."""
    ledger["trades"] = [_capped(), _roll(after_seconds=-30)]
    u.unwind()
    assert len(ledger["saved"]) == 2


def test_a_different_ticker_or_action_is_never_treated_as_a_roll(ledger):
    ledger["trades"] = [_capped(ticker="AAPL", action="BUY"),
                        _roll(ticker="MSFT", action="BUY"),
                        _roll(ticker="AAPL", action="SELL")]
    u.unwind()
    assert len(ledger["saved"]) == 3


def test_another_capped_close_is_not_consumed_as_a_roll(ledger):
    """Both are being reopened; treating one as the other's roll would delete a
    position that is itself due to be restored."""
    ledger["trades"] = [_capped(exit_at=_T0),
                        _capped(exit_at=_T0 + timedelta(seconds=5))]
    u.unwind()
    saved = ledger["saved"]
    assert len(saved) == 2
    assert all(t["status"] == "OPEN" for t in saved)


def test_each_roll_is_consumed_at_most_once(ledger):
    """Two cap closes seconds apart must not both claim the same re-entry."""
    c1 = _capped(exit_at=_T0)
    c2 = _capped(exit_at=_T0 + timedelta(seconds=1))
    ledger["trades"] = [c1, c2, _roll(after_seconds=5)]
    u.unwind()
    saved = ledger["saved"]
    assert len(saved) == 2, "one roll removal should leave both originals"
    assert all(t["status"] == "OPEN" for t in saved)


def test_a_capped_close_with_no_roll_just_reopens(ledger):
    ledger["trades"] = [_capped(), {"ticker": "MSFT", "status": "OPEN"}]
    u.unwind()
    saved = ledger["saved"]
    assert len(saved) == 2
    assert saved[0]["status"] == "OPEN" and saved[0]["exit_reason"] is None


def test_unrelated_trades_are_passed_through_untouched(ledger):
    other = {"ticker": "TSLA", "status": "CLOSED", "exit_reason": "trailing_stop",
             "exit_price": 42.0}
    ledger["trades"] = [_capped(), other]
    u.unwind()
    kept = [t for t in ledger["saved"] if t["ticker"] == "TSLA"]
    assert kept == [other], "a non-capped close was modified"


def test_a_missing_exit_timestamp_blocks_roll_matching(ledger):
    """Without a cap-exit time there is no window to match in — the migration
    must reopen the position but delete nothing on a guess."""
    c = _capped()
    c["exit_datetime"] = None
    ledger["trades"] = [c, _roll(after_seconds=5)]
    u.unwind()
    assert len(ledger["saved"]) == 2
    assert ledger["saved"][0]["status"] == "OPEN"


# ── the timestamp parser ────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "not-a-date", 12345, "2026-13-45"])
def test_unparseable_timestamps_are_none_not_exceptions(value):
    assert u._parse(value) is None


def test_parse_accepts_the_stored_formats():
    assert u._parse("2026-05-01T15:00:00") == datetime(2026, 5, 1, 15, 0)
    assert u._parse("2026-05-01") == datetime(2026, 5, 1)
