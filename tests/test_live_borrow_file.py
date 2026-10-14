"""A live short opens only on IBKR's own short-stock file (user 2026-10-12: "Make sure the live is using the real
fees and lendability from IBKR"): with no current file the pick WAITS — journaled `no_borrow_file_wait`, not consumed,
no same-bar fallback — and opens at the next tick that has the file."""
from __future__ import annotations

from datetime import datetime

import pytest

from config.settings import settings
from src.signals import sel_short as ss
from tests.test_sel_short_turn import ET, _pick, on  # noqa: F401  (the fixture)


def test_no_ibkr_file_the_pick_waits_then_opens(monkeypatch, on):  # noqa: F811
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_short_borrow_gate", True)
    monkeypatch.setattr(settings, "ibkr_borrow_require_file", True)
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", False)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    monkeypatch.setattr(settings, "enable_sel_short_vol_fallback", True)
    rec = _pick("VVV", arm="vol", target=9.0, pre5=9.0)
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: [rec])
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.5)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:03:00+00:00")
    fallbacks = []
    monkeypatch.setattr(ss, "fallback_candidates", lambda *a, **k: fallbacks.append(a) or [], raising=False)
    # tick 1: no IBKR file at all
    assert tracker.record_sel_short_trades(run_id="r1") == 0
    assert not [t for t in tracker._load_trades() if t.get("entry_mechanism") == "sel_short"]
    assert ss.consumed(datetime.fromisoformat(rec["bar_end"]).date()) == {}            # still pending
    lines = ss.entries_path(datetime.fromisoformat(rec["bar_end"]).date()).read_text(encoding="utf-8")
    assert "no_borrow_file_wait" in lines and "no_borrow_retry" not in lines
    # tick 2: IBKR's file is back and lends the name
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)
    monkeypatch.setattr(ibkr_borrow, "short_block",
                        lambda ticker, price, base=None, now=None, max_fee_pct="default":
                        (None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)))
    assert tracker.record_sel_short_trades(run_id="r2") == 1
