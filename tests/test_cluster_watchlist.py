"""Insider cluster watchlist (`src/data/cluster_watchlist.py`).

Extends a one-day insider-cluster detection into a 10-day tracking window by
injecting the ticker back into the analysis universe on every subsequent run.
So this file decides what gets ANALYSED — a broken expiry either drops a name
the system wanted to keep watching, or pins a stale one in the universe forever.

Two properties carry that:

* **detection is recorded once.** Re-detecting the same cluster on a later run
  must not refresh `detected_at`, or a ticker that keeps firing never expires —
  the watch becomes permanent, which is exactly the failure the window exists
  to bound;
* **expiry is measured from detection, not from last-seen**, and it runs on
  every update so an entry cannot outlive the window just because nothing new
  was detected that day.

Every test points `WATCHLIST_PATH` at tmp_path — the default is the relative
`cache/cluster_watchlist.json`, which is the live one.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from src.data import cluster_watchlist as cw


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "WATCHLIST_PATH", tmp_path / "cluster_watchlist.json")


def _sig(detected=True, size=4, summary="Tim Cook (CEO) + 3 others"):
    return SimpleNamespace(insider_cluster_detected=detected,
                           insider_cluster_size=size,
                           insider_summary=summary)


_TODAY = date(2026, 8, 14)


# ── persistence ─────────────────────────────────────────────────────────────

def test_a_missing_file_loads_as_empty():
    assert cw.load_cluster_watchlist() == {}


def test_a_corrupt_file_loads_as_empty_rather_than_raising():
    """This runs inside universe construction; a truncated write must not take
    the tick down."""
    cw.WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cw.WATCHLIST_PATH.write_text("{not json", encoding="utf-8")
    assert cw.load_cluster_watchlist() == {}


def test_save_then_load_round_trips():
    raw = {"AAPL": {"detected_at": "2026-08-10", "cluster_size": 4,
                    "insider_summary": "x"}}
    cw.save_cluster_watchlist(raw)
    assert cw.load_cluster_watchlist() == raw


def test_save_creates_the_cache_directory():
    cw.save_cluster_watchlist({"AAPL": {"detected_at": "2026-08-10"}})
    assert cw.WATCHLIST_PATH.exists()
    assert json.loads(cw.WATCHLIST_PATH.read_text(encoding="utf-8"))


# ── detection ───────────────────────────────────────────────────────────────

def test_a_detected_cluster_is_added_with_todays_date():
    raw = cw.update_cluster_watchlist({"AAPL": _sig()}, {}, today=_TODAY)
    assert raw["AAPL"]["detected_at"] == str(_TODAY)
    assert raw["AAPL"]["cluster_size"] == 4
    assert "Tim Cook" in raw["AAPL"]["insider_summary"]


def test_a_signal_without_a_cluster_is_not_added():
    assert cw.update_cluster_watchlist({"AAPL": _sig(detected=False)}, {},
                                       today=_TODAY) == {}


def test_a_signal_missing_the_cluster_attribute_is_not_added():
    """Most TickerSignals never carry these fields; `getattr` defaults must read
    as 'no cluster', not as a detection."""
    assert cw.update_cluster_watchlist({"AAPL": SimpleNamespace()}, {},
                                       today=_TODAY) == {}


def test_re_detection_does_not_refresh_the_detection_date():
    """The load-bearing one. Refreshing on every re-detection would make a
    persistently-clustered ticker immortal in the universe, which is precisely
    what the 10-day bound exists to prevent."""
    day1 = _TODAY - timedelta(days=5)
    raw = cw.update_cluster_watchlist({"AAPL": _sig()}, {}, today=day1)
    raw = cw.update_cluster_watchlist({"AAPL": _sig()}, raw, today=_TODAY)
    assert raw["AAPL"]["detected_at"] == str(day1)


def test_cluster_size_defaults_when_absent():
    sig = SimpleNamespace(insider_cluster_detected=True)
    raw = cw.update_cluster_watchlist({"AAPL": sig}, {}, today=_TODAY)
    assert raw["AAPL"]["cluster_size"] == 3      # the detection threshold


# ── expiry ──────────────────────────────────────────────────────────────────

def test_an_entry_expires_after_the_watch_window():
    old = _TODAY - timedelta(days=cw.WATCH_DAYS + 1)
    raw = {"AAPL": {"detected_at": str(old), "cluster_size": 3, "insider_summary": ""}}
    assert cw.update_cluster_watchlist({}, raw, today=_TODAY) == {}


def test_an_entry_inside_the_window_survives():
    fresh = _TODAY - timedelta(days=cw.WATCH_DAYS - 1)
    raw = {"AAPL": {"detected_at": str(fresh), "cluster_size": 3, "insider_summary": ""}}
    assert "AAPL" in cw.update_cluster_watchlist({}, raw, today=_TODAY)


def test_expiry_runs_even_when_nothing_was_detected_today():
    """Otherwise a quiet week leaves stale names in the universe indefinitely."""
    old = _TODAY - timedelta(days=cw.WATCH_DAYS + 5)
    raw = {"OLD": {"detected_at": str(old)}, "NEW": {"detected_at": str(_TODAY)}}
    out = cw.update_cluster_watchlist({}, raw, today=_TODAY)
    assert set(out) == {"NEW"}


def test_an_unparseable_detection_date_expires_rather_than_persisting():
    """A malformed entry defaults to the year 2000 — comfortably past any
    window — so a corrupt row leaves the universe instead of sticking forever."""
    raw = {"AAPL": {"cluster_size": 3}}          # no detected_at at all
    assert cw.update_cluster_watchlist({}, raw, today=_TODAY) == {}


# ── the context ─────────────────────────────────────────────────────────────

def test_context_reports_elapsed_and_remaining_days():
    detected = _TODAY - timedelta(days=3)
    raw = {"AAPL": {"detected_at": str(detected), "cluster_size": 4,
                    "insider_summary": "s"}}
    ctx = cw.build_cluster_watchlist_context(raw, today=_TODAY)
    e = ctx.entries[0]
    assert e.ticker == "AAPL" and e.days_elapsed == 3
    assert e.days_remaining == cw.WATCH_DAYS - 3
    assert ctx.active_tickers == ["AAPL"]


def test_days_remaining_never_goes_negative():
    """The context is built from whatever is on disk, which can include an entry
    the update pass has not expired yet."""
    raw = {"AAPL": {"detected_at": str(_TODAY - timedelta(days=99))}}
    assert cw.build_cluster_watchlist_context(raw, today=_TODAY).entries[0].days_remaining == 0


def test_an_empty_watchlist_produces_an_empty_context():
    ctx = cw.build_cluster_watchlist_context({}, today=_TODAY)
    assert ctx.entries == [] and ctx.active_tickers == []
    assert "No active" in ctx.summary


def test_entries_are_sorted_and_summarised():
    raw = {"MSFT": {"detected_at": str(_TODAY)}, "AAPL": {"detected_at": str(_TODAY)}}
    ctx = cw.build_cluster_watchlist_context(raw, today=_TODAY)
    assert ctx.active_tickers == ["AAPL", "MSFT"]
    assert "AAPL" in ctx.summary and "MSFT" in ctx.summary
    assert "2 ticker(s)" in ctx.summary


def test_the_full_cycle_holds_a_ticker_for_exactly_the_window(tmp_path):
    """End to end over simulated days: detected once, present through the
    window, gone the day after."""
    detected_on = date(2026, 8, 1)
    raw = cw.update_cluster_watchlist({"AAPL": _sig()}, {}, today=detected_on)
    for offset in range(0, cw.WATCH_DAYS):
        raw = cw.update_cluster_watchlist({}, raw, today=detected_on + timedelta(days=offset))
        ctx = cw.build_cluster_watchlist_context(raw, today=detected_on + timedelta(days=offset))
        assert "AAPL" in ctx.active_tickers, f"dropped on day {offset}"
    raw = cw.update_cluster_watchlist(
        {}, raw, today=detected_on + timedelta(days=cw.WATCH_DAYS + 1))
    assert raw == {}
