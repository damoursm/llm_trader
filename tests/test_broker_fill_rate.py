"""The fill rate must count intended TRADES, not order events (2026-08-17).

The Execution tab reported `fill_outcomes()["fill_rate"]` = filled EVENTS /
terminal EVENTS, which is the wrong unit for either question anyone asks of it
and read 37.6% against a 20.1% per-retry reality. Two compounding reasons:

  1. one intended trade emits several events (SUBMIT, a SETTLE_REANCHOR every
     ~6s, then SETTLE_KILL or SETTLE_FILL), so a trade that re-anchored four
     times before filling counted as one fill and several non-fills;
  2. the event denominator DROPPED every row still marked `Submitted` as
     "working" — and most of those are re-anchor events belonging to intents
     that were subsequently killed, so the clearest failures were the rows being
     excluded.

Both are invisible from the number itself: any of 20%, 38% or 76% looks like a
plausible fill rate for a marketable-limit strategy. Only the UNIT gives it
away, so the unit is what these tests assert.

There are two legitimate units and the tab now shows both, because either one
alone misleads:

  PER RETRY  20.1%  -- of every order placed, did it fill? Execution quality,
                       low by design under settle-or-kill.
  PER TRADE  75.8%  -- of the trades we decided to make, did we get on? This is
                       whether the strategy is being executed at all.

They differ by 3.76x because a trade is resubmitted under a fresh `-rN` ref each
time, observed as deep as `-r85`.
"""

import pandas as pd

from src.analysis.broker_forensics import (base_ref, fill_outcomes,
                                           fill_rate_by_attempt)


def _orders(rows):
    return pd.DataFrame(rows)


def _rates(rows):
    out = fill_rate_by_attempt(_orders(rows))
    return out["per_retry"], out["per_trade"], out["by_intent"]


# ── the two units ───────────────────────────────────────────────────────────

def test_retries_of_one_trade_pool_onto_one_trade():
    """THE distinction. Three resubmissions of one intended trade, the last of
    which fills: 1/3 per RETRY (execution quality) but 1/1 per TRADE (we got it
    on). Reporting either alone misleads — measured 20.1% vs 75.8% live."""
    rows = [
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
        {"run_id": "r2", "client_ref": "abc-r1", "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
        {"run_id": "r3", "client_ref": "abc-r2", "intent": "ENTRY", "status": "Filled", "filled_qty": 10},
    ]
    retry, trade, _ = _rates(rows)
    assert (retry["n"], retry["filled"], retry["rate"]) == (3, 1, 33.3)
    assert (trade["n"], trade["filled"], trade["rate"]) == (1, 1, 100.0)
    assert trade["avg_retries"] == 3.0


def test_a_trade_that_never_fills_counts_against_both():
    rows = [
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
        {"run_id": "r2", "client_ref": "abc-r1", "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
    ]
    retry, trade, _ = _rates(rows)
    assert retry["rate"] == 0.0 and trade["rate"] == 0.0


def test_base_ref_strips_only_the_retry_suffix():
    """Exits are `<id>-exit` and retry to `<id>-exit-r1`; the `-exit` part is
    identity, not a retry, and collapsing it would merge an entry with its exit."""
    assert base_ref("abc") == "abc"
    assert base_ref("abc-r7") == "abc"
    assert base_ref("abc-exit") == "abc-exit"
    assert base_ref("abc-exit-r11") == "abc-exit"
    assert base_ref(None) == ""


def test_entry_and_exit_stay_separate_trades():
    """An entry and its exit share a recommendation id; they must not pool."""
    rows = [
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Filled", "filled_qty": 1},
        {"run_id": "r1", "client_ref": "abc-exit", "intent": "EXIT", "status": "Cancelled", "filled_qty": 0},
    ]
    _retry, trade, by_intent = _rates(rows)
    assert trade["n"] == 2 and trade["filled"] == 1
    assert by_intent["ENTRY"]["rate"] == 100.0
    assert by_intent["EXIT"]["rate"] == 0.0


def test_repeated_events_for_one_ref_are_one_attempt():
    """SUBMIT + several SETTLE_REANCHOR rows under the SAME ref are one order,
    not several — the event-level count is what read ~2x high."""
    rows = [
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Submitted", "filled_qty": 0},
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Submitted", "filled_qty": 0},
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Filled", "filled_qty": 5},
    ]
    retry, trade, _ = _rates(rows)
    assert retry["n"] == 1 and trade["n"] == 1
    assert retry["rate"] == 100.0


def test_an_intent_stuck_at_submitted_is_a_miss_the_event_metric_cannot_see():
    """The sharpest form of the old inflation. An order whose only rows are
    `Submitted` contributes NOTHING to the event denominator — every row is
    classed "working" and dropped — so the event metric scores a clean 100%
    while half the orders never filled."""
    rows = [
        {"run_id": "r1", "client_ref": "a", "intent": "ENTRY", "status": "Filled", "filled_qty": 5},
        {"run_id": "r1", "client_ref": "b", "intent": "ENTRY", "status": "Submitted", "filled_qty": 0},
        {"run_id": "r1", "client_ref": "b", "intent": "ENTRY", "status": "Submitted", "filled_qty": 0},
    ]
    assert fill_outcomes(_orders(rows))["fill_rate"] == 100.0, "fixture stopped reproducing it"
    retry, trade, _ = _rates(rows)
    assert retry["rate"] == 50.0 and trade["rate"] == 50.0


# ── definitions ─────────────────────────────────────────────────────────────

def test_a_partial_fill_counts_as_filled():
    """The position WAS established, just smaller — a fill, not a miss."""
    rows = [{"run_id": "r1", "client_ref": "abc", "intent": "ENTRY",
             "status": "Cancelled", "filled_qty": 3}]
    retry, trade, _ = _rates(rows)
    assert retry["rate"] == 100.0 and trade["rate"] == 100.0


def test_skipped_rows_are_not_attempts():
    """A duplicate ref / nothing-to-close / dry-run row never reached the
    broker, so it is not an execution attempt and must not dilute either rate."""
    rows = [
        {"run_id": "r1", "client_ref": "a", "intent": "ENTRY", "status": "Filled", "filled_qty": 5},
        {"run_id": "r1", "client_ref": "b", "intent": "ENTRY",
         "status": "DUPLICATE_REF_NOT_SUBMITTED", "filled_qty": 0},
    ]
    retry, trade, _ = _rates(rows)
    assert retry["n"] == 1 and trade["n"] == 1 and trade["rate"] == 100.0


def test_empty_and_malformed_input_is_safe():
    for df in (None, pd.DataFrame(), pd.DataFrame({"status": ["Filled"]})):
        out = fill_rate_by_attempt(df)
        assert out["per_retry"]["rate"] is None
        assert out["per_trade"]["rate"] is None
        assert out["by_intent"] == {}


def test_blank_client_refs_are_ignored():
    """Rows with no ref cannot be attributed to a trade; counting them would
    invent misses."""
    rows = [
        {"run_id": "r1", "client_ref": "abc", "intent": "ENTRY", "status": "Filled", "filled_qty": 5},
        {"run_id": "r1", "client_ref": "", "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
        {"run_id": "r1", "client_ref": None, "intent": "ENTRY", "status": "Cancelled", "filled_qty": 0},
    ]
    retry, trade, _ = _rates(rows)
    assert retry["rate"] == 100.0 and trade["rate"] == 100.0
