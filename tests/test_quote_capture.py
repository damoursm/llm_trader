"""Live-NBBO capture rate on the Execution tab
(`broker_forensics.quote_capture`).

The metric is the live quote feed's success rate as the ORDER PATH sees it.
Two properties carry it and are easy to get wrong:

* the denominator counts only events that ASK for a book (fill repairs / kills
  never price one, so counting them understates the feed), and
* it is EPOCH-GATED from the data — the bid/ask columns landed 2026-08-31, so
  pre-feature rows are structurally NULL and must be excluded, not scored as
  failures. "Not accruing" is a distinct state from 0%.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.broker_forensics import compute_forensics, quote_capture


def _orders(rows):
    return pd.DataFrame(rows)


def _row(ts, event="SUBMIT", bid=None, ask=None, ticker="AAPL"):
    return {"submitted_at": ts, "event": event, "ticker": ticker,
            "side": "BUY", "status": "Filled", "intent": "ENTRY",
            "bid_at_submit": bid, "ask_at_submit": ask, "fill_price": 100.0,
            "slippage_bps": 1.0}


def test_empty_or_missing_columns_is_not_accruing():
    assert quote_capture(pd.DataFrame())["accruing"] is False
    legacy = _orders([{"submitted_at": "2026-08-01T15:00:00+00:00",
                       "event": "SUBMIT", "ticker": "A", "side": "BUY"}])
    assert quote_capture(legacy)["accruing"] is False


def test_pre_epoch_rows_are_excluded_not_scored_as_failures():
    """The decisive case: months of NULL history plus one captured book must
    read as 100% since the epoch, never as a near-zero rate."""
    rows = [_row(f"2026-08-0{i}T15:00:00+00:00") for i in range(1, 9)]
    rows.append(_row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02))
    qc = quote_capture(_orders(rows))
    assert qc["accruing"] is True
    assert qc["since"].startswith("2026-08-31")
    assert qc["n_eligible"] == 1
    assert qc["rate"] == pytest.approx(100.0)


def test_rate_counts_only_book_pricing_events():
    """Fill repairs / kills never call _quote_for — including them would
    understate the feed by diluting the denominator."""
    rows = [
        _row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02),
        _row("2026-08-31T15:01:00+00:00", event="SETTLE_KILL"),
        _row("2026-08-31T15:02:00+00:00", event="FILL_REFRESH"),
        _row("2026-08-31T15:03:00+00:00", event="SETTLE_FILL"),
    ]
    qc = quote_capture(_orders(rows))
    assert qc["n_eligible"] == 1, "only the SUBMIT asks for a book"
    assert qc["rate"] == pytest.approx(100.0)


def test_partial_capture_and_session_split():
    rows = [
        _row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02),   # 11:00 ET rth
        _row("2026-08-31T15:30:00+00:00", bid=99.9, ask=100.1),     # rth
        _row("2026-08-31T16:00:00+00:00"),                          # rth, no book
        _row("2026-09-01T02:00:00+00:00"),                          # 22:00 ET overnight
    ]
    qc = quote_capture(_orders(rows))
    assert qc["n_eligible"] == 4
    assert qc["n_captured"] == 2
    assert qc["rate"] == pytest.approx(50.0)
    by = {r["session"]: r for r in qc["by_session"]}
    assert by["rth"]["orders"] == 3 and by["rth"]["captured"] == 2
    assert by["overnight"]["captured"] == 0
    # median quoted half-spread over the captured books: 2 bp and 10 bp → 6
    assert qc["median_half_bps"] == pytest.approx(6.0, abs=0.5)


def test_zero_percent_since_epoch_is_reported_as_zero_not_absent():
    """Feed broke AFTER working: still accruing, rate collapses. This is the
    failure the tile exists to surface, and it must NOT read as 'not
    measuring'."""
    rows = [
        _row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02),
        *[_row(f"2026-09-0{i}T15:00:00+00:00") for i in range(1, 9)],
    ]
    qc = quote_capture(_orders(rows))
    assert qc["accruing"] is True
    assert qc["n_eligible"] == 9 and qc["n_captured"] == 1
    assert qc["rate"] == pytest.approx(11.1, abs=0.2)


def test_crossed_or_zero_book_does_not_count_as_captured():
    rows = [
        _row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02),
        _row("2026-08-31T15:05:00+00:00", bid=0.0, ask=100.0),
    ]
    qc = quote_capture(_orders(rows))
    assert qc["n_captured"] == 1


def test_compute_forensics_exposes_the_block():
    rows = [_row("2026-08-31T15:00:00+00:00", bid=99.98, ask=100.02)]
    rep = compute_forensics(_orders(rows), pd.DataFrame())
    assert "quote_capture" in rep
    assert rep["quote_capture"]["rate"] == pytest.approx(100.0)


def test_execution_tab_renders_both_states():
    """The tile and section must render for accruing AND not-accruing without
    raising — a dashboard block that throws takes the whole tab down."""
    from dashboard.app import _quote_capture_kpi, _quote_capture_section
    for qc in ({}, {"accruing": False},
               {"accruing": True, "rate": 97.5, "n_captured": 39,
                "n_eligible": 40, "since": "2026-08-31T15:00:00+00:00",
                "median_half_bps": 2.4,
                "by_session": [{"session": "rth", "orders": 40,
                                "captured": 39, "rate": 97.5}]}):
        assert _quote_capture_kpi(qc) is not None
        assert _quote_capture_section(qc) is not None
