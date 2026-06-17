"""Look-ahead / forming-bar audit (item #6).

The pipeline runs intraday, so a freshly-fetched daily bar dated *today* is still
forming until the 16:00 ET close. ``market_data._drop_forming_bar`` removes it so
no daily indicator (or NAV walk) ever reads an unclosed bar. These tests freeze
the clock and assert:
  * before the close, today's bar is dropped; after it, kept;
  * a still-forming bar cannot retroactively change a past indicator value
    (the dropped frame is bit-identical to the completed-bars-only frame).
"""

from datetime import datetime

import pandas as pd

import src.data.market_data as md


def _freeze(monkeypatch, frozen: datetime):
    class _DT:
        @staticmethod
        def now(tz=None):
            return frozen
    monkeypatch.setattr(md, "_datetime", _DT)


def _df(tz):
    """Three daily bars: two completed (06-11, 06-12) + today (06-15)."""
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-11", tz=tz),
                            pd.Timestamp("2026-06-12", tz=tz),
                            pd.Timestamp("2026-06-15", tz=tz)])
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)


# ── drop/keep around the close ───────────────────────────────────────────────

def test_today_bar_dropped_before_close(monkeypatch):
    _freeze(monkeypatch, datetime(2026, 6, 15, 11, 0, tzinfo=md._NY_TZ))
    out = md._drop_forming_bar(_df("America/New_York"))
    assert len(out) == 2
    assert md._bar_session_date(out.index[-1]).isoformat() == "2026-06-12"


def test_today_bar_kept_after_close(monkeypatch):
    _freeze(monkeypatch, datetime(2026, 6, 15, 16, 30, tzinfo=md._NY_TZ))
    out = md._drop_forming_bar(_df("America/New_York"))
    assert len(out) == 3


def test_polygon_naive_index_also_dropped(monkeypatch):
    """Polygon's tz-naive midnight-UTC index must drop today's bar too."""
    _freeze(monkeypatch, datetime(2026, 6, 15, 11, 0, tzinfo=md._NY_TZ))
    out = md._drop_forming_bar(_df(None))
    assert len(out) == 2


def test_all_past_bars_unchanged(monkeypatch):
    """No bar dated today → nothing dropped even before the close."""
    _freeze(monkeypatch, datetime(2026, 6, 16, 11, 0, tzinfo=md._NY_TZ))
    df = _df("America/New_York")
    out = md._drop_forming_bar(df)
    assert len(out) == 3


def test_none_and_empty_safe(monkeypatch):
    _freeze(monkeypatch, datetime(2026, 6, 15, 11, 0, tzinfo=md._NY_TZ))
    assert md._drop_forming_bar(None) is None
    empty = pd.DataFrame({"Close": []})
    assert md._drop_forming_bar(empty).empty


# ── no retroactive change to a past indicator value ──────────────────────────

def test_forming_bar_cannot_change_past_indicator(monkeypatch):
    """A causal indicator computed as-of the last completed bar must be identical
    whether or not a still-forming bar is appended — i.e. no look-ahead."""
    _freeze(monkeypatch, datetime(2026, 6, 15, 11, 0, tzinfo=md._NY_TZ))
    full = _df("America/New_York")           # includes today's forming bar
    completed_only = full.iloc[:2].copy()    # what the indicator legitimately sees

    dropped = md._drop_forming_bar(full)
    pd.testing.assert_frame_equal(dropped, completed_only)

    # A causal indicator (trailing mean) over the dropped frame equals the one
    # over the completed-only frame — the forming bar moved nothing.
    def last_sma2(frame):
        return float(frame["Close"].rolling(2).mean().iloc[-1])

    assert last_sma2(dropped) == last_sma2(completed_only)
