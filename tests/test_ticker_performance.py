"""Per-ticker simulated performance (2026-07-25).

Answers "how is the strategy doing on THIS name?" over every scored ticker-day,
independent of whether the gates ever let it trade — the trade ledger only holds
gate survivors, so a pinned ticker that never clears the confidence bar would
otherwise be invisible while still being scored every tick.

The properties pinned here are the ones that make the number honest:
  * orientation — a BEARISH call on a stock that FELL is a WIN;
  * no-view days are excluded, not counted as zero (a flat score next to a huge
    move would otherwise manufacture a return);
  * a missing forward bar is an absent observation, not a zero.

All synthetic, no network, no production DB.
"""

import pandas as pd
import pytest

from src.analysis import ticker_performance as tp


def _panel(rows):
    """rows: (ticker, score, fwd_5d) — plus the columns build_panel supplies."""
    return pd.DataFrame([
        {"ticker": t, "universe_source": "watchlist", "signal_date": f"2026-07-{1+i:02d}",
         "combined_score": s, "confidence": 0.9, "direction":
             "BULLISH" if s > 0 else "BEARISH", "fwd_ret_5d": f}
        for i, (t, s, f) in enumerate(rows)
    ])


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Isolate from the real recommendations table and trade ledger."""
    monkeypatch.setattr(tp, "_funnel_counts", lambda days: pd.DataFrame(
        columns=["ticker", "recs", "dir_recs", "actionable"]))
    monkeypatch.setattr(tp, "_trade_counts", lambda: {})


def compute(rows, **kw):
    return tp.compute_ticker_perf(horizons=(5,), panel=_panel(rows),
                                  **kw).set_index("ticker")


# ── orientation ────────────────────────────────────────────────────────────

def test_bearish_call_on_a_falling_stock_is_a_win():
    """The whole point of orienting by the signal's own direction: a short that
    worked must not read as a loss just because the stock went down."""
    out = compute([("DOWN", -0.5, -10.0)])
    assert out.loc["DOWN", "ret_5d"] == pytest.approx(10.0)
    assert out.loc["DOWN", "hit_5d"] == 100.0


def test_bullish_call_on_a_falling_stock_is_a_loss():
    out = compute([("WRONG", 0.5, -10.0)])
    assert out.loc["WRONG", "ret_5d"] == pytest.approx(-10.0)
    assert out.loc["WRONG", "hit_5d"] == 0.0


def test_both_directions_score_symmetrically():
    out = compute([("A", 0.5, 10.0), ("B", -0.5, -10.0)])
    assert out.loc["A", "ret_5d"] == out.loc["B", "ret_5d"] == pytest.approx(10.0)


# ── what counts as an observation ──────────────────────────────────────────

def test_no_view_days_are_excluded_not_zeroed():
    """A ~flat score beside a huge move must contribute nothing. Counting it as
    a 0 return would silently dilute the ticker's real record toward zero."""
    out = compute([("FLAT", 0.001, 50.0)])
    assert out.loc["FLAT", "view_days"] == 0
    assert pd.isna(out.loc["FLAT", "ret_5d"])
    assert out.loc["FLAT", "signal_days"] == 1   # still SCORED, just no view


def test_missing_forward_bar_is_not_a_zero():
    """The newest signal days have no forward bar yet — they must not drag the
    mean toward zero."""
    p = _panel([("X", 0.5, 10.0), ("X", 0.5, None)])
    out = tp.compute_ticker_perf(horizons=(5,), panel=p).set_index("ticker")
    assert out.loc["X", "ret_5d"] == pytest.approx(10.0)
    assert out.loc["X", "n_5d"] == 1        # one usable observation
    assert out.loc["X", "view_days"] == 2   # both days had a view


def test_hit_rate_counts_only_scored_observations():
    out = compute([("M", 0.5, 10.0), ("M", 0.5, -10.0), ("M", 0.5, 10.0)])
    assert out.loc["M", "hit_5d"] == pytest.approx(66.7, abs=0.1)
    assert out.loc["M", "n_5d"] == 3


# ── grouping / filtering ───────────────────────────────────────────────────

def test_direction_mix_is_reported_per_ticker():
    out = compute([("D", 0.5, 1.0), ("D", -0.5, -1.0), ("D", 0.001, 99.0)])
    assert out.loc["D", "buy_days"] == 1
    assert out.loc["D", "sell_days"] == 1
    assert out.loc["D", "view_days"] == 2
    assert out.loc["D", "signal_days"] == 3


def test_source_filter_selects_one_universe_source():
    p = _panel([("W", 0.5, 10.0), ("T", 0.5, 10.0)])
    p.loc[p["ticker"] == "T", "universe_source"] = "trending"
    out = tp.compute_ticker_perf(horizons=(5,), panel=p, source="watchlist")
    assert list(out["ticker"]) == ["W"]


def test_unknown_source_returns_empty_not_error():
    out = tp.compute_ticker_perf(horizons=(5,), panel=_panel([("W", 0.5, 1.0)]),
                                 source="nope")
    assert out.empty


def test_min_days_drops_thin_names():
    p = _panel([("THIN", 0.5, 10.0), ("THICK", 0.5, 10.0), ("THICK", 0.5, 10.0)])
    out = tp.compute_ticker_perf(horizons=(5,), panel=p, min_days=2)
    assert list(out["ticker"]) == ["THICK"]


def test_empty_panel_degrades_quietly():
    assert tp.compute_ticker_perf(panel=pd.DataFrame()).empty
