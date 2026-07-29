"""Per-ticker synthesis-arm bake-off (2026-07-25).

Shadow arms ask EVERY prompt arm about EVERY ticker each tick, so the arms
become comparable on the SAME ticker-day instead of over whichever runs their
coin came up on. That pairing is the point: the 2026-07-22 bake-off showed the
unpaired shape of this comparison produces pure window artifacts.

These tests pin the two properties that make the evaluation trustworthy:
  * a declined call (HOLD/WATCH) earns 0, not NaN — the dual arm is explicitly
    allowed to decline, so scoring only its directional calls would hide the
    behaviour it was built for;
  * the head-to-head edge is measured on the DISAGREEMENT subset only, since a
    shared call is not evidence about either arm.

All synthetic, no network, no production DB.
"""

import pandas as pd
import pytest

from src.analysis import arm_eval


def _calls(rows):
    """rows: (day, arm, ticker, action) -> the frame load_arm_calls returns."""
    return pd.DataFrame([
        {"signal_date": d, "generated_at": f"{d}T20:00:00", "arm": a,
         "live": a == "dual", "ticker": t, "action": act,
         "direction": "BULLISH" if act == "BUY" else "BEARISH",
         "confidence": 0.9, "snap_price": 100.0}
        for d, a, t, act in rows
    ])


@pytest.fixture
def fixed_returns(monkeypatch):
    """AAA rises 10%, BBB falls 10% — one session after any signal date."""
    from datetime import date

    series = {
        "AAA": {date(2026, 7, 20): 100.0, date(2026, 7, 21): 110.0},
        "BBB": {date(2026, 7, 20): 100.0, date(2026, 7, 21): 90.0},
    }
    monkeypatch.setattr(arm_eval, "_close_series", lambda tk: series.get(tk, {}))


# ── orientation ────────────────────────────────────────────────────────────

def test_declining_earns_zero_not_nan(fixed_returns):
    """A HOLD on a falling stock must score 0 — better than the SELL-less BUY
    that took the loss, worse than the SELL that profited."""
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "BBB", "HOLD"),
        ("2026-07-20", "blind", "BBB", "BUY"),
        ("2026-07-20", "sighted", "BBB", "SELL"),
    ]), horizons=(1,))
    by_arm = df.set_index("arm")["ret_1d"]
    assert by_arm["dual"] == 0.0
    assert by_arm["blind"] == pytest.approx(-10.0)
    assert by_arm["sighted"] == pytest.approx(10.0)


def test_summary_separates_strategy_from_directional_return(fixed_returns):
    """An arm that declines half its calls has a strategy return diluted toward
    zero, while its directional return reflects only the calls it made."""
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "AAA", "BUY"),
        ("2026-07-20", "dual", "BBB", "HOLD"),
    ]), horizons=(1,))
    row = next(r for r in arm_eval.arm_summary(df, 1) if r["arm"] == "dual")
    assert row["mean_ret"] == pytest.approx(5.0)    # (10 + 0) / 2
    assert row["dir_ret"] == pytest.approx(10.0)    # the one real call
    assert row["dir_calls"] == 1
    assert row["flat_pct"] == 50.0


# ── pairing ────────────────────────────────────────────────────────────────

def test_edge_is_measured_only_where_the_arms_disagreed(fixed_returns):
    """Both arms call AAA identically (a wash) and split on BBB. The reported
    edge must come from BBB alone — otherwise the shared call dilutes it and a
    high agreement rate would mechanically shrink every difference to zero."""
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "AAA", "BUY"),      # both right, shared
        ("2026-07-20", "blind", "AAA", "BUY"),
        ("2026-07-20", "dual", "BBB", "SELL"),     # dual right  (+10)
        ("2026-07-20", "blind", "BBB", "BUY"),     # blind wrong (-10)
    ]), horizons=(1,))
    pair = next(p for p in arm_eval.arm_pairs(df, 1)
                if {p["a"], p["b"]} == {"dual", "blind"})
    assert pair["common"] == 2
    assert pair["agree_pct"] == 50.0
    assert pair["disagree"] == 1
    assert pair["a_ret"] == pytest.approx(10.0)
    assert pair["b_ret"] == pytest.approx(-10.0)
    assert pair["edge"] == pytest.approx(20.0)


def test_total_agreement_reports_no_edge(fixed_returns):
    """Two arms that never diverge carry no evidence about each other, however
    good or bad their shared calls were."""
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "AAA", "BUY"),
        ("2026-07-20", "blind", "AAA", "BUY"),
    ]), horizons=(1,))
    pair = next(p for p in arm_eval.arm_pairs(df, 1)
                if {p["a"], p["b"]} == {"dual", "blind"})
    assert pair["agree_pct"] == 100.0
    assert pair["disagree"] == 0
    assert pair["edge"] is None


def test_pairs_ignore_ticker_days_only_one_arm_answered(fixed_returns):
    """A ticker only one arm was asked about cannot be paired evidence."""
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "AAA", "BUY"),
        ("2026-07-20", "dual", "BBB", "SELL"),
        ("2026-07-20", "blind", "AAA", "SELL"),
    ]), horizons=(1,))
    pair = next(p for p in arm_eval.arm_pairs(df, 1)
                if {p["a"], p["b"]} == {"dual", "blind"})
    assert pair["common"] == 1        # BBB dropped — blind never saw it
    assert pair["disagree"] == 1


def test_missing_forward_return_is_excluded_from_the_edge(monkeypatch):
    """A ticker-day whose forward bar isn't cached yet must not count as a
    disagreement with a 0 return — that would silently pull every edge toward
    the arm that declined."""
    monkeypatch.setattr(arm_eval, "_close_series", lambda tk: {})
    df = arm_eval.attach_forward_returns(_calls([
        ("2026-07-20", "dual", "AAA", "BUY"),
        ("2026-07-20", "blind", "AAA", "SELL"),
    ]), horizons=(1,))
    pair = next(p for p in arm_eval.arm_pairs(df, 1)
                if {p["a"], p["b"]} == {"dual", "blind"})
    assert pair["disagree"] == 0
    assert pair["edge"] is None


# ── plumbing ───────────────────────────────────────────────────────────────

def test_empty_table_degrades_quietly():
    """The dashboard renders this before any shadow arm has run."""
    assert arm_eval.arm_summary(pd.DataFrame(), 5) == []
    assert arm_eval.arm_pairs(pd.DataFrame(), 5) == []
    assert arm_eval.attach_forward_returns(pd.DataFrame()).empty
