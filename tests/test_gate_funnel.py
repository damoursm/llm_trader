"""Per-gate funnel on the pivot target (restored to the dashboard 2026-08-18).

Three methodology choices carry the verdicts, and each is invisible in the
output if it regresses — the table still renders, just with different numbers:

  1. EXCESS over a SAME-SIDE random draw, not the raw oriented mean. A short's
     oriented return is minus the population drift by construction, so a
     short-heavy cohort looks worse than a long-heavy one for no reason but
     drift. Without this, gate verdicts track their BUY/SELL mix.
  2. DAY-CLUSTERED t. Same-day returns are correlated; a t on the raw row count
     is inflated several-fold and would turn "nothing is significant" into a
     table full of apparently-real effects.
  3. The label join keys on ``signal_date`` (ET), never ``generated_at`` (UTC).
     Every 20:00-23:59 ET overnight tick carries the NEXT UTC date, so joining on
     generated_at mis-assigns ~4 of 7 nightly slots — it flipped a gate's verdict
     during the original study.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import gate_funnel as gf


def _frame(sides, rets, days=None):
    n = len(sides)
    days = days or [f"2026-08-{1 + i % 10:02d}" for i in range(n)]
    sign = np.array([1.0 if s == "BUY" else -1.0 for s in sides])
    return pd.DataFrame({"sign": sign, "oriented": np.array(rets, dtype=float),
                         "day": days, "gate": ["pass"] * n})


# ── 1. excess must neutralise the side mix ──────────────────────────────────

def test_excess_removes_the_drift_advantage_of_a_long_cohort():
    """A long-only and a short-only cohort that BOTH merely earn the drift must
    score the same excess (~0). On raw oriented means the short looks far worse."""
    drift = 0.40
    longs = _frame(["BUY"] * 20, [drift] * 20)
    shorts = _frame(["SELL"] * 20, [-drift] * 20)
    L, S = gf._cohort_stats(longs, drift), gf._cohort_stats(shorts, drift)
    assert L["mean"] == pytest.approx(0.40)
    assert S["mean"] == pytest.approx(-0.40)      # raw means look opposite...
    assert L["exc"] == pytest.approx(0.0, abs=1e-9)
    assert S["exc"] == pytest.approx(0.0, abs=1e-9)   # ...excess agrees they tie


def test_excess_credits_genuine_outperformance_on_either_side():
    drift = 0.40
    good_short = _frame(["SELL"] * 20, [-drift + 1.0] * 20)   # 1pp better than drift
    assert gf._cohort_stats(good_short, drift)["exc"] == pytest.approx(1.0)


# ── 2. the t must be day-clustered ──────────────────────────────────────────

def test_t_is_day_clustered_not_row_counted():
    """200 rows spread over 5 days must produce a t built from 5 observations,
    not 200 — the raw-row t would be ~6x larger and manufacture significance."""
    rows, days = [], []
    for d in range(5):
        rows += [1.0] * 40
        days += [f"2026-08-0{d + 1}"] * 40
    f = _frame(["BUY"] * 200, rows, days)
    st = gf._cohort_stats(f, 0.0)
    assert st["days"] == 5
    # a constant per-day mean has zero across-day variance => no t at all
    assert np.isnan(st["t"])


def test_t_needs_at_least_five_days():
    f = _frame(["BUY"] * 40, [1.0] * 40, [f"2026-08-0{1 + i % 4}" for i in range(40)])
    st = gf._cohort_stats(f, 0.0)
    assert st["days"] == 4 and np.isnan(st["t"])


def test_t_is_computed_on_excess_not_the_raw_mean():
    """A short book that exactly earns the drift has a strongly negative raw
    oriented mean but ZERO excess — its t must reflect the excess, not the sign
    imposed by drift."""
    drift = 0.40
    days = [f"2026-08-{1 + i:02d}" for i in range(10)]
    f = _frame(["SELL"] * 10, [-drift] * 10, days)
    st = gf._cohort_stats(f, drift)
    assert st["mean"] < 0                      # raw mean is negative...
    assert st["exc"] == pytest.approx(0.0, abs=1e-9)   # ...but there is no edge
    assert np.isnan(st["t"]) or abs(st["t"]) < 1e-6


# ── 3. the ET/UTC join trap ─────────────────────────────────────────────────

def test_loader_joins_on_signal_date_not_generated_at():
    """Asserted at the source: generated_at is UTC, signal_date is ET, and an
    overnight tick carries the NEXT UTC date. A regression here silently
    mis-assigns ~4 of 7 nightly slots."""
    import inspect
    src = inspect.getsource(gf.load_gate_calls)
    assert "signal_date" in src
    assert "SELECT DISTINCT run_id, signal_date FROM signals" in src, (
        "the run_id -> ET day map must come from `signals`")
    # the day key must never be derived from generated_at
    assert 'rdf["day"] = rdf["run_id"]' in src


# ── cascade shape ───────────────────────────────────────────────────────────

def test_gate_stages_match_the_pipeline_order():
    """The table reads top-to-bottom as the funnel, so the order must equal the
    order pipeline._apply_actionable_gates applies the gates."""
    keys = [k for k, _ in gf.GATE_STAGES]
    assert keys == ["below_threshold", "low_agreement", "buy_blocked",
                    "earnings_blackout", "untradeable", "overextended"]


def test_gate_ic_needs_contrast():
    """A gate that never fired has no pass/drop contrast, so its IC is undefined
    rather than a spurious 0."""
    f = _frame(["BUY"] * 20, [1.0] * 20)
    ic, icir, days = gf._gate_ic(f, "buy_blocked")     # nothing carries that stamp
    assert np.isnan(ic) and np.isnan(icir) and days == 0


def test_empty_input_is_safe():
    """Degenerate input returns "no verdict", never a fabricated zero."""
    for frame in (None, pd.DataFrame()):
        st = gf._cohort_stats(frame, 0.4)
        assert st["n"] == 0
        assert np.isnan(st["exc"]) and np.isnan(st["t"])
    t, days = gf._day_clustered_t(pd.DataFrame())
    assert np.isnan(t) and days == 0


# ── simulated gates ─────────────────────────────────────────────────────────

def test_simulated_source_reads_confidence_UNMASKED():
    """THE silent failure. build_panel masks confidence to NaN wherever
    CONFIDENCE_EPOCH says it came from superseded code -- 90.5% of rows in the
    current window. `conf < threshold` is False for NaN, so Gate 1 rejects
    NOTHING while appearing to run: on the first attempt it dropped 0 of 20,381
    rows and the biggest gate in the cascade silently vanished. The simulation
    therefore sources confidence from the raw `signals` table."""
    import inspect
    src = inspect.getsource(gf.simulate_gate_calls)
    assert "FROM signals" in src, "confidence must come from the raw signals table"
    assert "_conf_raw" in src
    # ...and a row with no confidence at all is EXCLUDED, never passed silently
    assert "df = df[conf.notna()]" in src


def test_simulated_gates_are_point_in_time():
    """`pipeline._recent_runup_pct` and `liquidity.is_liquid` both read the
    CURRENT tail of the OHLCV cache -- correct live, look-ahead in a historical
    replay. The simulation must recompute both from bars visible on the signal
    date."""
    import inspect
    src = inspect.getsource(gf._pit_ohlcv_features)
    assert "searchsorted" in src, "bars must be cut as-of the signal date"
    assert "side=\"right\"" in src


def test_unsimulable_gates_are_declared_not_faked():
    """Gate 3 needs the historical earnings calendar, which is not stored. It
    must be reported as unsimulable -- a gate that always passes is
    indistinguishable from one that never rejects anything."""
    assert "earnings_blackout" in gf._SIM_UNSIMULATED


def test_regime_thresholds_match_the_documented_table():
    assert gf._REGIME_THRESHOLD == {"PANIC": 0.95, "RISK_OFF": 0.89,
                                    "CAUTION": 0.87, "NEUTRAL": 0.85,
                                    "RISK_ON": 0.79}
