"""The PIVOT evaluation basis (2026-08-12 user directive).

Pivot metrics drive every DECISION evaluation (weights, filter, inversion,
states, ML labels); fixed horizons stay for monitoring, holding periods and
exits. These tests pin the four load-bearing properties:

  1. PARITY — the evaluation-side label (`next_pivot_targets`) is the training
     target (`pivot_frame.sp_buy`) to the digit, on the same series.
  2. POINT-IN-TIME — a pivot label exists only once its CONFIRMING bar is
     inside the visible window; under an as-of cutoff the observation drops
     out entirely (never a peeked value).
  3. SUPERSESSION — when the pivot column is present, the filter and the
     method states judge on IT; the fixed grid decides only the holding
     period. Frames without pivot columns fall back to the legacy rules.
  4. SHARING — `compute_directional_perf` is memoised single-flight (five
     consumers were each paying a full panel pass), callers get copies, and
     a supplied ``sim_df`` bypasses the memo.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import settings


# ── 1. parity: eval label == training target ────────────────────────────────

def _wavy_closes(n=80, seed=7):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1.0, n))


def test_next_pivot_targets_matches_training_sp_buy(monkeypatch):
    import src.analysis.pivot_target as pt

    c = _wavy_closes()
    n = len(c)
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    sp, end = pt.next_pivot_targets(c, c * 1.01, c * 0.99)

    rows = pt._leg_target_rows(c, c * 1.01, c * 0.99, dates)
    assert rows, "training pass produced no rows"
    by_date = {r["signal_date"]: r for r in rows}
    checked = 0
    for i in range(n):
        r = by_date.get(dates[i].isoformat())
        if r is None or "sp_buy" not in r:
            continue
        assert sp[i] == pytest.approx(r["sp_buy"]), f"bar {i} diverges"
        assert dates[end[i]].isoformat() == r["sp_end"]
        checked += 1
    assert checked >= 40, "parity checked on too few settled bars"


def test_unsettled_rows_carry_no_target():
    from src.analysis.pivot_target import next_pivot_targets

    c = np.linspace(100.0, 160.0, 60)          # monotone: no pivot ever prints
    sp, end = next_pivot_targets(c, c * 1.01, c * 0.99)
    assert np.isnan(sp).all() and (end == -1).all()


# ── 2. point-in-time: as-of drops unconfirmed pivots ────────────────────────

def _pivot_series():
    """Closes with a clean trough at index 52 (confirmed by bar 53)."""
    c = list(np.linspace(100.0, 120.0, 50))                       # 0..49 up
    c += [118.0, 115.0, 110.0, 116.0, 121.0]                      # 50..54: trough @52
    dates = [date(2026, 6, 2) + timedelta(days=i) for i in range(len(c))]
    closes = {d: float(v) for d, v in zip(dates, c)}
    return dates, closes


def test_pivot_observation_needs_the_confirming_bar(monkeypatch):
    import src.analysis.simulated_trades as st

    dates, closes = _pivot_series()
    # Full visibility: the row on the way down (idx 51) settles at the trough (52).
    full_dates, sp, end = st._pivot_targets(dates, closes)
    i = 51
    assert end[i] == 52
    assert sp[i] == pytest.approx((closes[dates[52]] / closes[dates[51]] - 1) * 100)

    # As-of the day OF the confirming bar (53): series truncates to < cutoff, so
    # the trough at 52 has no bar 53 inside the window -> unsettled -> no label.
    from src.analysis.asof import analysis_asof
    with analysis_asof(dates[53].isoformat()):
        st._ASOF_DATE_CACHE = None
        cut_dates, sp2, end2 = st._pivot_targets(dates, closes)
        assert len(cut_dates) == 53                # bars 0..52 visible
        assert end2[51] == -1 and np.isnan(sp2[51])
    st._ASOF_DATE_CACHE = None


def test_fwd_daily_respects_the_asof_cutoff(monkeypatch):
    import src.analysis.simulated_trades as st

    dates, closes = _pivot_series()
    from src.analysis.asof import analysis_asof
    assert st._fwd_daily(dates, closes, dates[50], 3) is not None
    with analysis_asof(dates[52].isoformat()):
        st._ASOF_DATE_CACHE = None
        # end bar (53) lands at/after the cutoff -> unknown outcome -> None
        assert st._fwd_daily(dates, closes, dates[50], 3) is None
    st._ASOF_DATE_CACHE = None


# ── 3a. directional perf emits pv; filter judges on it alone ────────────────

def test_directional_perf_emits_pivot_columns(monkeypatch):
    import src.analysis.simulated_trades as st

    dates, closes = _pivot_series()
    monkeypatch.setattr(st, "_daily_series", lambda tk: (dates, closes))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": dates[50].isoformat(),
         "ticker": "STK", "method": "tech", "score": 0.5, "direction": "BUY"},
    ])
    perf = st.compute_directional_perf(sim_df=sim, min_n=1)
    row = perf[(perf.method == "tech") & (perf.side == "both")].iloc[0]
    assert row["n_pv"] == 1
    # ticker leg 50->52 vs benchmark (same series) over the same window -> rel 0,
    # so the yield is exactly 0 and the observation exists — the point is the
    # column contract, the arithmetic is covered by the parity test.
    assert row["yield_pv"] == pytest.approx(0.0, abs=1e-9)


def test_filter_judges_on_pivot_alone_when_present(monkeypatch):
    import src.signals.aggregator as agg

    monkeypatch.setattr(settings, "enable_market_relative_filter", True)
    monkeypatch.setattr(settings, "ic_weight_min_t", 2.0)
    monkeypatch.setattr(settings, "market_relative_min_obs", 200)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)

    def _panel(**kw):
        return pd.DataFrame([
            # pivot fine, EVERY fixed horizon significantly negative -> KEPT
            {"method": "hi52", "side": "both",
             "icir_pv": +0.10, "icdays_pv": 100, "n_pv": 5000,
             "icir_1w": -0.50, "icdays_1w": 100, "n_1w": 5000},
            # pivot significantly negative, fixed fine -> DROPPED
            {"method": "tech", "side": "both",
             "icir_pv": -0.30, "icdays_pv": 100, "n_pv": 5000,
             "icir_1w": +0.30, "icdays_1w": 100, "n_1w": 5000},
        ])
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf", _panel)
    agg.reset_winrate_filter_cache()
    out = agg._market_relative_filtered()
    assert "tech" in out and "hi52" not in out


# ── 3b. method states from pivot; holding period from the fixed curve ───────

def _mh(monkeypatch, row):
    import src.analysis.method_horizons as mh
    monkeypatch.setattr(settings, "enable_method_horizons", True)
    monkeypatch.setattr(settings, "method_horizon_alpha", 0.05)
    monkeypatch.setattr(settings, "method_horizon_min_obs", 1)
    monkeypatch.setattr(mh, "_CACHE", {}, raising=False)
    ev = pd.DataFrame([{"generated_at": "t", "signal_date": "2026-06-01",
                        "ticker": "STK", "method": row["method"],
                        "score": 0.9, "direction": "BUY"}])
    monkeypatch.setattr("src.analysis.simulated_trades.load_sim_entry_events",
                        lambda days=None: ev)
    monkeypatch.setattr("src.analysis.simulated_trades.compute_method_perf",
                        lambda **kw: pd.DataFrame([row]))
    return mh


def test_pivot_disproof_supersedes_a_flattering_fixed_curve(monkeypatch):
    mh = _mh(monkeypatch, {
        "method": "news",
        "win_pv": 40.0, "n_pv": 500,               # significantly < 50 on pivot
        "win_1d": 60.0, "n_1d": 500,               # looks great at 1d
        "win_3d": None, "n_3d": 0,
        "win_1w": None, "n_1w": 0, "win_2w": None, "n_2w": 0,
    })
    out = mh.compute_method_horizons()
    assert out["news"]["state"] == mh.DISPROVEN


def test_pivot_proof_takes_holding_period_from_fixed_curve(monkeypatch):
    mh = _mh(monkeypatch, {
        "method": "news",
        "win_pv": 60.0, "n_pv": 500,               # significantly > 50 on pivot
        "win_1d": 52.0, "n_1d": 400,               # suggestive, not significant
        "win_3d": 55.0, "n_3d": 400,               # the best fixed guess
        "win_1w": 51.0, "n_1w": 400, "win_2w": None, "n_2w": 0,
    })
    out = mh.compute_method_horizons()
    d = out["news"]
    assert d["state"] == mh.PROVEN
    assert d["best_horizon"] == "3d", "holding period must come from the FIXED curve"
    assert d["best_days"] == 3.0


def test_no_pivot_column_falls_back_to_legacy_states(monkeypatch):
    mh = _mh(monkeypatch, {
        "method": "news",
        "win_1d": 60.0, "n_1d": 500,               # significant at 1d
        "win_3d": None, "n_3d": 0,
        "win_1w": None, "n_1w": 0, "win_2w": None, "n_2w": 0,
    })
    out = mh.compute_method_horizons()
    assert out["news"]["state"] == mh.PROVEN       # legacy rule still governs


# ── 3c. stacker label resolution ─────────────────────────────────────────────

def test_stacker_label_prefers_settled_pivot_rank(monkeypatch):
    from src.analysis.ml_stacker import BUY_TRAIN_CONFIG, _label_cfg
    monkeypatch.setattr(settings, "stacker_label_basis", "pivot_rank")
    df = pd.DataFrame({"fwd_ret_rank_pv": np.linspace(-0.5, 0.5, 600),
                       "fwd_ret_rank_5d": np.linspace(-0.5, 0.5, 600)})
    cfg, ycol = _label_cfg(df, BUY_TRAIN_CONFIG, "t")
    assert ycol == "fwd_ret_rank_pv" and cfg["basis"] == "rank_pv"


def test_stacker_label_falls_back_when_pivot_thin(monkeypatch):
    from src.analysis.ml_stacker import BUY_TRAIN_CONFIG, _label_cfg
    monkeypatch.setattr(settings, "stacker_label_basis", "pivot_rank")
    df = pd.DataFrame({"fwd_ret_rank_pv": [0.1] * 10 + [None] * 590,
                       "fwd_ret_rank_5d": np.linspace(-0.5, 0.5, 600)})
    cfg, ycol = _label_cfg(df, BUY_TRAIN_CONFIG, "t")
    assert ycol == "fwd_ret_rank_5d" and cfg["basis"] == "rank"


def test_stacker_label_setting_pins_legacy(monkeypatch):
    from src.analysis.ml_stacker import BUY_TRAIN_CONFIG, _label_cfg
    monkeypatch.setattr(settings, "stacker_label_basis", "rank_5d")
    df = pd.DataFrame({"fwd_ret_rank_pv": np.linspace(-0.5, 0.5, 600)})
    cfg, ycol = _label_cfg(df, BUY_TRAIN_CONFIG, "t")
    assert ycol == "fwd_ret_rank_5d" and cfg["basis"] == "rank"


# ── 4. the shared memo ───────────────────────────────────────────────────────

def test_directional_perf_is_memoised_and_hands_out_copies(monkeypatch):
    import src.analysis.simulated_trades as st

    calls = {"n": 0}
    frame = pd.DataFrame([{"method": "tech", "side": "both", "icir_pv": 0.1}])

    def _impl(**kw):
        calls["n"] += 1
        return frame.copy()

    monkeypatch.setattr(st, "_directional_perf_impl", lambda **kw: _impl(**kw))
    st.reset_cache()

    a = st.compute_directional_perf(days=90)
    b = st.compute_directional_perf(days=90)          # same args -> memo hit
    assert calls["n"] == 1
    b.loc[0, "icir_pv"] = 999.0                        # mutate MY copy
    c = st.compute_directional_perf(days=90)
    assert c.loc[0, "icir_pv"] == pytest.approx(0.1), "cache must hand out copies"

    st.compute_directional_perf(days=30)               # different args -> new compute
    assert calls["n"] == 2

    sim = pd.DataFrame([{"generated_at": "t", "signal_date": "2026-06-01",
                         "ticker": "S", "method": "m", "score": 1.0}])
    st.compute_directional_perf(sim_df=sim)            # explicit frame bypasses
    assert calls["n"] == 3
    st.reset_cache()
    st.compute_directional_perf(days=90)
    assert calls["n"] == 4, "reset_cache must force a recompute"
    st.reset_cache()
