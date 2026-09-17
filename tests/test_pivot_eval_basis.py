"""The PIVOT evaluation basis (2026-08-12 user directive).

Pivot metrics drive every DECISION evaluation (weights, filter, inversion,
states, ML labels); fixed horizons stay for monitoring, holding periods and
exits. These tests pin the four load-bearing properties:

  1. PARITY — the trainer's label (`session_close_labels`, date-only rows at
     the session close) is the per-row resolver's (`pivot_rows.pivot_fwd_row`)
     to the digit, on the same 30-minute series.
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


# ── 1. parity: the training label == the per-row label on the same series ──

def _wavy_closes(n=80, seed=7):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1.0, n))


def test_training_label_matches_the_row_resolver(monkeypatch):
    """`session_close_labels` (the ml_ohlcv trainer's label, date-only rows
    anchored at the session close) and `pivot_rows.pivot_fwd_row` (the panel's
    per-row resolver on a date-only row) must agree to the digit."""
    import src.analysis.pivot_target as pt
    from src.analysis import pivot_rows as pr
    from tests.intraday_fixtures import replica_30m, stub_30m

    c = _wavy_closes()
    dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(len(c))]
    dates = [d for d in dates if d.weekday() < 5][:60]
    c = c[:len(dates)]
    idx, cc, hh, ll = replica_30m(dates, c)
    sp, end = pt.session_close_labels(idx, cc, hh, ll, dates, c)
    stub_30m(monkeypatch, {"STK": (idx, cc, hh, ll)})
    checked = 0
    for i, d in enumerate(dates):
        r = pr.pivot_fwd_row("STK", d, None, fallback_close=float(c[i]))
        if not (sp[i] == sp[i]):
            assert r is None, f"{d}: resolver settled a row the trainer did not"
            continue
        assert r is not None, f"{d}: trainer settled a row the resolver did not"
        assert r[0] == pytest.approx(sp[i]) and r[1].isoformat() == end[i]
        checked += 1
    assert checked >= 20, "parity checked on too few settled rows"


def test_unsettled_rows_carry_no_target():
    from src.analysis.pivot_target import session_close_labels
    from tests.intraday_fixtures import replica_30m

    dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(70)]
    dates = [d for d in dates if d.weekday() < 5]
    c = np.linspace(100.0, 160.0, len(dates))          # monotone: no pivot ever prints
    sp, end = session_close_labels(*replica_30m(dates, c), dates, c)
    assert np.isnan(sp).all() and all(e is None for e in end)


# ── 2. point-in-time: as-of drops unconfirmed pivots ────────────────────────

def _pivot_series():
    """Closes with a clean trough at index 52 (confirmed by bar 53)."""
    c = list(np.linspace(100.0, 120.0, 50))                       # 0..49 up
    c += [118.0, 115.0, 110.0, 116.0, 121.0]                      # 50..54: trough @52
    dates = [date(2026, 6, 1) + timedelta(days=i) for i in range(80)]
    dates = [d for d in dates if d.weekday() < 5][:len(c)]
    closes = {d: float(v) for d, v in zip(dates, c)}
    return dates, closes


def test_pivot_observation_needs_the_confirming_bar(monkeypatch):
    from src.analysis import pivot_rows as pr
    from tests.intraday_fixtures import replica_30m, stub_30m

    dates, closes = _pivot_series()
    stub_30m(monkeypatch, {"STK": replica_30m(dates, closes)})
    # Full visibility: the row on the way down (idx 51) settles at the trough (52).
    r = pr.pivot_fwd_row("STK", dates[51], None, fallback_close=closes[dates[51]])
    assert r is not None and r[1] == dates[52]
    assert r[0] == pytest.approx((closes[dates[52]] / closes[dates[51]] - 1) * 100)
    # As-of the day OF the confirming bar (53): only earlier sessions are visible,
    # so the trough at 52 has no confirming bar inside the window -> no label.
    assert pr.pivot_fwd_row("STK", dates[51], None, fallback_close=closes[dates[51]],
                            asof_day=dates[53]) is None
    assert pr.pivot_fwd_row("STK", dates[51], None, fallback_close=closes[dates[51]],
                            asof_day=dates[54]) is not None


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

    from src.analysis.pivot_target import session_close_utc
    from tests.intraday_fixtures import replica_30m, stub_30m

    dates, closes = _pivot_series()
    monkeypatch.setattr(st, "_daily_series", lambda tk: (dates, closes))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])
    stub_30m(monkeypatch, {"STK": replica_30m(dates, closes), "SPY": replica_30m(dates, closes)})
    sim = pd.DataFrame([
        {"generated_at": session_close_utc(dates[50]).isoformat(), "signal_date": dates[50].isoformat(),
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
