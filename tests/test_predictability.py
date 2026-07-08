"""Predictability-feature IC panel (Tier 0): feature math, no-look-ahead alignment,
and bucket-separation of combined_score's forward prediction.

Guards the three things that make this measurement trustworthy: the per-stock
features are computed correctly, they are read AS OF the signal date (never with
future bars), and the bucketed conditional IC actually separates a predictive
cohort from an anti-predictive one.
"""

from datetime import date

import pandas as pd
import pytest

import src.analysis.predictability as pf


# ── feature math ─────────────────────────────────────────────────────────────

def _closes(vals):
    idx = [date(2026, 1, 1) + pd.Timedelta(days=k) for k in range(len(vals))]
    idx = [d.date() if hasattr(d, "date") else d for d in idx]
    s = pd.Series([float(v) for v in vals], index=idx)
    return idx, s


def test_efficiency_ratio_monotone_is_one(monkeypatch):
    # A perfectly monotone climb has efficiency ratio 1.0 (net move == path length).
    vals = list(range(100, 125))            # 25 bars, +1 each
    idx, close = _closes(vals)
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idx, close, close, close, close))
    _, fs = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fs["eff_ratio"].iloc[-1] == pytest.approx(1.0)


def test_efficiency_ratio_zigzag_is_low(monkeypatch):
    # A pure oscillation goes nowhere: tiny net move over a long path → ER ≈ 0.
    vals = [100 + (k % 2) for k in range(25)]   # 100,101,100,101,…
    idx, close = _closes(vals)
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idx, close, close, close, close))
    _, fs = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fs["eff_ratio"].iloc[-1] < 0.2


def test_dollar_vol_is_20d_avg_in_millions(monkeypatch):
    idx, close = _closes([10.0] * 25)                       # flat $10
    vol = pd.Series([1_000_000.0] * 25, index=idx)          # 1M shares/day
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idx, close, close, close, vol))
    _, fs = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fs["dollar_vol"].iloc[-1] == pytest.approx(10.0)  # $10 × 1M / 1e6 = $10M


def test_price_and_volume_features_are_bucketed():
    # price (a signals-panel column) + dollar_vol (OHLCV-derived) are evaluated as
    # predictability features — the "are penny / thin-volume names predictable?" cut.
    rows = [{"signal_date": "2026-01-02", "combined_score": 0.3,
             "price": float(i + 1), "dollar_vol": float(i + 1),
             "fwd_ret_1d": float(i - 15)} for i in range(30)]
    ic = pf.compute_predictability_ic(pd.DataFrame(rows), horizons=(1,), min_n=5,
                                      n_buckets=3, features=["price", "dollar_vol"])
    assert {"price", "dollar_vol"} <= set(ic["feature"])
    price = ic[ic["feature"] == "price"]
    assert list(price["bucket"]) == ["Low", "Mid", "High"]   # penny → large-cap terciles


def test_er_signed_is_directional(monkeypatch):
    # Signed efficiency ratio: a clean climb → +1, a clean fall → −1 (unsigned
    # eff_ratio is 1.0 for both — the sign is what makes it a directional signal).
    up = list(range(100, 125))
    idx, close = _closes(up)
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idx, close, close, close, close))
    _, fs = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fs["er_signed"].iloc[-1] == pytest.approx(1.0)
    assert fs["eff_ratio"].iloc[-1] == pytest.approx(1.0)

    idxd, closed = _closes(list(range(124, 99, -1)))
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idxd, closed, closed, closed, closed))
    _, fsd = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fsd["er_signed"].iloc[-1] == pytest.approx(-1.0)


def test_adx_signed_follows_trend_direction(monkeypatch):
    # ADX is unsigned; adx_signed orients it by +DI vs −DI, so an uptrend is
    # positive and a downtrend negative.
    up = list(range(100, 145))
    idx, close = _closes(up)
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idx, close + 0.5, close - 0.5, close, close))
    _, fs = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fs["adx_signed"].iloc[-1] > 0

    idxd, closed = _closes(list(range(144, 99, -1)))
    monkeypatch.setattr(pf, "_hlc_by_session", lambda tk: (idxd, closed + 0.5, closed - 0.5, closed, closed))
    _, fsd = pf._feature_frame("X", er_window=20, vol_window=20, adx_period=14)
    assert fsd["adx_signed"].iloc[-1] < 0


def test_attach_feature_signals_asof(monkeypatch):
    idx = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
    fs = pd.DataFrame({"er_signed": [0.1, -0.2, 0.3], "adx_signed": [0.4, -0.5, 0.6]}, index=idx)
    monkeypatch.setattr(pf, "_feature_frame", lambda tk, *a, **k: (idx, fs))
    panel = pd.DataFrame([
        {"ticker": "X", "signal_date": "2026-01-05"},          # exact session
        {"ticker": "X", "signal_date": "2026-01-03"},          # anchors to 01-05
        {"ticker": "X", "signal_date": "2026-01-09"},          # past last bar → None
    ])
    out = pf.attach_feature_signals(panel)
    assert out.iloc[0]["er_signed"] == pytest.approx(-0.2)
    assert out.iloc[1]["adx_signed"] == pytest.approx(-0.5)
    assert pd.isna(out.iloc[2]["er_signed"])


def _dir_panel():
    """A signal that WORKS on longs (positive score → positive return, magnitude
    aligned) but FAILS on shorts (negative score → still positive return)."""
    rows = []
    for s, f in zip([0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
        rows.append({"signal_date": "2026-01-02", "er_signed": s, "fwd_ret_1d": f})
    for s, f in zip([-0.2, -0.4, -0.6, -0.8, -1.0], [1.0, 2.0, 3.0, 4.0, 5.0]):
        rows.append({"signal_date": "2026-01-02", "er_signed": s, "fwd_ret_1d": f})
    return pd.DataFrame(rows)


def test_directional_ic_splits_buy_and_sell():
    out = pf.compute_directional_feature_ic(_dir_panel(), horizons=(1,), min_n=5,
                                            features=["er_signed"])
    by = {r["side"]: r for _, r in out.iterrows()}
    assert set(by) == {"all", "buy", "sell"}
    # Long calls work: 100% hit, +3 avg, IC +1.
    assert by["buy"]["win_1d"] == pytest.approx(100.0)
    assert by["buy"]["sim_1d"] == pytest.approx(3.0)
    assert by["buy"]["ic_1d"] == pytest.approx(1.0)
    assert by["buy"]["n_1d"] == 5
    # Short calls fail (stock rose): 0% hit, −3 avg — the split exposes it.
    assert by["sell"]["win_1d"] == pytest.approx(0.0)
    assert by["sell"]["sim_1d"] == pytest.approx(-3.0)
    assert by["sell"]["side_label"] == "Sells (short calls)"
    assert by["all"]["n_1d"] == 10


def test_directional_ic_min_n_gates_stats():
    out = pf.compute_directional_feature_ic(_dir_panel(), horizons=(1,), min_n=20,
                                            features=["er_signed"])
    buy = next(r for _, r in out.iterrows() if r["side"] == "buy")
    assert buy["n_1d"] == 5                          # count still reported
    assert pd.isna(buy["win_1d"]) and pd.isna(buy["ic_1d"])


def test_directional_ic_empty():
    assert pf.compute_directional_feature_ic(pd.DataFrame(), horizons=(1,)).empty


def test_adx_series_is_causal_no_lookahead(monkeypatch):
    # ADX at bar i must equal ADX recomputed on the truncated series [0..i] — i.e.
    # it never peeks at future bars (Wilder smoothing is a causal recursion).
    import numpy as np
    rng = np.random.default_rng(0)
    vals = 100 + np.cumsum(rng.normal(0, 1, 60))
    idx, close = _closes(vals)
    high = close + 0.5
    low = close - 0.5
    full = pf._adx_series(high, low, close, 14)
    cut = 40
    trunc = pf._adx_series(high.iloc[:cut + 1], low.iloc[:cut + 1], close.iloc[:cut + 1], 14)
    assert full.iloc[cut] == pytest.approx(trunc.iloc[-1])


# ── as-of-date alignment (no look-ahead in the panel join) ───────────────────

def _fake_frame():
    idx = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
    fs = pd.DataFrame({"eff_ratio": [0.1, 0.2, 0.3], "adx": [10.0, 20.0, 30.0],
                       "realized_vol": [1.0, 2.0, 3.0]}, index=idx)
    return idx, fs


def test_feature_read_at_anchor_bar(monkeypatch):
    monkeypatch.setattr(pf, "_feature_frame", lambda tk, *a, **k: _fake_frame())
    panel = pd.DataFrame([
        {"ticker": "X", "signal_date": "2026-01-05", "combined_score": 0.5,
         "n_methods_agreeing": 3, "fwd_ret_1d": 1.0},
        # signal on a non-session day (2026-01-03) anchors to the next session (01-05),
        # the SAME bar build_panel anchors the forward return on — no look-ahead.
        {"ticker": "X", "signal_date": "2026-01-03", "combined_score": 0.5,
         "n_methods_agreeing": 5, "fwd_ret_1d": 1.0},
    ])
    fp = pf.build_feature_panel(panel)
    assert fp.iloc[0]["eff_ratio"] == pytest.approx(0.2)   # 01-05 bar
    assert fp.iloc[1]["eff_ratio"] == pytest.approx(0.2)   # anchored to 01-05
    assert fp.iloc[0]["breadth"] == 3                      # copied from n_methods_agreeing


def test_feature_none_past_last_bar(monkeypatch):
    monkeypatch.setattr(pf, "_feature_frame", lambda tk, *a, **k: _fake_frame())
    panel = pd.DataFrame([{"ticker": "X", "signal_date": "2026-01-09",  # after last bar
                           "combined_score": 0.5, "n_methods_agreeing": 2, "fwd_ret_1d": 1.0}])
    fp = pf.build_feature_panel(panel)
    assert pd.isna(fp.iloc[0]["adx"])


# ── bucketed conditional IC + edge summary ───────────────────────────────────

def _bucket_panel():
    # eff_ratio Low cohort is anti-predictive (score>0, fwd<0 → 0% hit); the High
    # cohort is predictive (score>0, fwd>0 → 100% hit); Mid is 50/50.
    rows = []
    eff = [0.10, 0.15, 0.20, 0.25,  0.45, 0.50, 0.55, 0.60,  0.85, 0.90, 0.95, 1.00]
    fwd = [-2, -1, -3, -2.5,        1, -1, 2, -2,            2, 1, 3, 2.5]
    sc = [0.4, 0.5, 0.6, 0.55] * 3
    for e, f, s in zip(eff, fwd, sc):
        rows.append({"signal_date": "2026-01-02", "combined_score": s,
                     "eff_ratio": e, "fwd_ret_1d": float(f)})
    return pd.DataFrame(rows)


def test_bucket_separation_high_beats_low():
    ic = pf.compute_predictability_ic(_bucket_panel(), horizons=(1,), min_n=3,
                                      n_buckets=3, features=["eff_ratio"])
    assert (ic["feature"] == pf.BASELINE_KEY).any()          # baseline row present
    eff = ic[ic["feature"] == "eff_ratio"].set_index("bucket")
    assert list(eff.index) == ["Low", "Mid", "High"]         # ordered Low→High
    assert eff.loc["High", "hit_1d"] == pytest.approx(100.0)
    assert eff.loc["Low", "hit_1d"] == pytest.approx(0.0)
    assert eff.loc["High", "n_1d"] == 4


def test_edge_summary_reports_spread_and_best_bucket():
    ic = pf.compute_predictability_ic(_bucket_panel(), horizons=(1,), min_n=3,
                                      n_buckets=3, features=["eff_ratio"])
    edges = pf.summarize_feature_edges(ic, horizons=(1,))
    row = edges[edges["feature"] == "eff_ratio"].iloc[0]
    assert row["hit_spread_1d"] == pytest.approx(100.0)      # best(100) − worst(0)
    assert row["hit_best_1d"] == "High"


def test_low_variance_or_thin_feature_skipped():
    # A feature with <2 distinct values yields no buckets (only the baseline row).
    panel = _bucket_panel().assign(adx=5.0)                  # constant
    ic = pf.compute_predictability_ic(panel, horizons=(1,), min_n=3,
                                      n_buckets=3, features=["adx"])
    assert list(ic["feature"].unique()) == [pf.BASELINE_KEY]


def test_empty_inputs():
    assert pf.compute_predictability_ic(pd.DataFrame(), horizons=(1,)).empty
    assert pf.summarize_feature_edges(pd.DataFrame(), horizons=(1,)).empty
    assert pf.build_feature_panel(pd.DataFrame()).empty
