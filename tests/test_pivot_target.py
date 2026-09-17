"""Probes for the pivot LABEL (30-minute bars) and the daily LEG FEATURES
(``analysis/pivot_target``).

The failure these guard is invisible and fatal: any leak across the pivot
confirmation seam hands a model the answer key and produces a spectacular,
entirely fake validation. The label tests attack that seam on the 30-minute
series the label runs on; the leg-feature tests attack it on the daily series
the features run on; the serving-parity test pins ``latest_leg_features`` to
the dataset pass so train/serve drift cannot open the same hole later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.pivot_target import (LEG_FEATURES, MAX_PIVOT_BARS_30M,
                                       intraday_pivot_targets, leg_feature_rows,
                                       within_day_rank)
from tests.intraday_fixtures import flat_series, plant_peak, sessions_30m


def _dates(n):
    return list(pd.date_range("2024-01-02", periods=n, freq="B").date)


def _rows(c, h=None, lo=None, **kw):
    c = np.asarray(c, dtype=float)
    h = c * 1.01 if h is None else np.asarray(h, dtype=float)
    lo = c * 0.99 if lo is None else np.asarray(lo, dtype=float)
    return leg_feature_rows(c, h, lo, _dates(len(c)), **kw)


def _base_series(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return 50 * np.cumprod(1 + rng.normal(0, 0.02, n))


# ── the label: worked example on 30-minute bars ──────────────────────────────

def test_worked_example_signed_target_on_30m_bars():
    """A peak on session 6 bar 6 (high 103) confirmed by the >1% drop after it,
    then a trough (low 99) confirmed by the recovery: an anchor before the peak
    targets the peak's HIGH, an anchor after it targets the trough's LOW."""
    idx = sessions_30m("2026-08-03", 10)
    c, h, lo = flat_series(idx)
    at = 6 * 13 + 6
    plant_peak(h, lo, c, at)                       # peak 103 @at; lows 99 on at+1..at+3
    # recovery confirms the trough: +2% off 99 on the bars after the lows
    for k in range(4, 8):
        c[at + k] = 101.5; h[at + k] = 101.8; lo[at + k] = 101.2
    tick_before = idx[at - 3] + pd.Timedelta(minutes=5)
    r = intraday_pivot_targets(idx, c, h, lo, [(tick_before, 100.0)])[0]
    assert r["resolved"] and r["is_peak"] is True
    assert r["target_pct"] == pytest.approx(3.0)            # 103 / 100 - 1
    assert pd.Timestamp(r["end_ts"]) == idx[at]
    assert pd.Timestamp(r["confirm_ts"]) == idx[at + 1]     # the first low that printed the 1% drop
    tick_after = idx[at] + pd.Timedelta(minutes=5)          # inside the peak bar: the peak is not eligible
    r2 = intraday_pivot_targets(idx, c, h, lo, [(tick_after, 102.0)])[0]
    assert r2["resolved"] and r2["is_peak"] is False
    assert r2["target_pct"] == pytest.approx((99.0 / 102.0 - 1) * 100)
    assert pd.Timestamp(r2["end_ts"]) > tick_after


def test_monotone_series_has_no_settled_targets():
    idx = sessions_30m("2026-08-03", 8)
    c = np.linspace(100.0, 160.0, len(idx)); h = c * 1.001; lo = c * 0.999
    res = intraday_pivot_targets(idx, c, h, lo, [(idx[5 * 13], 100.0), (idx[6 * 13] + pd.Timedelta(minutes=1), 120.0)])
    assert all(r is not None and r["resolved"] is False for r in res)
    assert res[0]["target_pct"] == pytest.approx((c[-1] / 100.0 - 1) * 100)   # marked at the last close


def test_max_pivot_bars_cap():
    """A pivot further out than MAX_PIVOT_BARS_30M (60 sessions) never settles a row."""
    n_sess = 70
    idx = sessions_30m("2026-01-05", n_sess)
    c = np.r_[np.linspace(100.0, 130.0, 65 * 13), np.linspace(129.5, 120.0, 5 * 13)]
    h, lo = c * 1.0005, c * 0.9995
    r_early = intraday_pivot_targets(idx, c, h, lo, [(idx[0] + pd.Timedelta(minutes=1), 100.0)])[0]
    assert r_early["resolved"] is False                     # the peak is > 60 sessions out
    r_late = intraday_pivot_targets(idx, c, h, lo, [(idx[60 * 13], 125.0)])[0]
    assert r_late["resolved"] is True and r_late["bars_ahead"] <= MAX_PIVOT_BARS_30M


def test_anchor_before_the_first_bar_is_unlabelled():
    idx = sessions_30m("2026-08-03", 8)
    c, h, lo = flat_series(idx)
    plant_peak(h, lo, c, 3 * 13 + 2)
    assert intraday_pivot_targets(idx, c, h, lo, [(idx[0] - pd.Timedelta(days=1), 100.0)])[0] is None


# ── causality at the confirmation seam (label) ───────────────────────────────

def test_settled_label_immune_to_future_mutation():
    """Rewriting every bar after a settled row's CONFIRMING bar must change
    nothing about that row — target, end bar or confirmation bar."""
    idx = sessions_30m("2026-08-03", 12)
    rng = np.random.default_rng(7)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.004, len(idx)))
    h, lo = c * 1.001, c * 0.999
    tick = idx[2 * 13] + pd.Timedelta(minutes=1)
    ref = intraday_pivot_targets(idx, c, h, lo, [(tick, float(c[2 * 13 - 1]))])[0]
    assert ref is not None and ref["resolved"], "need a settled probe row"
    j_conf = int(idx.get_loc(pd.Timestamp(ref["confirm_ts"])))
    c2, h2, lo2 = c.copy(), h.copy(), lo.copy()
    c2[j_conf + 1:] = c2[j_conf + 1:] * 7.0 + 3.0        # violent future rewrite
    h2[j_conf + 1:] = c2[j_conf + 1:] * 1.001; lo2[j_conf + 1:] = c2[j_conf + 1:] * 0.999
    new = intraday_pivot_targets(idx, c2, h2, lo2, [(tick, float(c[2 * 13 - 1]))])[0]
    assert new["target_pct"] == pytest.approx(ref["target_pct"])
    assert pd.Timestamp(new["end_ts"]) == pd.Timestamp(ref["end_ts"])
    assert pd.Timestamp(new["confirm_ts"]) == pd.Timestamp(ref["confirm_ts"])


# ── causality at the confirmation seam (leg features, daily bars) ────────────

def test_leg_features_use_only_confirmed_pivots():
    """A pivot at bar j is usable from bar j+1 only. Truncating the series at the
    pivot bar itself (so its confirming bar never prints) must leave bar j's leg
    features exactly as they were computed from the shorter history."""
    c = _base_series(seed=11)
    full = _rows(c)
    d = _dates(len(c))
    full_by_date = {r["signal_date"]: r for r in full}
    for i in (60, 80, 100):
        trunc = _rows(c[: i + 1], only_last=True)
        assert trunc, f"no serving row at {i}"
        t = trunc[0]
        f = full_by_date[d[i].isoformat()]
        for k in LEG_FEATURES:
            tv, fv = t.get(k), f.get(k)
            if tv is None or tv != tv:
                assert fv is None or fv != fv, f"{k}@{i}: NaN mismatch"
            else:
                assert tv == pytest.approx(fv), (
                    f"{k}@{i}: serving-from-truncated {tv} != full-pass {fv} — "
                    f"a future bar is leaking into the leg state")


def test_serving_only_last_matches_full_pass():
    c = _base_series(seed=23)
    d = _dates(len(c))
    full = {r["signal_date"]: r for r in _rows(c)}
    last = _rows(c, only_last=True)[0]
    ref = full[d[len(c) - 1].isoformat()]
    for k in LEG_FEATURES:
        a, b = last.get(k), ref.get(k)
        if a is None or a != a:
            assert b is None or b != b
        else:
            assert a == pytest.approx(b)


def test_leg_rows_carry_no_target():
    """Features only: the label is never produced by the daily pass."""
    rows = _rows(_base_series())
    assert rows and all("sp_buy" not in r and "sp_end" not in r for r in rows)


# ── the training label ───────────────────────────────────────────────────────

def test_within_day_rank_centred_and_per_day():
    y = np.array([1.0, 3.0, 2.0, -5.0, 0.0], dtype=np.float32)
    day = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    r = within_day_rank(y, day)
    assert r[:3] == pytest.approx([-0.5, 0.5, 0.0])       # day 0: 1 < 2 < 3
    assert r[3:] == pytest.approx([-0.5, 0.5])            # day 1: -5 < 0
    assert float(np.abs(r).max()) <= 0.5


def test_sub_threshold_wiggles_never_pivot():
    """The 2026-08-12 min-move directive: runs smaller than
    ``pivot_min_move_pct`` are not worth the spread and must not become
    pivots. A 0.5% saw inside a rising trend leaves the up-leg unbroken."""
    from src.analysis.pivot_target import _resolved_pivots

    base = np.linspace(100.0, 130.0, 80)
    saw = 1.0 + 0.0025 * np.where(np.arange(80) % 2 == 0, 1.0, -1.0)  # ±0.25%
    c = base * saw
    h, lo = c * 1.002, c * 0.998                    # bar range ~0.4% < 1%
    P, PP, FL, CF = _resolved_pivots(c, h, lo)
    # the seeding trough at the series start may confirm; no PEAK may exist —
    # the saw never retraces 1% from a running high inside the trend
    assert not FL.any(), f"sub-threshold wiggles produced peak pivots at {P[FL]}"


def test_pivot_confirmation_is_the_threshold_reversal_bar():
    """Variable-lag seam: the pivot's confirmation index is the bar whose
    adverse print reached the threshold — NOT extreme+1."""
    from src.analysis.pivot_target import _resolved_pivots

    c = np.r_[np.linspace(100, 120, 60),            # up-leg to the peak
              [119.9, 119.85, 119.7, 118.0, 116.0]]  # slow bleed, then the 1% print
    h, lo = c * 1.0001, c * 0.9999                   # razor-thin bars
    P, PP, FL, CF = _resolved_pivots(c, h, lo)
    peaks = [(int(p), int(cf)) for p, cf, f in zip(P, CF, FL) if f]
    assert peaks, "the peak must resolve once the 1% reversal prints"
    p, cf = peaks[-1]
    assert p == 59
    assert cf == 63, f"confirmation must be the 1% print (bar 63), got {cf}"
