"""Probes for the signed pivot target + leg features (``analysis/pivot_target``).

The failure these guard is invisible and fatal: ``sign(sp_buy) equals the sign
of TOMORROW's move`` by construction, so any leak across the one-bar pivot
confirmation seam hands the model the answer key and produces a spectacular,
entirely fake validation. Every test here attacks that seam or the worked-
example semantics; the serving-parity test pins ``latest_leg_features`` to the
dataset pass so train/serve drift cannot open the same hole later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.pivot_target import (LEG_FEATURES, MAX_PIVOT_DAYS,
                                       _leg_target_rows, within_day_rank)


def _dates(n):
    return list(pd.date_range("2024-01-02", periods=n, freq="B").date)


def _rows(c, h=None, lo=None, **kw):
    c = np.asarray(c, dtype=float)
    h = c * 1.01 if h is None else np.asarray(h, dtype=float)
    lo = c * 0.99 if lo is None else np.asarray(lo, dtype=float)
    return _leg_target_rows(c, h, lo, _dates(len(c)), **kw)


def _base_series(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return 50 * np.cumprod(1 + rng.normal(0, 0.02, n))


# ── worked example ───────────────────────────────────────────────────────────

def test_worked_example_signed_target():
    # Path: 10, 11, 12 (peak), 11, 10 (trough), 11, 12, ... peak at i=2, trough at i=4.
    c = np.array([10.0, 11, 12, 11, 10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 12,
                  13, 14, 13, 12, 13, 14, 15, 14, 13, 14, 15, 16, 15, 14, 15,
                  16, 17, 16, 15, 16, 17, 18, 17, 16, 17, 18, 19, 18, 17, 18,
                  19, 20, 19, 18, 19, 20, 21], dtype=float)
    rows = {r["signal_date"]: r for r in _rows(c)}
    d = _dates(len(c))
    # H/L basis: the helper feeds h = c*1.01 / lo = c*0.99, so the target lands
    # on the pivot bar's EXTREME. At i=0 (rising toward the i=2 peak):
    # target = (12*1.01)/10 - 1 = +21.2%, end = date[2].
    r0 = rows[d[0].isoformat()]
    assert r0["sp_buy"] == pytest.approx(21.2)
    assert r0["sp_end"] == d[2].isoformat()
    # At i=2 (the peak itself): next pivot is the trough at i=4 ->
    # (10*0.99)/12 - 1 = -17.5%.
    r2 = rows[d[2].isoformat()]
    assert r2["sp_buy"] == pytest.approx(-17.5, abs=1e-3)
    assert r2["sp_end"] == d[4].isoformat()
    # sign(target) == sign(tomorrow's move) on every emitted row.
    for i in range(len(c) - 1):
        r = rows.get(d[i].isoformat())
        if r is None or "sp_buy" not in r:
            continue
        move = c[i + 1] - c[i]
        if move != 0:
            assert np.sign(r["sp_buy"]) == np.sign(move), f"row {i}"


def test_monotone_series_has_no_settled_targets():
    # A strictly rising series has no interior pivot -> nothing to train on.
    rows = _rows(np.linspace(10, 60, 80))
    assert rows == []


def test_max_pivot_days_cap():
    # Rise 100 bars to a single peak then fall: early rows' pivot is > 60 days out.
    c = np.r_[np.linspace(10, 30, 100), np.linspace(29.9, 20, 20)]
    rows = _rows(c)
    gaps = [(pd.Timestamp(r["sp_end"]) - pd.Timestamp(r["signal_date"])).days
            for r in rows]
    # calendar gaps exceed trading gaps, so just assert the trading-day cap held
    settled_idx = [r["signal_date"] for r in rows]
    assert settled_idx, "cap test needs some settled rows"
    d = _dates(len(c))
    for r in rows:
        i = d.index(pd.Timestamp(r["signal_date"]).date())
        j = d.index(pd.Timestamp(r["sp_end"]).date())
        assert j - i <= MAX_PIVOT_DAYS


# ── causality at the confirmation seam ───────────────────────────────────────

def test_settled_rows_immune_to_future_mutation():
    """Rewriting every bar after a settled row's pivot END must change nothing
    about that row — target, end date, or any leg feature."""
    c = _base_series()
    rows_before = {r["signal_date"]: r for r in _rows(c)}
    assert rows_before, "need settled rows"
    # take a row whose pivot end is safely inside the series
    d = _dates(len(c))
    probe = None
    for r in rows_before.values():
        j = d.index(pd.Timestamp(r["sp_end"]).date())
        if j < len(c) - 36:
            probe = (r, j)
    assert probe is not None
    r_ref, j_end = probe
    c2 = c.copy()
    # The threshold zigzag confirms a pivot only when the 1% reversal PRINTS
    # (variable lag), so immunity is guaranteed only past the confirmation
    # bar; +6 bars of 2% noise is deterministic slack for seed 7.
    c2[j_end + 6:] = c2[j_end + 6:] * 7.0 + 3.0          # violent future rewrite
    rows_after = {r["signal_date"]: r for r in _rows(c2)}
    r_new = rows_after[r_ref["signal_date"]]
    for k, v in r_ref.items():
        got = r_new[k]
        if isinstance(v, float) and v == v:
            assert got == pytest.approx(v), f"{k} moved under future mutation"
        elif v == v:                                       # str fields
            assert got == v, f"{k} moved under future mutation"


def test_leg_features_use_only_confirmed_pivots():
    """A pivot at bar j is usable from bar j+1 only. Truncating the series at the
    pivot bar itself (so its confirming bar never prints) must leave bar j's leg
    features exactly as they were computed from the shorter history."""
    c = _base_series(seed=11)
    full = _rows(c, require_target=False)
    # find an interior pivot: leg_dir flips at the bar AFTER the pivot prints
    d = _dates(len(c))
    full_by_date = {r["signal_date"]: r for r in full}
    # serve at bar i from the truncated series c[:i+1]; compare vs the full pass
    for i in (60, 80, 100):
        trunc = _rows(c[: i + 1], require_target=False, only_last=True)
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
    full = {r["signal_date"]: r for r in _rows(c, require_target=False)}
    last = _rows(c, only_last=True, require_target=False)[0]
    ref = full[d[len(c) - 1].isoformat()]
    for k in LEG_FEATURES:
        a, b = last.get(k), ref.get(k)
        if a is None or a != a:
            assert b is None or b != b
        else:
            assert a == pytest.approx(b)


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
    import numpy as np

    from src.analysis.pivot_target import _resolved_pivots

    base = np.linspace(100.0, 130.0, 80)
    saw = 1.0 + 0.0025 * np.where(np.arange(80) % 2 == 0, 1.0, -1.0)  # ±0.25%
    c = base * saw
    h, lo = c * 1.002, c * 0.998                    # intraday range ~0.4% < 1%
    P, PP, FL, CF = _resolved_pivots(c, h, lo)
    # the seeding trough at the series start may confirm; no PEAK may exist —
    # the saw never retraces 1% from a running high inside the trend
    assert not FL.any(), f"sub-threshold wiggles produced peak pivots at {P[FL]}"


def test_pivot_confirmation_is_the_threshold_reversal_bar():
    """Variable-lag seam: the pivot's confirmation index is the bar whose
    adverse print reached the threshold — NOT extreme+1."""
    import numpy as np

    from src.analysis.pivot_target import _resolved_pivots

    c = np.r_[np.linspace(100, 120, 60),            # up-leg to the peak
              [119.9, 119.85, 119.7, 118.0, 116.0]]  # slow bleed, then the 1% print
    h, lo = c * 1.0001, c * 0.9999                   # razor-thin bars
    P, PP, FL, CF = _resolved_pivots(c, h, lo)
    pk = [k for k in range(len(P)) if FL[k]]
    assert pk, "need the peak"
    k = pk[-1]
    assert P[k] == 59                                # the extreme bar
    # 120*0.99 = 118.8 — first low at/below that is bar 63 (118.0*0.9999)
    assert CF[k] == 63, f"confirmation must wait for the 1% print (got {CF[k]})"
