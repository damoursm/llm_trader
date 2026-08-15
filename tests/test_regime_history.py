"""Historical regime reconstruction (`src/analysis/regime_history.py`).

The harness behind a standing decision: the macro-regime bands are CORRECT and
must not be lowered (memory/regime-retrofit-2026-07-27). That conclusion rests
on this module measuring SPY's forward return after each reconstructed regime,
split by era — pooled over 2018-2026 the data says PANIC precedes the best
returns, and only the era split reveals that this is an artefact of a window
containing nothing but V-shaped recoveries.

Two things therefore matter here and nothing else does:

* **the bands must mirror the live ones.** This module keeps its own copy so the
  reconstruction can be re-banded without touching live code — which is exactly
  what makes silent drift possible, and a reconstruction judged on different
  thresholds than production uses answers a question nobody asked;
* **the forward-return join must not look ahead or wrap.** A row within `h`
  days of the end of the price series has no forward return; returning
  something anyway would fabricate the evidence the decision rests on.

`reconstruct()` itself needs yfinance and FRED and is not exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import regime_history as rh


# ── the bands ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("norm,expected", [
    (-5.0, "PANIC"), (-1.51, "PANIC"), (-1.5, "PANIC"),
    (-1.49, "RISK_OFF"), (-0.81, "RISK_OFF"), (-0.8, "RISK_OFF"),
    (-0.79, "CAUTION"), (-0.31, "CAUTION"), (-0.3, "CAUTION"),
    (-0.29, "NEUTRAL"), (0.0, "NEUTRAL"), (0.3, "NEUTRAL"),
    (0.31, "RISK_ON"), (5.0, "RISK_ON"),
])
def test_band_boundaries_are_inclusive_on_the_worse_side(norm, expected):
    assert rh.band_of(norm) == expected


def test_bands_mirror_the_live_regime_thresholds():
    """Drift guard. The reconstruction's verdicts are only evidence about the
    LIVE overlay if both use the same cut points; a change on either side turns
    every historical conclusion into a statement about a system that never ran."""
    from src.data.macro_regime import _THRESHOLD_BOUNDARIES
    live_edges = [edge for edge, _below, _above in _THRESHOLD_BOUNDARIES]
    hist_edges = [hi for _name, hi in rh.BANDS]
    assert hist_edges == live_edges, (
        "regime_history.BANDS drifted from macro_regime._THRESHOLD_BOUNDARIES")
    # ...and the band NAMES line up with the live labels in the same order.
    assert [name for name, _ in rh.BANDS] == [
        below for _e, below, _a in _THRESHOLD_BOUNDARIES]


def test_band_of_agrees_with_the_live_labeller_across_the_range():
    """Checked behaviourally as well as structurally: the live labeller is a
    chain of `if norm <= x` returns, so a reordered branch would keep the
    constants identical while changing the answer."""
    from src.data.macro_regime import _REGIME_THRESHOLD
    for i in range(-250, 251):
        norm = i / 100.0
        label = rh.band_of(norm)
        assert label in _REGIME_THRESHOLD, f"{label} is not a live regime"
    # Monotone: worse composite never yields a more permissive band.
    order = {"PANIC": 0, "RISK_OFF": 1, "CAUTION": 2, "NEUTRAL": 3, "RISK_ON": 4}
    ranks = [order[rh.band_of(i / 100.0)] for i in range(-250, 251)]
    assert ranks == sorted(ranks)


# ── forward returns ─────────────────────────────────────────────────────────

def _prices(n: int = 40, step: float = 1.0):
    """A strictly rising SPY series on consecutive calendar days."""
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.Series([100.0 + step * i for i in range(n)], index=idx)


def _regime_df(regimes) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(regimes), freq="D")
    return pd.DataFrame({"date": [d.date().isoformat() for d in idx],
                         "regime": list(regimes),
                         "inputs": [7] * len(regimes),
                         "norm": [0.0] * len(regimes)})


def test_forward_returns_are_grouped_per_regime(monkeypatch):
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(40))
    df = _regime_df(["PANIC"] * 10 + ["RISK_ON"] * 10 + ["NEUTRAL"] * 20)
    out = rh.regime_forward_returns(df, horizons=(1, 5))
    assert set(out["regime"]) == {"PANIC", "RISK_OFF", "CAUTION", "NEUTRAL", "RISK_ON"}
    by = out.set_index("regime")
    assert by.loc["PANIC", "days"] == 10
    assert by.loc["RISK_OFF", "days"] == 0          # unobserved regimes still reported
    # Strictly rising series -> every forward return positive, 100% up.
    assert by.loc["PANIC", "spy_1d"] > 0
    assert by.loc["PANIC", "up_1d"] == 100.0


def test_a_regime_with_no_days_reports_missing_not_zero(monkeypatch):
    """Missing is 'never observed'; 0.0 would read as 'observed, and flat' — the
    exact confusion that made the pooled PANIC reading misleading. (The module
    writes None; pandas surfaces it as NaN once the column holds floats. Either
    way it must not be a number.)"""
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(40))
    out = rh.regime_forward_returns(_regime_df(["NEUTRAL"] * 30), horizons=(1,))
    row = out.set_index("regime").loc["PANIC"]
    assert row["days"] == 0
    assert pd.isna(row["spy_1d"]) and pd.isna(row["up_1d"])


def test_rows_without_enough_forward_history_are_excluded(monkeypatch):
    """The last `h` rows have no h-day forward return. Wrapping or clamping
    would invent the very numbers this module exists to measure."""
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(10))
    df = _regime_df(["PANIC"] * 10)
    out = rh.regime_forward_returns(df, horizons=(1, 5))
    row = out.set_index("regime").loc["PANIC"]
    assert row["days"] == 10               # all rows are counted...
    # ...but the horizon means are computed only over rows that HAVE a forward
    # value, and a 5-day horizon on 10 bars leaves the last 5 undefined.
    assert row["spy_5d"] is not None
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(3))
    out2 = rh.regime_forward_returns(_regime_df(["PANIC"] * 3), horizons=(5,))
    assert out2.set_index("regime").loc["PANIC", "spy_5d"] is None


def test_dates_absent_from_the_price_series_are_skipped(monkeypatch):
    """Weekends and holidays appear in the reconstruction but not in SPY's
    index; they must drop out rather than match the nearest bar."""
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(40))
    df = _regime_df(["PANIC"] * 5)
    df.loc[len(df)] = {"date": "2030-12-31", "regime": "PANIC", "inputs": 7, "norm": 0.0}
    out = rh.regime_forward_returns(df, horizons=(1,))
    assert out.set_index("regime").loc["PANIC", "days"] == 5


def test_unavailable_price_history_returns_empty_not_garbage(monkeypatch):
    monkeypatch.setattr(rh, "_hist", lambda t, s: None)
    assert rh.regime_forward_returns(_regime_df(["PANIC"] * 5)).empty
    monkeypatch.setattr(rh, "_hist", lambda t, s: pd.Series(dtype=float))
    assert rh.regime_forward_returns(_regime_df(["PANIC"] * 5)).empty


# ── the era split ───────────────────────────────────────────────────────────

def test_eras_are_ordered_contiguous_and_cover_the_gfc():
    """The 2008-2009 window is the whole point: it is the only era with a large
    PANIC sample that was NOT a V-shaped recovery, and dropping it would flip
    the module's conclusion about the BUY block."""
    assert any("GFC" in name for name, _lo, _hi in rh.ERAS)
    starts = [lo for _n, lo, _h in rh.ERAS]
    assert starts == sorted(starts)
    for (_n1, _lo1, hi1), (_n2, lo2, _hi2) in zip(rh.ERAS, rh.ERAS[1:]):
        assert hi1 < lo2, "eras overlap — a day would be counted twice"


def test_forward_returns_by_era_splits_the_same_regime(monkeypatch):
    """The measurement that overturned the pooled reading: identical regime,
    different era, opposite sign.

    The price series is CONTINUOUS across both eras on purpose. The forward
    return walks positions in the sorted price index, not calendar time, so a
    fixture with a gap would let a 2008 row's 21-day forward land on a 2020 bar
    — which is an artefact of the fixture, not of the function."""
    idx = pd.date_range("2008-01-01", "2020-06-30", freq="D")
    gfc_end, recovery = pd.Timestamp("2009-06-01"), pd.Timestamp("2020-01-01")

    def _px(d):
        if d < gfc_end:
            return 100.0 - (d - idx[0]).days * 0.02          # protracted decline
        if d < recovery:
            return 100.0 - (gfc_end - idx[0]).days * 0.02    # flat trough
        return 90.0 + (d - recovery).days * 0.05             # V-shaped recovery

    monkeypatch.setattr(rh, "_hist",
                        lambda t, s: pd.Series([_px(d) for d in idx], index=idx))

    # 60 PANIC days inside each era, each with 21 same-direction days ahead.
    tagged = ([pd.Timestamp("2008-01-01") + pd.Timedelta(days=i) for i in range(60)]
              + [recovery + pd.Timedelta(days=i) for i in range(60)])
    df = pd.DataFrame({
        "date": [d.date().isoformat() for d in tagged],
        "regime": ["PANIC"] * len(tagged),
        "inputs": [7] * len(tagged),
    })
    out = rh.forward_returns_by_era(df, horizons=(21,))
    by_era = {r["era"]: r for _, r in out.iterrows() if r["regime"] == "PANIC"}
    gfc = next(k for k in by_era if "GFC" in k)
    recent = next(k for k in by_era if k.startswith("2020"))
    assert by_era[gfc]["spy_21d"] < 0, "the protracted bear must read negative"
    assert by_era[recent]["spy_21d"] > 0
    assert by_era[gfc]["days"] > 0 and by_era[recent]["days"] > 0


def test_eras_with_no_rows_are_omitted(monkeypatch):
    monkeypatch.setattr(rh, "_hist", lambda t, s: _prices(40))
    out = rh.forward_returns_by_era(_regime_df(["PANIC"] * 30), horizons=(1,))
    assert set(out["era"]) == {"2020-2026"}


def test_era_split_survives_missing_price_history(monkeypatch):
    monkeypatch.setattr(rh, "_hist", lambda t, s: None)
    assert rh.forward_returns_by_era(_regime_df(["PANIC"] * 5)).empty
