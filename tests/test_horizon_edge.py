"""Edge-decay time-stop: the realized edge-by-horizon curve, the calibrated
edge-positive window, and its evidence-throttled feed into the exit floor.

Guards that the curve measures the oriented edge per horizon, that the calibration
finds the last-positive-edge window, that the floor nudge is inert on thin data /
within the window and bounded past it, and that the per-position exit pressure
ramps once a position is held past the window.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import src.analysis.horizon_edge as he
from config.settings import settings


@pytest.fixture(autouse=True)
def _reset():
    he.reset_cache()
    yield
    he.reset_cache()


def _panel():
    """Actionable rows whose oriented edge is POSITIVE at 1–3d and NEGATIVE at
    5–7d — the peak-and-decay shape the time-stop targets. combined_score>0, so
    oriented return == forward return."""
    f = {1: [1.0]*40 + [-1.0]*20,     # win 66.7%, edge +0.33
         3: [1.0]*33 + [-1.0]*27,     # win 55.0%, edge +0.10
         5: [-1.0]*40 + [1.0]*20,     # win 33.3%, edge −0.33
         7: [-1.0]*45 + [1.0]*15}     # win 25.0%, edge −0.50
    rows = []
    for i in range(60):
        # combined_score varies but stays positive (so oriented edge == fwd ret,
        # and Spearman is non-degenerate — no constant-input warnings).
        rows.append({"signal_date": "2026-01-02", "combined_score": 0.30 + 0.006 * i,
                     "confidence": 0.80, "fwd_ret_1d": f[1][i], "fwd_ret_3d": f[3][i],
                     "fwd_ret_5d": f[5][i], "fwd_ret_7d": f[7][i]})
    return pd.DataFrame(rows)


def test_edge_curve_measures_peak_and_decay():
    c = he.compute_horizon_edge_curve(panel=_panel(), horizons=(1, 3, 5, 7),
                                      conf_min=0.78, min_n=10)
    by = {int(r["horizon"]): r for _, r in c.iterrows()}
    assert by[1]["edge"] > 0 and by[1]["win"] > 50       # edge present short-horizon
    assert by[5]["edge"] < 0 and by[5]["win"] < 50       # decayed / reversed by 5d


def test_confidence_filter_excludes_below_threshold():
    p = _panel()
    p.loc[:29, "confidence"] = 0.50                       # half below the 0.78 gate
    c = he.compute_horizon_edge_curve(panel=p, horizons=(1,), conf_min=0.78, min_n=5)
    assert int(c.iloc[0]["n"]) == 30                      # only the actionable half counts


def test_edge_exhaustion_days_is_last_positive():
    c = he.compute_horizon_edge_curve(panel=_panel(), horizons=(1, 3, 5, 7),
                                      conf_min=0.78, min_n=10)
    assert he.edge_exhaustion_days(c, min_n=10) == 3      # positive through 3d, negative at 5d


def test_calibration_finds_window_and_gentle_strength(monkeypatch):
    monkeypatch.setattr(settings, "edge_decay_min_n", 10)
    cal = he.calibrate_edge_horizon(panel=_panel())
    assert cal["edge_days"] == 3
    assert 0.0 < cal["strength"] < 0.5                   # evidence-throttled (n small vs prior)
    assert cal["peak_day"] == 1                           # +0.33 at 1d is the max here


def test_calibration_inert_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_edge_decay_exit", False)
    assert he.calibrate_edge_horizon(panel=_panel()) == dict(he._INERT)


def test_calibration_inert_when_no_decay(monkeypatch):
    # Edge positive at EVERY horizon → never exhausts → no window → inert.
    monkeypatch.setattr(settings, "edge_decay_min_n", 10)
    p = _panel()
    for h in (5, 7):
        p[f"fwd_ret_{h}d"] = 1.0
    cal = he.calibrate_edge_horizon(panel=p)
    assert cal["strength"] == 0.0


# ── the exit-pressure signal + the floor nudge ───────────────────────────────

def _cal(edge_days=3, strength=0.2):
    return {"edge_days": edge_days, "strength": strength, "n_obs": 55, "peak_day": 2}


def test_edge_decay_pressure_zero_within_window_negative_past(monkeypatch):
    from src.analysis import exit_methods as em
    # _edge_decay_pressure does `from horizon_edge import calibrate_edge_horizon`
    # at call time, so patch the source module.
    monkeypatch.setattr(he, "calibrate_edge_horizon", lambda *a, **k: _cal(edge_days=3))
    fresh = {"entry_date": (date.today() - timedelta(days=1)).isoformat()}   # ~1 trading day
    old = {"entry_date": (date.today() - timedelta(days=20)).isoformat()}    # well past 3d
    assert em._edge_decay_pressure(fresh) == 0.0
    assert em._edge_decay_pressure(old) < 0.0


def test_floor_adjustment_raises_floor_only_past_window():
    cal = _cal(edge_days=3, strength=0.2)
    assert he.edge_decay_floor_adjustment({"edge_decay": 0.0}, cal) == 0.0     # within window
    adj = he.edge_decay_floor_adjustment({"edge_decay": -1.0}, cal)            # fully past
    assert adj == pytest.approx(0.2 * settings.edge_decay_floor_cap)           # strength × cap
    assert adj <= settings.edge_decay_floor_cap                                # bounded


def test_floor_adjustment_inert_when_strength_zero():
    assert he.edge_decay_floor_adjustment({"edge_decay": -1.0}, _cal(strength=0.0)) == 0.0
