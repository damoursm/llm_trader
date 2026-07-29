"""Per-method horizon skill + the method-derived exit horizon (2026-07-26).

A method is not good or bad — it is good over a particular HOLDING PERIOD.
Judged at 1 day only, exactly one method cleared p<0.05 over the solo-method
simulation and the book would have collapsed to a single signal. Judged across
1/3/5/10 days, five clear with coherent shapes: `sent_velocity` spikes at 1d and
decays (it measures a rate of change, so that IS its expected profile),
`pattern` builds monotonically and clears at both 5d and 10d, `oi_skew` is
slowest.

Two consumers are pinned here:
  * WEIGHTING — three states. Proven keeps full weight, DISPROVEN (significantly
    below 50% at every judgeable horizon) is dropped, and everything between
    gets REDUCED weight. Absence of evidence is not evidence of absence, and
    zeroing the middle would collapse the ensemble that coherence,
    sources_agreeing, family agreement and Gate 1b all need.
  * EXIT — a position's target hold is the conviction-weighted average of its
    own methods' best horizons, so a `sent_velocity` trade is stale after a day
    while a `pattern` trade still has a week to run. It joins the exit consensus
    rather than replacing any existing rule.

All synthetic, no network, no DB.
"""

import pandas as pd
import pytest

from config.settings import settings
import src.analysis.method_horizons as mh


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "enable_method_horizons", True)
    monkeypatch.setattr(settings, "method_horizon_alpha", 0.05)
    monkeypatch.setattr(settings, "method_horizon_min_obs", 30)
    mh.reset_cache()
    yield
    mh.reset_cache()


def _perf(rows):
    """rows: {method: {horizon: (win_pct, n)}} -> a compute_method_perf frame."""
    recs = []
    for m, by_h in rows.items():
        r = {"method": m}
        for h, (w, n) in by_h.items():
            r[f"win_{h}"], r[f"n_{h}"] = w, n
        recs.append(r)
    return pd.DataFrame(recs)


def _install(monkeypatch, rows, events=None):
    monkeypatch.setattr(mh, "_CACHE", {})
    import src.analysis.simulated_trades as st
    ev = events if events is not None else pd.DataFrame(
        {"score": [0.5] * 10, "method": ["x"] * 10})
    monkeypatch.setattr(st, "load_sim_entry_events", lambda days=None: ev)
    monkeypatch.setattr(st, "compute_method_perf", lambda **kw: _perf(rows))


# ── state classification ───────────────────────────────────────────────────

def test_a_method_proven_at_a_LATER_horizon_is_kept(monkeypatch):
    """The whole point. `pattern` is noise at 1d and real at 5d — judging it on
    a one-day hold would have discarded a working signal."""
    _install(monkeypatch, {"pattern": {"1d": (48.6, 1701), "3d": (51.4, 1624),
                                       "1w": (52.8, 1500), "2w": (52.8, 1400)}})
    d = mh.compute_method_horizons()["pattern"]
    assert d["state"] == mh.PROVEN
    assert d["best_horizon"] == "1w"
    assert d["best_days"] == 5.0


def test_a_fast_method_is_proven_at_1d(monkeypatch):
    _install(monkeypatch, {"sent_velocity": {"1d": (52.4, 3348), "3d": (50.0, 2978),
                                             "1w": (50.9, 2800), "2w": (50.7, 2600)}})
    d = mh.compute_method_horizons()["sent_velocity"]
    assert d["state"] == mh.PROVEN and d["best_days"] == 1.0


def test_disproven_requires_losing_at_EVERY_horizon(monkeypatch):
    """Dropping a method is the strong action, so one bad horizon is not enough
    — it must fail everywhere it can be judged."""
    _install(monkeypatch, {
        "market_momentum": {"1d": (45.3, 1993), "3d": (46.5, 1818),
                            "1w": (43.9, 1700), "2w": (43.0, 1600)},
        "vwap":            {"1d": (45.0, 1500), "3d": (50.5, 1500),
                            "1w": (50.2, 1500), "2w": (49.8, 1500)},
    })
    res = mh.compute_method_horizons()
    assert res["market_momentum"]["state"] == mh.DISPROVEN
    assert res["vwap"]["state"] == mh.UNPROVEN, "one bad horizon is not disproof"


def test_the_middle_is_UNPROVEN_not_disproven(monkeypatch):
    _install(monkeypatch, {"vwap": {"1d": (50.2, 2692), "3d": (48.3, 2478),
                                    "1w": (47.5, 2300), "2w": (50.2, 2100)}})
    assert mh.compute_method_horizons()["vwap"]["state"] == mh.UNPROVEN


def test_thin_horizons_are_ignored_not_guessed(monkeypatch):
    _install(monkeypatch, {"pead": {"1d": (90.0, 3), "3d": (10.0, 2)}})
    assert mh.compute_method_horizons().get("pead", {}).get("state") != mh.PROVEN


# ── coherence: the guard against a lone significant spike ──────────────────

def test_adjacent_support_reads_as_coherent(monkeypatch):
    _install(monkeypatch, {"pattern": {"1d": (48.6, 1701), "3d": (51.4, 1624),
                                       "1w": (52.8, 1500), "2w": (52.8, 1400)}})
    assert mh.compute_method_horizons()["pattern"]["coherent"] is True


def test_a_lone_spike_is_flagged_incoherent(monkeypatch):
    """Correlated horizons make a count-based correction the wrong tool, so
    SHAPE is the guard: significance at one horizon with noise on both sides is
    the profile a fluke produces."""
    # iv_expr is the real-world example of this shape: 50.0 / 56.6 / 54.3 / 47.6.
    _install(monkeypatch, {"iv_expr": {"1d": (44.0, 2000), "3d": (55.0, 2000),
                                       "1w": (44.0, 2000), "2w": (44.0, 2000)}})
    d = mh.compute_method_horizons()["iv_expr"]
    assert d["state"] == mh.PROVEN and d["coherent"] is False


# ── the score gate ─────────────────────────────────────────────────────────

def test_only_gated_events_are_measured(monkeypatch):
    """A solo method fires a direction only at |score| >= buy_sell_diff_threshold,
    so ungated calls would credit it for trades the system would never take."""
    seen = {}
    ev = pd.DataFrame({"score": [0.05, 0.20, -0.30, -0.01], "method": ["m"] * 4})
    import src.analysis.simulated_trades as st
    monkeypatch.setattr(mh, "_CACHE", {})
    monkeypatch.setattr(st, "load_sim_entry_events", lambda days=None: ev)

    def spy(**kw):
        seen["n"] = len(kw.get("sim_df"))
        return _perf({"vwap": {"1d": (52.0, 500)}})
    monkeypatch.setattr(st, "compute_method_perf", spy)
    mh.compute_method_horizons()
    assert seen["n"] == 2, "only |score| >= 0.15 should reach the measurement"


def test_disabled_returns_nothing(monkeypatch):
    monkeypatch.setattr(settings, "enable_method_horizons", False)
    assert mh.compute_method_horizons() == {}


def test_failure_is_fail_soft(monkeypatch):
    import src.analysis.simulated_trades as st
    monkeypatch.setattr(mh, "_CACHE", {})
    monkeypatch.setattr(st, "load_sim_entry_events",
                        lambda days=None: (_ for _ in ()).throw(RuntimeError("no panel")))
    assert mh.compute_method_horizons() == {}


# ── weighting: three states ────────────────────────────────────────────────

def test_unproven_gets_reduced_weight_not_zero(monkeypatch):
    import src.signals.aggregator as agg
    monkeypatch.setattr(settings, "unproven_weight_multiplier", 0.5)
    monkeypatch.setattr(mh, "compute_method_horizons", lambda: {
        "pattern": {"state": mh.PROVEN}, "vwap": {"state": mh.UNPROVEN},
        "market_momentum": {"state": mh.DISPROVEN}})
    mults = agg.method_state_multipliers()
    assert mults["pattern"] == 1.0
    assert mults["vwap"] == 0.5
    assert "market_momentum" not in mults, "disproven is the filter's job, not a weight"
    assert agg.horizon_disproven_methods() == frozenset({"market_momentum"})


# ── the exit method ────────────────────────────────────────────────────────

def _trade(scores, entry="2026-07-01"):
    return {"method_scores": scores, "entry_date": entry}


def test_target_hold_follows_the_methods_that_drove_the_entry(monkeypatch):
    from src.analysis import exit_methods as em
    monkeypatch.setattr(mh, "compute_method_horizons", lambda: {
        "sent_velocity": {"state": mh.PROVEN, "best_days": 1.0},
        "pattern":       {"state": mh.PROVEN, "best_days": 5.0}})
    fast = em.method_horizon_days(_trade({"sent_velocity": 0.8}))
    slow = em.method_horizon_days(_trade({"pattern": 0.8}))
    assert fast == pytest.approx(1.0)
    assert slow == pytest.approx(5.0)
    mixed = em.method_horizon_days(_trade({"sent_velocity": 0.8, "pattern": 0.8}))
    assert fast < mixed < slow, "a mixed trade sits between its methods' horizons"


def test_conviction_weights_the_horizon(monkeypatch):
    """A trade carried mostly by the fast method should get a fast clock."""
    from src.analysis import exit_methods as em
    monkeypatch.setattr(mh, "compute_method_horizons", lambda: {
        "sent_velocity": {"state": mh.PROVEN, "best_days": 1.0},
        "pattern":       {"state": mh.PROVEN, "best_days": 5.0}})
    loud_fast = em.method_horizon_days(_trade({"sent_velocity": 0.9, "pattern": 0.1}))
    loud_slow = em.method_horizon_days(_trade({"sent_velocity": 0.1, "pattern": 0.9}))
    assert loud_fast < loud_slow


def test_unproven_only_trade_gets_NO_deadline(monkeypatch):
    """Better no view than a fabricated one — an unproven method has no measured
    horizon to offer."""
    from src.analysis import exit_methods as em
    monkeypatch.setattr(mh, "compute_method_horizons", lambda: {
        "tech": {"state": mh.UNPROVEN, "best_days": None}})
    assert em.method_horizon_days(_trade({"tech": 0.9})) is None
    assert em._method_horizon_pressure(_trade({"tech": 0.9})) == 0.0


def test_pressure_is_zero_inside_the_window_and_negative_past_it(monkeypatch):
    from datetime import date, timedelta
    from src.analysis import exit_methods as em
    monkeypatch.setattr(mh, "compute_method_horizons", lambda: {
        "pattern": {"state": mh.PROVEN, "best_days": 5.0}})
    fresh = _trade({"pattern": 0.8}, entry=str(date.today()))
    old = _trade({"pattern": 0.8}, entry=str(date.today() - timedelta(days=60)))
    assert em._method_horizon_pressure(fresh) == 0.0
    assert em._method_horizon_pressure(old) < 0.0
    assert em._method_horizon_pressure(old) >= -1.0, "bounded like every exit score"


def test_it_participates_in_the_exit_consensus():
    """The user's requirement: another input weighted into exit confidence, not
    a replacement for the existing rules. Unlike `horizon` (LLM-stated) and
    `edge_decay` (one global window), this is a statement BY the methods."""
    from src.analysis.exit_conviction import _CONSENSUS_SKIP, exit_method_consensus
    assert "method_horizon" not in _CONSENSUS_SKIP
    assert "horizon" in _CONSENSUS_SKIP and "edge_decay" in _CONSENSUS_SKIP
    c = exit_method_consensus({"tech": 0.0, "method_horizon": -1.0})
    assert c is not None and c < 0, "it must be able to move the consensus"
