"""Panel-ICIR arm of the inversion detector (2026-07-25).

The ledger arm is starved (~150 attributed trades, so nothing clears
Bonferroni). This arm reads the `simulated_trades` panel — tens of thousands of
solo directional calls — through `compute_directional_perf`, the documented
market-relative inversion readout.

**The shared-decay control is the design, not a refinement.** Measured on the
live panel, the share of methods with negative ICIR runs 35% @30m → 61% @1d →
76% @2w → 79% @1m, while `combined_score` ITSELF runs −0.227 @1d to −2.187 @2w.
That is the system's holding-period edge decay, shared by every method. Scoring
a method against ZERO at a long horizon therefore flags almost the whole book —
it measures how long a position is held, not whether the signal points
backwards. Inverting a method that merely shares the decay cannot help, because
the decay is about duration, not direction.

All synthetic, no network.
"""

import pytest

from config.settings import settings
import src.performance.tracker as tk


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "enable_inversion_panel_arm", True)
    monkeypatch.setattr(settings, "inversion_panel_horizons", "1d,3d")
    monkeypatch.setattr(settings, "inversion_panel_min_t", 2.0)
    tk._PANEL_INVERSION_CACHE.clear()
    yield
    tk._PANEL_INVERSION_CACHE.clear()


def _panel(monkeypatch, rows, days=100):
    """rows: {method: {horizon: icir}} — always includes combined_score."""
    import pandas as pd
    recs = []
    for m, by_h in rows.items():
        r = {"method": m, "side": "both"}
        for h, v in by_h.items():
            r[f"icir_{h}"] = v
            r[f"icdays_{h}"] = days
        recs.append(r)
    df = pd.DataFrame(recs)
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf",
                        lambda **kw: df)
    tk._PANEL_INVERSION_CACHE.clear()


# ── the shared-decay control ───────────────────────────────────────────────

def test_a_method_that_merely_SHARES_the_decay_is_not_inverted(monkeypatch):
    """THE test. Both the method and the system are deeply negative — the
    method is no worse than the book it lives in, so flipping it would be
    betting against holding period, not correcting a backwards signal."""
    _panel(monkeypatch, {
        "combined_score": {"1d": -0.90, "3d": -1.20},
        "tech":           {"1d": -0.90, "3d": -1.20},   # identical to the system
    })
    ev = tk.panel_inversion_evidence()
    assert ev["tech"]["excess"]["1d"] == pytest.approx(0.0)
    assert not ev["tech"]["qualifies"], (
        "a method matching the system's own decay must not be inverted")


def test_uncontrolled_negativity_would_have_flagged_it(monkeypatch):
    """Proves the control is load-bearing: the same method IS flagged once it
    is genuinely worse than the system, on identical raw numbers for itself."""
    _panel(monkeypatch, {
        "combined_score": {"1d": 0.00, "3d": 0.00},     # system healthy…
        "tech":           {"1d": -0.90, "3d": -1.20},   # …method is not
    })
    ev = tk.panel_inversion_evidence()
    assert ev["tech"]["excess"]["1d"] == pytest.approx(-0.90)
    assert ev["tech"]["qualifies"]


def test_a_method_BETTER_than_the_system_is_never_inverted(monkeypatch):
    """Both negative in absolute terms, but the method is the better one."""
    _panel(monkeypatch, {
        "combined_score": {"1d": -1.50, "3d": -2.00},
        "insider":        {"1d": -0.50, "3d": -0.60},   # less bad = positive excess
    })
    ev = tk.panel_inversion_evidence()
    assert ev["insider"]["excess"]["1d"] > 0
    assert not ev["insider"]["qualifies"]


# ── "stays negative across horizons" ───────────────────────────────────────

def test_must_be_negative_at_EVERY_horizon(monkeypatch):
    """The documented criterion. A sign flip between horizons is noise, not a
    reliably backwards signal."""
    _panel(monkeypatch, {
        "combined_score": {"1d": 0.0, "3d": 0.0},
        "flip":           {"1d": -0.90, "3d": 0.90},
    })
    assert not tk.panel_inversion_evidence().get("flip", {}).get("qualifies", False)


# ── confidence ─────────────────────────────────────────────────────────────

def test_thin_evidence_does_not_qualify(monkeypatch):
    """t = |excess| · sqrt(signal-days); few days ⇒ no confidence."""
    _panel(monkeypatch, {
        "combined_score": {"1d": 0.0, "3d": 0.0},
        "tech":           {"1d": -0.30, "3d": -0.30},
    }, days=4)
    ev = tk.panel_inversion_evidence()
    assert ev["tech"]["t"]["1d"] < 2.0
    assert not ev["tech"]["qualifies"]


def test_the_bar_does_not_depend_on_how_many_methods_are_judged(monkeypatch):
    """No multiple-comparison bump (2026-07-26). The methods are ~0.72
    correlated, so a threshold scaled by the test COUNT punishes a method for
    the company it keeps rather than for its own evidence. Replication across
    the two arms is the safeguard instead."""
    def bar(extra):
        rows = {"combined_score": {"1d": 0.0, "3d": 0.0},
                "tech": {"1d": -0.25, "3d": -0.25}}
        rows.update({m: {"1d": 0.0, "3d": 0.0} for m in extra})
        _panel(monkeypatch, rows, days=100)
        return tk.panel_inversion_evidence()["tech"]["min_t"]

    assert bar([]) == bar(["momentum", "vwap", "news", "pattern", "iv_rank"])
    assert bar([]) == pytest.approx(settings.inversion_panel_min_t)


# ── plumbing ───────────────────────────────────────────────────────────────

def test_missing_combined_score_row_disables_the_arm(monkeypatch):
    """Without the yardstick there is no control, so the arm must abstain
    rather than fall back to testing against zero."""
    _panel(monkeypatch, {"tech": {"1d": -0.90, "3d": -1.20}})
    assert tk.panel_inversion_evidence() == {}


def test_disabled_arm_returns_nothing(monkeypatch):
    monkeypatch.setattr(settings, "enable_inversion_panel_arm", False)
    _panel(monkeypatch, {"combined_score": {"1d": 0.0}, "tech": {"1d": -9.0}})
    assert tk.panel_inversion_evidence() == {}


def test_panel_failure_is_fail_soft(monkeypatch):
    def boom(**kw): raise RuntimeError("panel unavailable")
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf", boom)
    tk._PANEL_INVERSION_CACHE.clear()
    assert tk.panel_inversion_evidence() == {}


def test_only_combine_methods_are_candidates(monkeypatch):
    _panel(monkeypatch, {
        "combined_score": {"1d": 0.0, "3d": 0.0},
        "not_a_method":   {"1d": -9.0, "3d": -9.0},
    })
    assert "not_a_method" not in tk.panel_inversion_evidence()
