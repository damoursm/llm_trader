"""Market-relative method weighting (2026-07-27).

An ABSOLUTE win rate cannot separate "this signal works" from "the market went
up". Measured on 296k ticker-days, a low-volatility screen won 53.5% at 5 days
and only 46.1% net of SPY — the entire apparent edge was beta.

Two things are pinned here, and the second is the one that is easy to get wrong:

  * WEIGHTING uses the market-relative basis, and falls back to the absolute
    ledger basis whenever the panel is unavailable — so the change can only ever
    add information, never remove it.
  * The neutral point is the MEASURED baseline (~48.6%), NOT 50%. The
    cap-weighted index beats its typical constituent, so the median stock is
    market-relative-negative. A half-migrated basis — relative numerator against
    a 50% bar — is worse than either pure basis, because it reads every ordinary
    method as weak.

Scope is deliberate: sizing, the edge blend and all P&L stay ABSOLUTE. The book
is outright long/short, so alpha it cannot capture must not size it.
"""

import pytest

from config.settings import settings
import src.analysis.market_relative as mrel
import src.signals.aggregator as agg


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "enable_market_relative_weighting", True)
    monkeypatch.setattr(settings, "enable_side_adaptive_weights", True)
    monkeypatch.setattr(settings, "side_weight_prior_n", 10)
    monkeypatch.setattr(settings, "side_weight_min_multiplier", 0.5)
    monkeypatch.setattr(settings, "side_weight_max_multiplier", 2.0)
    mrel.reset_cache()
    agg.reset_winrate_filter_cache()
    yield
    mrel.reset_cache()
    agg.reset_winrate_filter_cache()


def _skill(monkeypatch, table, baseline=48.6):
    """table: {method: (win_pct, n)}"""
    data = {m: {"win_rate": w, "trades": n, "baseline": baseline}
            for m, (w, n) in table.items()}
    monkeypatch.setattr(agg, "_side_skill", lambda: {"buy": data, "sell": data})


# ── the baseline is measured, not 50% ──────────────────────────────────────

def test_the_neutral_point_is_the_BASELINE_not_fifty(monkeypatch):
    """A method exactly AT the market-relative baseline must get a 1.0x
    multiplier. Centring on 0.5 instead would penalise it for being ordinary —
    on this basis the median stock is below 50% by construction."""
    _skill(monkeypatch, {"tech": (48.6, 5000)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["tech"] == pytest.approx(1.0, abs=0.01)


def test_a_method_at_fifty_percent_is_ABOVE_baseline(monkeypatch):
    """50% is a genuinely good market-relative result — it means beating the
    index on half your calls when the median stock does not."""
    _skill(monkeypatch, {"tech": (50.0, 5000)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["tech"] > 1.0


def test_below_baseline_is_downweighted(monkeypatch):
    _skill(monkeypatch, {"tech": (44.0, 5000)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["tech"] < 1.0


def test_baseline_shift_changes_the_verdict(monkeypatch):
    """The same 49% method is weak against a 50% bar and strong against 48.6% —
    which is precisely why the bar must be measured, not assumed."""
    _skill(monkeypatch, {"tech": (49.0, 5000)}, baseline=50.0)
    against_fifty = agg.side_weight_multipliers("buy")["tech"]
    mrel.reset_cache(); agg.reset_winrate_filter_cache()
    _skill(monkeypatch, {"tech": (49.0, 5000)}, baseline=48.6)
    against_measured = agg.side_weight_multipliers("buy")["tech"]
    assert against_fifty < 1.0 < against_measured


# ── shrinkage and clamps still hold ────────────────────────────────────────

def test_thin_evidence_shrinks_toward_neutral(monkeypatch):
    _skill(monkeypatch, {"tech": (80.0, 2)}, baseline=48.6)
    m = agg.side_weight_multipliers("buy")["tech"]
    assert 1.0 < m < 1.5, "2 observations must not earn a large tilt"


def test_multiplier_is_clamped(monkeypatch):
    _skill(monkeypatch, {"tech": (100.0, 100000)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["tech"] <= 2.0
    mrel.reset_cache()
    _skill(monkeypatch, {"tech": (0.0, 100000)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["tech"] >= 0.5


def test_a_method_with_no_record_is_neutral(monkeypatch):
    _skill(monkeypatch, {"tech": (60.0, 500)}, baseline=48.6)
    assert agg.side_weight_multipliers("buy")["vwap"] == pytest.approx(1.0, abs=0.01)


# ── fallback: never worse than the absolute basis ──────────────────────────

def test_disabled_returns_nothing_and_weighting_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "enable_market_relative_weighting", False)
    assert mrel.market_relative_skill("buy") == {}


def test_panel_failure_is_fail_soft(monkeypatch):
    def boom(**kw):
        raise RuntimeError("panel unavailable")
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf", boom)
    mrel.reset_cache()
    assert mrel.market_relative_skill("buy") == {}
    assert mrel.market_relative_baseline() == pytest.approx(48.6)


def test_baseline_is_observation_weighted(monkeypatch):
    """The bar is the panel's OWN up-share, so a method with 10x the
    observations must count 10x toward it — not one vote per method."""
    import pandas as pd
    df = pd.DataFrame([
        {"method": "a", "side": "both", "hit_1w": 40.0, "n_1w": 9000},
        {"method": "b", "side": "both", "hit_1w": 90.0, "n_1w": 1000},
    ])
    assert mrel._measure_baseline(df, "1w") == pytest.approx(45.0)


# ── the HARD filter: promotion logic (IC significance), 2026-08-12 rebasis ───

def _dirpanel(monkeypatch, rows):
    """Install a fake directional panel. ``rows`` = {method: {horizon: (icir,
    icdays, n)}}; every method lands on the ``both`` side the filter reads."""
    import pandas as pd
    recs = []
    for m, hs in rows.items():
        r = {"method": m, "side": "both"}
        for h, (icir, days, n) in hs.items():
            r[f"icir_{h}"], r[f"icdays_{h}"], r[f"n_{h}"] = icir, days, n
        recs.append(r)
    df = pd.DataFrame(recs)
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf",
                        lambda **kw: df)
    agg.reset_winrate_filter_cache()


@pytest.fixture
def _filter_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_market_relative_filter", True)
    monkeypatch.setattr(settings, "ic_weight_min_t", 2.0)
    monkeypatch.setattr(settings, "market_relative_min_obs", 200)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)


def test_significantly_negative_ic_is_filtered(monkeypatch, _filter_on):
    """ICIR −0.30 over 100 days → t = 3.0: confidently anti-predictive at every
    judgeable horizon → dropped. A healthy method is untouched."""
    _dirpanel(monkeypatch, {
        "tech": {"1w": (-0.30, 100, 5000)},
        "vwap": {"1w": (+0.20, 100, 5000)},
    })
    out = agg._market_relative_filtered()
    assert "tech" in out and "vwap" not in out


def test_insignificant_negative_ic_is_KEPT(monkeypatch, _filter_on):
    """THE POINT of the rebasis: below-par on the live window is not evidence.
    ICIR −0.05 over 60 days → t ≈ 0.39 — the old point-estimate rule would
    have dropped a method reading like this; promotion logic keeps it."""
    _dirpanel(monkeypatch, {"hi52": {"1w": (-0.05, 60, 5000)}})
    assert agg._market_relative_filtered() == frozenset()


def test_one_healthy_horizon_saves_the_method(monkeypatch, _filter_on):
    """DISPROVEN requires significantly negative at EVERY judgeable horizon
    (the method_horizons rule). Bad at 1d but fine at 1w → kept."""
    _dirpanel(monkeypatch, {
        "momentum": {"1d": (-0.40, 100, 5000), "1w": (+0.05, 100, 5000)},
    })
    assert agg._market_relative_filtered() == frozenset()


def test_thin_evidence_is_exempt(monkeypatch, _filter_on):
    """Unproven is not disproven — rows below the observation floor (or a
    missing ICIR) leave a horizon unjudgeable; no judgeable horizon → kept at
    full weight, the promotion posture for a fresh method."""
    _dirpanel(monkeypatch, {"pead": {"1w": (-0.90, 30, 50)}})       # n < 200
    assert agg._market_relative_filtered() == frozenset()


def test_inverted_methods_are_exempt(monkeypatch, _filter_on):
    """Their sign is already corrected — a confidently negative RAW IC is the
    reason they are kept, and the inversion machinery owns that verdict."""
    monkeypatch.setattr(settings, "inverted_methods", "tech")
    _dirpanel(monkeypatch, {"tech": {"1w": (-0.50, 100, 9000)}})
    assert "tech" not in agg._market_relative_filtered()


def test_disabled_filter_drops_nothing(monkeypatch, _filter_on):
    monkeypatch.setattr(settings, "enable_market_relative_filter", False)
    _dirpanel(monkeypatch, {"tech": {"1w": (-0.90, 200, 9000)}})
    assert agg._market_relative_filtered() == frozenset()


def test_filter_failure_is_fail_soft(monkeypatch, _filter_on):
    def boom(**kw): raise RuntimeError("panel gone")
    monkeypatch.setattr("src.analysis.simulated_trades.compute_directional_perf", boom)
    agg.reset_winrate_filter_cache()
    assert agg._market_relative_filtered() == frozenset()
