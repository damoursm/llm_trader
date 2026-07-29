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


# ── the HARD filter on the market-relative basis ───────────────────────────

def _rel(monkeypatch, table, baseline=48.1):
    data = {m: {"win_rate": w, "trades": n, "baseline": baseline}
            for m, (w, n) in table.items()}
    monkeypatch.setattr("src.analysis.market_relative.market_relative_skill",
                        lambda side=None: data)
    monkeypatch.setattr("src.analysis.market_relative.market_relative_baseline",
                        lambda: baseline)
    agg.reset_winrate_filter_cache()


@pytest.fixture
def _filter_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_market_relative_filter", True)
    monkeypatch.setattr(settings, "market_relative_filter_baseline", False)
    monkeypatch.setattr(settings, "winrate_filter_threshold", 0.50)
    monkeypatch.setattr(settings, "market_relative_min_obs", 200)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)


def test_below_fifty_is_filtered(monkeypatch, _filter_on):
    _rel(monkeypatch, {"tech": (46.6, 9000), "vwap": (51.7, 9000)})
    out = agg._market_relative_filtered()
    assert "tech" in out and "vwap" not in out


def test_thin_evidence_is_exempt(monkeypatch, _filter_on):
    """Unproven is not disproven — a method below the observation floor keeps
    full weight however bad the point estimate looks."""
    _rel(monkeypatch, {"pead": (20.0, 50)})
    assert agg._market_relative_filtered() == frozenset()


def test_inverted_methods_are_exempt(monkeypatch, _filter_on):
    """Their sign is already corrected, so a low RAW rate is the reason they're
    kept — the same exemption the absolute filter has always had."""
    monkeypatch.setattr(settings, "inverted_methods", "tech")
    _rel(monkeypatch, {"tech": (30.0, 9000)})
    assert "tech" not in agg._market_relative_filtered()


def test_baseline_mode_keeps_the_between_methods(monkeypatch, _filter_on):
    """The two bars differ only for methods sitting between them — ext_gap at
    49.7% and insider at 49.2% are below 50% but above the ~48.1% baseline."""
    _rel(monkeypatch, {"ext_gap": (49.7, 2275), "insider": (49.2, 7845)})
    assert agg._market_relative_filtered() == frozenset({"ext_gap", "insider"})
    monkeypatch.setattr(settings, "market_relative_filter_baseline", True)
    agg.reset_winrate_filter_cache()
    assert agg._market_relative_filtered() == frozenset()


def test_disabled_filter_drops_nothing(monkeypatch, _filter_on):
    monkeypatch.setattr(settings, "enable_market_relative_filter", False)
    _rel(monkeypatch, {"tech": (10.0, 9000)})
    assert agg._market_relative_filtered() == frozenset()


def test_filter_failure_is_fail_soft(monkeypatch, _filter_on):
    def boom(side=None): raise RuntimeError("panel gone")
    monkeypatch.setattr("src.analysis.market_relative.market_relative_skill", boom)
    agg.reset_winrate_filter_cache()
    assert agg._market_relative_filtered() == frozenset()


# ── the re-entrancy guard ──────────────────────────────────────────────────

def test_inverted_methods_is_reentrancy_safe(monkeypatch):
    """The auto-detector's bar 2 asks for the EFFECTIVE win rate, which calls
    back into `_inverted_methods` — so computing the inverted set depends on
    already knowing it. Unguarded that chain re-entered 22 times per filter
    evaluation and blew the recursion limit inside `simulated_trades`.

    While a computation is in flight the caller must see the MANUAL pins only:
    terminating, and semantically right — the detector cannot depend on its own
    output."""
    monkeypatch.setattr(settings, "enable_auto_inversion", True)
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    depth = {"n": 0, "max": 0}

    def recursive_detector():
        depth["n"] += 1
        depth["max"] = max(depth["max"], depth["n"])
        try:
            inner = agg._inverted_methods()          # the re-entrant call
            assert inner == frozenset({"insider"}), "in-flight must yield pins only"
            return {}
        finally:
            depth["n"] -= 1

    monkeypatch.setattr("src.performance.tracker.calibrate_method_inversion",
                        recursive_detector)
    agg._INVERSION_IN_FLIGHT["v"] = False
    out = agg._inverted_methods()
    assert out == frozenset({"insider"})
    assert depth["max"] == 1, "the detector must not be re-entered"


def test_guard_is_released_after_an_exception(monkeypatch):
    """A failing detector must not leave the guard stuck on, or every later
    call would silently degrade to manual pins forever."""
    monkeypatch.setattr(settings, "enable_auto_inversion", True)
    monkeypatch.setattr(settings, "inverted_methods", "")
    def boom():
        raise RuntimeError("detector broke")
    monkeypatch.setattr("src.performance.tracker.calibrate_method_inversion", boom)
    agg._INVERSION_IN_FLIGHT["v"] = False
    agg._inverted_methods()
    assert agg._INVERSION_IN_FLIGHT["v"] is False
