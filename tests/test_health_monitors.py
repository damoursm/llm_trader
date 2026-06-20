"""Monitoring for three silent-default sites that feed decisions but used to fail
invisibly: FX sizing-rate fallback, correlation-pair compute failures, and
macro-regime/market-mode input coverage (+ the regime fail-CAUTIOUS guard).

Each exposes a per-run health snapshot the pipeline surfaces through
``_collect_sources`` → run_sources (Data Quality tab) + the email/dashboard banner.
"""

from types import SimpleNamespace as NS

import src.broker.fx as fx
import src.performance.correlation as corr
import src.data.macro_regime as mr
import src.data.market_mode as mm


# ── FX sizing rate ────────────────────────────────────────────────────────────
def test_fx_usd_is_not_an_fx_event(monkeypatch):
    fx.reset_fx_health()
    assert fx.usd_per_unit("USD") == 1.0
    h = fx.get_fx_health()
    assert h["live"] == 0 and h["fallback"] == 0 and h["assumed_one"] == 0


def test_fx_live_rate_recorded(monkeypatch):
    fx.reset_fx_health()
    monkeypatch.setattr(fx, "_live_rate", lambda pair, day: 0.73)
    assert fx.usd_per_unit("CAD") == 0.73
    assert fx.get_fx_health()["live"] == 1


def test_fx_fallback_recorded(monkeypatch):
    fx.reset_fx_health()
    monkeypatch.setattr(fx, "_live_rate", lambda pair, day: None)   # live quote down
    monkeypatch.setattr(fx.settings, "broker_fx_fallback_cad_usd", 0.71)
    assert fx.usd_per_unit("CAD") == 0.71
    h = fx.get_fx_health()
    assert h["fallback"] == 1 and h["live"] == 0


def test_fx_unknown_ccy_assumes_one(monkeypatch):
    fx.reset_fx_health()
    monkeypatch.setattr(fx, "_live_rate", lambda pair, day: None)
    assert fx.usd_per_unit("EUR") == 1.0
    assert fx.get_fx_health()["assumed_one"] == 1


# ── Correlation sizing ────────────────────────────────────────────────────────
def test_correlation_counts_failed_pairs(monkeypatch):
    corr.reset_correlation_health()
    # No OHLCV cache for these → every pair fails to compute.
    corr.mean_pairwise_correlation("FAKE1", ["FAKE2", "FAKE3"])
    h = corr.get_correlation_health()
    assert h["attempted"] == 2 and h["failed"] == 2


def test_correlation_counts_successes(monkeypatch):
    corr.reset_correlation_health()
    monkeypatch.setattr(corr, "pairwise_correlation", lambda a, b, days=None: 0.4)
    corr.mean_pairwise_correlation("AAA", ["BBB", "CCC", "AAA"])  # self-pair skipped
    h = corr.get_correlation_health()
    assert h["attempted"] == 2 and h["failed"] == 0


# ── Macro regime coverage + fail-CAUTIOUS guard ───────────────────────────────
def test_regime_low_coverage_forces_caution(monkeypatch):
    monkeypatch.setattr(mr.settings, "macro_regime_min_inputs", 3)
    r = mr.compute_macro_regime(vix_context=NS(vix_signal="NORMAL"))  # 1 input, neutral
    assert r.regime == "CAUTION"                 # would be NEUTRAL; forced up
    assert r.inputs_available == 1 and r.inputs_total == 10
    assert mr.get_regime_coverage() == {"available": 1, "total": 10}


def test_regime_full_coverage_stays_neutral(monkeypatch):
    monkeypatch.setattr(mr.settings, "macro_regime_min_inputs", 3)
    r = mr.compute_macro_regime(
        vix_context=NS(vix_signal="NORMAL"),
        breadth_context=NS(signal="BREADTH_MIXED"),
        credit_context=NS(signal="NEUTRAL"),
    )
    assert r.regime == "NEUTRAL" and r.inputs_available == 3


def test_regime_protective_verdict_never_loosened(monkeypatch):
    monkeypatch.setattr(mr.settings, "macro_regime_min_inputs", 3)
    # A lone PANIC signal must still produce PANIC despite low coverage — the
    # guard only blocks fail-OPEN (permissive), never fail-closed (protective).
    r = mr.compute_macro_regime(vix_context=NS(vix_signal="PANIC"))
    assert r.regime == "PANIC"


def test_market_mode_coverage_fields():
    m = mm.compute_market_mode(vix_context=NS(vix_signal="LOW"))
    assert m.inputs_available == 1 and m.inputs_total == 4
    assert mm.get_mode_coverage() == {"available": 1, "total": 4}


# ── Pipeline source-health surfacing ──────────────────────────────────────────
def test_collect_sources_flags_unhealthy(monkeypatch):
    import src.pipeline as pl
    fx._FX_HEALTH.update(live=2, fallback=3, assumed_one=0)
    corr._CORR_HEALTH.update(attempted=10, failed=7)
    mr._LAST_COVERAGE.update(available=1, total=10)
    mm._LAST_COVERAGE.update(available=0, total=4)
    srcs = {s["label"]: s for s in pl._collect_sources()}
    for prefix in ("FX rate", "Correlation sizing", "Macro regime inputs", "Market mode inputs"):
        match = next(s for lbl, s in srcs.items() if lbl.startswith(prefix))
        assert match["ok"] is False and match["error"]


def test_collect_sources_silent_when_not_exercised(monkeypatch):
    # No FX/correlation activity and regime/mode not computed this run → no entries
    # (total=0 marks "not computed"), so a no-broker / disabled-regime run is clean.
    import src.pipeline as pl
    fx.reset_fx_health()
    corr.reset_correlation_health()
    mr.reset_regime_coverage()
    mm.reset_mode_coverage()
    labels = {s["label"] for s in pl._collect_sources()}
    assert not any(l.startswith(("FX rate", "Correlation sizing", "Macro regime", "Market mode"))
                   for l in labels)
