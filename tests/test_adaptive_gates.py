"""Adaptive gates + learned sizing (2026-07-03): the engine-relative actionable
threshold, the calibrated exit-confidence floor, and the unified expected-edge
sizing blend. All three follow the house pattern: documented prior, evidence
from the system's own tables, Bayesian shrinkage, hard clamps, fail-soft to
the static value.
"""

import pandas as pd
import pytest

from config.settings import settings


# ── #2: engine-relative actionable threshold ──────────────────────────────────

def _rec_df(rows):
    return pd.DataFrame(rows, columns=["llm_provider", "confidence"])


def _patch_recs(monkeypatch, rows):
    import src.analysis.threshold_calibration as tc
    from src.db import repo
    df = _rec_df(rows)
    monkeypatch.setattr(repo, "fetch_df", lambda sql, params=None: df)
    tc.reset_cache()
    return tc


def test_threshold_translates_into_inflated_engine_scale(monkeypatch):
    monkeypatch.setattr(settings, "threshold_engine_relative_enabled", True)
    monkeypatch.setattr(settings, "threshold_min_global_recs", 100)
    monkeypatch.setattr(settings, "threshold_engine_prior_n", 0)     # pure translation
    monkeypatch.setattr(settings, "threshold_max_shift", 0.20)
    # Global book: engine A (calibrated, conf 0.70–0.89) + engine B (inflated,
    # conf 0.85–1.00). A 0.78 static threshold rejects a chunk of A's calls;
    # for B the same SELECTIVITY lands materially higher than 0.78.
    rows = ([("A", 0.70 + 0.001 * i) for i in range(190)]
            + [("B", 0.85 + 0.00075 * i) for i in range(190)])
    tc = _patch_recs(monkeypatch, rows)
    eff_b, meta = tc.engine_relative_threshold(0.78, "B")
    assert meta["applied"] and eff_b > 0.85              # inflated engine → higher bar
    eff_a, _ = tc.engine_relative_threshold(0.78, "A")
    assert eff_a < eff_b                                  # calibrated engine keeps a lower bar


def test_threshold_shrinks_and_clamps(monkeypatch):
    monkeypatch.setattr(settings, "threshold_engine_relative_enabled", True)
    monkeypatch.setattr(settings, "threshold_min_global_recs", 100)
    monkeypatch.setattr(settings, "threshold_max_shift", 0.03)       # tight clamp
    monkeypatch.setattr(settings, "threshold_engine_prior_n", 150)
    rows = ([("A", 0.70 + 0.001 * i) for i in range(150)]
            + [("B", 0.95 + 0.0002 * i) for i in range(150)])        # wildly inflated
    tc = _patch_recs(monkeypatch, rows)
    eff, meta = tc.engine_relative_threshold(0.78, "B")
    assert meta["applied"]
    assert eff <= 0.78 + 0.03 + 1e-9                     # clamp: drifts, never jumps


def test_threshold_static_when_thin_or_disabled(monkeypatch):
    monkeypatch.setattr(settings, "threshold_engine_relative_enabled", True)
    monkeypatch.setattr(settings, "threshold_min_global_recs", 300)
    tc = _patch_recs(monkeypatch, [("A", 0.8)] * 50)     # far below the min sample
    eff, meta = tc.engine_relative_threshold(0.78, "A")
    assert eff == pytest.approx(0.78) and not meta["applied"]
    monkeypatch.setattr(settings, "threshold_engine_relative_enabled", False)
    eff, meta = tc.engine_relative_threshold(0.82, "A")
    assert eff == pytest.approx(0.82) and not meta["applied"]


# ── #3: calibrated exit-confidence floor ──────────────────────────────────────

def _patch_reviews(monkeypatch, pairs):
    """pairs = [(confidence, oriented_fwd_pct)] — patch the outcome join."""
    import src.analysis.exit_floor_calibration as ef
    monkeypatch.setattr(ef, "_review_outcomes", lambda days: pairs)
    ef.reset_cache()
    return ef


def test_exit_floor_moves_toward_observed_boundary(monkeypatch):
    monkeypatch.setattr(settings, "exit_floor_calibration_enabled", True)
    monkeypatch.setattr(settings, "signal_decay_confidence_floor", 0.45)
    monkeypatch.setattr(settings, "exit_floor_prior_n", 50)
    monkeypatch.setattr(settings, "exit_floor_min_side", 25)
    # Below 0.60 holding loses money, above it makes money → observed boundary 0.60.
    pairs = ([(0.40 + 0.004 * i, -1.0) for i in range(50)]      # conf 0.40–0.60 → bleed
             + [(0.62 + 0.003 * i, +1.0) for i in range(50)])   # conf 0.62–0.77 → gain
    ef = _patch_reviews(monkeypatch, pairs)
    floor = ef.calibrated_exit_floor()
    # The scan picks the LOWEST defensible boundary (deliberately conservative
    # toward holding): at 0.50 the below-floor mean is already ≤ 0 while the
    # above-floor mix is net-positive → obs = 0.50, and
    # shrink(0.45 prior_n=50, obs=0.50, n=100) = (22.5 + 50)/150 ≈ 0.483.
    assert floor == pytest.approx(0.4833, abs=0.01)
    assert floor > 0.45                                    # pulled UP toward the boundary
    assert settings.exit_floor_min <= floor <= settings.exit_floor_max


def test_exit_floor_static_when_no_separating_boundary(monkeypatch):
    monkeypatch.setattr(settings, "exit_floor_calibration_enabled", True)
    monkeypatch.setattr(settings, "signal_decay_confidence_floor", 0.45)
    # Profitable at every confidence → no boundary → prior holds exactly.
    ef = _patch_reviews(monkeypatch, [(0.3 + 0.005 * i, 1.0) for i in range(100)])
    assert ef.calibrated_exit_floor() == pytest.approx(0.45)
    # And fully disabled → static, no computation.
    monkeypatch.setattr(settings, "exit_floor_calibration_enabled", False)
    ef.reset_cache()
    assert ef.calibrated_exit_floor() == pytest.approx(0.45)


# ── #6: unified expected-edge sizing ─────────────────────────────────────────

def _mk_trade(breadth_n, ret, n_set=20, action="BUY", conf=0.8, status="CLOSED",
              news=0.5, momentum=0.5, combined=0.4, session="rth"):
    ms = {f"m{i}": (0.5 if i < breadth_n else 0.0) for i in range(n_set)}
    ms["news"] = news if breadth_n else 0.0
    ms["momentum"] = momentum if breadth_n else 0.0
    return {"ticker": "T", "action": action, "status": status, "return_pct": ret,
            "confidence": conf, "entry_session": session, "method_scores": ms,
            "signal_at_entry": {"combined_score": combined}}


def test_edge_model_learns_breadth_return_relation(monkeypatch):
    from src.performance import edge_sizing as es
    monkeypatch.setattr(settings, "edge_sizing_enabled", True)
    monkeypatch.setattr(settings, "edge_min_closed", 20)
    monkeypatch.setattr(settings, "edge_prior_n", 30)
    monkeypatch.setattr(settings, "edge_size_span", 0.25)
    # Broad trades won (+3%), narrow lost (−3%): the model must size the broad
    # profile ABOVE the narrow one.
    trades = ([_mk_trade(16, +3.0) for _ in range(30)]
              + [_mk_trade(4, -3.0) for _ in range(30)])
    model = es.fit_edge_model(trades)
    assert model is not None and model["n"] == 60
    broad = es.trade_features(_mk_trade(16, 0.0, status="OPEN"))
    narrow = es.trade_features(_mk_trade(4, 0.0, status="OPEN"))
    r_broad, m1 = es.edge_blend_ratio(broad, 1.0, model)
    r_narrow, m2 = es.edge_blend_ratio(narrow, 1.0, model)
    assert r_broad > 1.0 > r_narrow
    assert m1["w"] == pytest.approx(60 / 90, abs=1e-4)
    assert 0.6 <= r_narrow and r_broad <= 1.6              # hard ratio clamp


def test_edge_layer_is_inert_below_min_closed(monkeypatch):
    from src.performance import edge_sizing as es
    monkeypatch.setattr(settings, "edge_sizing_enabled", True)
    monkeypatch.setattr(settings, "edge_min_closed", 20)
    trades = [_mk_trade(16, 3.0) for _ in range(10)]       # only 10 closes
    assert es.fit_edge_model(trades) is None
    ratio, meta = es.edge_blend_ratio([0.8, 0.8, 0.4, 0.5, 0.5, 0.0], 1.2, None)
    assert ratio == pytest.approx(1.0) and meta["w"] == 0.0


def test_edge_model_is_deterministic(monkeypatch):
    from src.performance import edge_sizing as es
    monkeypatch.setattr(settings, "edge_sizing_enabled", True)
    monkeypatch.setattr(settings, "edge_min_closed", 20)
    trades = ([_mk_trade(16, +3.0) for _ in range(15)]
              + [_mk_trade(4, -3.0) for _ in range(15)])
    m1, m2 = es.fit_edge_model(trades), es.fit_edge_model(trades)
    assert m1["beta"] == m2["beta"] and m1["target_std"] == m2["target_std"]


def test_record_new_trades_applies_edge_blend(monkeypatch):
    from src.performance import tracker
    from src.performance import edge_sizing as es
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(settings, "edge_sizing_enabled", True)
    monkeypatch.setattr(settings, "edge_min_closed", 20)
    monkeypatch.setattr(settings, "edge_prior_n", 30)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    # Pin breadth neutral so the assertion isolates the edge layer.
    monkeypatch.setattr(tracker, "_breadth_calibration",
                        lambda trades=None: {"center": 0.5, "half_width": 0.1,
                                             "span_eff": 0.0, "n_cal": 0, "n_closed": 0})
    # Ledger whose lesson is "broad+news wins": model sizes the new broad entry up.
    ledger = ([_mk_trade(16, +3.0) for _ in range(30)]
              + [_mk_trade(4, -3.0) for _ in range(30)])
    monkeypatch.setattr(tracker, "_load_trades", lambda: list(ledger))
    monkeypatch.setattr(tracker, "_save_trades", lambda t: ledger.__setitem__(slice(None), t))
    monkeypatch.setattr(tracker, "_method_scores_from_signal",
                        lambda tk, d, s: {**{f"m{i}": (0.5 if i < 16 else 0.0) for i in range(20)},
                                          "news": 0.5, "momentum": 0.5})
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from src.models import Recommendation
    rec = Recommendation(ticker="EDGY", type="STOCK", direction="BULLISH",
                         confidence=0.80, action="BUY", time_horizon="SWING",
                         rationale="t", generated_at=datetime.now(timezone.utc))
    sig = SimpleNamespace(direction="BULLISH", combined_score=0.4, confidence=0.9,
                          pattern_name="")
    diag = tracker.record_new_trades([rec], signals_by_ticker={"EDGY": sig}, run_id="e1")
    assert diag["opened"] == 1 and diag["edge_blend_applied"] == 1
    t = next(t for t in ledger if t.get("ticker") == "EDGY")
    assert t["edge_size_ratio"] > 1.0                      # broad profile sized up
    assert t["edge_model_weight"] == pytest.approx(60 / 90, abs=1e-4)
    assert t["position_size_multiplier"] > 1.0
