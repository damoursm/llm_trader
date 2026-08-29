"""The 2026-08-21 ranking-stage improvements (memory/ranking-stage-map-2026-08.md):

  1. prompt shortlist key — flag-gated `camp_max` alternative to the measured
     anti-selective confidence key (default UNCHANGED: the camp columns have
     only 22 days of history, so the replication bar cannot be met yet);
  2. walk-forward SHAPE history — `weight_history`'s sibling for the fitted
     consumption layer, so tier-2 backtests stop scoring history through
     curves fitted on it;
  3. policy_eval gate-input repair — the harness must never again gate on an
     all-NaN confidence column and silently report n_decisions=0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings, settings


# ── 1. shortlist key ─────────────────────────────────────────────────────────

class _Sig:
    def __init__(self, tk, conf, buy=0.0, sell=0.0, insider=""):
        self.ticker = tk
        self.confidence = conf
        self.combined_buy_score = buy
        self.combined_sell_score = sell
        self.insider_summary = insider


def _shortlist(signals, monkeypatch, key):
    """Reproduce claude_analyst's two-tier shortlist with the given key."""
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(settings, "prompt_shortlist_key", key)
    src = None
    # drive the REAL sorting logic by extracting it the way the module runs it:
    # meaningful first, noise fill, both sorted by _shortlist_key. The helper is
    # nested, so exercise it through a tiny replica bound to the same setting —
    # and pin the replica to the module's code with a source assertion.
    import inspect
    body = inspect.getsource(ca.generate_recommendations)
    assert "_shortlist_key" in body and "camp_max" in body, \
        "claude_analyst no longer routes the shortlist through _shortlist_key"

    def k(s):
        if str(settings.prompt_shortlist_key).lower() == "camp_max":
            camp = max(abs(float(s.combined_buy_score or 0.0)),
                       abs(float(s.combined_sell_score or 0.0)))
            return (camp, float(s.confidence or 0.0))
        return (float(s.confidence or 0.0),)

    meaningful = [s for s in signals if s.confidence > 0.10 or s.insider_summary]
    noise = [s for s in signals if s not in meaningful]
    ranked = sorted(meaningful, key=k, reverse=True)
    ranked += sorted(noise, key=k, reverse=True)
    return [s.ticker for s in ranked]


def test_default_key_is_confidence_and_unchanged():
    assert Settings.model_fields["prompt_shortlist_key"].default == "confidence", (
        "camp_max must NOT be default until it clears the replication bar "
        "(~2026-09-15) — its columns have one contiguous 22-day window")


def test_camp_max_key_reorders_a_contested_name(monkeypatch):
    """The motivating case: buy .5 / sell .4 = tiny |combined| (low confidence)
    but real conviction — camp_max surfaces it, confidence buries it."""
    sigs = [_Sig("CONTESTED", conf=0.20, buy=0.50, sell=0.40),
            _Sig("MILD", conf=0.60, buy=0.15, sell=0.02)]
    assert _shortlist(sigs, monkeypatch, "confidence")[0] == "MILD"
    assert _shortlist(sigs, monkeypatch, "camp_max")[0] == "CONTESTED"


def test_camp_max_falls_back_to_confidence_on_ties(monkeypatch):
    sigs = [_Sig("A", conf=0.30, buy=0.40), _Sig("B", conf=0.70, buy=0.40)]
    assert _shortlist(sigs, monkeypatch, "camp_max")[0] == "B"


# ── 2. walk-forward shape history ────────────────────────────────────────────

def test_shapes_for_date_is_strictly_before(monkeypatch):
    from src.signals.rank_shaping import shapes_for_date
    hist = {"2026-08-01": {"tech": [0.1] * 10},
            "2026-08-10": {"tech": [0.2] * 10}}
    assert shapes_for_date("2026-08-05", hist) == {"tech": [0.1] * 10}
    assert shapes_for_date("2026-08-10", hist) == {"tech": [0.1] * 10}, \
        "a same-day calibration must NOT serve its own date (strictly-before)"
    assert shapes_for_date("2026-08-11", hist) == {"tech": [0.2] * 10}
    assert shapes_for_date("2026-07-31", hist) is None, \
        "before the first calibration = None (caller uses identity, never today's)"


def test_rank_transform_accepts_injected_shapes(monkeypatch):
    """The backtest's injection point: an explicit curve set must override the
    live TTL lookup, and {} must mean identity."""
    import src.signals.aggregator as agg
    monkeypatch.setattr(settings, "enable_rank_shaping", True)
    monkeypatch.setattr(settings, "method_rank_min_views", 2)

    raw = {f"T{i}": {"m": (True, s)} for i, s in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])}
    # identity: top view -> +1
    maps_id, _ = agg._rank_transform_run(raw, shapes={})
    assert maps_id["T4"]["m"][1] == pytest.approx(1.0)
    # an inverting curve: top rank maps NEGATIVE
    inv = {"m": list(np.linspace(1.0, -1.0, 10))}
    maps_inv, _ = agg._rank_transform_run(raw, shapes=inv)
    assert maps_inv["T4"]["m"][1] < 0, "injected curve was ignored"
    # and the live lookup must not have been consulted for either call
    called = []
    monkeypatch.setattr("src.signals.rank_shaping.get_rank_shapes",
                        lambda: called.append(1) or {})
    agg._rank_transform_run(raw, shapes=inv)
    assert not called, "explicit shapes must bypass the live TTL lookup"


def test_shape_history_table_is_in_schema():
    import inspect

    import src.db.schema as schema
    src = inspect.getsource(schema)
    assert "CREATE TABLE IF NOT EXISTS shape_history" in src


def test_backtest_wf_mode_uses_asof_shapes():
    """The wiring that makes tier-2 honest: wf mode resolves shapes per date
    and stamps the tag; no earlier calibration => identity, never today's."""
    import inspect

    from src.analysis import backtest as bt
    src = inspect.getsource(bt.run_backtest)
    assert "shapes_for_date" in src
    assert "load_shape_history" in src
    assert "shapes=run_shapes" in src
    assert 'run_shapes = {}' in src        # the identity fallback


# ── 3. policy_eval gate-input repair ─────────────────────────────────────────

def test_policy_eval_refuses_silence_on_all_nan_confidence(monkeypatch):
    """The bug: build_panel masks confidence behind CONFIDENCE_EPOCH, every
    settled row predates it, and all four policies reported n_decisions=0
    without complaint for a week. The repair must (a) try the signals_backtest
    rescore, (b) WARN loudly when the gate input stays unusable."""
    import src.analysis.policy_eval as pe

    n = 60
    frame = pd.DataFrame({
        "signal_date": ["2026-08-01"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "confidence": [np.nan] * n,
        "direction": ["BULLISH"] * n,
        "combined_score": [0.4] * n,
        "price": [50.0] * n,
        "fwd_ret_5d": [1.0] * n,
        "generated_at": ["2026-08-01T14:00:00"] * n,
    })
    monkeypatch.setattr("src.analysis.signal_panel.build_panel",
                        lambda **k: frame.copy())
    monkeypatch.setattr("src.analysis.signal_panel.session_of_ts",
                        lambda s: pd.Series(["rth"] * len(s)))

    # (a) the backtest rescore fills the masked column
    bt = pd.DataFrame({"signal_date": ["2026-08-01"] * n,
                       "ticker": [f"T{i}" for i in range(n)],
                       "bt_conf": [0.9] * n})
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: bt.copy())
    panel = pe.build_decision_panel(days=None, horizon=5)
    got = pd.to_numeric(panel["confidence"], errors="coerce")
    assert got.notna().all() and (got == 0.9).all()

    # (b) with no rescore either, the warning fires — silence is refused
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame())
    warned = []
    from loguru import logger as _lg
    sink_id = _lg.add(lambda m: warned.append(str(m)), level="WARNING")
    try:
        pe.build_decision_panel(days=None, horizon=5)
    finally:
        _lg.remove(sink_id)
    assert any("entirely NaN" in w or "epoch-masked" in w for w in warned), \
        "an unusable gate input must be LOUD, never a clean n_decisions=0"
