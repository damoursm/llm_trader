"""Tier 2 backtest — and the firewall that keeps it out of calibration.

The computation is the easy half. The property that actually matters is
NEGATIVE: a backtested `combined_score`/`confidence` must never reach a
calibration, because today's weights are fitted FROM the signals panel, so
fitting them on values derived from those same weights is self-confirmation
dressed as evidence — and it would look like the configuration improving.

So the tests fall in three groups:
  1. the extracted combine is behaviour-identical to the live path (a duplicate
     would silently drift — the reason it was extracted at all);
  2. the derived layer is computed correctly and stamped with the weight set;
  3. the FIREWALL: no calibration path can read it, by import or by table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import backtest as bt


# ── 1. the extracted combine matches the live path ────────────────────────────

def test_extracted_combine_matches_a_hand_computed_average():
    """buy = weighted mean over the BULLISH camp only; sell likewise. The camps
    must not dilute each other — that was the whole point of the 2026-07-22
    split, and a regression here silently rebalances every signal."""
    from src.signals.aggregator import combine_buy_sell

    msm = {"a": (True, 0.8), "b": (True, 0.4), "c": (True, -0.6)}
    w = {"a": 0.5, "b": 0.5, "c": 1.0}
    buy, sell = combine_buy_sell(msm, w)

    assert buy == pytest.approx((0.5 * 0.8 + 0.5 * 0.4) / 1.0)
    assert sell == pytest.approx((1.0 * 0.6) / 1.0)


def test_inverted_weight_flips_the_camp():
    """A negative weight IS the inversion, so a bullish raw score must land in
    the BEARISH camp."""
    from src.signals.aggregator import combine_buy_sell

    buy, sell = combine_buy_sell({"a": (True, 0.8)}, {"a": -0.5})
    assert buy == 0.0 and sell == pytest.approx(0.8)


def test_side_filter_removes_a_method_from_one_camp_only():
    from src.signals.aggregator import combine_buy_sell

    msm = {"a": (True, 0.8), "b": (True, -0.8)}
    w = {"a": 1.0, "b": 1.0}
    buy, sell = combine_buy_sell(msm, w, buy_filtered={"a"})
    assert buy == 0.0, "filtered from the bullish camp"
    assert sell == pytest.approx(0.8), "its bearish counterpart is untouched"


def test_abstainers_do_not_dilute_a_camp():
    from src.signals.aggregator import combine_buy_sell

    lone = combine_buy_sell({"a": (True, 0.8)}, {"a": 1.0})
    with_abstainer = combine_buy_sell(
        {"a": (True, 0.8), "b": (True, 0.0)}, {"a": 1.0, "b": 5.0})
    assert lone == with_abstainer


# ── 2. the derived layer ──────────────────────────────────────────────────────

def test_backtest_row_produces_the_full_derived_layer():
    w = {"m1": 0.5, "m2": 0.5}
    out = bt.backtest_row({"m1": 0.9, "m2": 0.7}, w,
                          movement_factor=1.1, tape_score=0.5, vol_ratio=1.4)
    for k in ("combined_score", "raw_confidence", "coherence_factor",
              "confidence", "direction"):
        assert k in out
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["direction"] == "BULLISH"


def test_direction_uses_the_configured_band():
    from config.settings import settings
    thr = float(settings.buy_sell_diff_threshold)
    w = {"m1": 1.0}
    assert bt.backtest_row({"m1": thr + 0.05}, w)["direction"] == "BULLISH"
    assert bt.backtest_row({"m1": -(thr + 0.05)}, w)["direction"] == "BEARISH"
    assert bt.backtest_row({"m1": thr / 2}, w)["direction"] == "NEUTRAL"


def test_missing_context_degrades_without_inventing_a_factor():
    """A component whose OHLCV input is absent must be OMITTED, not defaulted to
    1.0 and reported as though it had been measured."""
    out = bt.backtest_row({"m1": 0.9}, {"m1": 1.0})
    assert "movement_factor" not in out
    assert "volume_factor" not in out
    assert "tape_conf_factor" not in out
    assert "confidence" in out


def test_empty_or_unweighted_scores_yield_nothing():
    assert bt.backtest_row({}, {"m1": 1.0}) == {}
    assert bt.backtest_row({"unknown": 0.9}, {"m1": 1.0}) == {}


def test_weight_set_hash_is_stable_and_sensitive():
    """Rows under different weights are not comparable, so the stamp must change
    when the weights do — otherwise a backtest silently claims to be something
    it is not."""
    import src.signals.aggregator as agg
    h1 = bt.weight_set_hash()
    assert h1 == bt.weight_set_hash(), "hash must be stable within a config"

    orig = dict(agg._BASE_WEIGHTS)
    try:
        k = next(iter(agg._BASE_WEIGHTS))
        agg._BASE_WEIGHTS[k] = orig[k] + 0.123
        assert bt.weight_set_hash() != h1, "hash ignored a weight change"
    finally:
        agg._BASE_WEIGHTS.clear()
        agg._BASE_WEIGHTS.update(orig)


# ── 3. the firewall ───────────────────────────────────────────────────────────

_CALIBRATION_MODULES = (
    "src/analysis/signal_panel.py",
    "src/analysis/replay.py",
    "src/performance/tracker.py",
    "src/performance/edge_sizing.py",
    "src/performance/predictability_sizing.py",
    "src/performance/confidence_sizing.py",
    "src/analysis/threshold_calibration.py",
    "src/analysis/horizon_edge.py",
    "src/analysis/market_relative.py",
    "src/signals/aggregator.py",
)


def test_no_calibration_module_imports_the_backtest():
    """The structural firewall. A backtested value reaching a weight fit would
    fit the weights on values derived from themselves."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for rel in _CALIBRATION_MODULES:
        p = root / rel
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        if "analysis.backtest" in src or "from src.analysis import backtest" in src:
            offenders.append(rel)
    assert not offenders, (
        "these calibration/live modules import the tier-2 BACKTEST, which would "
        f"fit weights on values derived from those same weights: {offenders}")


def test_no_calibration_module_reads_the_backtest_table():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = [rel for rel in _CALIBRATION_MODULES
                 if (root / rel).exists()
                 and "signals_backtest" in (root / rel).read_text(encoding="utf-8")]
    assert not offenders, f"calibration modules querying signals_backtest: {offenders}"


def test_panel_restore_reads_only_the_replay_table():
    """`build_panel` must pull from `signals_replay` (tier 1 recoveries) and
    never from `signals_backtest` — that separation IS the firewall."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/analysis/replay.py").read_text(encoding="utf-8")
    assert "signals_replay" in src
    assert "signals_backtest" not in src
