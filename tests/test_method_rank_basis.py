"""Method RANK basis (2026-08-13 user directive) — `aggregator._rank_transform_run`
+ the `method_score_basis` consumption contract.

The transform replaces each method's ACTIVE, non-zero scores with their centered
within-run rank at the COMBINE INPUT only; every persisted surface keeps the raw
score (the inversion architecture — no scorer epoch fires). These tests pin the
transform's math and its abstention/fail-soft edges; the confidence-scale shift
it causes is registered as CONFIDENCE_EPOCH 2026-08-14 (test below).
"""
from datetime import datetime, timezone

import pytest

from src.signals.aggregator import _rank_transform_run


def _maps(scores, active=None, method="m"):
    active = active or {}
    return {f"T{i}": {method: (active.get(f"T{i}", True), s)}
            for i, s in enumerate(scores)}


def test_centered_rank_extremes_and_median():
    out, abstained = _rank_transform_run(_maps([0.9, 0.3, 0.1, -0.2, -0.7]))
    assert abstained == frozenset()
    vals = {t: v for t, (on, v) in ((t, m["m"]) for t, m in out.items())}
    assert vals["T0"] == 1.0            # strongest -> +1
    assert vals["T4"] == -1.0           # weakest  -> -1
    assert vals["T2"] == 0.0            # median   ->  0
    # order preserved, symmetric grid
    assert vals["T1"] == 0.5 and vals["T3"] == -0.5


def test_rank_is_monotone_in_raw_score():
    raw = [0.05, 0.9, -0.4, 0.2, -0.1, 0.33, -0.02]
    out, _ = _rank_transform_run(_maps(raw))
    ranked = [out[f"T{i}"]["m"][1] for i in range(len(raw))]
    order_raw = sorted(range(len(raw)), key=lambda i: raw[i])
    order_rank = sorted(range(len(raw)), key=lambda i: ranked[i])
    assert order_raw == order_rank


def test_zero_scores_abstain_and_get_no_rank():
    out, _ = _rank_transform_run(_maps([0.9, 0.0, 0.1, -0.2, -0.7, 0.4]))
    assert out["T1"]["m"] == (True, 0.0)
    # the zero must not occupy a rank slot: 5 views -> extremes still ±1
    assert out["T0"]["m"][1] == 1.0 and out["T4"]["m"][1] == -1.0


def test_inactive_methods_untouched():
    out, abstained = _rank_transform_run(_maps([0.9, 0.3, 0.1, -0.2, -0.7],
                                               active={"T2": False}))
    assert out["T2"]["m"] == (False, 0.1)          # raw kept, still inactive
    # remaining 4 views < min_views(5) -> WEIGHT-0 abstention, raw scores kept
    assert "m" in abstained
    assert out["T0"]["m"] == (True, 0.9)


def test_below_min_views_abstains_with_weight_zero():
    """No absolute fallback and no faked zero score (2026-08-13 directive,
    clarified same day): a cross-section too thin to rank keeps its RAW scores
    in the map — truthful — and the method comes back in the ``abstained`` set,
    which the combine turns into WEIGHT 0 + exclusion from coherence /
    sources_agreeing / family votes (the win-rate-filter idiom)."""
    out, abstained = _rank_transform_run(_maps([0.9, 0.3, -0.2]))
    assert abstained == frozenset({"m"})
    for i, s in enumerate([0.9, 0.3, -0.2]):
        assert out[f"T{i}"]["m"] == (True, s)      # raw, not zeroed


def test_all_zero_views_not_flagged_abstained():
    """A method whose every score is 0 has nothing to weight either way — it
    must not be reported as a thin cross-section."""
    out, abstained = _rank_transform_run(_maps([0.0, 0.0, 0.0]))
    assert abstained == frozenset()


def test_ties_share_a_rank():
    out, _ = _rank_transform_run(_maps([0.5, 0.2, 0.2, 0.2, -0.3]))
    tied = {out[f"T{i}"]["m"][1] for i in (1, 2, 3)}
    assert len(tied) == 1                            # one shared average rank
    assert out["T0"]["m"][1] == 1.0 and out["T4"]["m"][1] == -1.0


def test_methods_ranked_independently():
    maps = {
        "A": {"x": (True, 0.9), "y": (True, -0.5)},
        "B": {"x": (True, 0.1), "y": (True, 0.8)},
        "C": {"x": (True, -0.4), "y": (True, 0.1)},
        "D": {"x": (True, 0.2), "y": (True, -0.9)},
        "E": {"x": (True, 0.5), "y": (True, 0.3)},
    }
    out, _ = _rank_transform_run(maps)
    assert out["A"]["x"][1] == 1.0 and out["A"]["y"][1] == -0.5
    assert out["D"]["y"][1] == -1.0


def test_input_maps_not_mutated():
    maps = _maps([0.9, 0.3, 0.1, -0.2, -0.7])
    before = {t: dict(m) for t, m in maps.items()}
    _rank_transform_run(maps)
    assert maps == before                            # pure function


def test_setting_default_is_rank_and_revert_exists():
    """The FIELD default is "rank" (the live directive); the conftest pins the
    RUNTIME value to "absolute" suite-wide so legacy fixtures stay meaningful —
    assert the default on the model field, not the pinned instance."""
    from config.settings import Settings, settings
    assert Settings.model_fields["method_score_basis"].default == "rank"
    assert int(settings.method_rank_min_views) >= 2


def test_confidence_epoch_registered_for_the_switch():
    """The rank basis changes |combined|'s scale — the switch must be epoch-
    registered so confidence calibrations exclude absolute-era rows."""
    from src.signals.method_epochs import CONFIDENCE_EPOCH
    assert CONFIDENCE_EPOCH >= datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)


def test_abs_shadow_columns_registered():
    """The absolute-basis shadow combine (2026-08-14): schema columns exist,
    auto-migrated, and the TickerSignal carries the fields."""
    from src.db.schema import _ADD_COLUMNS, SIGNAL_ABS_SHADOW_COLUMNS
    from src.models import TickerSignal
    assert SIGNAL_ABS_SHADOW_COLUMNS == (
        "combined_score_abs", "combined_buy_score_abs", "combined_sell_score_abs")
    migrated = {(t, c) for t, c, _ in _ADD_COLUMNS}
    for c in SIGNAL_ABS_SHADOW_COLUMNS:
        assert ("signals", c) in migrated
        assert c in TickerSignal.model_fields


def test_abs_shadow_is_actually_assigned_on_the_signal():
    """...and the aggregator POPULATES them.

    The plumbing test above passed for the whole 2026-08-14 commit while the
    aggregator computed `_abs_buy`/`_abs_sell` and then dropped them on the
    floor: schema, model fields, repo writer and pipeline reader were all
    correct, so the column simply stayed NULL on every row and nothing warned.
    That silently defeated the `ex_combine` basis-invariance guarantee too — the
    fallback to `combined_score` is unconditional when the twin is never
    written, which is the RANKED combine post-restart.

    Checked on the AST rather than by running `build_signals` (which needs the
    full feed stack): the three fields must appear as KEYWORDS on the
    TickerSignal construction, which a dead store cannot fake.
    """
    import ast
    import inspect
    from src.db.schema import SIGNAL_ABS_SHADOW_COLUMNS
    from src.signals import aggregator

    tree = ast.parse(inspect.getsource(aggregator))
    kwargs: set = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TickerSignal"):
            kwargs |= {kw.arg for kw in node.keywords if kw.arg}
    assert kwargs, "no TickerSignal(...) construction found in aggregator"
    missing = [c for c in SIGNAL_ABS_SHADOW_COLUMNS if c not in kwargs]
    assert not missing, f"shadow combine computed but never persisted: {missing}"


def test_raw_confidence_scale_follows_the_basis():
    """One resolver, both sites. The cross-sectional overlay re-derives
    raw_confidence on the adjusted combined score; when it kept a hard-coded
    /0.5 while the first pass moved to the rank scale, every overlay-touched
    row persisted components that no longer multiplied back to its confidence."""
    import ast
    import inspect
    from config.settings import settings
    from src.signals import aggregator
    from src.signals.aggregator import _raw_confidence_scale

    _basis = settings.method_score_basis
    try:
        settings.method_score_basis = "rank"
        assert _raw_confidence_scale() == pytest.approx(
            settings.rank_raw_confidence_scale)
        settings.method_score_basis = "absolute"
        assert _raw_confidence_scale() == 0.5
    finally:
        settings.method_score_basis = _basis

    # Neither site may hard-code the divisor again. A combined-derived numerator
    # divided by a NUMERIC LITERAL is the defect's signature; dividing by another
    # expression (the overlay's `abs(new_combined) / old_abs` ratio) is fine.
    src = inspect.getsource(aggregator)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        if "combined" not in ast.unparse(node.left):
            continue
        assert not (isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, (int, float))), (
            f"raw_confidence divisor must resolve through "
            f"_raw_confidence_scale(), found literal in: {ast.unparse(node)}")


def test_confidence_formula_has_one_implementation():
    """`confidence` = raw x the five factors, capped and 2dp-rounded — computed
    by ONE function, so the cross-sectional overlay cannot drift from the first
    pass again (it has twice: a proportional rescale exact only below the cap /
    rounding / 0.05 floor, and a stale 0.5 divisor)."""
    from src.signals.aggregator import _confidence_from

    # The formula itself.
    assert _confidence_from(0.8, 1.0, 1.0, 1.0, 1.0, 1.0) == 0.8
    assert _confidence_from(0.5, 1.2, 1.1, 1.0, 1.0, 1.0) == pytest.approx(0.66)
    assert _confidence_from(0.9, 1.3, 1.2, 1.1, 1.1, 1.1) == 1.0        # capped
    assert _confidence_from(0.4, 0.5, 1.0, 1.0, 1.0, 1.0) == 0.2

    # A raw base and neutral factors must reproduce the base exactly — the
    # property confidence_components.py relies on to isolate one factor.
    for raw in (0.0, 0.13, 0.5, 0.87, 1.0):
        assert _confidence_from(raw, 1.0, 1.0, 1.0, 1.0, 1.0) == round(raw, 2)

    # The sector multiplier defaults to neutral (the two passes that run before
    # sector alignment is known) and multiplies in like any other factor.
    assert _confidence_from(0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) == 0.8
    assert _confidence_from(0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75) == 0.6
    assert _confidence_from(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.10) == pytest.approx(0.55)


def test_cross_sectional_overlay_does_not_rescale_confidence():
    """The overlay must RE-DERIVE confidence from the adjusted score, never
    scale the already-rounded, already-capped `sig.confidence`. Pinned on the
    AST because the rescale reads as reasonable code and its output stays
    plausible — only the stored components stop multiplying back."""
    import ast
    import inspect
    from src.signals import aggregator

    tree = ast.parse(inspect.getsource(aggregator))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        operands = {ast.unparse(node.left), ast.unparse(node.right)}
        assert "sig.confidence" not in operands, (
            f"cross-sectional overlay is rescaling confidence again: "
            f"{ast.unparse(node)}")

    # All THREE confidence sites go through the shared helper: the score pass,
    # the cross-sectional overlay, and the sector-alignment pass.
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_confidence_from"]
    assert len(calls) == 3, (
        f"expected the score pass, cross-sectional overlay and sector pass to "
        f"be the only confidence sites, found {len(calls)}")


def test_sector_alignment_factor_is_a_persisted_component():
    """The sector multiplier (1.10 aligned / 0.75 contradicted) applied to
    `confidence` for its entire life while being stored nowhere, so a
    sector-adjusted row could never multiply back from its components and
    confidence_components.py silently attributed the haircut to whichever
    factor it was isolating. It is now a first-class component end to end."""
    import ast
    import inspect
    from src.analysis.confidence_components import VARIANTS
    from src.db.schema import _ADD_COLUMNS, SIGNAL_CONFIDENCE_COMPONENT_COLUMNS
    from src.models import TickerSignal
    from src.signals import aggregator
    from src.signals.method_epochs import CONFIDENCE_EPOCH_COLUMNS

    assert "sector_conf_factor" in SIGNAL_CONFIDENCE_COMPONENT_COLUMNS
    assert ("signals", "sector_conf_factor") in {(t, c) for t, c, _ in _ADD_COLUMNS}
    assert "sector_conf_factor" in TickerSignal.model_fields
    assert TickerSignal.model_fields["sector_conf_factor"].default == 1.0
    assert "sector_conf_factor" in CONFIDENCE_EPOCH_COLUMNS
    assert "sector_conf_factor" in {v[2] for v in VARIANTS}

    # The sector pass must STORE it, not just consume it — the dead-store class.
    tree = ast.parse(inspect.getsource(aggregator))
    stored = {kw.arg for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              for kw in node.keywords if kw.arg}
    keys = {k.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert "sector_conf_factor" in (stored | keys), (
        "sector factor is applied but never written back onto the signal")


def test_ml_combine_has_its_own_scale():
    """The ML stacker combine is ~5x smaller than the weighted one, so sharing
    the weighted band + confidence divisor left the arm at 4.2% directional
    rows (0% once the rank bands went live) and 0.2% past Gate 1 — an A/B
    comparing a near-empty book against a full one. Both knobs are now
    combine-aware, and ONLY for the full swap."""
    from src.signals.aggregator import _direction_bands, _raw_confidence_scale
    from config.settings import settings

    w_long, w_short = _direction_bands("weighted")
    m_long, m_short = _direction_bands("ml")
    assert (m_long, m_short) == (settings.ml_diff_threshold_long,
                                 settings.ml_diff_threshold_short)
    assert m_long < w_long and m_short < w_short      # the whole point

    assert _raw_confidence_scale("ml") == settings.ml_raw_confidence_scale
    assert _raw_confidence_scale("ml") < _raw_confidence_scale("weighted")

    # A partial swap mixes two scales in one difference -> weighted (conservative).
    for src in ("ml_buy", "ml_sell", "weighted", "", None):
        assert _direction_bands(src) == (w_long, w_short)
        assert _raw_confidence_scale(src) == _raw_confidence_scale("weighted")


def test_held_positions_prompt_ab_is_retired():
    """Adopted ON 2026-08-14: the flip is gone and the setting deleted (a
    pinned-to-1.0 knob reads as live and would fail the inert-config guard)."""
    import inspect
    from config.settings import Settings
    from src import pipeline

    assert "open_positions_prompt_share" not in Settings.model_fields
    src = inspect.getsource(pipeline)
    assert "open_positions_prompt_share" not in src
    # The prompt is now conditioned only on there being open positions.
    assert "hold_prompt_active = bool(open_position_summaries)" in src


# ── payoff-shaped rank mapping (2026-08-14) ─────────────────────────────────

def test_shape_score_identity_without_curve():
    from src.signals.rank_shaping import shape_score
    assert shape_score("nope", 0.0, {}) == -1.0
    assert shape_score("nope", 0.5, {}) == 0.0
    assert shape_score("nope", 1.0, {}) == 1.0


def test_shape_score_interpolates_and_can_flip_the_extreme():
    """The whole point: a method whose measured top decile is BEARISH maps its
    best rank to a NEGATIVE score (momentum-family n-shape)."""
    from src.signals.rank_shaping import shape_score
    curve = [-0.2, 0.1, 0.3, 0.5, 0.7, 1.0, 0.7, 0.3, -0.4, -1.0]
    assert shape_score("m", 0.95, {"m": curve}) == pytest.approx(-1.0)  # top -> bearish
    assert shape_score("m", 0.55, {"m": curve}) == pytest.approx(1.0)   # mid-upper peak
    v = shape_score("m", 0.60, {"m": curve})                # between mids: interp
    assert 0.7 < v < 1.0


def test_calibration_shapes_are_bounded_and_demeaned(monkeypatch):
    """Synthetic panel with a KNOWN n-shape: the calibrated curve must bound to
    [-1,1], flip the top decile negative, and survive the shrink+demean chain."""
    import numpy as np
    import pandas as pd

    from src.signals import rank_shaping as rs
    rng = np.random.default_rng(0)
    days, per_day = 20, 120
    rows = []
    for d in range(days):
        day = f"2026-07-{d % 28 + 1:02d}"
        scores = rng.uniform(-1, 1, per_day)
        pct = pd.Series(scores).rank(pct=True).to_numpy()
        # payoff rises with rank then CRASHES in the top decile
        payoff = np.where(pct > 0.9, -3.0, 3.0 * (pct - 0.4)) + rng.normal(0, 0.5, per_day)
        for i in range(per_day):
            rows.append({"signal_date": day, "ticker": f"T{i}", "tech": scores[i],
                         "fwd_ret_pivot": payoff[i]})
    panel = pd.DataFrame(rows)
    monkeypatch.setattr(rs, "_MIDS", rs._MIDS)  # no-op, keeps import explicit
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "build_panel", lambda **k: panel)
    monkeypatch.setattr(settings := __import__("config.settings", fromlist=["settings"]).settings,
                        "rank_shape_min_rows", 500)
    shapes = rs.calibrate_rank_shapes()
    assert "tech" in shapes
    curve = shapes["tech"]
    assert max(abs(v) for v in curve) == 1.0                # normalized
    assert curve[-1] < 0                                    # crash decile flipped
    assert curve[7] > 0                                     # rising mid retained


def test_transform_applies_shapes(monkeypatch):
    """_rank_transform_run maps percentiles through the curve when shaping is on."""
    import src.signals.aggregator as agg
    from src.signals import rank_shaping as rs
    curve = [0.0] * 9 + [1.0]
    monkeypatch.setattr(rs, "get_rank_shapes", lambda: {"m": curve})
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_rank_shaping", True)
    maps = {f"T{i}": {"m": (True, s)} for i, s in enumerate([0.9, 0.5, 0.3, -0.2, -0.6])}
    out, _ = agg._rank_transform_run(maps)
    assert out["T0"]["m"][1] == 1.0                        # top of the run -> curve end
    assert abs(out["T2"]["m"][1]) < 0.3                    # middle -> flat region


def test_shaping_off_keeps_linear_grid(monkeypatch):
    import src.signals.aggregator as agg
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_rank_shaping", False)
    maps = {f"T{i}": {"m": (True, s)} for i, s in enumerate([0.9, 0.5, 0.3, -0.2, -0.6])}
    out, _ = agg._rank_transform_run(maps)
    assert out["T0"]["m"][1] == 1.0 and out["T4"]["m"][1] == -1.0
    assert out["T2"]["m"][1] == 0.0


# ── go-live threshold quantile match (2026-08-14) ───────────────────────────

def test_rank_thresholds_defaults_are_the_calibrated_values():
    """2026-08-14 final calibration (tradeable-ranked + shaped combine):
    symmetric fallback 0.182 / conf scale 0.642, and the ADOPTED asymmetric
    bands — bullish ≈ the tradeable top 10%, bearish ≈ the bottom 5%."""
    from config.settings import Settings
    assert Settings.model_fields["rank_diff_threshold"].default == 0.182
    assert Settings.model_fields["rank_raw_confidence_scale"].default == 0.642
    assert Settings.model_fields["rank_diff_threshold_long"].default == 0.324
    assert Settings.model_fields["rank_diff_threshold_short"].default == 0.346


# ── tradeable-only rank pool (2026-08-14) ───────────────────────────────────

def test_tradeable_pool_ranks_only_tradeables_and_interpolates_the_rest(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_rank_shaping", False)
    maps = _maps([0.9, 0.5, 0.3, -0.2, -0.6, 0.4])   # T0..T5
    tradeable = {"T0", "T1", "T2", "T3", "T4"}        # T5 (0.4) observe-only
    out, abst = _rank_transform_run(maps, tradeable=tradeable)
    assert abst == frozenset()
    # tradeable grid over 5 names: 0.9->1.0, 0.5->0.5, 0.3->0.0, -0.2->-0.5, -0.6->-1.0
    assert out["T0"]["m"][1] == 1.0 and out["T4"]["m"][1] == -1.0
    assert out["T2"]["m"][1] == 0.0
    # T5=0.4 sits between 0.3 and 0.5 in the tradeable distribution ->
    # avg-rank 3.5 -> pct (3.5-1)/4 = 0.625 -> centered 0.25
    assert out["T5"]["m"][1] == pytest.approx(0.25)


def test_tradeable_pool_outsider_above_max_clamps_to_top(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_rank_shaping", False)
    maps = _maps([0.9, 0.5, 0.3, -0.2, -0.6, 2.0])
    out, _ = _rank_transform_run(maps, tradeable={"T0", "T1", "T2", "T3", "T4"})
    assert out["T5"]["m"][1] == 1.0                   # clamped to the pool top


def test_thinness_judged_on_the_tradeable_count(monkeypatch):
    """6 views but only 4 tradeable -> the method abstains (weight 0)."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_rank_shaping", False)
    maps = _maps([0.9, 0.5, 0.3, -0.2, -0.6, 0.4])
    out, abst = _rank_transform_run(maps, tradeable={"T0", "T1", "T2", "T3"})
    assert abst == frozenset({"m"})
    assert out["T0"]["m"] == (True, 0.9)              # raw kept, weight-0 idiom


# ── ex_combine basis-invariance (2026-08-14, item 5) ────────────────────────

def test_live_exit_features_prefer_absolute_combine():
    """ml_exit's ex_combine/ex_combine_delta standardize on the ABSOLUTE
    combine: the rank basis (or a shaped curve refreshing under its TTL) must
    never shift a trained exit model's feature scale mid-hold."""
    from types import SimpleNamespace

    from src.analysis.ml_exit_dataset import live_exit_features
    sig = SimpleNamespace(combined_score=0.80, combined_score_abs=0.20)
    trade = {"action": "BUY", "direction": "BULLISH", "ticker": "ZZTEST",
             "entry_price": 100.0, "current_price": 105.0,
             "entry_date": "2026-08-01",
             "signal_at_entry": {"combined_score": 0.70, "combined_score_abs": 0.10}}
    f = live_exit_features(trade, {}, sig)
    assert f is not None
    assert f["ex_combine"] == pytest.approx(0.20)              # abs, not 0.80
    assert f["ex_combine_delta"] == pytest.approx(0.20 - 0.10)  # abs - abs


def test_live_exit_features_fall_back_to_live_combine():
    """Pre-shadow rows (combined_score_abs None) fall back to combined_score —
    which IS absolute-basis on those rows, so the series stays one basis."""
    from types import SimpleNamespace

    from src.analysis.ml_exit_dataset import live_exit_features
    sig = SimpleNamespace(combined_score=0.44, combined_score_abs=None)
    trade = {"action": "SELL", "direction": "BEARISH", "ticker": "ZZTEST",
             "entry_price": 100.0, "current_price": 95.0,
             "entry_date": "2026-08-01",
             "signal_at_entry": {"combined_score": -0.30}}
    f = live_exit_features(trade, {}, sig)
    assert f is not None
    assert f["ex_combine"] == pytest.approx(-0.44)             # ds(-1) * 0.44
