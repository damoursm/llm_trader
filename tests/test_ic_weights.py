"""IC-informed adaptive weights (``enable_ic_weights``, ON but confidence-gated).

Tilts each method's weight by its reliability-adjusted IC (ICIR) over the unbiased
signals panel, but ONLY for methods whose IC clears a statistical-confidence gate
(t = |ICIR|·sqrt(n_days) ≥ ic_weight_min_t). These tests lock: the confidence gate,
the relative-to-median boost, the floor for confidently-anti-predictive methods, and
the thin-panel no-op.
"""

import pandas as pd
import pytest

from config import settings
import src.signals.aggregator as agg


def _ic_df(rows, h: int = 5) -> pd.DataFrame:
    """Build a minimal compute_ic-shaped frame: method + icir_<h>d + icdays_<h>d."""
    return pd.DataFrame([
        {"method": r["method"], f"icir_{h}d": r.get("icir"), f"icdays_{h}d": r.get("icdays")}
        for r in rows
    ])


def test_ic_weights_enabled_by_default():
    assert settings.enable_ic_weights is True


def test_confident_methods_tilt_toward_higher_icir(monkeypatch):
    monkeypatch.setattr(settings, "ic_weight_min_multiplier", 0.25)
    monkeypatch.setattr(settings, "ic_weight_max_multiplier", 3.0)
    # n_days=100 ⇒ all clear t≥2 at default min_t; median positive ICIR = 0.5 → 1.0×.
    m = agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": 1.0, "icdays": 100},
        {"method": "tech", "icir": 0.5, "icdays": 100},
        {"method": "vwap", "icir": 0.25, "icdays": 100},
    ]), "5d")
    assert m["tech"] == pytest.approx(1.0, abs=0.05)
    assert m["news"] > m["tech"] > m["vwap"]
    assert m["news"] > 1.2 and m["vwap"] < 1.0


def test_confidence_gate_excludes_low_evidence():
    # High ICIR but only 2 signal-days → t = 1.0·√2 ≈ 1.41 < 2 → NOT reweighted;
    # a moderate ICIR with many days → t = 0.5·√100 = 5 ≥ 2 → reweighted.
    m = agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": 1.0, "icdays": 2},
        {"method": "tech", "icir": 0.5, "icdays": 100},
    ]), "5d")
    assert "news" not in m            # below the gate → keeps base weight
    assert "tech" in m


def test_no_confident_method_is_noop():
    # Nothing clears the t-gate → {} (every base weight untouched).
    assert agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": 0.3, "icdays": 4},     # t ≈ 0.6
        {"method": "tech", "icir": 0.2, "icdays": 9},     # t = 0.6
    ]), "5d") == {}


def test_confident_nonpositive_is_floored(monkeypatch):
    monkeypatch.setattr(settings, "ic_weight_min_multiplier", 0.25)
    m = agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": 0.5, "icdays": 100},   # confident positive → anchor
        {"method": "tech", "icir": -0.3, "icdays": 100},  # confident anti-predictive → floor
    ]), "5d")
    assert m["tech"] == pytest.approx(settings.ic_weight_min_multiplier)
    assert m["news"] == pytest.approx(1.0, abs=0.05)


def test_all_confident_negative_floored():
    m = agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": -0.5, "icdays": 100},
        {"method": "tech", "icir": -0.4, "icdays": 100},
    ]), "5d")
    assert m == {"news": settings.ic_weight_min_multiplier,
                 "tech": settings.ic_weight_min_multiplier}


def test_thin_method_keeps_base_weight():
    # No usable ICIR yet (None / 0 days) ⇒ omitted ⇒ base weight kept.
    m = agg._ic_mults_from_ic_table(_ic_df([
        {"method": "news", "icir": 0.5, "icdays": 100},
        {"method": "tech", "icir": None, "icdays": 0},
    ]), "5d")
    assert "tech" not in m and "news" in m


def test_mapping_works_for_shadow_horizon_label():
    # Horizon-agnostic: the shadow basis (default) passes a label key like "1w".
    df = pd.DataFrame([
        {"method": "news", "icir_1w": 0.5, "icdays_1w": 100},
        {"method": "tech", "icir_1w": 1.0, "icdays_1w": 100},
    ])
    m = agg._ic_mults_from_ic_table(df, "1w")
    assert m["tech"] > m["news"] > 0


def test_shadow_is_the_default_basis():
    assert settings.ic_weight_basis == "shadow"


def test_ic_weight_multipliers_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_ic_weights", False)
    agg.reset_ic_weight_cache()
    assert agg._ic_weight_multipliers() == {}


def test_compute_ic_exposes_icdays_and_feeds_weights(monkeypatch):
    # End-to-end column wiring: compute_ic emits icdays_<h>d, which the gate reads.
    from src.analysis.signal_panel import compute_ic
    monkeypatch.setattr(settings, "ic_weight_min_t", 0.4)   # +0.29 ICIR over 3 days → t≈0.50 clears
    data = []
    for day, fwds in (("2026-01-02", [1, 2, 3]), ("2026-01-05", [3, 2, 1]),
                      ("2026-01-06", [1, 2, 3])):              # per-day ICs +1/-1/+1 → ICIR>0
        for sc, fw in zip([0.1, 0.2, 0.3], fwds):
            data.append({"signal_date": day, "tech": sc, "fwd_ret_5d": float(fw)})
    ic = compute_ic(pd.DataFrame(data), horizons=(5,), min_n=9, min_per_day=3, min_days=3)
    assert "icdays_5d" in ic.columns
    assert int(ic[ic.method == "tech"].iloc[0]["icdays_5d"]) == 3
    assert "tech" in agg._ic_mults_from_ic_table(ic, "5d")
