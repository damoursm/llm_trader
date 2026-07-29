"""Confidence epoch + component-capture correctness (2026-07-27).

Two related fixes, both prompted by "can you retroactively rescore the whole
database with the current confidence setup?".

**The retrofit was REJECTED** and this is what replaced it. Reasons, in order of
severity: today's weights are calibrated FROM this panel, so rescoring the past
with them injects future information into the exact dataset that feeds the
IC-weight layer, predictability sizing and policy eval; 78.3% of rows predate
the component capture entirely; the OHLCV cache is retroactively split-adjusted
so movement/volume/tape cannot be reproduced as they were. A rescored panel
would look authoritative and be partly invented — strictly worse than a panel
known to be heterogeneous.

**The component-capture bug** was found while testing that: multiplying the six
persisted components reproduced the persisted confidence for only 24.7% of rows
— but 92.1% of the rows the cross-sectional overlay never touched. The overlay
runs AFTER capture and rescales confidence without updating `raw_confidence`, so
`confidence_components.py` was isolating factors of a value that was not the
final confidence.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
import src.signals.method_epochs as me


# ── the epoch ──────────────────────────────────────────────────────────────

def test_epoch_is_the_buy_sell_split(monkeypatch):
    """Registered because the SPLIT changed what combined_score MEANS — a
    difference of two camp averages, not one pooled average — and
    raw_confidence derives straight from |combined|. Level shifts alone would
    be a refinement; a change of meaning is categorical."""
    monkeypatch.setattr(settings, "enable_confidence_epoch", True)
    assert me.confidence_epoch() == date(2026, 7, 22)


def test_epoch_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_confidence_epoch", False)
    assert me.confidence_epoch() is None


def test_it_governs_the_components_too():
    """A component is only interpretable next to the confidence it explains, so
    masking one without the others would leave a misleading half-record."""
    assert "confidence" in me.CONFIDENCE_EPOCH_COLUMNS
    for c in ("raw_confidence", "coherence_factor", "movement_factor",
              "volume_factor", "family_conf_factor", "tape_conf_factor"):
        assert c in me.CONFIDENCE_EPOCH_COLUMNS


def test_masking_blanks_confidence_but_KEEPS_the_row(monkeypatch):
    """The whole point of masking over deletion: the row's other evidence —
    method scores, prices, forward returns — was never affected by the change
    and stays usable."""
    monkeypatch.setattr(settings, "enable_confidence_epoch", True)
    from src.analysis.signal_panel import build_panel
    df = pd.DataFrame({
        "signal_date": ["2026-07-20", "2026-07-25"],
        "ticker": ["AAA", "BBB"],
        "run_id": ["r1", "r2"],
        "generated_at": ["2026-07-20T12:00:00", "2026-07-25T12:00:00"],
        "confidence": [0.90, 0.80],
        "raw_confidence": [0.5, 0.5],
        "combined_score": [0.4, 0.4],
        "tech": [0.3, 0.3],
        "price": [10.0, 10.0],
    })
    out = build_panel(horizons=(1,), signals_df=df, dedupe="all")
    pre = out[out["signal_date"] == "2026-07-20"].iloc[0]
    post = out[out["signal_date"] == "2026-07-25"].iloc[0]
    assert pd.isna(pre["confidence"]), "pre-epoch confidence must be masked"
    assert pd.isna(pre["raw_confidence"]), "its components too"
    assert post["confidence"] == 0.80, "post-epoch is untouched"
    assert pre["tech"] == 0.3, "method scores survive — they are still evidence"
    assert pre["combined_score"] == 0.4
    assert len(out) == 2, "rows are kept, never dropped"


def test_disabled_leaves_history_visible(monkeypatch):
    monkeypatch.setattr(settings, "enable_confidence_epoch", False)
    from src.analysis.signal_panel import build_panel
    df = pd.DataFrame({
        "signal_date": ["2026-07-20"], "ticker": ["AAA"], "run_id": ["r1"],
        "generated_at": ["2026-07-20T12:00:00"], "confidence": [0.90],
        "combined_score": [0.4], "price": [10.0],
    })
    out = build_panel(horizons=(1,), signals_df=df, dedupe="all")
    assert out.iloc[0]["confidence"] == 0.90


# ── the component-capture fix ──────────────────────────────────────────────

def test_components_reconstruct_confidence_after_the_overlay():
    """The bug: the cross-sectional overlay rescales confidence AFTER the
    components are captured, leaving `raw_confidence` describing the pre-overlay
    combined score. Reconstruction was 24.7% on overlay-touched rows versus
    92.1% on untouched ones."""
    import src.signals.aggregator as agg
    sigs = agg.build_signals(["AAPL", "MSFT", "GLD"], [])
    assert sigs, "need signals to check"
    for s in sigs:
        prod = (s.raw_confidence * s.coherence_factor * s.movement_factor
                * s.volume_factor * s.family_conf_factor * s.tape_conf_factor)
        recon = round(min(1.0, prod), 2)
        assert abs(recon - s.confidence) <= 0.011, (
            f"{s.ticker}: components give {recon} but confidence is {s.confidence} "
            f"(cross_sectional={s.cross_sectional_score})")


def test_raw_confidence_tracks_the_POST_overlay_combined_score():
    """raw_confidence is min(1, |combined|/0.5) — it must describe the combined
    score the row actually ends up carrying, not the pre-overlay one."""
    import src.signals.aggregator as agg
    for s in agg.build_signals(["AAPL", "NVDA", "TSLA"], []):
        expected = min(1.0, abs(s.combined_score) / 0.5)
        assert abs(s.raw_confidence - expected) <= 0.02, (
            f"{s.ticker}: raw_confidence {s.raw_confidence} does not match "
            f"|combined|/0.5 = {expected}")
