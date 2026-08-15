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

def test_epoch_is_the_method_rank_basis(monkeypatch):
    """The epoch tracks the LATEST categorical change to what |combined| (and
    hence confidence) MEANS. History: 2026-07-22 the buy/sell split (difference
    of camp averages replaced one pooled average); 2026-08-14 the method RANK
    basis (the combine consumes centered within-run ranks, so mean |eff score|
    jumps from ~0.1-0.2 to ~0.5 by construction — same categorical test).
    02:00 UTC mid-day change -> the day-after convention yields 08-15."""
    monkeypatch.setattr(settings, "enable_confidence_epoch", True)
    assert me.confidence_epoch() == date(2026, 8, 15)


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
        "signal_date": ["2026-08-10", "2026-08-20"],
        "ticker": ["AAA", "BBB"],
        "run_id": ["r1", "r2"],
        "generated_at": ["2026-08-10T12:00:00", "2026-08-20T12:00:00"],
        "confidence": [0.90, 0.80],
        "raw_confidence": [0.5, 0.5],
        "combined_score": [0.4, 0.4],
        "tech": [0.3, 0.3],
        "price": [10.0, 10.0],
    })
    out = build_panel(horizons=(1,), signals_df=df, dedupe="all")
    pre = out[out["signal_date"] == "2026-08-10"].iloc[0]
    post = out[out["signal_date"] == "2026-08-20"].iloc[0]
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

def _reconstruct(s) -> float:
    """The confidence formula from the PERSISTED components — all SEVEN.

    `sector_conf_factor` joined the set on 2026-08-14: the sector-alignment
    multiplier (1.10 aligned / 0.75 contradicted) had been applied to
    `confidence` since the pass was written while being stored nowhere, so a
    sector-touched row could not multiply back BY CONSTRUCTION. Leaving it out
    of this reconstruction would re-open exactly that hole from the test side —
    the check would keep passing on fixtures where the factor happens to be
    neutral and fail on the ones where the code is right.
    """
    return round(min(1.0, s.raw_confidence * s.coherence_factor
                     * s.movement_factor * s.volume_factor
                     * s.family_conf_factor * s.tape_conf_factor
                     * s.sector_conf_factor), 2)


def test_components_reconstruct_confidence_after_the_overlay():
    """The bug: the cross-sectional overlay rescales confidence AFTER the
    components are captured, leaving `raw_confidence` describing the pre-overlay
    combined score. Reconstruction was 24.7% on overlay-touched rows versus
    92.1% on untouched ones."""
    import src.signals.aggregator as agg
    sigs = agg.build_signals(["AAPL", "MSFT", "GLD"], [])
    assert sigs, "need signals to check"
    checked = 0
    for s in sigs:
        if s.confidence == 0.0:
            # Gate-zeroed (neutral direction): the stored components describe
            # the pre-gate FORMULA by design — a gate decision is not a factor
            # and reconstruction is undefined here. (Surfaced 2026-08-12 when
            # the ml_ohlcv basis guard shifted a fixture ticker into the
            # neutral branch; any method abstaining can do this.)
            continue
        recon = _reconstruct(s)
        assert abs(recon - s.confidence) <= 0.011, (
            f"{s.ticker}: components give {recon} but confidence is {s.confidence} "
            f"(cross_sectional={s.cross_sectional_score})")
        checked += 1
    assert checked >= 1, "every fixture row was gate-zeroed — test lost its subject"


def test_reconstruction_holds_when_the_sector_pass_actually_FIRES(monkeypatch):
    """The seventh factor's own case.

    `_sector_alignment_factor` needs the ticker's sector ETF to be in the same
    run AND to carry a non-trivial sentiment/technical/insider blend, which no
    offline fixture produces — so the reconstruction above has never once seen a
    non-neutral sector factor. Forcing it is the only way to check that the
    persisted factor is the one the formula used (a 0.75 haircut stored as 1.0
    would sail through every other test in this file)."""
    import src.signals.aggregator as agg
    monkeypatch.setattr(agg, "_sector_alignment_factor", lambda *a, **k: 0.75)
    sigs = agg.build_signals(["AAPL", "MSFT", "GLD"], [])
    touched = [s for s in sigs if s.confidence > 0.0]
    assert touched, "every fixture row was gate-zeroed — test lost its subject"
    for s in touched:
        assert s.sector_conf_factor == 0.75, "the applied factor was not persisted"
        assert abs(_reconstruct(s) - s.confidence) <= 0.011, (
            f"{s.ticker}: sector-adjusted confidence {s.confidence} does not "
            f"reconstruct from its components")


def test_raw_confidence_tracks_the_POST_overlay_combined_score():
    """raw_confidence is min(1, |combined|/scale) — it must describe the combined
    score the row actually ends up carrying, not the pre-overlay one.

    The divisor is resolved through `aggregator._raw_confidence_scale()` rather
    than written out: it is 0.5 only on the ABSOLUTE basis (which the conftest
    pins suite-wide), 0.642 under the live rank basis and 0.0658 on an ML-combine
    row. Hard-coding it here would re-encode precisely the literal the 2026-08-14
    fix centralised — and `test_method_rank_basis.py` has an AST guard forbidding
    that same literal in `src/`."""
    import src.signals.aggregator as agg
    scale = agg._raw_confidence_scale()
    for s in agg.build_signals(["AAPL", "NVDA", "TSLA"], []):
        expected = min(1.0, abs(s.combined_score) / scale)
        assert abs(s.raw_confidence - expected) <= 0.02, (
            f"{s.ticker}: raw_confidence {s.raw_confidence} does not match "
            f"|combined|/{scale} = {expected}")
