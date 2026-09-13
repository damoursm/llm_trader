"""The MECHANICAL entry selector — the rank rule that replaced the LLM (2026-09-04).

Measured before the switch (pre-registered, `/evaluate`, 1,245 actionable LLM
decisions over 16 days, per-run per-side count-matched, H/L pivot label, Gate-4
pool): the LLM funnel returned +0.19%/decision against a RANDOM control of
-0.04% (difference t +0.56 — not distinguishable from chance), while the same
selection made by the within-run rank of `combined_score` returned +1.56%
(vs random +1.60, t +2.11, same-sign halves). So the rank decides and the LLM
is demoted to a shadow arm.

What is pinned here is the SHAPE of the rule, because each part carries a
measured lesson:

  * the direction BAND still has to fire. That is the abstention half — a pure
    rank gate forces trades on runs the book judged weak, and when that exact
    question was asked of Gate 1c the pure-rank variant measured WORSE
    (-0.36..-0.50 %/day);
  * selection is capped PER SIDE, so a run cannot come out all-BUY the way the
    pooled Gate 1c did in 188 of 662 SELL-passing runs;
  * only Gate-4 TRADEABLE names can be selected, and the ranking is computed
    over that pool — ranking an observe-only name against the tradeable
    cross-section is what Gate 4 exists to prevent;
  * Gates 1 and 1c are skipped in rank mode and SAY SO (`gate1_mode`). Both are
    thresholds on the LLM's stated confidence; re-pointing them at the
    aggregator's confidence would silently re-filter on a scale nobody
    calibrated, and a gate that always passes is indistinguishable from one
    that never rejects.
"""

import pytest

from config.settings import settings
import src.pipeline as pl
from src.models import Recommendation, TickerSignal
from src.performance.tracker import _provider_of_synth_model
from src.signals import rank_entry


def _sig(ticker, score, direction="BULLISH", confidence=0.9):
    return TickerSignal(ticker=ticker, direction=direction, confidence=confidence,
                        combined_score=score, sources_agreeing=3, rationale="r",
                        sentiment_score=0.0, technical_score=0.0)


def _universe():
    """Six tradeable names spanning the score range, plus one thin name whose
    score would otherwise put it top of the book."""
    return [
        _sig("AAA", 0.90), _sig("BBB", 0.80), _sig("CCC", 0.70), _sig("DDD", 0.60),
        _sig("NEU", 0.00, "NEUTRAL"),
        _sig("SSS", -0.80, "BEARISH"), _sig("TTT", -0.90, "BEARISH"),
        _sig("THIN", 0.99),
    ]
    # THIN is left out of the tradeable set by the caller.


TRADEABLE = {"AAA", "BBB", "CCC", "DDD", "NEU", "SSS", "TTT"}


def _by_action(recs):
    out = {}
    for r in recs:
        out.setdefault(r.action, []).append(r.ticker)
    return out


def test_selects_the_extremes_of_the_tradeable_cross_section():
    recs = rank_entry.build_rank_recommendations(_universe(), tradeable=TRADEABLE,
                                                 max_per_side=2)
    acts = _by_action(recs)
    assert acts["BUY"] == ["AAA", "BBB"]        # top-2 by combined_score
    assert set(acts["SELL"]) == {"TTT", "SSS"}  # bottom-2 (rows keep input order)
    # The rationale carries the selection RANK, so the book is auditable after
    # the fact without re-deriving the cross-section.
    assert "bottom #1" in next(r for r in recs if r.ticker == "TTT").rationale
    assert "top #1" in next(r for r in recs if r.ticker == "AAA").rationale
    assert len(recs) == 8                       # one row per signal, always
    assert all(r.rationale for r in recs)


def test_an_untradeable_name_can_never_be_selected():
    """THIN outscores every tradeable name. Gate 4 would drop it downstream, but
    it must not consume a slot in the first place, and it must not enter the
    ranking it is being compared against."""
    recs = rank_entry.build_rank_recommendations(_universe(), tradeable=TRADEABLE,
                                                 max_per_side=2)
    thin = next(r for r in recs if r.ticker == "THIN")
    assert thin.action == "HOLD"
    assert _by_action(recs)["BUY"] == ["AAA", "BBB"]


def test_the_band_decides_whether_anything_trades_at_all():
    """The abstention half. A run where no direction fired trades NOTHING, even
    though a highest-ranked name always exists — the pure-rank variant of this
    rule measured worse for exactly this reason."""
    flat = [_sig("AAA", 0.42, "NEUTRAL"), _sig("BBB", 0.11, "NEUTRAL"),
            _sig("CCC", -0.30, "NEUTRAL")]
    recs = rank_entry.build_rank_recommendations(flat, tradeable={"AAA", "BBB", "CCC"})
    assert {r.action for r in recs} == {"WATCH"}


def test_a_fired_band_outside_the_cap_holds_rather_than_trades():
    recs = rank_entry.build_rank_recommendations(_universe(), tradeable=TRADEABLE,
                                                 max_per_side=1)
    acts = _by_action(recs)
    assert acts["BUY"] == ["AAA"] and acts["SELL"] == ["TTT"]
    # BBB/CCC/DDD had a view and no slot; SSS likewise on the other side.
    assert set(acts["HOLD"]) == {"BBB", "CCC", "DDD", "SSS", "THIN"}


def test_each_side_is_capped_on_its_own(monkeypatch):
    """Per-side, not pooled: a pooled top-K came out all-BUY in 188 of 662
    SELL-passing runs, starving the side the funnel measures as profitable."""
    monkeypatch.setattr(settings, "gate1_rank_cap", 3)
    recs = rank_entry.build_rank_recommendations(_universe(), tradeable=TRADEABLE)
    acts = _by_action(recs)
    assert acts["BUY"] == ["AAA", "BBB", "CCC"]
    assert set(acts["SELL"]) == {"TTT", "SSS"}  # only two bearish names exist
    assert len(acts["BUY"]) == 3 and len(acts["SELL"]) == 2


def test_zero_cap_trades_nothing():
    recs = rank_entry.build_rank_recommendations(_universe(), tradeable=TRADEABLE,
                                                 max_per_side=0)
    assert not [r for r in recs if r.action in ("BUY", "SELL")]


def test_confidence_is_carried_through_for_sizing():
    """Confidence is no longer a gate, but it still drives the sizing ramp — so
    it must survive onto the recommendation unchanged."""
    sigs = [_sig("AAA", 0.9, confidence=0.63)]
    rec = rank_entry.build_rank_recommendations(sigs, tradeable={"AAA"})[0]
    assert rec.confidence == pytest.approx(0.63)
    assert rec.action == "BUY" and rec.time_horizon == "SWING"
    assert not rec.rule_filled                  # a decision, not a back-fill


def test_the_rank_rule_has_its_own_provenance():
    """`rank-v1` must not resolve to an LLM engine: it rides
    runs.llm_synthesis_provider and the per-trade stamp, so an eval splits this
    era from the LLM one, and a rank-opened position is never pinned to an LLM
    hold review."""
    assert rank_entry.RANK_PROVIDER == "rank" and rank_entry.RANK_MODEL == "rank-v1"
    assert _provider_of_synth_model(rank_entry.RANK_MODEL) == "rank"
    from src.performance.tracker import _LLM_ENGINES
    assert "rank" not in _LLM_ENGINES


# ── the gate cascade in rank mode ────────────────────────────────────────────

def _rec(ticker, action="BUY", confidence=0.40):
    from datetime import datetime, timezone
    return Recommendation(ticker=ticker, direction="BULLISH" if action == "BUY" else "BEARISH",
                          action=action, confidence=confidence, rationale="r",
                          generated_at=datetime.now(timezone.utc))


def _run_gates(recs, rank_mode, monkeypatch):
    diag = {k: 0 for k in ("buy_sell_candidates", "dropped_below_threshold",
                           "dropped_rank_cap", "dropped_low_agreement",
                           "dropped_buy_blocked", "dropped_earnings_blackout",
                           "dropped_illiquid", "dropped_wide_book",
                           "dropped_overextended", "actionable_survivors")}
    outcomes = {}
    monkeypatch.setattr(pl, "_passes_agreement_gate", lambda *a, **k: True)
    monkeypatch.setattr(pl, "_is_tradeable", lambda *a, **k: True)
    monkeypatch.setattr(pl, "_is_wide_book", lambda *a, **k: False)
    monkeypatch.setattr(pl, "_is_overextended", lambda *a, **k: False)
    out = pl._apply_actionable_gates(
        recs, confidence_threshold=0.85, side_threshold_adj={},
        signals_by_ticker={}, allow_buys=True, earnings_blackout=set(),
        trade_gate_budget={"n": 0}, gate_diag=diag, gate_outcomes=outcomes,
        rank_mode=rank_mode)
    return out, diag, outcomes


def test_rank_mode_skips_the_llm_confidence_gates(monkeypatch):
    """Gate 1 is a floor on the LLM's stated confidence and Gate 1c a rank of
    it. In rank mode the candidates carry the AGGREGATOR's confidence, which
    lives on a different scale — applying the 0.85 floor to it would drop the
    whole book for a reason nobody measured."""
    monkeypatch.setattr(settings, "gate1_rank_cap", 1)
    recs = [_rec("AAA", confidence=0.40), _rec("BBB", confidence=0.35)]

    kept, diag, _ = _run_gates(recs, rank_mode=True, monkeypatch=monkeypatch)
    assert [r.ticker for r in kept] == ["AAA", "BBB"]
    assert diag["dropped_below_threshold"] == 0 and diag["dropped_rank_cap"] == 0
    assert diag["gate1_mode"] == "rank"


def test_llm_mode_is_unchanged_by_the_new_parameter(monkeypatch):
    """The default path must be byte-identical: same drops, same stamps."""
    monkeypatch.setattr(settings, "gate1_rank_cap", 1)
    recs = [_rec("AAA", confidence=0.40), _rec("BBB", confidence=0.95),
            _rec("CCC", confidence=0.90)]

    kept, diag, outcomes = _run_gates(recs, rank_mode=False, monkeypatch=monkeypatch)
    assert [r.ticker for r in kept] == ["BBB"]          # floor, then top-1
    assert diag["dropped_below_threshold"] == 1 and outcomes["AAA"] == "below_threshold"
    assert diag["dropped_rank_cap"] == 1 and outcomes["CCC"] == "rank_capped"
    assert diag["gate1_mode"] == "llm_confidence"


def test_the_other_gates_still_apply_in_rank_mode(monkeypatch):
    """Only Gates 1 and 1c are not applicable. The PANIC buy block, the
    earnings blackout, liquidity, book width and the anti-chase rule are
    orthogonal to who chose the candidate and must keep firing."""
    monkeypatch.setattr(settings, "gate1_rank_cap", 0)
    diag = {k: 0 for k in ("buy_sell_candidates", "dropped_below_threshold",
                           "dropped_rank_cap", "dropped_low_agreement",
                           "dropped_buy_blocked", "dropped_earnings_blackout",
                           "dropped_illiquid", "dropped_wide_book",
                           "dropped_overextended", "actionable_survivors")}
    outcomes = {}
    monkeypatch.setattr(pl, "_passes_agreement_gate", lambda *a, **k: True)
    monkeypatch.setattr(pl, "_is_tradeable", lambda *a, **k: True)
    monkeypatch.setattr(pl, "_is_wide_book", lambda *a, **k: False)
    monkeypatch.setattr(pl, "_is_overextended", lambda *a, **k: False)

    kept = pl._apply_actionable_gates(
        [_rec("AAA"), _rec("BBB", "SELL")], confidence_threshold=0.85,
        side_threshold_adj={}, signals_by_ticker={}, allow_buys=False,
        earnings_blackout={"BBB"}, trade_gate_budget={"n": 0},
        gate_diag=diag, gate_outcomes=outcomes, rank_mode=True)

    assert kept == []
    assert outcomes["AAA"] == "buy_blocked" and outcomes["BBB"] == "earnings_blackout"
