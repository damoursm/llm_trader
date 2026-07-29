"""Top-N recommendation ranking: LLM calls outrank rule-based fills (2026-07-27).

USER-REPORTED BUG: "the confidence scores from the recommendation emails are now
at 100%". They were — and not one of them came from the model.

Synthesis is only asked about the top ~40 tickers; the rest are back-filled by
`_fallback_recommendations` so open positions never fall silent. A fill carries
the AGGREGATOR's confidence verbatim (`confidence=s.confidence`), and that
saturates at 1.00 whenever the coherence factor hits its 1.35x cap. The top-N
sort keyed on `-confidence` alone, so a saturated fill outranked every genuine
call.

Measured over 2026-07-24..27: ALL 227 recommendations at 100% confidence were
`rule-based (fill)` and NONE came from `deepseek-v4-flash`; on the worst runs the
entire top-10 email was fills. The report was showing aggregator scores labelled
as recommendations while hiding what the LLM actually said.

The fix ranks fills LAST rather than discarding them — they still exist for the
positions they cover, they just stop displacing real recommendations.
"""

from datetime import datetime, timezone

import pytest

from src.models import Recommendation

_ACTION_RANK = {"BUY": 0, "SELL": 0, "HOLD": 1, "WATCH": 2}


def _rank(r):
    """Mirror of the pipeline's top-N key."""
    return (_ACTION_RANK.get(r.action, 3), bool(getattr(r, "rule_filled", False)),
            -r.confidence)


def _rec(ticker, conf, fill, action="BUY"):
    r = Recommendation(ticker=ticker, type="STOCK", action=action,
                       direction="BULLISH" if action == "BUY" else "BEARISH",
                       confidence=conf, rationale="x",
                       generated_at=datetime.now(timezone.utc))
    r.rule_filled = fill
    return r


def test_a_saturated_fill_does_not_outrank_a_genuine_call():
    """The exact reported bug: 1.00-confidence fills crowding out real calls."""
    recs = [_rec("FILL", 1.00, True), _rec("LLM", 0.88, False)]
    top = sorted(recs, key=_rank)
    assert top[0].ticker == "LLM", (
        "a rule-based fill at 100% must not outrank a genuine 88% LLM call")


def test_an_all_fill_top_ten_becomes_llm_first():
    """On the worst observed runs every one of the ten emailed rows was a fill."""
    recs = [_rec(f"F{i}", 1.00, True) for i in range(10)]
    recs += [_rec("LLM1", 0.90, False), _rec("LLM2", 0.86, False)]
    top = sorted(recs, key=_rank)[:10]
    assert [r.ticker for r in top[:2]] == ["LLM1", "LLM2"]
    assert sum(1 for r in top if r.rule_filled) == 8, "fills fill the remainder"


def test_action_rank_still_dominates():
    """BUY/SELL must still precede HOLD/WATCH — the fill tier is a tie-break
    WITHIN an action class, not above it."""
    recs = [_rec("HOLDLLM", 1.00, False, "HOLD"), _rec("BUYFILL", 0.10, True, "BUY")]
    assert sorted(recs, key=_rank)[0].ticker == "BUYFILL"


def test_confidence_still_orders_within_each_tier():
    llm = [_rec("A", 0.70, False), _rec("B", 0.95, False)]
    fills = [_rec("C", 0.80, True), _rec("D", 0.99, True)]
    top = sorted(llm + fills, key=_rank)
    assert [r.ticker for r in top] == ["B", "A", "D", "C"]


def test_fills_are_kept_not_discarded():
    """They exist so open positions never fall silent — ranking them last must
    not drop them."""
    recs = [_rec("F", 1.00, True), _rec("L", 0.90, False)]
    assert len(sorted(recs, key=_rank)) == 2


def test_missing_flag_is_treated_as_llm_authored():
    """Legacy rows predate the `rule_filled` stamp; defaulting them to 'fill'
    would silently demote real history."""
    r = _rec("LEGACY", 0.90, False)
    del r.rule_filled
    assert _rank(r)[1] is False


def test_pipeline_uses_this_key():
    """Guard against the pipeline and this test drifting apart."""
    import inspect
    import src.pipeline as pl
    src = inspect.getsource(pl.run_pipeline)
    assert "rule_filled" in src, "the top-N sort must consider rule_filled"
    assert "_ACTION_RANK.get(r.action, 3), is_fill" in src
