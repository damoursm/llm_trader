"""Catalyst timing is computed BEFORE LLM synthesis and reaches the prompt.

Pipeline order (2026-06-12): Step 4.5 computes the catalyst context so the
LLM can reason WITH the event calendar (don't waste a BUY on an
earnings-blackout name; weigh an 8-K + insider setup); the mechanical
guarantees — WATCH elevation on the ranked top-10 and the earnings-blackout
gate on the actionable set — still run on the LLM's output afterwards.
"""

import json

from src.models import CatalystSetup, CatalystTimingContext, TickerSignal


def _signal(ticker="XLE"):
    return TickerSignal(
        ticker=ticker, direction="BULLISH", confidence=0.4,
        action_suggestion="WATCH", news_sentiment_score=0.0,
        sentiment_score=0.0, insider_score=0.0, technical_score=0.1,
        sources_agreeing=1, key_reasons=["r"], rationale="r", price=57.0,
    )


def _context():
    return CatalystTimingContext(
        earnings_blackout_tickers=["NVDA"],
        earnings_blackout_details={"NVDA": 1},
        opex_max_pain_weight=0.28, opex_boost_active=True,
        opex_is_triple_witching=True, opex_signal="TRIPLE_WITCHING_WEEK",
        catalyst_setups=[CatalystSetup(
            ticker="MTDR", has_8k=True, has_insider_buy=True,
            catalyst_reason="8-K material agreement + officer cluster buy")],
        watch_elevation_tickers=["MTDR"],
        summary="1 blackout, 1 setup, triple witching",
    )


def test_catalyst_context_reaches_the_synthesis_prompt(monkeypatch):
    import src.analysis.claude_analyst as ca

    captured = {}

    def fake(prompt):
        captured["prompt"] = prompt
        return json.dumps([{"ticker": "XLE", "type": "ETF",
                            "direction": "BULLISH", "action": "WATCH",
                            "confidence": 0.5, "rationale": "x"}])

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)

    recs = ca.generate_recommendations([_signal()], catalyst_timing_context=_context())
    assert len(recs) == 1
    p = captured["prompt"]
    assert "<catalyst_timing_context>" in p
    assert "NVDA: earnings in 1 day(s)" in p
    assert "MTDR: 8-K=yes, insider buy=yes" in p
    assert "26. Catalyst timing overlay" in p
    assert "ACTIVE — triple witching" in p


def test_no_catalyst_context_means_no_block(monkeypatch):
    import src.analysis.claude_analyst as ca

    captured = {}

    def fake(prompt):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)

    ca.generate_recommendations([_signal()], catalyst_timing_context=None)
    assert "<catalyst_timing_context>" not in captured["prompt"]
    assert "26. Catalyst timing overlay" not in captured["prompt"]
