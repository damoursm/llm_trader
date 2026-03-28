"""Final synthesis: Claude generates a high-level market report and recommendations."""

import anthropic
import json
from loguru import logger
from typing import List
from datetime import datetime, timezone
from config import settings
from src.models import TickerSignal, Recommendation


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_recommendations(signals: List[TickerSignal]) -> List[Recommendation]:
    """
    Feed all ticker signals to Claude and get final actionable recommendations.
    """
    if not signals:
        return []

    signals_text = "\n".join(
        f"- {s.ticker}: direction={s.direction}, confidence={s.confidence:.0%}, "
        f"news_sentiment={s.sentiment_score:+.2f}\n"
        f"  News rationale: {s.rationale}"
        for s in signals
    )

    prompt = f"""You are a high-conviction portfolio manager who makes decisive long and short calls every trading day. Your edge is identifying stocks and ETFs where recent news creates a clear asymmetric opportunity.

Today's date: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

INPUT — ticker signals derived from recent news sentiment:
<signals>
{signals_text}
</signals>

YOUR TASK:
1. Identify the 3-5 BEST opportunities across the full list — both longs (BUY) and shorts (SELL).
   - Every day you MUST find at least one actionable BUY and one actionable SELL if the news supports it.
   - Prefer high-momentum, high-news-volume tickers where the catalyst is fresh (< 48 h).
   - If a trending/newly-discovered ticker has strong news flow, do NOT ignore it just because it's not a mega-cap.

2. Distinguish time horizons:
   - "SWING" (2-10 days): catalyst-driven, price hasn't fully priced in the news yet.
   - "SHORT-TERM" (1-4 weeks): sector rotation, earnings run-up/fade, macro shift.
   - "POSITION" (1-3 months): structural change — regulatory, competitive moat disruption, macro theme.

3. Conviction rules:
   - confidence ≥ 0.75 → eligible for BUY / SELL.
   - confidence 0.50-0.74 → HOLD (monitor closely).
   - confidence < 0.50 → WATCH only.
   - Do NOT inflate confidence. A 90%+ call should be rare and backed by a dominant, unambiguous catalyst.

4. Short-selling discipline:
   - SELL means initiating a short position (or buying an inverse ETF).
   - Only short when: (a) clearly negative catalyst, (b) no counter-narrative, (c) broad market not in capitulation.

5. Do NOT factor in technical charts, moving averages, or price history — news catalysts only.

Output a JSON array where each element has:
- "ticker": string
- "type": "STOCK" | "ETF"
- "direction": "BULLISH" | "BEARISH" | "NEUTRAL"
- "action": "BUY" | "SELL" | "HOLD" | "WATCH"
- "time_horizon": "SWING" | "SHORT-TERM" | "POSITION" | "N/A"
- "confidence": float 0.0-1.0
- "rationale": 2-3 sentences — (1) name the specific catalyst, (2) explain the price mechanism, (3) state expected time horizon and risk.

Return ALL tickers from the input. No markdown, JSON only."""

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown fences if the model wrapped the response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        logger.debug(f"Raw Claude response ({len(raw)} chars): {raw[:200]}")
        data = json.loads(raw)
        now = datetime.now(timezone.utc)
        recommendations = [
            Recommendation(
                ticker=r["ticker"],
                type=r.get("type", "STOCK"),
                direction=r["direction"],
                action=r["action"],
                confidence=float(r["confidence"]),
                time_horizon=r.get("time_horizon", "N/A"),
                rationale=r["rationale"],
                generated_at=now,
            )
            for r in data
        ]
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Claude recommendations failed: {e}")
        # Fallback: convert signals directly to recommendations
        return _fallback_recommendations(signals)


def _fallback_recommendations(signals: List[TickerSignal]) -> List[Recommendation]:
    now = datetime.now(timezone.utc)
    recs = []
    for s in signals:
        if s.direction == "BULLISH" and s.confidence >= 0.6:
            action = "BUY"
        elif s.direction == "BEARISH" and s.confidence >= 0.6:
            action = "SELL"
        elif s.confidence < 0.4:
            action = "WATCH"
        else:
            action = "HOLD"
        recs.append(Recommendation(
            ticker=s.ticker,
            direction=s.direction,
            action=action,
            confidence=s.confidence,
            rationale=s.rationale,
            generated_at=now,
        ))
    return recs
