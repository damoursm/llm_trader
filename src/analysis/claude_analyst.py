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
        f"sentiment={s.sentiment_score:+.2f}, technical={s.technical_score:+.2f}\n"
        f"  Rationale: {s.rationale}"
        for s in signals
    )

    prompt = f"""You are a senior portfolio strategist. Based on the following ticker signals combining news sentiment and technical analysis, produce final actionable recommendations.

<signals>
{signals_text}
</signals>

Today's date: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

For each ticker, output a JSON object with:
- "ticker": string
- "direction": "BULLISH" | "BEARISH" | "NEUTRAL"
- "action": "BUY" | "SELL" | "HOLD" | "WATCH"
- "confidence": float 0.0-1.0
- "rationale": 2-3 sentence explanation

Return a JSON array of these objects. No markdown, JSON only."""

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        now = datetime.now(timezone.utc)
        recommendations = [
            Recommendation(
                ticker=r["ticker"],
                direction=r["direction"],
                action=r["action"],
                confidence=float(r["confidence"]),
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
