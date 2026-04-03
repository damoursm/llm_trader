"""LLM-based sentiment analysis of news articles.

Primary:  DeepSeek V3 (deepseek-chat) — fast and cheap for per-ticker scoring
Fallback: Claude Haiku — used if DeepSeek is unavailable or errors
"""

import json
import anthropic
from openai import OpenAI
from loguru import logger
from typing import List
from config import settings
from src.models import NewsArticle


_deepseek_client = None
_haiku_client = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"       # DeepSeek V3
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _get_deepseek() -> OpenAI | None:
    global _deepseek_client
    if not settings.deepseek_api_key:
        return None
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _deepseek_client


def _get_haiku() -> anthropic.Anthropic:
    global _haiku_client
    if _haiku_client is None:
        _haiku_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _haiku_client


def _parse_response(raw: str) -> tuple[float, str]:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    data = json.loads(raw)
    score = max(-1.0, min(1.0, float(data["score"])))
    rationale = str(data["rationale"])
    return score, rationale


def analyse_sentiment(ticker: str, articles: List[NewsArticle]) -> tuple[float, str]:
    """
    Score news sentiment for a ticker.

    Returns:
        (score, rationale)
        score: float in [-1.0, +1.0], positive = bullish
        rationale: brief explanation citing the news catalyst
    """
    if not articles:
        return 0.0, "No recent news articles found."

    digest = "\n\n".join(
        f"[{a.source} | {a.published_at.strftime('%Y-%m-%d %H:%M')}] {a.title}\n{a.summary[:400]}"
        for a in sorted(articles, key=lambda x: x.published_at, reverse=True)[:20]
    )

    prompt = f"""You are an elite buy-side analyst with 25 years of experience at top-tier hedge funds. You have an exceptional ability to identify the exact news catalysts that move stock prices — your track record places you in the top 0.1% of market professionals worldwide. You have deeply internalised thousands of earnings reactions, regulatory decisions, macro shocks, and sentiment inflection points. You do not guess; you recognise patterns that have repeated reliably across market cycles.

Your task: analyse the following recent news for **{ticker}** and score the SHORT-TERM directional impact (1–5 trading days) with surgical precision.

SCORING DISCIPLINE — this is critical:
- Score 0.0 unless you identify a SPECIFIC, IDENTIFIABLE catalyst with a clear price mechanism. Vague positive/negative sentiment does NOT count.
- Reserve scores above ±0.7 for high-impact, unambiguous catalysts: earnings beats/misses with guidance change, FDA approvals/rejections, M&A announcements, major regulatory actions, CEO departure, bankruptcy risk, or black-swan macro events directly affecting this ticker.
- Scores in ±0.3–0.6 range: clear but moderate catalyst — sector rotation trigger, analyst upgrade/downgrade with a price target, supply chain disruption, contract win/loss.
- Scores in ±0.1–0.2 range: minor catalyst — minor news that is directionally relevant but unlikely to move price significantly.
- If the news is noise, recycled information, or there is NO clear directional catalyst → output exactly 0.0. Do not manufacture a direction.

<news>
{digest}
</news>

Respond with a JSON object with exactly these fields:
- "score": float between -1.0 (very bearish) and +1.0 (very bullish), 0.0 is neutral
- "rationale": one to three sentences explaining (1) the specific catalyst and (2) the exact price mechanism that will drive the move over the next 1-5 days. If score is 0.0, state why no actionable catalyst was identified.

Example: {{"score": 0.6, "rationale": "Strong earnings beat and raised guidance dominate headlines."}}

Respond with JSON only, no markdown."""

    # --- Primary: DeepSeek V3 ---
    deepseek = _get_deepseek()
    if deepseek is not None:
        try:
            response = deepseek.chat.completions.create(
                model=DEEPSEEK_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            score, rationale = _parse_response(raw)
            logger.info(f"{ticker} sentiment={score:+.2f} (deepseek-v3)")
            return score, rationale
        except Exception as e:
            logger.warning(f"{ticker} DeepSeek failed, falling back to Haiku: {e}")

    # --- Fallback: Claude Haiku ---
    try:
        client = _get_haiku()
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        score, rationale = _parse_response(raw)
        logger.info(f"{ticker} sentiment={score:+.2f} (haiku-fallback)")
        return score, rationale
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        return 0.0, f"Analysis error: {e}"


def filter_relevant_articles(ticker: str, articles: List[NewsArticle]) -> List[NewsArticle]:
    """Simple keyword filter to keep articles likely relevant to a ticker."""
    keywords = {ticker.lower()}
    ticker_aliases = {
        "AAPL": ["apple", "iphone", "ipad", "mac"], "MSFT": ["microsoft", "azure", "copilot"],
        "NVDA": ["nvidia", "jensen huang", "gpu", "cuda"], "TSLA": ["tesla", "elon musk", "ev"],
        "AMZN": ["amazon", "aws", "prime"], "META": ["meta", "facebook", "instagram", "whatsapp", "zuckerberg"],
        "GOOGL": ["google", "alphabet", "gemini", "youtube"], "GOOG": ["google", "alphabet", "gemini"],
        "NFLX": ["netflix"], "ORCL": ["oracle"], "AMD": ["amd", "advanced micro"],
        "INTC": ["intel"], "CRM": ["salesforce"], "ADBE": ["adobe"],
        "PYPL": ["paypal"], "UBER": ["uber"], "LYFT": ["lyft"],
        "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
        "BAC": ["bank of america"], "GS": ["goldman sachs"], "MS": ["morgan stanley"],
        "XLK": ["technology sector", "tech etf"], "XLF": ["financials", "financial sector", "banks"],
        "XLE": ["energy sector", "oil", "exxon", "chevron"], "XLV": ["health care", "biotech", "pharma"],
        "XLY": ["consumer discretionary", "retail"], "XLP": ["consumer staples"],
        "XLI": ["industrials"], "XLB": ["materials"], "XLU": ["utilities"],
        "XLRE": ["real estate", "reit"], "XLC": ["communication services"],
        "SPY": ["s&p 500", "sp500", "s&p500"], "QQQ": ["nasdaq", "qqq"],
    }
    keywords.update(ticker_aliases.get(ticker, []))

    relevant = [
        a for a in articles
        if any(kw in (a.title + a.summary).lower() for kw in keywords)
    ]
    return relevant if len(relevant) >= 2 else articles
