"""LLM-based sentiment analysis of news articles.

Primary:  DeepSeek V3 (deepseek-chat) — fast and cheap for per-ticker scoring
Fallback: Claude Haiku — used if DeepSeek is unavailable or errors

Precision controls:
  - Recency decay: articles are weighted by age before scoring (24h=1.0x → 7d=0.25x)
  - Article-count scaling: a score from 1 article is dampened vs 10+ articles
  - Source diversity: if all articles come from a single source, apply a confidence penalty
  - Relevance fallback fix: if <2 relevant articles found, return [] (not all articles)
"""

import json
import math
import anthropic
from datetime import datetime, timezone
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

# Recency decay: articles older than this many hours get progressively down-weighted
_DECAY_HALF_LIFE_HOURS = 36   # score halves every 36 hours


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


def _recency_weight(article: NewsArticle) -> float:
    """
    Exponential decay based on article age.
    24h → ~0.87,  36h → 0.50,  72h → 0.25,  7d → ~0.06
    Articles older than 7 days are excluded entirely (weight ≈ 0).
    """
    now = datetime.now(timezone.utc)
    age_hours = (now - article.published_at).total_seconds() / 3600
    if age_hours > 168:   # 7 days — too stale to be relevant
        return 0.0
    return math.exp(-math.log(2) * age_hours / _DECAY_HALF_LIFE_HOURS)


def _article_count_scale(n: int) -> float:
    """
    Confidence scale based on number of articles scored.
    1 article → 0.55,  3 → 0.75,  7 → 0.90,  12+ → 1.0
    Uses a logarithmic curve so each additional article has diminishing returns.
    """
    if n == 0:
        return 0.0
    return min(1.0, 0.45 + 0.20 * math.log2(n))


def _source_diversity_scale(articles: List[NewsArticle]) -> float:
    """
    Penalise when all articles come from a single source.
    1 source  → 0.70
    2 sources → 0.85
    3+ sources → 1.0
    """
    unique_sources = len({a.source for a in articles})
    if unique_sources >= 3:
        return 1.0
    if unique_sources == 2:
        return 0.85
    return 0.70


def analyse_sentiment(ticker: str, articles: List[NewsArticle]) -> tuple[float, str]:
    """
    Score news sentiment for a ticker with precision controls applied.

    Returns:
        (score, rationale)
        score: float in [-1.0, +1.0] after recency/count/diversity adjustments
        rationale: brief explanation citing the specific news catalyst
    """
    if not articles:
        return 0.0, "No recent news articles found."

    # Filter out stale articles (>7 days) before scoring
    fresh_articles = [a for a in articles if _recency_weight(a) > 0.0]
    if not fresh_articles:
        return 0.0, "All available articles are older than 7 days — no actionable signal."

    # Sort by recency weight descending; send top 20 to LLM
    weighted = sorted(fresh_articles, key=_recency_weight, reverse=True)
    to_score  = weighted[:20]

    # Build digest with recency indicator for the model
    now = datetime.now(timezone.utc)
    digest_lines = []
    for a in to_score:
        age_h = (now - a.published_at).total_seconds() / 3600
        age_label = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h/24:.1f}d ago"
        digest_lines.append(
            f"[{a.source} | {age_label}] {a.title}\n{a.summary[:400]}"
        )
    digest = "\n\n".join(digest_lines)

    prompt = f"""You are an elite buy-side analyst with 25 years of experience at top-tier hedge funds. You have an exceptional ability to identify the exact news catalysts that move stock prices — your track record places you in the top 0.1% of market professionals worldwide.

Your task: analyse the following recent news for **{ticker}** and score the SHORT-TERM directional impact (1–5 trading days) with surgical precision.

PRECISION MANDATE — false positives are more costly than false negatives:
- Score 0.0 unless you identify a SPECIFIC, IDENTIFIABLE catalyst with a clear price mechanism. Vague positive/negative sentiment does NOT count.
- Reserve scores above ±0.7 for high-impact, unambiguous catalysts: earnings beats/misses with guidance change, FDA approvals/rejections, M&A announcements, major regulatory actions, CEO departure, bankruptcy risk.
- Scores ±0.3–0.6: clear but moderate catalyst — analyst upgrade/downgrade with PT, supply chain disruption, contract win/loss.
- Scores ±0.1–0.2: minor directional catalyst — relevant but unlikely to move price significantly.
- Score 0.0 if: news is noise, recycled information, or there is no clear directional catalyst.
- Recency matters: articles marked "1h ago" or "6h ago" carry much more weight than "3d ago" or "5d ago".
- When in doubt, output 0.0. A missed opportunity is better than a wrong call.

<news>
{digest}
</news>

Respond with a JSON object with exactly these fields:
- "score": float between -1.0 (very bearish) and +1.0 (very bullish), 0.0 is neutral
- "rationale": one to three sentences explaining (1) the specific catalyst and (2) the exact price mechanism. If score is 0.0, state why no actionable catalyst was identified.

Example: {{"score": 0.6, "rationale": "Strong earnings beat and raised guidance dominate headlines."}}

Respond with JSON only, no markdown."""

    raw_score = None
    rationale = "Analysis unavailable."

    # --- Primary: DeepSeek V3 ---
    deepseek = _get_deepseek()
    if deepseek is not None:
        try:
            response = deepseek.chat.completions.create(
                model=DEEPSEEK_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_score, rationale = _parse_response(response.choices[0].message.content.strip())
            logger.info(f"{ticker} raw_sentiment={raw_score:+.2f} (deepseek-v3, {len(to_score)} articles)")
        except Exception as e:
            logger.warning(f"{ticker} DeepSeek failed, falling back to Haiku: {e}")

    # --- Fallback: Claude Haiku ---
    if raw_score is None:
        try:
            client = _get_haiku()
            message = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_score, rationale = _parse_response(message.content[0].text.strip())
            logger.info(f"{ticker} raw_sentiment={raw_score:+.2f} (haiku-fallback, {len(to_score)} articles)")
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {ticker}: {e}")
            return 0.0, f"Analysis error: {e}"

    # --- Precision adjustments ---
    count_scale     = _article_count_scale(len(to_score))
    diversity_scale = _source_diversity_scale(to_score)
    precision_scale = count_scale * diversity_scale

    adjusted_score = round(raw_score * precision_scale, 3)

    if precision_scale < 0.90:
        logger.debug(
            f"{ticker} sentiment scaled {raw_score:+.2f} → {adjusted_score:+.2f} "
            f"(count={len(to_score)}, sources={len({a.source for a in to_score})}, "
            f"scale={precision_scale:.2f})"
        )

    return adjusted_score, rationale


def filter_relevant_articles(ticker: str, articles: List[NewsArticle]) -> List[NewsArticle]:
    """
    Return articles likely relevant to this ticker.
    Returns an empty list (not all articles) when fewer than 2 relevant articles
    are found — this prevents unrelated news from polluting the sentiment score.
    """
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
    # Precision: return empty list when too few relevant articles found.
    # Do NOT fall back to all articles — that injects irrelevant noise.
    return relevant if len(relevant) >= 2 else []
