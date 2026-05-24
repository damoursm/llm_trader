"""Sentiment velocity — the rate of change of news tone (Δsentiment, not level).

The *change* in sentiment leads short-horizon (1–5 day) price moves better than the
absolute level: a stock improving from very negative toward neutral often rallies even
while still mildly negative (the second derivative turned up), and a stock fading from
very positive often sells off even while still net-positive.

This module reuses the article timestamps we already store (``NewsArticle.published_at``)
and adds **no LLM/API cost**: it scores each article with a deterministic financial
lexical polarity, buckets articles by recency into a *recent* and a *prior* window, and
takes the difference:

    velocity = mean_tone(recent window) − mean_tone(prior window)        ∈ [-2, +2]
    score    = tanh(velocity / SCALE) × confidence                       ∈ [-1, +1]

``confidence`` damps the score when either window is thin (few articles). Returns 0 when
either window lacks articles — no change can be measured. Differencing a *consistent*
lexical measure across windows isolates the shift in tone, so absolute lexicon
calibration matters far less than it would for a level signal.
"""

import math
import re
from datetime import datetime, timezone
from statistics import mean
from typing import List, Tuple

from src.models import NewsArticle

# ── Compact financial polarity lexicon ───────────────────────────────────────
_POS = frozenset({
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "jump",
    "jumps", "upgrade", "upgrades", "upgraded", "record", "strong", "growth", "raise",
    "raises", "raised", "outperform", "buy", "bullish", "gain", "gains", "tops", "top",
    "win", "wins", "approval", "approved", "breakthrough", "expand", "expands",
    "expansion", "profit", "profits", "rebound", "recovery", "optimistic", "upbeat",
    "accelerate", "accelerating", "milestone", "partnership", "contract", "dividend",
    "buyback", "boost", "boosts", "rises", "rise", "higher", "robust", "exceeds",
    "exceeded", "momentum", "demand", "winning",
})
_NEG = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "sink", "sinks", "downgrade", "downgrades", "downgraded", "weak", "weakness",
    "decline", "declines", "cut", "cuts", "slash", "slashes", "underperform", "sell",
    "bearish", "loss", "losses", "lawsuit", "probe", "investigation", "recall",
    "bankruptcy", "default", "warning", "warn", "warns", "fraud", "layoff", "layoffs",
    "resign", "resigns", "plummet", "plummets", "slump", "halt", "halts", "delay",
    "delays", "concern", "concerns", "risk", "risks", "fear", "fears", "selloff",
    "lower", "slowdown", "disappoint", "disappoints", "disappointing", "guidance cut",
    "downbeat", "tumble", "tumbles", "crash",
})

_TOKEN_RE = re.compile(r"[a-z']+")


def _lexical_polarity(article: NewsArticle) -> float:
    """Per-article tone ∈ [-1, +1] from positive/negative financial keyword balance."""
    text = f"{article.title} {article.summary}".lower()
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POS)
    neg = sum(1 for t in tokens if t in _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _count_scale(n: int) -> float:
    """Confidence scale by article count: 1→0.55, 3→0.75, 7→0.90, 12+→1.0."""
    if n <= 0:
        return 0.0
    return min(1.0, 0.45 + 0.20 * math.log2(n))


# Tanh scale: a recent-vs-prior tone gap of ~0.6 maps to ~tanh(1)=0.76 before confidence.
_SCALE = 0.6


def compute_sentiment_velocity(
    ticker: str,
    articles: List[NewsArticle],
    recent_hours: int = 24,
    prior_hours: int = 96,
) -> Tuple[float, float, float, int]:
    """Compute the sentiment velocity for ``ticker`` from already-filtered articles.

    Args:
        ticker: ticker symbol (for logging/clarity only).
        articles: relevant articles for the ticker (already keyword-filtered).
        recent_hours: upper age bound (hours) of the *recent* window.
        prior_hours: upper age bound (hours) of the *prior* window (must be > recent_hours).

    Returns:
        ``(velocity_score, recent_tone, prior_tone, n_articles_used)``.
        ``velocity_score`` ∈ [-1, +1]; 0.0 when either window is empty.
    """
    if not articles or prior_hours <= recent_hours:
        return 0.0, 0.0, 0.0, 0

    now = datetime.now(timezone.utc)
    recent: List[float] = []
    prior: List[float] = []
    for a in articles:
        try:
            age_h = (now - a.published_at).total_seconds() / 3600.0
        except (TypeError, AttributeError):
            continue
        if age_h < 0:
            age_h = 0.0
        if age_h <= recent_hours:
            recent.append(_lexical_polarity(a))
        elif age_h <= prior_hours:
            prior.append(_lexical_polarity(a))

    # Need at least one article in each window to measure a change.
    if not recent or not prior:
        return 0.0, 0.0, 0.0, len(recent) + len(prior)

    recent_tone = mean(recent)
    prior_tone = mean(prior)
    raw = recent_tone - prior_tone                          # ∈ [-2, +2]
    confidence = _count_scale(len(recent)) * _count_scale(len(prior))
    score = math.tanh(raw / _SCALE) * confidence
    score = round(max(-1.0, min(1.0, score)), 3)

    return score, round(recent_tone, 3), round(prior_tone, 3), len(recent) + len(prior)
