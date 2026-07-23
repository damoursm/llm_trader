"""LLM-based sentiment analysis of news articles.

Engines: DeepSeek V4-Flash (deepseek-v4-flash, non-thinking) and Claude Haiku.
Which one is PRIMARY is flipped once per run (settings.llm_ab_anthropic_share,
default 50/50) so both accumulate comparable samples for the dashboard's
per-LLM evaluation; the other engine is the error fallback.

Precision controls:
  - Recency decay: articles weighted by age before scoring (fresh=1.0x, 18h=0.5x, ~2d=0.16x, >7d dropped)
  - Article-count scaling: a score from 1 article is dampened vs 10+ articles
  - Source diversity: if all articles come from a single source, apply a confidence penalty
  - Relevance fallback fix: if <2 relevant articles found, return [] (not all articles)
"""

import hashlib
import json
import math
import random
import re
import threading
import time
import anthropic
from datetime import datetime, timezone
from openai import OpenAI
from loguru import logger
from typing import List, Optional
from config import settings
from src.models import NewsArticle


_deepseek_client = None
_haiku_client = None
_qwen_client = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"   # DeepSeek V4-Flash — cheapest/latest (replaces deprecated deepseek-chat)
# DeepSeek reasoning mode (extra_body). OFF (default) keeps bulk sentiment cheap/
# deterministic; ON is used under the maximum-thinking policy (llm_max_thinking).
_DEEPSEEK_THINKING_OFF = {"thinking": {"type": "disabled"}}
_DEEPSEEK_THINKING_ON = {"thinking": {"type": "enabled"}}
HAIKU_MODEL = "claude-haiku-4-5-20251001"
# Qwen (DashScope direct or OpenRouter — settings.qwen_base_url/qwen_model pick
# the route; src/analysis/qwen_api.py picks the thinking dialect). 2026-07-13:
# scores ~10% of runs (sentiment_qwen_share; DeepSeek-flash the rest) — the pricier
# engine, kept to a minority for cost. Reasoning is billed separately from the
# answer's max_tokens on both routes.
QWEN_MODEL = settings.qwen_model      # resolved at import from the active route

# Output ceiling for the sentiment call. The old 256 truncated a verbose DeepSeek
# rationale mid-JSON (XBI) → invalid JSON → lost score. max_tokens is only a CEILING
# — the model stops at the JSON close and is billed per ACTUAL token, so a score +
# short rationale still emits ~100 tokens regardless. A generous cap therefore costs
# nothing and makes truncation impossible. Shared by both engines (≤ Haiku 4.5's
# 8192 hard limit; DeepSeek accepts far more — synthesis uses 32000).
_SENTIMENT_MAX_TOKENS = 4096

# Thread-safe tally of which provider answered each per-ticker sentiment call this
# run (sentiment runs concurrently across tickers). Surfaced to the pipeline as the
# run's llm_sentiment_provider so silent DeepSeek→Haiku fallbacks are observable.
_PROVIDER_COUNTS: dict = {}
_PROVIDER_LOCK = threading.Lock()

# Which engine scores this run — re-flipped per run in reset_sentiment_providers()
# (2026-07-13: ~90% DeepSeek / ~10% Qwen via sentiment_qwen_share, cost tune), so
# both providers accumulate whole-run samples for the dashboard's per-LLM evaluation
# rows and the run's dominant sentiment model attributes cleanly. The other engine
# remains the per-call error fallback.
_PRIMARY_SENTIMENT_ENGINE = "deepseek"


def reset_sentiment_providers() -> None:
    global _PRIMARY_SENTIMENT_ENGINE
    with _PROVIDER_LOCK:
        _PROVIDER_COUNTS.clear()
    # Per-run sentiment engine flip (2026-07-13 cost tune). Qwen is pricier, so it
    # scores only ``sentiment_qwen_share`` of runs (default 10%) and DeepSeek-flash
    # the rest; the non-primary engine is the per-call error fallback. A per-RUN flip
    # (not per-call) so each engine accrues whole-run samples for the dashboard's
    # per-LLM eval and the run's dominant sentiment model attributes cleanly. Claude
    # sentiment, when explicitly enabled, keeps its own A/B ahead of the Qwen split
    # (Claude is otherwise reserved for synthesis).
    if settings.enable_claude_sentiment and random.random() < settings.llm_ab_anthropic_share:
        _PRIMARY_SENTIMENT_ENGINE = "anthropic"
        logger.info(
            f"[sentiment] A/B routing this run: primary=anthropic "
            f"(anthropic share={settings.llm_ab_anthropic_share:.0%})"
        )
    elif settings.qwen_api_key and random.random() < settings.sentiment_qwen_share:
        _PRIMARY_SENTIMENT_ENGINE = "qwen"
        logger.info(f"[sentiment] Qwen-primary this run "
                    f"(qwen share={settings.sentiment_qwen_share:.0%}; DeepSeek is the error fallback)")
    else:
        _PRIMARY_SENTIMENT_ENGINE = "deepseek"
        logger.info("[sentiment] DeepSeek-primary this run"
                    + ("" if settings.enable_claude_sentiment else " (Claude sentiment reserved for synthesis)"))


# Cross-engine resilience order appended after the preferred engine, so a single
# provider outage can never leave a ticker unscored. DeepSeek first: it is the
# cheap, funded workhorse and the engine the rest of the system falls back to.
_SENTIMENT_FALLBACK_ORDER = ("deepseek", "qwen", "anthropic")


def _sentiment_engine_order(force_engine: Optional[str]) -> list:
    """Engine try-order for one sentiment call — preferred engine first, then EVERY
    remaining engine as successive fallbacks.

    2026-07-13 cost tune: DeepSeek-flash scores ~90% of runs and Qwen ~10%
    (``sentiment_qwen_share``). Hold-review pins are HONORED — the opener engine
    re-judges its own position first (Fix #2 same-engine invariant).

    2026-07-22: a pin used to return that engine ALONE, so when the pinned engine
    was down the call fell through to ``0.0`` + "Analysis error" — a NEUTRAL
    sentiment silently fed into the hold review that decides whether to keep or
    close the position. Observed with Qwen out of OpenRouter credits: WKC, MOH,
    SEIC, GLD and AGEN (all held) scored 0.0 on a DeepSeek-primary run, having
    never tried DeepSeek. A pin is a PREFERENCE, not a suicide pact: an honest
    score from the other engine beats a fabricated neutral, and the provider
    tally still records whichever engine actually answered, so attribution stays
    truthful. This mirrors what the synthesis layer already does one level up
    (``pipeline._review``: "pinned engine X produced no review — re-judged by Y").

    Claude is reserved for synthesis: unless ``enable_claude_sentiment`` is set,
    anthropic is stripped entirely (even a ``force_engine='anthropic'`` pin
    coerces to deepseek), so Claude is never called for sentiment.
    """
    if force_engine in ("deepseek", "anthropic", "qwen"):
        order = [force_engine]
    elif _PRIMARY_SENTIMENT_ENGINE == "qwen":
        order = ["qwen", "deepseek"]
    elif _PRIMARY_SENTIMENT_ENGINE == "anthropic":
        order = ["anthropic", "deepseek"]
    else:                                    # deepseek primary → Qwen is the error fallback
        order = ["deepseek", "qwen"]
    order += [e for e in _SENTIMENT_FALLBACK_ORDER if e not in order]
    if not settings.enable_claude_sentiment:
        order = [e for e in order if e != "anthropic"] or ["deepseek"]
    return order


def _record_sentiment_provider(provider: str) -> None:
    with _PROVIDER_LOCK:
        _PROVIDER_COUNTS[provider] = _PROVIDER_COUNTS.get(provider, 0) + 1


def get_sentiment_provider_summary() -> Optional[str]:
    """Compact summary of providers used this run, e.g. 'deepseek×40, anthropic×2'.

    Returns None when no per-tickerAdd sentiment LLM call was made this run.
    """
    with _PROVIDER_LOCK:
        if not _PROVIDER_COUNTS:
            return None
        items = sorted(_PROVIDER_COUNTS.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{name}×{n}" for name, n in items)


# provider name (as tallied above) → the exact model id that provider runs.
SENTIMENT_PROVIDER_MODELS: dict = {
    "deepseek": DEEPSEEK_MODEL,
    "anthropic": HAIKU_MODEL,
    "qwen": QWEN_MODEL,
}


def get_dominant_sentiment_model() -> Optional[str]:
    """Exact model id of the provider that scored the most tickers this run.

    Stamped onto each new trade for per-LLM performance attribution. A run with
    a few fallback calls (e.g. 'deepseek×40, anthropic×2') attributes to the
    majority engine; returns None when no sentiment LLM call was made.
    """
    with _PROVIDER_LOCK:
        counted = {k: v for k, v in _PROVIDER_COUNTS.items() if k in SENTIMENT_PROVIDER_MODELS}
        if not counted:
            return None
        top = max(counted.items(), key=lambda kv: kv[1])[0]
        return SENTIMENT_PROVIDER_MODELS[top]

# Fixed seed for OpenAI-compatible APIs (DeepSeek). Combined with temperature=0
# this gives near-deterministic output across runs for the same prompt — the
# user requirement is that two pipeline runs over the same cached inputs
# produce only marginally different recommendations.
_LLM_SEED = 1729


# ── Sentiment LLM cache (latency + cost) ─────────────────────────────────────
# Caches the RAW LLM verdict per (ticker, engine, exact top-20 article set).
# The key hashes the ARTICLE SET, so a new article (or one aging out of the
# 7-day window / top-20 cutoff) changes the key and forces a fresh score —
# the news fast-lane's reactivity is preserved exactly; only a repeat scoring
# of the IDENTICAL digest is skipped (the common case: the next 30-min tick,
# and the hold-review re-scoring held tickers minutes after the main pass).
# Only the raw LLM output is cached; the count/diversity precision scales are
# recomputed live (they are pure functions of the same article set), and the
# recency decay lives in top-20 SELECTION + the digest's age labels — bounded
# by the TTL (an unchanged set is re-judged at most every TTL even so).
# Thread-safe (scoring runs on many worker threads) and persisted to disk so
# the warm cache survives the supervisor's process restarts.
_SENT_CACHE_LOCK = threading.Lock()
_SENT_CACHE: Optional[dict] = None          # key → {"raw_score","rationale","engine","ts"}
_SENT_CACHE_DIRTY = False
_SENT_CACHE_LAST_FLUSH = 0.0
_SENT_CACHE_FLUSH_EVERY_S = 20.0            # debounce disk writes


def _sent_cache_path():
    from src.data.cache import CACHE_DIR
    return CACHE_DIR / "sentiment_llm.json"


def _sent_cache_ttl_seconds() -> float:
    return max(1.0, float(getattr(settings, "sentiment_cache_ttl_minutes", 180) or 180)) * 60.0


def _sentiment_cache_key(ticker: str, engine: str, articles: List[NewsArticle]) -> str:
    """Hash of the exact article set that would be sent to the LLM (order-free)."""
    ids = sorted(
        f"{a.url or ''}|{a.source}|{a.title}|{a.published_at.isoformat()}"
        for a in articles
    )
    payload = f"{ticker.upper()}|{engine}|{SENTIMENT_PROVIDER_MODELS.get(engine, engine)}|" + "\n".join(ids)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def _sent_cache_load_locked() -> dict:
    """Load (once) + expire-prune the persisted cache. Caller holds the lock."""
    global _SENT_CACHE
    if _SENT_CACHE is None:
        data: dict = {}
        try:
            path = _sent_cache_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"[sentiment] cache load failed ({e}) — starting empty")
            data = {}
        cutoff = time.time() - _sent_cache_ttl_seconds()
        _SENT_CACHE = {k: v for k, v in data.items()
                       if isinstance(v, dict) and float(v.get("ts", 0)) >= cutoff}
    return _SENT_CACHE


def _sent_cache_flush_locked(force: bool = False) -> None:
    """Debounced disk write. Caller holds the lock."""
    global _SENT_CACHE_DIRTY, _SENT_CACHE_LAST_FLUSH
    if _SENT_CACHE is None or not _SENT_CACHE_DIRTY:
        return
    now = time.time()
    if not force and now - _SENT_CACHE_LAST_FLUSH < _SENT_CACHE_FLUSH_EVERY_S:
        return
    try:
        path = _sent_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{threading.get_ident()}")
        tmp.write_text(json.dumps(_SENT_CACHE), encoding="utf-8")
        import os
        os.replace(tmp, path)
        _SENT_CACHE_DIRTY = False
        _SENT_CACHE_LAST_FLUSH = now
    except Exception as e:
        logger.debug(f"[sentiment] cache flush failed: {e}")


def _sentiment_cache_get(key: str) -> Optional[dict]:
    if not getattr(settings, "enable_sentiment_cache", True):
        return None
    with _SENT_CACHE_LOCK:
        entry = _sent_cache_load_locked().get(key)
        if entry and time.time() - float(entry.get("ts", 0)) <= _sent_cache_ttl_seconds():
            return dict(entry)
    return None


def _sentiment_cache_put(key: str, raw_score: float, rationale: str, engine: str) -> None:
    global _SENT_CACHE_DIRTY
    if not getattr(settings, "enable_sentiment_cache", True):
        return
    with _SENT_CACHE_LOCK:
        cache = _sent_cache_load_locked()
        cache[key] = {"raw_score": raw_score, "rationale": rationale,
                      "engine": engine, "ts": time.time()}
        _SENT_CACHE_DIRTY = True
        _sent_cache_flush_locked()


def _reset_sentiment_cache_for_tests() -> None:
    """Test hook: drop the in-memory cache so a test's tmp CACHE_DIR is re-read."""
    global _SENT_CACHE, _SENT_CACHE_DIRTY
    with _SENT_CACHE_LOCK:
        _SENT_CACHE = None
        _SENT_CACHE_DIRTY = False

# Recency decay: articles older than this many hours get progressively down-weighted.
# Tightened 36h → 18h (2026-06-17) to react faster to news catalysts — a same-day
# catalyst dominates the score while yesterday's news fades quickly
# (18h=0.5x, ~24h=0.40x, ~2d=0.16x, >7d dropped entirely).
_DECAY_HALF_LIFE_HOURS = 18   # score halves every 18 hours


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


def _log_cache_hit(ticker: str, engine: str, response) -> None:
    """DEBUG-log prefix-cache hits on a non-streaming sentiment call (Qwen
    ``prompt_tokens_details.cached_tokens`` / DeepSeek ``prompt_cache_hit_tokens``)
    so the provider-side automatic caching is verifiable without spamming INFO
    (sentiment runs dozens of calls per tick). NOTE the shared sentiment prefix is
    ~600 tokens — BELOW Qwen's ~1000-token implicit-cache minimum — so Qwen hits
    here are expected to be rare/zero; the app-level article-digest verdict cache
    is the layer that actually eliminates repeat sentiment cost. Fail-soft."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached:
            logger.debug(f"{ticker} sentiment {engine} prefix-cache hit: "
                         f"{cached}/{getattr(usage, 'prompt_tokens', '?')} input tok")
    except Exception:
        pass


def _get_qwen() -> OpenAI | None:
    """OpenAI-compatible Qwen client (DashScope). None when no key → the caller
    falls through to the next engine in the order (DeepSeek)."""
    global _qwen_client
    if not settings.qwen_api_key:
        return None
    if _qwen_client is None:
        _qwen_client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
    return _qwen_client


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
    try:
        data = json.loads(raw)
        score = max(-1.0, min(1.0, float(data["score"])))
        rationale = str(data["rationale"])
        return score, rationale
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Salvage a truncated/malformed response (e.g. DeepSeek hitting the
        # 256-token cap mid-rationale → invalid JSON, observed for XBI). The score
        # is the only field that feeds the aggregator, so recover it by regex
        # rather than lose the signal to a 0.0 fallback — especially valuable when
        # the other engine is rate-limited and can't be tried.
        m = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
        if not m:
            raise
        score = max(-1.0, min(1.0, float(m.group(1))))
        rm = re.search(r'"rationale"\s*:\s*"(.*?)(?:"\s*[},]|$)', raw, re.DOTALL)
        rationale = rm.group(1).strip() if rm else "Rationale unavailable (truncated response)."
        return score, rationale


def _recency_weight(article: NewsArticle) -> float:
    """
    Exponential decay based on article age.
    24h → ~0.87,  36h → 0.50,  72h → 0.25,  7d → ~0.06
    Articles older than 7 days are excluded entirely (weight ≈ 0).

    Time-quantised to the hour: the "now" used for age calculation is bucketed
    to the top of the current UTC hour so two pipeline runs within the same
    hour produce identical recency weights. Without this, two back-to-back
    runs could flip the top-20 article cutoff on borderline-stale articles
    and feed the LLM a different digest, producing different sentiment scores.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
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


# Provider sentiment LABEL → unit score (scaled by provider_sentiment_magnitude).
_PROVIDER_LABEL_UNIT = {
    "positive": 1.0, "bullish": 1.0, "buy": 1.0,
    "negative": -1.0, "bearish": -1.0, "sell": -1.0,
    "neutral": 0.0, "hold": 0.0,
}


def _provider_sentiment_score(ticker: str,
                              fresh_articles: List[NewsArticle]) -> Optional[tuple[float, str]]:
    """Per-ticker news score derived from PRE-COMPUTED provider sentiment
    (e.g. Polygon insights), skipping the LLM. Returns ``(score, rationale)`` or
    ``None`` to defer to the LLM (flag off, or too few provider-scored articles).

    Mirrors the LLM path's precision adjustments (recency-weighted blend, then
    article-count × source-diversity scaling) so the two are comparable."""
    if not settings.enable_provider_sentiment:
        return None
    tkr = ticker.upper()
    mag = float(settings.provider_sentiment_magnitude)
    scored = []
    for a in fresh_articles:
        label = (a.provider_insights or {}).get(tkr) or (a.provider_insights or {}).get(ticker)
        if not label:
            continue
        unit = _PROVIDER_LABEL_UNIT.get(str(label).strip().lower())
        if unit is None:
            continue
        scored.append((a, unit * mag))
    if len(scored) < int(settings.provider_sentiment_min_articles):
        return None

    wsum = sum(_recency_weight(a) for a, _ in scored)
    if wsum <= 0:
        return None
    raw = sum(_recency_weight(a) * s for a, s in scored) / wsum
    arts = [a for a, _ in scored]
    precision = _article_count_scale(len(arts)) * _source_diversity_scale(arts)
    score = round(raw * precision, 3)
    src = next((a.provider_sentiment_source for a in arts if a.provider_sentiment_source), "provider")
    pos = sum(1 for _, s in scored if s > 0)
    neg = sum(1 for _, s in scored if s < 0)
    rationale = (f"Provider sentiment ({src}): {len(arts)} pre-scored article(s) "
                 f"({pos} positive / {neg} negative) → {score:+.2f}; LLM scorer skipped.")
    return score, rationale


# FIXED, ticker-FREE instruction prefix — identical for every ticker, so it forms
# a shared prefix that DeepSeek auto-caches across the ~40 per-ticker calls in a
# run (and Anthropic would cache via cache_control IF it met the per-model minimum
# — it does NOT: this prefix is ~600 tok, far below Haiku 4.5's 4096-tok minimum,
# so cache_control is a silent no-op on Haiku; the real saving here is DeepSeek's
# automatic prefix caching). The per-ticker name + news digest go in the suffix.
_SENTIMENT_PREFIX = """You are an elite buy-side analyst with 25 years of experience at top-tier hedge funds. You have an exceptional ability to identify the exact news catalysts that move stock prices — your track record places you in the top 0.1% of market professionals worldwide.

Your task: analyse the recent news for THE TARGET TICKER (specified at the end) and score the SHORT-TERM directional impact (1–5 trading days) with surgical precision.

PRECISION MANDATE — false positives are more costly than false negatives:
- Score 0.0 unless you identify a SPECIFIC, IDENTIFIABLE catalyst with a clear price mechanism. Vague positive/negative sentiment does NOT count.
- Reserve scores above ±0.7 for high-impact, unambiguous catalysts: earnings beats/misses with guidance change, FDA approvals/rejections, M&A announcements, major regulatory actions, CEO departure, bankruptcy risk.
- Scores ±0.3–0.6: clear but moderate catalyst — analyst upgrade/downgrade with PT, supply chain disruption, contract win/loss.
- Scores ±0.1–0.2: minor directional catalyst — relevant but unlikely to move price significantly.
- Score 0.0 if: news is noise, recycled information, or there is no clear directional catalyst.
- Score ONLY news materially about the target ticker itself — ignore passing mentions, sector round-ups, or macro pieces that merely list it; those are not ticker-specific catalysts.
- If catalysts conflict, NET them by magnitude and recency — do not mechanically average to 0; the dominant, most recent, highest-impact catalyst drives the sign.
- Recency matters: articles marked "1h ago" or "6h ago" carry much more weight than "3d ago" or "5d ago".
- When in doubt, output 0.0. A missed opportunity is better than a wrong call.

SOURCE WEIGHTING — the digest mixes hard catalysts with soft sentiment; weight them differently. Each article is tagged "[source | age]":
- HARD sources move price directly and can justify scores up to ±1.0: SEC 8-K filings, earnings/EPS surprises, analyst rating & price-target changes, and primary financial news (M&A, FDA, guidance, legal/regulatory).
- SOFT sources are attention/positioning, NOT catalysts, and cap at ±0.2 on their own: Reddit/WSB social sentiment, Google Trends search spikes, short-interest shifts. They are corroborating color, never a standalone thesis. A soft source ALIGNED with a hard catalyst modestly amplifies conviction; if it CONTRADICTS the hard catalyst, discount it.

Respond with a JSON object with exactly these fields:
- "score": float between -1.0 (very bearish) and +1.0 (very bullish), 0.0 is neutral
- "rationale": one to three sentences explaining (1) the specific catalyst and (2) the exact price mechanism. If score is 0.0, state why no actionable catalyst was identified.

Examples:
{"score": 0.6, "rationale": "Q3 earnings beat with raised FY guidance (hard catalyst) reprices forward estimates over the next few sessions."}
{"score": 0.0, "rationale": "Only a Reddit mention spike and a generic sector round-up; no ticker-specific hard catalyst identified."}"""


def _anthropic_user_content(prefix: str, suffix: str, model: str):
    """Anthropic ``content`` for a sentiment call: a cache_control prefix block +
    variable suffix when caching is on AND the prefix meets the model's minimum
    (Haiku 4.5 = 4096 tok). The sentiment prefix is ~600 tok, so this currently
    returns a single concatenated string (cache_control would silently no-op);
    kept so it caches automatically if the prompt ever grows past the minimum."""
    if settings.enable_prompt_caching:
        min_tokens = 4096 if "haiku" in (model or "") else 1024
        if len(prefix) >= min_tokens * 4:   # ~4 chars/token
            return [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": suffix},
            ]
    return prefix + suffix


def analyse_sentiment(ticker: str, articles: List[NewsArticle],
                      force_engine: Optional[str] = None) -> tuple[float, str]:
    """
    Score news sentiment for a ticker with precision controls applied.

    ``force_engine`` ('deepseek' | 'anthropic') pins scoring to exactly that
    engine with NO cross-engine fallback — used by the opener-pinned hold-review
    so a position is always re-scored by the same sentiment engine that opened it
    (apples-to-apples). On a forced-engine failure the score is the usual
    (0.0, "error") rather than silently switching engines. ``None`` keeps the
    per-run A/B order (`_PRIMARY_SENTIMENT_ENGINE` first, the other as fallback).

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

    # Provider-sentiment hybrid (latency win): when enough fresh articles already
    # carry a provider sentiment (e.g. Polygon insights), score from those and
    # skip the LLM call entirely. Bypassed for force_engine (the opener-pinned
    # hold-review must re-judge with its OWN LLM engine for apples-to-apples).
    if force_engine is None:
        provider = _provider_sentiment_score(ticker, fresh_articles)
        if provider is not None:
            _record_sentiment_provider("provider")
            logger.info(f"{ticker} provider_sentiment={provider[0]:+.2f} (LLM scorer skipped)")
            return provider

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

    # Ticker-free fixed prefix (shared → DeepSeek auto-caches it across tickers) +
    # per-ticker variable suffix. The prefix carries ALL instructions/examples; the
    # suffix carries only the target ticker and the news digest.
    suffix = (f"\n\nTARGET TICKER: {ticker}\n\n<news>\n{digest}\n</news>\n\n"
              "Respond with JSON only, no markdown.")
    prompt = _SENTIMENT_PREFIX + suffix

    raw_score = None
    rationale = "Analysis unavailable."

    # Primary engine per the run's A/B flip (reset_sentiment_providers); the
    # other provider remains the error fallback. Determinism per engine:
    # temperature=0 (+ a stable seed on DeepSeek; Anthropic exposes no seed)
    # so two runs scoring the same digest on the same engine agree.
    order = _sentiment_engine_order(force_engine)

    # Provenance rule: only UNforced calls tally into the run's provider counts —
    # a forced (opener-pinned hold-review) call is not "the run's sentiment
    # engine", and with the review branch running concurrently with the main
    # scoring pass it would otherwise pollute llm_sentiment_provider mid-run.
    _tally = force_engine is None

    # Cache: identical (ticker, engine, article set) → reuse the raw LLM verdict.
    # temperature=0 makes the call deterministic anyway; this skips the latency.
    cache_key = _sentiment_cache_key(ticker, order[0], to_score)
    cached = _sentiment_cache_get(cache_key)
    if cached is not None:
        raw_score = float(cached["raw_score"])
        rationale = str(cached.get("rationale") or "Rationale unavailable (cached).")
        if _tally:
            _record_sentiment_provider(str(cached.get("engine") or order[0]))
        logger.debug(
            f"{ticker} raw_sentiment={raw_score:+.2f} "
            f"({cached.get('engine')}, {len(to_score)} articles, cached)"
        )
    last_err: Exception | None = None
    # Maximum-thinking policy: reason on every call. Qwen bills reasoning tokens
    # separately, so its answer cap stays small; DeepSeek shares max_tokens with
    # thinking, so give it headroom. thinking_budget is OMITTED → DashScope defaults
    # it to the model's maximum chain-of-thought (true "maximum thinking").
    _max_think = settings.llm_max_thinking
    _think_mt = (settings.llm_thinking_sentiment_max_tokens
                 if _max_think else _SENTIMENT_MAX_TOKENS)
    for engine in (order if raw_score is None else []):
        try:
            if engine == "qwen":
                qwen = _get_qwen()
                if qwen is None:            # no API key — try the next engine
                    continue
                from src.analysis.qwen_api import thinking_body
                response = qwen.chat.completions.create(
                    model=settings.qwen_model,
                    max_tokens=_SENTIMENT_MAX_TOKENS,   # answer only (thinking billed separately)
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    seed=_LLM_SEED,
                    extra_body=thinking_body(_max_think),   # route dialect (qwen_api)
                )
                _log_cache_hit(ticker, "qwen", response)
                raw_score, rationale = _parse_response(response.choices[0].message.content.strip())
            elif engine == "deepseek":
                deepseek = _get_deepseek()
                if deepseek is None:        # no API key — try the other engine
                    continue
                response = deepseek.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    max_tokens=_think_mt,   # DeepSeek thinking shares this budget
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    seed=_LLM_SEED,
                    extra_body=(_DEEPSEEK_THINKING_ON if _max_think else _DEEPSEEK_THINKING_OFF),
                )
                _log_cache_hit(ticker, "deepseek", response)
                raw_score, rationale = _parse_response(response.choices[0].message.content.strip())
            else:
                client = _get_haiku()
                message = client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=_SENTIMENT_MAX_TOKENS,
                    messages=[{"role": "user",
                               "content": _anthropic_user_content(_SENTIMENT_PREFIX, suffix, HAIKU_MODEL)}],
                    temperature=0,
                )
                raw_score, rationale = _parse_response(message.content[0].text.strip())
            logger.info(f"{ticker} raw_sentiment={raw_score:+.2f} ({engine}, {len(to_score)} articles)")
            if _tally:
                _record_sentiment_provider(engine)
            _sentiment_cache_put(cache_key, raw_score, rationale, engine)
            break
        except Exception as e:
            last_err = e
            logger.warning(f"{ticker} sentiment via {engine} failed: {e}")

    if raw_score is None:
        logger.error(f"Sentiment analysis failed for {ticker}: {last_err}")
        if _tally:
            _record_sentiment_provider("none")
        return 0.0, f"Analysis error: {last_err}"

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

    tkr = ticker.upper()
    relevant = [
        a for a in articles
        # Per-ticker feeds (yfinance) tag the symbol explicitly → exact match,
        # no fuzzy keyword needed; general-market feeds fall back to keyword.
        if tkr in (getattr(a, "tickers", None) or [])
        or any(kw in (a.title + a.summary).lower() for kw in keywords)
    ]
    # Precision: return empty list when too few relevant articles found.
    # Do NOT fall back to all articles — that injects irrelevant noise.
    return relevant if len(relevant) >= 2 else []
