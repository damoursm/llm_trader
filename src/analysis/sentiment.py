"""LLM-based sentiment analysis of news articles.

Engines: DeepSeek V4-Flash (deepseek-v4-flash, non-thinking) and Claude Haiku.
Which one is PRIMARY is flipped once per run (settings.llm_ab_anthropic_share,
default 50/50) so both accumulate comparable samples for the dashboard's
per-LLM evaluation; the other engine is the error fallback.

Precision controls:
  - Recency decay: articles weighted by age before scoring (fresh=1.0x, 18h=0.5x, ~2d=0.16x, >7d dropped)
  - Evidence-mass scaling: the raw verdict is scaled by Σ per-article recency
    weights (continuous since 2026-08-14 — 3 fresh articles ≠ 3 stale ones)
  - Source diversity: if all articles come from a single source, apply a confidence penalty
  - Relevance fallback fix: if <2 relevant articles found, return [] (not all articles)
  - Prompt (v3, `_SENT_PROMPT_VERSION`-salted into the verdict cache key):
    band-then-placement magnitude rubric, priced-in/remaining-move check,
    4-tier source-credibility ladder, rationale emitted before the score
"""

import hashlib
import json
import math
import random
import re
import threading
import time
import anthropic
from collections import OrderedDict
from datetime import datetime, timezone
from openai import OpenAI
from loguru import logger
from typing import List, Optional, Tuple
from config import settings
from src.analysis import local_llm
from src.models import NewsArticle


# The run every shadow verdict is stamped with (set once per tick by the
# pipeline via set_current_run; None outside a run - a CLI or a test).
_CURRENT_RUN_ID = None

_deepseek_client = None
_haiku_client = None
_qwen_client = None
_local_client = None

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

# Per-run tally of WHY tallied calls failed: engine → [failed calls, last error].
# The provider tally says THAT every call fell to "none"; this says which engine
# refused and with what, so the health alert names the real fault instead of a
# fixed guess (2026-09-21: a local-only tier sat dead for 20 ticks after a reboot
# while the alert said to top up hosted API credits).
_ENGINE_ERRORS: dict = {}

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
        _ENGINE_ERRORS.clear()
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
    # NOTE (2026-09-03): these shares are sampled SEQUENTIALLY, so they are not
    # independent - a later branch sees only the runs the earlier ones declined.
    # With local at 0.30 and qwen at S, qwen's realised share is S x 0.70, not S.
    # Harmless while sentiment_qwen_share is 0.0 (the live setting), but set both
    # above zero and the qwen arm silently under-samples. Fix by normalising the
    # draw across the active shares, not by reordering the branches.
    elif settings.enable_local_llm and random.random() < settings.sentiment_local_share:
        _PRIMARY_SENTIMENT_ENGINE = "local"
        _tier = ("hosted engines are the fallback"
                 if getattr(settings, "enable_hosted_sentiment_engines", True)
                 else "LOCAL-ONLY — no hosted fallback, this server is the whole tier")
        logger.info(f"[sentiment] LOCAL-primary this run "
                    f"({settings.local_sentiment_model} @ {settings.local_sentiment_base_url}; "
                    f"share={settings.sentiment_local_share:.0%}; {_tier})")
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
# `local` is LAST among the hosted engines by default but is the one that cannot
# be unfunded — the 2026-09-01 billing outage took DeepSeek AND Qwen together and
# left the 0.40-weight `news` method reading exactly 0.0000 for a full day, which
# no amount of cross-HOSTED-engine ordering could have prevented.
_SENTIMENT_FALLBACK_ORDER = ("deepseek", "qwen", "local", "anthropic")


def _fallback_order() -> tuple:
    """`_SENTIMENT_FALLBACK_ORDER` minus the engines that are switched OFF.

    `local` is gated on its flag rather than appended unconditionally (the
    convention the API engines follow, where a missing key just makes the loop
    `continue`): the flag is a hard on/off, so with `enable_local_llm=False`
    the try-order — and therefore this whole module — is byte-identical to
    before the local engine existed."""
    order = _SENTIMENT_FALLBACK_ORDER
    if not settings.enable_local_llm:
        order = tuple(e for e in order if e != "local")
    if not getattr(settings, "enable_hosted_sentiment_engines", True):
        # HOSTED ENGINES OFF (2026-09-18): keep only the self-hosted one. A tier
        # that returns HTTP 402 on every call is not a fallback — it is two dead
        # round trips per failure and a log that buries the real warnings.
        # Guarded so an accidental local-off + hosted-off never empties the
        # order: an empty try-order fabricates a neutral 0.0 in the 0.40-weight
        # `news` method, which is the one failure this module must never have.
        order = tuple(e for e in order if e == "local") or _SENTIMENT_FALLBACK_ORDER
    return order


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
    if force_engine in ("deepseek", "anthropic", "qwen", "local"):
        order = [force_engine]
    elif _PRIMARY_SENTIMENT_ENGINE == "local":
        order = ["local", "deepseek"]
    elif _PRIMARY_SENTIMENT_ENGINE == "qwen":
        order = ["qwen", "deepseek"]
    elif _PRIMARY_SENTIMENT_ENGINE == "anthropic":
        order = ["anthropic", "deepseek"]
    else:                                    # deepseek primary → Qwen is the error fallback
        order = ["deepseek", "qwen"]
    order += [e for e in _fallback_order() if e not in order]
    if not settings.enable_claude_sentiment:
        order = [e for e in order if e != "anthropic"] or ["deepseek"]
    if not settings.enable_local_llm:
        order = [e for e in order if e != "local"] or ["deepseek"]
    if (not getattr(settings, "enable_hosted_sentiment_engines", True)
            and settings.enable_local_llm):
        # Applied AFTER the pin, on purpose: a `force_engine="deepseek"` pin —
        # from a hold review or a shadow arm — must coerce to local rather than
        # dead-end, exactly as an anthropic pin coerces when Claude is off. A
        # pin is a preference, not a suicide pact (2026-07-22).
        order = ["local"]
    return order


def _record_sentiment_provider(provider: str) -> None:
    with _PROVIDER_LOCK:
        _PROVIDER_COUNTS[provider] = _PROVIDER_COUNTS.get(provider, 0) + 1


def _record_engine_error(engine: str, err: Exception) -> None:
    with _PROVIDER_LOCK:
        entry = _ENGINE_ERRORS.setdefault(engine, [0, ""])
        entry[0] += 1
        entry[1] = str(err)[:200]


def get_sentiment_engine_errors() -> dict:
    """``{engine: (failed_calls, last_error)}`` over this run's tallied calls.

    Empty when no engine RAISED — which includes a run where no engine was even
    available to try (every client unconfigured), so an empty dict on a run whose
    calls all fell to "none" means exactly that."""
    with _PROVIDER_LOCK:
        return {e: (n, msg) for e, (n, msg) in _ENGINE_ERRORS.items()}


def set_current_run(run_id: Optional[str]) -> None:
    """Stamp the run every shadow verdict belongs to (called once per tick)."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = str(run_id) if run_id else None


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
    # `local` is deliberately NOT a static entry — see sentiment_model_for().
    "local": None,
}


def sentiment_model_for(engine: str) -> str:
    """Exact model id for a sentiment engine — the string the ledger stamps.

    `local` resolves at CALL time from `local_sentiment_model`, unlike the hosted
    engines whose ids are import-time constants, for two reasons:

      * PROVENANCE. Swapping the self-hosted model is the expected operation
        here (that is the whole point of running one), and a stale import-time
        id would silently label the new model's verdicts as the old model's.
        The `local/` prefix also keeps a self-hosted qwen3:8b from ever being
        confused with the hosted qwen3.7-plus in the per-LLM eval, which groups
        on exactly this string.
      * CACHE CORRECTNESS. `_sentiment_cache_key` salts on this id, so
        resolving it dynamically means changing the model INVALIDATES its
        cached verdicts — the same discipline `_SENT_PROMPT_VERSION` enforces
        for a prompt edit. A static id would keep serving the previous model's
        answers for the TTL.
    """
    if engine == "local":
        return f"local/{settings.local_sentiment_model}"
    return SENTIMENT_PROVIDER_MODELS.get(engine) or engine


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
        return sentiment_model_for(top)

def _local_extra_body() -> dict:
    """`local_sentiment_extra_body` parsed — the local server's own dialect for
    switching reasoning OFF (see the setting for the 2026-09-03 probe).

    Load-bearing, not a tuning knob: with reasoning ON, qwen3:8b spent the whole
    256-token answer budget on its chain and returned content="" with
    finish_reason=length. That is a LOST verdict, and because
    `analyse_sentiment` scores an unparseable answer as 0.0, it would have been
    lost SILENTLY into the 0.40-weight news method.

    Fail-soft to {} on bad JSON: a malformed knob must not take the engine down,
    and the `<think>`-stripping in `_parse_response` still covers the inline
    case."""
    raw = (settings.local_sentiment_extra_body or "").strip()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
        return body if isinstance(body, dict) else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"[sentiment] local_sentiment_extra_body is not valid JSON "
                       f"({raw[:60]!r}) — sending none")
        return {}


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


# Salted into the cache key so a PROMPT change invalidates cached raw verdicts.
# Found 2026-08-14: the key hashed only (ticker, engine, article set), so after
# a prompt edit the cache kept serving verdicts produced by the OLD prompt for
# up to the TTL — two tickers scored seconds apart could be on different
# scoring standards with nothing recording which. Bump on any _SENTIMENT_PREFIX
# change that could move the score.
# v3-2026-08-15: derivation-based two-decimal placement (v2's example values
# 0.47/−0.62/0.71 had become the modal raw verdicts), priced-in/remaining-move
# check, company-PR + aggregator source tiers, rationale-before-score order.
# v4-2026-08-15: + "catalyst" output field (NEWS_CATALYST_TYPES) so every
# verdict lands in the news-event dataset typed; score semantics unchanged.
# v5-2026-09-04: the ZERO changed meaning. v4's PRECISION MANDATE ("score 0.0
# unless you identify a SPECIFIC, IDENTIFIABLE catalyst … when in doubt,
# output 0.0") made 0.0 the modal verdict (73–79% of calls) — but a 0.0 is an
# ABSTENTION in the rank-consumed combine (zeros are never ranked), so every
# soft read silently removed its ticker from the 0.40-weight method. v5 tells
# the model exactly that, reserves 0.0 for "no information about the company"
# (or a genuinely nil remaining move), routes uncertainty into MAGNITUDE via a
# new ±0.01–0.10 LEAN band, and drops the worked examples (their numbers had
# become modal outputs twice) for a placeholder-only format skeleton.
# v6-2026-09-04 (v5 never went live): the abstention cases NARROWED. Paired on
# the v5 zeros, three legitimate-looking families remained — routine
# company-specific items (13F/13D-G stakes, Form 4 / 10b5-1 sales, dividend
# declarations, index changes), sector/index FUNDS whose digest is "about the
# sector", and peer/customer read-throughs — each of which v5's "score ONLY the
# company itself" clause sent to 0.0. v6 says those ARE information with a
# conventional direction (LEAN/MINOR, typed — a typed lean is a labelled event
# `catalyst_tilt` can re-orient; a 0.0 teaches it nothing), flags a FUND target
# in the per-ticker header so its holdings count as the company, and names the
# company beside the symbol (articles say "Antero", not "AR"; the header is
# salted into the verdict cache key). 0.0 is now: nothing in the digest
# connects to the target at all, or a genuinely nil remaining move.
_SENT_PROMPT_VERSION = "v6-2026-09-04"


def _sentiment_cache_key(ticker: str, engine: str, articles: List[NewsArticle],
                        extra: str = "") -> str:
    """Hash of the exact article set that would be sent to the LLM (order-free),
    salted with the prompt version, the engine's model id and ``extra`` — the
    per-ticker prompt header (company name + fund flag), so a name resolving
    later, or a registrant rename, re-scores rather than serving a verdict
    formed under a different prompt."""
    ids = sorted(
        f"{a.url or ''}|{a.source}|{a.title}|{a.published_at.isoformat()}"
        for a in articles
    )
    payload = (f"{ticker.upper()}|{engine}|{sentiment_model_for(engine)}|"
               f"{_prompt_pair()[1]}|{extra}|" + "\n".join(ids))
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def digest_id_for(ticker: str, articles: List[NewsArticle]) -> str:
    """ENGINE-FREE id of an article digest (2026-09-06).

    The same sorted per-article id strings `_sentiment_cache_key` hashes,
    WITHOUT the engine / model / prompt-version / header salts — so the two
    engines scoring one digest share one id, and a later catalyst repair keys
    on the digest the model actually typed rather than on which engine typed
    it. The verdict cache keeps its salted key; this id is the JOIN key
    (`signals.news_digest_id`, `sentiment_shadow.digest_id`,
    `sentiment_digests`, `catalyst_repairs`)."""
    ids = sorted(
        f"{a.url or ''}|{a.source}|{a.title}|{a.published_at.isoformat()}"
        for a in articles
    )
    payload = f"{ticker.upper()}|" + "\n".join(ids)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


# ── Digest store ─────────────────────────────────────────────────────────────
# The verdict cache stores a HASH of the article set, not the articles, so a
# past verdict's exact input could never be replayed (which is why the engine
# bake-off could not be re-run on cached digests). The digest store keeps the
# TEXT the scorer saw, keyed by the engine-free id above, so a catalyst label
# can be re-judged later on the SAME digest (`catalyst_repair`). Buffered
# in-process and drained by `_persist_run` (fail-soft, idempotent per id).
_DIGEST_LOCK = threading.Lock()
_DIGEST_ROWS: list = []
_DIGEST_SEEN: "OrderedDict[str, None]" = OrderedDict()   # bounded per-process dedupe
_DIGEST_SEEN_MAX = 20_000


def _record_digest(digest_id: str, ticker: str, articles: List[NewsArticle],
                   digest_text: str) -> None:
    """Buffer one digest for persistence. Called on the cache-hit path too:
    a verdict served from cache still needs its digest on disk, because the
    cache predates the store."""
    if not getattr(settings, "enable_sentiment_digest_store", True):
        return
    try:
        with _DIGEST_LOCK:
            if digest_id in _DIGEST_SEEN:
                return
            _DIGEST_SEEN[digest_id] = None
            while len(_DIGEST_SEEN) > _DIGEST_SEEN_MAX:
                _DIGEST_SEEN.popitem(last=False)
            _DIGEST_ROWS.append({
                "digest_id": digest_id,
                "ticker": (ticker or "").strip().upper(),
                "run_id": _CURRENT_RUN_ID,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_articles": len(articles),
                "digest_text": digest_text,
                "articles_json": json.dumps([{
                    "url": a.url, "source": a.source, "title": a.title,
                    "published_at": a.published_at.isoformat(),
                    "summary": (a.summary or "")[:400],
                } for a in articles], ensure_ascii=False),
            })
    except Exception as e:                    # noqa: BLE001 - store is measurement only
        logger.debug(f"[sentiment-digest] {ticker} not buffered: {e}")


def pop_sentiment_digest_rows() -> list:
    """Drain the buffered digests (called by `_persist_run`)."""
    with _DIGEST_LOCK:
        rows, _DIGEST_ROWS[:] = list(_DIGEST_ROWS), []
    return rows


def _reset_digest_state() -> None:
    """Test hook: forget every digest this process has buffered or seen."""
    with _DIGEST_LOCK:
        _DIGEST_ROWS.clear()
        _DIGEST_SEEN.clear()


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


def _sentiment_cache_put(key: str, raw_score: float, rationale: str, engine: str,
                         catalyst: Optional[str] = None) -> None:
    global _SENT_CACHE_DIRTY
    if not getattr(settings, "enable_sentiment_cache", True):
        return
    with _SENT_CACHE_LOCK:
        cache = _sent_cache_load_locked()
        cache[key] = {"raw_score": raw_score, "rationale": rationale,
                      "catalyst": catalyst, "engine": engine, "ts": time.time()}
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


def _warn_if_empty_local(ticker: str, response) -> None:
    """Shout when the local model returns nothing.

    The 2026-09-03 failure: reasoning left ON, the chain consumed all 256 answer
    tokens, `content` came back "" with finish_reason="length" and the reasoning
    in a side field. `_parse_response` then raises, the engine loop falls to the
    next provider, and if none answers the ticker scores 0.0 — a real verdict
    replaced by a neutral one, with nothing in the log saying so. This makes that
    exact shape loud, because it is a CONFIG error (wrong dialect / too small an
    answer cap), not a transient."""
    try:
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if content:
            return
        reasoned = bool((choice.message.model_extra or {}).get("reasoning"))
        logger.error(
            f"[sentiment] local model returned an EMPTY verdict for {ticker} "
            f"(finish_reason={choice.finish_reason}"
            f"{', all tokens went to REASONING' if reasoned else ''}) — check "
            f"local_sentiment_extra_body (the server's no-reasoning dialect) and "
            f"the answer cap; this is a config error, not a transient"
        )
    except Exception:
        pass


def _get_local() -> OpenAI | None:
    """OpenAI-compatible client for the SELF-HOSTED model (Ollama / llama.cpp /
    vLLM). Returns None when the local engine is off or unconfigured, so the
    engine loop falls through to the next provider exactly like a missing key."""
    global _local_client
    if not settings.enable_local_llm or not settings.local_sentiment_base_url:
        return None
    if _local_client is None:
        _local_client = OpenAI(
            api_key=settings.local_sentiment_api_key or "local",
            base_url=settings.local_sentiment_base_url,
            timeout=settings.local_sentiment_timeout_seconds,
            max_retries=0,      # a local box does not need SDK backoff; fail to the next engine
        )
    return _local_client


def _log_cache_hit(ticker: str, engine: str, response) -> None:
    """DEBUG-log prefix-cache hits on a non-streaming sentiment call (Qwen
    ``prompt_tokens_details.cached_tokens`` / DeepSeek ``prompt_cache_hit_tokens``)
    so the provider-side automatic caching is verifiable without spamming INFO
    (sentiment runs dozens of calls per tick). NOTE the shared sentiment prefix measures
    **2,176 tokens** (DeepSeek's own count on this exact string, 2026-09-05 — it
    was ~600 before the v3-v6 rubric grew), so it is now ABOVE Qwen's ~1000-token
    implicit-cache minimum and Qwen hits are expected rather than rare; the
    app-level article-digest verdict cache is still the layer that eliminates
    repeat sentiment cost outright. Fail-soft."""
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


# Every numeric field the model may emit. v7d drops "score" and adds
# "catalyst_score"/"priced_in", so a repair anchored on "score" alone would go
# INERT on the exact prompt that replaced it — and a `+0.15` would again be lost
# as an abstention inside the 0.40-weight `news` method. The alternation is
# still anchored on a quoted field name, so it can never rewrite a `+` inside
# the rationale text ("revenue +12% yoy").
_NUM_FIELDS = '"(?:score|catalyst_score|priced_in)"'
_SCORE_PLUS_RE     = re.compile(r'(' + _NUM_FIELDS + r'\s*:\s*)\+')
_SCORE_BARE_DOT_RE = re.compile(r'(' + _NUM_FIELDS + r'\s*:\s*-?)(\.\d)')


def _repair_score_number(raw: str) -> str:
    """Make two NON-JSON number forms parseable, on the `score` field only.

    JSON forbids a leading `+` and requires a digit before the decimal point,
    but the rubric asks for a float "between -1.0 and +1.0" — and the local
    qwen3:8b duly emitted `"score": +0.15`. That raised, ALSO defeated the
    salvage regex below (which allowed `-` but not `+`), and so lost an
    otherwise perfect verdict to a neutral 0.0 inside the 0.40-weight `news`
    method. Measured on the 2026-09-03 acceptance run: **5 of 25 tickers**, and
    it read as the model abstaining rather than as a parser bug.

    Anchored on `"score"` rather than on any `:` so it can never rewrite a `+`
    that legitimately appears inside the rationale text ("revenue +12% yoy").
    Inert on already-valid JSON, so no engine's output changes and this is a
    parser repair, not a scorer change.
    """
    if not any(f in raw for f in ('"score"', '"catalyst_score"', '"priced_in"')):
        return raw
    if "+" in raw:
        raw = _SCORE_PLUS_RE.sub(r"\1", raw)
    return _SCORE_BARE_DOT_RE.sub(r"\g<1>0\2", raw)


PRICED_IN_MAX = 1.5         # past this an "overshoot" is a mis-stated fraction
_MIN_LEAN = 0.01            # the LEAN band floor: soft must never become silent


def apply_priced_in(catalyst_score: float, priced_in: float) -> float:
    """``catalyst_score x (1 - priced_in)`` — the v7d score, computed by US.

    The whole point of the decomposition is that this multiplication is not
    optional, so it lives here rather than in the prompt. Four rules, each
    load-bearing:

    * A zero catalyst is abstention case (a) and stays exactly 0.0 whatever the
      fraction says — no catalyst, no read.
    * A fraction AT 1.0 is abstention case (b), the genuinely nil remaining move
      (an all-cash target pinned at the offer), and is the ONLY way a real
      catalyst reaches exactly 0.0. Everything else keeps a signed lean: a 0.0
      is an ABSTENTION in the rank-consumed combine, which removes the ticker
      from the cross-section entirely, so a merely-well-priced read must land at
      `_MIN_LEAN`, not at nothing. That is the v5/v6 lesson restated in
      arithmetic.
    * The fraction may exceed 1.0, which flips the sign — the overshoot the v6
      prose asked for ("after an outsized one-day spike that residual is small
      and often NEGATIVE") and could not express. Clamped at `PRICED_IN_MAX`,
      and negative fractions (a model inventing "not yet priced, so -0.2") are
      clamped to 0.0 rather than used to AMPLIFY a catalyst past its band.
    * Six decimals, matching what the news family persists: the rank transform
      orders on this value, so a coarse round here re-merges what the two
      fields separated. Note the product of two two-decimal numbers is finer
      than either — a side benefit on an engine that quantises its own output.
    """
    try:
        cs = float(catalyst_score)
        pi = float(priced_in)
    except (TypeError, ValueError):
        raise ValueError("catalyst_score/priced_in not numeric")
    if cs != cs or pi != pi:
        raise ValueError("catalyst_score/priced_in NaN")
    cs = max(-1.0, min(1.0, cs))
    if cs == 0.0:
        return 0.0
    pi = max(0.0, min(PRICED_IN_MAX, pi))
    if pi == 1.0:
        return 0.0
    val = cs * (1.0 - pi)
    if abs(val) < _MIN_LEAN:
        val = _MIN_LEAN if val > 0 else -_MIN_LEAN
    return round(max(-1.0, min(1.0, val)), 6)


_DIRECTION_RE = re.compile(r'"direction"\s*:\s*"?(UP|DOWN|NONE)"?', re.I)


# The setting's own default, resolved once — see `apply_catalyst_cap`.
try:
    _CAP_LIMIT_DEFAULT = float(type(settings).model_fields["catalyst_cap_limit"].default)
except Exception:                                               # noqa: BLE001
    _CAP_LIMIT_DEFAULT = 0.03


def _cap_classes() -> frozenset:
    return frozenset(x.strip().lower() for x in
                     str(getattr(settings, "catalyst_cap_classes", "") or "").split(",")
                     if x.strip())


def apply_catalyst_cap(score: float, catalyst: Optional[str]) -> float:
    """Hold the named catalyst classes to the LEAN band.

    MEASURED (2026-09-11, 67 days / 5,959 labelled+typed rows, per-day pivot IC
    paired against the uncapped baseline, day-clustered): capping `analyst` to
    +/-0.10 scores **+0.0121, t +2.46, same-sign halves** — the one class-level
    intervention of eight tested that clears the house bar. Analyst reads are
    weak on the tape (oriented -0.882 pp, hit 47.1% over 745 rows), which is the
    same conclusion `memory/prompt-gate-selector-2026-08` reached about analyst
    price-target moves from the other direction.

    Sign is NEVER touched — only magnitude — so this is a conviction limit, not
    a direction claim. `analyst` alone ships; `legal_regulatory` (+0.0060,
    t +1.64) and `ma_deal` (which did NOT replicate: +0.0029 live, -0.0023
    backfill) are deliberately excluded, and the setting is a LIST so adding one
    later is config rather than code.

    Read the size honestly: the news method's own baseline IC over that window
    is **-0.0339**, so this improves a signal that is currently negative. And
    the cell was found by scanning eight classes — it clears the bar on the
    pooled window but on NEITHER half independently (t +1.66 live, +1.88
    backfill), so re-check it once v7dir-era labels settle.
    """
    if not bool(getattr(settings, "enable_catalyst_class_cap", True)):
        return score
    cls = str(catalyst or "").strip().lower()
    if not cls or cls not in _cap_classes():
        return score
    # `or <default>` would be wrong here: 0.0 is falsy but MEANINGFUL — it clips
    # the class to zero, i.e. abstains on it, which was one of the eight measured
    # arms. Only a missing/None value falls back.
    #
    # The fallback reads the SETTING's own default rather than repeating the
    # number: a constant duplicated across two sites is the failure this repo
    # tests for mechanically, and it fired here — the sweep moved the default to
    # 0.03 while a hardcoded 0.10 stayed behind in this function.
    _lim = getattr(settings, "catalyst_cap_limit", None)
    lim = abs(float(_CAP_LIMIT_DEFAULT if _lim is None else _lim))
    return max(-lim, min(lim, score))


def _finite_score(value) -> float:
    """The model's number, clamped to [-1, 1] — and REFUSED if it is not finite.

    `json.loads` accepts bare `NaN` and `Infinity` by default, and the obvious
    clamp is silently wrong on both: every comparison against NaN is False, so
    `max(-1.0, min(1.0, nan))` returns **+1.0** — a malformed answer arriving as
    the strongest possible BULLISH verdict in the 0.40-weight `news` method,
    which in a rank-consumed combine puts the ticker at the top of the
    cross-section. `Infinity` lands on +1.0 the same way.

    Raising instead hands the ticker to the next engine, which is what every
    other unreadable response already does. The module's own standing lesson is
    that a parse failure and an abstention are the same observable; a parse
    failure that produces CONVICTION is worse than either."""
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite sentiment score: {value!r}")
    return max(-1.0, min(1.0, f))


def _parse_response(raw: str) -> tuple[float, str, Optional[str]]:
    """→ ``(score, rationale, catalyst)`` with the catalyst-class cap applied.

    Every caller outside `analyse_sentiment`'s own engine loop wants this one:
    the shadow pass, the repair specialist and the cached-verdict paths all read
    a finished verdict. `analyse_sentiment` uses `_parse_response_uncapped` and
    caps once at the end, because the logprob expectation has to be measured
    against the model's OWN argmax — comparing it to an already-clamped value
    would make `MAX_SHIFT` reject or accept on the wrong quantity."""
    score, rationale, catalyst = _parse_response_uncapped(raw)
    return apply_catalyst_cap(score, catalyst), rationale, catalyst


def _parse_response_uncapped(raw: str) -> tuple[float, str, Optional[str]]:
    """→ ``(score, rationale, catalyst)``; catalyst normalized onto
    NEWS_CATALYST_TYPES (None when the model omitted the field — pre-v4 cache
    entries and degraded responses). The class cap is NOT applied here."""
    # Strip an inline reasoning block before anything else (2026-09-03): local
    # Qwen3 checkpoints reason by default and emit <think>…</think> ahead of the
    # answer. Inert for the hosted engines, which return reasoning in a separate
    # response field and never inline it.
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    raw = _repair_score_number(raw)
    try:
        data = json.loads(raw)
        if "catalyst_score" in data and "priced_in" in data:
            score = apply_priced_in(data["catalyst_score"], data["priced_in"])
        else:
            # v6 and every cached pre-v7d verdict: the model's own number.
            score = _finite_score(data["score"])
        rationale = str(data["rationale"])
        # v7dir: the number must follow the direction the model named first.
        score = apply_direction(score, data.get("direction"))
        return score, rationale, normalize_catalyst(data.get("catalyst"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Salvage a truncated/malformed response (e.g. DeepSeek hitting the
        # 256-token cap mid-rationale → invalid JSON, observed for XBI). The score
        # is the only field that feeds the aggregator, so recover it by regex
        # rather than lose the signal to a 0.0 fallback — especially valuable when
        # the other engine is rate-limited and can't be tried.
        cm_ = re.search(r'"catalyst_score"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
        pm_ = re.search(r'"priced_in"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
        if cm_ and pm_:
            score = apply_priced_in(float(cm_.group(1)), float(pm_.group(1)))
        else:
            # `"score"` cannot match inside `"catalyst_score"` — the opening
            # quote is part of the pattern — so a truncated v7d response that
            # lost its `priced_in` correctly falls through to the raise below
            # rather than being read as an UNDISCOUNTED verdict.
            m = re.search(r'"score"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
            if not m:
                raise
            score = _finite_score(m.group(1))
        rm = re.search(r'"rationale"\s*:\s*"(.*?)(?:"\s*[},]|$)', raw, re.DOTALL)
        rationale = rm.group(1).strip() if rm else "Rationale unavailable (truncated response)."
        cm = re.search(r'"catalyst"\s*:\s*"([A-Za-z_\- ]+)"', raw)
        dm = _DIRECTION_RE.search(raw)
        score = apply_direction(score, dm.group(1) if dm else None)
        return score, rationale, normalize_catalyst(cm.group(1) if cm else None)


def _clock(as_of: Optional[datetime] = None) -> datetime:
    """The instant article ages are measured against, quantised to the hour.

    ``None`` means NOW — the live path, byte-identical to what it always did.
    A REPLAY passes the tick it is reproducing: a scorer that measures age
    against today would find every historical article older than its 7-day
    window and abstain on the whole digest, which is exactly what the
    2026-09-07 backfill pre-flight hit (`src/analysis/news_replay.py`).

    Hour-quantised for the reason `_recency_weight` documents: two runs inside
    one hour must produce identical weights, or a borderline article flips the
    top-20 cut and re-keys the verdict cache.
    """
    base = as_of or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.replace(minute=0, second=0, microsecond=0)


def _recency_weight(article: NewsArticle, as_of: Optional[datetime] = None) -> float:
    """
    Exponential decay based on article age, half-life ``_DECAY_HALF_LIFE_HOURS``
    (18h): 6h → 0.79, 18h → 0.50, 24h → 0.40, 48h → 0.16, 72h → 0.063,
    4d → 0.025, 7d → 0.0016. Articles older than 7 days are excluded entirely
    (weight = 0.0). (The figures quoted here before 2026-09-07 were a 36h
    half-life left behind by an earlier constant.)

    Time-quantised to the hour: the "now" used for age calculation is bucketed
    to the top of the current UTC hour so two pipeline runs within the same
    hour produce identical recency weights. Without this, two back-to-back
    runs could flip the top-20 article cutoff on borderline-stale articles
    and feed the LLM a different digest, producing different sentiment scores.
    """
    now = _clock(as_of)
    age_hours = (now - article.published_at).total_seconds() / 3600
    if age_hours > 168:   # 7 days — too stale to be relevant
        return 0.0
    return math.exp(-math.log(2) * age_hours / _DECAY_HALF_LIFE_HOURS)


def recent_cluster(articles: List[NewsArticle], ratio: float = 3.0,
                   floor_hours: float = 24.0,
                   as_of: Optional[datetime] = None) -> List[NewsArticle]:
    """The FRESHEST cluster of a digest: articles no older than
    ``max(floor_hours, ratio × the freshest article's age)``.

    NOT WIRED into scoring — this is the treatment arm of a pre-registered A/B
    (`scripts/compare_news_truncation.py`), kept here so the offline replay and
    any future live arm cut the digest with the SAME code.

    A ticker's digest often holds two groups: a few articles from an hour ago
    and a few from days back, usually a different catalyst whose move is
    already in the price. The question the test asks is whether the older group
    dilutes the read or supplies the context that makes "how much is left"
    judgeable.

    The cut is RELATIVE, not a fixed age, for two reasons. A fixed window
    empties the digest on a quiet ticker — most names go days without coverage,
    and an empty digest is an abstention, which is a different (and worse)
    change than a narrower one. And a relative cut cannot be gamed by news
    volume: the freshest article always survives (its own age is ≤ ratio × its
    age), so the result is never empty. The ``floor_hours`` term keeps a
    30-minute article from cutting at 90 minutes and discarding this morning's
    coverage of the same story.

    ``now`` is quantised to the hour exactly as ``_recency_weight`` does it, so
    two runs inside one hour cut identically — a borderline article flipping in
    and out would re-key the verdict cache and re-score the ticker.
    """
    if not articles:
        return []
    now = _clock(as_of)
    ages = [(a, max(0.0, (now - a.published_at).total_seconds() / 3600)) for a in articles]
    freshest = min(age for _, age in ages)
    cutoff = max(float(floor_hours), float(ratio) * freshest)
    return [a for a, age in ages if age <= cutoff]


def _article_count_scale(n: int) -> float:
    """
    LEGACY (superseded 2026-08-14 by `_evidence_scale` on recency MASS —
    3 fresh articles are not the same evidence as 3 six-day-old ones).
    Kept for reference/tests; the live path no longer calls it.
    1 article → 0.55,  3 → 0.75,  7 → 0.90,  12+ → 1.0
    """
    if n == 0:
        return 0.0
    return min(1.0, 0.45 + 0.20 * math.log2(n))


def attention_mass(articles: List[NewsArticle],
                   as_of: Optional[datetime] = None) -> tuple[int, float]:
    """``(n_fresh, Σ recency weights)`` over the fresh (<7d) articles.

    The MASS is the freshness-weighted evidence quantity: one article from an
    hour ago contributes ~1.0, one from three days ago ~0.06 — so mass is
    continuous in article ages where a bare count is not. Consumed by the
    news score's evidence scale AND persisted per ticker
    (`signals.news_recency_mass`) as the `news_shock` baseline series.
    """
    fresh = [a for a in articles if _recency_weight(a, as_of) > 0.0]
    return len(fresh), round(sum(_recency_weight(a, as_of) for a in fresh), 4)


def _evidence_scale(mass: float) -> float:
    """Continuous confidence scale on recency mass (2026-08-14, replaces the
    article-count scale). mass 0.5 → ~0.57, 1 → 0.65, 3 → 0.85, 7+ → 1.0 —
    same anchors as the old count curve when every article is fresh, smoothly
    smaller as the set ages."""
    if mass <= 0:
        return 0.0
    return min(1.0, 0.45 + 0.20 * math.log2(1.0 + mass))


def _source_diversity_scale(articles: List[NewsArticle]) -> float:
    """
    Penalise a single-source article set — smoothly (2026-08-14; the old
    0.70/0.85/1.0 steps quantized the final score into a few branches).
    1 source → 0.70, 2 → 0.85, 3 → 0.925, 4 → 0.9625, → 1.0 asymptotically.
    """
    unique_sources = len({a.source for a in articles})
    if unique_sources <= 0:
        return 0.70
    return round(1.0 - 0.30 * (0.5 ** (unique_sources - 1)), 4)


# Provider sentiment LABEL → unit score (scaled by provider_sentiment_magnitude).
_PROVIDER_LABEL_UNIT = {
    "positive": 1.0, "bullish": 1.0, "buy": 1.0,
    "negative": -1.0, "bearish": -1.0, "sell": -1.0,
    "neutral": 0.0, "hold": 0.0,
}


def _provider_sentiment_score(ticker: str,
                              fresh_articles: List[NewsArticle],
                              as_of: Optional[datetime] = None) -> Optional[tuple[float, str]]:
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

    wsum = sum(_recency_weight(a, as_of) for a, _ in scored)
    if wsum <= 0:
        return None
    raw = sum(_recency_weight(a, as_of) * s for a, s in scored) / wsum
    arts = [a for a, _ in scored]
    # Mirrors the LLM path's continuous scalers (2026-08-14) — the two paths
    # must stay comparable, that is this function's contract.
    _n, _mass = attention_mass(arts, as_of)
    precision = _evidence_scale(_mass) * _source_diversity_scale(arts)
    score = round(raw * precision, 4)
    src = next((a.provider_sentiment_source for a in arts if a.provider_sentiment_source), "provider")
    pos = sum(1 for _, s in scored if s > 0)
    neg = sum(1 for _, s in scored if s < 0)
    rationale = (f"Provider sentiment ({src}): {len(arts)} pre-scored article(s) "
                 f"({pos} positive / {neg} negative) → {score:+.2f}; LLM scorer skipped.")
    return score, rationale


# ── News-event taxonomy (2026-08-15) ─────────────────────────────────────────
# The fixed catalyst classes the sentiment LLM (and the historical backfill
# classifier — src/analysis/news_backfill.py) must choose from, so the live and
# backfilled halves of the news-event dataset pool. Persisted per (run, ticker)
# as `signals.news_catalyst` and joined against the pivot forward return by
# `python -m src.analysis.news_events` — the "what kind of news → what kind of
# move" event study. Append-only in practice: renaming a class forks its
# history (the analysis groups on the stored string).
NEWS_CATALYST_TYPES = (
    "earnings",              # reported results beat/miss (no guidance change)
    "guidance",              # outlook raised/cut (with or without results)
    "analyst",               # rating / price-target actions
    "ma_deal",               # M&A, tender, take-private, stake-with-intent
    "fda_clinical",          # FDA decisions, trial readouts, medical data
    "legal_regulatory",      # lawsuits, investigations, fines, policy actions
    "management",            # CEO/CFO/board changes
    "capital_structure",     # offering/dilution, buyback, dividend, split, debt
    "distress",              # going concern, bankruptcy, delisting risk
    "contract_partnership",  # contract wins/losses, partnerships with terms
    "product",               # launches, recalls, operational incidents
    "index_membership",      # index adds/drops
    "insider_activity",      # insider/13D-G buying-selling as the news itself
    "short_squeeze_social",  # social/positioning attention (Reddit, short interest)
    "macro_sector",          # market/sector-wide, not ticker-specific
    "company_pr",            # promotional company-issued release, no hard numbers
    "other",                 # a real ticker-specific catalyst outside the classes
    "none",                  # no event at all
)


def _target_identity(ticker: str) -> Tuple[str, Optional[str], Optional[str], bool]:
    """``(symbol, name, industry, fund)`` for the per-ticker header and the fund
    override — ONE resolution so the two can never disagree about what the
    target is. Name and fund word from the registrant list
    (``company_names.name_keywords``); industry from the cached Polygon
    reference line (``industry_of``, primed for the universe in Step 1, so the
    scoring path stays cache-only). A target is a FUND when Polygon types it as an
    ETF/ETN/ETV/closed-end fund (a wrapper whose name lacks a fund word,
    ``SPDR Gold Shares``, is still a fund) OR when its NAME says so and the
    industry line does not overrule it — ``is_fund`` matches the bare token
    "trust", so every REIT tripped it (``industry_is_operating_trust``). Fail-soft: no
    name ⇒ ``(sym, None, None, False)`` and no industry lookup at all."""
    sym = (ticker or "").strip().upper()
    try:
        from src.data.company_names import (name_keywords, industry_of,
                                            industry_is_fund,
                                            industry_is_operating_trust,
                                            security_type, type_is_fund)
        kw = name_keywords(sym)
    except Exception:
        return sym, None, None, False
    name = (kw.get("name") or "").strip() if isinstance(kw, dict) else ""
    if not name:
        return sym, None, None, False
    try:
        industry = (industry_of(sym) or "").strip() or None
    except Exception:
        industry = None
    # A Polygon fund TYPE is a direct statement about the security and always
    # wins. A fund word in the NAME is an inference, and an operating industry
    # line overrules it — `is_fund` matches the bare token "trust", which every
    # REIT carries.
    # Polygon's security TYPE is the authoritative tier — a statement about the
    # instrument. The industry line and the name heuristic run behind it because
    # the type table covers 97.2% of this universe and types closed-end funds as
    # `CS` (CCD/CHI/CHY/CSQ), so the name is still what catches those.
    fund = (type_is_fund(security_type(sym))
            or industry_is_fund(industry)
            or (bool(kw.get("fund")) and not industry_is_operating_trust(industry)))
    return sym, name, industry, fund


def _target_header(ticker: str) -> str:
    """The per-ticker line(s) that open the variable suffix (v6).

    ``TARGET TICKER: AR — Antero Resources Corp (industry: Crude Petroleum &
    Natural Gas)`` — articles name the COMPANY, not the symbol (only ~32% of
    the per-ticker yfinance feed mentions the symbol at all), and a symbol →
    company mapping is exactly what a small local model cannot be trusted to
    know for a mid-cap. The industry line (Polygon ``sic_description``, cached
    beside the names) is the hook that lets the model notice that "EQT AB", a
    Swedish private-equity firm, is not "EQT Corp", a natural-gas producer —
    the same-name-other-entity confusion behind a share of the catalyst
    mislabels. A FUND gets a second line telling the scorer that its holdings
    ARE the company, so sector / index / commodity coverage is information
    about the target rather than the abstention case. The header text salts
    the verdict cache key, so a name or industry resolving later re-scores
    that ticker once. Fail-soft: no name ⇒ the bare v5 header."""
    sym, name, industry, fund = _target_identity(ticker)
    if not name:
        return f"TARGET TICKER: {sym}"
    head = f"TARGET TICKER: {sym} — {name}"
    if industry:
        head += f" (industry: {industry})"
    if fund:
        head += ("\nThe target is a FUND: its \"company\" is what it holds. Score the "
                 "direction of its holdings / sector / index / underlying asset for the "
                 "fund's own price.")
    return head


def normalize_catalyst(value) -> Optional[str]:
    """Map a model-emitted catalyst label onto the fixed taxonomy.

    None/empty → None ("not captured" — distinct from "none", the model's
    explicit no-event verdict); a recognised label → itself; anything else →
    "other" (the model invented a class — keep the event, don't lose it)."""
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not v:
        return None
    return v if v in NEWS_CATALYST_TYPES else "other"


# ── Fund/ETF catalyst override (2026-09-06, mechanical, zero LLM cost) ───────
# Both engines type an ETF with a HOLDING's class (XLV → fda_clinical, GLD →
# macro... or earnings, SPY → earnings), because the scorer is told a fund's
# "company" is what it holds. A fund has no earnings, insiders, trials or
# offerings of its own, so any company-event class on a fund target is a
# read-through of its holdings — `macro_sector` by the taxonomy's own rule.
# Applied to the LABEL only, at read time (cached verdicts included), never to
# the score: no scorer epoch. The model's own label survives as `catalyst_raw`.
_FUND_KEEP_CLASSES = frozenset({"none", "macro_sector"})


def _is_fund_target(ticker: str) -> bool:
    """The header's own FUND verdict (``_target_identity``), so the override
    and the prompt line the model read agree by construction."""
    try:
        return _target_identity(ticker)[3]
    except Exception:
        return False


def fund_catalyst_override(ticker: str, catalyst: Optional[str]) -> Optional[str]:
    """`macro_sector` for a FUND target labelled with a company-event class;
    otherwise the label unchanged. Inert with `enable_catalyst_fund_override`
    off, on a missing label, and on the two classes a fund may legitimately
    carry (`none`, `macro_sector`)."""
    if catalyst is None or catalyst in _FUND_KEEP_CLASSES:
        return catalyst
    if not getattr(settings, "enable_catalyst_fund_override", True):
        return catalyst
    return "macro_sector" if _is_fund_target(ticker) else catalyst


# FIXED, ticker-FREE instruction prefix — identical for every ticker, so it forms
# a shared prefix that DeepSeek auto-caches across the ~40 per-ticker calls in a
# run (measured 2026-09-05: 9,528 chars = **2,176 prompt tokens**, cached in full
# on every call after the first). Anthropic would cache it via cache_control IF it
# met the per-model minimum — it does NOT: 2,176 tok is still below Haiku 4.5's
# 4096-tok minimum, so cache_control stays a silent no-op on Haiku and DeepSeek's
# automatic prefix caching is the real saving. The per-ticker name + news digest
# go in the suffix.
_SENTIMENT_PREFIX = """You are an elite buy-side analyst with 25 years of experience at top-tier hedge funds. You have an exceptional ability to identify the exact news catalysts that move stock prices — your track record places you in the top 0.1% of market professionals worldwide.

Your task: analyse the recent news for THE TARGET TICKER (specified at the end) and score the SHORT-TERM directional impact (1–5 trading days) with surgical precision.

HOW YOUR SCORE IS USED — read this before scoring:
- Your score is consumed CROSS-SECTIONALLY: it is ranked against every other ticker scored today. A score of exactly 0.0 is an ABSTENTION — the ticker is removed from the ranking altogether, as if its news had never been read. An abstention discards the whole read; a small lean keeps it.
- So 0.0 is reserved for two cases only: (a) NOTHING in the digest connects to the target — no article about the company, none about what it holds if it is a fund, and no named peer, customer, supplier, competitor or sector development with a read-through to it; (b) the remaining move is genuinely nil, e.g. an all-cash acquisition target pinned at a certain offer price.
- Whenever ANYTHING in the digest connects to the target, output a SIGNED lean. Uncertainty belongs in the MAGNITUDE, never in a retreat to 0.0: a weak, soft, stale, routine, indirect or largely-priced read is a SMALL nonzero number. Decide the sign first, then decide how small.
- ROUTINE company-specific items ARE information and carry a conventional direction — score them in the LEAN band, never 0.0 — MINOR only when the item discloses a size that is material to the company (a stake or buyback large relative to its market value, a dividend cut): a new or increased institutional stake (13F/13D/13G) leans up and a trimmed one leans down; insider open-market buying leans up, insider or 10b5-1 selling leans slightly down; a dividend declaration, raise or buyback leans up, a cut or suspension is a hard negative; index inclusion leans up, deletion down; a product note, a conference appearance, a routine filing or an earnings-date notice takes the smallest lean in the direction of its framing. When a routine item has no conventional direction, use the tone of the coverage and the smallest lean — do not abstain.
- INDIRECT information counts. A named peer's, customer's, supplier's or competitor's news with a plausible read-through to the target is a LEAN in the read-through direction (MINOR only when the link is explicit and material); a passing mention in a sector round-up, a listicle or a market recap is a LEAN in the framing direction of the piece.
- FUNDS: when the target header flags a FUND, its "company" is what it holds. Coverage of its holdings, sector, index, commodity, rates or crypto asset IS information about the target — score the direction for the fund's own price, on the same bands.
- If catalysts conflict, NET them by magnitude and recency — the dominant, most recent, highest-impact catalyst sets the sign and the runner-up shrinks the magnitude. Never mechanically average to 0.0.
- Recency matters: articles marked "1h ago" or "6h ago" carry much more weight than "3d ago" or "5d ago".

MAGNITUDE — build the score in two steps, band then exact placement:
- ±0.60–1.00 HIGH impact, unambiguous: earnings beat/miss WITH a guidance change, FDA approval/rejection, M&A, major regulatory or legal action, CEO departure, going-concern/bankruptcy or delisting risk, large dilutive offering.
- ±0.25–0.60 CLEAR but moderate: analyst upgrade/downgrade with price target, EPS surprise without guidance change, contract win/loss with disclosed size, supply-chain disruption.
- ±0.10–0.25 MINOR: a real, dated, ticker-specific event of modest price relevance — a small contract, a product update, a partnership without terms, a routine filing whose disclosed size is material.
- ±0.01–0.10 LEAN: no dated event, or only a routine or indirect one, but the coverage has a direction — favourable or critical commentary, an aggregator's bull or bear framing, a "why is X up/down" recap, a stale catalyst still being discussed, a routine filing or stake change, a peer or sector read-through, a passing mention in a round-up, social attention with a tilt. This band exists so that "soft" never becomes "silent". It is a lean, not a thesis, and it must never be inflated into the bands above.
- PLACE the score inside its band from the specifics: surprise size vs expectations, disclosed dollar amounts relative to the company's size, source independence, corroboration across independent outlets, freshness, and how much of the reaction has already happened. Each factor pushes the placement up or down inside the band; the result is the exact value.
- PRECISION: ties destroy the ranking. Express the score with TWO-decimal granularity where the second decimal comes from the placement weighing above, never from habit: two reads of visibly different strength must NOT share a value; round numbers, band edges and band midpoints are almost never the honest result of a real weighing; and the LEAN band needs the same care as the others — a hundred soft leans that all land on the same number are worth nothing.

PRICED-IN CHECK — score the REMAINING move from now, not the catalyst's total worth:
- If the digest itself reports that the stock already moved sharply on this catalyst ("shares surged 40%", "up 75% pre-market"), most of the catalyst is consumed. Score only the expected FOLLOW-THROUGH — after an outsized one-day spike that residual is small and often NEGATIVE (extended movers mean-revert).
- "Why is X up/down today" recap pieces report a move already taken; a recap is not a fresh catalyst — score the remaining move as a LEAN, in whichever direction the residual points.
- A hard catalyst 2–3 days old has mostly been traded; keep the sign and shrink the magnitude unless a multi-day repricing mechanism is still working (e.g. estimate revisions after a guidance change).
- Asymmetry: fresh NEGATIVE hard catalysts (misses, guidance cuts, going-concern, regulatory action) tend to keep drifting down for several sessions even after a first sell-off — do not treat the first red day as full pricing. Positive spikes are more often fully priced immediately.

SOURCE TIERS — each article is tagged "[source | age]"; weight the tag, not just the words:
- HARD (can justify up to ±1.0): SEC 8-K filings, earnings/EPS surprises, analyst rating & price-target changes, and primary financial journalism (Reuters, Bloomberg, WSJ, Barron's, CNBC) on M&A, FDA, guidance, legal/regulatory events.
- COMPANY-ISSUED (PRNewswire, GlobeNewswire, Business Wire, Accesswire, Proactive): self-selected promotion. A promotional release — partnership, LOI, product launch, "record" results without numbers — caps at ±0.30 unless it discloses binding dollar amounts material to the company's size or is corroborated by independent reporting. NEGATIVE facts inside company-issued text (going concern, dilution, guidance cut, compliance notice) are involuntary disclosures — weight them FULLY.
- AGGREGATOR/COMMENTARY (Zacks, Motley Fool, Simply Wall St., 24/7 Wall St., StockStory, Insider Monkey, Trefis, GuruFocus, MarketBeat, Barchart, "best stocks" listicles): opinion and recycled facts, not catalysts. Score an underlying fact on its own merits only if it is itself fresh and hard; commentary or a listicle mention alone is at most a LEAN in the direction of its framing.
- SOFT (cap ±0.2 on their own): Reddit/WSB, StockTwits, Google Trends spikes, short-interest shifts, dark-pool prints. Corroborating color, never a standalone thesis — aligned with a hard catalyst it modestly amplifies conviction; contradicting one, discount it.
- One EVENT is one catalyst: ten syndicated articles about the same event do not make it ten times bigger. Independent corroboration raises confidence in the FACT, not the magnitude.

Respond with ONLY a JSON object with exactly these fields, in this order:
- "rationale": one to three sentences naming (1) the specific catalyst or the direction of the coverage, (2) the price mechanism, and (3) what is already priced in. If the score is 0.0, state which abstention case applies AND why not even a lean was possible. The rationale comes FIRST so the score is decided from the reasoning, not before it.
- "catalyst": the DOMINANT catalyst class behind your read, exactly one of: __CATALYST_TYPES__. Use "none" when there is no dated event (a LEAN with no event is "none"); a routine item keeps its class even at LEAN size (a stake or insider transaction is "insider_activity", a dividend or buyback is "capital_structure", an index change is "index_membership"); a peer/sector read-through or a fund scored on its holdings is "macro_sector"; when a real event nets to 0.0 (e.g. fully priced in), still name its class. "company_pr" = promotional company-issued release with no hard numbers.
- "score": float between -1.0 (very bearish) and +1.0 (very bullish); 0.0 only for the two abstention cases above.

Format skeleton (the angle-bracket placeholders are to be REPLACED, never copied; no worked example is given because any number written here would become a favourite answer):
{"rationale": "<one to three sentences>", "catalyst": "<one class from the list>", "score": <signed two-decimal number>}"""

# Interpolate the taxonomy once at import — the prompt and NEWS_CATALYST_TYPES
# cannot drift apart, and the string stays byte-stable for prefix caching.
_SENTIMENT_PREFIX = _SENTIMENT_PREFIX.replace(
    "__CATALYST_TYPES__", ", ".join(NEWS_CATALYST_TYPES))


# ── v7d: the priced-in discount, made ARITHMETIC ──────────────────────────────
# WHY (measured 2026-09-09, 146 re-scored clusters over 38 days with rationales
# kept): v6's rationale field REQUIRES the model to name "what is already priced
# in", so it says so in 77% of rationales — and then scores as if it had not.
# Clusters whose rationale states the move is already made are scored HIGHER,
# not lower (mean |s| 0.327 vs 0.309); 48% of them still land in the CLEAR band
# or above and 16% in the HIGH band. The worst case in the sample scored -0.85
# on a rationale reading "the stock has already reacted, with a steep decline".
# The instruction is satisfied VERBALLY and never reaches the number.
#
# The fix is not more prose — prose is what failed. The two judgements are split
# into two REQUIRED fields and the score becomes arithmetic we do ourselves:
#
#     score = catalyst_score * (1 - priced_in)
#
# so the discount cannot be skipped. The model no longer emits "score" at all;
# `_parse_response` computes it through `apply_priced_in`. Three properties are
# deliberate: the bands now describe the catalyst's FULL worth (the placement
# rule's "how much has already happened" clause MOVES to `priced_in`, or the
# discount is taken twice); `priced_in` may exceed 1.0, so an overshoot flips
# the sign, which is what the v6 prose already asked for and could not express;
# and a non-zero catalyst can never round to an exact 0.0 unless the model puts
# the fraction AT 1.0, because a 0.0 is an ABSTENTION that removes the ticker
# from the cross-section — the exact failure v5/v6 were built to end.
#
# Built by explicit replacement of v6 rather than as a second literal: three
# copies of a rubric is how the confidence prompt drifted, and here the DIFF is
# the experiment. Each replacement is asserted, so a v6 edit that invalidates
# one fails at IMPORT rather than silently shipping a half-converted prompt.
def _build_decomposed_prefix(base: str) -> str:
    def _sub(text: str, old: str, new: str) -> str:
        if old not in text:
            raise RuntimeError(
                "sentiment v7d: the v6 prompt no longer contains a passage the "
                f"decomposition rewrites: {old[:60]!r}")
        return text.replace(old, new, 1)

    out = base
    out = _sub(
        out,
        "- Whenever ANYTHING in the digest connects to the target, output a SIGNED lean. "
        "Uncertainty belongs in the MAGNITUDE, never in a retreat to 0.0: a weak, soft, "
        "stale, routine, indirect or largely-priced read is a SMALL nonzero number. Decide "
        "the sign first, then decide how small.",
        '- Whenever ANYTHING in the digest connects to the target, output a SIGNED '
        '"catalyst_score". Uncertainty belongs in the MAGNITUDE, never in a retreat to 0.0: '
        'a weak, soft, stale, routine or indirect read is a SMALL nonzero number. Decide the '
        'sign first, then decide how small. A read that is merely LARGELY PRICED keeps its '
        'full worth here and is discounted by "priced_in" instead — never shrink it twice.')
    out = _sub(
        out,
        "MAGNITUDE — build the score in two steps, band then exact placement:",
        'MAGNITUDE — "catalyst_score" is the catalyst’s FULL directional worth, judged as if '
        "the market had not yet traded a single share on it. Build it in two steps, band then "
        "exact placement:")
    out = _sub(
        out,
        "- PLACE the score inside its band from the specifics: surprise size vs expectations, "
        "disclosed dollar amounts relative to the company's size, source independence, "
        "corroboration across independent outlets, freshness, and how much of the reaction has "
        "already happened. Each factor pushes the placement up or down inside the band; the "
        "result is the exact value.",
        '- PLACE "catalyst_score" inside its band from the specifics: surprise size vs '
        "expectations, disclosed dollar amounts relative to the company's size, source "
        "independence, corroboration across independent outlets, and freshness. Each factor "
        "pushes the placement up or down inside the band; the result is the exact value. How "
        'much of the reaction has ALREADY happened does NOT belong here — that is the separate '
        '"priced_in" field, and taking it off in both places is the single most common error.')
    out = _sub(
        out,
        "- PRECISION: ties destroy the ranking. Express the score with TWO-decimal granularity",
        '- PRECISION: ties destroy the ranking. Express "catalyst_score" with TWO-decimal '
        "granularity")
    out = _sub(out, """PRICED-IN CHECK — score the REMAINING move from now, not the catalyst's total worth:
- If the digest itself reports that the stock already moved sharply on this catalyst ("shares surged 40%", "up 75% pre-market"), most of the catalyst is consumed. Score only the expected FOLLOW-THROUGH — after an outsized one-day spike that residual is small and often NEGATIVE (extended movers mean-revert).
- "Why is X up/down today" recap pieces report a move already taken; a recap is not a fresh catalyst — score the remaining move as a LEAN, in whichever direction the residual points.
- A hard catalyst 2–3 days old has mostly been traded; keep the sign and shrink the magnitude unless a multi-day repricing mechanism is still working (e.g. estimate revisions after a guidance change).
- Asymmetry: fresh NEGATIVE hard catalysts (misses, guidance cuts, going-concern, regulatory action) tend to keep drifting down for several sessions even after a first sell-off — do not treat the first red day as full pricing. Positive spikes are more often fully priced immediately.""",
                """PRICED-IN FRACTION — "priced_in" is the share of that full worth the market has ALREADY traded. It is a separate judgement, made after the magnitude and never folded into it:
- Judge it from what the digest itself reports about the move so far ("shares surged 40%", "up 75% pre-market", "the stock has already reacted", a red day after a miss) and from how long the catalyst has been public. Nothing traded on it yet: the fraction sits at the bottom of its range. The move fully made: it reaches one.
- It MAY exceed one. An outsized one-day spike often overshoots, and past one the residual points the OTHER way — extended movers mean-revert. Go above one only where the overshoot is visible in the digest, never as a hedge.
- A "why is X up/down today" recap reports a move already taken. That makes the FRACTION high; it does not make the catalyst small. Judge the two independently.
- A hard catalyst 2–3 days old has mostly been traded — unless a multi-day repricing mechanism is still working (e.g. estimate revisions after a guidance change), which holds the fraction down.
- Asymmetry: fresh NEGATIVE hard catalysts (misses, guidance cuts, going-concern, regulatory action) keep drifting down for several sessions, so a first red day does NOT put the fraction near one. Positive spikes are more often fully priced immediately.
- You do NOT multiply anything. The two fields are combined mechanically after you answer, so state each one honestly on its own terms.""")
    out = _sub(
        out,
        '- "rationale": one to three sentences naming (1) the specific catalyst or the direction '
        'of the coverage, (2) the price mechanism, and (3) what is already priced in. If the '
        'score is 0.0, state which abstention case applies AND why not even a lean was possible. '
        'The rationale comes FIRST so the score is decided from the reasoning, not before it.',
        '- "rationale": one to three sentences naming (1) the specific catalyst or the direction '
        'of the coverage, (2) the price mechanism, and (3) how much of the move has ALREADY '
        'happened and what in the digest tells you so. If nothing connects to the target, state '
        'which abstention case applies AND why not even a lean was possible. The rationale comes '
        'FIRST so the two numbers are decided from the reasoning, not before them.')
    out = _sub(out, """- "score": float between -1.0 (very bearish) and +1.0 (very bullish); 0.0 only for the two abstention cases above.

Format skeleton (the angle-bracket placeholders are to be REPLACED, never copied; no worked example is given because any number written here would become a favourite answer):
{"rationale": "<one to three sentences>", "catalyst": "<one class from the list>", "score": <signed two-decimal number>}""",
                """- "catalyst_score": float between -1.0 (very bearish) and +1.0 (very bullish) — the catalyst's FULL worth from the MAGNITUDE bands, before any discount for what has already been traded. This is where the SIGN is decided. Use 0.0 only for abstention case (a), nothing in the digest connecting to the target.
- "priced_in": float, from 0.0 when the market has not traded this catalyst at all, to 1.0 when the move is fully made, and above 1.0 only for a visible overshoot. Abstention case (b) — an all-cash target pinned at the offer — is a real catalyst with the fraction at 1.0.

Do NOT output a "score" field. The final score is computed from your two numbers.

Format skeleton (the angle-bracket placeholders are to be REPLACED, never copied; no worked example is given because any number written here would become a favourite answer):
{"rationale": "<one to three sentences>", "catalyst": "<one class from the list>", "catalyst_score": <signed two-decimal number>, "priced_in": <fraction>}""")
    return out


_SENTIMENT_PREFIX_DECOMPOSED = _build_decomposed_prefix(_SENTIMENT_PREFIX)
# v7d-2026-09-09: the priced-in discount became ARITHMETIC (see above). A
# different prompt AND a different output contract, so it carries its own
# version salt; flipping `enable_priced_in_decomposition` therefore re-scores
# rather than serving v6 verdicts, and needs a news-family scorer epoch placed
# by the RUNS of the restart that deploys it.
_SENT_PROMPT_VERSION_DECOMPOSED = "v7d-2026-09-09"


# ── v7dir: NAME THE DIRECTION BEFORE THE NUMBER ───────────────────────────────
# WHY. A blind judge shown only the rationale — never the score, so it cannot
# anchor — disagreed with the emitted SIGN on 7.8% of live verdicts. The model
# argues one way and scores the other.
#
# The obvious repair is to DETECT and flip, and it is unsafe: the best detector
# available (a bull/bear word count) is 21% precise, so it would damage four
# correct verdicts for every one it fixed. Concessive netting is exactly what it
# gets wrong — "beat estimates, HOWEVER the stock fell on weak guidance" is a
# correct bearish net, and the word count scores the losing clause.
#
# So make the inconsistency impossible instead of catching it. Field order IS
# generation order with reasoning off, so a `direction` emitted between the
# rationale and the score is chosen from the argument and the number then has to
# follow it. Measured paired on 90 digests, both arms judged blind:
#
#     contradiction   7.8% -> 3.3%      (B fixed 5, broke 1, both wrong 2)
#     direction agrees with its own score   100.0%, on 99% of rows
#     distinct scores 87 -> 87, mean |score| 0.367 -> 0.422
#     rank corr A vs B +0.768, 10/90 SIGN FLIPS
#
# SHIPPED ON THE USER'S DIRECTIVE with the caveats on the record: the McNemar
# one-sided exact p is **0.109 — NOT significant** (6 discordant pairs), it
# changes the SIGN of 11% of verdicts and lifts magnitude 15%, and internal
# consistency is not accuracy. v6-era pivot labels had not settled when this
# shipped, so "does B beat A on per-day pivot IC" is still unanswered and is the
# first thing to run once they do.
#
# The mechanical enforcement below is, today, a NO-OP: direction and score
# already agree 100% of the time. It is kept as the invariant's guarantee — if a
# future checkpoint drifts, the contradiction cannot reach the panel.
def _build_direction_prefix(base: str) -> str:
    def _sub(text: str, old: str, new: str) -> str:
        if old not in text:
            raise RuntimeError("sentiment v7dir: the v6 prompt no longer contains a "
                               f"passage the direction field needs: {old[:60]!r}")
        return text.replace(old, new, 1)

    out = _sub(base, '- "catalyst": the DOMINANT catalyst class behind your read, exactly one of: ',
               '- "direction": exactly one of "UP", "DOWN", "NONE" — the way you expect the '
               'stock to move, stated BEFORE the number so the number follows from it. If the '
               'digest weighs conflicting catalysts, this is the side you concluded is DOMINANT, '
               'not the one you mentioned first. "NONE" only for the abstention cases above.\n'
               '- "catalyst": the DOMINANT catalyst class behind your read, exactly one of: ')
    out = _sub(out,
               '{"rationale": "<one to three sentences>", "catalyst": "<one class from the list>", '
               '"score": <signed two-decimal number>}',
               '{"rationale": "<one to three sentences>", "direction": "<UP|DOWN|NONE>", '
               '"catalyst": "<one class from the list>", "score": <signed two-decimal number>}')
    return out


_SENTIMENT_PREFIX_DIRECTION = _build_direction_prefix(_SENTIMENT_PREFIX)
_SENT_PROMPT_VERSION_DIRECTION = "v7dir-2026-09-10"


def apply_direction(score: float, direction: Optional[str]) -> float:
    """Force the number to follow the direction the model just named.

    `NONE` abstains — it maps onto v6's own two abstention cases. A stated UP/DOWN
    that disagrees with the sign keeps the MAGNITUDE and takes the direction's
    sign: the direction is generated from the rationale and the number after it,
    so when they disagree the direction is the better-supported statement.
    """
    d = (direction or "").strip().upper()
    if d == "NONE":
        return 0.0
    if d not in ("UP", "DOWN") or score == 0.0:
        return score
    want = 1.0 if d == "UP" else -1.0
    return abs(score) * want if (score > 0) != (want > 0) else score


def _prompt_pair() -> tuple:
    """``(prefix, version)`` for THIS call.

    Reads the module attributes when the decomposition is off, so a harness that
    monkeypatches `_SENTIMENT_PREFIX` / `_SENT_PROMPT_VERSION` (the prompt A/B
    scripts) keeps working byte-for-byte.
    """
    if bool(getattr(settings, "enable_priced_in_decomposition", False)):
        return _SENTIMENT_PREFIX_DECOMPOSED, _SENT_PROMPT_VERSION_DECOMPOSED
    if bool(getattr(settings, "enable_direction_field", True)):
        return _SENTIMENT_PREFIX_DIRECTION, _SENT_PROMPT_VERSION_DIRECTION
    return _SENTIMENT_PREFIX, _SENT_PROMPT_VERSION


def _anthropic_user_content(prefix: str, suffix: str, model: str):
    """Anthropic ``content`` for a sentiment call: a cache_control prefix block +
    variable suffix when caching is on AND the prefix meets the model's minimum
    (Haiku 4.5 = 4096 tok). The sentiment prefix measures 2,176 tok (2026-09-05),
    so this still returns a single concatenated string (cache_control would
    silently no-op); kept so it caches automatically if the prompt ever grows past
    the minimum — the v3-v6 rubric already took it from ~600 tok to 2,176."""
    if settings.enable_prompt_caching:
        min_tokens = 4096 if "haiku" in (model or "") else 1024
        if len(prefix) >= min_tokens * 4:   # ~4 chars/token
            return [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": suffix},
            ]
    return prefix + suffix


DIGEST_MAX_ARTICLES = 20


def _excluded_sources() -> set:
    return {x.strip().lower() for x in
            str(getattr(settings, "source_tier_excluded", "") or "").split(",") if x.strip()}


def apply_source_tier(articles: List[NewsArticle]) -> List[NewsArticle]:
    """Drop aggregator/listicle outlets — UNLESS that empties the digest.

    The v6 prompt already tiers this material down ("commentary/listicles alone
    and recaps are at most a LEAN"), but a digest is ~58% of it and the residual
    still dilutes the read. The FALLBACK is the load-bearing half: an empty
    digest is an ABSTENTION, which removes the ticker from the cross-section
    entirely — a bigger and worse change than a thinner digest, and the exact
    failure the 2026-09-04 relevance rework was built to end. Measured on one
    day (n=60): a hard filter emptied 9 of 60 digests; with the fallback the
    abstention count went 3 -> 6 while the day-neutral rank IC went -0.052 ->
    +0.256.

    Returns the input unchanged when the filter is off or removes nothing.
    """
    if not getattr(settings, "enable_source_tier_filter", False) or not articles:
        return articles
    excl = _excluded_sources()
    if not excl:
        return articles
    kept = [a for a in articles if (a.source or "").strip().lower() not in excl]
    return kept if kept else articles


def _passing_mention_share(ticker: str, articles: List[NewsArticle]) -> Optional[float]:
    """Share of a digest whose articles name the target only in the BODY.

    The headline is where an article declares what it is ABOUT. A company named
    only further down is being listed, compared or held — a round-up, an
    ETF-holdings piece, a multi-ticker law-firm notice.

    None when the question cannot be answered, so an abstention is never
    triggered by a broken lookup — the failure mode would be a silent zero
    inside the 0.40-weight `news` method, exactly what the 2026-09-04 relevance
    rework existed to end.

    THE LOAD-BEARING GUARD: a ticker whose registrant name does not resolve to a
    phrase or a distinctive token (ARM -> `phrases: []`, `tokens: []`, because
    "Arm" is an ordinary word) is INVISIBLE to `mention_evidence` in a headline.
    Every one of its headlines then looks like a passing mention, the share
    reads 1.00, and the rule would abstain on that ticker ALWAYS — not because
    its coverage is round-ups but because the matcher is blind to it. Measured
    on the validation window: 6 of 130 tickers (ADT, AIR, AON, ARM, AVB, AZZ),
    21 of 655 rows, and abstaining on them wholesale scores **-0.0048 IC**. So
    they are skipped, and the shipped effect is measured WITHOUT them.
    """
    try:
        from src.data.company_names import mention_evidence, name_keywords
        arts = [a for a in (articles or []) if getattr(a, "title", None)]
        if not arts:
            return None
        kw = name_keywords(ticker) or {}
        if not (kw.get("phrases") or kw.get("tokens")):
            return None                 # matcher blind to this name — see above
        body_only = sum(1 for a in arts if not mention_evidence(ticker, a.title or ""))
        return body_only / len(arts)
    except Exception as e:                                      # noqa: BLE001
        logger.debug(f"[sentiment] passing-mention share failed for {ticker}: {e}")
        return None


def digest_articles(articles: List[NewsArticle],
                    as_of: Optional[datetime] = None) -> List[NewsArticle]:
    """The articles the model actually READS: fresh (<7d as of ``as_of``),
    recency-sorted, capped at ``DIGEST_MAX_ARTICLES``.

    Extracted 2026-09-07 so the scorer and anything measuring a digest agree on
    what "the digest" is. They did not: the replay was recording the PRE-cut
    relevant set (493 articles for NDAQ) while the live shadow row records the
    POST-cut list, so every digest-size comparison between them was
    apples-to-oranges. One function, both callers.
    """
    fresh = [a for a in articles if _recency_weight(a, as_of) > 0.0]
    return sorted(fresh, key=lambda a: _recency_weight(a, as_of),
                  reverse=True)[:DIGEST_MAX_ARTICLES]


def _digest_text(ticker: str, to_score: List[NewsArticle], as_of: Optional[datetime],
                 digest_override: Optional[tuple] = None) -> str:
    """The <news> block the model reads: one entry per article with its source
    and AGE, dated from ``as_of`` (None = now). ``digest_override`` =
    ``(digest_id, text)`` from `sentiment_digests`: returned verbatim when
    ``to_score`` is exactly that digest, so a re-score reads the article ages
    the original call read (a cache hit in a later run re-used a verdict scored
    at an earlier clock). Anything else rebuilds the text as live does."""
    if digest_override is not None and digest_override[0] == digest_id_for(ticker, to_score):
        return digest_override[1]
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    digest_lines = []
    for a in to_score:
        age_h = (now - a.published_at).total_seconds() / 3600
        age_label = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h/24:.1f}d ago"
        digest_lines.append(
            f"[{a.source} | {age_label}] {a.title}\n{a.summary[:400]}"
        )
    return "\n\n".join(digest_lines)


def analyse_sentiment(ticker: str, articles: List[NewsArticle], *,
                      allow_provider: Optional[bool] = None,
                      store_digest: bool = True,
                      force_engine: Optional[str] = None,
                      as_of: Optional[datetime] = None,
                      digest_override: Optional[tuple] = None) -> tuple[float, str, dict]:
    """
    Score news sentiment for a ticker with precision controls applied.

    ``force_engine`` ('deepseek' | 'anthropic') pins scoring to exactly that
    engine with NO cross-engine fallback — used by the opener-pinned hold-review
    so a position is always re-scored by the same sentiment engine that opened it
    (apples-to-apples). On a forced-engine failure the score is the usual
    (0.0, "error") rather than silently switching engines. ``None`` keeps the
    per-run A/B order (`_PRIMARY_SENTIMENT_ENGINE` first, the other as fallback).

    ``digest_override`` (the news re-score path only; None on every live call):
    ``(digest_id, text)`` of a digest a past run's scorer read — see
    `_digest_text`. The prompt carries that exact text when this call selects
    the same digest.

    Returns:
        (score, rationale, meta)
        score: float in [-1.0, +1.0] after recency/count/diversity adjustments
        rationale: brief explanation citing the specific news catalyst
        meta: {"catalyst": <NEWS_CATALYST_TYPES or None>, "raw_score": <pre-scaler
        verdict or None>} — the news-event dataset fields (2026-08-15). Empty-ish
        on degraded paths; consumers must .get(). The aggregator tolerates legacy
        2-tuples from test doubles, so meta is additive, never load-bearing.
    """
    if not articles:
        return 0.0, "No recent news articles found.", {}

    # Filter out stale articles (>7 days) before scoring
    fresh_articles = [a for a in articles if _recency_weight(a, as_of) > 0.0]
    if not fresh_articles:
        return 0.0, "All available articles are older than 7 days — no actionable signal.", {}

    # PASSING-MENTION ABSTENTION (2026-09-10). An article that names the target
    # only in its BODY, never in its headline, is a round-up, an ETF-holdings
    # piece or a multi-ticker newswire notice: it MENTIONS the company without
    # being ABOUT it. The relevance filter admits them correctly — they do
    # mention the target — but a digest made mostly of them is not coverage, and
    # the v6 prompt's own abstention case (a) already says so.
    #
    # Measured over 655 ticker-days / 50 days, oriented pivot return per
    # decision, day-clustered, by passing-mention share of the digest:
    #
    #     < 20%      +0.903 pp   hit 54.5%
    #     20-50%     -0.573 pp   hit 48.1%
    #     >= 50%     -2.223 pp   hit 41.9%     <- the read INVERTS
    #
    # A monotone dose-response, and the high cohort is anti-predictive rather
    # than merely weak — which is why this ABSTAINS instead of capping the
    # magnitude: capping preserves a sign the data says is wrong. Abstaining at
    # this threshold measured **+0.0093 per-day pivot IC (t +2.09, same-sign
    # halves)** against the unfiltered baseline — but that figure includes 2
    # rows that fired only because the name matcher is blind to their ticker.
    # With those correctly guarded out (see `_passing_mention_share`) the honest
    # number is **+0.0075, t +1.81, same-sign halves: BELOW the house bar**.
    # Shipped anyway on the user's standing instruction to take positive effects
    # even when small, and because the surrounding evidence is unusually
    # coherent for a below-bar cell: a monotone dose-response across three
    # buckets, positive at all three thresholds (0.35 / 0.50 / 0.65), positive
    # on BOTH the ranker and decider metrics, same-sign halves everywhere. The
    # looser 0.35 and 0.50 cuts are larger (+0.0274 / +0.0123) and less certain,
    # so the SHIPPED threshold is the conservative one.
    #
    # It also saves the LLM call, which is the rare case where the cheap thing
    # and the right thing agree.
    #
    # NO SCORER EPOCH is registered, deliberately. The change is confined to
    # ~2% of digests and turns a weak read into an ABSTENTION — a state the
    # panel already treats as "no view" and which the win-rate filter, the IC
    # tilt and the rank transform all skip. The other ~98% of scores are
    # byte-identical, so nothing about the SCALE or meaning of a nonzero verdict
    # moved. **If the threshold is ever lowered, revisit that**: at 0.35 it
    # touches 15% of digests and stops being a refinement.
    if bool(getattr(settings, "enable_passing_mention_abstention", True)):
        share = _passing_mention_share(ticker, fresh_articles)
        floor = float(getattr(settings, "passing_mention_abstain_share", 0.65) or 0.65)
        if share is not None and share >= floor:
            logger.debug(f"{ticker} sentiment abstained: {share:.0%} of the digest names it "
                         f"only in the body (>= {floor:.0%}) — a round-up, not coverage")
            return 0.0, (f"No usable read: {share:.0%} of the digest mentions {ticker} only "
                         f"in passing, never in a headline."), {}

    # Provider-sentiment hybrid (latency win): when enough fresh articles already
    # carry a provider sentiment (e.g. Polygon insights), score from those and
    # skip the LLM call entirely. Bypassed for force_engine (the opener-pinned
    # hold-review must re-judge with its OWN LLM engine for apples-to-apples).
    # No catalyst: provider insights carry a sentiment label, not an event class.
    #
    # `allow_provider` separates the two REASONS a caller forces an engine. A
    # hold review forces one to RE-JUDGE, and must not take the shortcut. The
    # news REPLAY forces one to PIN which engine answers, and must take it
    # wherever live did — measured 2026-09-12, the provider path serves ~42 of
    # ~130 scored tickers per tick (~32%), so a replay that always calls the LLM
    # diverges from live on a third of rows AND pays for calls live never made.
    _allow_provider = (force_engine is None) if allow_provider is None else bool(allow_provider)
    if _allow_provider:
        provider = _provider_sentiment_score(ticker, fresh_articles, as_of)
        if provider is not None:
            _record_sentiment_provider("provider")
            logger.info(f"{ticker} provider_sentiment={provider[0]:+.2f} (LLM scorer skipped)")
            return provider[0], provider[1], {}

    # Sort by recency weight descending; send the top slice to the LLM.
    # `digest_articles` is the shared definition — see its docstring.
    # Source tier BEFORE the top-20 cut, so the surviving slots go to the
    # articles that survived the tier rather than being filled by listicles the
    # cut happened to rank first.
    to_score = digest_articles(apply_source_tier(fresh_articles), as_of)

    # Build digest with recency indicator for the model. The age LABELS are part
    # of what the model reads ("3h ago" vs "5.2d ago" is the priced-in check), so
    # a replay must date them from its own tick too, not from today.
    digest = _digest_text(ticker, to_score, as_of, digest_override)

    # Ticker-free fixed prefix (shared → DeepSeek auto-caches it across tickers) +
    # per-ticker variable suffix. The prefix carries ALL instructions/examples; the
    # suffix carries only the target header (symbol, company name, fund flag)
    # and the news digest.
    header = _target_header(ticker)
    suffix = (f"\n\n{header}\n\n<news>\n{digest}\n</news>\n\n"
              "Respond with JSON only, no markdown.")
    _prefix = _prompt_pair()[0]
    prompt = _prefix + suffix

    raw_score = None
    rationale = "Analysis unavailable."
    catalyst: Optional[str] = None
    # The continuous reading of the same verdict (local engine + logprobs only).
    # It REPLACES the greedy value when available — see the local branch.
    _expected: Optional[float] = None
    _argmax: Optional[float] = None
    # The engine that actually PRODUCED the verdict — not `order[0]`, which is
    # only the one that led. A fallback answer belongs to the fallback, or the
    # shadow pass would pair a local-primary run's DeepSeek rescue against
    # DeepSeek itself and record it as a local verdict.
    _engine_used: Optional[str] = None

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
    cache_key = _sentiment_cache_key(ticker, order[0], to_score, extra=header)
    # Engine-free digest id + the digest text itself (the join key every
    # downstream catalyst surface uses; recorded before the cache lookup so a
    # cached verdict's digest reaches the store as well).
    digest_id = digest_id_for(ticker, to_score)
    # `store_digest=False` for the replay: `sentiment_digests` is the record of
    # what the LIVE scorer saw, and `news_replay` keeps no `digest_id`, so
    # replayed digests would land orphaned — unjoinable to the rows they came
    # from, stamped with whatever run id happened to be current, and counted
    # against the 180-day retention. ~34k rows of pure pollution on a full run.
    if store_digest:
        _record_digest(digest_id, ticker, to_score, digest)
    cached = _sentiment_cache_get(cache_key)
    if cached is not None:
        raw_score = float(cached["raw_score"])
        rationale = str(cached.get("rationale") or "Rationale unavailable (cached).")
        catalyst = normalize_catalyst(cached.get("catalyst"))
        _engine_used = str(cached.get("engine") or order[0])
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
            # Per-ATTEMPT, not per-call: the local branch sets these and only
            # then logs, tallies and writes the cache, so an exception in any of
            # those falls through to the next engine with the previous attempt's
            # logprob provenance still attached — a DeepSeek verdict wearing the
            # local model's `expected_score`.
            _expected = _argmax = None
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
                raw_score, rationale, catalyst = _parse_response_uncapped(response.choices[0].message.content.strip())
            elif engine == "local":
                local = _get_local()
                if local is None:           # engine off / unconfigured — next engine
                    continue
                # The LOCAL server's own dialect — never the hosted one. The
                # hosted routes take `enable_thinking` (DashScope) / `reasoning`
                # (OpenRouter); a local server ignores those SILENTLY, which is
                # indistinguishable from them working. See _local_extra_body:
                # with reasoning left on, the model burns the whole answer cap
                # on its chain and returns an empty verdict.
                # The sentiment prompt runs ~2.4k tokens at 10 articles and
                # ~4.7k at the 20-article cap, so it lives close to this
                # server's per-request context — and Ollama truncates the
                # OLDEST tokens (the rubric) silently rather than erroring. Both
                # guards RAISE, which falls through to the next engine: a
                # hosted verdict beats a local one scored without its rubric,
                # and an unparseable answer would land as a neutral 0.0 in the
                # 0.40-weight news method, indistinguishable from an abstention.
                _est = local_llm.check_fits(
                    prompt, context_tokens=settings.local_sentiment_context_tokens,
                    label=f"local sentiment {settings.local_sentiment_model} ({ticker})")
                # Logprobs are requested on the LOCAL engine only (2026-09-10):
                # they cost payload, not latency, and the continuous verdict they
                # buy is a fix for qwen3:8b's 0.05 grid specifically. Wrapped so
                # a server build without logprob support falls back to the plain
                # call rather than losing the verdict.
                _want_lp = bool(getattr(settings, "enable_logprob_expected_score", True))
                _kw = dict(model=settings.local_sentiment_model,
                           max_tokens=_SENTIMENT_MAX_TOKENS,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0, seed=_LLM_SEED,
                           extra_body=_local_extra_body())
                try:
                    response = local.chat.completions.create(
                        **_kw, logprobs=True,
                        top_logprobs=int(getattr(settings, "logprob_top_n", 5) or 5),
                    ) if _want_lp else local.chat.completions.create(**_kw)
                except Exception as _lp_err:                    # noqa: BLE001
                    if not _want_lp:
                        raise
                    logger.debug(f"[sentiment] {ticker}: logprobs refused "
                                 f"({str(_lp_err)[:60]}) — plain call")
                    response = local.chat.completions.create(**_kw)
                local_llm.check_reported(
                    getattr(getattr(response, "usage", None), "prompt_tokens", None),
                    estimate=_est,
                    context_tokens=settings.local_sentiment_context_tokens,
                    label=f"local sentiment {settings.local_sentiment_model} ({ticker})")
                _warn_if_empty_local(ticker, response)
                raw_score, rationale, catalyst = _parse_response_uncapped(response.choices[0].message.content.strip())
                if _want_lp:
                    from src.analysis.logprob_score import expected_score, tokens_of
                    _argmax = raw_score
                    _expected = expected_score(tokens_of(response), raw_score)
                    if _expected is not None:
                        # THE VERDICT, not a shadow (2026-09-10 user directive:
                        # improvements go 100% into production). The greedy
                        # value is kept as `argmax_score` for provenance only —
                        # nothing reads it, and there is no arm to compare.
                        raw_score = _expected
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
                raw_score, rationale, catalyst = _parse_response_uncapped(response.choices[0].message.content.strip())
            else:
                client = _get_haiku()
                message = client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=_SENTIMENT_MAX_TOKENS,
                    messages=[{"role": "user",
                               "content": _anthropic_user_content(_prefix, suffix, HAIKU_MODEL)}],
                    temperature=0,
                )
                raw_score, rationale, catalyst = _parse_response_uncapped(message.content[0].text.strip())
            logger.info(f"{ticker} raw_sentiment={raw_score:+.2f} ({engine}, {len(to_score)} articles)")
            _engine_used = engine
            if _tally:
                _record_sentiment_provider(engine)
            _sentiment_cache_put(cache_key, raw_score, rationale, engine, catalyst)
            break
        except Exception as e:
            last_err = e
            logger.warning(f"{ticker} sentiment via {engine} failed: {e}")
            if _tally:
                _record_engine_error(engine, e)

    if raw_score is None:
        logger.error(f"Sentiment analysis failed for {ticker}: {last_err}")
        if _tally:
            _record_sentiment_provider("none")
        return 0.0, f"Analysis error: {last_err}", {}

    # Catalyst label resolved BEFORE the cap, because the cap has to judge the
    # label the panel will store: `fund_catalyst_override` retypes a fund's
    # company-event class to `macro_sector`, and `macro_sector` is a class the
    # cap must NEVER touch. Applied the other way round, a fund labelled
    # `analyst` by the model was clipped and then persisted as `macro_sector` —
    # a row whose stored class and applied cap disagree, which would confound
    # exactly the re-measurement this cap is scheduled for.
    catalyst_raw = catalyst
    catalyst = fund_catalyst_override(ticker, catalyst_raw)

    # --- Catalyst-class conviction cap -------------------------------------
    # Applied HERE, not in the parser, so it lands on the value EVERY path
    # produced: a cached verdict, a fresh call, and — the one that made this
    # necessary — a local call whose logprob EXPECTATION replaced the greedy
    # value after `_parse_response` had already capped it, which silently
    # un-capped the engine carrying 100% of the primary route.
    #
    # The cache deliberately stores the UNCAPPED verdict (above), so editing
    # `catalyst_cap_classes` takes effect on the next tick instead of waiting
    # out the 3 h TTL — the cap is a clamp on a stored number, not a different
    # question to ask the model, so re-scoring to change it would be waste.
    raw_score = apply_catalyst_cap(raw_score, catalyst)
    if _argmax is not None:
        # keep the provenance columns on the SAME scale as the verdict, or the
        # "did the expectation beat the argmax" read they exist for would be
        # comparing a capped value against an uncapped one.
        _argmax = apply_catalyst_cap(_argmax, catalyst)
    if _expected is not None:
        # `expected_score` IS `raw_score` whenever it is present — that is the
        # documented contract of the column pair, and it has to hold after the
        # cap too. Assigning rather than re-capping makes it true by
        # construction instead of by applying the same clamp twice.
        _expected = raw_score

    # --- Precision adjustments (continuous since 2026-08-14, epoch "news") ---
    _n_fresh, _mass = attention_mass(to_score, as_of)
    evidence_scale  = _evidence_scale(_mass)
    diversity_scale = _source_diversity_scale(to_score)
    precision_scale = evidence_scale * diversity_scale

    adjusted_score = round(raw_score * precision_scale, 4)

    # Shadow pass (measurement only, off the critical path). Forced calls are
    # excluded: a hold review is already pinned to one engine on purpose, and
    # shadowing it would double the local server's load for a comparison the
    # entry-side rows already carry. The shadow row records the RAW label.
    if force_engine is None and _engine_used:
        _submit_shadow(ticker, to_score, _engine_used, raw_score, adjusted_score,
                       catalyst_raw, cache_key, digest_id=digest_id)
        # CLUSTER ARM (2026-09-08): the same digest, scored one cluster at a
        # time and blended. Off by default; decides nothing; drained with the
        # shadow rows.
        _submit_cluster_arm(ticker, to_score, _engine_used, raw_score,
                            adjusted_score, digest_id, as_of)
        # Catalyst-repair pass (2026-09-06): the PRIMARY verdict here; the shadow
        # verdict is submitted from `_run_shadow` under role="shadow" (own row),
        # and a hold review's label reaches no consumer. Background, fail-soft,
        # never a signal.
        try:
            from src.analysis import catalyst_repair
            catalyst_repair.maybe_submit(
                ticker=ticker, engine=_engine_used, first_pass=catalyst_raw,
                final_label=catalyst, rationale=rationale, digest_id=digest_id,
                digest_text=digest, articles=to_score)
        except Exception as e:                # noqa: BLE001 - measurement only
            logger.debug(f"[catalyst-repair] {ticker} not submitted: {e}")

    if precision_scale < 0.90:
        logger.debug(
            f"{ticker} sentiment scaled {raw_score:+.2f} → {adjusted_score:+.2f} "
            f"(count={len(to_score)}, sources={len({a.source for a in to_score})}, "
            f"scale={precision_scale:.2f})"
        )

    # `engine` is the engine that ANSWERED (a forced engine may have fallen
    # through); `digest` is the text it saw — both for the shadow pass, which
    # attributes its row and its repair submission on them.
    return adjusted_score, rationale, {"catalyst": catalyst, "catalyst_raw": catalyst_raw,
                                       "raw_score": raw_score, "digest_id": digest_id,
                                       "engine": _engine_used, "digest": digest,
                                       # continuous reading of the SAME verdict
                                       # (local + logprobs); None otherwise. When
                                       # present it IS `raw_score`; `argmax_score`
                                       # keeps the greedy value for provenance.
                                       "expected_score": _expected,
                                       "argmax_score": _argmax}


# --------------------------- SHADOW SENTIMENT ---------------------------
# Every ticker the primary engine scores with an LLM is ALSO scored by the OTHER
# engine on the SAME article set, so direction and magnitude can be compared per
# TICKER rather than per run. Three properties are load-bearing:
#
#   1. It is OFF the critical path. The local engine's measured ceiling is
#      ~0.54 calls/s (Ollama serves OLLAMA_NUM_PARALLEL=2 slots; client fan-out
#      past 2 only queues), so a ~175-call tick is ~325 s of shadow work - fine
#      beside a 30-min cadence, ruinous in front of the synthesis call. Rows are
#      drained by whichever `_persist_run` comes next, so a shadow that outlives
#      its own tick lands on the following write carrying its OWN run_id.
#   2. It never reaches the combine. The shadow verdict is persisted and nothing
#      else - no provider tally (forced calls do not tally), no score, no
#      fallback. A dead local server costs accrual, never a signal.
#   3. It scores the SAME digest: the shadow re-enters `analyse_sentiment` with
#      the primary's own `to_score` list, so a difference between the two
#      verdicts is a MODEL difference and not an input difference - the exact
#      confound that makes a cached head-to-head impossible (the sentiment cache
#      stores an article-set hash, not the articles).
_SHADOW_POOL = None
_SHADOW_LOCK = threading.Lock()
_SHADOW_ROWS: list = []
_SHADOW_PENDING = 0
_SHADOW_WARNED = False


def _shadow_engine_for(primary: str) -> Optional[str]:
    """The engine that should shadow *primary* this run, or None to skip."""
    if not getattr(settings, "enable_sentiment_shadow", False):
        return None
    pin = (getattr(settings, "sentiment_shadow_engine", "auto") or "auto").strip().lower()
    if pin and pin != "auto":
        engine = pin
    else:
        # The other side of the live pair: local shadows a hosted run, DeepSeek
        # shadows a local run, so every scored ticker accrues BOTH verdicts
        # whichever way the per-run flip landed.
        engine = "deepseek" if primary == "local" else "local"
    if engine == primary:
        return None
    if engine == "local" and not settings.enable_local_llm:
        return None
    if engine == "qwen" and not settings.qwen_api_key:
        return None
    if engine == "anthropic" and not settings.enable_claude_sentiment:
        return None
    return engine


def _shadow_sampled(ticker: str) -> bool:
    """Is this ticker in the shadow SAMPLE this run (`sentiment_shadow_share`)?

    Deterministic on (run_id, ticker) rather than a per-call random draw: the
    same ticker must get the same answer if it is scored twice in a run (a
    retry, a second digest), or the sample would quietly over-represent the
    tickers that failed once. Hash-based, so the sample is unbiased across
    tickers and stable across processes."""
    share = float(getattr(settings, "sentiment_shadow_share", 1.0) or 0.0)
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    key = f"{_CURRENT_RUN_ID or ''}|{ticker}".encode("utf-8")
    draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2 ** 64
    return draw < share


def _shadow_pool():
    """Lazily-built background pool. Two workers: the local server serves two
    slots and a third would only queue - measured 2026-09-04, throughput flat at
    0.54 calls/s from client concurrency 2 through 8 while per-call latency rose
    3.6 s to 13.0 s."""
    global _SHADOW_POOL
    if _SHADOW_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _SHADOW_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sent-shadow")
    return _SHADOW_POOL


def _run_shadow(run_id, ticker, articles, engine, digest_hash, primary):
    """One shadow verdict. Fail-soft: a shadow that raises is one missing row."""
    global _SHADOW_PENDING
    try:
        t0 = time.perf_counter()
        score, _rationale, meta = analyse_sentiment(ticker, articles, force_engine=engine)
        raw = meta.get("raw_score")
        if raw is None:                       # errored/degraded - nothing to compare
            return
        # The forced engine LEADS its order but the fallbacks are appended, so
        # the verdict may have come from another engine: attribute the row to
        # the one that ANSWERED, and drop a pair against the primary's own
        # engine — that is cost with no comparison in it.
        answered = str(meta.get("engine") or engine)
        if answered == primary.get("primary_engine"):
            logger.debug(f"[sentiment-shadow] {ticker}: forced {engine} answered by "
                         f"{answered} (the primary's engine) — self-pair dropped")
            return
        row = dict(primary)
        row.update({
            "run_id": run_id, "ticker": ticker, "digest_hash": digest_hash,
            "shadow_engine": answered, "shadow_model": sentiment_model_for(answered),
            "shadow_raw": float(raw), "shadow_score": float(score),
            # The engine's OWN label (the fund override is applied downstream
            # of both engines alike, so it would only mask the comparison).
            "shadow_catalyst": meta.get("catalyst_raw", meta.get("catalyst")),
            "shadow_latency_s": round(time.perf_counter() - t0, 3),
        })
        with _SHADOW_LOCK:
            _SHADOW_ROWS.append(row)
        # Catalyst-repair pass on the SHADOW first pass (own (digest_id, engine)
        # row, trigger stamped `<trigger>@shadow`) — the only place a local
        # first pass, a specialist verdict and a DeepSeek label coexist, which
        # is what the pre-registered `--eval` bar is measured on. `run_id` is
        # passed explicitly: this thread may outlive its tick.
        try:
            from src.analysis import catalyst_repair
            catalyst_repair.maybe_submit(
                ticker=ticker, engine=answered, first_pass=meta.get("catalyst_raw"),
                final_label=meta.get("catalyst"), rationale=_rationale,
                digest_id=primary.get("digest_id"), digest_text=meta.get("digest"),
                articles=articles, role="shadow", run_id=run_id)
        except Exception as e:                # noqa: BLE001 - measurement only
            logger.debug(f"[catalyst-repair] {ticker} shadow not submitted: {e}")
    except Exception as e:                    # noqa: BLE001 - measurement only
        logger.debug(f"[sentiment-shadow] {ticker} via {engine} failed: {e}")
    finally:
        with _SHADOW_LOCK:
            _SHADOW_PENDING -= 1


# --------------------------- CLUSTER ARM --------------------------------
# One call per NEWS CLUSTER instead of one call over the whole digest, blended
# by recency mass. Motivated by a measured anchoring failure: AAPL scored +0.35
# off a single bullish headline while the same digest carried a Senate China
# warning and Qualcomm guiding Apple revenue down (realized pivot -11.29%; both
# the cluster and thinking arms read it negative). Measured on 30 reconstructed
# digests, ONE day, NOT significant: direction hit 40% vs the single call's 31%
# and thinking-on's 41%, at 1.4x latency versus thinking-on's 11.8x.
#
# It DECIDES NOTHING. Like the shadow pass it runs on a background pool, is
# drained non-blocking by `_persist_run`, never tallies into
# `runs.llm_sentiment_provider`, and exists so the comparison can be made on
# LIVE digests instead of the reconstructed ones the pilot used.
_ARM_POOL = None
_ARM_LOCK = threading.Lock()
_ARM_ROWS: list = []
_ARM_PENDING = 0
_ARM_WARNED = False


def _arm_pool():
    global _ARM_POOL
    if _ARM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _ARM_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sent-arm")
    return _ARM_POOL


def _arm_sampled(ticker: str) -> bool:
    """Deterministic on (run_id, ticker), like the shadow sample: a RETRIED
    ticker must get the same decision, or the sample over-represents exactly
    the calls that failed once."""
    share = float(getattr(settings, "cluster_arm_share", 0.0) or 0.0)
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    h = hashlib.sha1(f"{_CURRENT_RUN_ID}|{ticker}|cluster-arm".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < share


def cluster_split(articles: List[NewsArticle],
                  as_of: Optional[datetime] = None) -> List[List[NewsArticle]]:
    """The digest partitioned into time clusters, freshest first — the same rule
    `news_priced_in.cluster_bounds` uses, so the two cannot disagree about what
    a cluster is."""
    out, rest = [], list(articles or [])
    guard = 0
    while rest and guard < 20:
        guard += 1
        g = recent_cluster(rest, as_of=as_of)
        if not g:
            break
        out.append(g)
        keep = {id(a) for a in g}
        rest = [a for a in rest if id(a) not in keep]
    return out


def _run_cluster_arm(ticker, articles, engine, primary_raw, primary_score,
                     digest_id, run_id, as_of=None) -> None:
    global _ARM_PENDING
    t0 = time.perf_counter()
    try:
        groups = cluster_split(articles, as_of)
        if len(groups) < 2:
            return                      # nothing to decompose — the arm IS the single call
        scores, sizes, num, den = [], [], 0.0, 0.0
        for g in groups:
            _s, _r, meta = analyse_sentiment(ticker, g, force_engine=engine, as_of=as_of)
            raw = (meta or {}).get("raw_score")
            # A pinned engine only LEADS its fallback order, so a fall-through
            # would mix engines inside one blend — drop the cluster instead.
            if raw is None or (meta or {}).get("engine") != engine:
                scores.append(None)
                sizes.append(len(g))
                continue
            w = sum(_recency_weight(a, as_of) for a in g)
            num, den = num + w * float(raw), den + w
            scores.append(round(float(raw), 6))
            sizes.append(len(g))
        if den <= 0:
            return
        arm_raw = round(num / den, 6)
        _n, mass = attention_mass(articles, as_of)
        arm_scaled = round(arm_raw * _evidence_scale(mass) * _source_diversity_scale(articles), 6)
        row = {"run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(),
               "ticker": (ticker or "").strip().upper(), "digest_id": digest_id,
               "engine": engine, "model": sentiment_model_for(engine),
               "n_articles": len(articles), "n_clusters": len(groups),
               "cluster_scores": json.dumps(scores), "cluster_sizes": json.dumps(sizes),
               "arm_raw": arm_raw, "arm_score": arm_scaled,
               "primary_raw": (None if primary_raw is None else float(primary_raw)),
               "primary_score": float(primary_score),
               "latency_s": round(time.perf_counter() - t0, 3)}
        with _ARM_LOCK:
            _ARM_ROWS.append(row)
    except Exception as e:                  # noqa: BLE001 - measurement only
        logger.debug(f"[cluster-arm] {ticker} failed: {e}")
    finally:
        with _ARM_LOCK:
            _ARM_PENDING -= 1


def _submit_cluster_arm(ticker, articles, engine, primary_raw, primary_score,
                        digest_id, as_of=None) -> None:
    """Queue the cluster arm for this ticker (never blocks)."""
    global _ARM_PENDING, _ARM_WARNED
    if not getattr(settings, "enable_cluster_arm", False) or primary_raw is None:
        return
    if not _arm_sampled(ticker):
        return
    cap = max(0, int(getattr(settings, "cluster_arm_max_pending", 200) or 0))
    with _ARM_LOCK:
        if cap and _ARM_PENDING >= cap:
            if not _ARM_WARNED:
                _ARM_WARNED = True
                logger.warning(f"[cluster-arm] {_ARM_PENDING} call(s) pending (cap {cap}) "
                               f"- skipping further arms; the {engine} engine is not keeping up")
            return
        _ARM_PENDING += 1
    run_id = _CURRENT_RUN_ID
    try:
        _arm_pool().submit(_run_cluster_arm, ticker, list(articles), engine,
                           primary_raw, primary_score, digest_id, run_id, as_of)
    except Exception as e:                  # noqa: BLE001
        with _ARM_LOCK:
            _ARM_PENDING -= 1
        logger.debug(f"[cluster-arm] {ticker} not queued: {e}")


def pop_cluster_arm_rows() -> List[dict]:
    """Drain finished rows (non-blocking — one still in flight lands on a later
    drain carrying its own run_id)."""
    with _ARM_LOCK:
        rows, _ARM_ROWS[:] = list(_ARM_ROWS), []
    return rows


def cluster_arm_pending() -> int:
    with _ARM_LOCK:
        return _ARM_PENDING


def _submit_shadow(ticker: str, articles: List[NewsArticle], primary_engine: str,
                   primary_raw: Optional[float], primary_score: float,
                   primary_catalyst: Optional[str], digest_hash: str,
                   digest_id: Optional[str] = None) -> None:
    """Queue the OTHER engine's verdict on this ticker's digest (never blocks)."""
    global _SHADOW_PENDING, _SHADOW_WARNED
    engine = _shadow_engine_for(primary_engine)
    if engine is None or primary_raw is None:
        return
    if not _shadow_sampled(ticker):
        return
    cap = max(0, int(getattr(settings, "sentiment_shadow_max_pending", 500) or 0))
    with _SHADOW_LOCK:
        if cap and _SHADOW_PENDING >= cap:
            if not _SHADOW_WARNED:
                _SHADOW_WARNED = True
                logger.warning(f"[sentiment-shadow] {_SHADOW_PENDING} call(s) pending "
                               f"(cap {cap}) - skipping further shadows; the "
                               f"{engine} engine is not keeping up")
            return
        _SHADOW_PENDING += 1
    primary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_articles": len(articles),
        "primary_engine": primary_engine,
        "primary_model": sentiment_model_for(primary_engine),
        "primary_raw": float(primary_raw),
        "primary_score": float(primary_score),
        "primary_catalyst": primary_catalyst,
        # Engine-free digest id: the join key shared with `signals.news_digest_id`
        # and `sentiment_digests` (`digest_hash` above stays the salted cache key).
        "digest_id": digest_id,
    }
    try:
        _shadow_pool().submit(_run_shadow, _CURRENT_RUN_ID, ticker, list(articles),
                              engine, digest_hash, primary)
    except Exception as e:                    # noqa: BLE001 - pool refused (shutdown)
        with _SHADOW_LOCK:
            _SHADOW_PENDING -= 1
        logger.debug(f"[sentiment-shadow] submit failed for {ticker}: {e}")


def pop_sentiment_shadow_rows() -> list:
    """Drain the shadow verdicts finished so far (each carries its own run_id).

    Non-blocking BY DESIGN: the persist step must never wait on a background
    LLM call. Whatever is still in flight is written by the next tick's drain."""
    global _SHADOW_WARNED
    with _SHADOW_LOCK:
        rows, _SHADOW_ROWS[:] = list(_SHADOW_ROWS), []
        _SHADOW_WARNED = False
    return rows


def sentiment_shadow_pending() -> int:
    """Shadow calls queued or running right now (0 when the pass has drained)."""
    with _SHADOW_LOCK:
        return _SHADOW_PENDING


def filter_relevant_articles(ticker: str, articles: List[NewsArticle]) -> List[NewsArticle]:
    """Articles about *ticker* — the input to the per-ticker sentiment digest.

    An article is relevant when a feed TAGGED it with the symbol (structured
    feeds tag what they are about; the search-derived feeds tag only what a
    mention confirmed — see ``news_fetcher._confirmed_tags``) or when its text
    MENTIONS the company (``company_names.mention_evidence``: an explicit
    ``(NYSE: AR)`` / ``$AR`` symbol, the registrant name or an alias as a whole
    phrase, or the bare symbol as a case-sensitive word for symbols that cannot
    be an ordinary acronym). Never the weakest single-token tier — that is for
    confirming a feed's own tag, not for sweeping the general pool.

    Returns ``[]`` below ``settings.news_relevance_min_articles`` rather than
    falling back to the whole pool — an unrelated digest is worse than none.

    Why (2026-09-04): the previous test was ``symbol.lower() in text`` — a bare
    substring, so ``"ar" in "market"`` was a hit and short symbols received the
    entire pool, capped to the 20 most recent random headlines; the scorer then
    correctly reported "about other companies" and abstained (measured 73–79%
    of calls, ~74% of the zero rationales). ``enable_name_relevance=False``
    restores that filter byte-for-byte.
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return []
    if not settings.enable_name_relevance:
        return _filter_relevant_articles_legacy(tkr, articles)
    from src.data.company_names import mentions

    relevant: List[NewsArticle] = []
    for a in articles:
        if tkr in (getattr(a, "tickers", None) or []):
            relevant.append(a)
            continue
        text = f"{a.title or ''} {a.summary or ''}"
        if mentions(tkr, text):
            relevant.append(a)
    min_n = max(1, int(settings.news_relevance_min_articles or 1))
    return relevant if len(relevant) >= min_n else []


def _filter_relevant_articles_legacy(ticker: str, articles: List[NewsArticle]) -> List[NewsArticle]:
    """The pre-2026-09-04 filter, kept verbatim behind ``enable_name_relevance=False``."""
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
        if tkr in (getattr(a, "tickers", None) or [])
        or any(kw in (a.title + a.summary).lower() for kw in keywords)
    ]
    return relevant if len(relevant) >= 2 else []
