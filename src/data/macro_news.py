"""Macro-news regime detector — scans the day's headlines for geopolitics,
trade/tariffs, energy shocks, central-bank moves, fiscal events, and black
swans, and produces a composite regime read.

Why this exists
---------------
Per-ticker sentiment captures stock-specific news. Numeric macro modules
(VIX, FRED, MOVE, credit, intermarket, …) capture market microstructure
and the slow-moving macro backdrop. Neither catches the **narrative** —
a war headline that just hit, an OPEC cut announced overnight, a tariff
escalation — which moves risk assets at the macro level long before it
shows up in FRED data or moves the VIX persistently.

This module reuses the news flow already fetched in step 1 of the pipeline
(no extra feed cost), pre-filters to macro-themed headlines, and either:
  * asks DeepSeek for a single structured macro classification call, or
  * falls back to a deterministic keyword-density heuristic when the key
    is absent.

The composite ``macro_news_score`` and ``composite_signal`` plug into the
Macro Regime Filter alongside VIX/MOVE/credit/breadth/intermarket so a
CRISIS-grade headline actually tightens the BUY threshold and blocks
new longs. The structured themes + sector tilts are also embedded in
the Claude analyst prompt so the LLM can lean trades toward (or away
from) the appropriate sectors (e.g., war → defense / oil / gold).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config import settings
from src.models import MacroNewsContext, MacroNewsTheme, NewsArticle


# Fixed seed for the DeepSeek macro-news classifier (OpenAI-compatible API
# supports the ``seed`` parameter). Combined with temperature=0 this gives
# near-deterministic regime labels across two runs over the same article set.
_MACRO_NEWS_SEED = 31337


# ─────────────────────────────────────────────────────────────────────────────
# Cache (hourly — matches news cache TTL)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "macro_news_"


def _cache_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{key}.json"


def _load_cache(key: str) -> Optional[MacroNewsContext]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        ctx = MacroNewsContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(
            f"[macro_news] Loaded from cache — {ctx.composite_signal} "
            f"score={ctx.macro_news_score:+.2f} themes={len(ctx.themes)}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[macro_news] Cache load failed: {e}")
        return None


def _save_cache(ctx: MacroNewsContext, key: str) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(key).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[macro_news] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword pre-filter
# ─────────────────────────────────────────────────────────────────────────────

# Each category maps to a list of keyword/phrase regexes (case-insensitive).
# Headlines/summaries matching any keyword for a category are tagged with that
# category. An article can match multiple categories.
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "geopolitical_conflict": [
        r"\bwar\b", r"\binvasion\b", r"\bairstrike\b", r"\bmissile\b",
        r"\bceasefire\b", r"\bsanctions?\b", r"\bnato\b", r"\bukraine\b",
        r"\brussia\b", r"\bisrael\b", r"\bhamas\b", r"\bhezbollah\b",
        r"\biran\b", r"\btaiwan\b", r"\bnorth korea\b", r"\bgaza\b",
        r"\bred sea\b", r"\bhouthi\b", r"\bgeopolitic\w*\b",
        r"\bescalation\b", r"\bmilitary strike\b",
    ],
    "trade_tariffs": [
        r"\btariff\w*\b", r"\btrade war\b", r"\btrade deal\b",
        r"\bexport controls?\b", r"\bimport ban\b", r"\bwto\b",
        r"\bsemiconductor (?:export|controls?|restrictions?)\b",
        r"\bchina (?:trade|tariffs?|sanctions?)\b",
        r"\bsection 301\b", r"\btrade representative\b", r"\bustr\b",
        r"\bcustoms\b", r"\bretaliat\w+\b",
    ],
    "energy_shock": [
        r"\bopec\b", r"\boil price\b", r"\bcrude\b", r"\bbrent\b", r"\bwti\b",
        r"\benergy crisis\b", r"\bpipeline\b", r"\brefinery (?:outage|fire|shutdown)\b",
        r"\bnatural gas\b", r"\boutput cut\b", r"\bproduction cut\b",
        r"\bbarrel\b", r"\boil sanctions\b", r"\benergy supply\b",
    ],
    "central_bank": [
        r"\bfed (?:hike|cut|pivot|pause|chair)\b", r"\bfomc\b",
        r"\brate (?:hike|cut|decision)\b", r"\bpowell\b", r"\bdovish\b",
        r"\bhawkish\b", r"\bquantitative (?:easing|tightening)\b",
        r"\bbank of japan\b", r"\bboj\b", r"\becb\b", r"\beuropean central bank\b",
        r"\bcurrency intervention\b", r"\byen intervention\b",
        r"\bpboc\b", r"\bpeople's bank of china\b",
    ],
    "fiscal_policy": [
        r"\bgovernment shutdown\b", r"\bdebt ceiling\b", r"\bcontinuing resolution\b",
        r"\bspending bill\b", r"\bfiscal cliff\b", r"\bstimulus package\b",
        r"\binfrastructure bill\b", r"\btax bill\b", r"\bbudget deal\b",
        r"\bcongress (?:passed|approved|vote)\b",
    ],
    "black_swan": [
        r"\bpandemic\b", r"\boutbreak\b", r"\bcyberattack\b", r"\bransomware\b",
        r"\bearthquake\b", r"\bhurricane\b", r"\btsunami\b",
        r"\bnuclear (?:incident|leak|test)\b", r"\bterrorist attack\b",
        r"\bflash crash\b", r"\bsystemic risk\b",
    ],
}

# Compile once at module load.
_CATEGORY_REGEXES: Dict[str, List[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _CATEGORY_KEYWORDS.items()
}


def _categorize_article(article: NewsArticle) -> List[str]:
    """Return the list of categories this article matches (may be empty)."""
    text = f"{article.title}  {article.summary}"
    hits: List[str] = []
    for cat, regexes in _CATEGORY_REGEXES.items():
        if any(rx.search(text) for rx in regexes):
            hits.append(cat)
    return hits


def _prefilter(articles: List[NewsArticle], cap: int) -> Tuple[List[NewsArticle], Dict[str, int]]:
    """Return (macro_articles, category_counts).

    Filters down to articles matching ANY macro category, capped at ``cap``.
    Sorted most-recent first so old headlines don't dilute the read.
    """
    matched: List[Tuple[NewsArticle, List[str]]] = []
    for a in articles:
        cats = _categorize_article(a)
        if cats:
            matched.append((a, cats))
    matched.sort(key=lambda pair: pair[0].published_at, reverse=True)
    matched = matched[:cap]

    counts: Dict[str, int] = {}
    for _, cats in matched:
        for c in cats:
            counts[c] = counts.get(c, 0) + 1

    return [a for a, _ in matched], counts


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Sector tilts per category — used by the heuristic fallback and merged
# with the LLM's per-theme tilts. ETF symbols are sector / commodity / region
# ETFs the system already tracks.
_DEFAULT_CATEGORY_TILTS: Dict[str, Dict[str, int]] = {
    "geopolitical_conflict": {"XLE": +1, "ITA": +1, "GLD": +1, "SLV": +1, "EEM": -1, "EFA": -1},
    "trade_tariffs":         {"EEM": -1, "EFA": -1, "XLI": -1, "XLB": -1, "GLD": +1},
    "energy_shock":          {"XLE": +1, "GLD": +1, "XLY": -1, "XLI": -1, "JETS": -1},
    "central_bank":          {"XLF": +1, "GLD": +1, "TLT": -1},  # tightening-biased default
    "fiscal_policy":         {"GLD": +1, "TLT": -1},
    "black_swan":            {"GLD": +1, "TLT": +1, "XLE": -1, "XLY": -1, "EEM": -1},
}

# Heuristic per-article weight by category — captures asymmetric importance.
# Higher = the theme moves markets harder when it fires.
_CATEGORY_WEIGHTS: Dict[str, float] = {
    "geopolitical_conflict": 0.20,
    "trade_tariffs":         0.15,
    "energy_shock":          0.15,
    "central_bank":          0.18,
    "fiscal_policy":         0.10,
    "black_swan":            0.30,
}


def _severity_from_count(n: int) -> str:
    if n >= 8:  return "EXTREME"
    if n >= 4:  return "HIGH"
    if n >= 2:  return "MEDIUM"
    return "LOW"


def _signal_from_score(score: float) -> str:
    if score <= -0.60:  return "CRISIS"
    if score <= -0.25:  return "ELEVATED_RISK"
    if score <= -0.10:  return "WATCH"
    return "STABLE"


def _heuristic_classify(
    articles: List[NewsArticle],
    category_counts: Dict[str, int],
) -> Tuple[List[MacroNewsTheme], float, Dict[str, int]]:
    """Build themes + composite score + merged sector tilts from keyword density.

    Used when DeepSeek isn't available. Deterministic and quick: every macro
    category with ≥1 article becomes a theme; severity scales with count;
    aggregate score is a capped sum of (count × category_weight) flipped to
    negative (macro headlines are overwhelmingly risk-off-leaning).
    """
    themes: List[MacroNewsTheme] = []
    raw_score = 0.0
    sector_tilts: Dict[str, int] = {}

    # Index articles by category for picking a representative headline.
    by_cat: Dict[str, List[NewsArticle]] = {c: [] for c in category_counts}
    for a in articles:
        for c in _categorize_article(a):
            if c in by_cat:
                by_cat[c].append(a)

    for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        if count <= 0:
            continue
        severity = _severity_from_count(count)
        weight = _CATEGORY_WEIGHTS.get(cat, 0.10)
        contribution = -min(0.75, count * weight)  # capped, sign = risk-off
        raw_score += contribution

        # Representative headline = most-recent article in this category.
        cat_articles = sorted(by_cat.get(cat, []), key=lambda a: a.published_at, reverse=True)
        headline = cat_articles[0].title if cat_articles else ""
        summary = (
            f"{count} {cat.replace('_', ' ')} article{'s' if count != 1 else ''} "
            f"in today's news flow"
        )

        tilts = _DEFAULT_CATEGORY_TILTS.get(cat, {})
        implications = [f"{etf}{'+' if v > 0 else '-'}" for etf, v in tilts.items()]

        themes.append(MacroNewsTheme(
            category=cat,
            severity=severity,
            direction="BEARISH" if contribution < -0.10 else "NEUTRAL",
            headline=headline,
            summary=summary,
            article_count=count,
            sector_implications=implications,
        ))

        # Merge tilts; the same ETF can appear in multiple categories — sum.
        for etf, v in tilts.items():
            sector_tilts[etf] = sector_tilts.get(etf, 0) + v

    score = round(max(-1.0, min(1.0, raw_score)), 3)
    return themes, score, sector_tilts


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek classifier
# ─────────────────────────────────────────────────────────────────────────────

_DEEPSEEK_PROMPT_TEMPLATE = """You are a senior macro strategist. Classify the macro-relevant headlines below into themes that move risk assets.

For EACH distinct theme that emerges from the headlines, output an object with:
- "category": one of {categories}
- "severity": "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
- "direction": "BULLISH" | "BEARISH" | "NEUTRAL" — for global risk assets (BEARISH = stocks down)
- "headline": one representative headline (verbatim from the input)
- "summary": one or two sentences summarising what's happening and why it matters for markets
- "article_count": integer — how many of the supplied headlines support this theme
- "sector_implications": array of strings like "XLE+", "ITA+", "GLD+", "EEM-" — use these ETF tickers when relevant: XLE (energy), XLF (financials), XLV (healthcare), XLY (consumer disc.), XLP (staples), XLI (industrials), XLB (materials), XLU (utilities), XLRE (real estate), XLK (tech), XLC (communications), GLD (gold), SLV (silver), GDX (gold miners), USO (oil), TLT (long bonds), EEM (emerging mkts), EFA (developed ex-US), ITA (defense), JETS (airlines), KRE (regional banks), SMH (semis). Suffix "+" for bullish or "-" for bearish.

Then output an overall assessment:
- "composite_signal": "STABLE" | "WATCH" | "ELEVATED_RISK" | "CRISIS"
- "macro_news_score": float in [-1.0, +1.0] — your aggregate read on near-term risk-asset bias from these headlines. Negative = stressed/bearish; positive = supportive
- "summary": 2–3 sentence narrative explaining the dominant story and how a portfolio manager should tilt

Return ONLY valid JSON with this exact shape:
{{
  "themes":            [...],
  "composite_signal":  "...",
  "macro_news_score":  ...,
  "summary":           "..."
}}

Headlines:
{headlines}
"""

_CATEGORIES_LIST = list(_CATEGORY_KEYWORDS.keys()) + ["other"]


def _deepseek_classify(
    articles: List[NewsArticle],
) -> Optional[Tuple[List[MacroNewsTheme], float, str, str, Dict[str, int]]]:
    """Call DeepSeek to classify the macro headlines. Returns None on any failure.

    Returns ``(themes, score, composite_signal, summary, sector_tilts)``.
    """
    if not settings.deepseek_api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")
    except Exception as e:
        logger.warning(f"[macro_news] DeepSeek client init failed: {e}")
        return None

    # Build prompt. Keep title-only to save tokens — summaries are noisy.
    lines = [
        f"- [{a.source} @ {a.published_at.strftime('%Y-%m-%d %H:%M UTC')}] {a.title}"
        for a in articles
    ]
    prompt = _DEEPSEEK_PROMPT_TEMPLATE.format(
        categories=", ".join(f'"{c}"' for c in _CATEGORIES_LIST),
        headlines="\n".join(lines),
    )

    try:
        # Determinism: temperature=0 + fixed seed so two runs on the same
        # filtered headline set produce the same theme classification.
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=_MACRO_NEWS_SEED,
            max_tokens=2400,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},  # non-thinking: cheap/fast/deterministic
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"[macro_news] DeepSeek call failed: {e}")
        return None

    # Parse — DeepSeek in JSON-mode returns clean JSON, but defensive strip
    # for ``` fences just in case.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"[macro_news] DeepSeek JSON parse failed: {e}")
        return None

    themes: List[MacroNewsTheme] = []
    sector_tilts: Dict[str, int] = {}
    for t in data.get("themes", []) or []:
        try:
            category = str(t.get("category", "other")).lower()
            if category not in _CATEGORIES_LIST:
                category = "other"
            severity = str(t.get("severity", "LOW")).upper()
            if severity not in ("LOW", "MEDIUM", "HIGH", "EXTREME"):
                severity = "LOW"
            direction = str(t.get("direction", "NEUTRAL")).upper()
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                direction = "NEUTRAL"
            implications = [str(x) for x in (t.get("sector_implications") or [])]
            themes.append(MacroNewsTheme(
                category=category,
                severity=severity,
                direction=direction,
                headline=str(t.get("headline", ""))[:300],
                summary=str(t.get("summary", ""))[:400],
                article_count=int(t.get("article_count", 0) or 0),
                sector_implications=implications,
            ))
            # Merge tilts
            for s in implications:
                m = re.match(r"^([A-Z]{1,5})([+-])$", s.strip())
                if m:
                    etf, sign = m.group(1), m.group(2)
                    sector_tilts[etf] = sector_tilts.get(etf, 0) + (1 if sign == "+" else -1)
        except Exception as e:
            logger.debug(f"[macro_news] Skipping bad theme entry: {e}")
            continue

    try:
        score = max(-1.0, min(1.0, float(data.get("macro_news_score", 0.0))))
    except Exception:
        score = 0.0
    composite = str(data.get("composite_signal", "STABLE")).upper()
    if composite not in ("STABLE", "WATCH", "ELEVATED_RISK", "CRISIS"):
        composite = _signal_from_score(score)
    summary = str(data.get("summary", ""))[:1000]

    return themes, round(score, 3), composite, summary, sector_tilts


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_macro_news_context(articles: Optional[List[NewsArticle]] = None) -> Optional[MacroNewsContext]:
    """Build a MacroNewsContext from today's news articles.

    Cache-first (hourly key). When the cache is empty, scan and classify
    via DeepSeek if available; otherwise fall back to the deterministic
    keyword-density heuristic so the feature still works without a key.
    Returns ``None`` only when the feature flag is off or the article list
    is too small to read.
    """
    if not settings.enable_macro_news:
        return None

    key = _cache_key()
    cached = _load_cache(key)
    if cached is not None:
        return cached

    articles = articles or []
    if len(articles) < int(settings.macro_news_min_articles):
        logger.info(f"[macro_news] Only {len(articles)} article(s) — skipping macro scan")
        return None

    macro_articles, counts = _prefilter(articles, cap=int(settings.macro_news_max_articles))
    if not macro_articles:
        logger.info("[macro_news] No macro-themed articles in today's flow → STABLE")
        ctx = MacroNewsContext(
            themes=[], composite_signal="STABLE", macro_news_score=0.0,
            sector_tilts={}, articles_scanned=len(articles), used_llm=False,
            report_date=date.today(),
            summary="No macro-themed headlines detected — no regime overlay applied.",
        )
        _save_cache(ctx, key)
        return ctx

    # Try DeepSeek first; fall back to heuristic on any failure.
    llm_result = _deepseek_classify(macro_articles)
    used_llm = False
    if llm_result is not None:
        themes, score, signal, summary, sector_tilts = llm_result
        used_llm = True
    else:
        themes, score, sector_tilts = _heuristic_classify(macro_articles, counts)
        signal = _signal_from_score(score)
        top_theme = themes[0].category.replace("_", " ") if themes else "—"
        summary = (
            f"Heuristic read (no DeepSeek key): {len(themes)} active macro theme(s), "
            f"led by {top_theme}. Score={score:+.2f} → {signal}."
        )

    if not themes:
        # No themes after parsing — degenerate state, treat as STABLE.
        signal = "STABLE"
        score = 0.0

    # Sort themes by severity then article count (clearest stories first)
    _SEV_ORDER = {"EXTREME": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    themes.sort(key=lambda t: (_SEV_ORDER.get(t.severity, 9), -t.article_count))

    ctx = MacroNewsContext(
        themes=themes,
        composite_signal=signal,
        macro_news_score=score,
        sector_tilts=sector_tilts,
        articles_scanned=len(macro_articles),
        used_llm=used_llm,
        report_date=date.today(),
        summary=summary,
    )

    logger.info(
        f"[macro_news] {signal}  score={score:+.2f}  themes={len(themes)}  "
        f"articles_scanned={len(macro_articles)}  llm={'yes' if used_llm else 'heuristic'}"
    )
    _save_cache(ctx, key)
    return ctx
