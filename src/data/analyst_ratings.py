"""
Analyst upgrades, downgrades, and price-target changes.

Source: yfinance ticker.upgrades_downgrades (free, no key required).
Returns List[NewsArticle] — injected into the articles list and scored by the
existing DeepSeek sentiment pipeline, same as 8-K and EPS surprise articles.

Signal logic:
  - Any upgrade (Action=="up") or downgrade (Action=="down"): always surface.
  - New coverage initiation with a bullish/bearish grade (Action=="init"): surface.
  - ≥2 firms raising price targets in the same direction: consensus signal.
  - Maintained/reiterated-only rows with no PT movement: filtered as noise.

Cached daily — analyst actions rarely change intraday.
"""

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import NewsArticle

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.35

# Grade → sentiment mapping
_BULLISH_GRADES = frozenset({
    "Buy", "Strong Buy", "Outperform", "Overweight", "Market Outperform",
    "Positive", "Top Pick", "Sector Outperform", "Add", "Accumulate",
    "Long-Term Buy",
})
_BEARISH_GRADES = frozenset({
    "Sell", "Strong Sell", "Underperform", "Underweight", "Market Underperform",
    "Negative", "Reduce", "Below Average", "Underperformer",
})

# Price-target action strings from yfinance
_PT_RAISE = frozenset({"Raises", "Initiated"})
_PT_CUT   = frozenset({"Lowers"})


def _cache_path() -> Path:
    return CACHE_DIR / f"analyst_ratings_{date.today().isoformat()}.json"


def _load_cache() -> Optional[List[NewsArticle]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[analyst] Loaded {len(articles)} cached analyst articles")
        return articles
    except Exception as e:
        logger.warning(f"[analyst] Cache load failed: {e}")
        return None


def _save_cache(articles: List[NewsArticle]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps([a.model_dump(mode="json") for a in articles], indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[analyst] Cache save failed: {e}")


def _grade_sentiment(grade: str) -> str:
    """Return 'bullish', 'bearish', or 'neutral' for a grade string."""
    g = grade.strip()
    if g in _BULLISH_GRADES:
        return "bullish"
    if g in _BEARISH_GRADES:
        return "bearish"
    return "neutral"


def _build_article(ticker: str, rows) -> Optional[NewsArticle]:
    """
    Build a NewsArticle from a filtered DataFrame of recent analyst actions.
    Returns None if the signal is too weak to surface.
    """
    upgrades    = rows[rows["Action"] == "up"]
    downgrades  = rows[rows["Action"] == "down"]
    initiations = rows[rows["Action"] == "init"]
    maintained  = rows[rows["Action"].isin({"main", "reit"})]

    # PT movements on maintained/reiterated coverage
    pt_raises = maintained[maintained["priceTargetAction"].isin(_PT_RAISE)]
    pt_cuts   = maintained[maintained["priceTargetAction"].isin(_PT_CUT)]

    # Determine if we have a meaningful signal to surface
    n_up   = len(upgrades)
    n_down = len(downgrades)
    n_init_bull = sum(1 for _, r in initiations.iterrows() if _grade_sentiment(r.get("ToGrade", "")) == "bullish")
    n_init_bear = sum(1 for _, r in initiations.iterrows() if _grade_sentiment(r.get("ToGrade", "")) == "bearish")
    n_pt_up   = len(pt_raises)
    n_pt_down = len(pt_cuts)

    # Signal threshold: at least one of these must be true
    has_signal = (
        n_up >= 1 or
        n_down >= 1 or
        n_init_bull >= 1 or
        n_init_bear >= 1 or
        n_pt_up >= 2 or
        n_pt_down >= 2
    )
    if not has_signal:
        return None

    # Overall direction
    bull_score = n_up * 2 + n_init_bull * 1.5 + n_pt_up * 0.5
    bear_score = n_down * 2 + n_init_bear * 1.5 + n_pt_down * 0.5
    if bull_score > bear_score:
        direction = "bullish"
        direction_word = "positive"
    elif bear_score > bull_score:
        direction = "bearish"
        direction_word = "negative"
    else:
        direction = "mixed"
        direction_word = "mixed"

    # Build action bullet list for the summary
    bullets: List[str] = []
    for _, r in upgrades.iterrows():
        from_g = r.get("FromGrade", "")
        to_g   = r.get("ToGrade", "")
        firm   = r.get("Firm", "Unknown")
        pt_str = f" (PT: ${r['currentPriceTarget']:.0f})" if r.get("currentPriceTarget") and r["currentPriceTarget"] > 0 else ""
        bullets.append(f"{firm} upgraded from {from_g} → {to_g}{pt_str}")

    for _, r in downgrades.iterrows():
        from_g = r.get("FromGrade", "")
        to_g   = r.get("ToGrade", "")
        firm   = r.get("Firm", "Unknown")
        pt_str = f" (PT: ${r['currentPriceTarget']:.0f})" if r.get("currentPriceTarget") and r["currentPriceTarget"] > 0 else ""
        bullets.append(f"{firm} downgraded from {from_g} → {to_g}{pt_str}")

    for _, r in initiations.iterrows():
        firm = r.get("Firm", "Unknown")
        to_g = r.get("ToGrade", "")
        pt_str = f" (PT: ${r['currentPriceTarget']:.0f})" if r.get("currentPriceTarget") and r["currentPriceTarget"] > 0 else ""
        bullets.append(f"{firm} initiated coverage: {to_g}{pt_str}")

    # Summarise notable PT moves on maintained coverage (top 3 by PT change magnitude)
    if n_pt_up >= 2 or n_pt_down >= 2:
        pool = pt_raises if n_pt_up >= n_pt_down else pt_cuts
        for _, r in pool.head(3).iterrows():
            firm = r.get("Firm", "Unknown")
            cur  = r.get("currentPriceTarget")
            prior = r.get("priorPriceTarget")
            if cur and prior and cur > 0 and prior > 0:
                chg = (cur - prior) / prior * 100
                action = "raised" if cur > prior else "cut"
                bullets.append(f"{firm} {action} PT ${prior:.0f} → ${cur:.0f} ({chg:+.0f}%)")

    # Compute average current PT across all rows that have one
    pts = [r["currentPriceTarget"] for _, r in rows.iterrows()
           if r.get("currentPriceTarget") and r["currentPriceTarget"] > 0]
    avg_pt_str = f" Average analyst PT: ${sum(pts)/len(pts):.0f} across {len(pts)} coverage note(s)." if pts else ""

    n_firms = rows["Firm"].nunique()
    n_actions = n_up + n_down + len(initiations)

    # Build title
    if n_up > 0 and n_down == 0:
        verb = f"{n_up} upgrade(s)"
    elif n_down > 0 and n_up == 0:
        verb = f"{n_down} downgrade(s)"
    elif n_up > 0 and n_down > 0:
        verb = f"{n_up} upgrade(s) and {n_down} downgrade(s)"
    elif n_init_bull > 0 or n_init_bear > 0:
        verb = f"{len(initiations)} new coverage initiation(s)"
    else:
        verb = f"{n_pt_up + n_pt_down} price-target revision(s)"

    title = f"Analyst ratings: {ticker} sees {verb} from {n_firms} firm(s)"

    summary = (
        f"{ticker} received {n_actions + n_pt_up + n_pt_down} analyst action(s) from "
        f"{n_firms} firm(s) in the recent lookback window — overall {direction_word} tone. "
        + " ".join(f"{b}." for b in bullets[:5])
        + avg_pt_str
        + (
            " Multiple analyst upgrades and rising consensus price targets signal growing institutional conviction — "
            "a strong confirming layer for a bullish thesis."
            if direction == "bullish" and n_up >= 2 else
            " Multiple downgrades signal deteriorating institutional sentiment — "
            "a confirming layer for a bearish thesis."
            if direction == "bearish" and n_down >= 2 else
            ""
        )
    )

    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://finance.yahoo.com/quote/{ticker}/analysis/",
        source="Analyst Ratings",
        published_at=datetime.now(timezone.utc),
    )


def fetch_analyst_ratings(
    tickers: List[str],
    lookback_days: int = 30,
) -> List[NewsArticle]:
    """
    Fetch recent analyst upgrades/downgrades/PT changes for each ticker.

    Args:
        tickers: list of ticker symbols to check
        lookback_days: how far back to look for analyst actions (default 30 days)

    Returns:
        List[NewsArticle] — one per ticker with a meaningful analyst signal.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[analyst_ratings] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles: List[NewsArticle] = []

    for sym in tickers:
        try:
            t  = yf.Ticker(sym)
            ud = t.upgrades_downgrades

            if ud is None or ud.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Index is GradeDate (DatetimeIndex, timezone-aware or naive depending on yfinance version)
            idx = ud.index
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC")
            recent = ud[idx >= cutoff]

            if recent.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            article = _build_article(sym, recent)
            if article:
                articles.append(article)
                n_up   = len(recent[recent["Action"] == "up"])
                n_down = len(recent[recent["Action"] == "down"])
                logger.info(
                    f"[analyst] {sym}: {len(recent)} action(s) — "
                    f"{n_up} up, {n_down} down, {recent['Firm'].nunique()} firms"
                )

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[analyst] {sym} fetch failed: {e}")
            time.sleep(_REQUEST_DELAY)

    logger.info(f"[analyst] {len(articles)} analyst articles from {len(tickers)} tickers")
    _save_cache(articles)
    return articles
