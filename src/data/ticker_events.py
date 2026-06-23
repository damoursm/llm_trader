"""
Ticker events watch — Massive corporate ticker-events (symbol/name changes, delistings).

Protective, not a directional alpha source: a rename or delisting can silently strand
a held position, break the OHLCV cache (the old symbol stops pricing), or leave the
system tracking a dead ticker. This surfaces RECENT events on the held + watchlist
names as material NewsArticles — pre-scored via provider_insights (a delisting is
negative, a rename neutral) so the LLM scorer is skipped — so the synthesis / exit
logic sees them. Cached daily.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from config import settings
from src.data import polygon_client
from src.data.market_data import is_valid_ticker
from src.models import NewsArticle

CACHE_DIR = Path("cache")


def _cache_path() -> Path:
    return CACHE_DIR / f"ticker_events_{date.today().isoformat()}.json"


def _pdate(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def fetch_ticker_events(tickers: List[str]) -> List[NewsArticle]:
    """Recent ticker symbol/name changes + delistings on *tickers* → NewsArticles.

    Daily-cached. Returns [] when disabled / Polygon unavailable. Seed with the held
    + watchlist names (one call per ticker)."""
    if not settings.enable_ticker_events or not polygon_client.is_available():
        return []

    path = _cache_path()
    if path.exists():
        try:
            return [NewsArticle.model_validate(a)
                    for a in json.loads(path.read_text(encoding="utf-8"))]
        except Exception as e:
            logger.warning(f"[ticker_events] cache load failed: {e}")

    if not settings.enable_fetch_data:
        return []

    cutoff = date.today() - timedelta(days=settings.ticker_events_lookback_days)
    out: List[NewsArticle] = []
    for tk in dict.fromkeys(t.upper() for t in tickers if is_valid_ticker(t)):
        for ev in polygon_client.get_ticker_events(tk):
            ed = _pdate(ev.get("date"))
            etype = (ev.get("type") or "").lower().replace("_", " ").strip()
            if not ed or ed < cutoff:
                continue
            is_delist = "delist" in etype
            if not (is_delist or "change" in etype):
                continue
            advice = (
                "A delisting removes the security from trading — close / raise cash on any exposure and stop tracking it."
                if is_delist else
                "A symbol/name change means this ticker may be superseded — verify the active symbol before pricing or trading it."
            )
            out.append(NewsArticle(
                title=f"{tk}: {etype} effective {ed.isoformat()}",
                summary=f"Massive corporate ticker event for {tk}: {etype} on {ed.isoformat()}. {advice}",
                url="https://massive.com/",
                source="Ticker Events",
                published_at=datetime.combine(ed, datetime.min.time(), tzinfo=timezone.utc),
                tickers=[tk],
                provider_insights={tk: "negative" if is_delist else "neutral"},
                provider_sentiment_source="massive",
            ))

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(json.dumps([a.model_dump(mode="json") for a in out], default=str),
                        encoding="utf-8")
    except Exception as e:
        logger.warning(f"[ticker_events] cache save failed: {e}")

    if out:
        logger.info(f"[ticker_events] {len(out)} recent event(s): {[a.tickers[0] for a in out]}")
    return out
