"""File-based cache for news articles and market snapshots.

Cache files are stored in cache/ and keyed by YYYY-MM-DD_HH so:
- Runs within the same hour reuse cached data (good for dev iteration).
- Each new hour fetches fresh live data and saves a new snapshot.
- Old files accumulate as a local historical archive.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from loguru import logger

from src.models import NewsArticle, TickerSnapshot

CACHE_DIR = Path("cache")


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(exist_ok=True)


def _hour_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")


def _news_path(key: str) -> Path:
    return CACHE_DIR / f"news_{key}.json"


def _snapshots_path(key: str) -> Path:
    return CACHE_DIR / f"snapshots_{key}.json"


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def load_news(key: Optional[str] = None) -> Optional[List[NewsArticle]]:
    """Return cached articles for the given hour key, or None if not cached."""
    path = _news_path(key or _hour_key())
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[cache] Loaded {len(articles)} news articles from {path.name}")
        return articles
    except Exception as e:
        logger.warning(f"[cache] Failed to load news cache {path.name}: {e}")
        return None


def save_news(articles: List[NewsArticle], key: Optional[str] = None) -> None:
    """Persist articles to the cache for the current hour."""
    _ensure_dir()
    path = _news_path(key or _hour_key())
    try:
        data = [a.model_dump(mode="json") for a in articles]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"[cache] Saved {len(articles)} news articles → {path.name}")
    except Exception as e:
        logger.warning(f"[cache] Failed to save news cache: {e}")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def load_snapshots(key: Optional[str] = None) -> Optional[List[TickerSnapshot]]:
    """Return cached snapshots for the given hour key, or None if not cached."""
    path = _snapshots_path(key or _hour_key())
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        snaps = [TickerSnapshot.model_validate(s) for s in data]
        logger.info(f"[cache] Loaded {len(snaps)} snapshots from {path.name}")
        return snaps
    except Exception as e:
        logger.warning(f"[cache] Failed to load snapshots cache {path.name}: {e}")
        return None


def save_snapshots(snapshots: List[TickerSnapshot], key: Optional[str] = None) -> None:
    """Persist snapshots to the cache for the current hour."""
    _ensure_dir()
    path = _snapshots_path(key or _hour_key())
    try:
        data = [s.model_dump(mode="json") for s in snapshots]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"[cache] Saved {len(snapshots)} snapshots → {path.name}")
    except Exception as e:
        logger.warning(f"[cache] Failed to save snapshots cache: {e}")


# ---------------------------------------------------------------------------
# Latest-snapshot fallback (used when live fetch is disabled)
# ---------------------------------------------------------------------------

def load_latest_snapshots() -> Optional[List[TickerSnapshot]]:
    """
    Return the most recently saved snapshot file, regardless of hour key.
    Used when ENABLE_FETCH_DATA=false so the pipeline still has price context
    from the last successful fetch.
    """
    if not CACHE_DIR.exists():
        return None
    files = sorted(CACHE_DIR.glob("snapshots_*.json"), reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            snaps = [TickerSnapshot.model_validate(s) for s in data]
            logger.info(f"[cache] Loaded {len(snaps)} snapshots from historical cache {path.name}")
            return snaps
        except Exception as e:
            logger.warning(f"[cache] Skipping corrupt snapshot file {path.name}: {e}")
    return None


# ---------------------------------------------------------------------------
# OHLCV cache (per-ticker, daily — used by chart builder)
# ---------------------------------------------------------------------------

OHLCV_DIR = CACHE_DIR / "ohlcv"


def _ohlcv_path(ticker: str) -> Path:
    return OHLCV_DIR / f"{ticker.upper()}.json"


def load_ohlcv(ticker: str) -> Optional["pd.DataFrame"]:
    """Return cached OHLCV DataFrame for a ticker, or None if not cached."""
    import pandas as pd
    path = _ohlcv_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_json(path, orient="split")
        df.index = pd.to_datetime(df.index)
        logger.debug(f"[cache] Loaded OHLCV for {ticker} from {path.name} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"[cache] Failed to load OHLCV cache for {ticker}: {e}")
        return None


def save_ohlcv(ticker: str, df: "pd.DataFrame") -> None:
    """Persist OHLCV DataFrame to disk, overwriting any previous version."""
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    path = _ohlcv_path(ticker)
    try:
        path.write_text(df.to_json(orient="split", date_format="iso"), encoding="utf-8")
        logger.debug(f"[cache] Saved OHLCV for {ticker} → {path.name}")
    except Exception as e:
        logger.warning(f"[cache] Failed to save OHLCV for {ticker}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_cached_keys() -> List[str]:
    """Return all hour keys that have both news and snapshots cached."""
    if not CACHE_DIR.exists():
        return []
    news_keys = {p.stem.replace("news_", "") for p in CACHE_DIR.glob("news_*.json")}
    snap_keys = {p.stem.replace("snapshots_", "") for p in CACHE_DIR.glob("snapshots_*.json")}
    return sorted(news_keys & snap_keys)
