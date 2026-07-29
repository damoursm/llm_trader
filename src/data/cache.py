"""File-based cache for news articles and market snapshots.

Cache files are stored in cache/ and keyed by YYYY-MM-DD_HH so:
- Runs within the same hour reuse cached data (good for dev iteration).
- Each new hour fetches fresh live data and saves a new snapshot.
- Old files accumulate as a local historical archive.
"""

import json
from collections import OrderedDict
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


def _ohlcv_dir(interval: str = "1d") -> Path:
    """OHLCV cache directory for an interval. Daily keeps the legacy path
    (``cache/ohlcv/``) so every existing caller is byte-for-byte unchanged;
    non-daily timeframes get a sibling namespace (``cache/ohlcv_30m/`` …)."""
    return OHLCV_DIR if interval == "1d" else CACHE_DIR / f"ohlcv_{interval}"


def _ohlcv_path(ticker: str, interval: str = "1d") -> Path:
    return _ohlcv_dir(interval) / f"{ticker.upper()}.json"


# Parsed-OHLCV memo, keyed by (path, mtime_ns, size) — see load_ohlcv.
#
# Bounded by BYTES, not entry count: a 30-min frame averages ~54 KB against a
# daily frame's ~8 KB, so an entry cap either thrashes on the intraday files or
# over-commits on the daily ones. The analytics panels sweep ~1,760 tickers in
# BOTH timeframes (~111 MB), and an undersized cache made each panel re-parse
# what the previous one had just evicted. Eviction is oldest-first (dicts keep
# insertion order), which fits the scan-everything access pattern better than
# true LRU bookkeeping.
_OHLCV_PARSE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_OHLCV_PARSE_BYTES = 0


def _ohlcv_cache_budget() -> int:
    from config.settings import settings
    return max(0, int(getattr(settings, "ohlcv_parse_cache_mb", 0) or 0)) * 1_000_000


def clear_ohlcv_parse_cache() -> None:
    """Drop the parsed-OHLCV memo (tests / forced refresh)."""
    global _OHLCV_PARSE_BYTES
    _OHLCV_PARSE_CACHE.clear()
    _OHLCV_PARSE_BYTES = 0


def ohlcv_parse_cache_stats() -> dict:
    """``{entries, mb, budget_mb}`` — for diagnostics/tests."""
    return {"entries": len(_OHLCV_PARSE_CACHE),
            "mb": round(_OHLCV_PARSE_BYTES / 1e6, 1),
            "budget_mb": round(_ohlcv_cache_budget() / 1e6, 1)}


def trim_trailing_nan_bars(df):
    """Drop trailing bars whose Close is NaN — no information, silently toxic.

    Only the TAIL, and only on a missing Close (the column every indicator
    reads). Returns the frame unchanged when the last bar is valid, so the
    common path costs one scalar check.
    """
    if df is None or getattr(df, "empty", True) or "Close" not in getattr(df, "columns", []):
        return df
    import pandas as pd          # imported lazily, as everywhere else in this module
    close = pd.to_numeric(df["Close"], errors="coerce")
    if not bool(pd.isna(close.iloc[-1])):
        return df                                  # fast path: tail is fine
    keep = len(close)
    while keep > 0 and bool(pd.isna(close.iloc[keep - 1])):
        keep -= 1
    return df.iloc[:keep]


def load_ohlcv(ticker: str, interval: str = "1d") -> Optional["pd.DataFrame"]:
    """Return cached OHLCV DataFrame for a ticker, or None if not cached.

    ``interval`` selects the timeframe namespace ("1d" = legacy daily cache,
    "30m" = the intraday cache, etc.).

    The parsed frame is memoised on (path, mtime, size) and returned as a COPY.
    ``pd.read_json`` was the single largest cost in the dashboard's cold load —
    4,151 parses, ~20s — because the analytics layers sweep the same few
    thousand tickers repeatedly. A copy costs microseconds against milliseconds
    to re-parse, and keeps the long-standing safety property that made this
    deliberately uncached before: callers (notably ``market_data.get_history``)
    MUTATE the frame they get back, so they must never share one. Keying on
    mtime+size means a pipeline rewrite invalidates the entry automatically, so
    this is safe in the writer process too.
    """
    import pandas as pd
    path = _ohlcv_path(ticker, interval)
    try:
        st = path.stat()
    except OSError:
        return None                      # missing / unreadable — same as before
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _OHLCV_PARSE_CACHE.get(key)
    if hit is not None:
        return hit.copy()
    try:
        df = pd.read_json(path, orient="split")
        df.index = pd.to_datetime(df.index)
        # Trailing rows with no Close carry NO information, and a consumer
        # handed one does not fail — it returns a DIFFERENT answer, silently
        # (2026-07-25). Trimmed HERE, at the single cache-read entry point,
        # so every consumer is covered: the scorers that go through
        # market_data.get_history AND the ones that read the cache directly
        # (agreement's tape confirmation, classic_anomalies, anchored_vwap,
        # cointegration, extended_session). Measured: one trailing NaN bar
        # flipped `tech` −0.065 → +0.400 and silently blanked the tape
        # confirmation from BULLISH_TAPE 0.704 to NO_DATA. Trimming happens
        # once per PARSE (the frame is then memoised), so the hot path is
        # unaffected. Interior NaNs are LEFT ALONE — a real gap in a real
        # series, which the indicators tolerate and whose removal would
        # silently change bar spacing.
        df = trim_trailing_nan_bars(df)
        logger.debug(f"[cache] Loaded OHLCV[{interval}] for {ticker} from {path.name} ({len(df)} rows)")
        budget = _ohlcv_cache_budget()
        if budget:
            global _OHLCV_PARSE_BYTES
            try:
                nbytes = int(df.memory_usage(deep=True).sum())
            except Exception:
                nbytes = st.st_size                      # good enough to bound growth
            if nbytes <= budget:                         # never evict all for one giant frame
                _OHLCV_PARSE_CACHE[key] = df
                _OHLCV_PARSE_BYTES += nbytes
                while _OHLCV_PARSE_BYTES > budget and len(_OHLCV_PARSE_CACHE) > 1:
                    _, victim = _OHLCV_PARSE_CACHE.popitem(last=False)
                    try:
                        _OHLCV_PARSE_BYTES -= int(victim.memory_usage(deep=True).sum())
                    except Exception:
                        _OHLCV_PARSE_BYTES = sum(
                            int(v.memory_usage(deep=True).sum())
                            for v in _OHLCV_PARSE_CACHE.values())
                        break
        return df.copy()
    except Exception as e:
        logger.warning(f"[cache] Failed to load OHLCV[{interval}] cache for {ticker}: {e}")
        return None


def save_ohlcv(ticker: str, df: "pd.DataFrame", interval: str = "1d") -> None:
    """Persist OHLCV DataFrame to disk, overwriting any previous version.

    Atomic (temp file + ``os.replace``): OHLCV caches are written from many
    threads (the scorer pool's cache-miss fetches, the tracker's open-trade
    refresh, the liquidity gate's pre-warm — some concurrent with readers), and
    a direct truncate-write let a concurrent reader catch a half-written file.
    The temp name embeds the thread id so concurrent writers of the SAME ticker
    can't collide on the temp file; last replace wins, and readers always see a
    complete document either way.
    """
    import os
    import threading
    _ohlcv_dir(interval).mkdir(parents=True, exist_ok=True)
    path = _ohlcv_path(ticker, interval)
    tmp = path.with_name(f"{path.name}.tmp{threading.get_ident()}")
    try:
        tmp.write_text(df.to_json(orient="split", date_format="iso"), encoding="utf-8")
        os.replace(tmp, path)
        logger.debug(f"[cache] Saved OHLCV[{interval}] for {ticker} → {path.name}")
    except Exception as e:
        logger.warning(f"[cache] Failed to save OHLCV[{interval}] for {ticker}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


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
