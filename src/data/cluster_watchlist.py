"""Insider Cluster Watchlist — persistent cross-run tracking of insider cluster signals.

When ≥3 different insiders buy the same ticker within 5 days (a cluster), the aggregator
fires a 1.75× score amplifier on the day of detection.  But historically, insider clusters
precede price movement by 5–20 days — not just the day the signal is detected.

This module extends that single-day signal into a 10-day tracking window:
  • On detection: ticker is added to `cache/cluster_watchlist.json` with the detection date.
  • On every subsequent run within 10 days: ticker is injected into the analysis universe
    so it is always re-evaluated, even if it has fallen off the trending/discovery list.
  • After 10 days: the entry expires and is removed from the watchlist.

The watchlist is a simple JSON dict keyed by ticker:
  {
    "AAPL": {
      "detected_at": "2026-04-15",
      "cluster_size": 4,
      "insider_summary": "Tim Cook (CEO) + 3 others — $500k–$1M"
    },
    ...
  }
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.models import ClusterWatchEntry, ClusterWatchlistContext

WATCHLIST_PATH = Path("cache") / "cluster_watchlist.json"
WATCH_DAYS = 10   # how many days after detection to keep the ticker on watch


# ── Persistence ───────────────────────────────────────────────────────────────

def load_cluster_watchlist() -> Dict[str, dict]:
    """Load the raw watchlist dict from disk.  Returns {} if not found or corrupt."""
    if not WATCHLIST_PATH.exists():
        return {}
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[cluster_watch] Failed to load watchlist: {e}")
        return {}


def save_cluster_watchlist(raw: Dict[str, dict]) -> None:
    """Persist the raw watchlist dict to disk."""
    WATCHLIST_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"[cluster_watch] Failed to save watchlist: {e}")


# ── Update logic ──────────────────────────────────────────────────────────────

def update_cluster_watchlist(
    signals_by_ticker: dict,           # Dict[str, TickerSignal]
    raw: Dict[str, dict],
    today: Optional[date] = None,
) -> Dict[str, dict]:
    """
    1. Add any newly detected clusters (insider_cluster_detected=True on a TickerSignal).
    2. Expire entries older than WATCH_DAYS.
    Returns the updated raw dict.
    """
    today = today or date.today()

    # 1. Add new clusters
    for ticker, sig in signals_by_ticker.items():
        if getattr(sig, "insider_cluster_detected", False) and ticker not in raw:
            raw[ticker] = {
                "detected_at": str(today),
                "cluster_size": getattr(sig, "insider_cluster_size", 3),
                "insider_summary": getattr(sig, "insider_summary", ""),
            }
            logger.info(
                f"[cluster_watch] NEW cluster entry: {ticker} "
                f"({raw[ticker]['cluster_size']} insiders) — watching for {WATCH_DAYS} days"
            )

    # 2. Expire stale entries
    cutoff = today - timedelta(days=WATCH_DAYS)
    expired = [
        t for t, v in raw.items()
        if date.fromisoformat(v.get("detected_at", "2000-01-01")) < cutoff
    ]
    for t in expired:
        logger.info(f"[cluster_watch] Expired: {t} (>{WATCH_DAYS}d since detection)")
        del raw[t]

    return raw


# ── Context builder ───────────────────────────────────────────────────────────

def build_cluster_watchlist_context(
    raw: Dict[str, dict],
    today: Optional[date] = None,
) -> ClusterWatchlistContext:
    """Convert the raw dict into a typed context object for the email / pipeline."""
    today = today or date.today()
    entries: List[ClusterWatchEntry] = []

    for ticker, v in sorted(raw.items()):
        detected_at = date.fromisoformat(v.get("detected_at", str(today)))
        days_elapsed  = (today - detected_at).days
        days_remaining = max(0, WATCH_DAYS - days_elapsed)
        entries.append(ClusterWatchEntry(
            ticker=ticker,
            detected_at=detected_at,
            cluster_size=v.get("cluster_size", 3),
            insider_summary=v.get("insider_summary", ""),
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
        ))

    active_tickers = [e.ticker for e in entries]

    if not entries:
        summary = "No active insider cluster watches."
    else:
        parts = [f"{e.ticker} ({e.days_remaining}d left)" for e in entries]
        summary = f"Cluster watch active for {len(entries)} ticker(s): {', '.join(parts)}"

    return ClusterWatchlistContext(
        entries=entries,
        active_tickers=active_tickers,
        summary=summary,
    )
