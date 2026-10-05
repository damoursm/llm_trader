"""ALL-SOURCE news coverage: every per-ticker news feed asks about EVERY name
the tick scores (2026-09-25, user directive: the news features must accrue from
2026-09-28 — test set 2 — under the updated all-source ingestion).

WHY
---
Step 1 fetches per-ticker news on the universe as it stands BEFORE the
smart-money, macro-discovery and cointegration-peer additions, which add ~270 of
the ~400 names a tick scores. Measured on three RTH runs of 2026-09-25: of ~253
smart-money names only ~68 had ANY relevant digest (1.4 articles on average),
against ~100% of the pre-fetch names — the per-ticker feeds (Google News,
Finnhub, the yfinance bundle, 8-K, analyst, EPS, short interest) had simply
never been asked about them. Finnhub was further capped at the first 60 names
and Google News at the first 150.

HOW, without serializing the tick
---------------------------------
1. PREVIOUS-UNIVERSE PREFETCH. Each tick writes its FINAL universe to
   ``cache/news_feed_universe.json``; the next tick's Step-1 news legs ask about
   Step 0's universe PLUS those names, inside the same parallel fetch pool. The
   additions are nearly the same names tick after tick (the smart-money list
   comes from date-keyed filing caches), so the pool absorbs them.
2. TOP-UP. Once the universe is final, the per-ticker legs run for the names
   neither list held — usually a handful (`pipeline._news_topup`).
3. PER-FILE COVERAGE SIDECARS. The date-keyed event caches (analyst ratings, EPS
   surprises, short interest, ticker events) and the hourly yfinance bundle held
   whatever the FIRST call of the day / hour asked for and served it to every
   later call, so a name that joined later never got those feeds. A sidecar
   under ``cache/coverage/`` records which tickers each cache file covers; a
   later call fetches only the missing names and appends them. Sidecars live in
   their own directory because `cache/news_*.json` is globbed as bundle files
   (`news_replay`) — a sidecar beside them would be read as one.

A cache file with NO sidecar (written before this module) has unknown coverage
and is treated as covering nothing: the next call refetches every name it is
asked about and merges on (url, title), so nothing is lost or duplicated.

FEED ATTRIBUTION. `feed_attribution` maps each article of the tick's pool to
EVERY leg that delivered it (the pool keeps only the first copy of a URL), so
`news_article_feeds` can rebuild any single feed's pool for a past run — the
per-source feature groups of `news_history`, forward from the deploy.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from loguru import logger

CACHE_DIR = Path("cache")
COVERAGE_DIR = CACHE_DIR / "coverage"
FEED_UNIVERSE_PATH = CACHE_DIR / "news_feed_universe.json"
# A previous universe older than this is not used for the prefetch — long enough
# to span a weekend or a holiday (Friday's names still describe Monday's), short
# enough that a scheduler stopped for a week does not fetch a stale list.
FEED_UNIVERSE_MAX_AGE_HOURS = 96.0

# The instant the all-source ingestion went LIVE (the scheduler restart that
# deployed it, UTC ISO — compared as a string against `signals.generated_at`).
# Per-ticker news coverage roughly tripled for the ~270 names that join the
# universe after the fetch, so a consumer whose meaning is the LEVEL of a name's
# coverage relative to its own history — `news_shock`'s attention baseline —
# must not mix the two regimes (the standing rule: fit only data the current
# code produced). Set from the ACTUAL restart, never from intent.
ALL_SOURCE_SINCE = "2026-09-25T20:30:00+00:00"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _norm(tickers: Iterable[str]) -> List[str]:
    """Upper-cased, order-preserving, de-duplicated, blanks dropped."""
    return list(dict.fromkeys(str(t).strip().upper() for t in (tickers or []) if str(t).strip()))


# ── 1. previous-universe prefetch ─────────────────────────────────────────────

def save_feed_universe(tickers: Sequence[str], run_id: str = "",
                       path: Optional[Path] = None) -> None:
    """Record the tick's FINAL universe (fail-soft: a lost file costs one tick's
    prefetch, which the top-up then covers)."""
    path = Path(path) if path else FEED_UNIVERSE_PATH
    try:
        _atomic_write(path, json.dumps({
            "run_id": str(run_id or ""),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "tickers": _norm(tickers)}))
    except Exception as exc:                          # noqa: BLE001
        logger.warning(f"[news-coverage] could not save the feed universe: {exc}")


def load_feed_universe(max_age_hours: float = FEED_UNIVERSE_MAX_AGE_HOURS,
                       path: Optional[Path] = None,
                       now: Optional[datetime] = None) -> List[str]:
    """The last tick's final universe, or [] when missing, unreadable or older
    than ``max_age_hours``."""
    path = Path(path) if path else FEED_UNIVERSE_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        saved = datetime.fromisoformat(str(data.get("saved_at")))
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=timezone.utc)
        if (now or datetime.now(timezone.utc)) - saved > timedelta(hours=max_age_hours):
            return []
        return _norm(data.get("tickers") or [])
    except FileNotFoundError:
        return []
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"[news-coverage] feed universe unreadable: {exc}")
        return []


def news_tickers(step0: Sequence[str], previous: Sequence[str]) -> List[str]:
    """Step 0's universe first (its order is what the per-ticker feeds walk),
    then the previous tick's other names in their stored order."""
    head = list(step0 or [])
    seen = {str(t).upper() for t in head}
    return head + [t for t in _norm(previous) if t not in seen]


# ── 2. per-file coverage sidecars ─────────────────────────────────────────────

def coverage_path(cache_file: Path) -> Path:
    return COVERAGE_DIR / Path(cache_file).name


def load_covered(cache_file: Path) -> Optional[Set[str]]:
    """Tickers the cache file covers; None when no sidecar exists (unknown)."""
    p = coverage_path(cache_file)
    try:
        return {str(t).upper() for t in json.loads(p.read_text(encoding="utf-8"))}
    except FileNotFoundError:
        return None
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"[news-coverage] sidecar {p.name} unreadable: {exc}")
        return None


def save_covered(cache_file: Path, tickers: Iterable[str]) -> None:
    try:
        _atomic_write(coverage_path(cache_file), json.dumps(sorted({str(t).upper() for t in tickers})))
    except Exception as exc:                          # noqa: BLE001
        logger.warning(f"[news-coverage] could not save sidecar for {Path(cache_file).name}: {exc}")


def _article_key(a) -> tuple:
    return (str(getattr(a, "url", "") or ""), str(getattr(a, "title", "") or ""))


def merge_articles(first: Sequence, second: Sequence) -> list:
    """``first`` then the articles of ``second`` not already present, keyed on
    (url, title) — NOT on the URL alone: several builders stamp one URL on
    every article they make (ticker events, Quiver contracts), and a URL-only
    merge would collapse them to one."""
    seen = {_article_key(a) for a in first}
    out = list(first)
    for a in second:
        k = _article_key(a)
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def fetch_with_coverage(label: str, cache_file: Path, tickers: Sequence[str],
                        load_cached: Callable[[], Optional[list]],
                        save_cached: Callable[[list], None],
                        fetch_missing: Callable[[List[str]], Optional[list]]) -> list:
    """Serve a date/hour-keyed article cache INCREMENTALLY: fetch only the
    tickers its sidecar does not cover, append, record them as covered.

    A name that was asked about and produced nothing is still COVERED (asking
    again the same day would return the same nothing). ``fetch_missing`` raising
    leaves the cache and sidecar untouched, so the next call retries."""
    cached = load_cached()
    covered = load_covered(cache_file) if cached is not None else set()
    if cached is None:
        cached = []
    if covered is None:                               # legacy file: coverage unknown
        covered = set()
    missing = [t for t in _norm(tickers) if t not in covered]
    if not missing:
        return cached
    new = fetch_missing(missing) or []
    merged = merge_articles(cached, new)
    save_cached(merged)
    save_covered(cache_file, covered | set(missing))
    logger.info(f"[{label}] +{len(missing)} ticker(s) fetched into today's cache "
                f"({len(new)} article(s); {len(covered) + len(missing)} covered)")
    return merged


# ── 3. feed attribution ───────────────────────────────────────────────────────

def url_hash(article) -> Optional[str]:
    """The archive's key (`repo.insert_news_articles`): sha1 of the URL, or of
    the title when there is no URL."""
    url = str(getattr(article, "url", "") or "").strip()
    title = str(getattr(article, "title", "") or "").strip()
    if not url and not title:
        return None
    return hashlib.sha1((url or title).encode("utf-8", errors="replace")).hexdigest()


def feed_attribution(chunks: Dict[str, Optional[Sequence]]) -> Dict[str, Dict[str, list]]:
    """``{url_hash: {leg: tickers-as-that-leg-tagged-them}}`` over every leg's
    articles BEFORE the pool's URL dedupe — the one place an article's second
    and third deliverers are still visible. The first copy per (hash, leg)
    wins, mirroring the pool."""
    out: Dict[str, Dict[str, list]] = {}
    for leg, arts in (chunks or {}).items():
        for a in arts or []:
            h = url_hash(a)
            if h is None:
                continue
            legs = out.setdefault(h, {})
            if leg not in legs:
                legs[leg] = [str(t).upper() for t in (getattr(a, "tickers", None) or [])]
    return out
