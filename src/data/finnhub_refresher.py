"""Finnhub company-news for EVERY scored name inside the free tier's 60 calls/min
(2026-09-25, the all-source news ingestion — `src/data/news_coverage.py`).

The Finnhub leg asked about the first 60 names of a tick, inline, one request
per name. At 60 calls/min a ~400-name universe would hold Step 1 for ~7
minutes, so the calls move OFF the tick: a daemon thread in the scheduler
process keeps a per-ticker cache of the SAME request the leg makes
(`provider_news.finnhub_company_news`: from = today-3d, to = today, newest
first, noise dropped, 15 kept) over the last tick's universe, stalest first, and
`provider_news.fetch_finnhub_news` reads it at the tick.

Served entries must have been fetched TODAY (the request's date window moves at
midnight) and within ``finnhub_cache_max_age_seconds``; a name without one is
fetched inline through the SAME limiter, up to ``finnhub_inline_budget`` per
tick. The price is freshness: at ``finnhub_refresh_cycle_seconds`` (600) every
name is re-asked every ~10 minutes against a 30-minute tick, while Google News,
RSS and Polygon are still fetched at the tick itself.

The thread idles when no tick has handed it targets for
``_IDLE_AFTER_SECONDS`` (weekends, holidays) and the scheduler wakes it before
each slot (`keepalive`). Everything is fail-soft: a dead refresher degrades to
the inline budget — the leg's pre-2026-09-25 behaviour.

It publishes its state (``STATE_PATH``: active / idle + a timestamp) because
another PROCESS shares the key's 60 calls/min: the Finnhub history backfill
(`src/analysis/news_finnhub_backfill.py`) pulls only while this reports idle.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from config import settings

CACHE_PATH = Path("cache/finnhub_news_cache.json")
STATE_PATH = Path("cache/finnhub_refresher_state.json")
# No targets for this long -> idle. Every weekday gap between two slots is at
# most 90 min (23:30 -> 01:00 overnight) and the runner wakes the thread 20 min
# before each slot, so 2 h never idles it on a trading day, while the weekend
# window (Friday's last tick -> Sunday 20:10) opens ~2 h after Friday's 19:50
# tick instead of ~6 h later: the backfill's quota.
_IDLE_AFTER_SECONDS = 2 * 3600
_STATE_EVERY_SECONDS = 30
_SAVE_EVERY_SECONDS = 300            # persist the cache so a restart is not a cold start
_RATE_LIMIT_BACKOFF_SECONDS = 65.0   # a 429 blocks every caller for a full minute window

_LOCK = threading.Lock()
_ENTRIES: Dict[str, dict] = {}       # TICKER -> {"t": epoch s, "day": iso date, "items": [...]}
_TARGETS: List[str] = []
_TARGETS_AT = 0.0
_THREAD: Optional[threading.Thread] = None
_STOP = threading.Event()
_STATS = {"fetched": 0, "failed": 0, "rate_limited": 0}


class _Limiter:
    """Minimum spacing between Finnhub requests, shared by the refresher and the
    tick's inline fetches, with a hard block after a 429."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0
        self._blocked_until = 0.0

    def acquire(self, per_minute: float, max_wait: Optional[float] = None) -> bool:
        """Reserve the next slot and sleep until it; False (nothing reserved)
        when the wait would exceed ``max_wait``."""
        spacing = 60.0 / max(1.0, float(per_minute))
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next, self._blocked_until)
            if max_wait is not None and start - now > max_wait:
                return False
            self._next = start + spacing
        wait = start - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        return True

    def block(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)


_LIMITER = _Limiter()


def _per_minute() -> float:
    return float(getattr(settings, "finnhub_calls_per_minute", 50))


def cached_items(ticker: str, max_age_seconds: float) -> Optional[Tuple[List[dict], float]]:
    """``(items, age_s)`` for a usable cache entry, else None."""
    with _LOCK:
        e = _ENTRIES.get(str(ticker).upper())
    if not e or e.get("day") != date.today().isoformat():
        return None
    age = time.time() - float(e.get("t", 0))
    if age > max_age_seconds:
        return None
    return list(e.get("items") or []), age


def fetch_now(ticker: str, lookback_days: int = 3, max_per_ticker: int = 15,
              max_wait: Optional[float] = None) -> Optional[List[dict]]:
    """Fetch one ticker through the shared limiter and cache it. None on any
    failure (a 429 also blocks the limiter), or when the limiter's wait would
    exceed ``max_wait``."""
    from src.data import provider_news as pn
    tk = str(ticker).upper()
    if not _LIMITER.acquire(_per_minute(), max_wait=max_wait):
        return None
    try:
        items, _dropped = pn.finnhub_company_news(tk, lookback_days=lookback_days,
                                                  max_per_ticker=max_per_ticker)
    except pn.FinnhubRateLimited:
        _LIMITER.block(_RATE_LIMIT_BACKOFF_SECONDS)
        with _LOCK:
            _STATS["rate_limited"] += 1
        logger.warning(f"[finnhub] rate limited at {tk} — every caller blocked "
                       f"{_RATE_LIMIT_BACKOFF_SECONDS:.0f}s")
        return None
    except Exception as exc:                          # noqa: BLE001 — a skip, not a stop
        with _LOCK:
            _STATS["failed"] += 1
        logger.debug(f"[finnhub] {tk} failed: {exc}")
        return None
    with _LOCK:
        _ENTRIES[tk] = {"t": time.time(), "day": date.today().isoformat(), "items": items}
        _STATS["fetched"] += 1
    return items


def set_targets(tickers: Sequence[str]) -> None:
    """The universe to keep fresh — called by the pipeline with each tick's
    final universe."""
    global _TARGETS, _TARGETS_AT
    names = list(dict.fromkeys(str(t).strip().upper() for t in (tickers or []) if str(t).strip()))
    with _LOCK:
        _TARGETS = names
        _TARGETS_AT = time.time()


def keepalive() -> None:
    """Keep the current targets live (the scheduler calls this before a slot so
    the first tick after a weekend finds a warm cache)."""
    global _TARGETS_AT
    with _LOCK:
        if _TARGETS:
            _TARGETS_AT = time.time()


def _next_due(now: float, cycle: float) -> Tuple[Optional[str], float]:
    """The stalest target and how long until it is due (0 = now)."""
    today = date.today().isoformat()
    with _LOCK:
        targets = list(_TARGETS)
        entries = dict(_ENTRIES)
    best, best_age = None, -1.0
    for tk in targets:
        e = entries.get(tk)
        age = float("inf") if not e or e.get("day") != today else now - float(e.get("t", 0))
        if age > best_age:
            best, best_age = tk, age
    if best is None:
        return None, cycle
    return best, max(0.0, cycle - best_age)


def _save() -> None:
    try:
        with _LOCK:
            payload = json.dumps(_ENTRIES)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, CACHE_PATH)
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"[finnhub] cache save failed: {exc}")


def _load() -> int:
    """Today's entries from disk; returns how many."""
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"[finnhub] cache unreadable: {exc}")
        return 0
    today = date.today().isoformat()
    keep = {str(k).upper(): v for k, v in (data or {}).items()
            if isinstance(v, dict) and v.get("day") == today}
    with _LOCK:
        for k, v in keep.items():
            if k not in _ENTRIES or float(_ENTRIES[k].get("t", 0)) < float(v.get("t", 0)):
                _ENTRIES[k] = v
    return len(keep)


def write_state(active: bool, path: Optional[Path] = None) -> None:
    """Publish active / idle for the other processes sharing the key (fail-soft)."""
    path = Path(path) if path else STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"active": bool(active), "at": time.time(), "pid": os.getpid()}),
                       encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"[finnhub] state write failed: {exc}")


def read_state(path: Optional[Path] = None) -> Optional[dict]:
    """The refresher's last published state, or None (never written / unreadable)."""
    path = Path(path) if path else STATE_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                 # noqa: BLE001
        return None


def _loop() -> None:
    cycle = float(getattr(settings, "finnhub_refresh_cycle_seconds", 600))
    last_save = time.time()
    last_state = 0.0
    while not _STOP.is_set():
        now = time.time()
        with _LOCK:
            idle = (not _TARGETS) or (now - _TARGETS_AT > _IDLE_AFTER_SECONDS)
        if now - last_state >= _STATE_EVERY_SECONDS:
            write_state(not idle)
            last_state = now
        if idle:
            _STOP.wait(30)
            continue
        tk, wait = _next_due(now, cycle)
        if tk is None or wait > 0:
            _STOP.wait(min(30.0, max(1.0, wait)))
        else:
            fetch_now(tk)
        if time.time() - last_save > _SAVE_EVERY_SECONDS:
            _save()
            last_save = time.time()


def start() -> bool:
    """Start the refresher thread (idempotent). Seeds the targets from the last
    tick's saved universe and the entries from the disk cache, so a scheduler
    restart mid-session serves the next tick warm."""
    global _THREAD, _TARGETS, _TARGETS_AT
    if not (getattr(settings, "enable_finnhub_refresher", True)
            and settings.enable_finnhub_news and settings.finnhub_api_key):
        return False
    if _THREAD is not None and _THREAD.is_alive():
        return True
    n = _load()
    try:
        from src.data import news_coverage
        prev = news_coverage.load_feed_universe()
        saved = json.loads(news_coverage.FEED_UNIVERSE_PATH.read_text(encoding="utf-8")).get("saved_at")
        from datetime import datetime
        saved_ts = datetime.fromisoformat(str(saved)).timestamp() if saved else 0.0
    except Exception:                                 # noqa: BLE001
        prev, saved_ts = [], 0.0
    with _LOCK:
        if prev and not _TARGETS:
            _TARGETS = prev
            _TARGETS_AT = saved_ts
    _STOP.clear()
    _THREAD = threading.Thread(target=_loop, name="finnhub-refresher", daemon=True)
    _THREAD.start()
    logger.info(f"[finnhub] refresher started: {len(prev)} target(s) from the last tick, "
                f"{n} cached entr{'y' if n == 1 else 'ies'} from today; "
                f"{_per_minute():.0f} calls/min, cycle "
                f"{float(getattr(settings, 'finnhub_refresh_cycle_seconds', 600)):.0f}s")
    return True


def stop(timeout: float = 5.0) -> None:
    _STOP.set()
    if _THREAD is not None:
        _THREAD.join(timeout)
    write_state(False)


def is_running() -> bool:
    return _THREAD is not None and _THREAD.is_alive()


def status() -> dict:
    """Coverage snapshot for logs / tests."""
    now = time.time()
    today = date.today().isoformat()
    with _LOCK:
        ages = [now - float(e.get("t", 0)) for e in _ENTRIES.values() if e.get("day") == today]
        return {"targets": len(_TARGETS), "entries_today": len(ages), "running": is_running(),
                "median_age_s": sorted(ages)[len(ages) // 2] if ages else None, **_STATS}


def reset() -> None:
    """Test hook — drop every entry, target and counter."""
    global _TARGETS, _TARGETS_AT, _LIMITER
    with _LOCK:
        _ENTRIES.clear()
        _TARGETS = []
        _TARGETS_AT = 0.0
        for k in _STATS:
            _STATS[k] = 0
    _LIMITER = _Limiter()
