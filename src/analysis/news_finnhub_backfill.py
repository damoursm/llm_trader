"""FINNHUB news history, one year back, for the whole liquid universe
(2026-09-25, user directive: "Start the finnhub pull and scoring this weekend.
Continue the next weekend's if needed.").

WHAT IT BUILDS
--------------
Rows in `news_replay` under pool_spec ``pre:finnhub``: for every NYSE session D
from 2025-10-06 through 2026-09-24 and every name the model arrays call liquid
that day (`ml30`'s ``sel30`` arrays: price >= $5 and 20-session dollar volume
>= $5M; ~1,985 names a day, 2,492 distinct, ~484k name-days), the whole news
family computed
from that name's OWN Finnhub feed as of 08:30 ET on D:

* the input is live's request (`provider_news.finnhub_company_news`: from =
  D-3, to = D) answered from the provider's history, one request per
  (ticker, week) and a per-day re-ask when a week comes back at the ~250-item
  cap; live's rule is applied at the instant (`news_history._finnhub_rule`:
  published <= 08:30 ET, UTC dates [D-3, D], newest first, noise dropped, the
  15 newest kept);
* the digest is scored ALONE through today's news code on the local engine
  (`news_replay._replay_one`: relevance, the v7dir verdict with its logprob
  expectation, the analyst cap, the passing-mention abstention, clustering, the
  derived family), `news_shock`'s baseline from this pool spec's own rows.

WHY 08:30 ET AND THE NAME'S OWN FEED
------------------------------------
08:30 ET is the knowledge cutoff every deep feature uses (`deep_features`), so
a 30-minute or daily model row of session D reads this without look-ahead, and
the same cutoff can be served live. The OWN feed keeps the feature independent
of which other names were asked about: the backfill's universe (~2,000 liquid
names) is not live's scored universe (~400), and a digest pooled across names
would change with that choice. Live has the same input for every scored name
since the all-source ingestion (the Finnhub refresher's cache at 08:30, or the
archive's `news_article_feeds` rows with feed ``finnhub``).

This is NOT the set-1 ``src:finnhub`` group: `news_history` rebuilt the leg as
live ASKED it on each live run (the first 60 names, the last run of the day,
the run's pooled relevance) to measure fidelity to live. This one is a MODEL
feature. Rows never mix: pool_spec ``pre:finnhub`` / ``pre:events``, run ids
``pre-<date>-b<band>``, replay_version ``finnhub-preopen-v1`` /
``events-preopen-v1``.

LIMITS AND SCHEDULE
-------------------
* Finnhub's free tier: 60 calls/min per key, and history is a ROLLING window
  (measured 2026-09-25: the first served day was 2025-09-30, so the window is
  ~360 days and moves a day a day). The pull therefore goes OLDEST WEEK FIRST —
  the next to expire — most liquid names first within a week; D starts at
  2025-10-06 so that [D-3, D] was served when the pull began.
* The key is shared with the live Finnhub refresher (50 calls/min whenever
  ticks run), so the pull runs only while the refresher reports IDLE
  (`finnhub_refresher.STATE_PATH`; it idles ~2 h after Friday's last tick and
  wakes 20 min before Sunday's first), keeps `x-ratelimit-remaining` above a
  floor, waits out any 429, and pauses while a live tick fetches.
* Scoring holds the local LLM only while no live tick scores sentiment. It runs
  band by band (500 names, most liquid first) and oldest day first within a
  band, because `news_shock`'s baseline reads a name's own earlier rows.
* ``--window weekend`` (default) works Saturday 01:00 -> Sunday 19:55 ET and
  sleeps through the week; one process spans as many weekends as it needs.
* Resumable throughout: raw answers are cached per (ticker, week) / (ticker,
  day) as trimmed gzip JSON (`RAW_DIR`), and a (day, band) is written whole or
  not at all.

THE EVENTS SOURCE (``pre:events``: same days, names and cutoff) rebuilds live's
event feeds from the deep store through live's own builders — the comment at
`EventTables` says what each leg reads and where it is approximate. It needs no
provider quota, so it can run on weekdays (``--window any``), holding the LLM
only between live ticks' sentiment phases.

CLI
    python -m src.analysis.news_finnhub_backfill --plan
    python -m src.analysis.news_finnhub_backfill --acquire [--window weekend|any] [--until ISO]
    python -m src.analysis.news_finnhub_backfill --score [--source finnhub|events]
                                                 [--window weekend|any] [--until ISO]
    python -m src.analysis.news_finnhub_backfill --status [--source finnhub|events]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from config.settings import settings

SPEC = "pre:finnhub"
VERSIONS = {"finnhub": "finnhub-preopen-v1", "events": "events-preopen-v1"}
VERSION = VERSIONS["finnhub"]
CUTOFF_ET = dtime(8, 30)
START = date(2025, 10, 6)
BAND_SIZE = 500
ARRAYS_DIR = Path("cache/ml/sel30")
ROOT = Path("cache/news_hist/finnhub_pre")
MIN_PRICE, MIN_DV = 5.0, 5e6
CAP = 240                     # at/above this a window came back truncated (the newest ~250)
PACE_S = 60.0 / 55            # the free tier is 60 calls/min; leave a little room
REMAINING_FLOOR = 8           # never drive the key's per-minute budget below this
REFRESHER_STALE_S = 600       # an older refresher state means no refresher is running
PREWARM_WORKERS = 2           # the local server's parallel slots (OLLAMA_NUM_PARALLEL)
FRESH_MARGIN = timedelta(hours=1)
WINDOW_OPEN = dtime(1, 0)     # Saturday (user, 2026-09-25): well after Friday's 19:50 tick
WINDOW_CLOSE = dtime(19, 55)  # Sunday: before the refresher wakes (20:10) for 20:30
_ET = ZoneInfo("America/New_York")


def raw_dir() -> Path:
    return ROOT / "raw"


def plan_path() -> Path:
    return ROOT / "plan.json"


def progress_path() -> Path:
    return ROOT / "progress.json"


class Stop(Exception):
    """The run's deadline passed."""


# ── time ─────────────────────────────────────────────────────────────────────

def cutoff(d: date) -> datetime:
    """08:30 ET on session ``d``, aware UTC."""
    return datetime.combine(d, CUTOFF_ET, tzinfo=_ET).astimezone(timezone.utc)


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_window(wk: date) -> Tuple[date, date]:
    """Every day a session of week ``wk`` can reach: [monday-3, sunday]."""
    return wk - timedelta(days=3), wk + timedelta(days=6)


def in_window(now_et: datetime, window: str = "weekend") -> bool:
    """``any``: always. ``weekend``: Saturday from 01:00 ET to Sunday 19:55 ET."""
    if window == "any":
        return True
    wd, t = now_et.weekday(), now_et.time()
    if wd == 5:
        return t >= WINDOW_OPEN
    if wd == 6:
        return t < WINDOW_CLOSE
    return False


# ── the plan ─────────────────────────────────────────────────────────────────

def build_plan(start: date = START, end: Optional[date] = None, arrays_dir: Path = ARRAYS_DIR,
               band_size: int = BAND_SIZE) -> dict:
    """``{days, universe: {day: [tickers]}, rank: {ticker: int}, band_size}``
    from the model arrays: a name is in day D's universe when any of its rows
    that day is tradeable (price >= $5, 20-session dollar volume >= $5M, both
    point-in-time); names are ranked by their median dollar volume over the
    window (0 = most liquid)."""
    arrays_dir = Path(arrays_dir)
    meta = json.loads((arrays_dir / "meta.json").read_text(encoding="utf-8"))
    names = list(meta["tickers"])
    dn = np.load(arrays_dir / "dn.npy", mmap_mode="r")
    s = int(np.datetime64(start, "D").astype(np.int64))
    e = int(np.datetime64(end, "D").astype(np.int64)) if end else int(np.max(dn))
    m = (np.asarray(dn) >= s) & (np.asarray(dn) <= e)
    idx = np.flatnonzero(m)
    dd = np.asarray(dn)[idx]
    tt = np.load(arrays_dir / "tk.npy", mmap_mode="r")[idx]
    px = np.load(arrays_dir / "px.npy", mmap_mode="r")[idx]
    dv = np.nan_to_num(np.load(arrays_dir / "dv20.npy", mmap_mode="r")[idx])
    ok = (px >= MIN_PRICE) & (dv >= MIN_DV)
    dd, tt, dv = dd[ok], tt[ok], dv[ok]
    universe: Dict[str, List[str]] = {}
    for day in np.unique(dd):
        codes = np.unique(tt[dd == day])
        universe[str(np.datetime64(int(day), "D"))] = sorted(names[int(c)] for c in codes)
    med: Dict[str, float] = {}
    order = np.argsort(tt, kind="stable")
    tt_s, dv_s = tt[order], dv[order]
    bounds = np.flatnonzero(np.diff(tt_s)) + 1
    for code_arr, dv_arr in zip(np.split(tt_s, bounds), np.split(dv_s, bounds)):
        if len(code_arr):
            med[names[int(code_arr[0])]] = float(np.median(dv_arr))
    ranked = sorted(med, key=lambda k: (-med[k], k))
    filled = fill_thin_days(universe)
    plan = {"built_at": datetime.now(timezone.utc).isoformat(), "start": str(start),
            "end": str(np.datetime64(e, "D")), "band_size": int(band_size),
            "days": sorted(universe), "universe": universe, "filled_days": filled,
            "rank": {tk: i for i, tk in enumerate(ranked)}}
    return plan


def fill_thin_days(universe: Dict[str, List[str]], frac: float = 0.8) -> Dict[str, int]:
    """The arrays' tail is THIN, not quiet: the deep 30-minute store is extended
    per name on demand, so the last sessions hold only the names someone
    extended (measured 2026-09-25: 316-626 names a day from 09-15 against ~2,000
    before). A session under ``frac`` of the last full one takes the union with
    it — tradeability moves slowly, and a name scored one session too many costs
    an LLM call where a missing one is a hole in the feature. Returns
    ``{day: names added}``; mutates ``universe``."""
    added: Dict[str, int] = {}
    last_full: Optional[List[str]] = None
    for day in sorted(universe):
        names = universe[day]
        if last_full is not None and len(names) < frac * len(last_full):
            merged = sorted(set(names) | set(last_full))
            added[day] = len(merged) - len(names)
            universe[day] = merged
        else:
            last_full = names
    return added


def save_plan(plan: dict) -> Path:
    p = plan_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(plan), encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_plan() -> dict:
    return json.loads(plan_path().read_text(encoding="utf-8"))


def band_of(plan: dict, ticker: str) -> int:
    return int(plan["rank"].get(ticker, 10 ** 6)) // int(plan["band_size"])


def units(plan: dict) -> List[Tuple[date, str, List[date]]]:
    """``(week, ticker, days)`` in PULL order: oldest week first (the free tier's
    rolling window drops it next), most liquid name first within a week."""
    by: Dict[Tuple[date, str], List[date]] = defaultdict(list)
    for day, tks in plan["universe"].items():
        d = date.fromisoformat(day)
        for tk in tks:
            by[(week_start(d), tk)].append(d)
    rank = plan["rank"]
    keys = sorted(by, key=lambda k: (k[0], rank.get(k[1], 10 ** 6), k[1]))
    return [(wk, tk, sorted(by[(wk, tk)])) for wk, tk in keys]


# ── raw answers ──────────────────────────────────────────────────────────────

def week_path(tk: str, wk: date) -> Path:
    return raw_dir() / tk.upper() / f"w{wk.isoformat()}.json.gz"


def day_path(tk: str, d: date) -> Path:
    return raw_dir() / tk.upper() / f"d{d.isoformat()}.json.gz"


def read_raw(p: Path) -> Optional[dict]:
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:                                   # noqa: BLE001
        logger.debug(f"[finnhub-backfill] unreadable {p}: {exc}")
        return None


def write_raw(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, p)


def trim(items: Sequence[dict]) -> List[dict]:
    """The fields live's rule reads, summaries cut where live cuts them."""
    out = []
    for it in items or []:
        out.append({"datetime": it.get("datetime"), "headline": it.get("headline"),
                    "url": it.get("url"), "source": it.get("source"),
                    "summary": (it.get("summary") or "")[:1000]})
    return out


def fresh(meta: Optional[dict], when: datetime) -> bool:
    """An answer serves an instant only when it was fetched after it."""
    if not meta:
        return False
    try:
        f = datetime.fromisoformat(str(meta.get("fetched_at")))
    except (TypeError, ValueError):
        return False
    return f >= when + FRESH_MARGIN


def unit_tasks(tk: str, wk: date, days: Sequence[date]) -> List[Tuple[str, str, date]]:
    """The requests a (ticker, week) still needs: the week, then — when it came
    back at the cap — live's exact request for each session of it."""
    last = max(cutoff(d) for d in days)
    meta = read_raw(week_path(tk, wk))
    if not fresh(meta, last):
        return [("week", tk, wk)]
    if int(meta.get("n", 0)) < CAP:
        return []
    return [("day", tk, d) for d in days if not fresh(read_raw(day_path(tk, d)), cutoff(d))]


def items_for(tk: str, d: date) -> Optional[list]:
    """The raw answer that holds session ``d``'s request, or None (not acquired)."""
    meta = read_raw(week_path(tk, week_start(d)))
    if not fresh(meta, cutoff(d)):
        return None
    if int(meta.get("n", 0)) < CAP:
        return meta.get("items") or []
    dm = read_raw(day_path(tk, d))
    return (dm.get("items") or []) if fresh(dm, cutoff(d)) else None


# ── the pull ─────────────────────────────────────────────────────────────────

def refresher_active() -> bool:
    """True while the live Finnhub refresher reports it is using the key."""
    from src.data.finnhub_refresher import read_state
    st = read_state()
    if not st:
        return False
    if time.time() - float(st.get("at", 0)) > REFRESHER_STALE_S:
        return False                     # the scheduler is not running it
    return bool(st.get("active"))


class LivePhase:
    """The live tick's phase (``fetch`` | ``prep`` | ``sentiment`` | ``post``, None
    between ticks) from the scheduler's NEWEST log, read INCREMENTALLY, with
    `news_replay`'s markers and rules.

    `news_replay.live_tick_phase` scans a fixed 3 MB tail, and an RTH tick logs
    more than that at DEBUG — measured 2026-09-25: ten minutes into the 12:00
    tick its start line had already left the tail, so mid-sentiment it read "no
    tick" and a gate built on it never paused. This finds the last tick start
    once (reading backwards), then scans only the bytes appended since. The log
    is named by its process's start date and rotates 24 h later, so the file
    being written is the newest by mtime, not "today's"."""

    CHUNK = 4 << 20
    MAX_BACK = 256 << 20

    def __init__(self, log_dir: Path = Path("logs"), ttl: float = 5.0):
        import re

        from src.analysis import news_replay as nr
        self.log_dir, self.ttl = Path(log_dir), ttl
        self._start = re.compile(nr._TICK_START_RE.pattern.encode())
        self._end = re.compile(nr._TICK_END_RE.pattern.encode())
        self._marks = [(name, re.compile(rx.pattern.encode())) for name, rx in nr._PHASE_MARKS]
        self._stale = nr._STALE_TICK
        self._checked, self._value = -1e18, None
        self._reset(None)

    def _reset(self, path: Optional[Path]) -> None:
        self.path, self.pos, self.partial = path, 0, b""
        self.found, self.ended, self.start_at, self.seen = False, False, None, set()

    def _seek_last_start(self, fh, size: int) -> int:
        """Byte offset of the line holding the file's last tick start, or -1."""
        end = size
        while end > 0 and size - end < self.MAX_BACK:
            begin = max(0, end - self.CHUNK)
            fh.seek(begin)
            block = fh.read(min(size, end + 512) - begin)      # overlap: a marker on the seam
            hits = list(self._start.finditer(block))
            if hits:
                return begin + block.rfind(b"\n", 0, hits[-1].start()) + 1
            end = begin
        return -1

    def _scan(self, data: bytes) -> None:
        for line in data.split(b"\n"):
            if self._start.search(line):
                self.found, self.ended, self.seen = True, False, set()
                try:
                    self.start_at = datetime.strptime(line[:19].decode("ascii"), "%Y-%m-%d %H:%M:%S")
                except (UnicodeDecodeError, ValueError):
                    self.start_at = None
            elif self.found and not self.ended:
                if self._end.search(line):
                    self.ended = True
                    continue
                for name, rx in self._marks:
                    if name not in self.seen and rx.search(line):
                        self.seen.add(name)

    def _read(self, now: datetime) -> Optional[str]:
        files = sorted(self.log_dir.glob("llm_trader_*.log"), key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        path = files[-1]
        size = path.stat().st_size
        if path != self.path or size < self.pos:
            self._reset(path)
        with open(path, "rb") as fh:
            if self.pos == 0 and not self.found:
                off = self._seek_last_start(fh, size)
                self.pos = off if off >= 0 else size
            fh.seek(self.pos)
            data = self.partial + fh.read(size - self.pos)
        self.pos = size
        cut = data.rfind(b"\n")
        complete, self.partial = (data[:cut + 1], data[cut + 1:]) if cut >= 0 else (b"", data)
        self._scan(complete)
        if not self.found or self.ended:
            return None
        if self.start_at is not None and now - self.start_at > self._stale:
            return None                                  # never persisted: the watchdog killed it
        phase = "fetch"
        for name, _rx in self._marks:
            if name in self.seen:
                phase = name
        return phase

    def __call__(self, now: Optional[datetime] = None) -> Optional[str]:
        t = time.monotonic()
        if now is None and t - self._checked < self.ttl:
            return self._value
        self._checked = t
        try:
            self._value = self._read(now or datetime.now())
        except Exception as exc:                               # noqa: BLE001 — pacing, never a guard
            logger.debug(f"[finnhub-backfill] live phase unreadable: {exc}")
            self._value = None
        return self._value


_LIVE_PHASE: Optional[LivePhase] = None


def live_phase() -> Optional[str]:
    """The live tick's current phase (see `LivePhase`); one reader per process."""
    global _LIVE_PHASE
    if _LIVE_PHASE is None:
        _LIVE_PHASE = LivePhase()
    return _LIVE_PHASE()


class Gate:
    """When may the next request / LLM call go: inside the window, before the
    deadline, and never beside the live tick's own use of the same resource."""

    def __init__(self, what: str, window: str = "weekend", until: Optional[datetime] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], datetime] = lambda: datetime.now(_ET)):
        self.what, self.window, self.until = what, window, until
        self.sleep, self.now = sleep, now
        self._told: set = set()
        self._paused_at: Optional[float] = None

    def _say(self, key: str, msg: str) -> None:
        if key not in self._told:
            logger.info(f"[finnhub-backfill] {self.what}: {msg}")
            self._told.add(key)

    def wait(self) -> None:
        while True:
            now = self.now()
            if self.until and now >= self.until:
                raise Stop("until")
            if not in_window(now, self.window):
                self._say("window", "outside the weekend window — sleeping until it opens")
                self.sleep(600)
                continue
            self._told.discard("window")
            if self.what == "acquire" and refresher_active():
                self._say("refresher", "the live refresher is using the key — waiting")
                self.sleep(60)
                continue
            self._told.discard("refresher")
            phase = live_phase()
            if (self.what == "acquire" and phase == "fetch") or \
               (self.what == "score" and phase == "sentiment"):
                if self._paused_at is None:                # one line per live tick
                    self._paused_at = time.monotonic()
                    logger.info(f"[finnhub-backfill] {self.what}: live tick in {phase} — paused")
                self.sleep(15)
                continue
            if self._paused_at is not None:
                logger.info(f"[finnhub-backfill] {self.what}: resumed after "
                            f"{time.monotonic() - self._paused_at:.0f}s")
                self._paused_at = None
            return


def http_fetch(tk: str, frm: date, to: date) -> Tuple[int, list, Optional[int], Optional[float]]:
    """One company-news request: ``(status, items, remaining, reset_epoch)``."""
    import httpx

    from src.data.provider_news import _FINNHUB_NEWS
    r = httpx.get(_FINNHUB_NEWS, params={"symbol": tk.upper(), "from": frm.isoformat(),
                                         "to": to.isoformat(), "token": settings.finnhub_api_key},
                  timeout=20)

    def _num(h, cast):
        try:
            return cast(r.headers.get(h))
        except (TypeError, ValueError):
            return None
    rem, reset = _num("x-ratelimit-remaining", int), _num("x-ratelimit-reset", float)
    if r.status_code != 200:
        return r.status_code, [], rem, reset
    items = r.json() or []
    return 200, (items if isinstance(items, list) else []), rem, reset


def _write_progress(**kw) -> None:
    p = progress_path()
    try:
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:                                          # noqa: BLE001
        cur = {}
    cur.update(kw, updated_at=datetime.now(timezone.utc).isoformat())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cur), encoding="utf-8")
    os.replace(tmp, p)


def read_progress() -> dict:
    try:
        return json.loads(progress_path().read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}


def acquire(plan: dict, window: str = "weekend", until: Optional[datetime] = None,
            fetch: Callable = http_fetch, sleep: Callable[[float], None] = time.sleep,
            gate: Optional[Gate] = None, max_requests: Optional[int] = None) -> dict:
    """Fetch every missing answer, oldest week first. Each week's units are all
    attempted before the next week starts, and ``progress.json`` records the
    last week done — the scorer's frontier."""
    if not settings.finnhub_api_key:
        return {"skipped": "no FINNHUB_API_KEY"}
    gate = gate or Gate("acquire", window, until, sleep=sleep)
    t0 = time.monotonic()
    n_req = n_ok = 0
    errors: Counter = Counter()
    last_at = 0.0
    todo = units(plan)
    by_week: Dict[date, list] = defaultdict(list)
    for wk, tk, days in todo:
        by_week[wk].append((tk, days))
    logger.info(f"[finnhub-backfill] pull: {len(todo):,} (ticker, week) units over "
                f"{len(by_week)} weeks, {len(plan['rank']):,} names")
    try:
        for wk in sorted(by_week):
            for tk, days in by_week[wk]:
                pending = unit_tasks(tk, wk, days)
                while pending:
                    kind, tk_, x = pending.pop(0)
                    frm, to = week_window(x) if kind == "week" else (x - timedelta(days=3), x)
                    status, items, rem, reset = -1, [], None, None
                    for _attempt in range(4):
                        gate.wait()
                        gap = PACE_S - (time.monotonic() - last_at)
                        if gap > 0:
                            sleep(gap)
                        try:
                            status, items, rem, reset = fetch(tk_, frm, to)
                        except Exception as exc:               # noqa: BLE001
                            logger.debug(f"[finnhub-backfill] {tk_} {frm}: {exc}")
                            status = -1
                        last_at = time.monotonic()
                        n_req += 1
                        if rem is not None and rem < REMAINING_FLOOR:
                            sleep(max(1.0, min(61.0, (reset or 0) - time.time() + 1.0)))
                        if status == 429:
                            sleep(65)
                            continue
                        if status == -1:
                            sleep(5)
                            continue
                        break
                    if status != 200:
                        errors[status] += 1
                    else:
                        n_ok += 1
                        path = week_path(tk_, x) if kind == "week" else day_path(tk_, x)
                        write_raw(path, {"from": frm.isoformat(), "to": to.isoformat(),
                                         "fetched_at": datetime.now(timezone.utc).isoformat(),
                                         "n": len(items), "items": trim(items)})
                        if kind == "week" and len(items) >= CAP:
                            pending += [t for t in unit_tasks(tk, wk, days) if t not in pending]
                    if max_requests and n_req >= max_requests:
                        raise Stop("max_requests")
                    if status != 200:
                        continue
                    if n_req % 500 == 0:
                        logger.info(f"[finnhub-backfill] pull: {n_req:,} requests ({n_ok:,} ok, "
                                    f"errors {dict(errors)}), week {wk} in "
                                    f"{(time.monotonic() - t0) / 3600:.1f} h")
            _write_progress(week_done=str(wk), requests=n_req)
            logger.info(f"[finnhub-backfill] pull: week {wk} done ({n_req:,} requests so far)")
    except Stop as stop:
        return {"stopped": str(stop), "requests": n_req, "ok": n_ok, "errors": dict(errors)}
    _write_progress(pull_complete=True)
    return {"requests": n_req, "ok": n_ok, "errors": dict(errors),
            "hours": round((time.monotonic() - t0) / 3600, 2)}


# ── the EVENTS source (2026-09-25, user request) ─────────────────────────────
#
# `pre:events`: the same 08:30 ET per-name feature built from live's EVENT feeds,
# each rebuilt from the deep store's point-in-time rows through LIVE'S OWN
# article builders (the daily event caches exist only from June 2026 and cover
# only the ~130 pre-fetch names). What each leg reads, as of session D:
#
# * 8-K — filings accepted by 08:30 ET D, filed in the last `eight_k_lookback_days`
#   (`news_history._eight_k_leg` over the deep store's `sec_filings`): exact;
# * analyst — `yf_analyst` rows graded in the 30 days before D's first tick,
#   through D-1 (live built the day's list at its first tick), fed to
#   `analyst_ratings._build_article`;
# * EPS — the latest `yf_earnings` release in the 90 days before D and out by
#   D's first tick, |surprise| >= 5%, `earnings._build_surprise_article` (the
#   consensus is Yahoo's as stored, which it may have restated since);
# * short interest — APPROXIMATE: live reads yfinance's snapshot (short % of
#   FLOAT, days to cover, this and last month's shares short), which has no
#   history; here the latest Polygon settlement published by D (settlement + 10
#   business days), the report two settlements earlier, days to cover as
#   reported, and short interest / shares OUTSTANDING (float is not stored, so the
#   15% bar is met less often), plus FINRA's short-volume ratio for D-1 — the
#   same thresholds and builders;
# * dark pool — `quiver_dpi` rows dated <= D-2 (the deep features' lag) through
#   `quiver.fetch_offexchange`.
#
# Quiver gov-contracts and lobbying are NOT rebuilt: every article either builder
# makes carries ONE shared URL, and live's pool keeps the first copy of a URL, so
# ~1 contract article per tick reaches a live digest (the known, unfixed defect)
# — a rebuild that gave every name its own contract articles would be a feature
# live never produces. Ticker events (renames / delistings) are left out as rare.
# Analyst / EPS / short-interest articles are stamped at D's first tick (01:00
# ET), as live's builders stamp `now` when the day's cache is built.

EVENTS_SPEC = "pre:events"
FIRST_TICK_ET = dtime(1, 0)
SOURCES = {"finnhub": SPEC, "events": EVENTS_SPEC}


def first_tick(d: date) -> datetime:
    """Session ``d``'s first tick (the overnight 01:00 ET slot), aware UTC."""
    return datetime.combine(d, FIRST_TICK_ET, tzinfo=_ET).astimezone(timezone.utc)


def _business_days_after(d: date, n: int) -> date:
    out, k = d, 0
    while k < n:
        out += timedelta(days=1)
        if out.weekday() < 5:
            k += 1
    return out


class EventTables:
    """The deep-store rows the event legs read, loaded ONCE for a plan's names
    and window and split per ticker (every lookup is then a dict hit)."""

    def __init__(self, tickers: Sequence[str], start: date, end: date, deep_dir: Optional[Path] = None):
        import duckdb
        from src.data import deep
        base = Path(deep_dir) if deep_dir else deep.DEEP_DIR
        names = sorted({t.upper() for t in tickers})
        con = duckdb.connect()
        con.register("names_", __import__("pandas").DataFrame({"ticker": names}))

        def q(fam: str, where: str):
            p = base / f"{fam}.parquet"
            if not p.exists():
                return None
            return con.execute(f"SELECT * FROM read_parquet('{p.as_posix()}') t "
                               f"WHERE upper(t.ticker) IN (SELECT ticker FROM names_) AND {where}").fetchdf()
        s = lambda k: (start - timedelta(days=k)).isoformat()     # noqa: E731
        try:
            frames = {
                "analyst": q("yf_analyst", f"grade_date >= TIMESTAMP '{s(40)}' "
                                           f"AND grade_date <= TIMESTAMP '{end.isoformat()} 23:59:59'"),
                "earnings": q("yf_earnings", f"event_ts >= TIMESTAMP '{s(120)}' "
                                             f"AND event_ts <= TIMESTAMP '{end.isoformat()} 23:59:59' "
                                             "AND eps_reported IS NOT NULL AND eps_estimate IS NOT NULL"),
                "short_interest": q("short_interest", f"settlement_date >= '{s(150)}'"),
                "short_volume": q("short_volume", f"date >= '{s(10)}'"),
                "shares": q("yf_shares", f"date >= '{s(800)}'"),
                "dpi": q("quiver_dpi", f"Date >= '{s(90)}'"),
                "sec": q("sec_filings", f"form IN ('8-K', '8-K/A') AND filing_date >= '{s(10)}'"),
            }
        finally:
            con.close()
        self.by: Dict[str, Dict[str, object]] = {}
        for fam, df in frames.items():
            groups: Dict[str, object] = {}
            if df is not None and not df.empty:
                for tk, g in df.groupby(df["ticker"].str.upper(), sort=False):
                    groups[tk] = g
            self.by[fam] = groups

    def rows(self, fam: str, tk: str):
        return self.by.get(fam, {}).get(tk.upper())


def _stamp(a, when: datetime):
    """Live's builders stamp `now`; the rebuild stamps D's first tick."""
    try:
        return a.model_copy(update={"published_at": when})
    except AttributeError:                                     # pydantic v1
        return a.copy(update={"published_at": when})


def _analyst_article(t: EventTables, tk: str, d: date):
    import pandas as pd

    from src.data import analyst_ratings
    g = t.rows("analyst", tk)
    if g is None or g.empty:
        return None
    ft = first_tick(d).replace(tzinfo=None)
    lo = ft - timedelta(days=int(settings.analyst_ratings_lookback_days))
    day0 = datetime.combine(d, dtime(0, 0), tzinfo=_ET).astimezone(timezone.utc).replace(tzinfo=None)
    gd = pd.to_datetime(g["grade_date"])
    rec = g[(gd >= lo) & (gd < day0)]
    if rec.empty:
        return None
    rows = pd.DataFrame({"Firm": rec["firm"].values, "ToGrade": rec["to_grade"].fillna("").values,
                         "FromGrade": rec["from_grade"].fillna("").values,
                         "Action": rec["action"].values, "priceTargetAction": rec["pt_action"].values,
                         "currentPriceTarget": rec["pt_current"].values,
                         "priorPriceTarget": rec["pt_prior"].values},
                        index=pd.DatetimeIndex(pd.to_datetime(rec["grade_date"]).values, name="GradeDate"))
    rows = rows.sort_index(ascending=False)
    a = analyst_ratings._build_article(tk, rows)
    return _stamp(a, first_tick(d)) if a is not None else None


def _eps_article(t: EventTables, tk: str, d: date):
    import pandas as pd

    from src.data import earnings
    g = t.rows("earnings", tk)
    if g is None or g.empty:
        return None
    ft = first_tick(d).replace(tzinfo=None)
    lo = d - timedelta(days=int(settings.earnings_lookback_days))
    ev = pd.to_datetime(g["event_ts"])
    et_day = pd.to_datetime(g["event_et"]).dt.date
    ok = g[(ev <= ft) & (et_day >= lo)]
    if ok.empty:
        return None
    row = ok.loc[pd.to_datetime(ok["event_ts"]).idxmax()]
    actual, est = float(row["eps_reported"]), float(row["eps_estimate"])
    surprise = row.get("surprise_pct")
    if surprise is None or surprise != surprise:
        if abs(est) < 0.01:
            return None
        surprise = (actual - est) / abs(est) * 100
    surprise = float(surprise)
    if abs(surprise) < earnings._MIN_SURPRISE_PCT:
        return None
    rep = pd.to_datetime(row["event_et"]).date()
    return _stamp(earnings._build_surprise_article(tk, rep, actual, est, surprise), first_tick(d))


def _short_article(t: EventTables, tk: str, d: date):
    from src.data import short_interest as si
    g = t.rows("short_interest", tk)
    if g is None or g.empty:
        return None
    known = []
    for r in g.sort_values("settlement_date").itertuples(index=False):
        try:
            sd = date.fromisoformat(str(r.settlement_date)[:10])
        except ValueError:
            continue
        if _business_days_after(sd, 10) <= d - timedelta(days=1):
            known.append(r)
    if not known:
        return None
    cur = known[-1]
    prior = known[-3] if len(known) >= 3 else None
    sh = t.rows("shares", tk)
    shares = None
    if sh is not None and not sh.empty:
        shk = sh[sh["date"].astype(str) <= d.isoformat()].sort_values("date")
        if not shk.empty:
            shares = float(shk["shares"].iloc[-1])
    if not shares:
        return None
    short_pct = float(cur.short_interest) / shares
    dtc = si._safe_float(cur.days_to_cover)
    mom = None
    if prior is not None and prior.short_interest:
        mom = (float(cur.short_interest) - float(prior.short_interest)) / float(prior.short_interest)
    finra = None
    sv = t.rows("short_volume", tk)
    if sv is not None and not sv.empty:
        svk = sv[sv["date"].astype(str) < d.isoformat()].sort_values("date")
        if not svk.empty and svk["short_volume_ratio"].iloc[-1] == svk["short_volume_ratio"].iloc[-1]:
            finra = float(svk["short_volume_ratio"].iloc[-1]) / 100.0
    a = None
    if short_pct >= si._MIN_SHORT_PCT and dtc is not None and si._SQUEEZE_MIN_DTC <= dtc <= si._SQUEEZE_MAX_DTC:
        a = si._build_squeeze_article(tk, short_pct, dtc, finra)
    elif short_pct >= si._MIN_SHORT_PCT and mom is not None and mom >= si._MOM_THRESHOLD:
        a = si._build_bearish_article(tk, short_pct, mom, finra)
    elif mom is not None and mom <= -si._MOM_THRESHOLD:
        a = si._build_covering_article(tk, short_pct, mom)
    return _stamp(a, first_tick(d)) if a is not None else None


def _dpi_articles(t: EventTables, tickers: Sequence[str], d: date) -> Dict[str, list]:
    """`quiver.fetch_offexchange` on the deep store's rows dated <= D-2, all
    names in one call (the API cap lifted: nothing is fetched)."""
    from unittest.mock import patch

    from src.data import quiver
    limit = (d - timedelta(days=2)).isoformat()

    def rows_for(path: str, **_kw) -> list:
        tk = path.rstrip("/").rsplit("/", 1)[-1].upper()
        g = t.rows("dpi", tk)
        if g is None or g.empty:
            return []
        return g[g["Date"].astype(str) <= limit].to_dict("records")

    class _NoSleep:
        @staticmethod
        def sleep(_s):
            return None

    out: Dict[str, list] = defaultdict(list)
    with patch.object(quiver, "_get", rows_for), patch.object(quiver, "is_available", lambda: True), \
            patch.object(quiver, "time", _NoSleep), \
            patch.object(settings, "quiver_offexchange_max_tickers", 10 ** 6), \
            patch.object(settings, "enable_quiver_offexchange", True):
        for a in quiver.fetch_offexchange(list(tickers)):
            for tk in a.tickers or []:
                out[tk.upper()].append(a)
    return out


def _eight_k_articles(t: EventTables, tk: str, d: date) -> list:
    """`news_history._eight_k_leg` for one name: it reads the module's 8-K
    frame, so that frame is swapped for this name's rows for the call (one
    implementation, one thread — pools are built before the parallel pre-warm)."""
    from src.analysis import news_history as nh
    g = t.rows("sec", tk)
    if g is None or g.empty:
        return []
    saved = nh._SEC_8K
    nh._SEC_8K = g
    try:
        return nh._eight_k_leg([tk], cutoff(d), d)
    finally:
        nh._SEC_8K = saved


def events_pools(t: EventTables, tickers: Sequence[str], d: date) -> Dict[str, list]:
    """``{ticker: [NewsArticle]}`` — each name's own event articles as of 08:30
    ET on ``d``, in live's merge order (8-K, analyst, EPS, short interest, then
    the Quiver dark pool)."""
    dpi = _dpi_articles(t, tickers, d)
    out: Dict[str, list] = {}
    for tk in tickers:
        arts = list(_eight_k_articles(t, tk, d))
        for build in (_analyst_article, _eps_article, _short_article):
            try:
                a = build(t, tk, d)
            except Exception as exc:                           # noqa: BLE001 — one leg, one name
                logger.debug(f"[finnhub-backfill] events {build.__name__} {tk} {d}: {exc}")
                a = None
            if a is not None:
                arts.append(a)
        arts += dpi.get(tk.upper(), [])
        out[tk] = arts
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

_KEY_AS_OF: Optional[datetime] = None     # the instant being scored (set per day-band)
_LIVE_KEY: Optional[Callable] = None


def _prompt_key(ticker: str, engine: str, articles, extra: str = "") -> str:
    """The verdict cache keyed on what the model READS — the rendered digest
    (source, AGE label, title, summary, in digest order) plus everything
    `_sentiment_cache_key` salts with — instead of each article's publish
    instant. Live re-stamps its analyst / EPS / short-interest articles every
    day, so an unchanged EPS beat is a new key every session while the prompt
    the model gets is byte-identical: at 08:30 the article is always "8h old".
    Two identical prompts are one question, so the backfill asks it once (per
    cache TTL). Installed in the backfill PROCESS only (`_bound_verdict_cache`)."""
    import hashlib

    from src.analysis import sentiment
    if _KEY_AS_OF is None:
        return _LIVE_KEY(ticker, engine, articles, extra)
    text = sentiment._digest_text(ticker, list(articles), _KEY_AS_OF)
    payload = (f"{ticker.upper()}|{engine}|{sentiment.sentiment_model_for(engine)}|"
               f"{sentiment._prompt_pair()[1]}|{extra}|{text}")
    return "read:" + hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def _bound_verdict_cache() -> None:
    """This process's verdict cache: never the LIVE file, and in memory only —
    the file is rewritten whole on every flush, and a weekend of scoring would
    turn that into a 100+ MB JSON rewrite every 20 s. The cache only has to
    carry a verdict from the parallel pre-warm to the sequential feature pass
    (seconds) and a repeat prompt to the next day (minutes) — keyed on the
    prompt text (`_prompt_key`)."""
    global _LIVE_KEY
    from src.analysis import news_replay as nr
    from src.analysis import sentiment
    nr._isolate_sentiment_cache()
    sentiment._SENT_CACHE_FLUSH_EVERY_S = 1e12
    if _LIVE_KEY is None:
        _LIVE_KEY = sentiment._sentiment_cache_key
        sentiment._sentiment_cache_key = _prompt_key


def _cache_size() -> int:
    from src.analysis import sentiment
    with sentiment._SENT_CACHE_LOCK:
        return len(sentiment._SENT_CACHE or {})


def _prune_verdict_cache() -> int:
    from src.analysis import sentiment
    with sentiment._SENT_CACHE_LOCK:
        cache = sentiment._SENT_CACHE
        if not cache:
            return 0
        horizon = time.time() - sentiment._sent_cache_ttl_seconds()
        stale = [k for k, v in cache.items() if float(v.get("ts", 0)) < horizon]
        for k in stale:
            cache.pop(k, None)
        return len(stale)


def run_id_for(d: date, band: int) -> str:
    return f"pre-{d.isoformat()}-b{band}"


def scored_runs(spec: str = SPEC) -> set:
    from src.db import repo
    df = repo.fetch_df("SELECT DISTINCT run_id FROM news_replay WHERE pool_spec = ?", [spec])
    return set() if df is None or df.empty else set(df["run_id"].astype(str))


def prev_closes(tickers: Sequence[str], d: date) -> Dict[str, float]:
    """The last completed daily close before session ``d`` — the price a
    pre-market 08:30 read has."""
    import bisect

    from src.analysis.predictability import _hlc_by_session
    out: Dict[str, float] = {}
    for tk in tickers:
        try:
            dh = _hlc_by_session(tk)
            if dh is None:
                continue
            days = list(dh[0])
            i = bisect.bisect_left(days, d) - 1
            if i >= 0:
                out[tk] = float(dh[3].iloc[i])
        except Exception:                                      # noqa: BLE001
            continue
    return out


def _insert_rows(rows: List[dict], sleep: Callable[[float], None] = time.sleep) -> None:
    """Write a (day, band) — retried, since the live scheduler and its EOD
    thread hold the database's single write lock for short spells."""
    from src.db import repo
    for attempt in range(20):
        try:
            repo.insert_news_replay(rows)
            return
        except Exception as exc:                               # noqa: BLE001
            if attempt == 19:
                raise
            logger.debug(f"[finnhub-backfill] write retry {attempt + 1}: {exc}")
            sleep(min(60.0, 3.0 * (attempt + 1)))


def score_day_band(plan: dict, d: date, band: int, engine: str = "local",
                   gate: Optional[Gate] = None, source: str = "finnhub",
                   tables: Optional[EventTables] = None) -> dict:
    """Score one (session, band) of ``source`` whole, or report why not. The
    feature pass runs in ONE thread: the point-in-time cutoff (`analysis_asof`)
    is thread-local."""
    global _KEY_AS_OF
    from concurrent.futures import ThreadPoolExecutor

    from src.analysis import news_history as nh
    from src.analysis import news_replay as nr
    from src.analysis.asof import analysis_asof
    from src.analysis.news_clustering import cluster_mode, set_corpus
    from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
    rid = run_id_for(d, band)
    spec = SOURCES[source]
    tks = [t for t in plan["universe"].get(d.isoformat(), []) if band_of(plan, t) == band]
    if not tks:
        return {"run_id": rid, "status": "empty band"}
    when = cutoff(d)
    _KEY_AS_OF = when
    if source == "finnhub":
        raw = {tk: items_for(tk, d) for tk in tks}
        missing = sorted(tk for tk, v in raw.items() if v is None)
        ready = [tk for tk in tks if raw[tk] is not None]
        pools = {tk: nh._finnhub_rule(raw[tk], tk, when, d) for tk in ready}
    elif source == "events":
        if tables is None:
            raise ValueError("the events source needs its EventTables")
        missing, ready = [], list(tks)
        pools = events_pools(tables, ready, d)
    else:
        raise ValueError(f"unknown source {source!r}")
    union = [a for tk in ready for a in pools[tk]]
    info = {"run_id": rid, "tickers": ready, "when": when, "signal_date": d.isoformat(),
            "prices": prev_closes(ready, d)}
    try:
        if cluster_mode() in ("content", "hybrid"):
            set_corpus(union)
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[finnhub-backfill] clustering corpus failed: {exc}")
    t0 = time.perf_counter()

    def warm(tk: str) -> None:
        arts = filter_relevant_articles(tk, pools[tk]) if pools[tk] else []
        if not arts:
            return
        if gate:
            gate.wait()
        try:
            analyse_sentiment(tk, arts, force_engine=engine, as_of=when, allow_provider=True,
                              store_digest=False)
        except Exception as exc:                               # noqa: BLE001
            logger.debug(f"[finnhub-backfill] prewarm {tk}: {exc}")

    n_cached = _cache_size()
    with ThreadPoolExecutor(max_workers=PREWARM_WORKERS, thread_name_prefix="fb-warm") as ex:
        list(ex.map(warm, ready))
    t_warm = time.perf_counter() - t0
    n_llm = _cache_size() - n_cached        # verdicts the model produced (reuse adds none)
    prov = {"bundle_file": None, "pool_spec": spec}
    rows = []
    with analysis_asof(d.isoformat()):
        baselines = nr.baselines_as_of(d.isoformat(), spec)
        for tk in ready:
            if gate:
                gate.wait()
            try:
                r = nr._replay_one(tk, pools[tk], info, engine, {**prov, "n_pool": len(pools[tk])},
                                   baselines)
            except Exception as exc:                           # noqa: BLE001
                logger.warning(f"[finnhub-backfill] {rid} {tk}: {exc}")
                r = {"ticker": tk, "scorer_failed": True}
            rows.append(r)
        for i, r in enumerate(rows):
            if r.get("scorer_failed"):                 # the model is not deterministic: once more
                try:
                    rows[i] = nr._replay_one(r["ticker"], pools[r["ticker"]], info, engine,
                                             {**prov, "n_pool": len(pools[r["ticker"]])}, baselines)
                except Exception as exc:                       # noqa: BLE001
                    logger.warning(f"[finnhub-backfill] {rid} {r['ticker']}: {exc}")
    for r in rows:
        r["replay_version"] = VERSIONS[source]
    failed = [r["ticker"] for r in rows if r.get("scorer_failed")]
    if failed and not nh.content_failures_only(rows):
        logger.error(f"[finnhub-backfill] {rid}: {len(failed)} scorer failure(s) — NOT stored, "
                     f"retried next pass")
        return {"run_id": rid, "status": "scorer failures", "failed": len(failed)}
    _insert_rows(rows)
    views = sum(1 for r in rows if (r.get("news") or 0) != 0)
    logger.info(f"[finnhub-backfill] {spec} {rid}: {len(ready)} names ({len(missing)} not acquired), "
                f"{sum(1 for tk in ready if pools[tk])} with items, {views} views, "
                f"{n_llm} LLM calls, {time.perf_counter() - t0:.0f}s (verdicts {t_warm:.0f}s)")
    return {"run_id": rid, "status": "stored", "names": len(ready), "missing": len(missing),
            "views": views, "llm_calls": n_llm}


def score(plan: dict, window: str = "weekend", until: Optional[datetime] = None,
          engine: str = "local", bands: Optional[Sequence[int]] = None,
          sleep: Callable[[float], None] = time.sleep, poll_s: float = 300.0,
          source: str = "finnhub", tables: Optional[EventTables] = None) -> dict:
    """Band by band (most liquid first), oldest day first within a band. For
    Finnhub a day whose week the pull has not finished yet waits for it, and a
    finished week's still-missing answers (the provider failed them) leave those
    names without a row — NaN, never a 0.0 that would read as "no news". The
    events source reads the deep store and never waits."""
    _bound_verdict_cache()
    gate = Gate("score", window, until, sleep=sleep)
    n_bands = 1 + max(int(r) for r in plan["rank"].values()) // int(plan["band_size"])
    bands = list(bands) if bands is not None else list(range(n_bands))
    if source == "events" and tables is None:
        t0 = time.monotonic()
        tables = EventTables(list(plan["rank"]), date.fromisoformat(plan["start"]),
                             date.fromisoformat(plan["end"]))
        logger.info(f"[finnhub-backfill] event tables loaded in {time.monotonic() - t0:.0f}s")
    done = scored_runs(SOURCES[source])
    stored = fails = 0
    waited_on = None
    try:
        for band in bands:
            for day in plan["days"]:
                d = date.fromisoformat(day)
                rid = run_id_for(d, band)
                if rid in done:
                    continue
                while source == "finnhub":
                    gate.wait()
                    prog = read_progress()
                    frontier = prog.get("week_done")
                    if prog.get("pull_complete") or (frontier and week_start(d) <= date.fromisoformat(frontier)):
                        break
                    if waited_on != (day, frontier):     # once per frontier, not every poll
                        logger.info(f"[finnhub-backfill] score: {day} waits for the pull "
                                    f"(frontier {frontier})")
                        waited_on = (day, frontier)
                    sleep(poll_s)
                _prune_verdict_cache()
                res = score_day_band(plan, d, band, engine=engine, gate=gate, source=source,
                                     tables=tables)
                if res.get("status") == "stored":
                    stored += 1
                    done.add(rid)
                elif res.get("status") == "scorer failures":
                    fails += 1
                    if fails >= 5:
                        logger.error("[finnhub-backfill] repeated scorer failures — is the local "
                                     "LLM server up? stopping")
                        return {"stopped": "scorer failures", "stored": stored}
                    sleep(300)
    except Stop as stop:
        return {"stopped": str(stop), "stored": stored}
    return {"stored": stored}


# ── status ───────────────────────────────────────────────────────────────────

def status(plan: dict, source: str = "finnhub") -> dict:
    from src.db import repo
    df = repo.fetch_df(
        "SELECT run_id, count(*) AS n, sum(CASE WHEN news <> 0 THEN 1 ELSE 0 END) AS views "
        "FROM news_replay WHERE pool_spec = ? GROUP BY 1", [SOURCES[source]])
    per_band: Dict[int, dict] = defaultdict(lambda: {"days": 0, "rows": 0, "views": 0})
    for rid, n, v in ([] if df is None or df.empty else zip(df["run_id"], df["n"], df["views"])):
        b = int(str(rid).rsplit("-b", 1)[1])
        per_band[b]["days"] += 1
        per_band[b]["rows"] += int(n)
        per_band[b]["views"] += int(v or 0)
    n_bands = 1 + max(int(r) for r in plan["rank"].values()) // int(plan["band_size"])
    return {"source": source, "pool_spec": SOURCES[source],
            "plan": {"days": len(plan["days"]), "names": len(plan["rank"]), "bands": n_bands,
                     "start": plan["start"], "end": plan["end"]},
            "pull": read_progress() if source == "finnhub" else None,
            "scored": {f"band {b} (names {b * plan['band_size']}-{(b + 1) * plan['band_size'] - 1})":
                       per_band[b] for b in sorted(per_band)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Finnhub news history, 08:30 ET, liquid universe")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--acquire", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--window", default="weekend", choices=("weekend", "any"))
    ap.add_argument("--until", default="", help="stop at this ET time (YYYY-MM-DD HH:MM)")
    ap.add_argument("--bands", default="", help="comma-separated band numbers (score)")
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--source", default="finnhub", choices=sorted(SOURCES))
    a = ap.parse_args(argv)
    # One log per process (two processes rotating one file collide on Windows),
    # holding this module's lines and every warning: the scorer's per-ticker
    # INFO lines would be ~100k a day. stderr keeps warnings for a detached run.
    import sys
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    if a.acquire or a.score:
        tag = "acquire" if a.acquire else f"score-{a.source}"
        logger.add(f"logs/finnhub_backfill_{tag}.log", rotation="1 day", retention="30 days",
                   level="INFO", enqueue=True,
                   filter=lambda r: r["level"].no >= 30 or "[finnhub-backfill]" in r["message"])
    until = datetime.strptime(a.until, "%Y-%m-%d %H:%M").replace(tzinfo=_ET) if a.until else None
    if a.plan:
        plan = build_plan()
        p = save_plan(plan)
        n = sum(len(v) for v in plan["universe"].values())
        print(f"plan: {len(plan['days'])} sessions {plan['start']}..{plan['end']}, "
              f"{len(plan['rank']):,} names, {n:,} name-days -> {p}")
    if a.acquire:
        print(json.dumps(acquire(load_plan(), window=a.window, until=until,
                                 max_requests=a.max_requests or None), indent=2, default=str))
    if a.score:
        bands = [int(b) for b in a.bands.split(",") if b.strip()] or None
        print(json.dumps(score(load_plan(), window=a.window, until=until, bands=bands,
                               source=a.source), indent=2, default=str))
    if a.status:
        print(json.dumps(status(load_plan(), source=a.source), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
