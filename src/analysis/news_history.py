"""Per-SOURCE news history: every recoverable feed rebuilt point-in-time, each one
scored ON ITS OWN (2026-09-23, user directive: "Build the news history using all
the sources. Then, we'll separate them in different features groups by source.
I.e., all features built using the Polygon source. This way, we'll be able to know
if some sources are better quality and if they are matching better the live").

WHY BY SOURCE
-------------
The live `news` verdict is one LLM read of a digest pooled from ~10 feeds. Over the
archive era (840 live digests, 2026-09-12..09-23) the digest ARTICLES came from
Google News 37.1%, the yfinance/NewsAPI hourly bundle 29.9%, Finnhub 18.5%, Polygon
6.4%, the structured event feeds 6.8% (EPS 2.9, analyst 1.4, short interest 1.0,
8-K 0.9, Quiver 0.6) and RSS / press wires 1.2%. Those feeds come back with very
different fidelity — the bundle, the event caches and Polygon's as-of call ARE the
tick's own input, Finnhub and Google are re-asked from the provider's history — so
one pooled rebuild hides both which part of it is live-faithful and where its skill
comes from. Here each feed is a FEATURE GROUP: the whole news family (verdict,
catalyst, velocity, shock, fresh / quiet / unpriced reads, mass, article count)
computed from that feed's articles alone, plus `all`, every feed in live's merge
order. Rows land in `news_replay` with ``pool_spec = "src:<group>"``.

GROUPS
------
============  ==========================================  ==========================
group         history                                     fidelity to the live leg
============  ==========================================  ==========================
``bundle``    the hourly `cache/news_*.json` file for     EXACT (tags re-confirmed on
              the tick's hour (yfinance + NewsAPI)        pre-2026-09-04 files)
``polygon``   `/v2/reference/news` limit 1000 with        EXACT but for articles
              `published_utc.lte=<tick>` (live's call)    Polygon ingested late
``events``    8-K, analyst, ticker events, EPS, short     EXACT from the daily caches
              interest, Quiver contracts / lobbying /     (built articles, or raw
              dark pool                                   Quiver payloads re-built by
                                                          live's own builders); 8-Ks
                                                          from the deep store's SEC
                                                          filings, acceptance <= tick
``finnhub``   company-news from the provider's history    RE-ASKED: live's request is
              (free tier: one rolling year), live's       reproduced (UTC-date window
              rule applied per tick                       [D-3, D], newest 15 non-
                                                          noise at the tick)
``google``    date-bounded search (`after:`/`before:`),   RE-ASKED and THINNING WITH
              live's three queries, 24h window at the     AGE: Google returns fewer
              tick                                        items for older windows
``all``       every group above, live's merge order and   RSS / press wires are not
              URL dedupe                                  recoverable before the
                                                          archive and are in no group
============  ==========================================  ==========================

COVERAGE — live asked its per-ticker feeds about a THIRD of the names it scored.
Step 1 fetches on the universe as it stands BEFORE the smart-money, macro-discovery
and cointegration-peer additions (`POST_FETCH_SOURCES`), which more than double
it: a median 128 pre-fetch names of 388 scored. Google, Finnhub, the 8-K scan,
Polygon's universe filter and the Quiver builders all ran on that pre-fetch set,
so here they do too (`_feed_universe`, from `signals.universe_source`). Within it
live asks Finnhub for the FIRST 60 and Google for the first 150 eligible of its
Step-0 order — watchlist, sector ETFs, trending picks, commodities, factor ETFs,
then discovered names — rebuilt by `_live_order` (pinned lists in settings order,
discovered names alphabetical: their order is not stored). Before 2026-07-03 no
`universe_source` exists, so live's coverage is unknowable there and the
`finnhub` / `google` / `all` groups are not built for those 10 runs, like RSS.
Scoring still covers every scored name, exactly as live's relevance filter does.

MEASURED FIDELITY (archive era, 10 runs, LLM-free `--calibrate`): Polygon's as-of
set was 95% held by live's pool (the rest: late ingestion); 8-K articles 100%;
EPS 99.5%; Finnhub recall 60% / precision 69% against the articles live held for
the names it asked (the archive's first-sighting tags and cross-ticker URL dedupe
attribute shared articles differently, so this understates the article overlap);
Google recall 61% / precision 50% by URL — by headline no better (54% / 44% per
ticker, 57% / 38% pool-wide), so this is not unstable redirect links: a
date-bounded search ranks a different slice than live's unbounded 24h query did
(~50% more headlines, ~60% of live's recovered). Google is the least faithful
leg and nothing can make it more so: search answers were never archived.
Before the pre-fetch restriction the same checks read Polygon 61%, 8-K 28%.

HOW THE PROVIDERS ARE ASKED (measured 2026-09-23)
-------------------------------------------------
* Finnhub returns only the NEWEST ~250 items of a window, filtered by UTC date.
  One request per (ticker, ISO week of the tick's ET date) covering the week's
  `[monday-3, sunday]`; a week that comes back at the cap is re-asked per day with
  live's exact request (`from=D-3, to=D`).
* Google returns a relevance-thinned subset, and a NARROW window returns 2-3x more
  per day than a wide one (AAPL "apple" stock, June: 9-day window 16 items on
  06-18, 3-day window 55). One request per (ticker, week, query) over
  `[monday-2, sunday+2]`; a query returning `GOOGLE_SPLIT_MIN` or more items is
  re-asked per run with the 3-day window the 2026-09-23 pilot validated.
* Both pause while a live tick FETCHES its news (the live pipeline shares this
  machine's Google and Finnhub quotas) and pace themselves; Google stops on
  repeated non-200 answers and holds off whenever the LIVE Google leg reports
  trouble. Scoring pauses while a live tick scores its own sentiment (the one
  local LLM server). A live tick runs ~35 min overnight but holds the feeds for
  ~3 and the LLM for ~6 (`news_replay.live_tick_phase`).

NO TIME TRAVEL
--------------
Every leg is cut at the tick: bundle files stamped at/before it, Polygon's
`lte`, Finnhub and Google items published at/before it, 8-Ks accepted at/before
it, and the event caches are the ones the tick itself read (their ET date).
Scoring runs inside `analysis_asof(signal_date)` exactly like `news_replay`.

CLI
    python -m src.analysis.news_history --status
    python -m src.analysis.news_history --acquire finnhub [--budget-hours H]
    python -m src.analysis.news_history --acquire google  [--budget-hours H]
    python -m src.analysis.news_history --score [--groups events,polygon] [--loop]
    python -m src.analysis.news_history --calibrate      # article-level fidelity, LLM-free
    python -m src.analysis.news_history --report         # per-source quality + fidelity
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from copy import copy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch
from zoneinfo import ZoneInfo

from loguru import logger

from config.settings import settings
from src.analysis import news_replay as nr

HISTORY_VERSION = "news-sources-v1"
SPEC_PREFIX = "src:"
GROUPS = ("events", "polygon", "bundle", "finnhub", "google", "all")
_ET = ZoneInfo("America/New_York")

# Live's merge order (pipeline Step 1: the news leg, then 8k, analyst, ticker
# events, EPS, short, polygon, finnhub, google, the Quiver legs) — first URL wins.
EVENT_LEGS = ("8k", "analyst", "ticker_events", "eps", "short",
              "quiver_contracts", "quiver_lobbying", "quiver_darkpool")
ALL_LEGS = ("bundle", "8k", "analyst", "ticker_events", "eps", "short", "polygon",
            "finnhub", "google", "quiver_contracts", "quiver_lobbying", "quiver_darkpool")
GROUP_LEGS = {"bundle": ("bundle",), "polygon": ("polygon",), "finnhub": ("finnhub",),
              "google": ("google",), "events": EVENT_LEGS, "all": ALL_LEGS}

# The event feeds that cache their BUILT articles per ET day (the tick read them).
_DAILY_ARTICLE_CACHES = {"analyst": "analyst_ratings", "ticker_events": "ticker_events",
                         "eps": "earnings_surprises", "short": "short_interest"}

GOOGLE_PACE_S = 2.0
GOOGLE_SPLIT_MIN = 30          # a weekly query at/above this is re-asked per run
GOOGLE_MAX_BAD = 2             # our own non-200 answers (after one cool-off) that stop the leg
GOOGLE_COOLOFF_S = 1800.0
NET_TIMEOUT_S = 30.0           # every socket in a fetch process (feedparser has no timeout of its own)
NET_RETRY_S = 60.0             # a transport failure (timeout, DNS, reset) waits this and retries
NET_MAX_FAILS = 15             # consecutive transport failures that stop the leg (network down)
FINNHUB_PACE_S = 1.5           # 40/min against the shared free 60/min
FINNHUB_CAP = 240              # at/above this the window came back truncated (the newest ~250)
_FRESH_MARGIN = timedelta(hours=1)


def pool_spec(group: str) -> str:
    return SPEC_PREFIX + group


def local_day(when: datetime) -> date:
    """The ET calendar date the live tick keyed its daily caches and its
    `date.today()` windows on (the machine runs on Eastern time)."""
    return when.astimezone(_ET).date()


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


# ── the run plan ────────────────────────────────────────────────────────────

def plan_runs(days: Optional[int] = None, only: Optional[Sequence[str]] = None) -> List[dict]:
    """The target runs (the last tick of each signal date, what `build_panel`
    keeps), oldest first, each with its universe, instant and ET date."""
    rids = list(only) if only else nr.target_runs(days=days)
    # the signal date still in progress has no LAST tick yet: the panel's row
    # for it will come from a later run than the one planned now
    today = local_day(datetime.now(timezone.utc)).isoformat()
    out = []
    for rid in rids:
        info = nr._tick_universe(rid)
        if info and (only or str(info["signal_date"])[:10] < today):
            info["day"] = local_day(info["when"])
            info["feed_tickers"], info["feed_known"], info["feed_sources"] = \
                _feed_universe(rid, info["tickers"])
            out.append(info)
    return sorted(out, key=lambda r: r["when"])


# Names the live tick added AFTER its Step-1 news fetch (`pipeline.run_pipeline`:
# smart-money filers, macro discovery, cointegration peer legs). They are scored,
# but no per-ticker feed was ever asked about them — Google, Finnhub, the 8-K scan
# and Polygon's universe filter all ran on the universe as it stood at Step 1.
POST_FETCH_SOURCES = ("smart_money", "macro_discovery", "coint_peer")


def _all_source_at(when: datetime) -> bool:
    """A tick at/after the ALL-SOURCE ingestion (`news_coverage.ALL_SOURCE_SINCE`,
    2026-09-25): from then every per-ticker feed asked about EVERY scored name,
    with no Finnhub or Google cap — so neither the pre-fetch restriction nor the
    caps below apply to it."""
    from src.data.news_coverage import ALL_SOURCE_SINCE
    w = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return w >= datetime.fromisoformat(ALL_SOURCE_SINCE)


def _run_instant(run_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(run_id)[:17], "%Y-%m-%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _feed_universe(run_id: str, tickers: Sequence[str]) -> Tuple[List[str], bool, Dict[str, str]]:
    """The run's PRE-FETCH universe — the names live's per-ticker feeds were
    asked about (a median 128 of 388 scored names). ``(tickers, known,
    sources)``: runs before `universe_source` was recorded (2026-07-03) fall back
    to every scored name, flagged unknown — the snapshot cache cannot recover it
    either (it holds the post-fetch top-up too)."""
    from src.db import repo
    df = repo.fetch_df("SELECT ticker, universe_source FROM signals WHERE run_id = ?", [run_id])
    if df is None or df.empty or df["universe_source"].isna().any():
        return sorted(tickers), False, {}
    sources = {str(t): str(s) for t, s in zip(df["ticker"], df["universe_source"])}
    when = _run_instant(run_id)
    if when is not None and _all_source_at(when):
        return sorted(tickers), True, sources          # every scored name was asked
    keep = {t for t, s in sources.items() if s not in POST_FETCH_SOURCES}
    return sorted(keep), True, sources


def _feed(run: dict) -> List[str]:
    return run.get("feed_tickers") or run["tickers"]


# Live's Step-0 order (`pipeline.run_pipeline`): `get_trending_tickers` returns
# watchlist + sector ETFs + trending picks, then pinned commodities, pinned factor
# ETFs, the screener, earnings / analyst discovery, the insider cluster watch,
# open-position pins and related peers. Live asks Finnhub for the FIRST 60 of it
# and Google for the first 150 eligible, so the order decides who got news.
_STEP0_ORDER = ("watchlist", "sector_etf", "trending", "commodity", "factor_etf", "screener",
                "earnings_discovery", "analyst_discovery", "insider_cluster",
                "open_position_pin", "related_peer")
LIVE_FINNHUB_MAX = 60           # `fetch_finnhub_news(max_tickers=60)` until the all-source ingestion
# `google_news_max_tickers` while the set-1 history was built (the live setting
# 2026-09-04 .. the all-source ingestion, which lifted it to 0 = no cap). Pinned
# here so a rebuild of a pre-switch run keeps the coverage that run had.
LIVE_GOOGLE_MAX = 150


def _live_order(run: dict) -> List[str]:
    """The pre-fetch universe in live's Step-0 order: the pinned lists in their
    settings order, discovered names alphabetical (their discovery order is not
    stored). Measured on the archive era: the first 60 hold ~44-49 of the
    ~60 names live asked Finnhub about."""
    sources = run.get("feed_sources") or {}
    pinned = {"watchlist": list(settings.stocks_list), "sector_etf": list(settings.sectors_list),
              "commodity": list(settings.commodities_list), "factor_etf": list(settings.factor_list)}
    by: Dict[str, List[str]] = defaultdict(list)
    for tk in _feed(run):
        by[sources.get(tk, "")].append(tk)
    order: List[str] = []
    for g in _STEP0_ORDER + tuple(sorted(k for k in by if k not in _STEP0_ORDER)):
        names = by.get(g, [])
        if g in pinned:
            rank = {t: i for i, t in enumerate(pinned[g])}
            names = sorted(names, key=lambda t: (rank.get(t, 10 ** 6), t))
        else:
            names = sorted(names)
        order += names
    return order


def _finnhub_names(run: dict) -> List[str]:
    names = _live_order(run)
    return names if _all_source_at(run["when"]) else names[:LIVE_FINNHUB_MAX]


def _google_names(run: dict) -> List[str]:
    names = [t for t in _live_order(run) if _google_eligible(t)]
    return names if _all_source_at(run["when"]) else names[:LIVE_GOOGLE_MAX]


# Groups resting on a per-ticker feed whose live coverage is unknowable before
# 2026-07-03 (no `universe_source`): not built for those runs, like RSS.
_COVERAGE_GROUPS = ("finnhub", "google", "all")


def _google_eligible(tk: str) -> bool:
    # live skips futures and index symbols (`fetch_google_news`)
    return bool(tk) and "=" not in tk and not tk.startswith("^")


def _week_runs(runs: Sequence[dict], pick=None) -> Dict[Tuple[str, date], List[dict]]:
    """``{(ticker, week monday): [runs]}`` — the unit both providers are asked
    in, over the names ``pick(run)`` says live asked (runs whose coverage is
    unknown are left out)."""
    out: Dict[Tuple[str, date], List[dict]] = defaultdict(list)
    for r in runs:
        if pick is not None and not r.get("feed_known", True):
            continue
        wk = week_start(r["day"])
        for tk in (pick(r) if pick is not None else _feed(r)):
            out[(tk, wk)].append(r)
    return out


# ── raw provider caches ──────────────────────────────────────────────────────

def _raw_dir(kind: str, ticker: str) -> Path:
    from src.data.cache import CACHE_DIR
    d = CACHE_DIR / "news_hist" / "raw" / kind / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        return None


def _write_json(p: Path, obj: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    tmp.replace(p)


def _fresh(meta: Optional[dict], when: datetime) -> bool:
    """A cached answer serves a run only when it was fetched after that run —
    a week asked mid-week holds nothing published after the ask."""
    if not meta:
        return False
    try:
        f = datetime.fromisoformat(str(meta.get("fetched_at")))
    except (TypeError, ValueError):
        return False
    return f >= when + _FRESH_MARGIN


def _google_queries(tk: str) -> List[Tuple[str, str]]:
    """Live's three Google queries (`news_fetcher._fetch_google_news_for_ticker`)."""
    from src.data.news_fetcher import _google_name_query
    qs = [("sym", f'"{tk}" stock')]
    name_q = _google_name_query(tk)
    if name_q:
        qs.append(("name", name_q))
    if settings.google_news_business_wire:
        qs.append(("bw", f'"{tk}" site:businesswire.com'))
    return qs


def _gweek_path(tk: str, wk: date, qkind: str) -> Path:
    return _raw_dir("google", tk) / f"w{wk.isoformat()}_{qkind}.json"


def _grun_path(tk: str, when: datetime, qkind: str) -> Path:
    return _raw_dir("google", tk) / f"r{when:%Y%m%dT%H%M%S}_{qkind}.json"


def _gweek_window(wk: date) -> Tuple[date, date]:
    return wk - timedelta(days=2), wk + timedelta(days=8)


def _grun_window(when: datetime) -> Tuple[date, date]:
    return (when - timedelta(days=2)).date(), (when + timedelta(days=1)).date()


def _fweek_path(tk: str, wk: date) -> Path:
    return _raw_dir("finnhub", tk) / f"w{wk.isoformat()}.json"


def _fday_path(tk: str, d: date) -> Path:
    return _raw_dir("finnhub", tk) / f"d{d.isoformat()}.json"


def _fweek_window(wk: date) -> Tuple[date, date]:
    return wk - timedelta(days=3), wk + timedelta(days=6)


class _Pacer:
    """Hold a request RATE: each ``wait()`` returns no sooner than ``interval``
    seconds after the previous one returned, so the request's own latency counts
    toward the spacing instead of adding to it (a plain sleep after each request
    ran Google at 0.34 req/s against the 0.5 intended)."""

    def __init__(self, interval: float):
        self.interval = float(interval)
        self._last: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            gap = self.interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
        self._last = time.monotonic()


# ── acquisition: Google ──────────────────────────────────────────────────────

_LIVE_GOOGLE_BAD = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| \w+\s*\| [^-]*- Google News: "
    r"(?:(\d+) feed\(s\) returned a non-200|0 articles across (\d+) tickers)", re.M)


def live_google_trouble_since(t0: datetime, log_dir: Path = Path("logs")) -> Optional[str]:
    """The live Google leg's own alarm since ``t0`` (local time): a non-200
    tally, or a sweep of 20+ tickers that returned nothing. Either means the
    provider is unhappy with this machine, so the history leg must back off —
    the live leg carries ~37% of live digest articles."""
    d = datetime.now().date()
    for day in (d - timedelta(days=1), d):
        text = nr._log_tail(Path(log_dir) / f"llm_trader_{day:%Y-%m-%d}.log", 2_000_000)
        for m in _LIVE_GOOGLE_BAD.finditer(text):
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts < t0:
                continue
            if m.group(2) or (m.group(3) and int(m.group(3)) >= 20):
                return m.group(0)[-160:]
    return None


def _google_fetch(q: str, after: date, before: date) -> Tuple[Optional[int], list]:
    """One date-bounded Google News search → ``(status, entries)``; entries
    carry exactly the fields live reads off a feed entry."""
    import feedparser
    from urllib.parse import quote_plus

    from src.data.news_fetcher import _GOOGLE_NEWS_RSS, _google_entry_source, _parse_feed_date
    feed = feedparser.parse(_GOOGLE_NEWS_RSS.format(
        q=quote_plus(f"{q} after:{after.isoformat()} before:{before.isoformat()}")))
    status = getattr(feed, "status", None)
    entries = []
    for e in feed.entries:
        pub = _parse_feed_date(e)
        entries.append({"title": (e.get("title") or "").strip(), "summary": e.get("summary", "") or "",
                        "link": e.get("link", ""), "source": _google_entry_source(e),
                        "published": pub.isoformat() if pub else None})
    if status is None and getattr(feed, "bozo", False) and not entries:
        status = -1                                              # transport failure
    return (int(status) if status is not None else 200), entries


def _google_unit_tasks(tk: str, wk: date, rs: Sequence[dict]) -> List[tuple]:
    """The Google requests one (ticker, week) still needs. A per-run split is
    only known to be needed once the weekly answer is in."""
    tasks = []
    last = max(r["when"] for r in rs)
    for qkind, q in _google_queries(tk):
        meta = _read_json(_gweek_path(tk, wk, qkind))
        if not _fresh(meta, last):
            tasks.append(("week", tk, wk, qkind, q, None))
            continue
        if int(meta.get("n", 0)) >= GOOGLE_SPLIT_MIN:
            for r in rs:
                if not _fresh(_read_json(_grun_path(tk, r["when"], qkind)), r["when"]):
                    tasks.append(("run", tk, wk, qkind, q, r["when"]))
    return tasks


def _unit_order(runs: Sequence[dict], first_since: Optional[date] = None, pick=None):
    """(ticker, week) units, weeks from ``first_since`` first — the archive era,
    where a rebuilt leg can be checked against what live held — then oldest
    first, so runs complete in order."""
    def key(kv):
        (tk, wk) = kv[0]
        return (0 if first_since and wk >= week_start(first_since) else 1, wk, tk)
    return sorted(_week_runs(runs, pick).items(), key=key)


def _google_units(runs: Sequence[dict], first_since: Optional[date] = None
                  ) -> List[Tuple[str, date, List[dict]]]:
    return [(tk, wk, rs) for (tk, wk), rs in _unit_order(runs, first_since, _google_names)]


def google_tasks(runs: Sequence[dict]) -> List[tuple]:
    """Every Google request the plan needs and does not yet hold (splits of a
    week not yet asked are not known and not listed)."""
    return [t for tk, wk, rs in _google_units(runs) for t in _google_unit_tasks(tk, wk, rs)]


def acquire_google(runs: Sequence[dict], budget_s: Optional[float] = None,
                   pace_s: float = GOOGLE_PACE_S, tick_aware: bool = True,
                   max_requests: Optional[int] = None, first_since: Optional[date] = None) -> dict:
    """Fetch every missing Google answer (weekly, then per-run splits), paced,
    tick-aware and self-stopping on throttling. Resumable: answers are cached
    as they arrive, and only a 200 answer is cached."""
    t0, started = time.monotonic(), datetime.now()
    pacer = _Pacer(pace_s)
    net_fails = 0
    n_req, n_ok, statuses, bad = 0, 0, Counter(), 0
    stopped = None
    units = _google_units(runs, first_since)
    logger.info(f"[news-history] google: {len(units)} (ticker, week) unit(s) to check")
    for i_unit, (u_tk, u_wk, u_rs) in enumerate(units, 1):
        if stopped:
            break
        # A unit's week answers first; any split they reveal is asked at once,
        # so every run of a week is complete as soon as its week has been swept.
        pending = _google_unit_tasks(u_tk, u_wk, u_rs)
        seen = set()
        while pending:
            task = pending.pop(0)
            if task in seen:
                continue
            seen.add(task)
            kind, tk, wk, qkind, q, when = task
            if budget_s and time.monotonic() - t0 > budget_s:
                stopped = "budget"
                break
            if max_requests and n_req >= max_requests:
                stopped = "max_requests"
                break
            if tick_aware:
                nr.wait_while_tick_running("historical Google News", phases=("fetch",))
                trouble = live_google_trouble_since(started)
                if trouble:
                    logger.warning(f"[news-history] the LIVE Google leg reported trouble "
                                   f"({trouble.strip()}) — history leg holding "
                                   f"{GOOGLE_COOLOFF_S / 60:.0f} min")
                    time.sleep(GOOGLE_COOLOFF_S)
                    started = datetime.now()
            pacer.wait()
            after, before = _gweek_window(wk) if kind == "week" else _grun_window(when)
            try:
                status, entries = _google_fetch(q, after, before)
            except Exception as exc:                            # noqa: BLE001
                logger.debug(f"[news-history] google {tk} {qkind}: {exc}")
                status, entries = -1, []
            n_req += 1
            if status == -1:
                # No HTTP answer at all (timeout, DNS, reset): the network, not
                # Google's verdict on this machine — wait and ask again, and stop
                # only if it stays down.
                net_fails += 1
                if net_fails >= NET_MAX_FAILS:
                    stopped = f"network down ({net_fails} transport failures in a row)"
                    logger.warning(f"[news-history] {stopped} — Google leg STOPPED")
                    break
                logger.warning(f"[news-history] google transport failure ({net_fails}) — "
                               f"retrying in {NET_RETRY_S:.0f}s")
                time.sleep(NET_RETRY_S)
                seen.discard(task)
                pending.insert(0, task)
                continue
            net_fails = 0
            if status != 200:
                statuses[status] += 1
                bad += 1
                if bad > GOOGLE_MAX_BAD:
                    stopped = f"google answered non-200 {dict(statuses)}"
                    logger.warning(f"[news-history] {stopped} — Google leg STOPPED")
                    break
                logger.warning(f"[news-history] google non-200 ({status}) — cooling off "
                               f"{GOOGLE_COOLOFF_S / 60:.0f} min, then slower")
                time.sleep(GOOGLE_COOLOFF_S)
                pacer.interval *= 1.5
                seen.discard(task)
                pending.insert(0, task)          # the same request again, after the cool-off
                continue
            bad = 0
            n_ok += 1
            path = _gweek_path(tk, wk, qkind) if kind == "week" else _grun_path(tk, when, qkind)
            _write_json(path, {"q": q, "after": after.isoformat(), "before": before.isoformat(),
                               "fetched_at": datetime.now(timezone.utc).isoformat(),
                               "n": len(entries), "entries": entries})
            if kind == "week":
                pending += [t for t in _google_unit_tasks(u_tk, u_wk, u_rs)
                            if t not in seen and t not in pending]
            if n_req % 200 == 0:
                logger.info(f"[news-history] google: {n_req} requests ({n_ok} ok), unit "
                            f"{i_unit}/{len(units)} (week {u_wk}) in "
                            f"{(time.monotonic() - t0) / 60:.0f} min")
    return {"requests": n_req, "ok": n_ok, "non200": dict(statuses), "stopped": stopped,
            "elapsed_min": round((time.monotonic() - t0) / 60, 1),
            "remaining": len(google_tasks(runs)) if not stopped else None}


# ── acquisition: Finnhub ─────────────────────────────────────────────────────

def _finnhub_fetch(tk: str, frm: date, to: date) -> Tuple[int, list]:
    import httpx

    from src.data.provider_news import _FINNHUB_NEWS
    r = httpx.get(_FINNHUB_NEWS, params={"symbol": tk.upper(), "from": frm.isoformat(),
                                         "to": to.isoformat(), "token": settings.finnhub_api_key},
                  timeout=20)
    if r.status_code != 200:
        return r.status_code, []
    items = r.json() or []
    return 200, items if isinstance(items, list) else []


def _finnhub_unit_tasks(tk: str, wk: date, rs: Sequence[dict]) -> List[tuple]:
    """The Finnhub requests one (ticker, week) still needs: the week, then —
    when it came back at the cap — live's exact request for each ET day."""
    meta = _read_json(_fweek_path(tk, wk))
    if not _fresh(meta, max(r["when"] for r in rs)):
        return [("week", tk, wk, None)]
    tasks = []
    if int(meta.get("n", 0)) >= FINNHUB_CAP:
        for d in sorted({r["day"] for r in rs}):
            lastd = max(r["when"] for r in rs if r["day"] == d)
            if not _fresh(_read_json(_fday_path(tk, d)), lastd):
                tasks.append(("day", tk, wk, d))
    return tasks


def finnhub_tasks(runs: Sequence[dict]) -> List[tuple]:
    return [t for (tk, wk), rs in _unit_order(runs, None, _finnhub_names)
            for t in _finnhub_unit_tasks(tk, wk, rs)]


def acquire_finnhub(runs: Sequence[dict], budget_s: Optional[float] = None,
                    pace_s: float = FINNHUB_PACE_S, tick_aware: bool = True,
                    max_requests: Optional[int] = None, first_since: Optional[date] = None) -> dict:
    """Fetch every missing Finnhub answer (weekly, then live's exact per-day
    request where a week came back truncated), one (ticker, week) at a time so
    runs complete in order. Resumable, paced, tick-aware; a 429 waits a minute
    and retries (the free tier's window is per minute)."""
    if not settings.finnhub_api_key:
        return {"skipped": "no FINNHUB_API_KEY"}
    t0 = time.monotonic()
    pacer = _Pacer(pace_s)
    n_req, n_ok, errors, stopped = 0, 0, Counter(), None
    units = _unit_order(runs, first_since, _finnhub_names)
    logger.info(f"[news-history] finnhub: {len(units)} (ticker, week) unit(s) to check")
    for i_unit, ((u_tk, u_wk), u_rs) in enumerate(units, 1):
        if stopped:
            break
        pending, seen = _finnhub_unit_tasks(u_tk, u_wk, u_rs), set()
        while pending:
            task = pending.pop(0)
            if task in seen:
                continue
            seen.add(task)
            kind, tk, wk, d = task
            if budget_s and time.monotonic() - t0 > budget_s:
                stopped = "budget"
                break
            if max_requests and n_req >= max_requests:
                stopped = "max_requests"
                break
            frm, to = _fweek_window(wk) if kind == "week" else (d - timedelta(days=3), d)
            status, items = -1, []
            for _attempt in range(4):
                if tick_aware:
                    nr.wait_while_tick_running("historical Finnhub", phases=("fetch",))
                pacer.wait()
                try:
                    status, items = _finnhub_fetch(tk, frm, to)
                except Exception as exc:                        # noqa: BLE001
                    logger.debug(f"[news-history] finnhub {tk}: {exc}")
                    status = -1
                n_req += 1
                if status != 429:
                    break
                time.sleep(65)
            if status != 200:
                errors[status] += 1
                continue
            n_ok += 1
            path = _fweek_path(tk, wk) if kind == "week" else _fday_path(tk, d)
            _write_json(path, {"from": frm.isoformat(), "to": to.isoformat(),
                               "fetched_at": datetime.now(timezone.utc).isoformat(),
                               "n": len(items), "items": items})
            if kind == "week":
                pending += [t for t in _finnhub_unit_tasks(u_tk, u_wk, u_rs)
                            if t not in seen and t not in pending]
            if n_req % 500 == 0:
                logger.info(f"[news-history] finnhub: {n_req} requests ({n_ok} ok), unit "
                            f"{i_unit}/{len(units)} (week {u_wk}) in "
                            f"{(time.monotonic() - t0) / 60:.0f} min")
    return {"requests": n_req, "ok": n_ok, "errors": dict(errors), "stopped": stopped,
            "elapsed_min": round((time.monotonic() - t0) / 60, 1),
            "remaining": len(finnhub_tasks(runs)) if not stopped else None}


# ── per-run legs ─────────────────────────────────────────────────────────────

def google_leg(run: dict) -> Tuple[Optional[list], dict]:
    """The run's Google articles, live's rules: 24h window at the tick, the
    tag a mention confirms, the publisher-labelled source, URL-deduped (a URL
    two tickers' searches both returned keeps both confirmed tags). None when
    any eligible ticker's answer is not in yet."""
    from src.data.news_fetcher import _confirmed_tags
    from src.data.provider_news import _parse_iso
    from src.models import NewsArticle
    when = run["when"]
    lo = when - timedelta(hours=24)
    wk = week_start(run["day"])
    by_url: Dict[str, NewsArticle] = {}
    missing, undated = 0, 0
    if not run.get("feed_known", True):
        return None, {"google_coverage": "unknown"}
    for tk in _google_names(run):
        entries = []
        for qkind, _q in _google_queries(tk):
            meta = _read_json(_gweek_path(tk, wk, qkind))
            if not _fresh(meta, when):
                missing += 1
                continue
            entries += meta.get("entries") or []
            if int(meta.get("n", 0)) >= GOOGLE_SPLIT_MIN:
                sm = _read_json(_grun_path(tk, when, qkind))
                if not _fresh(sm, when):
                    missing += 1
                    continue
                entries += sm.get("entries") or []
        for e in entries:
            pub = _parse_iso(e.get("published"))
            if pub is None:
                undated += 1                  # live stamps these with fetch time: unreproducible
                continue
            if not (lo <= pub <= when) or not e.get("title"):
                continue
            url = e.get("link") or ""
            tags = _confirmed_tags(tk, e["title"], e.get("summary") or "")
            have = by_url.get(url)
            if have is not None:
                if tags and not set(tags) <= set(have.tickers or []):
                    have.tickers = sorted(set(have.tickers or []) | set(tags))
                continue
            by_url[url] = NewsArticle(title=e["title"], summary=e.get("summary") or "", url=url,
                                      source=e.get("source") or "google_news", published_at=pub,
                                      tickers=list(tags))
    prov = {"n_google": len(by_url), "google_missing": missing, "google_undated": undated}
    return (None if missing else list(by_url.values())), prov


def _finnhub_rule(items: list, tk: str, when: datetime, d: date,
                  max_per_ticker: int = 15) -> list:
    """`provider_news.fetch_finnhub_news` at the tick: UTC-date window
    [D-3, D] (what `from`/`to` select), nothing published after the tick,
    newest first, noise dropped, the 15 newest kept."""
    from src.data.provider_news import _is_finnhub_noise
    from src.models import NewsArticle
    lo, hi = d - timedelta(days=3), d
    rows = []
    for it in items or []:
        ts = it.get("datetime")
        try:
            pub = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            continue
        if pub > when or not (lo <= pub.date() <= hi):
            continue
        rows.append((pub, it))
    rows.sort(key=lambda x: x[0], reverse=True)
    out = []
    for pub, it in rows:
        if len(out) >= max_per_ticker:
            break
        headline = (it.get("headline") or "").strip()
        url = (it.get("url") or "").strip()
        if not headline or not url:
            continue
        source = it.get("source") or "Finnhub"
        if _is_finnhub_noise(headline, source):
            continue
        out.append(NewsArticle(title=headline, summary=(it.get("summary") or "")[:1000], url=url,
                               source=source, published_at=pub, tickers=[tk.upper()],
                               provider_sentiment_source="finnhub"))
    return out


def finnhub_leg(run: dict) -> Tuple[Optional[list], dict]:
    from src.data.provider_news import _dedupe
    when, d = run["when"], run["day"]
    wk = week_start(d)
    out, missing, truncated = [], 0, 0
    if not run.get("feed_known", True):
        return None, {"finnhub_coverage": "unknown"}
    for tk in _finnhub_names(run):
        meta = _read_json(_fweek_path(tk, wk))
        if not _fresh(meta, when):
            missing += 1
            continue
        items = meta.get("items") or []
        if int(meta.get("n", 0)) >= FINNHUB_CAP:
            truncated += 1
            dm = _read_json(_fday_path(tk, d))
            if not _fresh(dm, when):
                missing += 1
                continue
            items = dm.get("items") or []
        out += _finnhub_rule(items, tk, when, d)
    out = _dedupe(out)
    prov = {"n_finnhub": len(out), "finnhub_missing": missing, "finnhub_truncated_weeks": truncated}
    return (None if missing else out), prov


def _daily_articles(prefix: str, d: date) -> Optional[list]:
    """A feed's BUILT articles for ET date ``d`` — the file every tick of that
    day read (written by the day's first tick). None when the day has none."""
    from src.data.cache import CACHE_DIR
    from src.models import NewsArticle
    p = CACHE_DIR / f"{prefix}_{d.isoformat()}.json"
    if not p.exists():
        return None
    try:
        return [NewsArticle.model_validate(a) for a in json.loads(p.read_text(encoding="utf-8"))]
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"[news-history] unreadable {p.name}: {exc}")
        return None


def _quiver_legs(tickers: Sequence[str], d: date) -> Tuple[Dict[str, list], dict]:
    """Quiver contracts / lobbying / dark pool exactly as live built them on ET
    date ``d``: live's own builders, fed the raw payloads cached that day and
    that day's `date.today()`. Cache-only — a missing payload yields nothing,
    never a fetch (a fetch today would return today's data)."""
    from src.data import quiver
    from src.data.cache import CACHE_DIR

    def cached_get(path: str, *, raise_on_error: bool = False) -> list:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")
        p = CACHE_DIR / f"quiver_{slug}_{d.isoformat()}.json"
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:                                       # noqa: BLE001
            return []

    class _Day(date):
        @classmethod
        def today(cls):
            return d

    universe = sorted({t.upper() for t in tickers})
    have_dp = [t for t in universe
               if (CACHE_DIR / f"quiver_historical_offexchange_{t}_{d.isoformat()}.json").exists()]
    out = {"quiver_contracts": [], "quiver_lobbying": [], "quiver_darkpool": []}
    with patch.object(quiver, "_get", cached_get), patch.object(quiver, "date", _Day), \
            patch.object(quiver, "is_available", lambda: True):
        out["quiver_contracts"] = quiver.fetch_gov_contracts(universe)
        out["quiver_lobbying"] = quiver.fetch_lobbying(universe)
        # Live walks its first `quiver_offexchange_max_tickers` names in a
        # discovery order that is not stored; the names it walked that day are
        # the ones whose payload was cached, so those are rebuilt (in batches,
        # since the builder applies the cap per call).
        cap = max(1, int(settings.quiver_offexchange_max_tickers or 60))
        for i in range(0, len(have_dp), cap):
            out["quiver_darkpool"] += quiver.fetch_offexchange(have_dp[i:i + cap])
    return out, {"quiver_darkpool_payloads": len(have_dp)}


_SEC_8K = None


def _sec_8k_frame():
    """8-K / 8-K/A rows of the deep store's SEC filings (acceptance is UTC)."""
    global _SEC_8K
    if _SEC_8K is None:
        import duckdb
        import pandas as pd

        from src.data.cache import CACHE_DIR
        p = CACHE_DIR / "ml" / "deep" / "sec_filings.parquet"
        if not p.exists():
            _SEC_8K = pd.DataFrame(columns=["ticker", "cik", "accession", "filing_date",
                                            "acceptance", "form", "items", "primary_doc"])
        else:
            con = duckdb.connect()
            try:
                _SEC_8K = con.execute(
                    "SELECT ticker, cik, accession, filing_date, acceptance, form, items, primary_doc "
                    f"FROM read_parquet('{p.as_posix()}') WHERE form IN ('8-K', '8-K/A') "
                    "AND filing_date >= '2026-01-01'").fetchdf()
            finally:
                con.close()
    return _SEC_8K


def _eight_k_leg(tickers: Sequence[str], when: datetime, d: date) -> list:
    """`eight_k.fetch_8k_articles` at the tick, from the deep store: every
    material 8-K accepted by the tick and filed within live's look-back, built
    by live's own `_build_article`."""
    import pandas as pd

    from src.data import eight_k
    from src.data.company_names import company_name
    df = _sec_8k_frame()
    if df is None or df.empty:
        return []
    cutoff = (d - timedelta(days=int(settings.eight_k_lookback_days))).isoformat()
    t_naive = when.astimezone(timezone.utc).replace(tzinfo=None)
    uni = {t.upper() for t in tickers}
    acc = pd.to_datetime(df["acceptance"])
    sub = df[df["ticker"].str.upper().isin(uni) & (df["filing_date"] >= cutoff)
             & (df["filing_date"] <= d.isoformat()) & (acc <= t_naive)]
    out, seen = [], set()
    for r in sub.sort_values(["ticker", "acceptance"], ascending=[True, False]).itertuples(index=False):
        items = eight_k._parse_items(str(r.items or ""))
        if not items:
            continue
        try:
            fd = date.fromisoformat(str(r.filing_date)[:10])
            cik_int = int(str(r.cik))
        except (TypeError, ValueError):
            continue
        tk = str(r.ticker).upper()
        a = eight_k._build_article(tk, company_name(tk) or tk, cik_int, fd, items,
                                   str(r.accession or ""), str(r.primary_doc or ""))
        if a and a.url not in seen:
            seen.add(a.url)
            out.append(a)
    return out


def event_legs(run: dict) -> Tuple[Dict[str, list], dict]:
    d, when = run["day"], run["when"]
    legs: Dict[str, list] = {}
    prov: Dict[str, object] = {}
    missing = []
    for leg, prefix in _DAILY_ARTICLE_CACHES.items():
        arts = _daily_articles(prefix, d)
        if arts is None:
            missing.append(leg)
            arts = []
        legs[leg] = arts
    q, qprov = _quiver_legs(_feed(run), d)
    legs.update(q)
    legs["8k"] = _eight_k_leg(_feed(run), when, d)
    prov.update(qprov)
    prov["events_missing_caches"] = missing
    for leg in EVENT_LEGS:
        prov[f"n_{leg}"] = len(legs.get(leg) or [])
    return legs, prov


def run_legs(run: dict, index=None, need: Sequence[str] = ALL_LEGS) -> Tuple[Dict[str, Optional[list]], dict]:
    """Every leg of one run (None = not acquired yet), with provenance."""
    legs: Dict[str, Optional[list]] = {}
    prov: Dict[str, object] = {"n_feed_tickers": len(_feed(run)),
                               "feed_universe_known": bool(run.get("feed_known"))}
    need = set(need)
    if "bundle" in need:
        index = index if index is not None else nr._bundle_index()
        bundle, _stamp, path = nr._bundle_leg(index, run["when"])
        legs["bundle"] = nr._reconfirm_tags(bundle)
        prov["bundle_file"] = path.name if path else None
        prov["n_bundle"] = len(bundle)
    if "polygon" in need:
        legs["polygon"] = nr._polygon_as_of(run["when"], set(_feed(run)))
        prov["n_polygon"] = len(legs["polygon"])
    if need & set(EVENT_LEGS):
        ev, eprov = event_legs(run)
        legs.update(ev)
        prov.update(eprov)
    if "finnhub" in need:
        legs["finnhub"], fprov = finnhub_leg(run)
        prov.update(fprov)
    if "google" in need:
        legs["google"], gprov = google_leg(run)
        prov.update(gprov)
    return legs, prov


def group_pool(group: str, legs: Dict[str, Optional[list]]) -> Optional[list]:
    """The group's pool in live's merge order, URL-deduped (first wins); None
    while any of its legs is not acquired."""
    from src.data.news_fetcher import _dedupe_by_url
    parts = []
    for leg in GROUP_LEGS[group]:
        arts = legs.get(leg)
        if arts is None:
            return None
        parts += arts
    return _dedupe_by_url(parts)


# ── scoring ─────────────────────────────────────────────────────────────────

def groups_done(run_id: str) -> Dict[str, int]:
    from src.db import repo
    df = repo.fetch_df("SELECT pool_spec, count(*) AS n FROM news_replay WHERE run_id = ? "
                       "AND pool_spec LIKE 'src:%' GROUP BY 1", [run_id])
    return {} if df is None or df.empty else {str(s)[len(SPEC_PREFIX):]: int(n)
                                              for s, n in zip(df["pool_spec"], df["n"])}


class _MentionIndex:
    """`company_names.mentions` answered from a per-run table.

    `filter_relevant_articles` asks `mentions(ticker, title + summary)` for every
    (ticker, untagged article) — ~32 us a pair, so a run's ~7k articles x ~380
    tickers cost ~85 s, and six groups ask it again over the same texts. The
    index asks each (ticker, distinct text) ONCE over the union of the run's
    legs and keeps only the positives; any text or ticker it was not built for
    falls through to the real function, so a miss can never change an answer."""

    def __init__(self, articles: Sequence, tickers: Sequence[str]):
        from src.data import company_names as cn
        self._cn = cn
        self._real = cn.mentions
        self._texts = {f"{a.title or ''} {a.summary or ''}" for a in articles}
        self._tickers = {str(t).upper() for t in tickers}
        self._pos: Dict[str, set] = {t: set() for t in self._tickers}
        for text in self._texts:
            for tk in self._tickers:
                if self._real(tk, text):
                    self._pos[tk].add(text)

    def mentions(self, ticker: str, text: str, allow_token: bool = False) -> bool:
        if not allow_token and ticker in self._tickers and text in self._texts:
            return text in self._pos[ticker]
        return self._real(ticker, text, allow_token=allow_token)

    def __enter__(self):
        self._cn.mentions = self.mentions
        return self

    def __exit__(self, *exc):
        self._cn.mentions = self._real
        return False


def _write_run_prov(run_id: str, prov: dict) -> None:
    from src.data.cache import CACHE_DIR
    d = CACHE_DIR / "news_hist" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.json"
    old = _read_json(p) or {}
    old.update(prov)
    _write_json(p, old)


PREWARM_WORKERS = 2        # the local server's parallel slots (`OLLAMA_NUM_PARALLEL`)
# A call that fails while the model is answering everything else is a CONTENT
# failure: the answer for that digest could not be read (no `score`, a word where
# a number belongs — SPY 2026-09-10, twice). Live records exactly 0.0 for that
# ticker, so the history does too, after one retry. Many failures, or failures
# with few good answers beside them, mean the SERVER is the problem, and nothing
# is stored.
MAX_CONTENT_FAILS = 3
MIN_GOOD_PER_FAIL = 5


def content_failures_only(rows: Sequence[dict]) -> bool:
    """True when a group's failed calls are the model's, not the server's: few
    of them, beside enough readable LLM verdicts to show the server answering."""
    failed = sum(1 for r in rows if r.get("scorer_failed"))
    good = sum(1 for r in rows if not r.get("scorer_failed") and r.get("news_raw_score") is not None)
    return 0 < failed <= MAX_CONTENT_FAILS and good >= MIN_GOOD_PER_FAIL * failed


def _prewarm_verdicts(pool: list, tickers: Sequence[str], when: datetime, engine: str,
                      tick_aware: bool) -> None:
    """Fill the verdict cache for a group on the server's parallel slots.

    The feature pass must run in ONE thread: the point-in-time cutoff
    (`analysis_asof`) is thread-local, so a worker thread would compute the
    derived family with no cutoff at all. The LLM call is the part worth
    parallelising and it needs no cutoff (its inputs are the digest and the
    tick instant), so it runs here first with exactly `_replay_one`'s
    arguments; the sequential pass then reads every verdict from the cache."""
    from concurrent.futures import ThreadPoolExecutor

    from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles

    def one(tk: str) -> None:
        arts = filter_relevant_articles(tk, pool)
        if not arts:
            return
        if tick_aware:
            nr.wait_while_tick_running("per-source scoring", phases=("sentiment",))
        try:
            analyse_sentiment(tk, arts, force_engine=engine, as_of=when, allow_provider=True,
                              store_digest=False)
        except Exception as exc:                                # noqa: BLE001
            logger.debug(f"[news-history] prewarm {tk}: {exc}")

    with ThreadPoolExecutor(max_workers=PREWARM_WORKERS, thread_name_prefix="nh-prewarm") as ex:
        list(ex.map(one, tickers))


def score_run(run: dict, groups: Sequence[str] = GROUPS, engine: str = "local",
              index=None, tick_aware: bool = True) -> dict:
    """Score every ready, not-yet-stored group of one run. A group is written
    whole or not at all (an engine failure reads as an abstention, so a run with
    any failed call is left undone for the next pass)."""
    from src.analysis.asof import analysis_asof
    from src.analysis.news_clustering import cluster_mode, set_corpus
    from src.db import repo
    done = groups_done(run["run_id"])
    todo = [g for g in groups if g not in done]
    if not run.get("feed_known", True):
        todo = [g for g in todo if g not in _COVERAGE_GROUPS]
    if not todo:
        return {"run_id": run["run_id"], "status": "done"}
    need = sorted({leg for g in todo for leg in GROUP_LEGS[g]})
    legs, prov = run_legs(run, index=index, need=need)
    _write_run_prov(run["run_id"], {k: v for k, v in prov.items()})
    info = {k: run[k] for k in ("run_id", "tickers", "when", "signal_date", "prices")}
    out = {"run_id": run["run_id"], "when": run["when"].isoformat(), "groups": {}}
    pools = {g: group_pool(g, legs) for g in todo}
    for g in todo:
        if pools[g] is None:
            out["groups"][g] = "not acquired"
    todo = [g for g in todo if pools[g] is not None]
    if not todo:
        return out
    ready_legs = sorted({leg for g in todo for leg in GROUP_LEGS[g]})
    union = [a for leg in ready_legs for a in (legs.get(leg) or [])]
    t_idx = time.perf_counter()
    index_ctx = _MentionIndex(union, run["tickers"])
    logger.info(f"[news-history] {run['run_id']}: relevance index over {len(union)} articles "
                f"x {len(run['tickers'])} tickers in {time.perf_counter() - t_idx:.0f}s")
    with analysis_asof(run["signal_date"]), index_ctx:
        for g in todo:
            pool = pools[g]
            spec = pool_spec(g)
            gprov = {"n_pool": len(pool), "pool_spec": spec,
                     "bundle_file": prov.get("bundle_file") if "bundle" in GROUP_LEGS[g] else None}
            try:
                if cluster_mode() in ("content", "hybrid"):
                    set_corpus(pool)            # the group's OWN pool: a source-only feature
            except Exception as exc:                            # noqa: BLE001
                logger.warning(f"[news-history] clustering corpus failed: {exc}")
            baselines = nr.baselines_as_of(run["signal_date"], spec)
            rows, t0 = [], time.perf_counter()
            _prewarm_verdicts(pool, run["tickers"], run["when"], engine, tick_aware)
            t_warm = time.perf_counter() - t0
            for tk in run["tickers"]:
                if tick_aware:
                    nr.wait_while_tick_running("per-source scoring", phases=("sentiment",))
                try:
                    r = nr._replay_one(tk, pool, info, engine, gprov, baselines)
                except Exception as exc:                        # noqa: BLE001
                    logger.warning(f"[news-history] {run['run_id']} {g} {tk}: {exc}")
                    r = {"ticker": tk, "scorer_failed": True}
                r["replay_version"] = HISTORY_VERSION
                rows.append(r)
            for i, r in enumerate(rows):
                if r.get("scorer_failed"):               # the model is not deterministic: once more
                    try:
                        rows[i] = nr._replay_one(r["ticker"], pool, info, engine, gprov, baselines)
                        rows[i]["replay_version"] = HISTORY_VERSION
                    except Exception as exc:                    # noqa: BLE001
                        logger.warning(f"[news-history] {run['run_id']} {g} {r['ticker']}: {exc}")
            failed_tks = [r["ticker"] for r in rows if r.get("scorer_failed")]
            views = sum(1 for r in rows if (r.get("news") or 0) != 0)
            if failed_tks and not content_failures_only(rows):
                out["groups"][g] = f"{len(failed_tks)} scorer failure(s) — not stored"
                logger.error(f"[news-history] {run['run_id']} {g}: {len(failed_tks)} scorer "
                             f"failure(s) — group NOT stored, retried next pass")
                continue
            if failed_tks:
                # every field of a failed row is already 0.0 (`_replay_one` scores the
                # derived family from news = 0): exactly what live records
                logger.warning(f"[news-history] {run['run_id']} {g}: unreadable answer for "
                               f"{failed_tks} after a retry — stored as 0.0, as live records it")
            repo.insert_news_replay(rows)
            out["groups"][g] = {"pool": len(pool), "views": views,
                                "seconds": round(time.perf_counter() - t0, 1)}
            logger.info(f"[news-history] {run['run_id']} {spec}: pool {len(pool)}, "
                        f"{views}/{len(rows)} views, {time.perf_counter() - t0:.0f}s "
                        f"(verdicts {t_warm:.0f}s on {PREWARM_WORKERS} slots)")
    return out


def score_all(runs: Sequence[dict], groups: Sequence[str] = GROUPS, engine: str = "local",
              tick_aware: bool = True, loop: bool = False, poll_s: float = 600.0,
              budget_s: Optional[float] = None) -> dict:
    """Score runs oldest first (so each group's `news_shock` baseline builds in
    order), every ready group of each; with ``loop``, keep waiting for the
    acquisition legs until every group of every run is stored."""
    index = nr._bundle_index()
    t0 = time.monotonic()
    fails = 0
    while True:
        progressed, pending = False, 0
        for run in runs:
            if budget_s and time.monotonic() - t0 > budget_s:
                return {"stopped": "budget"}
            try:
                res = score_run(run, groups=groups, engine=engine, index=index,
                                tick_aware=tick_aware)
            except Exception as exc:                            # noqa: BLE001
                # a leg that trips a point-in-time guard stops THIS run, loudly
                logger.error(f"[news-history] {run['run_id']}: {exc}")
                pending += 1
                continue
            gs = res.get("groups", {})
            if any(isinstance(v, dict) for v in gs.values()):
                progressed = True
            pending += sum(1 for v in gs.values() if not isinstance(v, dict))
            if any(isinstance(v, str) and "failure" in v for v in gs.values()):
                fails += 1
                if fails >= 5:
                    logger.error("[news-history] repeated scorer failures — is the local "
                                 "LLM server up? stopping")
                    return {"stopped": "scorer failures"}
                time.sleep(300)
        if not pending or not loop:
            return {"pending": pending}
        if not progressed:
            logger.info(f"[news-history] {pending} group-run(s) waiting on acquisition — "
                        f"sleeping {poll_s / 60:.0f} min")
            time.sleep(poll_s)


# ── status ──────────────────────────────────────────────────────────────────

def status(runs: Sequence[dict]) -> dict:
    from src.db import repo
    rids = [r["run_id"] for r in runs]
    df = repo.fetch_df(
        "SELECT pool_spec, count(DISTINCT run_id) AS runs, count(*) AS n, "
        "       sum(CASE WHEN news <> 0 THEN 1 ELSE 0 END) AS views "
        f"FROM news_replay WHERE pool_spec LIKE 'src:%' AND run_id IN ({', '.join('?' * len(rids))}) "
        "GROUP BY 1 ORDER BY 1", rids) if rids else None
    scored = {} if df is None or df.empty else {
        str(s): {"runs": int(r), "rows": int(n), "views": int(v or 0)}
        for s, r, n, v in zip(df["pool_spec"], df["runs"], df["n"], df["views"])}
    return {"runs": len(runs), "first": runs[0]["run_id"] if runs else None,
            "last": runs[-1]["run_id"] if runs else None,
            "google_requests_pending": len(google_tasks(runs)),
            "finnhub_requests_pending": len(finnhub_tasks(runs)),
            "scored": scored}


# ── evaluation ──────────────────────────────────────────────────────────────

_FEATURES = ("news", "news_raw_score", "news_catalyst", "news_article_count", "n_articles")

# Every column a source group stores: the whole news family plus the inputs
# the stacker reads beside it.
FEATURE_COLUMNS = nr.NEWS_REPLAY_COLUMNS + ("news_catalyst", "news_recency_mass",
                                            "news_article_count", "n_articles")


def source_feature_frame(groups: Sequence[str] = GROUPS,
                         features: Sequence[str] = FEATURE_COLUMNS,
                         runs: Optional[Sequence[str]] = None):
    """The BY-SOURCE feature matrix: one row per (run_id, ticker), one column per
    ``<feature>@<group>`` — ``news@polygon``, ``news_raw_score@google``,
    ``news_shock@finnhub`` … — the shape a model trains on, keyed run-exact so it
    joins the panel on (run_id, ticker).

    NaN and 0.0 mean different things and are kept apart: 0.0 is a source that
    was read and had no view on the name (no relevant article, or the scorer
    abstained); NaN is a source not built for that run (its live coverage is
    unknown before 2026-07-03, or it is not scored yet). A model must not be
    handed one as the other."""
    import pandas as pd

    from src.db import repo
    groups = [g for g in groups if g in GROUPS]
    specs = [pool_spec(g) for g in groups]
    where = f"pool_spec IN ({', '.join('?' * len(specs))})"
    params: list = list(specs)
    if runs:
        where += f" AND run_id IN ({', '.join('?' * len(runs))})"
        params += list(runs)
    cols = [c for c in features]
    df = repo.fetch_df(
        f"SELECT run_id, ticker, signal_date, generated_at, pool_spec, {', '.join(cols)} "
        f"FROM news_replay WHERE {where}", params)
    if df is None or df.empty:
        return pd.DataFrame()
    df["group"] = df["pool_spec"].str[len(SPEC_PREFIX):]
    keys = ["run_id", "ticker"]
    meta = df.groupby(keys, as_index=False)[["signal_date", "generated_at"]].first()
    wide = df.pivot(index=keys, columns="group", values=cols)
    wide.columns = [f"{feat}@{grp}" for feat, grp in wide.columns]
    ordered = [f"{feat}@{grp}" for grp in groups for feat in cols if f"{feat}@{grp}" in wide.columns]
    return meta.merge(wide[ordered].reset_index(), on=keys, how="left")


def load_group_features(groups: Sequence[str] = GROUPS) -> "object":
    """Long frame: one row per (run, ticker, group) of the stored source groups."""
    from src.db import repo
    specs = [pool_spec(g) for g in groups]
    return repo.fetch_df(
        f"SELECT run_id, ticker, signal_date, generated_at, pool_spec, {', '.join(_FEATURES)} "
        f"FROM news_replay WHERE pool_spec IN ({', '.join('?' * len(specs))})", specs)



def _tstat(s) -> Tuple[float, float, int]:
    s = s.dropna()
    n = len(s)
    if n < 3:
        return float("nan"), float("nan"), n
    sd = float(s.std(ddof=1))
    return float(s.mean()), (float(s.mean()) / sd * math.sqrt(n)) if sd > 0 else float("nan"), n


def _halves(s) -> Tuple[float, float]:
    s = s.dropna()
    if len(s) < 4:
        return float("nan"), float("nan")
    h = len(s) // 2
    return float(s.iloc[:h].mean()), float(s.iloc[h:].mean())



def quality(groups: Sequence[str] = GROUPS) -> dict:
    """Per group, the house model metrics (`eval_metrics`: IC to the next H/L
    pivot, its day-clustered t, the top / bottom 5% and 3% returns) on EACH test
    set, over the rows where the group holds a view — a 0.0 is an abstention,
    not a score, and ranking it would put a tied block in the middle."""
    import pandas as pd

    from src.analysis.eval_metrics import by_test_set
    from src.analysis.signal_panel import build_panel
    feats = load_group_features(groups)
    if feats is None or feats.empty:
        return {"status": "no stored source rows"}
    panel = build_panel(horizons=(5,), dedupe="last")
    lab = panel[["run_id", "ticker", "fwd_ret_pivot"]].copy()
    lab["ticker"] = lab["ticker"].astype(str)
    out = {}
    for g in groups:
        f = feats[feats["pool_spec"] == pool_spec(g)].copy()
        if f.empty:
            continue
        f["news"] = pd.to_numeric(f["news"], errors="coerce").astype(float).fillna(0.0)
        m = f.merge(lab, on=["run_id", "ticker"], how="inner")
        views = m[m["news"] != 0]
        out[g] = {"rows": int(len(m)), "views": int(len(views)),
                  "coverage": float(len(views) / len(m)) if len(m) else float("nan"),
                  "test_sets": by_test_set(views, "news", label="fwd_ret_pivot")}
    return out


def paired(groups: Sequence[str] = ("events", "polygon", "bundle", "finnhub", "google"),
           label: str = "fwd_ret_pivot", min_n: int = 8) -> dict:
    """Source A against source B ON THE SAME NAMES: per day, the IC of each
    group over the rows where BOTH have a view, and the day-clustered t of the
    difference — the like-for-like answer to "which source reads better",
    since each group's own IC is measured on a different population."""
    import itertools

    import pandas as pd

    from src.analysis.signal_panel import build_panel
    feats = load_group_features(groups)
    if feats is None or feats.empty:
        return {"status": "no stored source rows"}
    feats["news"] = pd.to_numeric(feats["news"], errors="coerce").astype(float).fillna(0.0)
    wide = feats.pivot_table(index=["run_id", "ticker", "signal_date"], columns="pool_spec",
                             values="news", aggfunc="last").reset_index()
    panel = build_panel(horizons=(5,), dedupe="last")
    lab = panel[["run_id", "ticker", label]].copy()
    m = wide.merge(lab, on=["run_id", "ticker"], how="inner")
    m[label] = pd.to_numeric(m[label], errors="coerce").astype(float)
    from src.analysis.eval_metrics import split_test_sets
    out: dict = {}
    for set_name, part in split_test_sets(m).items():      # never pooled across the sets
        res = {}
        for a, b in itertools.combinations(groups, 2):
            ca, cb = pool_spec(a), pool_spec(b)
            if ca not in part.columns or cb not in part.columns:
                continue
            sub = part[(part[ca].fillna(0) != 0) & (part[cb].fillna(0) != 0) & part[label].notna()]
            diffs = {}
            for d, g in sub.groupby("signal_date"):
                if len(g) < min_n:
                    continue
                ra, rb, rl = g[ca].rank(), g[cb].rank(), g[label].rank()
                diffs[d] = ra.corr(rl) - rb.corr(rl)
            ser = pd.Series(diffs, dtype=float).sort_index()
            mean, t, n = _tstat(ser)
            h1, h2 = _halves(ser)
            res[f"{a} - {b}"] = {"rows": int(len(sub)), "days": n, "ic_diff": mean, "t": t,
                                 "half1": h1, "half2": h2}
        out[set_name] = res
    return out


def coverage_by_month(groups: Sequence[str] = GROUPS) -> dict:
    """Share of rows with a view, per group and calendar month — the age
    profile of each leg (Google's search thins with age; a leg that fades
    toward June is a different feature in June than in September)."""
    import pandas as pd
    feats = load_group_features(groups)
    if feats is None or feats.empty:
        return {}
    feats["news"] = pd.to_numeric(feats["news"], errors="coerce").astype(float).fillna(0.0)
    feats["month"] = feats["signal_date"].astype(str).str[:7]
    tab = (feats.assign(view=feats["news"] != 0)
           .groupby(["pool_spec", "month"])["view"].mean().unstack("month"))
    return {str(k)[len(SPEC_PREFIX):]: {m: round(float(v), 3) for m, v in row.items() if v == v}
            for k, row in tab.iterrows()}


def fidelity_vs_live(groups: Sequence[str] = GROUPS, since: str = "2026-09-12") -> dict:
    """Each group's verdict against the LIVE `news` on the same (run, ticker),
    over runs where live ran the current news scorer (the archive era): rank
    correlation where either side has a view, sign agreement where both do,
    direction agreement counting no-view, live views the group keeps, and the
    group's views live agrees with."""
    import pandas as pd

    from src.analysis.signal_panel import _spearman
    from src.db import repo
    feats = load_group_features(groups)
    if feats is None or feats.empty:
        return {"status": "no stored source rows"}
    feats = feats[feats["signal_date"].astype(str) >= since]
    rids = sorted(feats["run_id"].astype(str).unique())
    if not rids:
        return {"status": f"no stored source rows since {since}"}
    live = repo.fetch_df(f"SELECT run_id, ticker, news AS live FROM signals "
                         f"WHERE run_id IN ({', '.join('?' * len(rids))})", rids)
    out = {}
    for g in groups:
        f = feats[feats["pool_spec"] == pool_spec(g)].merge(live, on=["run_id", "ticker"])
        if f.empty:
            continue
        a = pd.to_numeric(f["news"], errors="coerce").astype(float).fillna(0.0)
        b = pd.to_numeric(f["live"], errors="coerce").astype(float).fillna(0.0)
        either, both = (a != 0) | (b != 0), (a != 0) & (b != 0)
        sa, sb = a.apply(lambda x: (x > 0) - (x < 0)), b.apply(lambda x: (x > 0) - (x < 0))
        out[g] = {"rows": int(len(f)), "runs": int(f["run_id"].nunique()),
                  "views": int((a != 0).sum()), "live_views": int((b != 0).sum()),
                  "spearman": float(_spearman(a[either], b[either]) or float("nan"))
                  if either.sum() > 2 else float("nan"),
                  "sign_when_both": float((sa[both] == sb[both]).mean()) if both.any() else float("nan"),
                  "direction_incl_no_view": float((sa[either] == sb[either]).mean())
                  if either.any() else float("nan"),
                  "live_views_kept": float((sa[b != 0] == sb[b != 0]).mean()) if (b != 0).any() else float("nan"),
                  "group_views_right": float((sa[a != 0] == sb[a != 0]).mean()) if (a != 0).any() else float("nan")}
    return out


def calibrate(runs: Sequence[dict]) -> dict:
    """ARTICLE-level fidelity, LLM-free, on archive-era runs: each rebuilt leg
    against the articles the live tick actually held (`news_articles`):
    Finnhub and Google by URL on the tickers live asked, Polygon by the share
    of its as-of set the live pool held, the event feeds by title."""
    import pandas as pd

    from src.data.provider_news import _parse_iso
    from src.db import repo
    t0 = nr.archive_start()
    res = defaultdict(Counter)
    for run in runs:
        when = run["when"]
        if t0 is None or when < t0:
            continue
        arch = repo.fetch_df(
            "SELECT url, title, source, tickers_json FROM news_articles "
            "WHERE CAST(first_seen_at AS TIMESTAMPTZ) <= ? AND CAST(last_seen_at AS TIMESTAMPTZ) >= ?",
            [when, when])
        if arch is None or arch.empty:
            continue
        arch_urls = set(arch["url"].astype(str))
        live_by_feed = defaultdict(lambda: defaultdict(set))       # feed -> ticker -> urls
        for r in arch.itertuples(index=False):
            try:
                tags = json.loads(r.tickers_json) if r.tickers_json else []
            except (TypeError, ValueError):
                tags = []
            feed = ("finnhub" if "finnhub.io" in str(r.url)
                    else "google" if str(r.source).startswith("google_news") else None)
            if feed:
                for t in tags:
                    live_by_feed[feed][str(t).upper()].add(str(r.url))
        legs, _prov = run_legs(run, need=("finnhub", "google", "polygon") + EVENT_LEGS)
        for feed in ("finnhub", "google"):
            arts = legs.get(feed)
            if arts is None:
                res[feed]["runs_not_acquired"] += 1
                continue
            mine = defaultdict(set)
            for a in arts:
                for t in (a.tickers or []):
                    mine[str(t).upper()].add(a.url)
            asked = set(live_by_feed[feed])
            for tk in asked:
                lv, mv = live_by_feed[feed][tk], mine.get(tk, set())
                res[feed]["live"] += len(lv)
                res[feed]["rebuilt_on_live_tickers"] += len(mv)
                res[feed]["both"] += len(lv & mv)
            res[feed]["rebuilt_all_tickers"] += sum(len(v) for v in mine.values())
            res[feed]["runs"] += 1
        pol = legs.get("polygon") or []
        res["polygon"]["asof"] += len(pol)
        res["polygon"]["held_live"] += sum(1 for a in pol if a.url in arch_urls)
        res["polygon"]["runs"] += 1
        live_titles = set(arch.loc[arch["source"].isin(
            ["Earnings/EPS", "Analyst Ratings", "Short Interest", "SEC 8-K Filing", "Quiver Dark Pool",
             "Quiver Gov Contracts", "Quiver Lobbying", "Ticker Events"]), "title"].astype(str))
        for leg in EVENT_LEGS:
            arts = legs.get(leg) or []
            res[f"event:{leg}"]["rebuilt"] += len(arts)
            res[f"event:{leg}"]["held_live"] += sum(1 for a in arts if a.title in live_titles)
    out = {}
    for k, c in res.items():
        d = dict(c)
        if "live" in c:
            d["recall"] = c["both"] / c["live"] if c["live"] else float("nan")
            d["precision_on_live_tickers"] = (c["both"] / c["rebuilt_on_live_tickers"]
                                              if c["rebuilt_on_live_tickers"] else float("nan"))
        if "asof" in c:
            d["share_held_live"] = c["held_live"] / c["asof"] if c["asof"] else float("nan")
        if "rebuilt" in c:
            d["share_held_live"] = c["held_live"] / c["rebuilt"] if c["rebuilt"] else float("nan")
        out[k] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-source news history (build, score, evaluate)")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--only-runs", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--acquire", default=None, help="finnhub | google")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--pace", type=float, default=None, help="seconds between provider requests")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--loop", action="store_true", help="--score: wait for acquisition until done")
    ap.add_argument("--no-tick-aware", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--repair-shock", action="store_true",
                    help="after the build: recompute news_shock per group on the whole window "
                         "(order-independent baseline); dry run unless --apply")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    if argv is None:
        import sys
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    only = [r.strip() for r in args.only_runs.split(",") if r.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f"unknown group(s) {bad}; choose from {GROUPS}")
    tick_aware = not args.no_tick_aware
    budget = args.budget_hours * 3600 if args.budget_hours else None
    if args.status or args.calibrate or args.report:
        from src.db import repo
        repo.set_read_only(True)
    runs = [] if args.repair_shock else plan_runs(days=args.days, only=only or None)
    if args.status:
        print(json.dumps(status(runs), indent=2, default=str))
        return 0
    if args.repair_shock:
        print(json.dumps({g: nr.repair_shock(pool_spec=pool_spec(g), apply=args.apply)
                          for g in groups}, indent=2, default=str))
        return 0
    if args.calibrate:
        print(json.dumps(calibrate(runs), indent=2, default=str))
        return 0
    if args.report:
        single = [g for g in groups if g != "all"]
        print(json.dumps({"quality": quality(groups), "paired_pivot": paired(single),
                          "coverage_by_month": coverage_by_month(groups),
                          "fidelity_vs_live": fidelity_vs_live(groups)},
                         indent=2, default=str))
        return 0
    if args.acquire or args.score:
        # a background build: never compete with the live tick for the CPU
        from src.data.deep.refresh import _below_normal_priority
        _below_normal_priority()
    if args.acquire:
        # feedparser opens URLs with NO timeout, so one stalled read blocked the
        # Google leg for 3 hours (2026-09-24 01:25-04:18). Every socket in a fetch
        # process now gives up after NET_TIMEOUT_S; the loop retries it.
        import socket
        socket.setdefaulttimeout(NET_TIMEOUT_S)
    if args.acquire:
        t_arch = nr.archive_start()
        kw = {"budget_s": budget, "tick_aware": tick_aware, "max_requests": args.max_requests,
              "first_since": local_day(t_arch) if t_arch else None}
        if args.pace:
            kw["pace_s"] = args.pace
        if args.acquire == "google":
            print(json.dumps(acquire_google(runs, **kw), indent=2, default=str))
        elif args.acquire == "finnhub":
            print(json.dumps(acquire_finnhub(runs, **kw), indent=2, default=str))
        else:
            ap.error("--acquire takes finnhub or google")
        return 0
    if args.score:
        nr._isolate_sentiment_cache()
        print(json.dumps(score_all(runs, groups=groups, tick_aware=tick_aware, loop=args.loop,
                                   budget_s=budget), indent=2, default=str))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
