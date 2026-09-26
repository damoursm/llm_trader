"""Per-TICK news replay — today's pipeline, re-run on the data that tick had
(2026-09-07, user directive: "as representative of the current live as possible…
the articles data needs to be fetched by tick, using the same time window as the
current live pipeline… no time traveling obviously").

WHAT A TICK'S POOL WAS, AND WHAT OF IT SURVIVES
-----------------------------------------------
`pipeline` assembles one article pool per tick:

    articles = dedupe(cached_bundle(hour) + fresh_rss)
               + 8k + trends + reddit + analyst + ticker_events + eps + short
               + polygon_news + finnhub_news + google_news + av_news
               + stocktwits + quiver_*

Of that, exactly two legs are recoverable for a PAST tick:

  * `cached_bundle(hour)` — EXACT. The hourly per-ticker yfinance + NewsAPI
    bundle is on disk (`cache/news_YYYY-MM-DD_HH.json`, 1,132 files back to
    2026-06-16). The live tick used the bundle for ITS hour and nothing else
    from that leg, so replaying one hour's file is not an approximation.
  * `polygon_news` — FAITHFULLY re-fetchable. The live call is one market-wide
    `/v2/reference/news?order=desc&limit=1000`; the same endpoint with
    `published_utc.lte=<tick>` returns the same 1,000-most-recent set as of that
    instant. This is the only fetch this module makes, and it is made only
    because the live pipeline demonstrably made it too.

Everything else is GONE and is not substituted: RSS wires and per-ticker Google
News are search/feed reads with no archive (both age-capped at 24h at fetch
time), and the rest are event feeds whose live look-back windows are not
recoverable per tick. Re-fetching any of them today would return articles the
tick could not have seen — the definition of time travel — so this module does
not. **Measured consequence: a replayed digest carries ~3 of the ~7.3 articles
the live digest had** (2026-09-07 pilot, 200 v6-era rows). That is the ceiling
of the archive, not a bug in the replay, and it is reported per run.

WHAT "REPLAY" MEANS HERE
------------------------
The same thing `src/analysis/replay.py` means by it: **what TODAY's code would
have produced given the data available THEN** — not a re-enactment of the code
that ran that day. So the pool is the tick's, and every transform is the current
one: `news_fetcher._confirmed_tags`' rule re-derives the archived (pre-2026-09-04,
unconditional) ticker tags, then `sentiment.filter_relevant_articles`,
`analyse_sentiment` on the pinned engine, and the whole derived news family
(`sent_velocity`, `news_shock`, `news_bear_fresh`, `catalyst_tilt`,
`news_unpriced`, `news_unpriced_all`). That is what makes a replayed value
comparable with a live one instead of with a retired scorer.

NO TIME TRAVEL — the three places it could enter, each closed:
  1. POOL: only bundles stamped at or before the tick, only articles published
     at or before it, and the Polygon query is `published_utc.lte=<tick>`. Every
     article is re-asserted before scoring; a violation aborts the tick.
  2. DERIVED METHODS: the run executes inside `analysis_asof(signal_date)`, so
     `catalyst_tilt`'s map and every other panel-fitted layer see strictly
     earlier rows. `news_shock`'s baseline is bounded by the tick instant.
  3. PRICES: `news_unpriced` anchors on completed daily bars at or before each
     news cluster and marks at the tick's own `signals.price`, never today's.

SCOPE: one tick per DAY by default — the LAST one, because `build_panel`'s
`dedupe="last"` keeps exactly that row per (ticker, day), so it is the row every
downstream training set consumes. ~58 days x ~400 tickers is ~23k local calls at
the 0.54 calls/s ceiling: ~12 hours, resumable, budgeted, and off the tick path.

Rows land in `news_replay` (its OWN table, with provenance: pool composition and
article counts), NOT in `signals_replay` — wiring them into the panel is gated
on the fidelity report clearing a pre-registered bar, and `--fidelity` prints
that report over the v6-era overlap where live verdicts exist to compare with.

TWO MORE POOL SOURCES (2026-09-23)
-----------------------------------
``--source archive`` — FAITHFUL, and the reason the news history no longer has
to reset when the scorer changes. From 2026-09-11 ~16:50 UTC every tick's merged
pool is archived in `news_articles` (`first_seen_at` / `last_seen_at` are run
`generated_at` stamps), so a run's pool is the articles with
`first_seen_at <= run <= last_seen_at`, carrying the tags they were archived
with (confirmed at fetch, as live) — no substitution, no re-derived tags. Only
Polygon's sentiment `insights` are not archived; they are re-attached by URL
from the same as-of Polygon call, so the provider shortcut fires where live's
did. `compare_to_live` CERTIFIES the source: the rebuilt digest's
`news_digest_id` must equal the one the live scorer recorded. These rows — and
ONLY these (`RESTORABLE_POOL_SPECS`) — are restored into the panel by
`restore_news_rescored`, under the news-family epoch they were scored under:
after a scorer change, `--source archive` re-scores every archived run with the
new logic and the history comes back instead of being masked.

``--source hist`` — the June–September question: the bundle and Polygon legs as
above, plus the two feeds that made the old reconstruction weak, rebuilt from
their providers' HISTORY — Finnhub company-news (the free tier serves one
rolling year) and Google News date-bounded search (`after:`/`before:`), with
each fetcher's own live rules (queries, 24h / 3-day windows, caps, noise filter,
tag confirmation). A research arm: its rows are compared with live and never
restored. Both external legs are paced, pause while a live tick runs, and the
Google leg aborts on repeated non-200 answers — the live pipeline shares this
machine's Google and Finnhub quotas.

CLI
    python -m src.analysis.news_replay --days 7 [--limit-tickers N] [--budget-seconds S]
    python -m src.analysis.news_replay --run-id 2026-09-04_235002
    python -m src.analysis.news_replay --fidelity
    python -m src.analysis.news_replay --source archive --only-runs R1,R2 --tick-aware
    python -m src.analysis.news_replay --compare archive --only-runs R1,R2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from bisect import bisect_right
from collections import Counter
from copy import copy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from config.settings import settings

REPLAY_VERSION = "news-replay-v1"
ARCHIVE_POOL_SPEC = "archive"
HIST_POOL_SPEC = "hist-liveset"
# The ONLY pool shapes the panel may restore from: inputs reproduced exactly,
# certified per row by the live `news_digest_id`. Every reconstruction (the
# bundle `faithful`/`union*h` shapes, the `hist-*` pilot) is a different
# quantity from a live score and stays in this table.
RESTORABLE_POOL_SPECS = (ARCHIVE_POOL_SPEC,)
_RECENCY_DAYS = 7               # the scorer's own window (`_recency_weight` zeroes past 168h)
_POLYGON_LIMIT = 1000           # the live `get_news(limit=1000)` call
_FETCH_GRACE_MINUTES = 30       # a run's own news fetch takes minutes

# The news-family columns this module regenerates, in the order they are stored.
NEWS_REPLAY_COLUMNS = (
    "news", "news_raw_score", "sent_velocity", "news_shock", "news_bear_fresh",
    "catalyst_tilt", "news_unpriced", "news_unpriced_all",
    # 2026-09-12: both shipped AFTER this module was written and both are
    # WEIGHTED in the live book (`news_quiet` 0.10, `news_bull_fresh` 0.08), so a
    # backfill without them would regenerate a news family that is missing its
    # two newest voters — and the gap would be invisible, since the columns would
    # simply not exist rather than read wrong.
    "news_quiet", "news_bull_fresh",
)


# ── the tick's article pool ─────────────────────────────────────────────────

def _bundle_index() -> List[Tuple[datetime, Path]]:
    from src.data.cache import CACHE_DIR
    out = []
    for p in CACHE_DIR.glob("news_*.json"):
        try:
            stamp = datetime.strptime(p.name[5:18], "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append((stamp, p))
    return sorted(out)


def _bundle_for(index, when: datetime) -> Tuple[Optional[datetime], Optional[Path]]:
    """The bundle the tick actually used: the hourly file for ITS hour, i.e. the
    newest one stamped at or before it. Never a later file."""
    stamps = [s for s, _p in index]
    pos = bisect_right(stamps, when) - 1
    return index[pos] if pos >= 0 else (None, None)


def _load_bundle(path: Path) -> list:
    from src.models import NewsArticle
    out = []
    try:
        for a in json.loads(path.read_text(encoding="utf-8")):
            try:
                out.append(NewsArticle.model_validate(a))
            except Exception:                                   # noqa: BLE001
                continue
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"[news-replay] unreadable bundle {path.name}: {exc}")
    return out


def _polygon_as_of(when: datetime, universe: set) -> list:
    """The live `fetch_polygon_news` call, as of ``when``.

    Live makes ONE market-wide `/v2/reference/news?order=desc&limit=1000` and
    keeps the articles touching the universe. `published_utc.lte` reproduces the
    same 1,000-most-recent set at a past instant — the same endpoint, ordering
    and limit, so this is a fetch the tick itself made, not new information.
    Cached on disk per tick hour so a resumed run costs nothing.
    """
    if not settings.enable_polygon_news:
        return []
    from src.data.cache import CACHE_DIR
    from src.data.provider_news import _parse_iso
    from src.models import NewsArticle
    # Keyed to the SECOND (2026-09-23): an hourly key served one tick's as-of
    # set to every tick of that hour — up to 30 min stale for the later one,
    # and a tripwire for the earlier one if a later tick's file was written
    # first (its newer articles fail the `published > when` guard below).
    cache = CACHE_DIR / f"polygon_asof_{when:%Y-%m-%dT%H%M%S}.json"
    raw = None
    if cache.exists():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            raw = None
    if raw is None:
        from src.data import polygon_client
        if not polygon_client.is_available():
            return []
        j = polygon_client._get("/v2/reference/news", {
            "published_utc.lte": when.isoformat(), "order": "desc",
            "limit": _POLYGON_LIMIT,
        })
        raw = (j or {}).get("results") or []
        try:
            cache.write_text(json.dumps(raw), encoding="utf-8")
        except Exception:                                       # noqa: BLE001
            pass
    out = []
    for item in raw:
        tags = [t.upper() for t in (item.get("tickers") or [])]
        # Polygon's `insights` ride the history endpoint too, and live maps them
        # onto `provider_insights` (`provider_news.fetch_polygon_news`) — without
        # this the replay cannot take the provider shortcut live took on ~32% of
        # tickers and would LLM-score them instead, diverging on a third of rows
        # AND paying for calls live never made. The RAW label is kept, exactly as
        # live does; `sentiment.py` maps label -> numeric score downstream.
        _ins = {}
        for _i in (item.get("insights") or []):
            _tk = str(_i.get("ticker") or "").upper()
            _lab = str(_i.get("sentiment") or "").strip().lower()
            if _tk and _lab:
                _ins[_tk] = _lab
        if universe and not (universe & set(tags)):
            continue
        published = _parse_iso(item.get("published_utc"))
        title = (item.get("title") or "").strip()
        url = (item.get("article_url") or item.get("amp_url") or "").strip()
        if not title or not url or published is None:
            continue
        if published > when:
            # `published_utc.lte` was ignored — do not quietly drop the row and
            # carry on with a pool that may be silently wrong.
            raise RuntimeError(f"polygon returned an article published {published.isoformat()} "
                               f"after the requested cutoff {when.isoformat()}")
        out.append(NewsArticle(
            title=title, summary=(item.get("description") or title)[:600], url=url,
            source=((item.get("publisher") or {}).get("name") or "Polygon"),
            published_at=published, tickers=tags,
            provider_insights=_ins,
            provider_sentiment_source="polygon" if _ins else None))
    return out


def _reconfirm_tags(articles: list) -> list:
    """Re-derive archived ticker tags with the CURRENT rule.

    The bundles predate the 2026-09-04 relevance fix, so their tags are the old
    unconditional ones (yfinance's related feed and the Google symbol query
    tagged every result with the queried symbol). `filter_relevant_articles`
    gives a tagged article a free pass, so keeping them would feed the scorer
    other companies' news — measured on the pilot as roughly double the live
    abstention rate. Mirrors `news_fetcher._confirmed_tags`; an article that
    loses its tag stays in the pool untagged, where the filter can still attach
    it by company name.
    """
    from copy import copy

    from src.data.company_names import mention_evidence
    if not settings.enable_name_relevance:
        return articles
    out = []
    for a in articles:
        tags = list(getattr(a, "tickers", None) or [])
        if not tags:
            out.append(a)
            continue
        text = f"{a.title or ''} {a.summary or ''}"
        kept = [t for t in tags if mention_evidence(t, text, allow_token=True)]
        b = copy(a)
        b.tickers = kept
        out.append(b)
    return out


def _earlier_bundle_articles(index, when: datetime, hours: float) -> list:
    """Articles from EARLIER bundles, published within ``hours`` of the tick.

    A stand-in for the half of the live digest that cannot be recovered — RSS
    wires and per-ticker Google News. Those were fetched fresh with a 24h age
    cap, so the substitute is chosen to match their AGE PROFILE rather than to
    maximise count: reaching 7 days back does add articles, but they are older
    than anything the missing feeds could have contributed, which is why the
    7-day union ends up 2.3x the live digest size.

    Point-in-time unchanged: only bundles STAMPED at or before the tick are
    read, so these are articles the system genuinely had in hand.
    """
    if hours <= 0:
        return []
    floor = when - timedelta(hours=float(hours))
    seen, out = set(), []
    for stamp, path in index:
        if stamp > when or stamp < when - timedelta(days=_RECENCY_DAYS):
            continue
        for a in _load_bundle(path):
            u = getattr(a, "url", None)
            pub = getattr(a, "published_at", None)
            if not u or u in seen or pub is None or not (floor <= pub <= when):
                continue
            seen.add(u)
            out.append(a)
    return out


# ── courtesy to the live scheduler ──────────────────────────────────────────

_TICK_START_RE = re.compile(r"\[scheduler\] (?:CATCH-UP )?tick for ")
_TICK_END_RE = re.compile(r"\[db\] Persisted run ")


def _log_tail(path: Path, nbytes: int = 3_000_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def live_tick_running(log_dir: Path = Path("logs"), today: Optional[date] = None) -> bool:
    """True while the live scheduler is inside a tick: its last `tick for` line
    is newer than its last `Persisted run` line (a tick's GPU and feed use are
    over by the persist; the email ~10 s later uses neither). Read from the
    scheduler's own log, so nothing in the live process changes; the previous
    day's file answers when today's has no tick yet. Unknown reads as NOT
    running — this is pacing, never a correctness guard."""
    d = today or date.today()
    for day in (d, d - timedelta(days=1)):
        text = _log_tail(Path(log_dir) / f"llm_trader_{day:%Y-%m-%d}.log")
        starts = [m.start() for m in _TICK_START_RE.finditer(text)]
        if not starts:
            continue
        ends = [m.start() for m in _TICK_END_RE.finditer(text)]
        return starts[-1] > (ends[-1] if ends else -1)
    return False


# The phases of a live tick, in the order its log marks them. A tick holds the
# news feeds only while it FETCHES (~3 min: Google, Finnhub and the rest are
# asked in Step 1) and the local LLM only while it scores SENTIMENT (~6 min of
# Step 4), out of ~35 min overnight — so a background job that shares only one
# of the two need not sit out the rest.
_PHASE_MARKS = (
    ("prep", re.compile(r"Steps 1.3: \d+ total articles assembled")),
    ("sentiment", re.compile(r"Signal weights \[")),
    ("post", re.compile(r"\[aggregator\] rank pool: ")),
)
_STALE_TICK = timedelta(minutes=50)       # a tick never persisted (killed) stops counting


def live_tick_phase(log_dir: Path = Path("logs"), today: Optional[date] = None,
                    now: Optional[datetime] = None) -> Optional[str]:
    """The live tick's current phase — ``fetch`` | ``prep`` | ``sentiment`` |
    ``post`` — or None between ticks. Read from the scheduler's own log like
    `live_tick_running`; unknown reads as None (pacing, never a correctness
    guard), and a tick older than 50 min without a persist line is treated as
    over (the watchdog killed it)."""
    d = today or date.today()
    now = now or datetime.now()
    for day in (d, d - timedelta(days=1)):
        text = _log_tail(Path(log_dir) / f"llm_trader_{day:%Y-%m-%d}.log")
        starts = list(_TICK_START_RE.finditer(text))
        if not starts:
            continue
        s = starts[-1].start()
        line_start = text.rfind("\n", 0, s) + 1
        try:
            t_start = datetime.strptime(text[line_start:line_start + 19], "%Y-%m-%d %H:%M:%S")
            if now - t_start > _STALE_TICK:
                return None
        except ValueError:
            pass
        tail = text[s:]
        if _TICK_END_RE.search(tail):
            return None
        phase = "fetch"
        for name, rx in _PHASE_MARKS:
            if rx.search(tail):
                phase = name
        return phase
    return None


def wait_while_tick_running(what: str, poll_s: float = 15.0, max_wait_s: float = 3600.0,
                            phases: Optional[Sequence[str]] = None) -> None:
    """Hold GPU and feed use while a live tick runs (bounded; logged once).
    With ``phases``, hold only while the tick is in one of them."""
    t0, told = time.monotonic(), False

    def busy() -> bool:
        if phases is None:
            return live_tick_running()
        return live_tick_phase() in set(phases)

    while busy() and time.monotonic() - t0 < max_wait_s:
        if not told:
            logger.info(f"[news-replay] live tick running — {what} paused")
            told = True
        time.sleep(poll_s)


# ── FAITHFUL source: the archived merged pool (2026-09-23) ──────────────────

def _archive_pool(when: datetime) -> list:
    """The run's MERGED pool as the live run held it (see the module doc):
    `first_seen_at <= when <= last_seen_at` over `news_articles`, tags as
    archived. An article that left every feed and came back reads as present
    in the gap — the one approximation, measured by `compare_to_live`."""
    from src.data.news_fetcher import _dedupe_by_url
    from src.data.provider_news import _parse_iso
    from src.db import repo
    from src.models import NewsArticle
    df = repo.fetch_df(
        "SELECT url, title, source, published_at, summary, tickers_json FROM news_articles "
        "WHERE CAST(first_seen_at AS TIMESTAMPTZ) <= ? AND CAST(last_seen_at AS TIMESTAMPTZ) >= ? "
        "ORDER BY first_seen_at, url_hash", [when, when])
    if df is None or df.empty:
        return []
    grace = when + timedelta(minutes=_FETCH_GRACE_MINUTES)
    out = []
    for r in df.itertuples(index=False):
        pub = _parse_iso(r.published_at)
        if pub is None:
            continue
        if pub > grace:
            # first_seen_at <= when, so a fetch cannot have returned this —
            # the archive's stamps are wrong, and that must be loud.
            raise RuntimeError(f"archived article published {pub.isoformat()} is visible to "
                               f"the run at {when.isoformat()} — news_articles is not point-in-time")
        try:
            tags = json.loads(r.tickers_json) if r.tickers_json else []
        except (TypeError, ValueError):
            tags = []
        out.append(NewsArticle(title=r.title or "", summary=r.summary or "", url=r.url or "",
                               source=r.source or "", published_at=pub, tickers=list(tags)))
    return _dedupe_by_url(out)


def _attach_polygon_insights(pool: list, when: datetime, universe: set) -> Tuple[list, int]:
    """Re-attach Polygon's per-ticker sentiment labels (not archived) by URL from
    the same as-of call live made, so the provider shortcut fires where live's
    did instead of paying an LLM call live never made."""
    by_url = {a.url: a.provider_insights for a in _polygon_as_of(when, universe)
              if a.provider_insights}
    out, n = [], 0
    for a in pool:
        ins = by_url.get(a.url)
        if ins and not a.provider_insights:
            b = copy(a)
            b.provider_insights = dict(ins)
            b.provider_sentiment_source = "polygon"
            out.append(b)
            n += 1
        else:
            out.append(a)
    return out, n


# ── HISTORICAL source: provider history (the June–September pilot) ───────────

_HIST_GOOGLE_PACE_S = 2.0          # ~0.5 req/s — the live tick bursts ~300 in ~30 s
_HIST_GOOGLE_MAX_BAD = 3           # consecutive non-200 answers abort the leg
_HIST_FINNHUB_PACE_S = 1.5         # 40/min against the shared free 60/min


def _hist_cache_path(kind: str, ticker: str, when: datetime) -> Path:
    from src.data.cache import CACHE_DIR
    d = CACHE_DIR / "news_hist"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{when:%Y-%m-%dT%H%M%S}_{ticker.upper()}.json"


def _hist_cached(kind: str, ticker: str, when: datetime) -> Optional[list]:
    """One ticker's historical fetch for one tick, as stored (resumable, and
    shared between the LLM-free recall check and the scoring arm)."""
    from src.models import NewsArticle
    p = _hist_cache_path(kind, ticker, when)
    if not p.exists():
        return None
    try:
        return [NewsArticle.model_validate(x) for x in json.loads(p.read_text(encoding="utf-8"))]
    except Exception:                                           # noqa: BLE001
        return None


def _hist_store(kind: str, ticker: str, when: datetime, arts: list) -> None:
    try:
        _hist_cache_path(kind, ticker, when).write_text(
            json.dumps([a.model_dump(mode="json") for a in arts]), encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass


def live_query_sets(when: datetime) -> Dict[str, List[str]]:
    """PILOT ONLY — which tickers the live tick sent to Google News and to
    Finnhub, read off its archived pool (tickers carrying a Google article the
    fetch confirmed, or any Finnhub article). Live takes the first 150 / 60 of a
    per-tick discovery order that is not stored, so a June–September rebuild
    cannot read this and must approximate the selection."""
    from src.db import repo
    df = repo.fetch_df(
        "SELECT source, url, tickers_json FROM news_articles "
        "WHERE CAST(first_seen_at AS TIMESTAMPTZ) <= ? AND CAST(last_seen_at AS TIMESTAMPTZ) >= ? "
        "AND (source LIKE 'google_news%' OR url LIKE '%finnhub.io%')", [when, when])
    google, finnhub = set(), set()
    for r in ([] if df is None else df.itertuples(index=False)):
        try:
            tags = json.loads(r.tickers_json) if r.tickers_json else []
        except (TypeError, ValueError):
            tags = []
        (finnhub if "finnhub.io" in str(r.url) else google).update(str(t).upper() for t in tags)
    return {"google": sorted(google), "finnhub": sorted(finnhub)}


def _google_history(tickers: Sequence[str], when: datetime,
                    tick_aware: bool = True) -> Tuple[list, dict]:
    """Google News as the live tick fetched it (`news_fetcher.
    _fetch_google_news_for_ticker`: symbol, company-name and Business Wire
    queries, confirmed tags, publisher-labelled source), rebuilt from Google's
    date-bounded search and cut to live's own 24h age window at the tick.
    Undated entries are dropped (live stamps them with fetch time, which a past
    tick cannot reproduce)."""
    import feedparser
    from urllib.parse import quote_plus

    from src.data.news_fetcher import (_GOOGLE_NEWS_RSS, _confirmed_tags, _dedupe_by_url,
                                       _google_entry_source, _google_name_query,
                                       _parse_feed_date)
    from src.models import NewsArticle
    lo = when - timedelta(hours=24)
    bound = f" after:{(when - timedelta(days=2)).date().isoformat()} " \
            f"before:{(when + timedelta(days=1)).date().isoformat()}"
    out, statuses, n_req, bad, aborted = [], Counter(), 0, 0, False
    for tk in tickers:
        cached = _hist_cached("google", tk, when)
        if cached is not None:
            out.extend(cached)
            continue
        mine, complete = [], True
        queries = [f'"{tk}" stock']
        name_q = _google_name_query(tk)
        if name_q:
            queries.append(name_q)
        if settings.google_news_business_wire:
            queries.append(f'"{tk}" site:businesswire.com')
        for q in queries:
            if tick_aware:
                wait_while_tick_running("historical Google News")
            time.sleep(_HIST_GOOGLE_PACE_S)
            try:
                feed = feedparser.parse(_GOOGLE_NEWS_RSS.format(q=quote_plus(q + bound)))
            except Exception as exc:                            # noqa: BLE001
                logger.debug(f"[news-replay] google history {tk}: {exc}")
                complete = False
                continue
            n_req += 1
            status = getattr(feed, "status", None)
            if status is not None and int(status) != 200:
                statuses[int(status)] += 1
                complete = False
                bad += 1
                if bad >= _HIST_GOOGLE_MAX_BAD:
                    aborted = True
                    break
                continue
            bad = 0
            for entry in feed.entries:
                pub = _parse_feed_date(entry)
                if pub is None or not (lo <= pub <= when):
                    continue
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                summary = entry.get("summary", "") or ""
                mine.append(NewsArticle(title=title, summary=summary, url=entry.get("link", ""),
                                        source=_google_entry_source(entry), published_at=pub,
                                        tickers=_confirmed_tags(tk, title, summary)))
        out.extend(mine)
        if complete and not aborted:
            _hist_store("google", tk, when, mine)        # only a ticker every query answered
        if aborted:
            logger.warning(f"[news-replay] Google News answered non-200 {_HIST_GOOGLE_MAX_BAD}x "
                           f"in a row {dict(statuses)} — historical Google leg ABORTED "
                           f"(the live pipeline shares this machine's quota)")
            break
    return _dedupe_by_url(out), {"requests": n_req, "non200": dict(statuses), "aborted": aborted}


def _finnhub_history(tickers: Sequence[str], when: datetime, tick_aware: bool = True,
                     lookback_days: int = 3, max_per_ticker: int = 15) -> Tuple[list, dict]:
    """`provider_news.fetch_finnhub_news` as of a past tick: the same 3-day
    date window, noise filter and 15-most-recent cap, applied to what was
    published by the tick. The free tier serves one rolling year."""
    import httpx

    from src.data.provider_news import _FINNHUB_NEWS, _dedupe, _is_finnhub_noise
    from src.models import NewsArticle
    if not settings.finnhub_api_key:
        return [], {"requests": 0, "skipped": "no key"}
    frm = (when - timedelta(days=lookback_days)).date().isoformat()
    to = when.date().isoformat()
    out, n_req, limited = [], 0, False
    for tk in tickers:
        cached = _hist_cached("finnhub", tk, when)
        if cached is not None:
            out.extend(cached)
            continue
        if tick_aware:
            wait_while_tick_running("historical Finnhub")
        time.sleep(_HIST_FINNHUB_PACE_S)
        try:
            r = httpx.get(_FINNHUB_NEWS, params={"symbol": tk.upper(), "from": frm, "to": to,
                                                 "token": settings.finnhub_api_key}, timeout=15)
            n_req += 1
            if r.status_code == 429:
                limited = True
                logger.warning("[news-replay] Finnhub rate-limited — historical leg stopped")
                break
            r.raise_for_status()
            items = r.json() or []
        except Exception as exc:                                # noqa: BLE001
            logger.debug(f"[news-replay] finnhub history {tk}: {exc}")
            continue
        items = sorted(items, key=lambda x: x.get("datetime") or 0, reverse=True)
        kept, mine = 0, []
        for item in items:
            if kept >= max_per_ticker:
                break
            ts, headline = item.get("datetime"), (item.get("headline") or "").strip()
            url = (item.get("url") or "").strip()
            if not ts or not headline or not url:
                continue
            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OSError, TypeError):
                continue
            if published > when:
                continue                         # not yet published at the tick
            source = item.get("source") or "Finnhub"
            if _is_finnhub_noise(headline, source):
                continue
            mine.append(NewsArticle(title=headline, summary=(item.get("summary") or "")[:1000],
                                    url=url, source=source, published_at=published,
                                    tickers=[tk.upper()], provider_sentiment_source="finnhub"))
            kept += 1
        out.extend(mine)
        _hist_store("finnhub", tk, when, mine)
    return _dedupe(out), {"requests": n_req, "rate_limited": limited}


def build_tick_pool(when: datetime, universe: set, index=None,
                    union_hours: float = 0.0, source: str = "bundle",
                    hist_tickers: Optional[Dict[str, List[str]]] = None,
                    tick_aware: bool = False) -> Tuple[list, dict]:
    """The recoverable half of the tick's pool, with its provenance.

    ``union_hours`` = 0 is the FAITHFUL pool: exactly the bundle the tick used
    plus the Polygon call it made. Above 0 it adds articles from earlier
    bundles published within that many hours — a deliberate departure from what
    the tick held, so it is stamped into the row's `pool_spec` and can never be
    confused with the faithful one.

    ``source="archive"`` returns the run's archived merged pool instead (the
    FAITHFUL source, see the module doc); ``source="hist"`` adds the historical
    Finnhub and Google legs for ``hist_tickers`` (``{"google": [...],
    "finnhub": [...]}``) to the bundle + Polygon pool.
    """
    from src.data.news_fetcher import _dedupe_by_url
    if source == "archive":
        archived = _archive_pool(when)
        pool, n_ins = _attach_polygon_insights(archived, when, universe)
        return pool, {"bundle_file": None, "bundle_stamp": None, "n_bundle": 0,
                      "n_polygon": n_ins, "n_extra": 0, "n_archive": len(archived),
                      "n_pool": len(pool), "pool_spec": ARCHIVE_POOL_SPEC}
    index = index if index is not None else _bundle_index()
    bundle, stamp, path = _bundle_leg(index, when)
    polygon = _polygon_as_of(when, universe)
    if source == "hist":
        sel = hist_tickers or {}
        finnhub, fstat = _finnhub_history(sel.get("finnhub") or [], when, tick_aware=tick_aware)
        google, gstat = _google_history(sel.get("google") or [], when, tick_aware=tick_aware)
        # Live's merge order (bundle, polygon, finnhub, google — first URL wins)
        # and live's tags: only the bundle leg is re-derived (idempotent on
        # bundles written after the 2026-09-04 fix); Polygon keeps its own tags,
        # Finnhub its queried symbol, Google the tag its fetch confirmed.
        pool = _dedupe_by_url(_reconfirm_tags(bundle) + polygon + finnhub + google)
        return pool, {"bundle_file": (path.name if path else None),
                      "bundle_stamp": (stamp.isoformat() if stamp else None),
                      "n_bundle": len(bundle), "n_polygon": len(polygon), "n_extra": 0,
                      "n_finnhub": len(finnhub), "n_google": len(google),
                      "finnhub_fetch": fstat, "google_fetch": gstat,
                      "n_pool": len(pool), "pool_spec": HIST_POOL_SPEC}
    extra = _earlier_bundle_articles(index, when, union_hours)
    pool = _dedupe_by_url(_reconfirm_tags(bundle + polygon + extra))
    prov = {"bundle_file": (path.name if path else None),
            "bundle_stamp": (stamp.isoformat() if stamp else None),
            "n_bundle": len(bundle), "n_polygon": len(polygon), "n_extra": len(extra),
            "n_pool": len(pool),
            "pool_spec": ("faithful" if union_hours <= 0 else f"union{int(union_hours)}h")}
    return pool, prov


def _bundle_leg(index, when: datetime) -> Tuple[list, Optional[datetime], Optional[Path]]:
    """The hourly yfinance + NewsAPI bundle the tick used, point-in-time checked
    and cut to the scorer's 7-day window (tags NOT yet re-confirmed — the caller
    decides, since the merge order matters for which copy of a URL survives)."""
    stamp, path = _bundle_for(index, when)
    raw = _load_bundle(path) if path else []
    # The guard runs on the RAW load, BEFORE the recency window trims anything.
    # Trimming first would silently drop a future article and leave the check
    # unable to fire — a tripwire whose failure mode is indistinguishable from
    # normal operation. A future article in a bundle stamped before the tick
    # means the archive itself is wrong (a misnamed file, a clock problem), and
    # that must be loud, not quietly cleaned up.
    # A bundle's hour-STAMP is not its write time: the hour-H file is written BY
    # the tick that runs in hour H, during that tick's own news fetch, so an
    # article published a minute into the fetch lands in it with a timestamp
    # after the run's `generated_at`. That article WAS available to the run —
    # it postdates only the run's START, which is the conservative bound this
    # module measures against. So a few such rows are benign and are DROPPED
    # (the recency window would exclude them anyway); a large share means the
    # archive itself is wrong (a misnamed file, a clock problem) and still
    # raises, because that is not something to quietly clean up.
    # The discriminator is TIME, not count: a fetch overlap puts articles a few
    # MINUTES past the run's start, while a misnamed file or a clock problem puts
    # them hours or days out. Anything inside the grace window is dropped (the
    # recency window would drop it anyway); anything beyond it raises.
    grace = when + timedelta(minutes=_FETCH_GRACE_MINUTES)
    beyond = [a for a in raw if a.published_at and a.published_at > grace]
    if beyond:
        newest = max(a.published_at for a in beyond)
        raise RuntimeError(
            f"{len(beyond)} article(s) in {path.name if path else '?'} published up to "
            f"{newest.isoformat()}, more than {_FETCH_GRACE_MINUTES} min after the tick "
            f"instant {when.isoformat()} — the archive is not point-in-time")
    future = [a for a in raw if a.published_at and a.published_at > when]
    if future:
        logger.debug(f"[news-replay] {path.name if path else '?'}: dropped {len(future)} "
                     f"article(s) published within {_FETCH_GRACE_MINUTES} min after "
                     f"{when:%H:%M:%S} (fetched during the run itself)")
        raw = [a for a in raw if not (a.published_at and a.published_at > when)]
    floor = when - timedelta(days=_RECENCY_DAYS)
    bundle = [a for a in raw if a.published_at and floor <= a.published_at <= when]
    return bundle, stamp, path


def baselines_as_of(signal_date: str, pool_spec: Optional[str] = None) -> Dict[str, float]:
    """``{ticker: median daily attention mass}`` over the window ENDING at the
    replayed date.

    `news_shock.load_attention_baselines` hardcodes `CURRENT_DATE`, which is
    correct live and is time travel in a replay — it would hand a 2026-07 tick a
    baseline computed through today. `analysis_asof` cannot save us here either:
    it bounds the panel LOADERS, and that function issues its own SQL. So the
    replay computes its own, identical apart from the two date bounds, and a
    ticker with too few covered days is simply absent (the method abstains, the
    same way it does live before history accrues).

    BASIS (2026-09-12). `news_shock` is a RATIO — today's attention mass over the
    ticker's own normal — so both sides must come from the SAME generator. With
    `pool_spec` given the baseline is built from `news_replay`'s own stored mass
    for that pool shape; without it, from the live `signals` panel. Reading the
    LIVE series under a REPLAYED numerator is what made the first full backfill
    score `news_shock` on 1.2% of rows against 13.7% on the consistent basis: a
    replayed pool carries ~0.19x the live mass (its articles are older and only
    the yfinance/NewsAPI leg survives), so the ratio sits structurally below 1,
    and `clip(log2(ratio)/3, 0, 1)` is then exactly 0 — an abstention produced by
    the basis mismatch, indistinguishable from "attention is normal".

    Point-in-time either way: only signal dates strictly BEFORE `signal_date`
    are read, and a replayed mass for a past day was itself reconstructed from
    the pool as of that day. No fallback between the two bases on purpose — a
    baseline that silently mixes generators is the defect this parameter exists
    to remove, so a pool with too little replayed history abstains instead.
    """
    from src.db import repo
    days = max(5, int(getattr(settings, "news_shock_baseline_days", 20)))
    min_days = max(2, int(getattr(settings, "news_shock_min_days", 5)))
    day = str(signal_date)[:10]
    floor = (datetime.fromisoformat(day) - timedelta(days=days + 5)).date().isoformat()
    src = "signals" if not pool_spec else "news_replay"
    extra = "" if not pool_spec else " AND coalesce(pool_spec, 'faithful') = ?"
    params = [floor, day] + ([] if not pool_spec else [pool_spec])
    try:
        df = repo.fetch_df(f"""
            SELECT ticker, median(day_mass) AS base
            FROM (
                SELECT ticker, substr(signal_date, 1, 10) AS d,
                       max(news_recency_mass) AS day_mass
                FROM {src}
                WHERE signal_date >= ? AND signal_date < ?
                  AND news_recency_mass IS NOT NULL AND news_recency_mass > 0
                  {extra}
                GROUP BY 1, 2
            )
            GROUP BY ticker
            HAVING count(*) >= {min_days}
        """, params)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"[news-replay] baseline query failed (news_shock abstains): {exc}")
        return {}
    if df is None or df.empty:
        return {}
    return {str(t): float(b) for t, b in zip(df["ticker"], df["base"]) if b == b and b > 0}


# ── one tick ────────────────────────────────────────────────────────────────

def news_epoch_stamp() -> Optional[str]:
    """ISO instant of the news-family scorer epoch in force (`method_epochs`)."""
    try:
        from src.signals.method_epochs import METHOD_SCORER_EPOCH
        ep = METHOD_SCORER_EPOCH.get("news")
        return ep.isoformat() if ep else None
    except Exception:                                           # noqa: BLE001
        return None


def _tick_universe(run_id: str):
    """The run's own scored universe and instant — no discovery to reconstruct."""
    from src.db import repo
    df = repo.fetch_df(
        "SELECT ticker, generated_at, signal_date, price FROM signals WHERE run_id = ?",
        [run_id])
    if df is None or df.empty:
        return None
    when = datetime.fromisoformat(str(df["generated_at"].iloc[0]))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    prices = {str(t): (float(p) if p == p and p is not None else None)
              for t, p in zip(df["ticker"], df["price"])}
    return {"run_id": run_id, "tickers": sorted(prices), "when": when,
            "signal_date": str(df["signal_date"].iloc[0]), "prices": prices}


def paired_tickers(run_id: str) -> List[str]:
    """Tickers in this run that carry a LIVE local verdict to compare against.

    The fidelity question needs pairs, not rows: a replayed ticker with no live
    local verdict costs a GPU call and answers nothing. A `sentiment_shadow` row
    carries the local verdict on the real pool, which is exactly the comparison —
    engine and prompt held constant, the pool the only difference.

    ROLE-AGNOSTIC on purpose. This asked for `shadow_engine='local'`, which was
    right while DeepSeek was primary — and went silently FALSE on 2026-09-09 when
    local took 100% of the primary route and DeepSeek became the shadow. Every
    recent run then returned ZERO paired tickers, reported as "no paired
    tickers", which is indistinguishable from "nothing accrued". Exactly the
    defect `catalyst_repair`'s pair query carried (see its `--eval`): a query
    keyed on today's engine ROLES breaks the day routing changes. What the
    comparison needs is a LOCAL verdict, on whichever side it sits."""
    from src.db import repo
    df = repo.fetch_df(
        "SELECT DISTINCT ticker FROM sentiment_shadow WHERE run_id = ? AND ("
        "  (shadow_engine = 'local'  AND shadow_raw  IS NOT NULL) OR "
        "  (primary_engine = 'local' AND primary_raw IS NOT NULL))",
        [run_id])
    return [] if df is None or df.empty else sorted(str(t) for t in df["ticker"])


def archive_start() -> Optional[datetime]:
    """The first archived run instant (`news_articles` began 2026-09-11 ~16:50
    UTC). An archive re-score of an earlier run would score its stored digests
    but rebuild everything else from an empty pool, so it is refused."""
    from src.data.provider_news import _parse_iso
    from src.db import repo
    try:
        df = repo.fetch_df("SELECT min(first_seen_at) AS t0 FROM news_articles")
        v = None if df is None or df.empty else df["t0"].iloc[0]
        return _parse_iso(str(v)) if v is not None and v == v else None
    except Exception:                                           # noqa: BLE001
        return None


def _live_digests(run_id: str) -> Dict[str, tuple]:
    """``{ticker: (digest_id, digest_text, [articles])}`` for every row of the
    run whose live scorer read a stored digest — the EXACT scorer input."""
    from src.data.provider_news import _parse_iso
    from src.db import repo
    from src.models import NewsArticle
    df = repo.fetch_df(
        "SELECT s.ticker, s.news_digest_id, d.digest_text, d.articles_json FROM signals s "
        "JOIN sentiment_digests d ON d.digest_id = s.news_digest_id WHERE s.run_id = ?",
        [run_id])
    out: Dict[str, tuple] = {}
    for r in ([] if df is None else df.itertuples(index=False)):
        try:
            arts = []
            for x in json.loads(r.articles_json or "[]"):
                pub = _parse_iso(x.get("published_at"))
                if pub is not None:
                    arts.append(NewsArticle(title=x.get("title") or "", summary=x.get("summary") or "",
                                            url=x.get("url") or "", source=x.get("source") or "",
                                            published_at=pub, tickers=[str(r.ticker)]))
            if arts:
                out[str(r.ticker)] = (str(r.news_digest_id), r.digest_text or "", arts)
        except (TypeError, ValueError):
            continue
    return out


def _live_verdicts(run_id: str) -> Dict[str, tuple]:
    """``{ticker: (news, news_raw_score, news_catalyst)}`` as the live run
    persisted them — kept for every ticker whose live scorer read no digest."""
    from src.db import repo
    df = repo.fetch_df("SELECT ticker, news, news_raw_score, news_catalyst FROM signals "
                       "WHERE run_id = ?", [run_id])
    out: Dict[str, tuple] = {}
    for r in ([] if df is None else df.itertuples(index=False)):
        raw = r.news_raw_score if r.news_raw_score == r.news_raw_score else None
        cat = r.news_catalyst if isinstance(r.news_catalyst, str) and r.news_catalyst else None
        out[str(r.ticker)] = (float(r.news) if r.news == r.news and r.news is not None else 0.0,
                              raw, cat)
    return out


def replay_tick(run_id: str, engine: str = "local", limit_tickers: Optional[int] = None,
                budget_seconds: Optional[float] = None, index=None,
                paired_only: bool = False, union_hours: float = 0.0,
                source: str = "bundle", tick_aware: bool = False) -> dict:
    """Regenerate the news family for one tick. Returns a summary; rows are
    persisted to ``news_replay``. ``source`` picks the pool (see
    `build_tick_pool`); ``tick_aware`` pauses GPU and feed use while a live
    tick runs."""
    from src.analysis.asof import analysis_asof
    from src.db import repo
    info = _tick_universe(run_id)
    if not info:
        return {"run_id": run_id, "status": "no signals rows"}
    tickers = info["tickers"]
    if paired_only:
        keep = set(paired_tickers(run_id))
        tickers = [t for t in tickers if t in keep]
        if not tickers:
            return {"run_id": run_id, "status": "no paired tickers"}
    tickers = tickers[:limit_tickers] if limit_tickers else tickers
    if source == "archive":
        t0 = archive_start()
        if t0 is None or info["when"] < t0:
            return {"run_id": run_id, "status": "before the news archive began",
                    "archive_start": t0.isoformat() if t0 else None}
    hist_tickers = live_query_sets(info["when"]) if source == "hist" else None
    if source == "archive":
        info["live_digests"] = _live_digests(run_id)
        info["live_verdicts"] = _live_verdicts(run_id)
    pool, prov = build_tick_pool(info["when"], set(tickers), index=index,
                                 union_hours=union_hours, source=source,
                                 hist_tickers=hist_tickers, tick_aware=tick_aware)
    logger.info(f"[news-replay] {run_id} @ {info['when']:%Y-%m-%d %H:%M}Z — {prov['pool_spec']} pool "
                f"{prov['n_pool']} ({prov.get('n_archive', prov['n_bundle'])} "
                f"{'archived' if source == 'archive' else 'bundle'}, {prov['n_polygon']} polygon"
                + (f", {prov['n_finnhub']} finnhub, {prov['n_google']} google" if source == "hist" else "")
                + f"), {len(tickers)} tickers")
    # STORY-CLUSTERING CORPUS (2026-09-12). `news_cluster_mode` is "hybrid", and
    # the content-aware modes merge on IDF weights taken from the TICK'S WHOLE
    # POOL. `cluster_articles` degrades to the bare TIME partition when no corpus
    # is installed (`corpus_size() == 0`) — a SAFE fallback live, but a SILENT
    # infidelity here: the backfill would be labelled "today's code" while
    # clustering under the rule today's code replaced.
    #
    # Installed ONCE PER TICK from `pool`, mirroring `build_signals`, and never
    # per ticker from its own digest: within-digest IDF zeroes exactly the terms
    # that identify a shared story, which is the defect pool IDF exists to fix.
    try:
        from src.analysis.news_clustering import cluster_mode, corpus_size, set_corpus
        if cluster_mode() in ("content", "hybrid"):
            set_corpus(pool)
            logger.info(f"[news-replay] clustering corpus: {corpus_size()} documents")
    except Exception as _e:                                     # noqa: BLE001
        logger.warning(f"[news-replay] clustering corpus failed ({_e}) — "
                       f"hybrid merges degrade to the TIME partition")

    # The baseline comes from the SAME pool shape as this tick's mass — see
    # `baselines_as_of`. A tick replayed before its own trailing window has been
    # replayed simply finds too few covered days and `news_shock` abstains, so
    # the repair pass (`--repair-shock`) recomputes the column once the whole
    # window exists, where the coverage is maximal and order-independent.
    # The archived pool IS the live generator, so its baseline is the live
    # panel's own mass series; every reconstruction divides by its own.
    baselines = baselines_as_of(info["signal_date"],
                                None if source == "archive" else prov["pool_spec"])
    logger.info(f"[news-replay] attention baselines as of {info['signal_date']} "
                f"({prov['pool_spec']} basis): {len(baselines)} ticker(s)")
    rows, t0 = [], time.perf_counter()
    stopped = None
    # Point-in-time for every panel-fitted layer underneath (catalyst_tilt's map
    # above all): inside this block they can only see strictly-earlier rows.
    with analysis_asof(info["signal_date"]):
        for i, tk in enumerate(tickers, 1):
            if budget_seconds and (time.perf_counter() - t0) > budget_seconds:
                stopped = "budget"
                break
            if tick_aware:
                wait_while_tick_running("scoring")
            try:
                rows.append(_replay_one(tk, pool, info, engine, prov, baselines))
            except Exception as exc:                            # noqa: BLE001
                logger.debug(f"[news-replay] {tk}: {exc}")
            if i % 50 == 0:
                logger.info(f"[news-replay] {run_id}: {i}/{len(tickers)}")
    n_failed = sum(1 for r in rows if r.get("scorer_failed"))
    if n_failed:
        # A failed call reads 0.0, exactly like an abstention: storing the run
        # would restore false "no view" rows. Nothing is written, so the run
        # stays undone and a re-run scores it again.
        logger.error(f"[news-replay] {run_id}: {n_failed} scorer failure(s) — run NOT stored")
        return {"run_id": run_id, "status": "scorer failures", "failed": n_failed,
                "rows": len(rows), "stopped": stopped, **prov}
    if rows:
        repo.insert_news_replay(rows)
    return {"run_id": run_id, "signal_date": info["signal_date"],
            "when": info["when"].isoformat(), "tickers": len(tickers),
            "rows": len(rows), "elapsed_s": round(time.perf_counter() - t0, 1),
            "stopped": stopped, **prov}


def _replay_one(ticker: str, pool: list, info: dict, engine: str, prov: dict,
                baselines: Dict[str, float]) -> dict:
    """The CURRENT pipeline's news feature block for one ticker."""
    from src.analysis.replay import visible_history
    from src.analysis.sentiment import (analyse_sentiment, attention_mass,
                                        digest_articles, filter_relevant_articles)
    from src.data.cache import load_ohlcv
    from src.signals.catalyst_tilt import calibrate_catalyst_tilt, compute_catalyst_tilt_score
    from src.signals.news_bear_fresh import compute_news_bear_fresh
    from src.signals.news_bull_fresh import compute_news_bull_fresh
    from src.signals.news_quiet import compute_news_quiet
    from src.signals.news_priced_in import compute_news_priced_in
    from src.signals.news_shock import compute_news_shock
    from src.signals.sentiment_velocity import compute_sentiment_velocity
    arts = filter_relevant_articles(ticker, pool)
    row = {"run_id": info["run_id"],
           "ticker": ticker, "signal_date": info["signal_date"],
           "generated_at": info["when"].isoformat(),
           "replayed_at": datetime.now(timezone.utc).isoformat(),
           "replay_version": REPLAY_VERSION, "engine": engine,
           # n_articles is the digest the model READ (post 7-day cut, post
           # top-20), which is what the live shadow row records — the two must
           # mean the same thing to be comparable. n_relevant keeps the pre-cut
           # count, which is a property of the POOL, not of the digest.
           "n_articles": len(digest_articles(arts, info["when"])),
           "n_relevant": len(arts), "n_pool": prov["n_pool"],
           "bundle_file": prov["bundle_file"], "pool_spec": prov["pool_spec"],
           # The news-family scorer epoch these values were produced under:
           # `restore_news_rescored` takes a row only while it is still the
           # epoch in force, so a later scorer change retires it automatically.
           "news_epoch": news_epoch_stamp()}
    # The EXACT input, where there is one: an archive re-score scores the digest
    # the live scorer READ (`sentiment_digests`, stored from 2026-09-08) with its
    # original text — the pool rebuild reproduces only ~45% of digests exactly
    # (tags and titles are archived as first seen, and the archive cannot say
    # whether an article sat in a given run's pool). The rebuilt pool still
    # feeds the derived family's inputs and every ticker live did not score.
    live_d = (info.get("live_digests") or {}).get(ticker)
    # A ticker the live scorer did NOT read a digest for — it abstained, found
    # nothing, or took the provider shortcut — keeps LIVE's verdict: the pool
    # rebuild cannot say whether the run's true pool held a digest for it, and
    # scoring one anyway invented views live never had (59 of 61 extras on the
    # 2026-09-23 pilot; keeping live's verdict took the re-score's views that
    # match live from 84.6% to 94.6%). The derived family is still recomputed.
    kept = None
    if info.get("live_verdicts") is not None and live_d is None:
        kept = info["live_verdicts"].get(ticker, (0.0, None, None))
    score_arts = live_d[2] if live_d else arts
    if not arts and not live_d and not (kept and kept[0]):
        return {**row, **{c: 0.0 for c in NEWS_REPLAY_COLUMNS}}
    if not arts:
        arts = score_arts                 # the digest is the best evidence left
    # ``as_of`` is the tick instant: the scorer measures article age against it
    # instead of against today. Without it every historical article falls outside
    # the 7-day window, the digest is discarded whole and the scorer abstains —
    # which is exactly what the 2026-09-07 pre-flight found, silently, on every
    # tick older than a week.
    when = info["when"]
    # `allow_provider=True`: the engine is forced here to PIN which model answers,
    # not to re-judge, so the provider shortcut must fire wherever live's would
    # have (~32% of scored tickers). `store_digest=False`: `sentiment_digests` is
    # the record of what the LIVE scorer saw, and `news_replay` keeps no
    # `digest_id`, so replayed digests would land orphaned.
    # A stored digest exists only where the provider shortcut did NOT fire, so
    # that path is closed for it (`allow_provider=False`).
    failed = False
    if kept is not None:
        news, meta = float(kept[0] or 0.0), {"raw_score": kept[1], "catalyst": kept[2]}
    else:
        score, _rationale, meta = analyse_sentiment(
            ticker, score_arts, force_engine=engine, as_of=when,
            allow_provider=live_d is None, store_digest=False,
            digest_override=(live_d[0], live_d[1]) if live_d else None)
        news = float(score or 0.0)
        # An engine failure returns 0.0 — the same value as an abstention — so
        # it is flagged for the caller, which must not store it as a verdict.
        failed = isinstance(_rationale, str) and _rationale.startswith("Analysis error")
    n_art, mass = attention_mass(arts, when)
    velocity, _r, _p, _n = compute_sentiment_velocity(
        ticker, arts, recent_hours=settings.sentiment_velocity_recent_hours,
        prior_hours=settings.sentiment_velocity_prior_hours, as_of=when)
    # The OHLCV the tick actually had: bars strictly before the signal date
    # unless the tick ran at/after the 16:00 ET close (`replay.visible_history`,
    # the same rule `market_data._drop_forming_bar` applies live). Passing the
    # full cached frame would let a July tick read September bars — the exact
    # time travel this module exists to avoid.
    df = visible_history(load_ohlcv(ticker), info["signal_date"], info["when"])
    shock = compute_news_shock(news, mass, baselines.get(ticker)) \
        if settings.enable_news_shock else 0.0
    bear, _z3 = compute_news_bear_fresh(ticker, news, df=df) \
        if settings.enable_news_bear_fresh else (0.0, 0.0)
    tilt = compute_catalyst_tilt_score(news, (meta or {}).get("catalyst"),
                                       calibrate_catalyst_tilt()) \
        if settings.enable_catalyst_tilt else 0.0
    # `news_quiet` reads the RAW verdict (the scaled one shrinks exactly the thin
    # quiet digests the method exists to express) and pins mode="time" at its own
    # call site, so it is unaffected by the corpus question below.
    quiet, _age_h = compute_news_quiet(ticker, (meta or {}).get("raw_score"), arts,
                                       as_of=when) \
        if settings.enable_news_quiet else (0.0, None)
    bull, _bz3 = compute_news_bull_fresh(ticker, news, df=df) \
        if settings.enable_news_bull_fresh else (0.0, 0.0)
    unpriced, unpriced_all, _diag = compute_news_priced_in(
        ticker, news, arts, price_now=info["prices"].get(ticker), df=df, as_of=when) \
        if settings.enable_news_priced_in else (0.0, 0.0, {})
    return {**row,
            "news": news, "news_raw_score": (meta or {}).get("raw_score"),
            "sent_velocity": velocity, "news_shock": shock, "news_bear_fresh": bear,
            "catalyst_tilt": tilt, "news_unpriced": unpriced,
            "news_unpriced_all": unpriced_all,
            "news_quiet": quiet, "news_bull_fresh": bull,
            "news_catalyst": (meta or {}).get("catalyst"),
            "news_recency_mass": mass, "news_article_count": n_art,
            # the digest the scorer READ — equal to the live row's
            # `signals.news_digest_id` exactly when the pool was reproduced
            "news_digest_id": (meta or {}).get("digest_id"),
            # not a stored column (`insert_news_replay` writes only its own)
            "scorer_failed": failed}


# ── tick selection ──────────────────────────────────────────────────────────

def target_runs(days: Optional[int] = None, run_id: Optional[str] = None,
                all_ticks: bool = False) -> List[str]:
    """The runs to replay: by default the LAST tick of each day.

    `build_panel(dedupe="last")` keeps exactly one row per (ticker, signal_date)
    — the last run's — and that is the row every training set consumes, so
    replaying earlier ticks of the same day costs GPU hours no downstream
    surface would read. `--all-ticks` overrides for a within-day study.
    """
    from src.db import repo
    if run_id:
        return [run_id]
    where, params = "", []
    if days:
        from datetime import date
        where = "WHERE signal_date >= ?"
        params = [(date.today() - timedelta(days=int(days))).isoformat()]
    if all_ticks:
        df = repo.fetch_df(
            f"SELECT DISTINCT run_id FROM signals {where} ORDER BY run_id", params)
    else:
        df = repo.fetch_df(
            "SELECT run_id FROM (SELECT run_id, signal_date, "
            "       row_number() OVER (PARTITION BY signal_date ORDER BY generated_at DESC) rn "
            f"      FROM (SELECT DISTINCT run_id, signal_date, max(generated_at) generated_at "
            f"            FROM signals {where} GROUP BY 1, 2)) WHERE rn = 1 "
            "ORDER BY signal_date", params)
    return [] if df is None or df.empty else [str(r) for r in df["run_id"]]


def already_done(run_id: str, pool_spec: str = "faithful") -> int:
    from src.db import repo
    df = repo.fetch_df("SELECT count(*) AS n FROM news_replay WHERE run_id = ? "
                       "AND coalesce(pool_spec, 'faithful') = ?", [run_id, pool_spec])
    return 0 if df is None or df.empty else int(df["n"].iloc[0])


# ── fidelity ────────────────────────────────────────────────────────────────

def fidelity(days: int = 30, runs: Optional[List[str]] = None,
             pool_spec: Optional[str] = None) -> dict:
    """Replayed vs LIVE verdict on the rows where both exist.

    The only comparable live rows are v6-era LOCAL verdicts (the shadow pass and
    the local-primary run), since engine and prompt have to be held constant for
    the difference to mean "the pool", which is the one thing the replay cannot
    reproduce.
    """
    import pandas as pd

    from src.analysis.signal_panel import _spearman
    from src.db import repo
    where, params = ["news_raw_score IS NOT NULL"], []
    if runs:
        where.append("run_id IN (" + ", ".join("?" * len(runs)) + ")")
        params += list(runs)
    if pool_spec:
        where.append("coalesce(pool_spec, 'faithful') = ?")
        params.append(pool_spec)
    rep = repo.fetch_df(
        "SELECT run_id, ticker, pool_spec, news_raw_score AS replay_raw, "
        "       news AS replay_scaled, n_articles "
        f"FROM news_replay WHERE {' AND '.join(where)}", params)
    if rep is None or rep.empty:
        return {"status": "no replayed rows yet"}
    live = repo.fetch_df(
        # Same role-agnostic rule as `paired_tickers`: take the LOCAL side
        # whichever it is, or this reads zero rows for everything after
        # 2026-09-09 (when local became primary).
        "SELECT run_id, ticker, "
        "       CASE WHEN shadow_engine = 'local' THEN shadow_raw   ELSE primary_raw   END AS live_raw, "
        "       CASE WHEN shadow_engine = 'local' THEN shadow_score ELSE primary_score END AS live_scaled, "
        "       n_articles AS live_n "
        "FROM sentiment_shadow WHERE "
        "  (shadow_engine = 'local'  AND shadow_raw  IS NOT NULL) OR "
        "  (primary_engine = 'local' AND primary_raw IS NOT NULL)")
    m = rep.merge(live, on=["run_id", "ticker"], how="inner") if live is not None else None
    if m is None or m.empty:
        return {"status": "no overlapping live LOCAL verdicts", "n_replayed": int(len(rep))}
    both = m[(m.replay_raw != 0) & (m.live_raw != 0)]
    return {
        "pool_spec": (pool_spec or "all"),
        "runs": (list(runs) if runs else "all"),
        "n_pairs": int(len(m)),
        "abstain_live": float((m.live_raw == 0).mean()),
        "abstain_replay": float((m.replay_raw == 0).mean()),
        "digest_live": float(pd.to_numeric(m.live_n, errors="coerce").mean()),
        "digest_replay": float(pd.to_numeric(m.n_articles, errors="coerce").mean()),
        "spearman": float(_spearman(m.replay_raw, m.live_raw) or float("nan")),
        "spearman_both_nonzero": (float(_spearman(both.replay_raw, both.live_raw) or float("nan"))
                                  if len(both) > 2 else float("nan")),
        "sign_agree": (float(((both.replay_raw > 0) == (both.live_raw > 0)).mean())
                       if len(both) else float("nan")),
        # THE feature the stacker actually consumes is the SCALED score — the raw
        # verdict passes through the evidence-mass x diversity scaler, which is
        # precisely where a reconstruction's digest-size difference lands. A
        # reconstruction can match the raw verdict and still hand the model a
        # systematically larger or smaller number.
        "spearman_scaled": float(_spearman(m.replay_scaled, m.live_scaled) or float("nan")),
        "scaled_ratio": (float(pd.to_numeric(m.replay_scaled, errors="coerce").abs().mean()
                               / max(1e-9, pd.to_numeric(m.live_scaled, errors="coerce").abs().mean()))),
    }


def rebuilt_digest_id(ticker: str, pool: list, when: datetime) -> Optional[str]:
    """The `digest_id` the live scorer would record for ``ticker`` given
    ``pool``: `analyse_sentiment`'s pre-LLM selection step for step — relevance,
    7-day freshness, passing-mention abstention, provider shortcut, source tier,
    top-20 cut (`test_news_rescore` pins the order against the scorer). None
    where it would not score a digest at all."""
    from src.analysis.sentiment import (_passing_mention_share, _provider_sentiment_score,
                                        _recency_weight, apply_source_tier, digest_articles,
                                        digest_id_for, filter_relevant_articles)
    arts = filter_relevant_articles(ticker, pool)
    fresh = [a for a in arts if _recency_weight(a, when) > 0.0]
    if not fresh:
        return None
    if bool(getattr(settings, "enable_passing_mention_abstention", True)):
        share = _passing_mention_share(ticker, fresh)
        floor = float(getattr(settings, "passing_mention_abstain_share", 0.65) or 0.65)
        if share is not None and share >= floor:
            return None
    if _provider_sentiment_score(ticker, fresh, when) is not None:
        return None
    to_score = digest_articles(apply_source_tier(fresh), when)
    return digest_id_for(ticker, to_score) if to_score else None


def certify_archive(run_id: str) -> dict:
    """LLM-FREE certificate for one archived run: rebuild every ticker's digest
    from the archive and compare its id with the live `signals.news_digest_id`.

    ``match`` = live digests reproduced exactly (the same article set);
    ``mismatch`` = the live scorer read a digest and the rebuild read a
    different one (or none); ``extra`` = the rebuild would score a digest where
    live scored none. Costs one pool build and no model call, so it can certify
    every archived run before a single GPU second is spent."""
    from src.db import repo
    info = _tick_universe(run_id)
    if not info:
        return {"run_id": run_id, "status": "no signals rows"}
    pool, prov = build_tick_pool(info["when"], set(info["tickers"]), source="archive")
    live = repo.fetch_df("SELECT ticker, news_digest_id FROM signals WHERE run_id = ?", [run_id])
    live_ids = {str(t): str(d) for t, d in zip(live["ticker"], live["news_digest_id"])
                if d is not None and d == d and str(d)}
    match = mismatch = extra = 0
    examples = []
    for tk in info["tickers"]:
        rid, lid = rebuilt_digest_id(tk, pool, info["when"]), live_ids.get(tk)
        if lid:
            if rid == lid:
                match += 1
            else:
                mismatch += 1
                if len(examples) < 5:
                    examples.append(tk)
        elif rid:
            extra += 1
    n_live = match + mismatch
    stored = _live_digests(run_id)
    exact = sum(1 for tk, lid in live_ids.items() if tk in stored and stored[tk][0] == lid)
    return {"run_id": run_id, "when": info["when"].isoformat(), "n_archive": prov["n_archive"],
            "n_pool": prov["n_pool"], "insights_attached": prov["n_polygon"],
            "live_digests": n_live,
            # the re-score's input for a live-scored ticker is its STORED digest:
            # this is the share it can score exactly
            "exact_input": exact, "exact_input_rate": (exact / n_live) if n_live else float("nan"),
            # the pool rebuild on its own (what non-digest tickers and the derived
            # family's inputs rest on)
            "pool_match": match, "pool_mismatch": mismatch, "pool_extra": extra,
            "pool_match_rate": (match / n_live) if n_live else float("nan"),
            "mismatch_examples": examples}


def compare_to_live(pool_spec: str, runs: Optional[List[str]] = None) -> dict:
    """Replayed vs LIVE on the same (run, ticker), read from `signals` itself.

    ``digest_match`` is the certificate: the share of live digests (rows where
    the live scorer recorded a `news_digest_id`) whose re-scored digest is the
    SAME article set. On runs after the news epoch, live IS the current code, so
    the per-column block measures how faithfully a re-score reproduces it —
    ``exact`` (|diff| <= 1e-6), ``within_0.02``, ``zero_agree`` (both zero or
    both not) and the rank correlation. On a reconstruction it measures the
    whole gap.
    """
    import numpy as np
    import pandas as pd

    from src.analysis.signal_panel import _spearman
    from src.db import repo
    cols = list(NEWS_REPLAY_COLUMNS) + ["news_recency_mass", "news_article_count"]
    where, params = ["coalesce(pool_spec, 'faithful') = ?"], [pool_spec]
    if runs:
        where.append("run_id IN (" + ", ".join("?" * len(runs)) + ")")
        params += list(runs)
    rep = repo.fetch_df(
        f"SELECT run_id, ticker, news_digest_id AS r_dig, n_articles AS r_n, {', '.join(cols)} "
        f"FROM news_replay WHERE {' AND '.join(where)}", params)
    if rep is None or rep.empty:
        return {"status": "no replayed rows", "pool_spec": pool_spec}
    rids = sorted(rep["run_id"].astype(str).unique())
    live = repo.fetch_df(
        f"SELECT run_id, ticker, news_digest_id AS l_dig, {', '.join(cols)} FROM signals "
        f"WHERE run_id IN ({', '.join('?' * len(rids))})", rids)
    m = rep.merge(live, on=["run_id", "ticker"], how="inner", suffixes=("_r", "_l"))
    if m.empty:
        return {"status": "no overlapping live rows", "pool_spec": pool_spec}
    has_dig = m["l_dig"].notna() & (m["l_dig"].astype(str) != "")
    out = {"pool_spec": pool_spec, "runs": len(rids), "rows": int(len(m)),
           "live_digests": int(has_dig.sum()),
           "digest_match": (float((m.loc[has_dig, "r_dig"].astype(str)
                                   == m.loc[has_dig, "l_dig"].astype(str)).mean())
                            if has_dig.any() else float("nan")),
           "digest_articles_replay": float(pd.to_numeric(m["r_n"], errors="coerce").astype(float).mean())}
    per_col = {}
    for c in cols:
        r = pd.to_numeric(m[f"{c}_r"], errors="coerce").astype(float)
        l = pd.to_numeric(m[f"{c}_l"], errors="coerce").astype(float)
        rz, lz = r.fillna(0.0), l.fillna(0.0)
        either = (rz != 0) | (lz != 0)
        both = r.notna() & l.notna()
        per_col[c] = {
            "nonzero_live": float((lz != 0).mean()), "nonzero_replay": float((rz != 0).mean()),
            "zero_agree": float(((rz != 0) == (lz != 0)).mean()),
            "exact": float((np.abs(rz - lz) <= 1e-6).mean()),
            "within_0.02": float((np.abs(rz - lz) <= 0.02).mean()),
            "spearman": (float(_spearman(rz[either], lz[either]) or float("nan"))
                         if either.sum() > 2 else float("nan")),
            "sign_agree": (float(((rz[both & (rz != 0) & (lz != 0)] > 0)
                                  == (lz[both & (rz != 0) & (lz != 0)] > 0)).mean())
                           if (both & (rz != 0) & (lz != 0)).any() else float("nan")),
            "magnitude_ratio": (float(rz[either].abs().mean() / max(1e-12, lz[either].abs().mean()))
                                if either.any() else float("nan")),
        }
    out["columns"] = per_col
    # the live-scored subset: where the re-score read the EXACT stored digest
    sub = m[has_dig]
    if len(sub):
        rz = pd.to_numeric(sub["news_raw_score_r"], errors="coerce").astype(float).fillna(0.0)
        lz = pd.to_numeric(sub["news_raw_score_l"], errors="coerce").astype(float).fillna(0.0)
        out["raw_on_live_digests"] = {
            "rows": int(len(sub)), "exact": float((np.abs(rz - lz) <= 1e-6).mean()),
            "within_0.02": float((np.abs(rz - lz) <= 0.02).mean()),
            "spearman": float(_spearman(rz, lz) or float("nan")),
            "sign_agree": float(((rz > 0) == (lz > 0))[(rz != 0) & (lz != 0)].mean())
            if ((rz != 0) & (lz != 0)).any() else float("nan")}
    return out


# The columns an archive re-score restores into the panel: the regenerated news
# family plus the news-derived inputs the stacker masks itself.
RESTORE_COLUMNS = NEWS_REPLAY_COLUMNS + ("news_catalyst", "news_recency_mass",
                                         "news_article_count")


def restore_news_rescored(df):
    """Put archive re-scores back into a panel frame for the rows the news
    epoch would otherwise blank — the news family's counterpart of the OHLCV
    replay restore, and what stops a scorer change from resetting the models'
    news history.

    Takes only `RESTORABLE_POOL_SPECS` rows (the exact archived pool) scored
    under the news epoch IN FORCE (`news_epoch` = today's registry value), joined
    run-exact on (signal_date, ticker, generated_at), and only onto rows dated
    before that epoch's first full day — post-epoch rows already carry the live
    current-code value. Returns ``(df, {column: bool Series})``; the caller must
    exempt those cells from the epoch mask. Adds ``news_rescored`` so the
    stacker's own news mask can exempt them too. Fail-soft: any problem returns
    the frame untouched, and the mask then runs exactly as before.
    """
    import pandas as pd

    from src.db import repo
    from src.signals.method_epochs import epoch_for
    stamp, day = news_epoch_stamp(), epoch_for("news")
    if df is None or df.empty or stamp is None or day is None:
        return df, {}
    try:
        cols = [c for c in RESTORE_COLUMNS]
        specs = ", ".join("?" * len(RESTORABLE_POOL_SPECS))
        rep = repo.fetch_df(
            "SELECT signal_date, ticker, generated_at, replayed_at, "
            f"{', '.join(cols)} FROM news_replay "
            f"WHERE pool_spec IN ({specs}) AND news_epoch = ?",
            list(RESTORABLE_POOL_SPECS) + [stamp])
        if rep is None or rep.empty or "generated_at" not in df.columns:
            return df, {}
        key = ["signal_date", "ticker", "generated_at"]
        for k in key:
            rep[k] = rep[k].astype(str)
        rep = (rep.sort_values("replayed_at", kind="stable")
                  .drop_duplicates(subset=key, keep="last")
                  .drop(columns=["replayed_at"]))
        left = df[key].astype(str)
        merged = left.merge(rep, on=key, how="left")
        if len(merged) != len(left):
            raise AssertionError("news re-score join fanned out")
        merged.index = df.index
        pre = df["signal_date"].astype(str) < day.isoformat()
        restored, any_taken = {}, pd.Series(False, index=df.index)
        for c in cols:
            if c not in merged.columns:
                continue
            take = merged[c].notna() & pre
            if take.any():
                if c not in df.columns:
                    df[c] = None if c == "news_catalyst" else float("nan")
                df.loc[take, c] = merged.loc[take, c].values
                restored[c] = take
                any_taken |= take
        if any_taken.any():
            df["news_rescored"] = any_taken
            logger.info(f"[news-replay] restored {int(any_taken.sum())} pre-epoch panel row(s) "
                        f"from archive re-scores (epoch {stamp})")
        return df, restored
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"[news-replay] re-score restore unavailable: {exc}")
        return df, {}


def repair_shock(pool_spec: str = "union168h", since: Optional[str] = None,
                 apply: bool = False) -> dict:
    """Recompute `news_shock` for already-replayed rows on the POOL-CONSISTENT
    baseline, and (with ``apply``) write it back.

    Why a repair pass exists at all: `news_shock` is the one news feature that
    is a RATIO of today's evidence mass to the ticker's own normal, so it needs a
    trailing window of the SAME generator's mass. Two things make that awkward
    during a replay — the baseline a tick reads depends on which OTHER ticks have
    been replayed already (so a resumable run is order-dependent), and the first
    full backfill read the LIVE `signals` mass under a REPLAYED numerator, which
    scores 0 for a reason that has nothing to do with attention (see
    `baselines_as_of`). Recomputing after the fact fixes both: the whole window
    exists, so coverage is maximal and the result is order-independent.

    Costs NOTHING but a query — no LLM call, no re-scoring. The inputs are the
    `news` verdict and the `news_recency_mass` already stored on the row. Never
    touches `signals`: the panel records what actually happened.
    """
    import pandas as pd

    from src.db import repo
    days = max(5, int(getattr(settings, "news_shock_baseline_days", 20)))
    min_days = max(2, int(getattr(settings, "news_shock_min_days", 5)))
    where = ["coalesce(pool_spec, 'faithful') = ?"]
    params: list = [pool_spec]
    if since:
        where.append("replayed_at > ?")
        params.append(since)
    df = repo.fetch_df(
        "SELECT run_id, ticker, substr(signal_date, 1, 10) AS d, news, news_shock, "
        "       news_recency_mass AS mass "
        f"FROM news_replay WHERE {' AND '.join(where)}", params)
    if df is None or df.empty:
        return {"status": "no rows", "pool_spec": pool_spec}
    from src.signals.news_shock import compute_news_shock
    mass = pd.to_numeric(df["mass"], errors="coerce")
    news = pd.to_numeric(df["news"], errors="coerce")
    # One mass per (ticker, day): the MAX across that day's replayed ticks, the
    # same reduction the live baseline query applies across a day's runs.
    series = (df.assign(m=mass).loc[mass > 0]
              .groupby(["ticker", "d"], as_index=False)["m"].max()
              .sort_values(["ticker", "d"]))
    by_ticker = {t: g[["d", "m"]].reset_index(drop=True) for t, g in series.groupby("ticker")}
    updates, changed, nonzero_new = [], 0, 0
    for i, row in enumerate(df.itertuples()):
        m_today = float(mass.iat[i]) if mass.iat[i] == mass.iat[i] else 0.0
        base = None
        g = by_ticker.get(row.ticker)
        if g is not None:
            floor = (datetime.fromisoformat(row.d) - timedelta(days=days + 5)).date().isoformat()
            # STRICTLY earlier days only — the same point-in-time rule
            # `baselines_as_of` applies, restated here because this pass sees
            # the whole window at once and could otherwise read forward.
            w = g[(g["d"] >= floor) & (g["d"] < row.d)]
            if len(w) >= min_days:
                base = float(w["m"].median())
        n_today = float(news.iat[i]) if news.iat[i] == news.iat[i] else 0.0
        shock = compute_news_shock(n_today, m_today, base)
        if shock:
            nonzero_new += 1
        was = float(row.news_shock or 0.0)
        if abs(shock - was) > 1e-9:
            changed += 1
            updates.append((shock, row.run_id, row.ticker, pool_spec))
    if apply and updates:
        from src.db.connection import connect
        with connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            conn.executemany(
                "UPDATE news_replay SET news_shock = ? WHERE run_id = ? AND ticker = ? "
                "AND coalesce(pool_spec, 'faithful') = ?", updates)
            conn.execute("COMMIT")
    n = max(1, len(df))
    before = pd.to_numeric(df["news_shock"], errors="coerce").fillna(0)
    return {"pool_spec": pool_spec, "since": since, "rows": int(len(df)),
            "nonzero_before": float((before != 0).mean()),
            "nonzero_after": nonzero_new / n,
            "changed": changed, "applied": bool(apply and updates)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-tick news replay from the archived pool")
    ap.add_argument("--days", type=int, default=None, help="replay the last N days of ticks")
    ap.add_argument("--run-id", default=None, help="replay exactly one tick")
    ap.add_argument("--all-ticks", action="store_true",
                    help="every tick, not just the last of each day (the panel keeps the last)")
    ap.add_argument("--engine", default="local")
    ap.add_argument("--limit-tickers", type=int, default=None)
    ap.add_argument("--budget-seconds", type=float, default=None, help="per tick")
    ap.add_argument("--union-hours", type=float, default=0.0,
                    help="0 = the FAITHFUL pool (what the tick held). Above 0, add articles "
                         "from earlier bundles published within N hours, as a stand-in for the "
                         "unrecoverable RSS/Google half (which was itself age-capped at 24h)")
    ap.add_argument("--pool-spec", default=None, help="--fidelity: restrict to one pool shape")
    ap.add_argument("--only-runs", default="", help="comma list of run_ids (tune/test split)")
    ap.add_argument("--paired-only", action="store_true",
                    help="replay only the tickers with a live LOCAL verdict for that run — "
                         "every call then yields a fidelity pair")
    ap.add_argument("--redo", action="store_true", help="replay ticks that already have rows")
    ap.add_argument("--fidelity", action="store_true", help="report replayed vs live and exit")
    ap.add_argument("--repair-shock", action="store_true",
                    help="recompute news_shock for already-replayed rows on the pool-consistent "
                         "baseline (dry run unless --apply)")
    ap.add_argument("--since", default=None,
                    help="--repair-shock: only rows with replayed_at > this ISO stamp")
    ap.add_argument("--apply", action="store_true", help="--repair-shock: write the values back")
    ap.add_argument("--source", default="bundle", choices=["bundle", "archive", "hist"],
                    help="archive = the FAITHFUL archived pool (restorable); hist = bundle + "
                         "Polygon + historical Finnhub and Google (research only)")
    ap.add_argument("--tick-aware", action="store_true",
                    help="pause GPU and feed use while a live scheduler tick runs")
    ap.add_argument("--compare", default=None, metavar="POOL_SPEC",
                    help="report replayed vs live (digest certificate, verdict agreement) and exit")
    ap.add_argument("--certify", action="store_true",
                    help="LLM-free: rebuild each selected run's digests from the archive and "
                         "match their ids against the live ones")
    args = ap.parse_args(argv)
    _isolate_sentiment_cache()

    only = [r.strip() for r in args.only_runs.split(",") if r.strip()]
    if args.compare or args.certify or args.fidelity:
        # read-only reports: never take the write lock the live scheduler needs
        from src.db import repo
        repo.set_read_only(True)
    if args.compare:
        print(json.dumps(compare_to_live(args.compare, runs=only or None), indent=2, default=str))
        return 0
    if args.certify:
        for rid in (only or target_runs(days=args.days, run_id=args.run_id,
                                        all_ticks=args.all_ticks)):
            print(json.dumps(certify_archive(rid), default=str))
        return 0
    if args.repair_shock:
        print(json.dumps(repair_shock(pool_spec=(args.pool_spec or "union168h"),
                                      since=args.since, apply=args.apply), indent=2))
        return 0
    if args.fidelity:
        print(json.dumps(fidelity(runs=only or None, pool_spec=args.pool_spec), indent=2))
        return 0
    runs = only or target_runs(days=args.days, run_id=args.run_id, all_ticks=args.all_ticks)
    if not runs:
        print("no ticks selected")
        return 1
    index = _bundle_index()
    print(f"{len(runs)} tick(s) to replay; {len(index)} bundles on disk")
    for run_id in runs:
        spec = {"archive": ARCHIVE_POOL_SPEC, "hist": HIST_POOL_SPEC}.get(
            args.source, "faithful" if args.union_hours <= 0 else f"union{int(args.union_hours)}h")
        if not args.redo and already_done(run_id, spec):
            logger.info(f"[news-replay] {run_id}: already replayed, skipping")
            continue
        out = replay_tick(run_id, engine=args.engine, limit_tickers=args.limit_tickers,
                          budget_seconds=args.budget_seconds, index=index,
                          paired_only=args.paired_only, union_hours=args.union_hours,
                          source=args.source, tick_aware=args.tick_aware)
        print(json.dumps(out, indent=2, default=str))
    return 0


def _isolate_sentiment_cache() -> None:
    """A replay process must never share the LIVE verdict cache file: both
    processes read-modify-write it on their own debounce, so each would drop the
    other's entries (live then re-pays those calls), and a replay needs real
    re-scores anyway. Called by the CLI only — tests isolate the cache their
    own way."""
    from src.analysis import sentiment
    from src.data.cache import CACHE_DIR
    sentiment._sent_cache_path = lambda: CACHE_DIR / "sentiment_llm_replay.json"
    sentiment._reset_sentiment_cache_for_tests()


if __name__ == "__main__":
    raise SystemExit(main())
