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

CLI
    python -m src.analysis.news_replay --days 7 [--limit-tickers N] [--budget-seconds S]
    python -m src.analysis.news_replay --run-id 2026-09-04_235002
    python -m src.analysis.news_replay --fidelity
"""

from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config.settings import settings

REPLAY_VERSION = "news-replay-v1"
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
    cache = CACHE_DIR / f"polygon_asof_{when:%Y-%m-%d_%H}.json"
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


def build_tick_pool(when: datetime, universe: set, index=None,
                    union_hours: float = 0.0) -> Tuple[list, dict]:
    """The recoverable half of the tick's pool, with its provenance.

    ``union_hours`` = 0 is the FAITHFUL pool: exactly the bundle the tick used
    plus the Polygon call it made. Above 0 it adds articles from earlier
    bundles published within that many hours — a deliberate departure from what
    the tick held, so it is stamped into the row's `pool_spec` and can never be
    confused with the faithful one.
    """
    from src.data.news_fetcher import _dedupe_by_url
    index = index if index is not None else _bundle_index()
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
    polygon = _polygon_as_of(when, universe)
    extra = _earlier_bundle_articles(index, when, union_hours)
    pool = _dedupe_by_url(_reconfirm_tags(bundle + polygon + extra))
    prov = {"bundle_file": (path.name if path else None),
            "bundle_stamp": (stamp.isoformat() if stamp else None),
            "n_bundle": len(bundle), "n_polygon": len(polygon), "n_extra": len(extra),
            "n_pool": len(pool),
            "pool_spec": ("faithful" if union_hours <= 0 else f"union{int(union_hours)}h")}
    return pool, prov


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


def replay_tick(run_id: str, engine: str = "local", limit_tickers: Optional[int] = None,
                budget_seconds: Optional[float] = None, index=None,
                paired_only: bool = False, union_hours: float = 0.0) -> dict:
    """Regenerate the news family for one tick. Returns a summary; rows are
    persisted to ``news_replay``."""
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
    pool, prov = build_tick_pool(info["when"], set(tickers), index=index,
                                 union_hours=union_hours)
    logger.info(f"[news-replay] {run_id} @ {info['when']:%Y-%m-%d %H:%M}Z — pool {prov['n_pool']} "
                f"({prov['n_bundle']} bundle + {prov['n_polygon']} polygon), {len(tickers)} tickers")
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
    baselines = baselines_as_of(info["signal_date"], prov["pool_spec"])
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
            try:
                rows.append(_replay_one(tk, pool, info, engine, prov, baselines))
            except Exception as exc:                            # noqa: BLE001
                logger.debug(f"[news-replay] {tk}: {exc}")
            if i % 50 == 0:
                logger.info(f"[news-replay] {run_id}: {i}/{len(tickers)}")
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
           "bundle_file": prov["bundle_file"], "pool_spec": prov["pool_spec"]}
    if not arts:
        return {**row, **{c: 0.0 for c in NEWS_REPLAY_COLUMNS}}
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
    score, _rationale, meta = analyse_sentiment(ticker, arts, force_engine=engine,
                                               as_of=when, allow_provider=True,
                                               store_digest=False)
    news = float(score or 0.0)
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
            "news_recency_mass": mass, "news_article_count": n_art}


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
    args = ap.parse_args(argv)

    only = [r.strip() for r in args.only_runs.split(",") if r.strip()]
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
        spec = "faithful" if args.union_hours <= 0 else f"union{int(args.union_hours)}h"
        if not args.redo and already_done(run_id, spec):
            logger.info(f"[news-replay] {run_id}: already replayed, skipping")
            continue
        out = replay_tick(run_id, engine=args.engine, limit_tickers=args.limit_tickers,
                          budget_seconds=args.budget_seconds, index=index,
                          paired_only=args.paired_only, union_hours=args.union_hours)
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
