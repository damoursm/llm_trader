"""PILOT: can historical news features be REGENERATED from the stored hourly
bundles? (2026-09-07)

    .venv\\Scripts\\python.exe scripts\\pilot_news_backfill.py --n 200
    .venv\\Scripts\\python.exe scripts\\pilot_news_backfill.py --n 50 --workers 1 --dump runs/backfill

THE CLAIM UNDER TEST
--------------------
`cache/news_*.json` holds 1,132 hourly bundles back to 2026-06-16 — every
article with its title, summary, source, publish time and ticker tags. If a
digest can be rebuilt from the bundle that was current at a panel row's own
`generated_at`, then re-scoring it with today's local Qwen + v6 prompt makes the
news family REPLAYABLE the way the OHLCV methods already are: regenerate the
history the scorer epoch masks, instead of discarding it.

The doubt is the POOL, not the plumbing. The bundle carries the per-ticker
yfinance + NewsAPI leg only; the live pool also gets RSS wires, per-ticker Google
News and Polygon, and those two are age-capped at FETCH time (24h), so they are
gone. Measured on a live fetch today the bundle was 577 of a 1,129-article pool
(51%). A digest rebuilt from half the articles is a different digest, and the v6
prompt tiers sources — so the verdict can move for reasons that have nothing to
do with the date being replayed.

WHAT THIS PILOT MEASURES
------------------------
The POOL GAP alone, with engine and prompt held constant. Gold = local Qwen
verdicts that were produced LIVE in the v6 era on the real pool:

  * `sentiment_shadow` rows with `shadow_engine='local'` (the shadow pass), and
  * the local-PRIMARY run 2026-09-04_214338 (`signals.news_raw_score`).

For each sampled row it rebuilds the pool from the bundle at-or-before that
row's own `generated_at`, applies the PRODUCTION path unchanged — the same
`filter_relevant_articles` (name relevance), the same recency weighting and
top-20 cut, the same header and v6 prompt, `force_engine="local"` — and compares
the raw verdicts: Spearman (the order the rank transform consumes), sign
agreement, abstention on each side, mean |delta|, and the digest-size ratio,
which is the pool gap in the most direct form available.

NO LOOK-AHEAD: the bundle is chosen at-or-before the row's `generated_at` and
every article is asserted to have been published at or before it. A violation
aborts the row rather than being silently scored.

The verdict cache is monkeypatched off (a real call per row, and no test verdict
can reach the live cache); the DB is opened read-only; nothing is written to
`signals_replay` — this pilot only decides whether the full ~6-9 GPU-hour run is
worth starting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings                               # noqa: E402

LOCAL_PRIMARY_RUN = "2026-09-04_214338"
WIDEN = {"polygon": False, "gte": "", "lte": ""}
UNION = {"on": True, "days": 7, "reconfirm": True}   # trailing UNION + today's tag rule


def _spearman(xs, ys) -> float:
    import pandas as pd

    from src.analysis.signal_panel import _spearman as _sp
    r = _sp(pd.Series(list(xs), dtype=float), pd.Series(list(ys), dtype=float))
    return float("nan") if r is None else float(r)


# ── the stored bundles ───────────────────────────────────────────────────────

def _bundle_index():
    """[(bundle_utc_hour, path)] sorted — the hourly news caches on disk."""
    from src.data.cache import CACHE_DIR
    out = []
    for p in CACHE_DIR.glob("news_*.json"):
        try:
            stamp = datetime.strptime(p.name[5:18], "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append((stamp, p))
    return sorted(out)


def _bundle_at(index, when: datetime):
    """The bundle that was CURRENT at ``when`` — the newest one stamped at or
    before it. Never a later file: that is the whole point of the exercise."""
    stamps = [s for s, _p in index]
    pos = bisect_right(stamps, when) - 1
    return (index[pos] if pos >= 0 else (None, None))


_POOL_CACHE: dict = {}
_PG_CACHE: dict = {}                    # ticker -> [NewsArticle] over the whole window


def _polygon_history(ticker: str, gte: str, lte: str):
    """Polygon's own history for one ticker over the WHOLE window, fetched once.

    The bundle carries only the yfinance + NewsAPI leg; Polygon is the one other
    live feed that is re-fetchable after the fact (`get_ticker_news_history`,
    verified served ~2 months back). RSS wires and per-ticker Google News are
    age-capped at fetch time and are simply gone, which is the ceiling on any
    reconstruction. One paged call per ticker for the full window, then the
    articles are filtered per row by publish time — never one call per row.
    """
    if ticker in _PG_CACHE:
        return _PG_CACHE[ticker]
    from src.data import polygon_client
    from src.data.provider_news import _parse_iso
    from src.models import NewsArticle
    arts = []
    try:
        for item in polygon_client.get_ticker_news_history(ticker, gte, lte,
                                                           page_delay_s=0.2) or []:
            published = _parse_iso(item.get("published_utc"))
            title = (item.get("title") or "").strip()
            url = (item.get("article_url") or item.get("amp_url") or "").strip()
            if not title or not url or published is None:
                continue
            arts.append(NewsArticle(
                title=title, summary=(item.get("description") or title)[:600], url=url,
                source=((item.get("publisher") or {}).get("name") or "Polygon"),
                published_at=published,
                tickers=[t.upper() for t in (item.get("tickers") or [])]))
    except Exception as e:                                         # noqa: BLE001
        print(f"  [polygon] {ticker}: {e}", file=sys.stderr)
    _PG_CACHE[ticker] = arts
    return arts


def _load_pool(path: Path):
    if path in _POOL_CACHE:
        return _POOL_CACHE[path]
    from src.models import NewsArticle
    arts = []
    try:
        for a in json.loads(path.read_text(encoding="utf-8")):
            try:
                arts.append(NewsArticle.model_validate(a))
            except Exception:                                      # noqa: BLE001
                continue
    except Exception:                                              # noqa: BLE001
        arts = []
    if len(_POOL_CACHE) > 40:                                      # bounded
        _POOL_CACHE.clear()
    _POOL_CACHE[path] = arts
    return arts


_UNION_CACHE: dict = {}
_TAG_CACHE: dict = {}


def _reconfirm(arts):
    """Re-derive each archived article's ticker tags with TODAY's rule.

    The bundles were written before 2026-09-04, so their stored `tickers` are
    the LEGACY unconditional tags: yfinance's related feed and the Google
    symbol query tagged every result with the queried symbol, which is the
    defect the relevance fix removed. `filter_relevant_articles` gives a tagged
    article a free pass, so trusting those tags hands the scorer a digest full
    of other companies' news — and the v6 prompt then correctly returns 0.0.
    Mirrors `news_fetcher._confirmed_tags`: keep the tag only when the text
    mentions the company (single-token tier allowed, since the feed already
    vouched for the association). An article that loses its tag stays in the
    pool untagged, where the filter can still attach it by name.
    """
    from copy import copy

    from src.data.company_names import mention_evidence
    out = []
    for a in arts:
        tags = list(getattr(a, "tickers", None) or [])
        if not tags:
            out.append(a)
            continue
        text = f"{a.title or ''} {a.summary or ''}"
        kept = []
        for t in tags:
            key = (a.url, t)
            if key not in _TAG_CACHE:
                if len(_TAG_CACHE) > 200_000:
                    _TAG_CACHE.clear()
                _TAG_CACHE[key] = bool(mention_evidence(t, text, allow_token=True))
            if _TAG_CACHE[key]:
                kept.append(t)
        b = copy(a)
        b.tickers = kept
        out.append(b)
    return out


def _union_pool(index, when: datetime, days: float = 7.0):
    """Every article the system had FETCHED by ``when`` and still inside the
    scorer's recency window — the union of the hourly bundles, not one snapshot.

    A single bundle is one hour's fetch; the scorer's window is 7 days
    (`_recency_weight` zeroes past 168h), and an article fetched on Monday is
    still in Thursday's digest. Measured at 2026-09-04 23:00Z: one bundle held
    577 unique urls, the trailing-7d union 5,684 — 9.9x. Using the snapshot was
    the pilot's own bug, not a property of the archive.

    Point-in-time twice over: only bundles STAMPED at or before ``when`` are
    read (so an article the system had not fetched yet cannot appear), and each
    article must also have been PUBLISHED at or before it.
    """
    key = (when.replace(minute=0, second=0, microsecond=0), days)
    if key in _UNION_CACHE:
        return _UNION_CACHE[key]
    floor = when - timedelta(days=days)
    seen, out = set(), []
    for stamp, path in index:
        if stamp > when or stamp < floor:
            continue
        for a in _load_pool(path):
            u = getattr(a, "url", None)
            if not u or u in seen:
                continue
            pub = getattr(a, "published_at", None)
            if pub is None or pub > when or pub < floor:
                continue
            seen.add(u)
            out.append(a)
    if len(_UNION_CACHE) > 8:                                      # bounded (38 MB/union)
        _UNION_CACHE.clear()
    _UNION_CACHE[key] = out
    return out


# ── the gold rows ────────────────────────────────────────────────────────────

def _gold_rows(n: int, seed: int) -> list:
    """Local-engine verdicts produced LIVE in the v6 era, on the real pool."""
    from src.db import repo
    repo.set_read_only(True)
    rows = []
    sh = repo.fetch_df(
        "SELECT run_id, generated_at, ticker, shadow_raw AS live_raw, n_articles "
        "FROM sentiment_shadow WHERE shadow_engine = 'local' AND shadow_raw IS NOT NULL")
    if sh is not None and not sh.empty:
        rows += [dict(r, source="shadow") for r in sh.to_dict("records")]
    pr = repo.fetch_df(
        "SELECT run_id, generated_at, ticker, news_raw_score AS live_raw, "
        "       news_article_count AS n_articles "
        "FROM signals WHERE run_id = ? AND news_raw_score IS NOT NULL", [LOCAL_PRIMARY_RUN])
    if pr is not None and not pr.empty:
        rows += [dict(r, source="primary") for r in pr.to_dict("records")]
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows[:n]


# ── scoring one row through the PRODUCTION path ──────────────────────────────

def _score_row(row, index) -> dict:
    import src.analysis.sentiment as sent
    out = dict(row)
    try:
        when = datetime.fromisoformat(str(row["generated_at"]))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
    except Exception as e:                                         # noqa: BLE001
        return dict(out, err=f"bad timestamp: {e}")
    stamp, path = _bundle_at(index, when)
    if path is None:
        return dict(out, err="no bundle at or before this row")
    pool = list(_union_pool(index, when, UNION["days"]) if UNION["on"] else _load_pool(path))
    if UNION["reconfirm"]:
        pool = _reconfirm(pool)
    if WIDEN["polygon"]:
        # Widen the reconstruction with Polygon's own history, clipped to the
        # row's instant. Same no-look-ahead rule as the bundle.
        pool += [a for a in _polygon_history(str(row["ticker"]), WIDEN["gte"], WIDEN["lte"])
                 if a.published_at and a.published_at <= when]
        from src.data.news_fetcher import _dedupe_by_url
        pool = _dedupe_by_url(pool)
    # NO LOOK-AHEAD: a bundle written at hour H can only contain articles
    # published by H, but assert it rather than trusting it — one future article
    # would silently make the whole exercise a backtest of the future.
    future = [a for a in pool if a.published_at and a.published_at > when]
    if future:
        return dict(out, err=f"{len(future)} article(s) published after the row instant")
    ticker = str(row["ticker"])
    # PRODUCTION path from here down: the same relevance filter, recency
    # weighting, top-20 cut, header and v6 prompt the live tick uses.
    arts = sent.filter_relevant_articles(ticker, pool)
    out["bundle"] = path.name
    out["n_rebuilt"] = len(arts)
    if not arts:
        return dict(out, rebuilt_raw=0.0, note="no relevant article in the bundle")
    t0 = time.perf_counter()
    try:
        _score, rationale, meta = sent.analyse_sentiment(ticker, arts, force_engine="local")
    except Exception as e:                                         # noqa: BLE001
        return dict(out, err=str(e)[:90])
    raw = (meta or {}).get("raw_score")
    if raw is None or "Analysis error" in (rationale or ""):
        return dict(out, err=(rationale or "no verdict")[:90])
    out.update(rebuilt_raw=float(raw), catalyst=(meta or {}).get("catalyst"),
               lat=round(time.perf_counter() - t0, 2))
    return out


def _report(rows: list) -> dict:
    ok = [r for r in rows if "rebuilt_raw" in r and r.get("live_raw") is not None]
    errs = [r for r in rows if r.get("err")]
    live = [float(r["live_raw"]) for r in ok]
    rebuilt = [float(r["rebuilt_raw"]) for r in ok]
    both_nz = [(a, b) for a, b in zip(live, rebuilt) if a != 0 and b != 0]
    sizes = [(float(r.get("n_articles") or 0), float(r["n_rebuilt"])) for r in ok
             if (r.get("n_articles") or 0) > 0]
    rep = {
        "n_sampled": len(rows), "n_scored": len(ok), "n_errors": len(errs),
        "abstain_live": (sum(1 for x in live if x == 0) / len(live)) if live else float("nan"),
        "abstain_rebuilt": (sum(1 for x in rebuilt if x == 0) / len(rebuilt)) if rebuilt else float("nan"),
        "spearman_all": _spearman(live, rebuilt) if len(live) > 2 else float("nan"),
        "spearman_both_nonzero": (_spearman([a for a, _ in both_nz], [b for _, b in both_nz])
                                  if len(both_nz) > 2 else float("nan")),
        "sign_agree_both_nonzero": ((sum(1 for a, b in both_nz if (a > 0) == (b > 0)) / len(both_nz))
                                    if both_nz else float("nan")),
        "mean_abs_delta": (statistics.fmean([abs(a - b) for a, b in zip(live, rebuilt)])
                           if live else float("nan")),
        "digest_size_live": (statistics.fmean([a for a, _ in sizes]) if sizes else float("nan")),
        "digest_size_rebuilt": (statistics.fmean([b for _, b in sizes]) if sizes else float("nan")),
        "size_ratio": (statistics.fmean([b / a for a, b in sizes if a > 0]) if sizes else float("nan")),
        "mean_latency_s": (statistics.fmean([r["lat"] for r in ok if "lat" in r])
                           if any("lat" in r for r in ok) else float("nan")),
    }
    # Pre-registered read: the rebuilt score has to ORDER tickers like the live
    # one, since the combine consumes the rank. Levels moving is survivable;
    # the order collapsing is not.
    rep["VERDICT"] = (
        "GO — the pool gap is tolerable, run the full backfill"
        if (rep["n_scored"] >= 30 and rep["spearman_both_nonzero"] == rep["spearman_both_nonzero"]
            and rep["spearman_both_nonzero"] >= 0.60
            and rep["sign_agree_both_nonzero"] >= 0.75
            and abs(rep["abstain_rebuilt"] - rep["abstain_live"]) <= 0.20)
        else "NO-GO — regenerate as its OWN feature with provenance, not as `news`")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pilot: regenerate news verdicts from stored bundles")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=1, help="local Ollama serves 2 slots; 1 is safest")
    ap.add_argument("--union-days", type=float, default=7.0,
                    help="how far back the bundle union reaches (the scorer's own window is 7d; "
                         "a shorter one trades coverage for a digest closer to what live saw)")
    ap.add_argument("--trust-stored-tags", action="store_true",
                    help="keep the archived (pre-2026-09-04, unconditional) ticker tags instead "
                         "of re-deriving them with today's confirmation rule")
    ap.add_argument("--snapshot-only", action="store_true",
                    help="rebuild from the single nearest bundle instead of the trailing-7d "
                         "union (the pilot's original, wrong, unit — kept to reproduce it)")
    ap.add_argument("--polygon", action="store_true",
                    help="widen the reconstruction with Polygon's own per-ticker history "
                         "(one paged call per ticker, clipped per row by publish time)")
    ap.add_argument("--dump", default="", help="directory for the per-row JSON dump")
    args = ap.parse_args(argv)

    settings.enable_local_llm = True
    import src.analysis.sentiment as sent
    from src.data import company_names
    sent._sentiment_cache_get = lambda key: None
    sent._sentiment_cache_put = lambda *a, **k: None

    UNION.update(on=not args.snapshot_only, days=args.union_days,
                 reconfirm=not args.trust_stored_tags)
    index = _bundle_index()
    print(f"bundles on disk: {len(index)}  "
          f"{index[0][0].date() if index else '-'} -> {index[-1][0].date() if index else '-'}")
    rows = _gold_rows(args.n, args.seed)
    if not rows:
        print("no live LOCAL verdicts to compare against yet")
        return 1
    tickers = sorted({str(r["ticker"]) for r in rows})
    if args.polygon:
        stamps = sorted(str(r["generated_at"]) for r in rows)
        WIDEN.update(polygon=True, gte=stamps[0][:10], lte=stamps[-1][:19] + "Z")
        print(f"widening with Polygon history {WIDEN['gte']} -> {WIDEN['lte']} "
              f"({len(tickers)} tickers, one paged call each)")
    company_names.prime(tickers)
    company_names.prime_industries(tickers)
    print(f"gold rows: {len(rows)} ({sum(1 for r in rows if r['source'] == 'shadow')} shadow / "
          f"{sum(1 for r in rows if r['source'] == 'primary')} primary), {len(tickers)} tickers")

    done = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, r in enumerate(ex.map(lambda x: _score_row(x, index), rows), 1):
            done.append(r)
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", file=sys.stderr, flush=True)
    rep = _report(done)
    print("\n== rebuilt-from-bundle vs live local verdict (v6 era, engine held constant) ==")
    for k in ("n_sampled", "n_scored", "n_errors", "digest_size_live", "digest_size_rebuilt",
              "size_ratio", "abstain_live", "abstain_rebuilt", "spearman_all",
              "spearman_both_nonzero", "sign_agree_both_nonzero", "mean_abs_delta",
              "mean_latency_s"):
        v = rep[k]
        print(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
    print(f"\n  {rep['VERDICT']}")
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        path = os.path.join(args.dump,
                            f"pilot_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"report": rep, "rows": done}, fh, indent=2, default=str)
        print(f"  dump: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
