"""PAIRED test: FULL news digest vs the FRESHEST CLUSTER only (2026-09-07).

    .venv\\Scripts\\python.exe scripts\\compare_news_truncation.py --n 120
    .venv\\Scripts\\python.exe scripts\\compare_news_truncation.py --n 120 --engines local,deepseek
    .venv\\Scripts\\python.exe scripts\\compare_news_truncation.py --n 60 --ratio 3 --floor-hours 24 --out runs/trunc

THE QUESTION
------------
A ticker's digest often holds two groups of articles — a few from an hour ago
and a few from days back, frequently a DIFFERENT catalyst whose move is already
in the price. Two candidate rules:

  FULL  (today's behaviour)  every relevant article <7d, recency-sorted, top 20.
  TRUNC (the treatment)      only `sentiment.recent_cluster`: age <= max(24h,
                             3 x the freshest article's age).

The worry behind TRUNC is dilution: a stale, already-priced catalyst pulling the
sign of a read whose target is the REMAINING move. The worry behind it is the
mirror image: without the old group the model cannot tell new information from
the same story being re-reported, and "how much of the reaction has already
happened" is exactly what the rubric asks it to weigh.

WHAT THIS SCRIPT MEASURES (stage A), AND WHAT IT CANNOT
-------------------------------------------------------
It scores the SAME tickers from ONE article pool, fetched once, cut both ways,
with the verdict cache bypassed and one engine per arm. So the only thing that
differs between the two arms is the cut. It reports:

  * how often the cut CHANGES anything at all (digest differs, article counts)
  * per-call abstention on each arm (a 0.0 removes the ticker from the ranking)
  * SIGN FLIPS and |delta| on the tickers whose digest differs
  * Spearman of the two arms' raw verdicts — the ORDER is what the rank
    transform consumes, so two arms that reorder nothing cannot differ in the
    combine no matter how their levels move
  * the CONFLICT subset: tickers whose fresh and stale groups carry opposite
    signs, scored separately, which is where the two rules can disagree at all

It does NOT measure skill. Skill is per-day Spearman IC against the signed pivot
target with a day-clustered t (`.claude/skills/evaluate`), and one cross-section
is one day. Stage B does that OFFLINE once digests have accrued: replay the
stored `sentiment_digests` rows both ways (they carry per-article
published_at, so the cut is reproducible) and join to the panel's pivot labels.
No live arm is needed and none should be added before stage A says the cut moves
anything.

PRE-REGISTERED (write it down before running, house rule):
  H0 — the cut changes nothing that matters: |Spearman| between arms >= 0.95 and
       sign flips < 5% of scored tickers. Then TRUNC is not worth carrying and
       the question is closed on cost grounds alone.
  If H0 is rejected, stage B decides on skill, and the bar is the house one:
  per-day pivot IC difference with a day-clustered t >= 2 AND same-sign halves,
  split long/short. Agreement is not skill; a lower abstention rate is not
  skill either.
  Anything else — a nicer-looking distribution, fewer zeros, a tighter tail —
  is NOT a reason to ship the cut.

The DB is opened read-only; nothing here writes to it or to the verdict cache.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings                               # noqa: E402

ARMS = ("full", "trunc")


def _spearman(xs, ys) -> float:
    import pandas as pd

    from src.analysis.signal_panel import _spearman as _sp
    r = _sp(pd.Series(list(xs), dtype=float), pd.Series(list(ys), dtype=float))
    return float("nan") if r is None else float(r)


def _universe_sample(n: int, seed: int) -> list:
    """Seeded random sample of the latest run's scored universe."""
    from src.db import repo
    repo.set_read_only(True)
    df = repo.fetch_df(
        """
        SELECT DISTINCT ticker FROM signals
        WHERE run_id = (SELECT run_id FROM signals ORDER BY generated_at DESC LIMIT 1)
        """)
    tickers = sorted(str(t) for t in df["ticker"].tolist())
    rnd = random.Random(seed)
    rnd.shuffle(tickers)
    return sorted(tickers[:n])


def _latest_news_cache():
    """The most recent hourly bundle (per-ticker yfinance + NewsAPI), the same
    file `pipeline._fetch_news` reads.

    Load-bearing for THIS test: the fresh feeds are age-capped at fetch time
    (`fetch_rss_news` / `fetch_google_news` default `max_age_hours=24`), so
    anything older than a day reaches a digest only through this bundle and
    through Polygon. Building the pool without it would leave nothing for the
    cut to remove and "confirm" H0 for the wrong reason.
    """
    import json as _json

    from src.data.cache import CACHE_DIR
    from src.models import NewsArticle
    files = sorted(CACHE_DIR.glob("news_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return [], None
    path = files[-1]
    arts = []
    for a in _json.loads(path.read_text(encoding="utf-8")):
        try:
            arts.append(NewsArticle.model_validate(a))
        except Exception:                                          # noqa: BLE001
            continue
    return arts, path.name


def _build_pool(tickers: list):
    """One fetch, assembled as the live Step 1 does."""
    from src.data import news_fetcher as nf
    from src.data.provider_news import fetch_polygon_news
    cached, cache_name = _latest_news_cache()
    rss = nf.fetch_rss_news()
    google = nf.fetch_google_news(tickers)
    polygon = fetch_polygon_news(tickers)
    pool = nf._dedupe_by_url(cached + rss + polygon + google)
    return pool, {"cache_file": cache_name, "cached": len(cached), "rss": len(rss),
                  "polygon": len(polygon), "google": len(google), "pool": len(pool)}


def _age_h(article, now) -> float:
    return max(0.0, (now - article.published_at).total_seconds() / 3600)


def _digests(tickers, pool, ratio, floor_hours):
    """{arm: {ticker: [articles]}} plus per-ticker age stats."""
    import src.analysis.sentiment as sent
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    out = {a: {} for a in ARMS}
    stats = {}
    for tk in tickers:
        full = sent.filter_relevant_articles(tk, pool)
        # The live path scores the top 20 by recency; apply that to BOTH arms so
        # the contrast is the CUT, not the cap.
        full = sorted([a for a in full if sent._recency_weight(a) > 0.0],
                      key=sent._recency_weight, reverse=True)[:20]
        trunc = sent.recent_cluster(full, ratio=ratio, floor_hours=floor_hours)
        out["full"][tk], out["trunc"][tk] = full, trunc
        if full:
            ages = sorted(_age_h(a, now) for a in full)
            # identity, not equality: two wire-service copies of one story can
            # compare equal as models and would corrupt the dropped set.
            keep = {id(a) for a in trunc}
            dropped = [a for a in full if id(a) not in keep]
            stats[tk] = {
                "n_full": len(full), "n_trunc": len(trunc), "n_dropped": len(dropped),
                "freshest_h": round(ages[0], 2), "oldest_h": round(ages[-1], 2),
                "dropped_ages_h": [round(_age_h(a, now), 2) for a in dropped],
            }
    return out, stats


def _score_arm(arm, engine, tickers, digests, workers, quiet):
    import src.analysis.sentiment as sent
    todo = [tk for tk in tickers if digests[arm].get(tk)]
    rows = {}

    def one(tk):
        arts = digests[arm][tk]
        t0 = time.perf_counter()
        try:
            score, rationale, meta = sent.analyse_sentiment(tk, arts, force_engine=engine)
        except Exception as ex:                                    # noqa: BLE001
            return tk, {"err": str(ex)[:80], "lat": time.perf_counter() - t0,
                        "n_arts": len(arts)}
        lat = time.perf_counter() - t0
        raw = meta.get("raw_score") if isinstance(meta, dict) else None
        if raw is None or "Analysis error" in (rationale or ""):
            return tk, {"err": (rationale or "no verdict")[:80], "lat": lat,
                        "n_arts": len(arts)}
        return tk, {"raw": float(raw), "score": float(score), "catalyst": meta.get("catalyst"),
                    "rationale": (rationale or "")[:240], "lat": lat, "n_arts": len(arts)}

    n_workers = 1 if engine == "local" else workers
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (tk, rec) in enumerate(ex.map(one, todo), 1):
            rows[tk] = rec
            if not quiet and (i % 20 == 0 or i == len(todo)):
                print(f"    {engine}/{arm}: {i}/{len(todo)}", file=sys.stderr, flush=True)
    return rows


def _conflict_subset(tickers, digests, stats, engine, workers, quiet):
    """Tickers whose DROPPED group, scored alone, disagrees in sign with the
    fresh group — the only rows where the two rules can differ in direction.
    Scored as a third arm so the disagreement is measured, not assumed."""
    import src.analysis.sentiment as sent
    dropped = {}
    for tk, st in stats.items():
        if st["n_dropped"] <= 0:
            continue
        keep = set(id(a) for a in digests["trunc"][tk])
        dropped[tk] = [a for a in digests["full"][tk] if id(a) not in keep]
    if not dropped:
        return {}
    rows = {}
    todo = sorted(dropped)

    def one(tk):
        try:
            score, _, meta = sent.analyse_sentiment(tk, dropped[tk], force_engine=engine)
        except Exception as ex:                                    # noqa: BLE001
            return tk, {"err": str(ex)[:60]}
        raw = meta.get("raw_score") if isinstance(meta, dict) else None
        return tk, ({"raw": float(raw)} if raw is not None else {"err": "no verdict"})

    n_workers = 1 if engine == "local" else workers
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (tk, rec) in enumerate(ex.map(one, todo), 1):
            rows[tk] = rec
            if not quiet and (i % 20 == 0 or i == len(todo)):
                print(f"    {engine}/dropped-only: {i}/{len(todo)}", file=sys.stderr, flush=True)
    return rows


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * float(x):.1f}%"


def _report(engine, tickers, stats, scored, dropped_rows) -> dict:
    full, trunc = scored["full"], scored["trunc"]
    both = [tk for tk in tickers
            if "raw" in full.get(tk, {}) and "raw" in trunc.get(tk, {})]
    differs = [tk for tk in both if stats.get(tk, {}).get("n_dropped", 0) > 0]
    flips = [tk for tk in differs
             if full[tk]["raw"] * trunc[tk]["raw"] < 0
             or (full[tk]["raw"] == 0) != (trunc[tk]["raw"] == 0)]
    deltas = [abs(full[tk]["raw"] - trunc[tk]["raw"]) for tk in differs]
    rho = _spearman([full[tk]["raw"] for tk in both], [trunc[tk]["raw"] for tk in both])
    conflict = [tk for tk in differs
                if "raw" in dropped_rows.get(tk, {})
                and dropped_rows[tk]["raw"] * full[tk]["raw"] < 0]
    out = {
        "engine": engine,
        "n_scored_both": len(both),
        "n_digest_differs": len(differs),
        "share_digest_differs": (len(differs) / len(both)) if both else float("nan"),
        "abstain_full": (sum(1 for tk in both if full[tk]["raw"] == 0) / len(both)) if both else float("nan"),
        "abstain_trunc": (sum(1 for tk in both if trunc[tk]["raw"] == 0) / len(both)) if both else float("nan"),
        "spearman_raw": rho,
        "sign_flips": len(flips),
        "share_sign_flips": (len(flips) / len(both)) if both else float("nan"),
        "mean_abs_delta_on_differs": (statistics.fmean(deltas) if deltas else float("nan")),
        "max_abs_delta": (max(deltas) if deltas else float("nan")),
        "n_conflict_clusters": len(conflict),
        "conflict_tickers": sorted(conflict)[:15],
        "mean_articles_full": statistics.fmean([s["n_full"] for s in stats.values()]) if stats else float("nan"),
        "mean_articles_trunc": statistics.fmean([s["n_trunc"] for s in stats.values()]) if stats else float("nan"),
        "mean_latency_full": statistics.fmean([r["lat"] for r in full.values() if "lat" in r]) if full else float("nan"),
        "mean_latency_trunc": statistics.fmean([r["lat"] for r in trunc.values() if "lat" in r]) if trunc else float("nan"),
    }
    out["H0_cut_changes_nothing"] = bool(
        out["n_scored_both"] >= 30
        and out["spearman_raw"] == out["spearman_raw"]
        and abs(out["spearman_raw"]) >= 0.95
        and out["share_sign_flips"] < 0.05)
    return out


def _print(rep: dict) -> None:
    print(f"\n== {rep['engine']}: FULL vs FRESHEST-CLUSTER ==")
    print(f"  scored both arms      : {rep['n_scored_both']}")
    print(f"  digest actually differs: {rep['n_digest_differs']} "
          f"({_pct(rep['share_digest_differs'])} of scored)")
    print(f"  articles/ticker       : full {rep['mean_articles_full']:.1f}  "
          f"trunc {rep['mean_articles_trunc']:.1f}")
    print(f"  abstention (raw = 0)  : full {_pct(rep['abstain_full'])}  "
          f"trunc {_pct(rep['abstain_trunc'])}")
    print(f"  Spearman of raw verdicts: {rep['spearman_raw']:+.3f}   "
          f"sign flips {rep['sign_flips']} ({_pct(rep['share_sign_flips'])})")
    print(f"  |delta| on changed rows : mean {rep['mean_abs_delta_on_differs']:.3f}  "
          f"max {rep['max_abs_delta']:.3f}")
    print(f"  opposite-sign stale cluster: {rep['n_conflict_clusters']} "
          f"{rep['conflict_tickers']}")
    print(f"  latency/call          : full {rep['mean_latency_full']:.2f}s  "
          f"trunc {rep['mean_latency_trunc']:.2f}s")
    print(f"  H0 (cut changes nothing that matters): "
          f"{'NOT REJECTED — do not ship the cut' if rep['H0_cut_changes_nothing'] else 'REJECTED — go to stage B (skill)'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FULL digest vs freshest-cluster, paired")
    ap.add_argument("--n", type=int, default=120, help="universe tickers to sample (seeded)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--engines", default="local", help="comma list; 'local' needs Ollama up")
    ap.add_argument("--ratio", type=float, default=3.0, help="cut at ratio x freshest age")
    ap.add_argument("--floor-hours", type=float, default=24.0, help="never cut below this age")
    ap.add_argument("--no-conflict-arm", action="store_true",
                    help="skip scoring the dropped group alone (saves a third of the calls)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="directory for the per-row JSON dump")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if "local" in engines:
        settings.enable_local_llm = True

    import src.analysis.sentiment as sent
    from src.data import company_names
    # Real calls only, and never poison the live verdict cache with test digests.
    sent._sentiment_cache_get = lambda key: None
    sent._sentiment_cache_put = lambda *a, **k: None

    tickers = _universe_sample(args.n, args.seed)
    company_names.prime(tickers)
    company_names.prime_industries(tickers)
    pool, info = _build_pool(tickers)
    print(f"pool: {info}")
    digests, stats = _digests(tickers, pool, args.ratio, args.floor_hours)
    n_with = sum(1 for tk in tickers if digests["full"].get(tk))
    n_diff = sum(1 for s in stats.values() if s["n_dropped"] > 0)
    print(f"tickers sampled {len(tickers)}, with a digest {n_with}, "
          f"digest cut by the rule {n_diff}")
    if not n_diff:
        print("The cut removes nothing on this pool — H0 holds trivially today; "
              "re-run on a day with older coverage before concluding.")

    reports = {}
    for engine in engines:
        scored = {arm: _score_arm(arm, engine, tickers, digests, args.workers, args.quiet)
                  for arm in ARMS}
        dropped_rows = ({} if args.no_conflict_arm
                        else _conflict_subset(tickers, digests, stats, engine,
                                              args.workers, args.quiet))
        rep = _report(engine, tickers, stats, scored, dropped_rows)
        reports[engine] = rep
        _print(rep)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(args.out, f"truncation_{engine}_{stamp}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"report": rep, "stats": stats,
                           "rows": {arm: scored[arm] for arm in ARMS},
                           "dropped_only": dropped_rows}, fh, indent=2, default=str)
            print(f"  dump: {path}")
    print("\nStage B (skill) is the decision, and it needs accrued panel labels: "
          "replay the stored sentiment_digests both ways and read per-day pivot IC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
