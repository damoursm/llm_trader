"""Acceptance harness for the LOCAL sentiment engine (2026-09-03).

Run before pointing production at a self-hosted model, and again after every
model swap::

    .venv\\Scripts\\python.exe scripts\\bench_local_sentiment.py --n 40

WHAT THIS CAN AND CANNOT DECIDE
-------------------------------
It decides the MECHANICAL questions, which are the ones that can be answered
offline in minutes and each of which has a documented failure precedent here:

  * LATENCY - the sentiment pass makes ~68 fresh calls per tick and the tick
    has to finish inside its 30-minute slot (measured 2026-09-02: 2,378 fresh
    calls / 35 runs). A model that is accurate at 8 s/call is useless.
  * FORMAT - every verdict must parse. `_parse_response` has a regex salvage,
    but a salvage rate above a few percent means rationales are being lost and
    the news-event dataset degrades silently.
  * DISTRIBUTION - the pathology this codebase has been bitten by TWICE
    (sentiment prompt v2, confidence placement v1) is MODAL COLLAPSE: a model
    that emits the same few numbers turns the rank transform into one big tie
    group, and `news` is a rank-consumed method. Measured over the NONZERO
    verdicts ONLY, because "Zeros ABSTAIN (never ranked)" - a zero never
    enters the rank distribution, so counting abstentions as a tie group
    measures a population the combine never sees. (The first version of this
    harness got that wrong and failed a model on its abstention rate.)
  * ABSTENTION - checked as a BAND, not a floor. Never saying 0.0 means the
    model is pattern-matching the prompt rather than reading the news;
    abstaining on nearly everything means it contributes no cross-section.
    Reference: DeepSeek abstains on **73.0%** of news-carrying tickers -
    MEASURED head-to-head on identical article sets (63 tickers, 2026-09-03,
    scripts/compare_sentiment_engines.py), not derived. An earlier ~55% figure
    inferred from panel aggregates (`nonzero_news`/`has_articles`) was wrong:
    it divided a SCALED score by an article count and swept in the non-LLM
    provider-sentiment rows. Derive nothing you can measure directly.

It does NOT decide whether the model is any GOOD. Skill is per-day Spearman IC
of the score against the signed pivot target (`.claude/skills/evaluate`), which
needs panel rows the local engine has not produced yet. Nothing here is a
substitute for that; the numbers below are a GATE, not a verdict.

There is deliberately no head-to-head against DeepSeek: the sentiment cache
stores an article-set HASH, not the articles, so the exact digest a past
DeepSeek verdict saw cannot be replayed. Comparing against a re-fetched digest
would compare two different inputs and call the difference a model difference.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from config.settings import settings


# Nonzero verdicts needed before the tie-mass / variety checks mean anything.
_MIN_NONZERO = 30


def _universe(n: int) -> list:
    """Recent Gate-4 tradeable tickers that actually carried news - benchmarking
    on names with no news would measure the empty-input short circuit."""
    from src.db import repo
    repo.set_read_only(True)
    df = repo.fetch_df(
        """
        SELECT ticker, MAX(news_article_count) AS arts
        FROM signals
        WHERE signal_date >= (SELECT MAX(signal_date) FROM signals)
          AND news_article_count > 0
        GROUP BY ticker ORDER BY arts DESC LIMIT ?
        """, [int(n)])
    return [str(t) for t in df["ticker"].tolist()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local sentiment engine acceptance harness")
    ap.add_argument("--n", type=int, default=120,
                    help="tickers to score (>=100: the tie-mass check needs ~30 "
                         "NONZERO verdicts to be stable - at n=40 it landed on a "
                         "coin-flip boundary and flipped the verdict)")
    ap.add_argument("--model", default=None, help="override local_sentiment_model")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--use-cache", action="store_true",
                    help="allow cached verdicts (default: every call is real, "
                         "so the latency figure is one a tick would actually pay)")
    args = ap.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    if args.model:
        settings.local_sentiment_model = args.model
    settings.enable_local_llm = True

    import src.analysis.sentiment as sent
    from src.data.news_fetcher import fetch_ticker_news

    if sent._get_local() is None:
        print("local engine is not configured (enable_local_llm / local_sentiment_base_url)")
        return 2

    if not args.use_cache:
        # Every call must be a REAL call: the verdict cache is keyed on
        # (ticker, engine, article set, prompt version), so a second run over
        # the same tickers would serve the first run's answers at 0.00s and
        # report a latency that no tick will ever see.
        sent._sentiment_cache_get = lambda key: None
        sent._sentiment_cache_put = lambda *a, **k: None

    tickers = _universe(args.n)
    if not tickers:
        print("no recent tickers with news in the signals panel")
        return 2
    print(f"model={settings.local_sentiment_model}  endpoint={settings.local_sentiment_base_url}")
    print(f"scoring {len(tickers)} ticker(s) with FRESH news\n")

    rows, latencies, failures, salvaged = [], [], 0, 0
    raw_texts = []

    for i, tk in enumerate(tickers, 1):
        try:
            # fetch_ticker_news takes a LIST and tags each article with its
            # ticker; keep only this ticker's, exactly as the pipeline does.
            articles = [a for a in fetch_ticker_news([tk])
                        if tk in (getattr(a, "tickers", None) or [tk])]
        except Exception as e:
            logger.warning(f"{tk}: news fetch failed ({e})")
            continue
        if not articles:
            continue
        t0 = time.perf_counter()
        try:
            # force_engine pins the local engine with NO cross-engine fallback,
            # so a hosted engine can never silently answer for it and flatter
            # the benchmark.
            score, rationale, meta = sent.analyse_sentiment(tk, articles,
                                                            force_engine="local")
        except Exception as e:
            failures += 1
            logger.warning(f"{tk}: {e}")
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt)
        if "Analysis error" in (rationale or ""):
            failures += 1
            continue
        if "truncated response" in (rationale or ""):
            salvaged += 1
        rows.append({"ticker": tk, "score": score, "catalyst": meta.get("catalyst"),
                     "raw": meta.get("raw_score"), "n_articles": len(articles),
                     "seconds": round(dt, 2), "rationale": rationale})
        raw_texts.append(rationale or "")
        print(f"  [{i:3d}/{len(tickers)}] {tk:6s} {score:+.3f} "
              f"({meta.get('catalyst') or '-':<20s}) {dt:5.1f}s  {len(articles):2d} art")

    if not rows:
        print("\nno verdicts produced - the model or endpoint is not working")
        return 1

    raws = [r["raw"] for r in rows if r["raw"] is not None]
    nonzero = [r for r in raws if r != 0.0]
    # Tie mass is judged on the NONZERO verdicts: zeros abstain and are never
    # ranked, so they cannot form a tie group in the combine.
    distinct = len(set(nonzero))
    modal_share = (Counter(nonzero).most_common(1)[0][1] / len(nonzero)) if nonzero else 1.0
    abstain = 1 - len(nonzero) / max(1, len(raws))
    p50 = statistics.median(latencies)
    p90 = sorted(latencies)[max(0, int(0.9 * len(latencies)) - 1)]
    per_tick = p50 * 68     # the measured fresh-call count per tick

    print("\n" + "=" * 62)
    print(f"verdicts            {len(rows)}/{len(tickers)}   failures={failures} salvaged={salvaged}")
    print(f"latency             p50 {p50:.2f}s   p90 {p90:.2f}s")
    print(f"  -> 68 calls/tick  ~{per_tick / 60:.1f} min serial "
          f"(~{per_tick / 60 / 4:.1f} min at OLLAMA_NUM_PARALLEL=4)")
    print(f"distinct NONZERO    {distinct} of {len(nonzero)}   (zeros abstain, never ranked)")
    print(f"modal share NONZERO {modal_share:.1%}  (v2 prompt collapse was 98% tie mass)")
    print(f"abstention (0.0)    {abstain:.1%}   (DeepSeek measured 73.0%, paired)")
    if nonzero:
        print(f"nonzero |score|     mean {statistics.fmean(abs(x) for x in nonzero):.3f}  "
              f"min {min(abs(x) for x in nonzero):.2f}  max {max(abs(x) for x in nonzero):.2f}")
    print(f"catalysts           {dict(Counter(r['catalyst'] for r in rows).most_common(8))}")

    # The gate. Each threshold is a documented failure mode, not a preference.
    checks = {
        "parses (>=95%)":            len(rows) / max(1, len(rows) + failures) >= 0.95,
        "salvage (<5%)":             salvaged / max(1, len(rows)) < 0.05,
        "fits the tick (<8min)":     per_tick / 60 / 4 < 8,
        "nonzero variety (>=6)":     distinct >= 6,
        "nonzero modal (<50%)":      modal_share < 0.50,
        "abstention in 20-85%":      0.20 <= abstain <= 0.85,
        "takes both directions":     any(x > 0 for x in nonzero) and any(x < 0 for x in nonzero),
    }
    print("-" * 62)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = all(checks.values())

    # A gate whose verdict depends on how many tickers you happened to run is
    # not a gate. The tie-mass and variety checks are the sample-hungry ones:
    # measured 2026-09-03, the SAME model read 45.5% modal over 44 nonzero
    # verdicts (n=120, PASS) and 50.0% over 10 (n=40, FAIL). Below the floor the
    # run reports NO VERDICT rather than a number that looks decisive.
    if len(nonzero) < _MIN_NONZERO:
        print(f"\n  INCONCLUSIVE: only {len(nonzero)} nonzero verdicts "
              f"(need >={_MIN_NONZERO}). Re-run with a larger --n; the tie-mass "
              f"and variety checks above are not stable on this sample.")
        verdict = None
    print("=" * 62)
    label = "INCONCLUSIVE" if verdict is None else ("PASS" if verdict else "FAIL")
    print(f"MECHANICAL GATE: {label}   (skill is decided by panel IC, not here)")

    if args.json:
        print(json.dumps({"rows": rows, "checks": checks, "p50": p50,
                          "distinct": distinct, "modal_share": modal_share}, indent=2))
    return 0 if verdict else (2 if verdict is None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
