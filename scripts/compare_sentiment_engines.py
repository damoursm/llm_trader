"""PAIRED head-to-head between sentiment engines on IDENTICAL inputs (2026-09-03).

    .venv\\Scripts\\python.exe scripts\\compare_sentiment_engines.py --n 60
    .venv\\Scripts\\python.exe scripts\\compare_sentiment_engines.py --engines deepseek,local,qwen

WHY THIS IS VALID AND THE OBVIOUS ALTERNATIVE IS NOT
----------------------------------------------------
The tempting comparison is "score today with the new engine and compare against
the DeepSeek verdicts already in the panel". That is INVALID: the panel's
verdicts were formed on the article set visible at THEIR run time, and the
sentiment cache stores an article-set HASH rather than the articles, so that
digest cannot be reconstructed. Any difference measured that way is an INPUT
difference reported as a model difference.

This script instead fetches each ticker's news ONCE and hands the SAME
`List[NewsArticle]` to every engine in the same process, with the verdict cache
bypassed so each call is real. That makes the contrast genuinely paired: the
only thing that varies is the model.

WHAT IT MEASURES, AND WHAT IT CANNOT
------------------------------------
It measures AGREEMENT, which is the question that actually decides whether an
engine swap would change the book: correlation and sign agreement of the RAW
verdicts (raw, because the precision scalers are engine-independent), catalyst
agreement, magnitude bias, abstention rates, and latency.

It does NOT measure SKILL. Skill is per-day Spearman IC against the signed
pivot target with a day-clustered t (`.claude/skills/evaluate`), which needs
many DAYS of panel rows. One cross-section is one day: the `--label` read below
is a sanity check with n=1 day and no t, explicitly not a verdict. An engine
with no production history has no IC to compare, full stop.

The useful asymmetry to keep in mind while reading the output: HIGH agreement is
decisive (the engines are interchangeable, so switch on cost/latency and stop
thinking about it), while LOW agreement is NOT evidence that either is better -
it only says the choice matters and must be settled on the panel.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from config.settings import settings


def _spearman(xs, ys) -> float:
    """The panel's OWN Spearman (`signal_panel._spearman`), not a local copy.

    It is scipy-free by the house rule, and its np.errstate guard is
    load-bearing rather than cosmetic - the 2026-08-08/09 scheduler freezes were
    a RuntimeWarning flood from degenerate correlations filling an undrained
    stderr pipe. A second implementation here would silently lack that."""
    import pandas as pd
    from src.analysis.signal_panel import _spearman as _sp
    r = _sp(pd.Series(list(xs), dtype=float), pd.Series(list(ys), dtype=float))
    return float("nan") if r is None else float(r)


def _tickers_with_news(n: int) -> list:
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
    ap = argparse.ArgumentParser(description="Paired sentiment-engine comparison")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--engines", default="deepseek,local")
    ap.add_argument("--label", action="store_true",
                    help="also print a ONE-DAY pivot-IC read (a sanity check, not a verdict)")
    args = ap.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    settings.enable_local_llm = True

    import src.analysis.sentiment as sent
    from src.data.news_fetcher import fetch_ticker_news

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    # Real calls only: a cached verdict would make one engine look instant and,
    # worse, could serve an answer formed on a different article set.
    sent._sentiment_cache_get = lambda key: None
    sent._sentiment_cache_put = lambda *a, **k: None

    tickers = _tickers_with_news(args.n)
    print(f"engines: {engines}   tickers: {len(tickers)}   (same article set per ticker)\n")
    print(f"{'ticker':<8}" + "".join(f"{e:>22}" for e in engines))

    rows, lat = [], {e: [] for e in engines}
    for tk in tickers:
        try:
            arts = fetch_ticker_news([tk])
        except Exception:
            continue
        if not arts:
            continue
        rec = {"ticker": tk, "n_articles": len(arts)}
        ok = True
        for e in engines:
            t0 = time.perf_counter()
            try:
                _, rationale, meta = sent.analyse_sentiment(tk, arts, force_engine=e)
            except Exception as ex:
                print(f"{tk:<8}  {e} FAILED: {str(ex)[:60]}")
                ok = False
                break
            lat[e].append(time.perf_counter() - t0)
            raw = meta.get("raw_score")
            if raw is None or "Analysis error" in (rationale or ""):
                ok = False
                break
            rec[e] = float(raw)
            rec[e + "_cat"] = meta.get("catalyst")
        if not ok:
            continue
        rows.append(rec)
        print(f"{tk:<8}" + "".join(
            f"{rec[e]:>+12.3f} {str(rec[e+'_cat'] or '-')[:9]:<9}" for e in engines))

    if len(rows) < 5:
        print("\ntoo few paired verdicts to compare")
        return 1

    print("\n" + "=" * 70)
    print(f"PAIRED on {len(rows)} tickers, identical article sets\n")
    for e in engines:
        vals = [r[e] for r in rows]
        nz = [v for v in vals if v != 0.0]
        print(f"  {e:<10} abstain {1-len(nz)/len(vals):>5.1%} | "
              f"nonzero |v| mean {statistics.fmean(abs(v) for v in nz) if nz else 0:.3f} | "
              f"distinct {len(set(vals)):>3} | latency p50 {statistics.median(lat[e]):.2f}s")

    # Pairwise agreement - the number that decides whether a swap changes the book.
    print()
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            va, vb = [r[a] for r in rows], [r[b] for r in rows]
            both_nz = [(x, y) for x, y in zip(va, vb) if x != 0.0 and y != 0.0]
            sign_all = sum(1 for x, y in zip(va, vb)
                           if (x > 0) == (y > 0) and (x < 0) == (y < 0)) / len(va)
            cat = sum(1 for r in rows if r[a + "_cat"] == r[b + "_cat"]) / len(rows)
            print(f"  {a} vs {b}:")
            print(f"    Spearman (all rows)      {_spearman(va, vb):+.3f}")
            if len(both_nz) >= 5:
                print(f"    Spearman (both nonzero)  "
                      f"{_spearman([x for x,_ in both_nz],[y for _,y in both_nz]):+.3f}  (n={len(both_nz)})")
                dis = sum(1 for x, y in both_nz if (x > 0) != (y > 0))
                print(f"    OPPOSITE direction       {dis}/{len(both_nz)} "
                      f"({dis/len(both_nz):.1%}) <- these are the rows a swap moves")
            print(f"    exact sign agreement     {sign_all:.1%}")
            print(f"    catalyst agreement       {cat:.1%}")
            print(f"    mean {a}-{b}             {statistics.fmean(x-y for x,y in zip(va,vb)):+.4f}")

    if args.label:
        print("\n" + "-" * 70)
        print("ONE-DAY pivot-IC read - a SANITY CHECK, not a verdict (n=1 day, no t).")
        print("Skill needs many days of panel rows; see .claude/skills/evaluate.")
        try:
            from datetime import date as _date
            from src.analysis.pivot_target import _series, next_pivot_targets, session_close_utc
            from src.db import repo
            repo.set_read_only(True)
            day = str(repo.fetch_df("SELECT MAX(signal_date) d FROM signals").iloc[0]["d"])[:10]
            # the next 30-minute H/L pivot after that session's close, from its close
            lab = {}
            for _tk in sorted(set(r["ticker"] for r in rows)):
                _s = _series(_tk)
                if _s is None:
                    continue
                _idx, _c, _h, _l = _s
                try:
                    _i = list(_idx).index(_date.fromisoformat(day))
                except ValueError:
                    continue
                _r = next_pivot_targets(_tk, [(session_close_utc(day), float(_c[_i]))])[0]
                if _r is not None:
                    lab[(_tk, day)] = (float(_r["target_pct"]), bool(_r["resolved"]))
            ys, keep = [], []
            for r in rows:
                y, settled = lab.get((r["ticker"], day), (float("nan"), False))
                if y == y:
                    ys.append(y)
                    keep.append(r)
            if len(keep) >= 10:
                print(f"  labelled {len(keep)}/{len(rows)} rows on {day}")
                for e in engines:
                    print(f"    {e:<10} within-day IC {_spearman([r[e] for r in keep], ys):+.3f}")
            else:
                print(f"  only {len(keep)} labelled rows - EXPECTED when run intraday: the")
                print(f"  pivot label anchors on the signal date's OWN completed bar, and")
                print(f"  today's is still forming (`_drop_forming_bar`). This tool cannot")
                print(f"  produce an IC read on the same day it scores - which is the point:")
                print(f"  skill accrues over days, agreement is what you get today.")
        except Exception as ex:
            print(f"  label read unavailable: {ex}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
