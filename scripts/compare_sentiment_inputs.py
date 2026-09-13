"""PAIRED measurement of the news-abstention fixes on ONE shared article pool (2026-09-04).

    .venv\\Scripts\\python.exe scripts\\compare_sentiment_inputs.py --n 120
    .venv\\Scripts\\python.exe scripts\\compare_sentiment_inputs.py --n 120 --engines deepseek,local --thinking
    .venv\\Scripts\\python.exe scripts\\compare_sentiment_inputs.py --n 40 --variants old:v4:deepseek,new:v5:deepseek

WHAT CHANGED, AND WHY EACH FACTOR NEEDS ITS OWN ARM
---------------------------------------------------
Two things drive the `news` method's zero rate, and a single before/after read
would fold them together:

  INPUT  — the relevance FILTER that builds a ticker's digest. The old one was a
           bare lowercase substring test on the symbol (``"ar" in "market"``),
           so a short symbol swept the whole pool and the scorer was shown 20
           random headlines — it then (correctly) answered "about other
           companies" and abstained. The new one needs a confirmed tag or a
           company mention (`src/data/company_names.py`).
  PROMPT — v4 said "when in doubt, output 0.0"; v5 says a 0.0 is an ABSTENTION
           that removes the ticker from the ranking, reserves it for two cases,
           and gives uncertainty a MAGNITUDE home (the LEAN band).

Every variant here is (filter, prompt, engine[, thinking]) and every variant is
scored on the SAME tickers from the SAME article pool, fetched once, with the
verdict cache bypassed. The only thing that differs between two arms is the
factor named in their labels, so the contrasts are genuinely paired.

Faithful OLD inputs from one fetch: the search-derived feeds decide their tags
at FETCH time (`news_fetcher._confirmed_tags`), so the pool is fetched with the
NEW code and a second, OLD-shaped copy is derived: the queried symbol re-attached
unconditionally to every search-feed article (what the old fetch did), and the
legacy substring filter applied on top. Structured feeds (Polygon, RSS, NewsAPI)
tag identically under both.

WHAT IT MEASURES, AND WHAT IT CANNOT
------------------------------------
The abstention rate at TWO levels — per CALL (the model returned 0.0) and per
TICKER (no digest OR 0.0: what the combine actually sees, since both are the
same observable there) — plus the distribution shape the rank transform cares
about (distinct values, modal mass over the NONZERO verdicts, LEAN-band share),
sign agreement between prompts, and latency.

It does NOT measure skill. Skill is per-day Spearman IC against the signed
pivot target with a day-clustered t over accrued panel rows
(`.claude/skills/evaluate`); one cross-section is one day. A lower abstention
rate is the NECESSARY condition — a ticker the method never ranks cannot carry
information — not the sufficient one. Read the v5 panel IC 3–5 trading days
after deploy.

The DB is opened read-only; nothing here writes to it or to the verdict cache.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from config.settings import settings

LEAN_MAX = 0.10          # v5's LEAN band upper edge (±0.01–0.10)
INCONCLUSIVE_N = 30      # below this many NONZERO verdicts, shape stats are noise


# ── helpers ──────────────────────────────────────────────────────────────────

_LIVE_HEADER = None      # the module's _target_header, captured in main()


def _spearman(xs, ys) -> float:
    """The panel's own scipy-free Spearman (with its errstate guard)."""
    import pandas as pd
    from src.analysis.signal_panel import _spearman as _sp
    r = _sp(pd.Series(list(xs), dtype=float), pd.Series(list(ys), dtype=float))
    return float("nan") if r is None else float(r)


def _universe_sample(n: int, seed: int) -> list:
    """A SEEDED random sample of the latest run's scored universe. Random, not
    'top by article count': the latest run still counted articles under the OLD
    filter, so ranking by that column would over-sample the swept short symbols
    whose zero rate the fix is about."""
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


def _v4_prefix() -> str:
    """The v4 prefix as committed (git HEAD), with the taxonomy interpolated
    exactly as the module does it. The literal is parsed with ast so any escape
    sequence in it resolves the way Python resolved it."""
    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    src = subprocess.run(["git", "show", "HEAD:src/analysis/sentiment.py"],
                         capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout
    if 'v4-2026-08-15' not in src:
        return ""                      # HEAD has moved past v4: no v4 arm available
    start = src.index('_SENTIMENT_PREFIX = """') + len("_SENTIMENT_PREFIX = ")
    end = src.index('"""', start + 3) + 3
    text = ast.literal_eval(src[start:end])
    return text.replace("__CATALYST_TYPES__", ", ".join(NEWS_CATALYST_TYPES))


def _latest_news_cache():
    """The most recent hourly bundle (per-ticker yfinance + NewsAPI) — the same
    file `pipeline._fetch_news` reads. Its yfinance rows were tagged by whichever
    code wrote it, so both pool shapes are re-derived below from the text."""
    from src.data.cache import CACHE_DIR
    from src.models import NewsArticle
    files = sorted(CACHE_DIR.glob("news_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return [], None
    path = files[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    arts = []
    for a in data:
        try:
            arts.append(NewsArticle.model_validate(a))
        except Exception:
            continue
    return arts, path.name


def _build_pools(tickers: list):
    """One fetch → (pool_new, pool_old).

    pool_new: tags as the NEW code confirms them (search-feed tag kept only when
              the article's text mentions the company).
    pool_old: the queried symbol attached UNCONDITIONALLY to every search-feed
              article, as the old fetch did. Derived from the same objects, so
              titles, summaries, sources and timestamps are identical.
    """
    from src.data import news_fetcher as nf
    from src.data.company_names import mention_evidence
    from src.data.provider_news import fetch_polygon_news

    queried: dict = {}                    # (title, summary) → queried symbol(s)
    orig = nf._confirmed_tags

    def _capture(tk, title, summary):
        queried.setdefault((title or "", summary or ""), set()).add((tk or "").strip().upper())
        return orig(tk, title, summary)

    nf._confirmed_tags = _capture
    try:
        cached, cache_name = _latest_news_cache()
        rss = nf.fetch_rss_news()
        google = nf.fetch_google_news(tickers)
        polygon = fetch_polygon_news(tickers)
    finally:
        nf._confirmed_tags = orig

    # The cached bundle was written by the RUNNING scheduler. Re-derive both
    # tag shapes from its text: a cached yfinance row carries its queried
    # symbol as the (old, unconditional) tag; the new shape keeps it only when
    # the text confirms it — byte-for-byte what the new fetch produces.
    cached_new, cached_old = [], []
    for a in cached:
        tags_old = list(getattr(a, "tickers", None) or [])
        a_old = copy.deepcopy(a); a_old.tickers = tags_old
        a_new = copy.deepcopy(a)
        a_new.tickers = [t for t in tags_old
                         if mention_evidence(t, f"{a.title or ''} {a.summary or ''}", allow_token=True)]
        cached_old.append(a_old); cached_new.append(a_new)

    google_old = []
    for a in google:
        b = copy.deepcopy(a)
        q = queried.get((a.title or "", a.summary or ""))
        b.tickers = sorted(set(b.tickers or []) | (q or set()))
        google_old.append(b)

    pool_new = nf._dedupe_by_url(cached_new + rss + polygon + google)
    pool_old = nf._dedupe_by_url(cached_old + rss + copy.deepcopy(polygon) + google_old)
    info = {"cache_file": cache_name, "cached": len(cached), "rss": len(rss),
            "polygon": len(polygon), "google": len(google), "pool": len(pool_new)}
    return pool_new, pool_old, info


def _prefix_from_file(path: str) -> str:
    """A prompt prefix saved to disk (e.g. a never-deployed version kept for a
    paired arm), with the taxonomy interpolated the way the module does it."""
    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    text = Path(path).read_text(encoding="utf-8")
    return text.replace("__CATALYST_TYPES__", ", ".join(NEWS_CATALYST_TYPES))


def _bare_header(ticker: str) -> str:
    """The pre-v6 per-ticker header (symbol only, no company name, no fund flag)."""
    return f"TARGET TICKER: {(ticker or '').strip().upper()}"


def _digest_for(tk: str, which: str, pool_new, pool_old):
    import src.analysis.sentiment as sent
    if which == "old":
        return sent._filter_relevant_articles_legacy(tk, pool_old)
    return sent.filter_relevant_articles(tk, pool_new)


# ── scoring ──────────────────────────────────────────────────────────────────

def _score_variant(label, filt, prompt, engine, thinking, tickers, digests, prefixes,
                   workers: int, show_progress: bool):
    import src.analysis.sentiment as sent

    # A prompt label ending in "~" runs that prefix with the BARE pre-v6 header
    # (symbol only), so the v6 header (company name + fund flag) is its own
    # factor: v5~ -> v5 isolates the header, v5 -> v6 isolates the text.
    bare = prompt.endswith("~")
    key = prompt.rstrip("~")
    sent._SENTIMENT_PREFIX = prefixes[key]["text"]
    sent._SENT_PROMPT_VERSION = prefixes[key]["version"]
    sent._target_header = _bare_header if bare else _LIVE_HEADER
    settings.llm_max_thinking = bool(thinking)

    rows = {}
    todo = [tk for tk in tickers if digests[filt].get(tk)]

    def one(tk):
        arts = digests[filt][tk]
        t0 = time.perf_counter()
        try:
            score, rationale, meta = sent.analyse_sentiment(tk, arts, force_engine=engine)
        except Exception as ex:                                   # engine raised
            return tk, {"err": str(ex)[:80], "lat": time.perf_counter() - t0, "n_arts": len(arts)}
        lat = time.perf_counter() - t0
        raw = meta.get("raw_score") if isinstance(meta, dict) else None
        if raw is None or "Analysis error" in (rationale or ""):
            return tk, {"err": (rationale or "no verdict")[:80], "lat": lat, "n_arts": len(arts)}
        return tk, {"raw": float(raw), "score": float(score), "catalyst": meta.get("catalyst"),
                    "rationale": (rationale or "")[:240], "lat": lat, "n_arts": len(arts)}

    n_workers = 1 if engine == "local" else workers
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (tk, rec) in enumerate(ex.map(one, todo), 1):
            rows[tk] = rec
            if show_progress and (i % 20 == 0 or i == len(todo)):
                print(f"    {label}: {i}/{len(todo)}", file=sys.stderr, flush=True)
    return rows


def _summarise(label, tickers, digests_f, rows):
    N = len(tickers)
    n_digest = sum(1 for tk in tickers if digests_f.get(tk))
    sizes = [len(digests_f[tk]) for tk in tickers if digests_f.get(tk)]
    ok = {tk: r for tk, r in rows.items() if "raw" in r}
    err = {tk: r for tk, r in rows.items() if "err" in r}
    raws = [r["raw"] for r in ok.values()]
    zeros = sum(1 for v in raws if v == 0.0)
    nz = [v for v in raws if v != 0.0]
    lat = [r["lat"] for r in rows.values()]
    out = {
        "label": label, "N": N, "n_digest": n_digest,
        "digest_mean": statistics.fmean(sizes) if sizes else 0.0,
        "digest_capped": sum(1 for s in sizes if s >= 20),
        "calls_ok": len(ok), "calls_err": len(err),
        "abstain_call": zeros / len(ok) if ok else float("nan"),
        "zero_ticker": (N - n_digest + zeros + len(err)) / N if N else float("nan"),
        "n_nonzero": len(nz), "distinct_nz": len(set(nz)),
        "mean_abs_nz": statistics.fmean(abs(v) for v in nz) if nz else float("nan"),
        "pos": sum(1 for v in nz if v > 0), "neg": sum(1 for v in nz if v < 0),
        "lean_share": (sum(1 for v in nz if abs(v) <= LEAN_MAX) / len(nz)) if nz else float("nan"),
        "lat_p50": statistics.median(lat) if lat else float("nan"),
        "lat_p90": (sorted(lat)[int(0.9 * (len(lat) - 1))] if lat else float("nan")),
    }
    if nz:
        c = Counter(round(v, 2) for v in nz)
        top = c.most_common(4)
        out["modal"] = top[0][0]; out["modal_mass"] = top[0][1] / len(nz)
        out["top4_mass"] = sum(m for _, m in top) / len(nz)
    else:
        out["modal"] = float("nan"); out["modal_mass"] = float("nan"); out["top4_mass"] = float("nan")
    out["shape_verdict"] = "INCONCLUSIVE" if len(nz) < INCONCLUSIVE_N else "ok"
    return out


def _print_summary(summ):
    print(f"\n{'variant':<24}{'digest':>8}{'calls':>7}{'err':>5}{'abst/call':>10}{'zero/tkr':>9}"
          f"{'nz':>5}{'dist':>5}{'modal':>7}{'m.mass':>7}{'top4':>6}{'lean':>6}{'|v|':>6}"
          f"{'pos:neg':>9}{'p50s':>6}{'p90s':>6}")
    for s in summ:
        print(f"{s['label']:<24}{s['n_digest']:>5}/{s['N']:<3}{s['calls_ok']:>6}{s['calls_err']:>5}"
              f"{s['abstain_call']:>10.1%}{s['zero_ticker']:>9.1%}"
              f"{s['n_nonzero']:>5}{s['distinct_nz']:>5}{s['modal']:>+7.2f}{s['modal_mass']:>7.1%}"
              f"{s['top4_mass']:>6.0%}{s['lean_share']:>6.0%}{s['mean_abs_nz']:>6.2f}"
              f"{s['pos']:>5}:{s['neg']:<3}{s['lat_p50']:>6.1f}{s['lat_p90']:>6.1f}"
              + ("   <- shape INCONCLUSIVE (<30 nonzero)" if s["shape_verdict"] != "ok" else ""))
    print("\n  abst/call = model returned 0.0 | zero/tkr = no digest OR 0.0 OR error, over ALL sampled"
          "\n  tickers (what the combine sees) | modal/top4 = mass over NONZERO verdicts | lean = |v| <= 0.10")


def _paired(results, a, b, tickers, what):
    """Rows scored under BOTH arms; then the change from a to b."""
    ra, rb = results.get(a), results.get(b)
    if ra is None or rb is None:
        return
    both = [tk for tk in tickers if "raw" in ra.get(tk, {}) and "raw" in rb.get(tk, {})]
    if len(both) < 5:
        print(f"\n  {a} -> {b}: only {len(both)} rows scored by both")
        return
    va = [ra[tk]["raw"] for tk in both]; vb = [rb[tk]["raw"] for tk in both]
    both_nz = [(x, y) for x, y in zip(va, vb) if x != 0.0 and y != 0.0]
    a0_b1 = [y for x, y in zip(va, vb) if x == 0.0 and y != 0.0]
    a1_b0 = sum(1 for x, y in zip(va, vb) if x != 0.0 and y == 0.0)
    print(f"\n  {a} -> {b}  ({what}; n={len(both)} rows scored by both)")
    print(f"    Spearman (all rows)              {_spearman(va, vb):+.3f}")
    if len(both_nz) >= 5:
        agree = sum(1 for x, y in both_nz if (x > 0) == (y > 0)) / len(both_nz)
        print(f"    Spearman (both nonzero)          "
              f"{_spearman([x for x, _ in both_nz], [y for _, y in both_nz]):+.3f}  (n={len(both_nz)})")
        print(f"    sign agreement (both nonzero)    {agree:.1%}")
    print(f"    zero under {a[:14]:<14} -> nonzero under {b[:14]:<14}: {len(a0_b1):>3}"
          f"  (pos {sum(1 for v in a0_b1 if v > 0)} / neg {sum(1 for v in a0_b1 if v < 0)}, "
          f"mean |v| {statistics.fmean(abs(v) for v in a0_b1) if a0_b1 else 0:.3f})")
    print(f"    nonzero under {a[:11]:<11} -> zero under {b[:11]:<11}:    {a1_b0:>3}")


def _factors(label: str) -> tuple:
    """'new:v6:deepseek+think' -> ('new', 'v6', 'deepseek', True)."""
    base, _, think = label.partition("+")
    f, p, e = base.split(":")
    return f, p, e, bool(think)


def _one_factor_contrasts(results, order, tickers):
    """Every pair of scored arms that differ in exactly ONE factor (filter,
    prompt, engine, thinking), in the order the arms were listed."""
    names = ("filter", "prompt", "engine", "thinking")
    labels = [l for l in order if l in results]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            fa, fb = _factors(a), _factors(b)
            diff = [k for k in range(4) if fa[k] != fb[k]]
            if len(diff) != 1:
                continue
            k = diff[0]
            _paired(results, a, b, tickers, f"{names[k]}: {fa[k]} -> {fb[k]}")


def _digest_contrast(tickers, digests):
    old, new = digests["old"], digests["new"]
    swept = [tk for tk in tickers if len(old.get(tk) or []) >= 20]
    lost = [tk for tk in tickers if old.get(tk) and not new.get(tk)]
    gained = [tk for tk in tickers if new.get(tk) and not old.get(tk)]
    print("\nDIGEST CONTRAST (same pool, two filters)")
    print(f"  tickers with a digest: old {sum(1 for tk in tickers if old.get(tk))}  new "
          f"{sum(1 for tk in tickers if new.get(tk))}  of {len(tickers)}")
    print(f"  old digests at the 20-headline cap (the sweep): {len(swept)}"
          + (f"  e.g. {', '.join(swept[:8])}" if swept else ""))
    print(f"  had an old digest, none under new: {len(lost)}"
          + (f"  e.g. {', '.join(lost[:8])}" if lost else ""))
    print(f"  no old digest, one under new:      {len(gained)}"
          + (f"  e.g. {', '.join(gained[:8])}" if gained else ""))
    shrink = [(tk, len(old[tk]), len(new[tk])) for tk in tickers if old.get(tk) and new.get(tk)]
    if shrink:
        print(f"  both: mean size old {statistics.fmean(o for _, o, _ in shrink):.1f} -> new "
              f"{statistics.fmean(n for _, _, n in shrink):.1f}")


def _parse_variants(spec: str, engines: list, thinking: bool, live: str, have_v4: bool) -> list:
    """(label, filter, prompt, engine, thinking) tuples. *live* is the module
    prompt's label (``v6`` for ``v6-2026-09-04``)."""
    out = []
    if spec:
        for item in spec.split(","):
            base, _, think = item.strip().partition("+")
            f, p, e = base.split(":")
            out.append((item.strip(), f, p, e, bool(think)))
        return out
    # Default design: factor isolation on deepseek, both prompts on every other
    # engine under the NEW filter (the only one that ships), thinking as a
    # single opt-in arm on the deploy candidate.
    if have_v4:
        out += [("old:v4:deepseek", "old", "v4", "deepseek", False),   # as-was
                ("new:v4:deepseek", "new", "v4", "deepseek", False),   # input fix alone
                (f"old:{live}:deepseek", "old", live, "deepseek", False)]   # prompt alone
    out.append((f"new:{live}:deepseek", "new", live, "deepseek", False))   # deploy candidate
    for e in engines:
        if e == "deepseek":
            continue
        if have_v4:
            out.append((f"new:v4:{e}", "new", "v4", e, False))
        out.append((f"new:{live}:{e}", "new", live, e, False))
    if thinking:
        out.append((f"new:{live}:deepseek+think", "new", live, "deepseek", True))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paired abstention measurement: filter x prompt x engine")
    ap.add_argument("--n", type=int, default=120, help="universe tickers to sample (seeded)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--engines", default="deepseek", help="comma list; 'local' needs the Ollama box up")
    ap.add_argument("--thinking", action="store_true", help="add a DeepSeek thinking-on arm (v5, new filter)")
    ap.add_argument("--variants", default="",
                    help="explicit filter:prompt:engine[+think] list (overrides the design); a prompt "
                         "label ending in '~' runs with the bare pre-v6 header")
    ap.add_argument("--prefix-file", action="append", default=[], metavar="LABEL=PATH",
                    help="extra prompt prefix from a file (taxonomy placeholder interpolated)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="directory for per-row CSV + digest JSON (default: no dump)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from src.db import repo
    repo.set_read_only(True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if "local" in engines:
        settings.enable_local_llm = True

    import src.analysis.sentiment as sent
    from src.data import company_names
    # Real calls only — a cached verdict was formed under some other prompt or
    # digest, and this run must never poison the live cache with test prompts.
    sent._sentiment_cache_get = lambda key: None
    sent._sentiment_cache_put = lambda *a, **k: None
    live_prefix, live_version = sent._SENTIMENT_PREFIX, sent._SENT_PROMPT_VERSION
    live_label = live_version.split("-")[0]                      # "v6-2026-09-04" -> "v6"
    global _LIVE_HEADER
    _LIVE_HEADER = sent._target_header
    prefixes = {live_label: {"text": live_prefix, "version": live_version}}
    v4 = _v4_prefix()
    if v4:
        prefixes["v4"] = {"text": v4, "version": "v4-2026-08-15"}
    for item in args.prefix_file:
        label, _, path = item.partition("=")
        prefixes[label.strip()] = {"text": _prefix_from_file(path.strip()),
                                   "version": f"{label.strip()}-harness"}
    print(f"module prompt: {live_version}; prefixes available: {sorted(prefixes)}")

    tickers = _universe_sample(args.n, args.seed)
    company_names.prime(tickers)
    print(f"tickers sampled: {len(tickers)} (seed {args.seed}) from the latest run's universe")
    print("fetching the shared pool once (cached hourly bundle + RSS + Polygon + Google News) ...",
          flush=True)
    pool_new, pool_old, info = _build_pools(tickers)
    print(f"  pool: {info['pool']} articles  (bundle {info['cache_file']}: {info['cached']}, "
          f"rss {info['rss']}, polygon {info['polygon']}, google {info['google']})")

    digests = {"old": {tk: _digest_for(tk, "old", pool_new, pool_old) for tk in tickers},
               "new": {tk: _digest_for(tk, "new", pool_new, pool_old) for tk in tickers}}
    _digest_contrast(tickers, digests)

    variants = _parse_variants(args.variants, engines, args.thinking, live_label, bool(v4))
    missing = sorted({p.rstrip("~") for _, _, p, _, _ in variants} - set(prefixes))
    if missing:
        print(f"unknown prompt label(s) {missing}; pass --prefix-file LABEL=PATH")
        return 2
    results, summ = {}, []
    for label, filt, prompt, engine, think in variants:
        print(f"\nscoring {label} ...", flush=True)
        t0 = time.perf_counter()
        rows = _score_variant(label, filt, prompt, engine, think, tickers, digests, prefixes,
                              args.workers, not args.quiet)
        n_err = sum(1 for r in rows.values() if "err" in r)
        if rows and n_err == len(rows):
            print(f"  {label}: every call failed ({next(iter(rows.values()))['err']}) — arm dropped")
            continue
        results[label] = rows
        summ.append(_summarise(label, tickers, digests[filt], rows))
        print(f"  done in {time.perf_counter() - t0:.0f}s")

    # restore the module state we monkeypatched (cosmetic — the process exits)
    sent._SENTIMENT_PREFIX, sent._SENT_PROMPT_VERSION = live_prefix, live_version
    sent._target_header = _LIVE_HEADER
    settings.llm_max_thinking = False

    if not summ:
        print("\nno variant produced verdicts")
        return 1

    print("\n" + "=" * 100)
    print(f"PAIRED on {len(tickers)} tickers, one shared article pool")
    _print_summary(summ)

    print("\nPAIRED CONTRASTS (every pair of arms differing in exactly one factor)")
    _one_factor_contrasts(results, [v[0] for v in variants], tickers)
    if "old:v4:deepseek" in results and f"new:{live_label}:deepseek" in results:
        _paired(results, "old:v4:deepseek", f"new:{live_label}:deepseek", tickers,
                "as-was -> deploy candidate")

    # Zero rationales under the deploy candidate(s): are they the abstention
    # cases the prompt reserves 0.0 for? (Eyeball read — the numbers above decide.)
    for e in engines:
        cand_label = f"new:{live_label}:{e}"
        cand = results.get(cand_label) or {}
        zr = [(tk, r["rationale"]) for tk, r in cand.items() if r.get("raw") == 0.0]
        if zr:
            print(f"\nZERO rationales under {cand_label} ({len(zr)}), first 12:")
            for tk, why in zr[:12]:
                print(f"  {tk:<6} {why[:150].encode('ascii', 'replace').decode()}")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        with (out / "sentiment_inputs_rows.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["variant", "ticker", "n_arts", "raw", "score", "catalyst", "lat_s", "err", "rationale"])
            for label, rows in results.items():
                for tk, r in rows.items():
                    w.writerow([label, tk, r.get("n_arts"), r.get("raw"), r.get("score"),
                                r.get("catalyst"), f"{r.get('lat', 0):.2f}", r.get("err", ""),
                                r.get("rationale", "")])
        with (out / "sentiment_inputs_digests.json").open("w", encoding="utf-8") as fh:
            json.dump({f: {tk: [f"[{a.source}] {a.title}" for a in arts[:20]]
                           for tk, arts in d.items() if arts}
                       for f, d in digests.items()}, fh, indent=1)
        print(f"\nrows -> {out / 'sentiment_inputs_rows.csv'}   digests -> {out / 'sentiment_inputs_digests.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
