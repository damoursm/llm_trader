"""One-time historical backfill of news catalyst TYPES for the event study.

The `signals` panel has nonzero news reads back to 2026-06-17, but the catalyst
class is only captured live since prompt v4 (2026-08-15), and the news
rationale text was never persisted on `signals` — so history is typed from two
sources, best-first:

  SOURCE 1 — "rationale" (primary): `recommendations.rationale` BEGINS with the
             sentiment LLM's news rationale verbatim (the aggregator joins
             news_rationale + " | Technical score: …"), for every recommended
             ticker — 3k+ ticker-days of authentic run-time text, the events
             that actually drove decisions. Classified offline in batches; no
             fetching. Provider-skip / no-news texts are left to source 2.
  SOURCE 2 — "polygon" (remainder): re-fetched headlines from Polygon's
             `/v2/reference/news` (verified on the free key, but per-ticker
             density is THIN for mid-caps — Alcoa returned 3 articles over two
             months, so this source types only partial coverage). One paginated
             range query per ticker, cached to `cache/news_backfill/<T>.json`.

Both classify into `sentiment.NEWS_CATALYST_TYPES` with DeepSeek (temp 0) —
ONE taxonomy shared with the live prompt, so all three provenances pool — and
WRITE `news_event_backfill` rows in small idempotent batches (the live
scheduler may hold the DuckDB write lock — `connect()` retries). The
`classifier_version` string records which source typed each row; source 1 may
overwrite source-2/NULL rows (higher fidelity), source 2 never overwrites.

Politeness (source 2): the free Polygon tier allows 5 calls/min SHARED with the
live scheduler's snapshots, so every call waits `--ticker-delay` (default 13 s)
— a full run is a background job of hours, resumable at any point (Ctrl-C
safe). A ticker whose fetch returns nothing is SKIPPED, not marked (a
rate-limited response is indistinguishable from no coverage — retried next
run); a fetched ticker with no headlines near a target day writes a
catalyst-NULL row ("attempted, no coverage") so re-runs stop retrying that day.

CLI:
  python -m src.analysis.news_backfill --dry-run                # scope, no calls
  python -m src.analysis.news_backfill --source rationale       # minutes, no fetch
  python -m src.analysis.news_backfill --source polygon [--limit-tickers 8]
  python -m src.analysis.news_backfill                          # both, rationale first
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from src.analysis.sentiment import NEWS_CATALYST_TYPES, normalize_catalyst

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo("America/New_York")

CLASSIFIER_VERSION = "bf1-polygon-2026-08-15"
RATIONALE_CLASSIFIER_VERSION = "bf1-rationale-2026-08-15"
_HEADLINES_PER_DAY = 12
_WINDOW_PAD_DAYS = 3          # fetch pad + widened classify window
_MAX_TOKENS = 6000

# Rationale texts that carry no catalyst information — left to source 2.
_JUNK_RATIONALE_PREFIXES = ("Provider sentiment (", "No recent news",
                            "All available articles", "News sentiment disabled",
                            "Analysis error", "Rationale unavailable",
                            "Analysis unavailable", "No rationale available",
                            "Technical score:", "Put/call signal:")

# The aggregator builds `rationale` as news_rationale + these joined parts —
# strip them so the classifier sees the news text alone.
_METHOD_MARKERS = (" | Technical score:", " | Put/call signal:")

_CLASSIFY_PROMPT = """You are labeling historical stock-news days for an event study. For each numbered item below (one ticker on one date, with that day's news evidence — headlines or an analyst's news summary), pick the ONE DOMINANT catalyst class — the event most likely to have driven the stock's news that day.

Classes (use exactly these strings): {types}.
Guidance: "earnings" = reported results without an outlook change; "guidance" = outlook raised/cut; "company_pr" = promotional company-issued release with no hard numbers; "short_squeeze_social" = social/positioning attention; "macro_sector" = market- or sector-wide, not company-specific; "none" = no identifiable event; "other" = a real company-specific catalyst outside the classes.

Respond with a JSON array only, one entry per item, no other text:
[{{"i": <item number>, "catalyst": "<class>"}}, ...]

ITEMS:
{items}"""


def _cache_dir() -> Path:
    from src.data.cache import CACHE_DIR
    d = Path(CACHE_DIR) / "news_backfill"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_tables() -> None:
    """One short write-connect so `news_event_backfill` exists before the
    targets query reads it (schema is created on write connections)."""
    from src.db.connection import connect
    with connect():
        pass


def load_targets(limit_tickers: Optional[int] = None) -> dict[str, list[str]]:
    """{ticker: [signal_date, ...]} still needing a catalyst — no backfill row
    yet AND no live-captured catalyst for that (ticker, day)."""
    from src.db import repo
    df = repo.fetch_df("""
        SELECT t.ticker, t.signal_date FROM (
            SELECT DISTINCT ticker, signal_date FROM signals
            WHERE news IS NOT NULL AND news <> 0.0
        ) t
        WHERE NOT EXISTS (SELECT 1 FROM news_event_backfill b
                          WHERE b.ticker = t.ticker AND b.signal_date = t.signal_date)
          AND NOT EXISTS (SELECT 1 FROM signals s2
                          WHERE s2.ticker = t.ticker AND s2.signal_date = t.signal_date
                            AND s2.news_catalyst IS NOT NULL)
        ORDER BY t.ticker, t.signal_date
    """)
    out: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        tk = str(r["ticker"]).upper()
        if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", tk):
            continue                       # indices/futures — Polygon has no news
        out.setdefault(tk, []).append(str(r["signal_date"]))
    if limit_tickers:
        out = dict(list(out.items())[: int(limit_tickers)])
    return out


def _news_part(rationale: str) -> str:
    """The news rationale alone — the aggregator appends per-method score
    fragments after it."""
    text = str(rationale or "")
    for marker in _METHOD_MARKERS:
        text = text.split(marker)[0]
    return text.strip()


def load_rationale_targets() -> list[dict]:
    """[{ticker, signal_date, text}] — the FIRST recommendation rationale of
    each (ticker, day) whose panel news read is nonzero, minus keys already
    live-captured or already rationale-classified (source 1 may overwrite a
    source-2/NULL row — higher fidelity — so those keys stay IN)."""
    from src.db import repo
    df = repo.fetch_df("""
        SELECT ticker, signal_date, rationale FROM (
            SELECT s.ticker, s.signal_date, r.rationale, r.generated_at,
                   row_number() OVER (PARTITION BY s.ticker, s.signal_date
                                      ORDER BY r.generated_at) AS _rn
            FROM recommendations r
            JOIN signals s ON s.run_id = r.run_id AND s.ticker = r.ticker
            WHERE r.rationale IS NOT NULL AND length(r.rationale) > 30
              AND s.news IS NOT NULL AND s.news <> 0.0
              AND NOT EXISTS (SELECT 1 FROM signals s2
                              WHERE s2.ticker = s.ticker AND s2.signal_date = s.signal_date
                                AND s2.news_catalyst IS NOT NULL)
              AND NOT EXISTS (SELECT 1 FROM news_event_backfill b
                              WHERE b.ticker = s.ticker AND b.signal_date = s.signal_date
                                AND b.classifier_version LIKE '%rationale%')
        ) WHERE _rn = 1 ORDER BY ticker, signal_date
    """)
    out = []
    for _, r in df.iterrows():
        text = _news_part(r["rationale"])
        if len(text) < 30 or text.startswith(_JUNK_RATIONALE_PREFIXES):
            continue
        out.append({"ticker": str(r["ticker"]).upper(),
                    "signal_date": str(r["signal_date"]), "text": text[:600]})
    return out


def run_rationales(batch: int = 25, dry_run: bool = False) -> dict:
    """Source 1: classify the stored run-time news rationales. Minutes, no
    external fetching; per-batch DB flush so an interrupt loses ≤ one batch."""
    _ensure_tables()
    from src.db import repo
    targets = load_rationale_targets()
    logger.info(f"[news_backfill] rationale source: {len(targets)} ticker-days to classify")
    stats = {"targets": len(targets), "classified": 0}
    if dry_run or not targets:
        return stats
    audit = _cache_dir() / "classified.jsonl"
    for i0 in range(0, len(targets), int(batch)):
        chunk = targets[i0:i0 + int(batch)]
        items = [{"i": i0 + j + 1, "ticker": t["ticker"], "date": t["signal_date"],
                  "headlines": [t["text"]]} for j, t in enumerate(chunk)]
        got = classify_batch(items)
        now = datetime.now(timezone.utc).isoformat()
        rows = [{"ticker": it["ticker"], "signal_date": it["date"],
                 "catalyst": got[it["i"]], "headline_count": 1,
                 "top_headline": it["headlines"][0][:200],
                 "classifier_version": RATIONALE_CLASSIFIER_VERSION,
                 "classified_at": now}
                for it in items if it["i"] in got]
        if rows:
            repo.insert_news_event_backfill(rows)
            with audit.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            stats["classified"] += len(rows)
        logger.info(f"[news_backfill] rationale {min(i0 + len(chunk), len(targets))}"
                    f"/{len(targets)} classified={stats['classified']}")
    logger.info(f"[news_backfill] rationale source done: {stats}")
    return stats


def fetch_ticker_articles(ticker: str, days: list[str],
                          page_delay_s: float) -> Optional[list[dict]]:
    """Headlines covering [min(day)-pad, max(day)] for one ticker, via the
    per-ticker cache. None = fetch produced nothing (skip, retry next run);
    a list (possibly with zero rows near some days) = usable coverage."""
    gte = (date.fromisoformat(min(days)) - timedelta(days=_WINDOW_PAD_DAYS)).isoformat()
    lte = max(days)
    path = _cache_dir() / f"{ticker.replace('/', '_')}.json"
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            if blob.get("gte") <= gte and blob.get("lte") >= lte:
                return blob.get("articles") or []
        except Exception:
            pass
    from src.data import polygon_client
    raw = polygon_client.get_ticker_news_history(
        ticker, f"{gte}T00:00:00Z", f"{lte}T23:59:59Z", page_delay_s=page_delay_s)
    if not raw:
        return None
    arts = [{"t": a.get("published_utc"), "h": (a.get("title") or "")[:200],
             "p": ((a.get("publisher") or {}).get("name") or "")[:40]}
            for a in raw if a.get("title")]
    try:
        path.write_text(json.dumps({"gte": gte, "lte": lte, "articles": arts}),
                        encoding="utf-8")
    except Exception as e:
        logger.debug(f"[news_backfill] cache write failed for {ticker}: {e}")
    return arts


def _et_date(published_utc: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(str(published_utc).replace("Z", "+00:00"))
        return dt.astimezone(ET).date().isoformat()
    except Exception:
        return None


def headlines_for_day(articles: list[dict], day: str) -> list[str]:
    """Same-day + previous-day headlines (ET dates), most recent first; widened
    to `_WINDOW_PAD_DAYS` back when that window is empty."""
    d = date.fromisoformat(day)
    by_date: dict[str, list[tuple[str, str]]] = {}
    for a in articles:
        ad = _et_date(a.get("t"))
        if ad:
            by_date.setdefault(ad, []).append((a.get("t") or "", a.get("h") or ""))
    for back in (1, _WINDOW_PAD_DAYS):
        window = [(t, h) for off in range(back + 1)
                  for t, h in by_date.get((d - timedelta(days=off)).isoformat(), [])]
        if window:
            window.sort(reverse=True)
            return [h for _, h in window[:_HEADLINES_PER_DAY]]
    return []


def classify_batch(items: list[dict], client=None) -> dict[int, str]:
    """items: [{"i", "ticker", "date", "headlines": [...]}] → {i: catalyst}.
    Items the model skipped or mangled are simply absent (retried next run)."""
    if not items:
        return {}
    if client is None:
        from src.analysis import sentiment as _sent
        client = _sent._get_deepseek()
        if client is None:
            raise RuntimeError("DeepSeek key not configured — classifier unavailable")
    from src.analysis.sentiment import DEEPSEEK_MODEL, _DEEPSEEK_THINKING_OFF, _LLM_SEED
    lines = []
    for it in items:
        lines.append(f"#{it['i']} TICKER={it['ticker']} DATE={it['date']}")
        lines.extend(f"- {h}" for h in it["headlines"])
    prompt = _CLASSIFY_PROMPT.format(types=", ".join(NEWS_CATALYST_TYPES),
                                     items="\n".join(lines))
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL, max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, seed=_LLM_SEED, extra_body=_DEEPSEEK_THINKING_OFF,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.startswith("json") else raw
    out: dict[int, str] = {}
    try:
        for entry in json.loads(raw.strip()):
            cat = normalize_catalyst(entry.get("catalyst"))
            if cat is not None:
                out[int(entry["i"])] = cat
    except Exception as e:
        logger.warning(f"[news_backfill] classifier reply unparseable ({e}) — batch retried next run")
    return out


def run(limit_tickers: Optional[int] = None, batch: int = 25,
        ticker_delay_s: float = 13.0, dry_run: bool = False) -> dict:
    _ensure_tables()
    targets = load_targets(limit_tickers)
    n_days = sum(len(v) for v in targets.values())
    logger.info(f"[news_backfill] scope: {len(targets)} tickers / {n_days} ticker-days "
                f"(~{len(targets) * ticker_delay_s / 3600:.1f}h of fetch throttle)")
    if dry_run or not targets:
        return {"tickers": len(targets), "days": n_days, "classified": 0, "no_coverage": 0}

    from src.db import repo
    audit = _cache_dir() / "classified.jsonl"
    stats = {"tickers": len(targets), "days": n_days, "classified": 0,
             "no_coverage": 0, "skipped_tickers": 0}
    pending: list[dict] = []          # classify queue: {"i", "ticker", "date", "headlines"}
    rows_null: list[dict] = []        # attempted-no-coverage rows

    def _flush_classified() -> None:
        nonlocal pending
        if not pending:
            return
        got = classify_batch(pending)
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for it in pending:
            cat = got.get(it["i"])
            if cat is None:
                continue
            rows.append({"ticker": it["ticker"], "signal_date": it["date"],
                         "catalyst": cat, "headline_count": len(it["headlines"]),
                         "top_headline": it["headlines"][0][:200] if it["headlines"] else None,
                         "classifier_version": CLASSIFIER_VERSION, "classified_at": now})
        if rows:
            repo.insert_news_event_backfill(rows)
            with audit.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            stats["classified"] += len(rows)
        pending = []

    def _flush_null() -> None:
        nonlocal rows_null
        if rows_null:
            repo.insert_news_event_backfill(rows_null)
            stats["no_coverage"] += len(rows_null)
            rows_null = []

    seq = 0
    for n_done, (ticker, days) in enumerate(targets.items(), 1):
        arts = fetch_ticker_articles(ticker, days, page_delay_s=ticker_delay_s)
        if arts is None:
            stats["skipped_tickers"] += 1
            logger.info(f"[news_backfill] {n_done}/{len(targets)} {ticker}: "
                        f"no articles returned — skipped (retried next run)")
            time.sleep(ticker_delay_s)
            continue
        now = datetime.now(timezone.utc).isoformat()
        n_typed = 0
        for day in days:
            heads = headlines_for_day(arts, day)
            if not heads:
                rows_null.append({"ticker": ticker, "signal_date": day, "catalyst": None,
                                  "headline_count": 0, "top_headline": None,
                                  "classifier_version": CLASSIFIER_VERSION,
                                  "classified_at": now})
                continue
            seq += 1
            n_typed += 1
            pending.append({"i": seq, "ticker": ticker, "date": day, "headlines": heads})
            if len(pending) >= batch:
                _flush_classified()
        _flush_null()
        logger.info(f"[news_backfill] {n_done}/{len(targets)} {ticker}: "
                    f"{len(days)} days, {len(arts)} articles, {n_typed} queued "
                    f"(classified so far: {stats['classified']})")
        time.sleep(ticker_delay_s)
    _flush_classified()
    _flush_null()
    logger.info(f"[news_backfill] done: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill historical news catalyst types")
    ap.add_argument("--source", choices=("all", "rationale", "polygon"), default="all",
                    help="rationale = owned run-time text (minutes); polygon = "
                         "re-fetched headlines for the remainder (hours)")
    ap.add_argument("--limit-tickers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--ticker-delay", type=float, default=13.0,
                    help="seconds between Polygon calls (free tier: 5/min shared)")
    ap.add_argument("--dry-run", action="store_true", help="print scope, make no calls")
    args = ap.parse_args()
    stats: dict = {}
    if args.source in ("all", "rationale"):
        stats["rationale"] = run_rationales(batch=args.batch, dry_run=args.dry_run)
    if args.source in ("all", "polygon"):
        stats["polygon"] = run(limit_tickers=args.limit_tickers, batch=args.batch,
                               ticker_delay_s=args.ticker_delay, dry_run=args.dry_run)
    print(stats)


if __name__ == "__main__":
    main()
