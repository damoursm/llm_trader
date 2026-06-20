"""Quiver Quantitative alternative-data client (Hobbyist tier).

Datasets wired into the pipeline:
  • Congress trading      → List[InsiderTrade] (politician trades). Revives the
    congressional smart-money signal that died when the House/Senate Stock Watcher
    S3 buckets went 403 — flows into the existing insider_score + universe discovery.
  • Government contracts   → List[NewsArticle] (federal contract awards = revenue catalyst).
  • Corporate lobbying     → List[NewsArticle] (lobbying spend = regulatory-attention context).
  • Off-exchange / dark pool (DPI) → List[NewsArticle] (per-ticker institutional
    accumulation/distribution), emitted only on a meaningful shift.

The non-congress datasets are rendered as **synthetic NewsArticles** — the same
idiom the system already uses for Google Trends, Reddit, short interest, analyst
ratings and EPS surprises (all List[NewsArticle], scored by the sentiment pipeline
and surfaced in the synthesis prompt). Congress uses InsiderTrade because it IS a
smart-money trade, identical in shape to the old congressional feed.

All sources are key-gated (QUIVER_API_KEY) and fail-soft (return []). Raw API
responses are cached daily. Endpoints/auth follow the open-source `quiverquant`
client (base /beta/, ``Authorization: Token``). Field access is case-insensitive
with fallbacks — VERIFY the exact field names against your live key (built without
one, like the Finnhub integration).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from loguru import logger

from config import settings
from src.models import InsiderTrade, NewsArticle

_BASE = "https://api.quiverquant.com/beta"
_CACHE_DIR = Path("cache")
_TIMEOUT = 30.0


def is_available() -> bool:
    return bool(settings.quiver_api_key)


def _headers() -> dict:
    # The quiverquant client uses a Token auth header (not Bearer).
    return {"accept": "application/json", "Authorization": f"Token {settings.quiver_api_key}"}


def _cache_path(path: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")
    return _CACHE_DIR / f"quiver_{slug}_{date.today().isoformat()}.json"


def _get(path: str) -> list:
    """GET a Quiver endpoint → list of row dicts. Daily-cached (raw JSON); empty
    on any failure (NOT cached, so the next tick retries)."""
    if not is_available():
        return []
    cache = _cache_path(path)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        r = httpx.get(f"{_BASE}{path}", headers=_headers(), timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[quiver] GET {path} failed: {e}")
        return []
    if not isinstance(data, list):
        logger.warning(f"[quiver] GET {path}: unexpected payload type {type(data).__name__}")
        return []
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        cache.write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception:
        pass
    return data


def _g(row: dict, *keys, default=None):
    """Case-insensitive field access with fallbacks (Quiver field casing varies)."""
    low = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        v = low.get(str(k).lower())
        if v not in (None, ""):
            return v
    return default


def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    txt = str(s)[:10]
    try:
        return date.fromisoformat(txt)
    except (ValueError, TypeError):
        return None


def _amount(val) -> Optional[float]:
    """Parse a numeric amount from a string/number (strips $ and commas)."""
    if val is None:
        return None
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(val)) or 0) or None
    except (ValueError, TypeError):
        return None


def _fmt_usd(v: Optional[float]) -> str:
    return f"${v:,.0f}" if isinstance(v, (int, float)) and v else "an undisclosed amount"


# ── Congress trading → smart money (InsiderTrade) ─────────────────────────────

def fetch_congress_trades(tickers: Optional[List[str]] = None) -> List[InsiderTrade]:
    """Recent congressional trades (one live call). NOT filtered to the existing
    universe — like the old Stock-Watcher feed it returns market-wide trades so
    smart-money discovery can surface new tickers (then the liquidity gate filters
    them). Optionally narrowed to ``tracked_politicians_list`` when configured.
    Mapped to InsiderTrade(trader_type="politician") so it flows through the
    existing insider-score + discovery path exactly like the old feed did."""
    if not (settings.enable_quiver_congress and is_available()):
        return []
    rows = _get("/live/congresstrading")
    cutoff = date.today() - timedelta(days=settings.quiver_lookback_days)
    tracked = [p.lower() for p in (settings.tracked_politicians_list or [])]
    out: List[InsiderTrade] = []
    for r in rows or []:
        tk = str(_g(r, "Ticker", "ticker") or "").upper().strip()
        if not tk:
            continue
        txn = _parse_date(_g(r, "TransactionDate", "Transaction Date", "Traded"))
        if txn is None or txn < cutoff:
            continue
        rep = str(_g(r, "Representative", "Name", "Senator") or "Member of Congress")
        if tracked and not any(t in rep.lower() for t in tracked):
            continue
        raw_txn = str(_g(r, "Transaction", "transaction") or "").lower()
        if "purchase" in raw_txn or "buy" in raw_txn:
            tx_type = "purchase"
        elif "sale" in raw_txn or "sell" in raw_txn:
            tx_type = "sale"
        elif "exchange" in raw_txn:
            tx_type = "exchange"
        else:
            tx_type = raw_txn.replace(" ", "_") or "purchase"
        chamber = str(_g(r, "House", "Chamber") or "").lower()
        role = "Senator" if "senate" in chamber else "Representative"
        out.append(InsiderTrade(
            ticker=tk,
            trader_name=rep[:80],
            trader_type="politician",
            role=role,
            transaction_type=tx_type,
            amount_range=str(_g(r, "Range", "Amount", default="see filing")),
            transaction_date=txn,
            disclosure_date=_parse_date(_g(r, "ReportDate", "Report Date", "Filed")) or txn,
            notes=f"Congress ({role}) {raw_txn or tx_type} — {rep}",
        ))
    n_tk = len({t.ticker for t in out})
    logger.info(f"[quiver] Congress: {len(out)} trade(s) across {n_tk} ticker(s)")
    return out


# ── Government contracts → NewsArticle (revenue catalyst) ──────────────────────

def fetch_gov_contracts(tickers: List[str]) -> List[NewsArticle]:
    if not (settings.enable_quiver_gov_contracts and is_available()):
        return []
    rows = _get("/live/govcontractsall")
    universe = {t.upper() for t in (tickers or [])}
    cutoff = date.today() - timedelta(days=settings.quiver_lookback_days)
    out: List[NewsArticle] = []
    for r in rows or []:
        tk = str(_g(r, "Ticker", "ticker") or "").upper().strip()
        if not tk or (universe and tk not in universe):
            continue
        d = _parse_date(_g(r, "Date", "date"))
        if d is None or d < cutoff:
            continue
        amt = _amount(_g(r, "Amount", "amount"))
        agency = str(_g(r, "Agency", "agency") or "a federal agency")
        title = f"{tk} awarded {_fmt_usd(amt)} U.S. government contract ({agency})"
        out.append(NewsArticle(
            title=title,
            summary=(f"Quiver: {tk} received a U.S. federal government contract from {agency} "
                     f"for {_fmt_usd(amt)} (awarded {d.isoformat()}). Federal awards add backlog/"
                     f"revenue visibility — a fundamental positive catalyst for the awardee."),
            url="https://www.quiverquant.com/sources/govcontracts",
            source="Quiver Gov Contracts",
            published_at=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
            tickers=[tk],
        ))
    logger.info(f"[quiver] Gov contracts: {len(out)} award(s) for universe tickers")
    return out


# ── Corporate lobbying → NewsArticle (context) ────────────────────────────────

def fetch_lobbying(tickers: List[str]) -> List[NewsArticle]:
    if not (settings.enable_quiver_lobbying and is_available()):
        return []
    rows = _get("/live/lobbying")
    universe = {t.upper() for t in (tickers or [])}
    cutoff = date.today() - timedelta(days=settings.quiver_lookback_days)
    out: List[NewsArticle] = []
    for r in rows or []:
        tk = str(_g(r, "Ticker", "ticker") or "").upper().strip()
        if not tk or (universe and tk not in universe):
            continue
        d = _parse_date(_g(r, "Date", "date"))
        if d is None or d < cutoff:
            continue
        amt = _amount(_g(r, "Amount", "amount"))
        issue = str(_g(r, "Issue", "Specific_Issue", "Client") or "policy matters")
        out.append(NewsArticle(
            title=f"{tk} disclosed {_fmt_usd(amt)} in federal lobbying ({issue[:60]})",
            summary=(f"Quiver: {tk} reported {_fmt_usd(amt)} of federal lobbying spend on "
                     f"{issue[:200]} ({d.isoformat()}). Lobbying activity signals regulatory "
                     f"exposure/attention — context, not a standalone directional catalyst."),
            url="https://www.quiverquant.com/sources/lobbying",
            source="Quiver Lobbying",
            published_at=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
            tickers=[tk],
        ))
    logger.info(f"[quiver] Lobbying: {len(out)} disclosure(s) for universe tickers")
    return out


# ── Off-exchange / dark pool (DPI) → NewsArticle (per-ticker, on a shift) ──────

_DPI_MIN_BARS = 10        # need history to judge a shift
_DPI_SHIFT = 0.06         # recent-vs-baseline DPI delta to flag (≈6 percentage pts)


def fetch_offexchange(tickers: List[str]) -> List[NewsArticle]:
    """Per-ticker dark-pool index (DPI = off-exchange share of volume). Emits an
    article only when the recent DPI departs meaningfully from its baseline —
    elevated = institutional dark accumulation, depressed = distribution. Bounded +
    daily-cached so the per-ticker loop stays within Hobbyist rate limits."""
    if not (settings.enable_quiver_offexchange and is_available()):
        return []
    out: List[NewsArticle] = []
    for tk in [t.upper() for t in (tickers or [])][: settings.quiver_offexchange_max_tickers]:
        rows = _get(f"/historical/offexchange/{tk}")
        if not rows or len(rows) < _DPI_MIN_BARS:
            continue
        # Quiver returns rows NEWEST-FIRST; sort by date so "recent" is genuinely
        # the latest bars regardless of API order. DPI = off-exchange share (0..1).
        pts = []
        for r in rows:
            dt = _parse_date(_g(r, "Date", "date"))
            dpi = _g(r, "DPI", "dpi", "OffExchangeVolume", "Off_Exchange")
            try:
                if dt is not None:
                    pts.append((dt, float(dpi)))
            except (TypeError, ValueError):
                continue
        if len(pts) < _DPI_MIN_BARS:
            continue
        pts.sort(key=lambda x: x[0])
        series = [v for _, v in pts]
        recent = sum(series[-3:]) / 3
        baseline = sum(series[-_DPI_MIN_BARS:-3]) / max(1, len(series[-_DPI_MIN_BARS:-3]))
        shift = recent - baseline
        if abs(shift) < _DPI_SHIFT:
            continue
        direction = "accumulation" if shift > 0 else "distribution"
        d = pts[-1][0]
        out.append(NewsArticle(
            title=f"{tk} off-exchange (dark-pool) {direction}: DPI {recent*100:.0f}% vs {baseline*100:.0f}% baseline",
            summary=(f"Quiver dark-pool index for {tk} averaged {recent*100:.0f}% of volume off-exchange "
                     f"recently vs a {baseline*100:.0f}% baseline (Δ{shift*100:+.0f}pp, as of {d.isoformat()}). "
                     f"A rising off-exchange share can indicate institutional {direction}; it is a SOFT, "
                     f"positioning signal — corroborating, not a standalone thesis."),
            url=f"https://www.quiverquant.com/sources/offexchange/{tk}",
            source="Quiver Dark Pool",
            published_at=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
            tickers=[tk],
        ))
        time.sleep(0.1)   # gentle pacing for the per-ticker loop
    logger.info(f"[quiver] Off-exchange: {len(out)} dark-pool shift article(s)")
    return out
