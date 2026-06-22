"""
Company fundamentals — valuation, profitability, and leverage ratios.

Source: Massive/Polygon financials & ratios endpoint (``/stocks/financials/v1/ratios``)
— trailing-twelve-month ratios recomputed daily from SEC-filed statements + the
latest close. Requires the Stocks Advanced plan (or the ratios add-on); the free
tier 403s, in which case this degrades to None and the run proceeds without a
fundamentals block.

Returns a ``FundamentalsContext`` fed into the LLM synthesis prompt so the
recommendations weigh valuation / quality alongside the price / news / options
signals. This is a slow-moving QUALITY OVERLAY, not a per-tick timing signal and
not an aggregator scorer.

Cached daily — ratios only update once per trading day.
"""

import json
from datetime import date
from pathlib import Path
from typing import List, Optional

from loguru import logger

from config import settings
from src.data import polygon_client
from src.models import FundamentalsContext, FundamentalsSignal

CACHE_DIR = Path("cache")


def _cache_path() -> Path:
    return CACHE_DIR / f"fundamentals_{date.today().isoformat()}.json"


def _f(row: dict, key: str) -> Optional[float]:
    """Coerce a ratios field to float; None when absent / non-numeric."""
    v = row.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _build_signal(ticker: str, row: dict) -> FundamentalsSignal:
    """Map one ratios row → FundamentalsSignal with a compact human summary."""
    pe   = _f(row, "price_to_earnings")
    pb   = _f(row, "price_to_book")
    ev   = _f(row, "ev_to_ebitda")
    roe  = _f(row, "return_on_equity")
    de   = _f(row, "debt_to_equity")
    dy   = _f(row, "dividend_yield")

    parts: List[str] = []
    if pe is not None:  parts.append(f"P/E {pe:.1f}")
    if pb is not None:  parts.append(f"P/B {pb:.1f}")
    if ev is not None:  parts.append(f"EV/EBITDA {ev:.1f}")
    if roe is not None: parts.append(f"ROE {roe * 100:.0f}%")
    if de is not None:  parts.append(f"D/E {de:.2f}")
    if dy:              parts.append(f"yield {dy * 100:.1f}%")
    summary = f"{ticker}: " + (", ".join(parts) if parts else "no ratios available")

    as_of: Optional[date] = None
    raw_date = row.get("date")
    if raw_date:
        try:
            as_of = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            as_of = None

    return FundamentalsSignal(
        ticker=ticker,
        pe=pe, pb=pb, ps=_f(row, "price_to_sales"), ev_ebitda=ev,
        roe=roe, roa=_f(row, "return_on_assets"), debt_to_equity=de,
        dividend_yield=dy, current_ratio=_f(row, "current"),
        fcf=_f(row, "free_cash_flow"), market_cap=_f(row, "market_cap"),
        enterprise_value=_f(row, "enterprise_value"), as_of=as_of, summary=summary,
    )


def fetch_fundamentals_context(tickers: List[str]) -> Optional[FundamentalsContext]:
    """Fetch TTM valuation/quality ratios for ``tickers`` → FundamentalsContext.

    Daily-cached. Returns ``None`` when the feature is disabled, when fetching is
    off, when Polygon is unavailable, or when the endpoint returns nothing (free
    tier 403 / no coverage) — in every case the synthesis simply runs without a
    fundamentals block.
    """
    if not settings.enable_fundamentals:
        return None

    path = _cache_path()
    if path.exists():
        try:
            ctx = FundamentalsContext.model_validate_json(path.read_text(encoding="utf-8"))
            logger.info(f"[fundamentals] loaded {len(ctx.signals)} cached ratios")
            return ctx
        except Exception as e:
            logger.warning(f"[fundamentals] cache load failed: {e}")

    if not settings.enable_fetch_data or not polygon_client.is_available():
        return None

    rows = polygon_client.get_ratios_batch(list(dict.fromkeys(tickers)))
    if not rows:
        logger.info("[fundamentals] no ratios returned (entitlement / coverage) — skipping")
        return None

    signals = sorted((_build_signal(tk, r) for tk, r in rows.items()),
                     key=lambda s: s.ticker)
    ctx = FundamentalsContext(
        signals=signals,
        report_date=date.today(),
        summary=f"TTM valuation/quality ratios for {len(signals)} tickers (Massive/Polygon).",
    )

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[fundamentals] cache save failed: {e}")

    logger.info(f"[fundamentals] ratios for {len(signals)}/{len(set(tickers))} tickers")
    return ctx
