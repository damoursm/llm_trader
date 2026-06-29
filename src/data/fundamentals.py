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


def _enrich_signal(sig: FundamentalsSignal, ratios_row: dict) -> None:
    """Add positioning (short interest/volume) + growth/quality (latest income
    statement) to a signal, and extend its prompt summary. Best-effort per field —
    the polygon fetchers return None/[] on any miss, so this never raises."""
    tk = sig.ticker
    mc, px = ratios_row.get("market_cap"), ratios_row.get("price")
    shares = (mc / px) if mc and px else None   # float endpoint isn't on plan → implied shares

    si = polygon_client.get_short_interest(tk)
    if si:
        if shares and si.get("short_interest"):
            sig.short_pct = round(si["short_interest"] / shares * 100, 2)
        try:
            sig.days_to_cover = round(float(si["days_to_cover"]), 1) if si.get("days_to_cover") is not None else None
        except (TypeError, ValueError):
            pass

    sv = polygon_client.get_short_volume(tk)
    if sv and sv.get("short_volume_ratio") is not None:
        try:
            sig.short_volume_ratio = round(float(sv["short_volume_ratio"]), 1)
        except (TypeError, ValueError):
            pass

    inc = polygon_client.get_income_statements(tk, limit=6)
    if inc:
        r0 = inc[0]
        rev = r0.get("revenue")
        ni = r0.get("consolidated_net_income_loss")
        if rev and ni is not None:
            sig.net_margin = round(ni / rev * 100, 1)
        yago = next((r for r in inc[1:]
                     if r.get("fiscal_quarter") == r0.get("fiscal_quarter") and r.get("revenue")), None)
        if rev and yago:
            sig.rev_growth_yoy = round((rev - yago["revenue"]) / yago["revenue"] * 100, 1)

    extra = []
    if sig.short_pct is not None:
        extra.append(f"short {sig.short_pct}%" + (f"/{sig.days_to_cover}d" if sig.days_to_cover else ""))
    if sig.rev_growth_yoy is not None:
        extra.append(f"rev {sig.rev_growth_yoy:+.0f}% YoY")
    if sig.net_margin is not None:
        extra.append(f"net margin {sig.net_margin:.0f}%")
    if extra:
        sig.summary += " | " + ", ".join(extra)


def _c(x: float) -> float:
    return max(-1.0, min(1.0, x))


def factor_scores(fs: FundamentalsSignal) -> dict:
    """Signed [-1, +1] factor scores from a FundamentalsSignal (+ = hypothesized bullish).

    Sign follows the universal convention (see ``aggregator`` docstring): + = the
    factor's hypothesised UP direction (cheap value / high quality / high growth /
    crowded short → squeeze), − = down; magnitude (tanh-scaled) = how extreme the
    factor is. The Signal-IC table is what reveals whether each actually predicts
    forward returns. Persisted as the ``f_*`` signals columns for IC/Sim-win/Sim-ret
    AND folded into combined_score as a small additive overlay
    (``fundamental_factor_weight``, applied OUTSIDE the normalised weight pool) — but
    NOT in ``_ALL_METHODS`` / the solo perf tables. A missing input omits that factor
    (no fake 0 that would dilute its IC)."""
    from math import tanh
    out: dict = {}

    # VALUE — cheap (high earnings / book yield) hypothesised to outperform.
    vparts = []
    if fs.pe and fs.pe > 0:
        vparts.append(tanh((1.0 / fs.pe - 0.05) * 20))     # earnings-yield pivot ~5% (P/E 20)
    if fs.pb and fs.pb > 0:
        vparts.append(tanh((1.0 / fs.pb - 0.15) * 4))      # book-yield pivot ~0.15 (P/B ~6.7)
    if vparts:
        out["f_value"] = round(_c(sum(vparts) / len(vparts)), 3)

    # QUALITY — high ROE / net margin.
    qparts = []
    if fs.roe is not None:
        qparts.append(tanh(fs.roe * 2.0))                  # ROE 0.30 → 0.54
    if fs.net_margin is not None:
        qparts.append(tanh(fs.net_margin / 25.0))          # 25% → 0.76
    if qparts:
        out["f_quality"] = round(_c(sum(qparts) / len(qparts)), 3)

    # GROWTH — high YoY revenue growth.
    if fs.rev_growth_yoy is not None:
        out["f_growth"] = round(_c(tanh(fs.rev_growth_yoy / 25.0)), 3)

    # SHORT SQUEEZE — crowded short (high short% + days-to-cover); the IC reveals
    # whether that precedes up-moves (squeeze) or down-moves (shorts right).
    sparts = []
    if fs.short_pct is not None:
        sparts.append(tanh(fs.short_pct / 10.0))           # 10% → 0.76
    if fs.days_to_cover is not None:
        sparts.append(tanh(fs.days_to_cover / 8.0))        # 8d → 0.76
    if sparts:
        out["f_short_squeeze"] = round(_c(sum(sparts) / len(sparts)), 3)

    return out


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

    sig_by_ticker = {tk: _build_signal(tk, r) for tk, r in rows.items()}

    # Capped enrichment: short interest/volume + statement margin/growth for the
    # first N tickers in ratios/universe order (watchlist + early discovery first),
    # ~3 extra calls each. 0 disables; -1/large = all.
    cap = settings.fundamentals_enrich_max_tickers
    enriched = 0
    if cap != 0:
        for tk in rows:
            if cap > 0 and enriched >= cap:
                break
            try:
                _enrich_signal(sig_by_ticker[tk], rows[tk])
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"[fundamentals] enrich {tk} failed: {e}")
            enriched += 1
        logger.info(f"[fundamentals] enriched {enriched} ticker(s) with short-interest/volume + growth")

    signals = sorted(sig_by_ticker.values(), key=lambda s: s.ticker)
    n_pos = sum(1 for s in signals if s.short_pct is not None or s.rev_growth_yoy is not None)
    ctx = FundamentalsContext(
        signals=signals,
        report_date=date.today(),
        summary=(f"TTM valuation/quality ratios for {len(signals)} tickers "
                 f"({n_pos} with short-interest + revenue growth) — Massive/Polygon."),
    )

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[fundamentals] cache save failed: {e}")

    logger.info(f"[fundamentals] ratios for {len(signals)}/{len(set(tickers))} tickers")
    return ctx
