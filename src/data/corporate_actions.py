"""
Corporate actions — upcoming ex-dividends + recent/upcoming stock splits.

Source: Massive/Polygon dividends & splits calendars (market-wide, date-filtered —
two paginated calls), filtered to the scored universe. Returns a
CorporateActionsContext fed into the LLM synthesis prompt as a WHEN/mechanics
overlay (instruction §29), never a directional trigger:
  * ex-dividend date → price drops by ~the dividend (not weakness; mild income support),
  * stock split → price / share-count rescale (OHLCV signals around it can mislead).

Cached daily. Fail-graceful: any error / no entitlement → None (run continues).
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

from loguru import logger

from config import settings
from src.data import polygon_client
from src.models import CorporateActionsContext, DividendEvent, SplitEvent

CACHE_DIR = Path("cache")


def _cache_path() -> Path:
    return CACHE_DIR / f"corporate_actions_{date.today().isoformat()}.json"


def _pdate(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _split_ratio(frm: float, to: float) -> str:
    """'3:2' for a 2→3 forward split, '1:10' for a 10→1 reverse split."""
    try:
        return f"{int(to)}:{int(frm)}" if frm and to else ""
    except (TypeError, ValueError):
        return ""


def fetch_corporate_actions_context(tickers: List[str]) -> Optional[CorporateActionsContext]:
    """Upcoming ex-dividends + nearby splits for ``tickers`` → CorporateActionsContext.

    Daily-cached. Returns None when disabled, when fetching is off / Polygon is
    unavailable, or when the universe has no upcoming dividends/splits."""
    if not settings.enable_corporate_actions:
        return None

    path = _cache_path()
    if path.exists():
        try:
            return CorporateActionsContext.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[corp_actions] cache load failed: {e}")

    if not settings.enable_fetch_data or not polygon_client.is_available():
        return None

    universe = {t.upper() for t in tickers}
    today = date.today()
    div_end = today + timedelta(days=settings.corp_actions_div_lookahead_days)
    split_lo = today - timedelta(days=settings.corp_actions_split_window_days)
    split_hi = today + timedelta(days=settings.corp_actions_split_window_days)

    dividends: List[DividendEvent] = []
    for r in polygon_client.get_dividends_calendar(today.isoformat(), div_end.isoformat()):
        tk = (r.get("ticker") or "").upper()
        exd = _pdate(r.get("ex_dividend_date"))
        amt = r.get("cash_amount")
        if tk in universe and exd is not None and amt is not None:
            dividends.append(DividendEvent(
                ticker=tk, ex_dividend_date=exd, cash_amount=float(amt),
                frequency=int(r.get("frequency") or 0), pay_date=_pdate(r.get("pay_date")),
                days_until_ex=(exd - today).days,
            ))

    splits: List[SplitEvent] = []
    for r in polygon_client.get_splits_calendar(split_lo.isoformat(), split_hi.isoformat()):
        tk = (r.get("ticker") or "").upper()
        exe = _pdate(r.get("execution_date"))
        frm, to = r.get("split_from"), r.get("split_to")
        if tk in universe and exe is not None and frm and to:
            splits.append(SplitEvent(
                ticker=tk, execution_date=exe, split_from=float(frm), split_to=float(to),
                ratio=_split_ratio(frm, to), days_until=(exe - today).days,
            ))

    if not dividends and not splits:
        logger.info("[corp_actions] no upcoming dividends/splits in the universe")
        return None

    dividends.sort(key=lambda d: d.days_until_ex)
    splits.sort(key=lambda s: abs(s.days_until))
    ctx = CorporateActionsContext(
        dividends=dividends, splits=splits, report_date=today,
        summary=(f"{len(dividends)} upcoming ex-dividend(s) (≤{settings.corp_actions_div_lookahead_days}d) "
                 f"and {len(splits)} split(s) (±{settings.corp_actions_split_window_days}d) in the universe."),
    )

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[corp_actions] cache save failed: {e}")

    logger.info(f"[corp_actions] {len(dividends)} ex-div, {len(splits)} split(s) in universe")
    return ctx
