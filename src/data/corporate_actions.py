"""
Corporate actions — upcoming ex-dividends + recent/upcoming stock splits.

Source: Massive/Polygon dividends & splits calendars (market-wide, date-filtered —
paginated calls), filtered to the scored universe. Returns a
CorporateActionsContext fed into the LLM synthesis prompt as a WHEN/mechanics
overlay (instruction §29):
  * ex-dividend date → price drops by ~the dividend (not weakness; mild income support),
  * stock split → price / share-count rescale (OHLCV signals around it can mislead).

Plus the two directional factor scores (additive combine overlay + panel IC):
  * f_split — forward-split drift / reverse-split distress;
  * f_dividend — EVENT-windowed, RAISE-ONLY dividend-change score (2026-08-16
    rework, epoch "f_dividend"): declared raises drift up for ~the first week
    after DECLARATION, monotone in raise size; cuts abstain (their drift did
    not replicate out-of-sample) — see `_dividend_change_score` for the
    measured basis and memory/dividend-factor-rework-2026-08.md for the study.

Cached daily. Fail-graceful: any error / no entitlement → None (run continues).
"""

import json
from datetime import date, timedelta
from math import log, tanh
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config import settings
from src.data import polygon_client
from src.data.market_data import is_valid_ticker
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


def _split_factor(split_evs: List[SplitEvent], window: int) -> Optional[float]:
    """Directional split factor (+ = bullish). Forward split → + (post-split drift
    anomaly); reverse split → − (delisting-distress, the more robust side). Magnitude
    scales with the ratio and decays with distance from the execution date."""
    if not split_evs:
        return None
    latest = min(split_evs, key=lambda s: abs(s.days_until))   # most recent / imminent
    frm, to = latest.split_from, latest.split_to
    if not frm or not to:
        return None
    mag = tanh(log(max(to / frm, frm / to)))                   # 2:1 → 0.6, 10:1 → 0.98
    decay = max(0.3, 1.0 - abs(latest.days_until) / max(window, 1))
    if to > frm:                                               # forward split
        return round(min(1.0, mag * decay), 3)
    return round(max(-1.0, -max(0.5, mag * decay)), 3)         # reverse: stronger negative


def _dividend_change(div_rows: List[dict]) -> Optional[float]:
    """Relative change of the latest REGULAR dividend vs the prior comparable one.

    ``div_rows`` are one ticker's raw rows, NEWEST first. Comparable = both
    regular cash dividends (type CD — specials mixed into the comparison were
    the source of the old factor's fake deep cuts: 18 of 24 panel readings
    ≤ −0.9 decomposed as data artifacts, only 6 as real cuts) with the SAME
    frequency. None when no valid pair exists — which also retires the old
    "single row → +0.4 initiation" heuristic, measured at −0.61% mean pivot
    return / 42.9% win (truncated history masquerading as initiation)."""
    cds = [d for d in div_rows
           if d.get("cash_amount") and (d.get("dividend_type") or "CD") == "CD"]
    if len(cds) < 2:
        return None
    latest = cds[0]
    freq = latest.get("frequency")
    for prior in cds[1:]:
        if prior.get("frequency") == freq:
            p = float(prior["cash_amount"] or 0)
            if p <= 0:
                return None
            return float(latest["cash_amount"]) / p - 1.0
    return None


def _dividend_change_score(chg: Optional[float], days_since_decl: Optional[int],
                           window: int) -> Optional[float]:
    """EVENT-windowed dividend-RAISE score (+ = bullish stock), None = abstain.

    Measured basis (2026-08-16 two-year event study — declaration-date events,
    gated, signed pivot target, 5,804 labeled events;
    memory/dividend-factor-rework-2026-08.md): the RAISE side is monotone and
    replicates across both year-halves — flat +0.28% → raise +0.42% →
    big raise +1.17% (59.4% win) → huge raise +1.98% to the next pivot — and
    the drift is CONFINED to the first ~5 sessions after declaration (d5→d10
    ≈ 0 in every bucket), so the score decays to zero across ``window`` days
    and abstains beyond it (the old factor held a stale reading for a whole
    quarter). CUTS abstain entirely: the big-cut drift did NOT replicate
    (H1 −1.86% / H2 +0.27% — a sign that flips across halves does not ship),
    and small cuts measured positive in both halves — cut events are
    confounded with the distress-bounce mean reversion this system keeps
    measuring, so the honest score is no view.

    tanh(chg × 4): +5% → +0.20, +10% → +0.38, +25% → +0.76, +50% → +0.96.
    """
    if chg is None or days_since_decl is None or days_since_decl < 0 \
            or days_since_decl > window:
        return None
    if chg < 0.01:
        return None                    # cuts + flats: abstain (see docstring)
    decay = 1.0 - days_since_decl / (window + 1.0)
    score = round(min(1.0, tanh(chg * 4.0) * decay), 3)
    return score if score >= 0.05 else None


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

    # Upcoming ex-dividends (narrow window, market-wide) → §29 mechanics overlay.
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

    splits_by_tk: Dict[str, List[SplitEvent]] = {}
    splits: List[SplitEvent] = []
    for r in polygon_client.get_splits_calendar(split_lo.isoformat(), split_hi.isoformat()):
        tk = (r.get("ticker") or "").upper()
        exe = _pdate(r.get("execution_date"))
        frm, to = r.get("split_from"), r.get("split_to")
        if tk in universe and exe is not None and frm and to:
            ev = SplitEvent(ticker=tk, execution_date=exe, split_from=float(frm), split_to=float(to),
                            ratio=_split_ratio(frm, to), days_until=(exe - today).days)
            splits.append(ev)
            splits_by_tk.setdefault(tk, []).append(ev)

    # f_dividend — EVENT discovery (2026-08-16 rework, epoch "f_dividend"): one
    # market-wide declaration-window query finds every dividend DECLARED in the
    # trailing event window (full-universe coverage — the old per-ticker loop
    # reached only the first 60 insertion-order names, most of whose calls the
    # 5/min budget rejected anyway); per-ticker history is fetched only for the
    # handful of fresh in-universe declarers, to find the prior comparable
    # regular payment.
    div_factor_by_tk: Dict[str, float] = {}
    window = int(settings.corp_actions_div_event_window_days)
    decl_lo = today - timedelta(days=window)
    fetches = 0
    for r in polygon_client.get_recent_dividend_declarations(
            decl_lo.isoformat(), today.isoformat()):
        tk = (r.get("ticker") or "").upper()
        decl = _pdate(r.get("declaration_date"))
        if (tk not in universe or tk in div_factor_by_tk or decl is None
                or not r.get("cash_amount")
                or (r.get("dividend_type") or "CD") != "CD"
                or not is_valid_ticker(tk)):
            continue
        if fetches >= int(settings.corp_actions_div_max_event_fetches):
            break
        fetches += 1
        chg = _dividend_change(polygon_client.get_dividend_history(tk, limit=8))
        score = _dividend_change_score(chg, (today - decl).days, window)
        if score is not None:
            div_factor_by_tk[tk] = score

    # Directional factor scores per ticker — consumed by the panel IC, the aggregator
    # overlay, and the §29 synthesis block.
    factor_scores: Dict[str, Dict[str, float]] = {}
    for tk in set(div_factor_by_tk) | set(splits_by_tk):
        f: Dict[str, float] = {}
        sf = _split_factor(splits_by_tk.get(tk, []), settings.corp_actions_split_window_days)
        if sf is not None:
            f["f_split"] = sf
        if tk in div_factor_by_tk:
            f["f_dividend"] = div_factor_by_tk[tk]
        if f:
            factor_scores[tk] = f

    if not dividends and not splits and not factor_scores:
        logger.info("[corp_actions] no dividends/splits in the universe")
        return None

    dividends.sort(key=lambda d: d.days_until_ex)
    splits.sort(key=lambda s: abs(s.days_until))
    ctx = CorporateActionsContext(
        dividends=dividends, splits=splits, factor_scores=factor_scores, report_date=today,
        summary=(f"{len(dividends)} upcoming ex-dividend(s), {len(splits)} split(s), and "
                 f"{len(factor_scores)} ticker(s) with a directional split/dividend signal."),
    )

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[corp_actions] cache save failed: {e}")

    logger.info(f"[corp_actions] {len(dividends)} ex-div, {len(splits)} split(s) in universe")
    return ctx
