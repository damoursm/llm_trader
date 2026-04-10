"""
Fetch macro regime indicators from the St. Louis Fed FRED API.

Series fetched (all public, require a free API key):
  T10Y2Y       — 10Y-2Y Treasury yield spread (daily)         — yield curve
  DFF          — Effective Federal Funds Rate (daily)          — rate regime
  CPIAUCSL     — CPI All Urban (monthly, YoY computed)         — inflation
  UNRATE       — Unemployment rate (monthly)                   — labor market
  BAMLH0A0HYM2 — ICE BofA US High Yield OAS (daily, %)        — credit stress
  BAMLC0A0CM   — ICE BofA US IG Corporate OAS (daily, %)      — credit baseline
  M2SL         — M2 money supply (monthly, YoY computed)       — liquidity

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in .env. If the key is missing or a fetch fails the module
returns None and the pipeline continues without macro context.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import httpx
from loguru import logger

from src.models import MacroContext

_BASE = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 15


def _fetch(series_id: str, api_key: str, limit: int = 13) -> list[dict]:
    """Return the `limit` most-recent observations for a FRED series (newest first)."""
    try:
        resp = httpx.get(
            _BASE,
            params={
                "series_id":  series_id,
                "api_key":    api_key,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      limit,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        logger.warning(f"[fred] {series_id} fetch failed: {e}")
        return []


def _latest(obs: list[dict]) -> Optional[float]:
    """Return the most recent non-missing value from a sorted (newest-first) obs list."""
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                return float(v)
            except ValueError:
                continue
    return None


def _yoy(obs: list[dict]) -> Optional[float]:
    """
    Compute year-over-year percentage change from monthly observations.
    obs must be newest-first; index 0 = current month, index 12 = same month last year.
    Returns None if either endpoint is missing.
    """
    if len(obs) < 13:
        return None
    current = _latest(obs[:1])
    prior   = _latest(obs[12:13])
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def _trend(obs: list[dict], window: int = 4) -> str:
    """
    Classify direction of a series over the last `window` monthly observations.
    Returns 'RISING', 'FALLING', or 'STABLE'.
    """
    values = []
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                values.append(float(v))
            except ValueError:
                pass
        if len(values) >= window:
            break

    if len(values) < 2:
        return "STABLE"

    # Simple linear trend: compare first half avg to second half avg
    mid   = len(values) // 2
    newer = sum(values[:mid]) / mid          # more recent (obs is newest-first)
    older = sum(values[mid:]) / (len(values) - mid)
    diff  = newer - older
    threshold = abs(older) * 0.02            # 2% move required to call a trend

    if diff > threshold:
        return "RISING"
    elif diff < -threshold:
        return "FALLING"
    return "STABLE"


# ---------------------------------------------------------------------------
# Regime classification helpers
# ---------------------------------------------------------------------------

def _yield_curve_label(spread: Optional[float]) -> str:
    if spread is None:
        return "UNKNOWN"
    if spread < -0.25:
        return "INVERTED"
    if spread < 0.10:
        return "FLAT"
    if spread < 1.00:
        return "NORMAL"
    return "STEEP"


def _inflation_label(cpi_yoy: Optional[float]) -> str:
    if cpi_yoy is None:
        return "UNKNOWN"
    if cpi_yoy > 5.0:
        return "HIGH"
    if cpi_yoy > 3.0:
        return "ELEVATED"
    if cpi_yoy > 2.0:
        return "MODERATE"
    return "LOW"


def _credit_label(hy_spread: Optional[float]) -> str:
    """HY OAS in percentage points (FRED stores it this way)."""
    if hy_spread is None:
        return "UNKNOWN"
    if hy_spread > 6.0:
        return "STRESSED"
    if hy_spread > 4.5:
        return "ELEVATED"
    if hy_spread > 3.0:
        return "NORMAL"
    return "TIGHT"


def _regime(
    yield_curve: str,
    credit: str,
    unemployment_trend: str,
    inflation: str,
) -> str:
    """
    Overall macro regime:
      RECESSION   — inverted curve + rising unemployment
      LATE_CYCLE  — inverted/flat curve + stressed credit
      SLOWDOWN    — normal curve + elevated credit or rising unemployment
      EXPANSION   — normal/steep curve + tight/normal credit + stable employment
    """
    if yield_curve == "INVERTED" and unemployment_trend == "RISING":
        return "RECESSION"
    if yield_curve in ("INVERTED", "FLAT") and credit in ("STRESSED", "ELEVATED"):
        return "LATE_CYCLE"
    if credit in ("STRESSED", "ELEVATED") or unemployment_trend == "RISING":
        return "SLOWDOWN"
    return "EXPANSION"


def _build_summary(ctx: MacroContext) -> str:
    parts = []

    # Yield curve
    if ctx.yield_spread_10y2y is not None:
        sign = "+" if ctx.yield_spread_10y2y >= 0 else ""
        parts.append(
            f"Yield curve is {ctx.yield_curve_signal} "
            f"(10Y-2Y spread: {sign}{ctx.yield_spread_10y2y:.2f}%)."
        )

    # Fed funds rate
    if ctx.fed_funds_rate is not None:
        parts.append(f"Fed Funds Rate: {ctx.fed_funds_rate:.2f}%.")

    # Inflation
    if ctx.cpi_yoy is not None:
        parts.append(
            f"CPI inflation {ctx.cpi_yoy:+.1f}% YoY — {ctx.inflation_signal.lower()} inflation regime."
        )

    # Labor
    if ctx.unemployment_rate is not None:
        parts.append(
            f"Unemployment: {ctx.unemployment_rate:.1f}% and {ctx.unemployment_trend.lower()}."
        )

    # Credit
    if ctx.hy_spread is not None:
        parts.append(
            f"HY credit spread: {ctx.hy_spread:.2f}% ({ctx.credit_signal.lower()} credit environment)."
        )

    # M2
    if ctx.m2_growth_yoy is not None:
        liquidity = "expanding" if ctx.m2_growth_yoy > 2 else ("contracting" if ctx.m2_growth_yoy < -2 else "flat")
        parts.append(f"M2 money supply growth: {ctx.m2_growth_yoy:+.1f}% YoY ({liquidity} liquidity).")

    parts.append(f"Overall macro regime: {ctx.regime}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_macro_context(api_key: str) -> Optional[MacroContext]:
    """
    Fetch all FRED series and return a MacroContext summary.
    Returns None if the API key is missing or all fetches fail.
    """
    if not api_key:
        logger.info("[fred] FRED_API_KEY not set — skipping macro context")
        return None

    logger.info("[fred] Fetching macro indicators from FRED...")

    # Yield curve: T10Y2Y (daily) — 2 obs sufficient for current value
    obs_t10y2y = _fetch("T10Y2Y",       api_key, limit=5)
    # Fed funds rate (daily)
    obs_dff    = _fetch("DFF",           api_key, limit=5)
    # CPI (monthly) — need 13 for YoY
    obs_cpi    = _fetch("CPIAUCSL",      api_key, limit=13)
    # Unemployment (monthly) — need several for trend
    obs_urate  = _fetch("UNRATE",        api_key, limit=8)
    # HY credit spread (daily)
    obs_hy     = _fetch("BAMLH0A0HYM2",  api_key, limit=5)
    # IG credit spread (daily)
    obs_ig     = _fetch("BAMLC0A0CM",    api_key, limit=5)
    # M2 (monthly) — need 13 for YoY
    obs_m2     = _fetch("M2SL",          api_key, limit=13)

    if not any([obs_t10y2y, obs_dff, obs_cpi, obs_urate, obs_hy]):
        logger.warning("[fred] All FRED fetches returned empty — skipping macro context")
        return None

    yield_spread   = _latest(obs_t10y2y)
    fed_funds      = _latest(obs_dff)
    cpi_yoy        = _yoy(obs_cpi)
    unemployment   = _latest(obs_urate)
    unemp_trend    = _trend(obs_urate, window=4)
    hy_spread      = _latest(obs_hy)
    ig_spread      = _latest(obs_ig)
    m2_yoy         = _yoy(obs_m2)

    yield_curve    = _yield_curve_label(yield_spread)
    inflation      = _inflation_label(cpi_yoy)
    credit         = _credit_label(hy_spread)
    macro_regime   = _regime(yield_curve, credit, unemp_trend, inflation)

    ctx = MacroContext(
        yield_spread_10y2y  = yield_spread,
        yield_curve_signal  = yield_curve,
        fed_funds_rate      = fed_funds,
        cpi_yoy             = cpi_yoy,
        inflation_signal    = inflation,
        unemployment_rate   = unemployment,
        unemployment_trend  = unemp_trend,
        hy_spread           = hy_spread,
        ig_spread           = ig_spread,
        credit_signal       = credit,
        m2_growth_yoy       = m2_yoy,
        regime              = macro_regime,
        summary             = "",   # populated after construction
    )
    ctx = ctx.model_copy(update={"summary": _build_summary(ctx)})

    logger.info(
        f"[fred] Regime: {macro_regime} | "
        f"Curve: {yield_curve} ({yield_spread:+.2f}% spread) | "
        f"Credit: {credit} (HY {hy_spread:.2f}%) | "
        f"CPI YoY: {cpi_yoy:+.1f}% | "
        f"Unemployment: {unemployment:.1f}% ({unemp_trend})"
    )
    return ctx
