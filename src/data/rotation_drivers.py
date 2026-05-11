"""
Rotation Drivers — Federal Reserve rate-cycle phase tracker.

Synthesises the Fed's actual rate trajectory (FRED DFF: 3m/12m changes) and the
CPI inflation trend into a named rate-cycle phase, then maps the phase to
cross-asset rotation implications.

  EARLY_TIGHTENING  — Fed recently started hiking; inflation rising; real rate still accommodative
  PEAK_TIGHTENING   — Fed at/near peak rate; inflation elevated; real rate restrictive
  TIGHTENING_PAUSE  — Fed has stopped hiking; holding at peak; CPI moderating
  PIVOT_IMMINENT    — Fed paused + CPI declining + market pricing ≥25bp cuts
  EASING_CYCLE      — Fed actively cutting; risk-on rotation underway
  NEUTRAL           — No clear directional cycle driver

Data: FRED DFF (270 obs ≈ 13m daily) + CPIAUCSL (19 monthly obs). Requires FRED_API_KEY.
Cache: daily — cache/rotation_drivers_YYYY-MM-DD.json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import List, Optional

import httpx
from loguru import logger

from config import settings
from src.models import RotationDriversContext

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "rotation_drivers_"
_TIMEOUT      = 12
_FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

# Asset rotation per cycle phase
_PHASE_ASSETS = {
    "EARLY_TIGHTENING": {
        "favour": ["XLE", "XLF", "XLV"],
        "avoid":  ["XLRE", "XLU", "TLT", "XLK"],
        "narrative": (
            "Early tightening: Fed hiking, inflation rising, real rate still accommodative. "
            "Capital rotating from rate-sensitive growth (XLK, XLRE) into value/energy/financials. "
            "Favour short-duration assets; reduce long-duration/high-PE names."
        ),
    },
    "PEAK_TIGHTENING": {
        "favour": ["XLE", "GLD", "SLV", "XLF"],
        "avoid":  ["XLRE", "XLU", "TLT", "QQQ", "XLK"],
        "narrative": (
            "Peak tightening: Fed near or at terminal rate, real rate restrictive. "
            "Maximum rate headwind for rate-sensitive assets. "
            "Favour inflation hedges (gold, energy) and financials; avoid XLRE, XLU, TLT, and high-PE tech."
        ),
    },
    "TIGHTENING_PAUSE": {
        "favour": ["XLV", "XLP", "GLD"],
        "avoid":  ["XLRE", "XLU"],
        "narrative": (
            "Tightening pause: Fed holding at peak, inflation moderating. Transitional regime. "
            "Capital hasn't yet rotated back to risk-on. "
            "Defensive positioning appropriate (XLV, XLP, gold); avoid rate-sensitive until pivot confirmed."
        ),
    },
    "PIVOT_IMMINENT": {
        "favour": ["TLT", "XLRE", "XLU", "GLD", "IEF"],
        "avoid":  ["XLE", "TBF"],
        "narrative": (
            "Pivot imminent: Fed paused + CPI declining + market pricing 25bp+ of cuts. "
            "Accumulation phase for rate-sensitive assets ahead of the official pivot. "
            "Favour long-duration bonds (TLT), REITs (XLRE), utilities (XLU), and gold."
        ),
    },
    "EASING_CYCLE": {
        "favour": ["XLK", "XLY", "XLC", "QQQ", "XLRE"],
        "avoid":  ["XLP", "XLU"],
        "narrative": (
            "Easing cycle: Fed actively cutting. Risk-on rotation underway. "
            "Capital flowing from defensive/cash into growth (tech, discretionary) and rate-sensitive assets (XLRE). "
            "Reduce cash and defensive sectors; increase growth/cyclical/rate-sensitive exposure."
        ),
    },
    "NEUTRAL": {
        "favour": [],
        "avoid":  [],
        "narrative": (
            "Neutral rate cycle: no clear directional hiking or cutting trend. "
            "Rate-cycle rotation signal is absent — rely on other signal layers."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{date.today().isoformat()}.json"


def _load_cache() -> Optional[RotationDriversContext]:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        ctx = RotationDriversContext.model_validate(
            json.loads(p.read_text(encoding="utf-8"))
        )
        logger.info(
            f"[rotation_drivers] Loaded from cache — "
            f"{ctx.cycle_phase}  {ctx.cycle_direction}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[rotation_drivers] Cache load failed: {e}")
        return None


def _save_cache(ctx: RotationDriversContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[rotation_drivers] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FRED helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fred_series(series_id: str, api_key: str, limit: int) -> list[dict]:
    try:
        resp = httpx.get(
            _FRED_BASE,
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
        logger.warning(f"[rotation_drivers] FRED {series_id} failed: {e}")
        return []


def _latest_value(obs: list[dict]) -> Optional[float]:
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                return float(v)
            except ValueError:
                pass
    return None


def _value_at_index(obs: list[dict], idx: int) -> Optional[float]:
    for o in obs[idx:idx + 5]:
        v = o.get("value", ".")
        if v != ".":
            try:
                return float(v)
            except ValueError:
                pass
    return None


def _yoy(obs: list[dict]) -> Optional[float]:
    """CPI YoY from monthly obs (newest first, index 0=current, 12=year ago)."""
    if len(obs) < 13:
        return None
    cur  = _latest_value(obs[:1])
    prev = _latest_value(obs[12:13])
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / abs(prev) * 100, 2)


def _yoy_at_offset(obs: list[dict], offset: int) -> Optional[float]:
    """CPI YoY computed `offset` months ago."""
    if len(obs) < offset + 13:
        return None
    cur  = _latest_value(obs[offset:offset + 1])
    prev = _latest_value(obs[offset + 12:offset + 13])
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / abs(prev) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rate_trajectory(
    current: float,
    m3: Optional[float],
    m12: Optional[float],
) -> tuple[str, Optional[float], Optional[float]]:
    """Returns (label, change_3m_bp, change_12m_bp)."""
    chg3  = round((current - m3)  * 100, 1) if m3  is not None else None
    chg12 = round((current - m12) * 100, 1) if m12 is not None else None

    if chg12 is not None and chg12 > 75 and chg3 is not None and chg3 > 10:
        return "ACTIVE_HIKING",  chg3, chg12
    if chg12 is not None and chg12 > 25 and chg3 is not None and abs(chg3) <= 15:
        return "PAUSING",        chg3, chg12
    if chg12 is not None and chg12 < -75 and chg3 is not None and chg3 < -10:
        return "ACTIVE_CUTTING", chg3, chg12
    if chg12 is not None and chg12 < -25 and chg3 is not None and abs(chg3) <= 15:
        return "EASING_PAUSE",   chg3, chg12
    if chg12 is not None and chg12 > 25:
        return "ACTIVE_HIKING",  chg3, chg12
    if chg12 is not None and chg12 < -25:
        return "ACTIVE_CUTTING", chg3, chg12
    return "STABLE", chg3, chg12


def _inflation_trend(cpi_now: Optional[float], cpi_6m_ago: Optional[float]) -> str:
    if cpi_now is None:
        return "UNKNOWN"
    if cpi_6m_ago is None:
        return "ELEVATED_STABLE" if cpi_now > 4.0 else ("LOW_STABLE" if cpi_now < 2.5 else "UNKNOWN")
    delta = cpi_now - cpi_6m_ago
    if delta > 1.0:   return "ACCELERATING"
    if delta > 0.4:   return "RISING"
    if delta < -1.5:  return "DECLINING"
    if delta < -0.5:  return "MODERATING"
    if cpi_now > 4.0: return "ELEVATED_STABLE"
    if cpi_now < 2.5: return "LOW_STABLE"
    return "STABLE"


def _real_rate_regime(
    fed_rate: Optional[float],
    cpi_yoy: Optional[float],
) -> tuple[str, Optional[float]]:
    if fed_rate is None or cpi_yoy is None:
        return "UNKNOWN", None
    real = round(fed_rate - cpi_yoy, 2)
    if real > 1.5:  return "HIGHLY_RESTRICTIVE", real
    if real > 0.5:  return "RESTRICTIVE",         real
    if real > -0.5: return "NEUTRAL",              real
    return "ACCOMMODATIVE", real


def _classify_cycle_phase(
    rate_traj: str,
    infl_trend: str,
    real_rate_reg: str,
    implied_cuts_12m_bp: Optional[float],
) -> tuple[str, str]:
    """Returns (cycle_phase, equity_direction)."""
    if rate_traj == "ACTIVE_CUTTING":
        return "EASING_CYCLE", "BULLISH"
    if rate_traj == "EASING_PAUSE":
        return "NEUTRAL", "NEUTRAL"
    if rate_traj == "ACTIVE_HIKING":
        if real_rate_reg in ("RESTRICTIVE", "HIGHLY_RESTRICTIVE"):
            return "PEAK_TIGHTENING", "BEARISH"
        return "EARLY_TIGHTENING", "BEARISH"
    if rate_traj == "PAUSING":
        if infl_trend in ("DECLINING", "MODERATING", "LOW_STABLE"):
            if implied_cuts_12m_bp is not None and implied_cuts_12m_bp >= 25:
                return "PIVOT_IMMINENT", "BULLISH"
            return "TIGHTENING_PAUSE", "NEUTRAL"
        if infl_trend in ("ELEVATED_STABLE", "ACCELERATING", "RISING"):
            return "PEAK_TIGHTENING", "BEARISH"
        return "TIGHTENING_PAUSE", "NEUTRAL"
    # STABLE
    if infl_trend in ("DECLINING", "LOW_STABLE"):
        if implied_cuts_12m_bp is not None and implied_cuts_12m_bp >= 25:
            return "PIVOT_IMMINENT", "BULLISH"
    if infl_trend in ("ACCELERATING", "RISING"):
        return "EARLY_TIGHTENING", "BEARISH"
    return "NEUTRAL", "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rotation_drivers_context(
    fedwatch_context=None,  # Optional[FedWatchContext] — for implied_cuts_12m_bp
) -> Optional[RotationDriversContext]:
    """
    Compute rate-cycle phase from FRED DFF + CPIAUCSL trajectory.
    Optionally accepts the already-fetched FedWatch context to sharpen
    the PIVOT_IMMINENT classification.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    api_key = settings.fred_api_key
    if not api_key:
        logger.debug("[rotation_drivers] FRED_API_KEY not set — skipping")
        return None

    logger.info("[rotation_drivers] Fetching DFF + CPIAUCSL history from FRED...")

    # DFF: daily, ~270 obs ≈ 13 months
    obs_dff = _fred_series("DFF",      api_key, limit=270)
    # CPIAUCSL: monthly, 19 obs for YoY at 0 and 6 months ago
    obs_cpi = _fred_series("CPIAUCSL", api_key, limit=19)

    if not obs_dff and not obs_cpi:
        logger.warning("[rotation_drivers] All FRED fetches failed — skipping")
        return None

    # Fed Funds Rate at four points in time
    fed_current = _latest_value(obs_dff)
    fed_3m_ago  = _value_at_index(obs_dff, 63)
    fed_6m_ago  = _value_at_index(obs_dff, 126)
    fed_12m_ago = _value_at_index(obs_dff, 252)

    rate_traj, chg3, chg12 = _rate_trajectory(
        fed_current or 0.0, fed_3m_ago, fed_12m_ago
    )

    # CPI YoY now and 6 months ago
    cpi_now    = _yoy(obs_cpi)
    cpi_6m_ago = _yoy_at_offset(obs_cpi, 6)

    infl_trend          = _inflation_trend(cpi_now, cpi_6m_ago)
    real_rate_reg, real_val = _real_rate_regime(fed_current, cpi_now)

    # Optional: use FedWatch implied cuts to sharpen pivot detection
    implied_cuts: Optional[float] = None
    if fedwatch_context is not None:
        implied_cuts = getattr(fedwatch_context, "implied_cuts_12m_bp", None)

    cycle_phase, cycle_dir = _classify_cycle_phase(
        rate_traj, infl_trend, real_rate_reg, implied_cuts
    )

    assets = _PHASE_ASSETS.get(cycle_phase, _PHASE_ASSETS["NEUTRAL"])

    # Human-readable summary
    rate_desc = ""
    if fed_current is not None:
        rate_desc = f"FF rate {fed_current:.2f}%"
        if chg12 is not None:
            rate_desc += f" ({chg12:+.0f}bp over 12m)"
        if chg3 is not None:
            rate_desc += f", {chg3:+.0f}bp last 3m"

    cpi_desc = f"; CPI {cpi_now:+.1f}% YoY" if cpi_now is not None else ""
    if cpi_6m_ago is not None:
        cpi_desc += f" (6m ago: {cpi_6m_ago:+.1f}%)"
    cpi_desc += f"; inflation: {infl_trend.lower().replace('_', ' ')}"

    real_desc = (
        f"; real rate {real_val:+.2f}% ({real_rate_reg.replace('_', ' ').lower()})"
        if real_val is not None else ""
    )

    cut_desc = (
        f"; market pricing {implied_cuts:+.0f}bp of cuts" if implied_cuts is not None else ""
    )

    summary = (
        f"Rate Cycle — {cycle_phase.replace('_', ' ')}: {cycle_dir}. "
        f"{rate_desc}{cpi_desc}{real_desc}{cut_desc}. "
        f"{assets['narrative']}"
    )

    ctx = RotationDriversContext(
        report_date=date.today(),
        fed_rate_current=fed_current,
        fed_rate_3m_ago=fed_3m_ago,
        fed_rate_6m_ago=fed_6m_ago,
        fed_rate_12m_ago=fed_12m_ago,
        rate_change_3m_bp=chg3,
        rate_change_12m_bp=chg12,
        rate_trajectory=rate_traj,
        cpi_yoy_current=cpi_now,
        cpi_yoy_6m_ago=cpi_6m_ago,
        inflation_trend=infl_trend,
        real_rate=real_val,
        real_rate_regime=real_rate_reg,
        cycle_phase=cycle_phase,
        cycle_direction=cycle_dir,
        favoured_assets=assets["favour"],
        avoid_assets=assets["avoid"],
        summary=summary,
    )
    _save_cache(ctx)

    logger.info(
        f"[rotation_drivers] Phase={cycle_phase}  Traj={rate_traj}  "
        f"Inflation={infl_trend}  RealRate={real_rate_reg}  Dir={cycle_dir}"
    )
    return ctx
