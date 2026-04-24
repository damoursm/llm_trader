"""Global macro cross-asset regime — DXY strength, Copper/Gold ratio, Oil/Bonds divergence.

Three independent lenses on the global risk appetite and economic cycle:

  1. DXY (US Dollar Index) — DX-Y.NYB
     A rising dollar tightens global financial conditions: it raises the cost of
     dollar-denominated debt for EM borrowers, depresses commodity prices (priced
     in USD), and compresses the overseas earnings of US multinationals.
     Signal mapping (5-day return):
       STRONG_BULL (> +1.5%): significant dollar strength → headwind for EM, commodities, multinationals
       BULL        (> +0.5%): mild dollar strength
       NEUTRAL     (±0.5%):   no dollar signal
       BEAR        (< −0.5%): mild dollar weakness → tailwind for commodities/EM
       STRONG_BEAR (< −1.5%): significant dollar weakness → strong commodity/EM tailwind

  2. Copper/Gold ratio (HG=F / GC=F) — "Dr. Copper" economic barometer
     Copper prices track global industrial demand and economic growth expectations.
     Gold prices track safe-haven demand and real-rate sensitivity.
     A rising copper/gold ratio = industrial demand > safety demand = risk-on expansion.
     A declining ratio = safety demand > industrial demand = risk-off contraction.
     Signal mapping (20-day % change in ratio):
       RISK_ON_SURGE  (> +5%):   Dr. Copper strongly bullish on global growth
       RISK_ON        (> +2%):   mild risk-on signal
       NEUTRAL        (±2%):     no directional signal from cross-asset ratio
       RISK_OFF       (< −2%):   mild contraction signal
       RISK_OFF_CRASH (< −5%):   Dr. Copper signalling contraction / recessionary risk

  3. Oil/Bonds divergence (CL=F vs TLT — 5-day co-movement)
     Oil and Treasury bonds are normally inversely correlated: rising oil = inflation
     expectations → bond yields rise → TLT falls. When both move in the same direction,
     the normal macro framework is being overridden — this is a high-value divergence signal.
       Both up   (POLICY_PIVOT_SIGNAL):  market is pricing a Fed policy pivot despite oil →
                                          BULLISH for equities (growth support outweighs inflation)
       Oil up, bonds down (STAGFLATION_RISK): worst case — rising costs + tightening rates
       Oil down, bonds up (GROWTH_FEAR_RISK_OFF): demand destruction + flight to safety → BEARISH
       Both down (DEFLATION_SHOCK):      broad de-risking or liquidity squeeze → BEARISH

Cached daily — yfinance, no API key required.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import GlobalMacroContext

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "global_macro_"
_TICKERS      = ["DX-Y.NYB", "HG=F", "GC=F", "CL=F", "TLT"]

# DXY thresholds (5-day % return)
_DXY_STRONG_BULL  =  1.5
_DXY_BULL         =  0.5
_DXY_BEAR         = -0.5
_DXY_STRONG_BEAR  = -1.5

# Copper/Gold ratio thresholds (20-day % change)
_CG_RISK_ON_SURGE   =  5.0
_CG_RISK_ON         =  2.0
_CG_RISK_OFF        = -2.0
_CG_RISK_OFF_CRASH  = -5.0

# Oil/Bond divergence thresholds (5-day % return)
_OIL_UP   =  2.5   # oil rally threshold
_OIL_DOWN = -2.5   # oil selloff threshold
_TLT_UP   =  1.5   # bonds rallying (yields falling)
_TLT_DOWN = -1.5   # bonds selling off (yields rising)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[GlobalMacroContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = GlobalMacroContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(
            f"[global_macro] Loaded from cache — "
            f"DXY {ctx.dxy_signal} / CG {ctx.copper_gold_signal} / composite {ctx.composite_signal}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[global_macro] Cache load failed: {e}")
        return None


def _save_cache(ctx: GlobalMacroContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[global_macro] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_change(series, n: int) -> Optional[float]:
    """n-period % change from the last close. Returns None if insufficient data."""
    if series is None or len(series) <= n:
        return None
    start = float(series.iloc[-(n + 1)])
    end   = float(series.iloc[-1])
    if start == 0:
        return None
    return round((end / start - 1) * 100, 3)


def _last(series) -> Optional[float]:
    if series is None or series.empty:
        return None
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return None


def _nth_last(series, n: int) -> Optional[float]:
    s = series.dropna()
    if len(s) <= n:
        return None
    return float(s.iloc[-(n + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_dxy(ret_5d: Optional[float]):
    if ret_5d is None:
        return "UNKNOWN", "NEUTRAL"
    if ret_5d > _DXY_STRONG_BULL:
        return "STRONG_BULL", "BEARISH"   # for equities
    if ret_5d > _DXY_BULL:
        return "BULL", "BEARISH"
    if ret_5d < _DXY_STRONG_BEAR:
        return "STRONG_BEAR", "BULLISH"
    if ret_5d < _DXY_BEAR:
        return "BEAR", "BULLISH"
    return "NEUTRAL", "NEUTRAL"


def _classify_copper_gold(change_20d: Optional[float]):
    if change_20d is None:
        return "UNKNOWN", "NEUTRAL"
    if change_20d > _CG_RISK_ON_SURGE:
        return "RISK_ON_SURGE", "BULLISH"
    if change_20d > _CG_RISK_ON:
        return "RISK_ON", "BULLISH"
    if change_20d < _CG_RISK_OFF_CRASH:
        return "RISK_OFF_CRASH", "BEARISH"
    if change_20d < _CG_RISK_OFF:
        return "RISK_OFF", "BEARISH"
    return "NEUTRAL", "NEUTRAL"


def _classify_oil_bond(oil_5d: Optional[float], tlt_5d: Optional[float]):
    """Classify the oil/bond co-movement divergence signal."""
    if oil_5d is None or tlt_5d is None:
        return "NEUTRAL", "NEUTRAL"

    oil_up   = oil_5d   >  _OIL_UP
    oil_down = oil_5d   <  _OIL_DOWN
    tlt_up   = tlt_5d   >  _TLT_UP
    tlt_down = tlt_5d   <  _TLT_DOWN

    if oil_up and tlt_up:
        return "POLICY_PIVOT_SIGNAL", "BULLISH"   # unusual co-rally → Fed cutting despite oil
    if oil_down and tlt_down:
        return "DEFLATION_SHOCK",     "BEARISH"   # both selling off → demand destruction
    if oil_up and tlt_down:
        return "STAGFLATION_RISK",    "BEARISH"   # oil up + rates rising = worst combo
    if oil_down and tlt_up:
        return "GROWTH_FEAR_RISK_OFF","BEARISH"   # demand destruction + flight to safety
    return "NEUTRAL", "NEUTRAL"


def _composite(dxy_dir: str, cg_dir: str) -> tuple[str, str]:
    """Combine DXY and Copper/Gold directions into a composite regime."""
    # Strong DXY is bearish for equities; weak DXY is bullish
    # Copper/gold directions are already framed for equities
    _score = {
        "BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1, "UNKNOWN": 0,
    }
    net = _score[dxy_dir] + _score[cg_dir]
    if net >= 2:
        return "RISK_ON",      "BULLISH"
    if net == 1:
        return "CONSTRUCTIVE", "BULLISH"
    if net == 0:
        return "NEUTRAL",      "NEUTRAL"
    if net == -1:
        return "DEFENSIVE",    "BEARISH"
    return "RISK_OFF",     "BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_global_macro_context(today: Optional[date] = None) -> Optional[GlobalMacroContext]:
    """Fetch DXY and Copper/Gold ratio and compute global macro regime signals.

    Downloads ~90 calendar days of daily closes (covers the 20-day windows).
    Returns None if data is unavailable or fetching is disabled.  Cached daily.
    """
    if today is None:
        today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[global_macro] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    start = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            _TICKERS,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"[global_macro] yfinance download failed: {e}")
        return None

    if raw is None or raw.empty:
        logger.warning("[global_macro] No data returned from yfinance")
        return None

    try:
        closes = raw["Close"]
    except Exception as e:
        logger.warning(f"[global_macro] Could not extract Close prices: {e}")
        return None

    def _s(ticker: str):
        try:
            return closes[ticker].dropna()
        except Exception:
            return None

    dxy_s    = _s("DX-Y.NYB")
    copper_s = _s("HG=F")
    gold_s   = _s("GC=F")
    oil_s    = _s("CL=F")
    tlt_s    = _s("TLT")

    # Guard: at least DXY or both metals must have data
    if (dxy_s is None or dxy_s.empty) and (
        copper_s is None or copper_s.empty or gold_s is None or gold_s.empty
    ) and (oil_s is None or oil_s.empty):
        logger.warning("[global_macro] All tickers empty — likely rate-limited, not caching")
        return None

    # ── DXY ────────────────────────────────────────────────────────────────
    dxy         = _last(dxy_s)
    dxy_5d      = _pct_change(dxy_s, 5)
    dxy_20d     = _pct_change(dxy_s, 20)
    dxy_signal, dxy_direction = _classify_dxy(dxy_5d)

    # ── Copper/Gold ratio ───────────────────────────────────────────────────
    copper_price = _last(copper_s)
    gold_price   = _last(gold_s)

    cg_ratio = None
    if copper_price and gold_price and gold_price > 0:
        cg_ratio = round(copper_price / gold_price, 6)

    # Build ratio series for trend calculation
    cg_ratio_5d_ago  = None
    cg_ratio_20d_ago = None
    cg_change_5d     = None
    cg_change_20d    = None

    if copper_s is not None and gold_s is not None:
        min_len = min(len(copper_s), len(gold_s))
        if min_len > 5:
            # align by taking the shorter length from the right
            cu = copper_s.values[-min_len:]
            au = gold_s.values[-min_len:]
            ratios = cu / au  # element-wise ratio series

            cur = ratios[-1]

            if len(ratios) > 5:
                past_5  = ratios[-6]
                cg_ratio_5d_ago = round(float(past_5), 6)
                if past_5 > 0:
                    cg_change_5d = round((float(cur) / float(past_5) - 1) * 100, 3)

            if len(ratios) > 20:
                past_20 = ratios[-21]
                cg_ratio_20d_ago = round(float(past_20), 6)
                if past_20 > 0:
                    cg_change_20d = round((float(cur) / float(past_20) - 1) * 100, 3)

    cg_signal, cg_direction = _classify_copper_gold(cg_change_20d)
    composite_signal, composite_direction = _composite(dxy_direction, cg_direction)

    # ── Oil / Bond divergence ───────────────────────────────────────────────
    oil_price   = _last(oil_s)
    oil_5d      = _pct_change(oil_s, 5)
    oil_20d     = _pct_change(oil_s, 20)
    tlt_5d_ob   = _pct_change(tlt_s, 5)   # TLT for oil/bond calc

    ob_signal, ob_direction = _classify_oil_bond(oil_5d, tlt_5d_ob)

    # ── Summary ─────────────────────────────────────────────────────────────
    parts = []
    if dxy is not None and dxy_5d is not None:
        parts.append(f"DXY={dxy:.2f} ({dxy_5d:+.2f}% 5d, {dxy_signal})")
    if cg_ratio is not None:
        cg_part = f"Cu/Au={cg_ratio:.5f}"
        if cg_change_20d is not None:
            cg_part += f" ({cg_change_20d:+.1f}% 20d, {cg_signal})"
        parts.append(cg_part)
    if oil_price is not None and oil_5d is not None:
        parts.append(f"Oil=${oil_price:.1f} ({oil_5d:+.1f}% 5d)")
    if ob_signal != "NEUTRAL":
        parts.append(f"Oil/Bond={ob_signal}")

    summary = f"Global macro: {composite_signal} ({composite_direction}) — " + " | ".join(parts)

    # Don't cache if we got nothing meaningful
    if dxy is None and cg_ratio is None and oil_price is None:
        logger.warning("[global_macro] No signals computed — skipping cache")
        return None

    ctx = GlobalMacroContext(
        dxy=round(dxy, 3) if dxy else None,
        dxy_return_5d=dxy_5d,
        dxy_return_20d=dxy_20d,
        dxy_signal=dxy_signal,
        dxy_direction=dxy_direction,
        copper_price=round(copper_price, 4) if copper_price else None,
        gold_price=round(gold_price, 2) if gold_price else None,
        copper_gold_ratio=cg_ratio,
        copper_gold_ratio_5d_ago=cg_ratio_5d_ago,
        copper_gold_ratio_20d_ago=cg_ratio_20d_ago,
        copper_gold_change_5d=cg_change_5d,
        copper_gold_change_20d=cg_change_20d,
        copper_gold_signal=cg_signal,
        copper_gold_direction=cg_direction,
        oil_price=round(oil_price, 2) if oil_price else None,
        oil_return_5d=oil_5d,
        oil_return_20d=oil_20d,
        tlt_return_5d_ob=tlt_5d_ob,
        oil_bond_signal=ob_signal,
        oil_bond_direction=ob_direction,
        composite_signal=composite_signal,
        composite_direction=composite_direction,
        report_date=today,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(
        f"[global_macro] DXY={dxy:.2f} ({dxy_signal}) | "
        f"Cu/Au={'N/A' if cg_ratio is None else f'{cg_ratio:.5f}'} ({cg_signal}) | "
        f"Oil={'N/A' if oil_price is None else f'${oil_price:.1f}'} ob={ob_signal} | "
        f"composite={composite_signal}"
    )
    return ctx
