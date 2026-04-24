"""Bond market internals — macro regime signals from Treasury and credit ETFs.

Five independent lenses on the 1–8 week macro environment, all via yfinance:

  1. Yield curve shape (10Y − 3M spread)
     The 3-month/10-year inversion is the most reliable recession predictor available.
     Unlike the 2Y-10Y, the 10Y-3M spread incorporates actual near-term Fed policy
     expectations (the 3M bill reflects the expected average FF rate over 90 days).
     Inversions that persist for 3+ months historically precede recessions by 6–18 months
     with ~95% accuracy.

  2. TLT (20+ year Treasury ETF) price momentum — 1-week, 4-week, 8-week horizons
     TLT price is the inverse of 20+ year Treasury yields. A falling TLT = rising long
     rates = higher discount rates = headwind for long-duration assets (growth tech, REITs).
     A rallying TLT (falling rates) is a tailwind for the same names.

  3. Duration positioning: TLT vs IEF 5-day relative return
     When TLT underperforms IEF, the long end is selling off faster than intermediate
     → bear steepening → inflation/fiscal concerns or term premium re-pricing.
     When TLT outperforms IEF → bull flattening → risk-off flight to duration.

  4. Real yield proxy: TIP vs IEF 5-day relative return
     TIPS (TIP) embed inflation compensation. TIP outperforming IEF = inflation
     expectations rising = real rates falling. Constructive for commodities and gold;
     mixed for equities; headwind for long-duration tech on a real discount rate basis.

  5. IG credit risk premium: LQD vs TLT 5-day relative return
     IG corporates embed a spread above Treasuries. LQD underperforming TLT = IG
     spreads widening = rising corporate stress → typically leads equity weakness by
     1–5 days. LQD outperforming TLT = spreads tightening = constructive for risk assets.

Cached daily — yfinance, no API key required.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import BondInternalsContext

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "bond_internals_"
_TICKERS      = ["^TNX", "^IRX", "^FVX", "^TYX", "TLT", "IEF", "TIP", "LQD", "SPY"]

# Bond-equity divergence thresholds
_BOND_RALLY_STRONG  =  2.5   # TLT 5d return ≥ this = bonds rallying hard
_BOND_RALLY_MILD    =  1.5   # TLT 5d return ≥ this = bonds rallying mildly
_BOND_SELLOFF       = -2.0   # TLT 5d return ≤ this = bonds selling off
_BOND_SELLOFF_MILD  = -1.5   # for synchronized risk-off detection
_EQUITY_FLAT_HIGH   =  1.5   # SPY 5d return < this = equities "holding"
_EQUITY_FLAT_LOW    = -1.5   # SPY 5d return > this = equities not in freefall
_EQUITY_RALLY       =  2.0   # SPY 5d return > this = equities rallying
_EQUITY_SELLOFF     = -2.0   # SPY 5d return < this = equities in selloff


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[BondInternalsContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = BondInternalsContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(f"[bond_internals] Loaded from cache — {ctx.regime} ({ctx.direction})")
        return ctx
    except Exception as e:
        logger.warning(f"[bond_internals] Cache load failed: {e}")
        return None


def _save_cache(ctx: BondInternalsContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[bond_internals] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_yield(val: Optional[float]) -> Optional[float]:
    """Normalise ^TNX-style CBOE yield index to percent.

    The CBOE historically published yield indices ×10 (4.5% → 45.0). Modern
    yfinance feeds vary. We divide by 10 when the value is implausibly high
    for an interest rate (> 20 implies raw CBOE format, not decimal percent).
    """
    if val is None:
        return None
    return round(val / 10.0, 3) if val > 20.0 else round(val, 3)


def _pct_return(series, n: int) -> Optional[float]:
    """n-period % return from the last close. Returns None if insufficient data."""
    if series is None or len(series) <= n:
        return None
    start = float(series.iloc[-(n + 1)])
    end   = float(series.iloc[-1])
    if start == 0:
        return None
    return round((end / start - 1) * 100, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bond_internals_context(today: Optional[date] = None) -> Optional[BondInternalsContext]:
    """Fetch Treasury/credit ETF data and compute bond market regime signals.

    Fetches ~90 calendar days (≈ 63 trading days) to cover the 1-week through
    8-week momentum windows. Returns None if data is unavailable or fetching
    is disabled.  Cached daily.
    """
    if today is None:
        today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[bond_internals] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
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
        logger.warning(f"[bond_internals] yfinance download failed: {e}")
        return None

    if raw is None or raw.empty:
        logger.warning("[bond_internals] No data returned from yfinance")
        return None

    try:
        closes = raw["Close"]
    except Exception as e:
        logger.warning(f"[bond_internals] Could not extract Close prices: {e}")
        return None

    # Verify at least one ETF ticker has usable data (guard against all-NaN from rate limits)
    _etf_check = ["TLT", "IEF"]
    if all(closes.get(t) is None or closes[t].dropna().empty for t in _etf_check):
        logger.warning("[bond_internals] All ETF tickers empty — likely rate-limited, not caching")
        return None

    def _last(ticker: str) -> Optional[float]:
        try:
            s = closes[ticker].dropna()
            return float(s.iloc[-1]) if len(s) > 0 else None
        except Exception:
            return None

    def _ret(ticker: str, n: int) -> Optional[float]:
        try:
            s = closes[ticker].dropna()
            return _pct_return(s, n)
        except Exception:
            return None

    # ── Raw yields ────────────────────────────────────────────────────────────
    yield_10y = _norm_yield(_last("^TNX"))
    yield_3m  = _norm_yield(_last("^IRX"))
    yield_5y  = _norm_yield(_last("^FVX"))
    yield_30y = _norm_yield(_last("^TYX"))

    # ── Curve spreads ─────────────────────────────────────────────────────────
    spread_10y_3m  = round(yield_10y - yield_3m,  3) if yield_10y and yield_3m  else None
    spread_10y_5y  = round(yield_10y - yield_5y,  3) if yield_10y and yield_5y  else None
    spread_30y_10y = round(yield_30y - yield_10y, 3) if yield_30y and yield_10y else None

    if spread_10y_3m is not None:
        if   spread_10y_3m < -0.75: curve_signal = "DEEPLY_INVERTED"
        elif spread_10y_3m < -0.10: curve_signal = "INVERTED"
        elif spread_10y_3m <  0.50: curve_signal = "FLAT"
        elif spread_10y_3m <  1.50: curve_signal = "NORMAL"
        else:                        curve_signal = "STEEP"
    else:
        curve_signal = "UNKNOWN"

    # ── TLT momentum ─────────────────────────────────────────────────────────
    tlt_5d  = _ret("TLT",  5)
    tlt_20d = _ret("TLT", 20)
    tlt_40d = _ret("TLT", 40)

    if tlt_20d is not None:
        if   tlt_20d >  3.0: tlt_signal = "RALLYING_STRONG"
        elif tlt_20d >  1.0: tlt_signal = "RALLYING"
        elif tlt_20d > -1.0: tlt_signal = "FLAT"
        elif tlt_20d > -3.0: tlt_signal = "FALLING"
        else:                  tlt_signal = "FALLING_STRONG"
    else:
        tlt_signal = "UNKNOWN"

    # ── Duration spread: TLT vs IEF (5-day) ──────────────────────────────────
    ief_5d = _ret("IEF", 5)
    tlt_ief_spread_5d = (
        round(tlt_5d - ief_5d, 3) if tlt_5d is not None and ief_5d is not None else None
    )
    if tlt_ief_spread_5d is not None:
        if   tlt_ief_spread_5d < -0.30: tlt_ief_signal = "LONG_END_PRESSURE"  # bear steepening
        elif tlt_ief_spread_5d >  0.30: tlt_ief_signal = "LONG_END_RALLY"      # bull flattening
        else:                            tlt_ief_signal = "FLAT"
    else:
        tlt_ief_signal = "UNKNOWN"

    # ── Real yield proxy: TIP vs IEF (5-day) ─────────────────────────────────
    tip_5d = _ret("TIP", 5)
    tip_ief_spread_5d = (
        round(tip_5d - ief_5d, 3) if tip_5d is not None and ief_5d is not None else None
    )
    if tip_ief_spread_5d is not None:
        if   tip_ief_spread_5d >  0.20: real_yield_signal = "REAL_RATES_FALLING"
        elif tip_ief_spread_5d < -0.20: real_yield_signal = "REAL_RATES_RISING"
        else:                            real_yield_signal = "NEUTRAL"
    else:
        real_yield_signal = "UNKNOWN"

    # ── IG credit: LQD vs TLT (5-day) ────────────────────────────────────────
    lqd_5d = _ret("LQD", 5)
    lqd_tlt_spread_5d = (
        round(lqd_5d - tlt_5d, 3) if lqd_5d is not None and tlt_5d is not None else None
    )
    if lqd_tlt_spread_5d is not None:
        if   lqd_tlt_spread_5d < -0.30: ig_credit_signal = "IG_STRESS"
        elif lqd_tlt_spread_5d < -0.10: ig_credit_signal = "IG_CAUTION"
        elif lqd_tlt_spread_5d >  0.10: ig_credit_signal = "IG_STRONG"
        else:                             ig_credit_signal = "NEUTRAL"
    else:
        ig_credit_signal = "UNKNOWN"

    # ── Bond-equity divergence: TLT/IEF vs SPY (5-day) ──────────────────────
    spy_5d  = _ret("SPY", 5)
    spy_20d = _ret("SPY", 20)

    tlt_spy_div_5d = (
        round(tlt_5d - spy_5d, 3) if tlt_5d is not None and spy_5d is not None else None
    )
    ief_spy_div_5d = (
        round(ief_5d - spy_5d, 3) if ief_5d is not None and spy_5d is not None else None
    )

    bond_equity_signal    = "NEUTRAL"
    bond_equity_direction = "NEUTRAL"

    if tlt_5d is not None and spy_5d is not None:
        bonds_rallying_hard = tlt_5d >= _BOND_RALLY_STRONG
        bonds_rallying_mild = tlt_5d >= _BOND_RALLY_MILD
        bonds_selling_off   = tlt_5d <= _BOND_SELLOFF
        bonds_selling_mild  = tlt_5d <= _BOND_SELLOFF_MILD
        equities_flat       = _EQUITY_FLAT_LOW < spy_5d < _EQUITY_FLAT_HIGH
        equities_rallying   = spy_5d >= _EQUITY_RALLY
        equities_selling    = spy_5d <= _EQUITY_SELLOFF

        if bonds_rallying_hard and equities_flat:
            bond_equity_signal    = "EQUITY_CATCHUP_LIKELY"
            bond_equity_direction = "BULLISH"
        elif bonds_rallying_mild and equities_flat:
            bond_equity_signal    = "EQUITY_CATCHUP_POSSIBLE"
            bond_equity_direction = "BULLISH"
        elif bonds_rallying_mild and equities_rallying:
            bond_equity_signal    = "SYNCHRONIZED_RISK_ON"
            bond_equity_direction = "BULLISH"
        elif bonds_selling_off and equities_flat:
            bond_equity_signal    = "EQUITY_SELLOFF_RISK"
            bond_equity_direction = "BEARISH"
        elif bonds_selling_mild and equities_selling:
            bond_equity_signal    = "SYNCHRONIZED_RISK_OFF"
            bond_equity_direction = "BEARISH"

    # ── Composite regime ──────────────────────────────────────────────────────
    bear_pts = 0
    bull_pts = 0

    if   curve_signal == "DEEPLY_INVERTED": bear_pts += 2
    elif curve_signal == "INVERTED":        bear_pts += 1
    elif curve_signal == "STEEP":           bull_pts += 1

    if   tlt_signal == "FALLING_STRONG":    bear_pts += 1
    elif tlt_signal == "RALLYING_STRONG":   bull_pts += 1

    if   ig_credit_signal == "IG_STRESS":   bear_pts += 2
    elif ig_credit_signal == "IG_CAUTION":  bear_pts += 1
    elif ig_credit_signal == "IG_STRONG":   bull_pts += 1

    if real_yield_signal == "REAL_RATES_RISING":    bear_pts += 1
    if tlt_ief_signal    == "LONG_END_PRESSURE":    bear_pts += 1

    net = bull_pts - bear_pts
    if   net >= 2: regime, direction = "RISK_ON",      "BULLISH"
    elif net == 1: regime, direction = "CONSTRUCTIVE", "BULLISH"
    elif net == 0: regime, direction = "NEUTRAL",      "NEUTRAL"
    elif net ==-1: regime, direction = "DEFENSIVE",    "BEARISH"
    else:           regime, direction = "RISK_OFF",     "BEARISH"

    # REFLATIONARY override: rising nominal rates + inflation expectations rising
    if tlt_signal in ("FALLING", "FALLING_STRONG") and real_yield_signal == "REAL_RATES_FALLING":
        regime    = "REFLATIONARY"
        direction = "NEUTRAL"   # good for commodities/cyclicals; bad for long-duration tech

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if yield_10y: parts.append(f"10Y={yield_10y:.2f}%")
    if yield_3m:  parts.append(f"3M={yield_3m:.2f}%")
    if spread_10y_3m is not None:
        parts.append(f"10Y-3M={spread_10y_3m:+.2f}pp ({curve_signal})")
    if tlt_20d is not None:
        parts.append(f"TLT 4w={tlt_20d:+.1f}% ({tlt_signal})")
    if ig_credit_signal not in ("UNKNOWN",):
        parts.append(f"IG={ig_credit_signal}")
    if real_yield_signal not in ("UNKNOWN",):
        parts.append(f"real={real_yield_signal}")

    summary = f"Bond internals: {regime} ({direction}) — " + " | ".join(parts)

    # Don't cache if no signals resolved (e.g. partial rate-limit)
    if tlt_20d is None and spread_10y_3m is None:
        logger.warning("[bond_internals] No signals computed — skipping cache")
        return None

    ctx = BondInternalsContext(
        yield_10y=yield_10y,
        yield_3m=yield_3m,
        yield_5y=yield_5y,
        yield_30y=yield_30y,
        spread_10y_3m=spread_10y_3m,
        spread_10y_5y=spread_10y_5y,
        spread_30y_10y=spread_30y_10y,
        curve_signal=curve_signal,
        tlt_return_5d=tlt_5d,
        tlt_return_20d=tlt_20d,
        tlt_return_40d=tlt_40d,
        tlt_signal=tlt_signal,
        tlt_ief_spread_5d=tlt_ief_spread_5d,
        tlt_ief_signal=tlt_ief_signal,
        tip_ief_spread_5d=tip_ief_spread_5d,
        real_yield_signal=real_yield_signal,
        lqd_tlt_spread_5d=lqd_tlt_spread_5d,
        ig_credit_signal=ig_credit_signal,
        spy_return_5d=spy_5d,
        spy_return_20d=spy_20d,
        tlt_spy_div_5d=tlt_spy_div_5d,
        ief_spy_div_5d=ief_spy_div_5d,
        bond_equity_signal=bond_equity_signal,
        bond_equity_direction=bond_equity_direction,
        regime=regime,
        direction=direction,
        report_date=today,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(f"[bond_internals] {regime} ({direction}): {summary[:120]}")
    return ctx
