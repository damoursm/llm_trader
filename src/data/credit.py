"""
Credit market leading indicator — HYG vs SPY divergence.

High-yield bonds (HYG) typically lead equities (SPY) by 1–3 days because
credit markets are more sensitive to liquidity and risk appetite than equity
markets. When credit tightens (HYG underperforms SPY), equity weakness usually
follows. When credit strengthens relative to equities, it often foreshadows a
rally or sustained move.

Signal logic (5-day divergence = HYG_5d_return − SPY_5d_return):
  divergence < −3.0%   → CREDIT_STRESS    BEARISH  (credit significantly lagging)
  divergence < −1.5%   → CREDIT_CAUTION   BEARISH  (mild credit underperformance)
  −1.5% to +1.5%       → NEUTRAL          NEUTRAL
  divergence > +1.5%   → CREDIT_STRONG    BULLISH  (credit leading equities)
  divergence > +3.0%   → CREDIT_SURGE     BULLISH  (strong risk-on confirmation)

Cached daily — yfinance, no API key required.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import CreditContext

CACHE_DIR = Path("cache")
_LOOKBACK  = "20d"   # history window — enough for 5d and 10d return windows

_STRESS_THRESHOLD  = -3.0   # divergence % below which = CREDIT_STRESS
_CAUTION_THRESHOLD = -1.5   # divergence % below which = CREDIT_CAUTION
_STRONG_THRESHOLD  = +1.5   # divergence % above which = CREDIT_STRONG
_SURGE_THRESHOLD   = +3.0   # divergence % above which = CREDIT_SURGE


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"credit_{date.today().isoformat()}.json"


def _load_cache() -> Optional[CreditContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = CreditContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(
            f"[credit] Loaded from cache — signal={ctx.signal}, "
            f"div={ctx.divergence_5d:+.2f}%" if ctx.divergence_5d is not None else
            f"[credit] Loaded from cache — signal={ctx.signal}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[credit] Cache load failed: {e}")
        return None


def _save_cache(ctx: CreditContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[credit] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify(divergence: float):
    """Return (signal, direction) for the given 5-day HYG−SPY divergence."""
    if divergence < _STRESS_THRESHOLD:
        return "CREDIT_STRESS", "BEARISH"
    if divergence < _CAUTION_THRESHOLD:
        return "CREDIT_CAUTION", "BEARISH"
    if divergence > _SURGE_THRESHOLD:
        return "CREDIT_SURGE", "BULLISH"
    if divergence > _STRONG_THRESHOLD:
        return "CREDIT_STRONG", "BULLISH"
    return "NEUTRAL", "NEUTRAL"


def _pct_return(series, n: int) -> Optional[float]:
    """n-day % return from the last n+1 closes. Returns None if insufficient data."""
    if series is None or len(series) < n + 1:
        return None
    start = float(series.iloc[-(n + 1)])
    end   = float(series.iloc[-1])
    if start == 0:
        return None
    return round((end - start) / start * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_credit_context() -> Optional[CreditContext]:
    """
    Fetch HYG and SPY OHLCV from yfinance and compute the credit-market
    leading-indicator divergence signal.
    Returns CreditContext or None if data is unavailable.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[credit] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    try:
        data = yf.download(
            ["HYG", "SPY"],
            period=_LOOKBACK,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning(f"[credit] yfinance download failed: {e}")
        return None

    if data is None or data.empty:
        logger.warning("[credit] No data returned for HYG/SPY")
        return None

    try:
        # yfinance multi-ticker: columns are (field, ticker) MultiIndex
        close = data["Close"]
        hyg_close = close["HYG"].dropna()
        spy_close = close["SPY"].dropna()
    except Exception as e:
        logger.warning(f"[credit] Could not extract close prices: {e}")
        return None

    if hyg_close.empty or spy_close.empty:
        logger.warning("[credit] Empty close series for HYG or SPY")
        return None

    hyg_1d = _pct_return(hyg_close, 1)
    hyg_5d = _pct_return(hyg_close, 5)
    spy_1d = _pct_return(spy_close, 1)
    spy_5d = _pct_return(spy_close, 5)

    if hyg_5d is None or spy_5d is None:
        logger.warning("[credit] Insufficient history for 5-day return — need 6+ trading days")
        return None

    divergence = round(hyg_5d - spy_5d, 2)
    signal, direction = _classify(divergence)

    # Build summary
    lead_warn = " Credit typically leads equities 1–3 days." if signal != "NEUTRAL" else ""
    action_text = {
        "CREDIT_STRESS":  "Credit markets under significant stress — equity weakness likely to follow.",
        "CREDIT_CAUTION": "Credit mildly underperforming equities — watch for equity follow-through.",
        "NEUTRAL":        "HYG and SPY moving in tandem — no divergence signal.",
        "CREDIT_STRONG":  "Credit outperforming equities — risk-on confirmation, bullish leading signal.",
        "CREDIT_SURGE":   "Credit surging vs equities — strong risk-on; equities likely to follow.",
    }.get(signal, "")

    hyg_price = float(hyg_close.iloc[-1])
    spy_price = float(spy_close.iloc[-1])

    summary = (
        f"HYG {hyg_5d:+.2f}% vs SPY {spy_5d:+.2f}% (5d). "
        f"Divergence: {divergence:+.2f}% ({signal}). "
        f"{action_text}{lead_warn}"
    )

    ctx = CreditContext(
        hyg_price=round(hyg_price, 2),
        spy_price=round(spy_price, 2),
        hyg_return_1d=hyg_1d,
        hyg_return_5d=hyg_5d,
        spy_return_1d=spy_1d,
        spy_return_5d=spy_5d,
        divergence_5d=divergence,
        signal=signal,
        direction=direction,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(
        f"[credit] HYG {hyg_5d:+.2f}% / SPY {spy_5d:+.2f}% (5d) → "
        f"div={divergence:+.2f}% ({signal}, {direction})"
    )
    return ctx
