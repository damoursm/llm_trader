"""Intermarket divergence detector — broad-index ETFs vs SPY.

Catches cross-market regime tells the per-stock signals and per-sector
rotation miss: small-cap vs large-cap leadership, US vs international,
growth vs value, equal-weight vs cap-weight participation. The composite
intermarket_health composes with the rest of the macro stack via
``compute_macro_regime`` — narrow leadership tightens the BUY threshold;
broad participation relaxes it.

What this module does NOT duplicate:
  * Sector-vs-market lag — covered by ``sector_rotation`` for the 11 SPDR sectors.
  * Bonds vs equities — covered by ``bond_internals`` and ``credit``.
  * Volatility regime — covered by ``vix`` and ``move``.

The basket below is the canonical intermarket dashboard used by macro
strategists. Cache: daily JSON in ``cache/intermarket_YYYY-MM-DD.json``.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config import settings
from src.models import IntermarketContext, IntermarketEntry

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "intermarket_"

# Basket: ticker → (human name, regime role). The regime role is a hint used
# by the label compiler — multiple tickers can belong to the same regime
# (e.g. IWM + RSP both inform NARROW_LEADERSHIP). Avoid overlap with
# sector_rotation (sector ETFs) and bond_internals (TLT/HYG).
_INTERMARKET_BASKET: Dict[str, str] = {
    "IWM":  "Russell 2000 (small-caps)",
    "RSP":  "S&P 500 equal-weight",
    "QQQ":  "NASDAQ-100 (tech-heavy)",
    "DIA":  "Dow Jones Industrial",
    "MDY":  "S&P 400 mid-caps",
    "EFA":  "Developed ex-US",
    "EEM":  "Emerging Markets",
    "IWF":  "Russell 1000 growth",
    "IWD":  "Russell 1000 value",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[IntermarketContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = IntermarketContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(
            f"[intermarket] Loaded from cache — "
            f"{ctx.composite_signal} health={ctx.intermarket_health:+.2f} "
            f"labels={ctx.regime_labels}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[intermarket] Cache load failed: {e}")
        return None


def _save_cache(ctx: IntermarketContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[intermarket] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_change(series: pd.Series, n: int) -> Optional[float]:
    s = series.dropna()
    if len(s) <= n or n <= 0:
        return None
    start = float(s.iloc[-(n + 1)])
    end   = float(s.iloc[-1])
    if start <= 0:
        return None
    return round((end / start - 1) * 100, 3)


def _signal_from_rel(rel_1m: Optional[float]) -> str:
    """LEADING / NEUTRAL / LAGGING — 1-month residual vs SPY in pp."""
    if rel_1m is None:
        return "NEUTRAL"
    if rel_1m >= 1.0:
        return "LEADING"
    if rel_1m <= -1.0:
        return "LAGGING"
    return "NEUTRAL"


def _direction_from_role(etf: str, signal: str) -> str:
    """Map a leg's signal to a BULL/BEAR regime contribution.

    Most legs are straightforward (LEADING = bullish for the broad market),
    but defensive proxies and the dollar-sensitive ones invert. None apply
    here today — every ETF in the basket reads bullish-when-leading — but
    the function keeps the shape symmetric with SectorRotationEntry so the
    email template can treat both the same way.
    """
    if signal == "LEADING":
        return "BULLISH"
    if signal == "LAGGING":
        return "BEARISH"
    return "NEUTRAL"


def _by_etf(entries: List[IntermarketEntry]) -> Dict[str, IntermarketEntry]:
    return {e.etf: e for e in entries}


# ─────────────────────────────────────────────────────────────────────────────
# Regime labels — interpret the cross-section
# ─────────────────────────────────────────────────────────────────────────────

def _compute_regime_labels(entries: List[IntermarketEntry]) -> List[str]:
    """Return a list of named regimes that fired given the residual readings.

    Thresholds are 1m residuals in percentage points (pp). Picked at
    ±1pp / ±2pp boundaries that historically separate clean tells from
    noise on monthly windows.
    """
    by = _by_etf(entries)
    labels: List[str] = []

    def rel(etf: str) -> Optional[float]:
        e = by.get(etf)
        return e.relative_1m_pct if e is not None else None

    iwm = rel("IWM"); rsp = rel("RSP"); qqq = rel("QQQ"); dia = rel("DIA")
    mdy = rel("MDY"); efa = rel("EFA"); eem = rel("EEM")
    iwf = rel("IWF"); iwd = rel("IWD")

    # NARROW_LEADERSHIP — IWM + RSP both lag → mega-cap dependence
    if iwm is not None and rsp is not None and iwm <= -2.0 and rsp <= -1.0:
        labels.append("NARROW_LEADERSHIP")
    # BROAD_PARTICIPATION — IWM + RSP both lead → healthy rally
    if iwm is not None and rsp is not None and iwm >= 1.0 and rsp >= 0.5:
        labels.append("BROAD_PARTICIPATION")
    # GROWTH_ROTATION — QQQ leading AND IWF above IWD
    if qqq is not None and iwf is not None and iwd is not None \
            and qqq >= 1.0 and (iwf - iwd) >= 1.0:
        labels.append("GROWTH_ROTATION")
    # VALUE_ROTATION — IWD above IWF and QQQ flat/lagging
    if iwf is not None and iwd is not None and qqq is not None \
            and (iwd - iwf) >= 1.0 and qqq <= 0.0:
        labels.append("VALUE_ROTATION")
    # US_EXCEPTIONALISM — both EFA and EEM lag
    if efa is not None and eem is not None and efa <= -1.0 and eem <= -1.0:
        labels.append("US_EXCEPTIONALISM")
    # INTERNATIONAL_STRENGTH — EFA or EEM strongly leading
    if (efa is not None and efa >= 1.5) or (eem is not None and eem >= 1.5):
        labels.append("INTERNATIONAL_STRENGTH")
    # MID_CAP_LEADERSHIP — MDY leading while IWM neutral or lagging (institutional
    # tier participating, retail tier not yet)
    if mdy is not None and iwm is not None and mdy >= 1.0 and iwm <= 0.5:
        labels.append("MID_CAP_LEADERSHIP")
    # CYCLICAL_LEAD — DIA leading (industrials, financials, energy mega-caps)
    if dia is not None and dia >= 1.0:
        labels.append("CYCLICAL_LEAD")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Composite health score
# ─────────────────────────────────────────────────────────────────────────────

# Weights tuned so small-cap participation (IWM) carries the most regime
# information — it's the canonical risk-appetite tell — followed by equal-
# weight (RSP). International ETFs each get a smaller weight; growth/value
# net to zero in either rotation direction so they don't bias the composite.
_HEALTH_WEIGHTS: Dict[str, float] = {
    "IWM": 0.30,
    "RSP": 0.25,
    "MDY": 0.15,
    "QQQ": 0.10,
    "DIA": 0.05,
    "EFA": 0.075,
    "EEM": 0.075,
}

_HEALTH_TANH_SCALE = 4.0  # divisor for tanh; 4pp residual → ~tanh(1) ≈ 0.76


def _compute_health(entries: List[IntermarketEntry]) -> float:
    """Composite health ∈ [-1, +1] — positive = broad / healthy, negative = narrow / risk-off."""
    by = _by_etf(entries)
    total = 0.0
    weight_sum = 0.0
    for etf, w in _HEALTH_WEIGHTS.items():
        e = by.get(etf)
        if e is None or e.relative_1m_pct is None:
            continue
        contrib = math.tanh(e.relative_1m_pct / _HEALTH_TANH_SCALE) * w
        total += contrib
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    # Normalise so we don't penalise the composite when some ETFs are missing
    # from the cache. The /sum(weights) keeps the score in roughly [-1, +1].
    return round(max(-1.0, min(1.0, total / weight_sum)), 3)


def _composite_signal(health: float) -> str:
    """Map composite health to a regime label compatible with macro_regime score maps."""
    if health >= 0.40:
        return "BROAD_EXPANSION"
    if health >= 0.15:
        return "BROAD_HEALTHY"
    if health <= -0.40:
        return "NARROW_RISK_OFF"
    if health <= -0.15:
        return "NARROW_CAUTION"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_intermarket_context() -> Optional[IntermarketContext]:
    """Compute the intermarket leaderboard + regime read.

    Cache-first: if today's cache file exists we return it directly. On a
    cold cache the function pulls 3 months of OHLCV per ETF (cache-first via
    ``market_data.get_history``) and writes the result to disk.
    """
    today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    # Lazy import to avoid circular dependency (market_data imports models)
    from src.data.market_data import get_history

    tickers = list(_INTERMARKET_BASKET.keys()) + ["SPY"]
    histories: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = get_history(ticker, period="3mo")
            if df is not None and not df.empty and len(df) >= 10:
                histories[ticker] = df
        except Exception as exc:
            logger.debug(f"[intermarket] {ticker}: {exc}")

    if "SPY" not in histories:
        logger.warning("[intermarket] SPY history unavailable — skipping intermarket regime")
        return None

    spy_close   = histories["SPY"]["Close"]
    spy_ret_1m  = _pct_change(spy_close, 21)
    spy_ret_3m  = _pct_change(spy_close, min(63, max(0, len(spy_close.dropna()) - 2)))

    entries: List[IntermarketEntry] = []
    for etf, name in _INTERMARKET_BASKET.items():
        df = histories.get(etf)
        if df is None:
            continue
        close = df["Close"]
        ret_1m = _pct_change(close, 21)
        ret_3m = _pct_change(close, min(63, max(0, len(close.dropna()) - 2)))

        def _rel(ret, spy_ret):
            if ret is None or spy_ret is None:
                return None
            return round(ret - spy_ret, 3)

        rel_1m = _rel(ret_1m, spy_ret_1m)
        rel_3m = _rel(ret_3m, spy_ret_3m)
        sig    = _signal_from_rel(rel_1m)
        entries.append(IntermarketEntry(
            etf=etf,
            name=name,
            return_1m_pct=ret_1m,
            return_3m_pct=ret_3m,
            relative_1m_pct=rel_1m,
            relative_3m_pct=rel_3m,
            signal=sig,
            direction=_direction_from_role(etf, sig),
        ))

    if not entries:
        logger.warning("[intermarket] No entries computed — skipping")
        return None

    # Sort by 1m relative perf descending (leaders first)
    entries.sort(
        key=lambda e: (e.relative_1m_pct if e.relative_1m_pct is not None else -999),
        reverse=True,
    )

    leaders  = [e.etf for e in entries if e.signal == "LEADING"][:4]
    laggards = [e.etf for e in entries if e.signal == "LAGGING"][-4:]
    labels   = _compute_regime_labels(entries)
    health   = _compute_health(entries)
    sig      = _composite_signal(health)

    label_str = ", ".join(labels) if labels else "no canonical regime"
    leader_str  = ", ".join(leaders)  if leaders  else "—"
    laggard_str = ", ".join(laggards) if laggards else "—"
    summary = (
        f"Intermarket: {sig} (health={health:+.2f}). "
        f"Leaders: {leader_str}.  Laggards: {laggard_str}.  Labels: {label_str}."
    )

    ctx = IntermarketContext(
        entries=entries,
        regime_labels=labels,
        leaders=leaders,
        laggards=laggards,
        intermarket_health=health,
        composite_signal=sig,
        report_date=today,
        summary=summary,
    )

    logger.info(f"[intermarket] {sig}  health={health:+.2f}  labels={labels}")
    logger.debug(f"[intermarket] {summary}")

    _save_cache(ctx)
    return ctx
