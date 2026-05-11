"""Sector rotation / money-flow monitor — "Ebb and Flow" mechanism.

Money acts like water: when it floods into one sector it is usually exiting another.
This module tracks capital rotation across the 11 SPDR sector ETFs relative to SPY:

  • Per-sector rotation score ∈ [-1, +1]:
      Cross-sectional z-score of a weighted composite of 1-week, 1-month, and
      3-month excess returns vs SPY, adjusted upward when volume is elevated
      (confirming accumulation) or downward when volume accompanies outflows.

  • Flow signals:
      STRONG_INFLOW  (score ≥ +0.5): money actively flooding in
      INFLOW         (score ≥ +0.2): meaningful relative inflow
      NEUTRAL                       : no directional flow conviction
      OUTFLOW        (score ≤ -0.2): money leaving
      STRONG_OUTFLOW (score ≤ -0.5): significant capital exodus

  • Rotation regime (cyclical vs defensive balance):
      RISK_ON  (cyclical avg − defensive avg > +0.2): growth / cyclical rotation dominant
      NEUTRAL                                        : mixed or no clear bias
      RISK_OFF (spread < -0.2)                      : defensive rotation dominant

  • Explicit rotation pairs: e.g. "XLK → XLP (Technology → Consumer Staples)"

Data: 3-month OHLCV via market_data.get_history() (cached, no extra API key).
Cache: daily — cache/sector_rotation_YYYY-MM-DD.json
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config import settings
from src.models import SectorRotationContext, SectorRotationEntry

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "sector_rotation_"

_SECTOR_NAMES: Dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

# Cyclical sectors: benefit from economic expansion
_CYCLICAL_ETFS: frozenset = frozenset({"XLK", "XLF", "XLY", "XLI", "XLB", "XLC", "XLE"})
# Defensive sectors: sought in slowdowns / risk-off
_DEFENSIVE_ETFS: frozenset = frozenset({"XLV", "XLP", "XLU", "XLRE"})


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[SectorRotationContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = SectorRotationContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(
            f"[sector_rotation] Loaded from cache — "
            f"{ctx.rotation_regime}  inflow={ctx.top_inflow}  outflow={ctx.top_outflow}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[sector_rotation] Cache load failed: {e}")
        return None


def _save_cache(ctx: SectorRotationContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[sector_rotation] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_change(series: pd.Series, n: int) -> Optional[float]:
    s = series.dropna()
    if len(s) <= n:
        return None
    start = float(s.iloc[-(n + 1)])
    end   = float(s.iloc[-1])
    if start == 0:
        return None
    return round((end / start - 1) * 100, 3)


def _vol_ratio(volume: pd.Series) -> Optional[float]:
    """5-day avg volume / 20-day avg volume (excluding the 5 most recent bars from the base)."""
    v = volume.dropna()
    if len(v) < 22:
        return None
    recent = float(v.iloc[-5:].mean())
    base   = float(v.iloc[-22:-5].mean())
    if base <= 0:
        return None
    return round(recent / base, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sector_rotation_context() -> Optional[SectorRotationContext]:
    """Compute per-sector money-flow rotation scores and return a SectorRotationContext."""
    today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    # Lazy import to avoid circular dependency (market_data imports models)
    from src.data.market_data import get_history

    sector_etfs = settings.sectors_list  # ["XLK", "XLF", ...]
    needed      = sector_etfs + ["SPY"]

    histories: Dict[str, pd.DataFrame] = {}
    for ticker in needed:
        try:
            df = get_history(ticker, period="3mo")
            if df is not None and not df.empty and len(df) >= 10:
                histories[ticker] = df
        except Exception as exc:
            logger.debug(f"[sector_rotation] {ticker}: {exc}")

    if not histories:
        logger.warning("[sector_rotation] No OHLCV data available — skipping")
        return None

    # SPY reference for relative-return calculation
    spy_close  = histories.get("SPY", pd.DataFrame()).get("Close", pd.Series())
    spy_ret_5d  = _pct_change(spy_close, 5)  if not spy_close.empty else None
    spy_ret_21d = _pct_change(spy_close, 21) if not spy_close.empty else None
    spy_ret_63d = _pct_change(spy_close, min(63, max(0, len(spy_close) - 2))) if not spy_close.empty else None

    entries: List[SectorRotationEntry] = []
    for etf in sector_etfs:
        if etf not in histories:
            continue
        df     = histories[etf]
        close  = df.get("Close",  pd.Series())
        volume = df.get("Volume", pd.Series())

        n_bars = max(0, len(close.dropna()) - 2)

        ret_5d  = _pct_change(close, 5)
        ret_21d = _pct_change(close, 21)
        ret_63d = _pct_change(close, min(63, n_bars))

        def _rel(ret, spy_ret):
            if ret is None or spy_ret is None:
                return None
            return round(ret - spy_ret, 3)

        entries.append(SectorRotationEntry(
            etf=etf,
            name=_SECTOR_NAMES.get(etf, etf),
            return_5d=ret_5d,
            return_21d=ret_21d,
            return_63d=ret_63d,
            relative_5d=_rel(ret_5d,  spy_ret_5d),
            relative_21d=_rel(ret_21d, spy_ret_21d),
            relative_63d=_rel(ret_63d, spy_ret_63d),
            volume_ratio=_vol_ratio(volume) if not volume.empty else None,
        ))

    if not entries:
        logger.warning("[sector_rotation] No sector entries computed — skipping")
        return None

    # ── Weighted composite relative return (or absolute if SPY unavailable) ──
    raw_scores: List[float] = []
    for e in entries:
        parts = []
        if e.relative_5d  is not None: parts.append((0.50, e.relative_5d))
        if e.relative_21d is not None: parts.append((0.30, e.relative_21d))
        if e.relative_63d is not None: parts.append((0.20, e.relative_63d))
        if not parts:
            # fallback to absolute returns if SPY unavailable
            if e.return_5d  is not None: parts.append((0.50, e.return_5d))
            if e.return_21d is not None: parts.append((0.30, e.return_21d))
            if e.return_63d is not None: parts.append((0.20, e.return_63d))
        if parts:
            total_w = sum(w for w, _ in parts)
            raw_scores.append(sum(w * v for w, v in parts) / total_w)
        else:
            raw_scores.append(0.0)

    # ── Cross-sectional z-score normalisation ──
    n = len(raw_scores)
    mean_ = sum(raw_scores) / n
    std_  = (sum((v - mean_) ** 2 for v in raw_scores) / n) ** 0.5

    for i, entry in enumerate(entries):
        z = (raw_scores[i] - mean_) / std_ if std_ > 1e-9 else 0.0
        # Volume modifier: confirmed accumulation/distribution amplifies signal
        vol_mod = 0.0
        if entry.volume_ratio is not None and entry.volume_ratio > 1.15:
            boost = min(0.25, (entry.volume_ratio - 1.0) * 0.6)
            vol_mod = boost if z >= 0 else -boost

        score = max(-1.0, min(1.0, z / 2.0 + vol_mod))
        entry.rotation_score = round(score, 3)

        if   score >=  0.5: entry.flow_signal, entry.direction = "STRONG_INFLOW",  "BULLISH"
        elif score >=  0.2: entry.flow_signal, entry.direction = "INFLOW",          "BULLISH"
        elif score <= -0.5: entry.flow_signal, entry.direction = "STRONG_OUTFLOW",  "BEARISH"
        elif score <= -0.2: entry.flow_signal, entry.direction = "OUTFLOW",          "BEARISH"
        else:               entry.flow_signal, entry.direction = "NEUTRAL",          "NEUTRAL"

    entries.sort(key=lambda e: e.rotation_score, reverse=True)

    top_inflow  = [e.etf for e in entries if e.flow_signal in ("STRONG_INFLOW",  "INFLOW")][:3]
    top_outflow = [e.etf for e in entries if e.flow_signal in ("STRONG_OUTFLOW", "OUTFLOW")][:3]

    # ── Rotation regime: cyclical vs defensive balance ──
    cyc_scores = [e.rotation_score for e in entries if e.etf in _CYCLICAL_ETFS]
    def_scores = [e.rotation_score for e in entries if e.etf in _DEFENSIVE_ETFS]
    cyc_avg  = round(sum(cyc_scores) / len(cyc_scores), 3)  if cyc_scores  else 0.0
    def_avg  = round(sum(def_scores) / len(def_scores), 3)  if def_scores  else 0.0
    spread   = round(cyc_avg - def_avg, 3)

    if   spread > 0.20:  rotation_regime, rotation_direction = "RISK_ON",  "BULLISH"
    elif spread < -0.20: rotation_regime, rotation_direction = "RISK_OFF", "BEARISH"
    else:                rotation_regime, rotation_direction = "NEUTRAL",  "NEUTRAL"

    # ── Rotation pairs ──
    rotation_pairs: List[str] = []
    if top_inflow and top_outflow:
        seen = set()
        for out_etf in top_outflow:
            for in_etf in top_inflow:
                if (out_etf, in_etf) not in seen and out_etf != in_etf:
                    out_name = _SECTOR_NAMES.get(out_etf, out_etf)
                    in_name  = _SECTOR_NAMES.get(in_etf,  in_etf)
                    rotation_pairs.append(f"{out_etf} → {in_etf}  ({out_name} → {in_name})")
                    seen.add((out_etf, in_etf))
            if len(rotation_pairs) >= 3:
                break

    inflow_str  = ", ".join(top_inflow)  if top_inflow  else "none"
    outflow_str = ", ".join(top_outflow) if top_outflow else "none"
    summary = (
        f"{rotation_regime}: capital flowing into [{inflow_str}], "
        f"exiting [{outflow_str}]. "
        f"Cyclical avg={cyc_avg:+.2f}, defensive avg={def_avg:+.2f} "
        f"(spread={spread:+.2f})."
    )

    ctx = SectorRotationContext(
        sectors=entries,
        top_inflow=top_inflow,
        top_outflow=top_outflow,
        rotation_regime=rotation_regime,
        rotation_direction=rotation_direction,
        cyclical_avg=cyc_avg,
        defensive_avg=def_avg,
        cyc_def_spread=spread,
        rotation_pairs=rotation_pairs,
        report_date=today,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(
        f"[sector_rotation] {rotation_regime}  "
        f"inflow=[{inflow_str}]  outflow=[{outflow_str}]  spread={spread:+.2f}"
    )
    return ctx
