"""
Post-Earnings Announcement Drift (PEAD) signal.

Background
──────────
PEAD is one of the longest-replicated anomalies in academic finance: stocks
that beat (miss) earnings expectations tend to continue drifting in the
direction of the surprise for roughly 60 days after the announcement. The
market under-reacts to the information at announcement, and the drift
provides a tradable edge until the price fully absorbs the news.

Formula (per ticker)
────────────────────
    sue_normalized = tanh(surprise_pct / surprise_scale_pct)        ∈ [-1, +1]
    time_decay     = max(0, 1 - days_since_report / decay_window)   ∈ [0, 1]
    pead_score     = sue_normalized × time_decay                    ∈ [-1, +1]

Defaults:
  surprise_scale_pct = 25  → ±25% surprise saturates near ±0.76; a typical
                             ±10% surprise yields ±0.38 before decay.
  decay_window       = 60  → score fades linearly to 0 over 60 calendar days.

Worked example:
  AAPL reports +15% surprise 14 days ago →
    sue_norm = tanh(0.60)  = +0.537
    decay    = 1 - 14/60   = +0.767
    score    = 0.537 × 0.767 = +0.412   (mild bullish drift remaining)

Data source
───────────
yfinance ``Ticker.earnings_dates`` — same source as ``fetch_earnings_surprises``
but consumed structurally (per-ticker dict) rather than as articles. Cached
daily at ``cache/pead_YYYY-MM-DD.json``.

Returns ``None`` when ``ENABLE_FETCH_DATA`` is false and no cache exists, or
when no ticker has a reported earnings event within the decay window.
"""

from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import yfinance as yf
from loguru import logger

from config import settings
from src.models import PEADContext, PEADSignal


CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.35   # yfinance rate-limit buffer (matches earnings.py)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"pead_{date.today().isoformat()}.json"


def _load_cache() -> Optional[PEADContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = PEADContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(f"[pead] Loaded {len(ctx.signals)} PEAD signal(s) from cache")
        return ctx
    except Exception as e:
        logger.warning(f"[pead] cache load failed: {e}")
        return None


def _save_cache(ctx: PEADContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[pead] cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pead_score(surprise_pct: float, days_since_report: int,
                surprise_scale_pct: float, decay_window_days: int) -> Tuple[float, float, float]:
    """Return ``(pead_score, sue_normalized, time_decay)``.

    sue_normalized uses ``tanh`` for graceful saturation: ±25% surprise →
    ±0.76; ±100% → ±0.999. time_decay is linear so the score is easy to
    reason about ("half-life" at decay_window/2).
    """
    if decay_window_days <= 0 or surprise_scale_pct <= 0:
        return 0.0, 0.0, 0.0
    sue_norm = math.tanh(surprise_pct / surprise_scale_pct)
    decay = max(0.0, 1.0 - days_since_report / decay_window_days)
    return sue_norm * decay, sue_norm, decay


def _direction(score: float) -> str:
    if score >= 0.15:
        return "BULLISH"
    if score <= -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _build_signal(ticker: str, report_dt: date, actual: Optional[float],
                   estimate: Optional[float], surprise_pct: float, today: date,
                   surprise_scale_pct: float, decay_window_days: int) -> PEADSignal:
    days = (today - report_dt).days
    score, sue, decay = _pead_score(surprise_pct, days, surprise_scale_pct, decay_window_days)
    direction = _direction(score)
    summary = (
        f"{ticker}: {'+' if surprise_pct >= 0 else ''}{surprise_pct:.1f}% "
        f"EPS surprise {days}d ago → SUE={sue:+.2f} × decay={decay:.2f} = "
        f"PEAD={score:+.2f} ({direction.lower()} drift)"
    )
    return PEADSignal(
        ticker=ticker,
        report_date=report_dt,
        days_since_report=days,
        actual_eps=actual,
        estimated_eps=estimate,
        surprise_pct=round(surprise_pct, 2),
        sue_normalized=round(sue, 4),
        time_decay=round(decay, 4),
        pead_score=round(score, 4),
        direction=direction,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pead_context(
    tickers: List[str],
    decay_window_days: Optional[int] = None,
    surprise_scale_pct: Optional[float] = None,
) -> Optional[PEADContext]:
    """Build a ``PEADContext`` with per-ticker drift signals.

    For each ticker, fetches the most recent reported earnings within the
    decay window and computes (sue × decay). Tickers without a recent report
    or with insufficient data are silently omitted.

    Cached daily — within the same date a re-run is a one-line cache hit.
    """
    decay_window_days   = decay_window_days   or settings.pead_decay_window_days
    surprise_scale_pct  = surprise_scale_pct  or settings.pead_surprise_scale_pct

    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[pead] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    today = date.today()
    cutoff = today - timedelta(days=decay_window_days)
    signals: List[PEADSignal] = []

    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            ed = t.earnings_dates

            if ed is None or ed.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Past reports only (both Estimate and Reported EPS populated)
            reported = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
            if reported.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            in_window = reported[reported.index.date >= cutoff]
            if in_window.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Most recent report (yfinance sorts descending → first row)
            row         = in_window.iloc[0]
            report_dt   = in_window.index[0].date()
            actual      = _safe_float(row.get("Reported EPS"))
            estimate    = _safe_float(row.get("EPS Estimate"))
            if actual is None or estimate is None:
                time.sleep(_REQUEST_DELAY)
                continue

            surprise_pct = _safe_float(row.get("Surprise(%)"))
            if surprise_pct is None:
                if abs(estimate) < 0.01:
                    time.sleep(_REQUEST_DELAY)
                    continue
                surprise_pct = (actual - estimate) / abs(estimate) * 100.0

            sig = _build_signal(
                sym, report_dt, actual, estimate, surprise_pct,
                today, surprise_scale_pct, decay_window_days,
            )
            # Skip fully-decayed signals (no remaining edge)
            if abs(sig.pead_score) >= 0.01:
                signals.append(sig)

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[pead] {sym} fetch failed: {e}")
            time.sleep(_REQUEST_DELAY)

    if not signals:
        logger.info("[pead] No PEAD signals in window")
        return None

    # Sort by absolute score and surface top movers
    signals_sorted_desc = sorted(signals, key=lambda s: s.pead_score, reverse=True)
    top_bull = [s.ticker for s in signals_sorted_desc if s.pead_score > 0][:5]
    top_bear = [s.ticker for s in signals_sorted_desc if s.pead_score < 0][-5:][::-1]

    n_bull = sum(1 for s in signals if s.direction == "BULLISH")
    n_bear = sum(1 for s in signals if s.direction == "BEARISH")
    summary = (
        f"{len(signals)} ticker(s) with active PEAD drift: {n_bull} bullish, "
        f"{n_bear} bearish. "
        f"Score = tanh(surprise%/{surprise_scale_pct:.0f}) * max(0, 1 - days/{decay_window_days})."
    )
    if top_bull:
        summary += f" Top bullish drift: {', '.join(top_bull[:3])}."
    if top_bear:
        summary += f" Top bearish drift: {', '.join(top_bear[:3])}."

    ctx = PEADContext(
        signals=signals,
        report_date=today,
        decay_window_days=decay_window_days,
        surprise_scale_pct=surprise_scale_pct,
        top_drift_bullish=top_bull,
        top_drift_bearish=top_bear,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(f"[pead] {len(signals)} active drift signal(s) — {summary}")
    return ctx
