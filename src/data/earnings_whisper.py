"""
Earnings Whisper vs. Consensus Gap.

Estimates the implied "whisper number" — the buy-side's true EPS expectation,
which often differs from the published sell-side consensus.

Key insight: companies that consistently beat EPS estimates have the beat "baked in"
to the market's true expectation. A stock that beats the printed consensus but falls
short of the implied whisper will sell off even on a headline beat.

Signals derived from three free yfinance sources (no paid API required):
  1. Historical beat/miss record  — ticker.earnings_dates (last 4–8 quarters)
  2. Consensus revision trend     — ticker.eps_trend (how estimate moved over 7/30/60/90d)
  3. Net analyst revisions        — ticker.eps_revisions (up/down count over 7d/30d)

Methodology:
  - avg_eps_surprise_pct = mean of historical Surprise(%) over last N quarters
  - implied_whisper      = current_consensus × (1 + avg_eps_surprise_pct / 100)
    → the level the buy-side actually expects, based on historical patterns
  - whisper_gap_pct      = avg_eps_surprise_pct (the % by which whisper exceeds consensus)
  - Consensus trend: if the estimate is higher today than 30 days ago, the market is
    already chasing up toward the whisper — the effective bar is even higher.
  - Signal classification: beat_rate × avg_beat_magnitude × revision_direction

Only returns WhisperSignal entries for tickers that have:
  (a) upcoming earnings within upcoming_days, OR
  (b) sufficient historical beat/miss data (≥ 2 quarters)

Cached daily to cache/whisper_YYYY-MM-DD.json.
"""

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import WhisperContext, WhisperSignal

CACHE_DIR      = Path("cache")
_REQUEST_DELAY = 0.80   # higher than earnings.py (0.35) to avoid rate-limit collisions
_UPCOMING_DAYS = 21     # look-ahead window for upcoming earnings


def _cache_path() -> Path:
    return CACHE_DIR / f"whisper_{date.today().isoformat()}.json"


def _load_cache() -> Optional[WhisperContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = WhisperContext.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        logger.info(f"[whisper] Loaded cached whisper context ({len(ctx.signals)} signals)")
        return ctx
    except Exception as e:
        logger.warning(f"[whisper] Cache load failed: {e}")
        return None


def _save_cache(ctx: WhisperContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[whisper] Cache save failed: {e}")


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _get_df_val(df, row_options, col_options) -> Optional[float]:
    """Try multiple row/column label combinations; return first non-None value."""
    for r in row_options:
        for c in col_options:
            try:
                val = df.loc[r, c]
                f = _safe_float(val)
                if f is not None:
                    return f
            except (KeyError, TypeError):
                continue
    return None


def _fetch_ticker_whisper(sym: str, today: date, cutoff: date) -> Optional[WhisperSignal]:
    """
    Fetch whisper-proxy data for a single ticker.

    Returns None if there's insufficient data to compute a meaningful signal.
    """
    t = yf.Ticker(sym)

    # ── Historical beat/miss record ──────────────────────────────────────────
    earnings_date: Optional[date] = None
    days_until: Optional[int] = None
    current_eps_estimate: Optional[float] = None
    quarters_analyzed = 0
    beat_count = 0
    miss_count = 0
    avg_eps_surprise_pct = 0.0

    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            # ── Upcoming earnings date (future rows have no Reported EPS) ─────
            future_mask = ed.index.date > today
            if future_mask.any():
                future = ed[future_mask]
                if "Reported EPS" in future.columns:
                    future = future[future["Reported EPS"].isna()]
                if not future.empty:
                    # yfinance sorts descending → last row = earliest future date
                    earliest_date = future.index[-1].date()
                    if earliest_date <= cutoff:
                        earnings_date = earliest_date
                        days_until = (earliest_date - today).days
                        # Also grab the EPS estimate for that upcoming quarter
                        est_val = _safe_float(future.iloc[-1].get("EPS Estimate"))
                        if est_val is not None:
                            current_eps_estimate = est_val

            # ── Past quarters: beat/miss history ─────────────────────────────
            if "Reported EPS" in ed.columns:
                past = ed[ed["Reported EPS"].notna()].head(8)  # last 8 quarters
                quarters_analyzed = len(past)
                if quarters_analyzed > 0:
                    surprises = past["Surprise(%)"].dropna() if "Surprise(%)" in past.columns else None
                    if surprises is not None and not surprises.empty:
                        beat_count = int((past["Surprise(%)"] > 0).sum())
                        miss_count = quarters_analyzed - beat_count
                        avg_eps_surprise_pct = round(float(surprises.mean()), 2)
    except Exception as e:
        logger.debug(f"[whisper] {sym} earnings_dates failed: {e}")

    # ── Consensus estimate + revision trend ──────────────────────────────────
    eps_trend_current: Optional[float] = None
    eps_trend_7d: Optional[float] = None
    eps_trend_30d: Optional[float] = None
    eps_trend_direction = "STABLE"
    revisions_up_30d = 0
    revisions_down_30d = 0

    try:
        trend = t.eps_trend
        if trend is not None and not trend.empty:
            # eps_trend: index = period (0q/+1q/0y/+1y), columns = (current/7daysAgo/...)
            # OR transposed. Be defensive.
            curr_col_opts = ["current", "Current", "0"]
            ago7_col_opts = ["7daysAgo", "7DaysAgo", "7 Days Ago"]
            ago30_col_opts = ["30daysAgo", "30DaysAgo", "30 Days Ago"]
            qtr_row_opts  = ["0q", "0Q", "currentQtr"]

            # If the period labels are in COLUMNS (transposed layout)
            if any(r in trend.columns for r in curr_col_opts):
                eps_trend_current = _get_df_val(trend, qtr_row_opts, curr_col_opts)
                eps_trend_7d      = _get_df_val(trend, qtr_row_opts, ago7_col_opts)
                eps_trend_30d     = _get_df_val(trend, qtr_row_opts, ago30_col_opts)
            # If the period labels are in INDEX (standard layout)
            elif any(r in trend.index for r in curr_col_opts):
                eps_trend_current = _get_df_val(trend, curr_col_opts, qtr_row_opts)
                eps_trend_7d      = _get_df_val(trend, ago7_col_opts, qtr_row_opts)
                eps_trend_30d     = _get_df_val(trend, ago30_col_opts, qtr_row_opts)

            # Override current_eps_estimate with eps_trend if we got it
            if eps_trend_current is not None:
                current_eps_estimate = eps_trend_current

            # Compute revision direction
            if eps_trend_current is not None and eps_trend_30d is not None and eps_trend_30d != 0:
                pct_chg = (eps_trend_current - eps_trend_30d) / abs(eps_trend_30d) * 100
                if pct_chg > 1.5:
                    eps_trend_direction = "REVISING_UP"
                elif pct_chg < -1.5:
                    eps_trend_direction = "REVISING_DOWN"
    except Exception as e:
        logger.debug(f"[whisper] {sym} eps_trend failed: {e}")

    try:
        revisions = t.eps_revisions
        if revisions is not None and not revisions.empty:
            up30_opts   = ["upLast30days",   "upLast30Days"]
            down30_opts = ["downLast30days", "downLast30Days"]
            qtr_opts    = ["0q", "0Q", "currentQtr"]

            # Determine orientation same way as eps_trend
            if any(r in revisions.columns for r in up30_opts):
                # period in columns
                up   = _get_df_val(revisions, qtr_opts, up30_opts)
                down = _get_df_val(revisions, qtr_opts, down30_opts)
            else:
                up   = _get_df_val(revisions, up30_opts,   qtr_opts)
                down = _get_df_val(revisions, down30_opts, qtr_opts)

            if up is not None:
                revisions_up_30d = int(up)
            if down is not None:
                revisions_down_30d = int(down)
    except Exception as e:
        logger.debug(f"[whisper] {sym} eps_revisions failed: {e}")

    # ── Need at least some data ───────────────────────────────────────────────
    has_beat_data    = quarters_analyzed >= 2
    has_estimate     = current_eps_estimate is not None
    has_upcoming     = earnings_date is not None

    # Skip if we have nothing meaningful at all
    if not has_beat_data and not has_estimate:
        return None

    # ── Implied whisper computation ───────────────────────────────────────────
    implied_whisper: Optional[float] = None
    whisper_gap_pct = avg_eps_surprise_pct   # by definition: avg_eps_surprise_pct IS the gap

    if has_estimate and has_beat_data and current_eps_estimate != 0:
        implied_whisper = round(current_eps_estimate * (1 + avg_eps_surprise_pct / 100), 4)

    # ── Beat rate ─────────────────────────────────────────────────────────────
    beat_rate_pct = round(beat_count / max(1, quarters_analyzed) * 100, 1)

    # ── Signal classification ─────────────────────────────────────────────────
    not_revising_down = eps_trend_direction in ("REVISING_UP", "STABLE")
    net_revisions_pos = revisions_up_30d >= revisions_down_30d

    if (beat_rate_pct >= 75 and avg_eps_surprise_pct >= 3.0
            and not_revising_down and net_revisions_pos):
        signal = "BEAT_LIKELY"
        direction = "BULLISH"
    elif (beat_rate_pct >= 60 or avg_eps_surprise_pct >= 1.5
          or (eps_trend_direction == "REVISING_UP" and net_revisions_pos)):
        signal = "BEAT_POSSIBLE"
        direction = "BULLISH"
    elif (beat_rate_pct < 30 or avg_eps_surprise_pct <= -3.0
          or (eps_trend_direction == "REVISING_DOWN" and revisions_down_30d > revisions_up_30d)):
        signal = "MISS_LIKELY"
        direction = "BEARISH"
    elif (beat_rate_pct < 45 or avg_eps_surprise_pct <= -1.0
          or eps_trend_direction == "REVISING_DOWN"):
        signal = "MISS_POSSIBLE"
        direction = "BEARISH"
    else:
        signal = "NEUTRAL"
        direction = "NEUTRAL"

    # ── Human summary ─────────────────────────────────────────────────────────
    parts = []
    if has_beat_data:
        parts.append(
            f"{beat_rate_pct:.0f}% beat rate ({beat_count}/{quarters_analyzed}q, "
            f"avg surprise {avg_eps_surprise_pct:+.1f}%)"
        )
    if has_estimate:
        parts.append(f"consensus ${current_eps_estimate:.2f}")
    if implied_whisper is not None and implied_whisper != current_eps_estimate:
        parts.append(f"implied whisper ${implied_whisper:.2f} ({whisper_gap_pct:+.1f}%)")
    if eps_trend_direction != "STABLE":
        trend_verb = "rising" if eps_trend_direction == "REVISING_UP" else "falling"
        parts.append(f"consensus {trend_verb} vs 30d ago")
    if revisions_up_30d + revisions_down_30d > 0:
        parts.append(f"{revisions_up_30d}↑/{revisions_down_30d}↓ revisions (30d)")
    if has_upcoming:
        parts.append(f"reports in {days_until}d")

    summary = f"{sym}: {signal} — " + "; ".join(parts) + "." if parts else f"{sym}: {signal}."

    return WhisperSignal(
        ticker=sym,
        earnings_date=earnings_date,
        days_until_earnings=days_until,
        current_eps_estimate=current_eps_estimate,
        quarters_analyzed=quarters_analyzed,
        beat_count=beat_count,
        miss_count=miss_count,
        beat_rate_pct=beat_rate_pct,
        avg_eps_surprise_pct=avg_eps_surprise_pct,
        implied_whisper=implied_whisper,
        whisper_gap_pct=whisper_gap_pct if has_beat_data else None,
        eps_trend_current=eps_trend_current,
        eps_trend_7d=eps_trend_7d,
        eps_trend_30d=eps_trend_30d,
        eps_trend_direction=eps_trend_direction,
        revisions_up_30d=revisions_up_30d,
        revisions_down_30d=revisions_down_30d,
        signal=signal,
        direction=direction,
        summary=summary,
    )


def fetch_whisper_context(
    tickers: List[str],
    upcoming_days: int = _UPCOMING_DAYS,
) -> Optional[WhisperContext]:
    """
    Compute whisper-proxy signals for the watchlist.

    Returns a WhisperContext with per-ticker signals for tickers that have
    either upcoming earnings or sufficient historical beat/miss data.

    Caches the result for the current calendar day.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[whisper] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    today  = date.today()
    cutoff = today + timedelta(days=upcoming_days)
    signals: List[WhisperSignal] = []

    for sym in tickers:
        try:
            ws = _fetch_ticker_whisper(sym, today, cutoff)
            if ws is not None:
                signals.append(ws)
                logger.debug(
                    f"[whisper] {sym}: {ws.signal} | beat_rate={ws.beat_rate_pct:.0f}% "
                    f"avg_surprise={ws.avg_eps_surprise_pct:+.1f}% "
                    f"trend={ws.eps_trend_direction}"
                    + (f" | earnings in {ws.days_until_earnings}d" if ws.days_until_earnings is not None else "")
                )
            time.sleep(_REQUEST_DELAY)
        except Exception as e:
            logger.debug(f"[whisper] {sym} failed: {e}")
            time.sleep(_REQUEST_DELAY)

    if not signals:
        logger.info("[whisper] No whisper signals found")
        return None

    # Aggregate stats
    n_beat_likely  = sum(1 for s in signals if s.signal == "BEAT_LIKELY")
    n_beat_poss    = sum(1 for s in signals if s.signal == "BEAT_POSSIBLE")
    n_miss_likely  = sum(1 for s in signals if s.signal == "MISS_LIKELY")
    n_miss_poss    = sum(1 for s in signals if s.signal == "MISS_POSSIBLE")
    n_neutral      = sum(1 for s in signals if s.signal == "NEUTRAL")

    upcoming       = [s for s in signals if s.days_until_earnings is not None]
    upcoming_names = [f"{s.ticker}({s.days_until_earnings}d)" for s in upcoming[:5]]

    avg_beat_rate = (
        sum(s.beat_rate_pct for s in signals if s.quarters_analyzed > 0) /
        max(1, sum(1 for s in signals if s.quarters_analyzed > 0))
    )

    summary_parts = []
    if upcoming:
        summary_parts.append(f"Upcoming earnings: {', '.join(upcoming_names)}.")
    summary_parts.append(
        f"Whisper signals across {len(signals)} tickers: "
        f"{n_beat_likely} BEAT_LIKELY, {n_beat_poss} BEAT_POSSIBLE, "
        f"{n_neutral} NEUTRAL, {n_miss_poss} MISS_POSSIBLE, {n_miss_likely} MISS_LIKELY. "
        f"Average historical beat rate: {avg_beat_rate:.0f}%."
    )

    ctx = WhisperContext(
        signals=signals,
        n_beat_likely=n_beat_likely,
        n_beat_possible=n_beat_poss,
        n_miss_possible=n_miss_poss,
        n_miss_likely=n_miss_likely,
        avg_beat_rate_pct=round(avg_beat_rate, 1),
        report_date=today,
        summary=" ".join(summary_parts),
    )
    logger.info(
        f"[whisper] {len(signals)} signals | avg_beat={avg_beat_rate:.0f}% | "
        f"{n_beat_likely}BL {n_beat_poss}BP {n_miss_poss}MP {n_miss_likely}ML"
    )
    _save_cache(ctx)
    return ctx
