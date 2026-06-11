"""Extended-session gap momentum (``ext_gap``) — pre/after-market repricing signal.

The one signal that only exists outside RTH: how far the live extended-session
print has moved away from the last COMPLETED daily close, in units of the
ticker's own ATR. Pre-market this is the overnight gap forming in real time
(gap-and-go candidates); after-hours it is the immediate reaction to a
just-released catalyst (earnings, 8-K, guidance) — the move the next RTH open
will inherit.

Design
──────
* RTH ticks return ``0.0`` ("no view"): the open gap is already captured by
  the daily technical stack, and a second gap reading would double-count it.
  Zero scores are excluded from the signal panel's IC analysis by convention.
* Reference close: pre-market (and overnight before noon) → the previous
  market day's close; after the 16:00 close → today's completed close. The
  OHLCV cache only ever stores completed bars (``save_ohlcv`` runs after
  ``_drop_forming_bar``), so the row AT the expected reference date is the
  official close — if that exact row is missing the cache is stale and the
  scorer fails CLOSED (0.0) rather than measure a phantom gap against an old
  close.
* Normalisation: gap% / ATR%(14, Wilder). A fixed-percent scale would make
  the same 2% gap scream on KO and whisper on TSLA; ATR units make the score
  mean "how unusual is this move for THIS name".
* Deadband + tanh: |gap| < ``extended_gap_deadband_atr`` ATR → 0.0 (micro
  drift on a thin book is noise, not signal); above it the score is
  ``tanh(gap_atr / extended_gap_scale_atr)`` — monotone momentum-following.
  Gap-fade risk is handled by the architecture, not the scorer: a lone
  unconfirmed gap can never trigger a BUY/SELL (convergence rules), while a
  news-confirmed gap composes with the up-weighted news methods.

Cache-first like the other OHLCV scorers: works with ENABLE_FETCH_DATA=false
once the cache is warm; never fetches on its own.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Tuple

from loguru import logger

from config import settings
from src.data.cache import load_ohlcv
from src.performance.market_calendar import (
    MARKET_CLOSE_LOCAL,
    NY_TZ,
    current_session,
    is_market_day,
    previous_market_day,
)


def _reference_close_date(now_et: datetime):
    """Date of the last COMPLETED daily bar for a clock instant.

    Before the 16:00 close (pre-market, or the after-midnight tail of the
    overnight session) the last completed session is the previous market day;
    from the close onward it is today itself (when today is a session).
    """
    d = now_et.date()
    if now_et.time() >= MARKET_CLOSE_LOCAL and is_market_day(d):
        return d
    return previous_market_day(d)


def compute_extended_gap_score(
    ticker: str,
    live_price: Optional[float],
    session: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Tuple[float, float]:
    """Return ``(score, gap_pct)`` for *ticker* at the live extended print.

    ``score`` ∈ [-1, +1] (0.0 = no view: RTH run, missing/stale data, or gap
    inside the deadband); ``gap_pct`` is the raw move vs the reference close
    in percent (kept for display even when the score deadbands to 0).
    """
    if not settings.enable_extended_gap:
        return 0.0, 0.0
    sess = session or current_session(now)
    if sess == "rth":
        return 0.0, 0.0
    if live_price is None or live_price <= 0:
        return 0.0, 0.0

    bars = load_ohlcv(ticker)
    if bars is None or bars.empty or "Close" not in bars.columns:
        return 0.0, 0.0

    now_et = now or datetime.now(NY_TZ)
    now_et = now_et.astimezone(NY_TZ) if now_et.tzinfo is not None else now_et
    ref_date = _reference_close_date(now_et)

    # The completed close AT the expected reference date — fail closed when
    # the cache hasn't caught up (a gap vs a 3-day-old close is meaningless).
    day_rows = bars[[ts.date() == ref_date for ts in bars.index]]
    if day_rows.empty:
        logger.debug(
            f"[ext_gap] {ticker}: no cached close for reference session {ref_date} — no view"
        )
        return 0.0, 0.0
    ref_close = float(day_rows["Close"].iloc[-1])
    if ref_close <= 0:
        return 0.0, 0.0

    gap_pct = (float(live_price) - ref_close) / ref_close * 100.0

    atr_pct = _atr_pct(bars, ref_close)
    if atr_pct is None or atr_pct <= 0:
        return 0.0, round(gap_pct, 3)

    gap_atr = (gap_pct / 100.0) / atr_pct
    if abs(gap_atr) < float(settings.extended_gap_deadband_atr):
        return 0.0, round(gap_pct, 3)

    scale = max(0.1, float(settings.extended_gap_scale_atr))
    score = math.tanh(gap_atr / scale)
    return round(score, 3), round(gap_pct, 3)


def _atr_pct(bars, ref_close: float, period: int = 14) -> Optional[float]:
    """ATR(14, Wilder) as a fraction of the reference close.

    Falls back to the mean absolute close-to-close move when High/Low columns
    are unavailable (some cached series are close-only).
    """
    try:
        closes = bars["Close"].astype(float)
        if {"High", "Low"}.issubset(bars.columns):
            high = bars["High"].astype(float)
            low = bars["Low"].astype(float)
            prev_close = closes.shift(1)
            tr = (high - low).combine((high - prev_close).abs(), max).combine(
                (low - prev_close).abs(), max
            )
            tr = tr.dropna()
            if len(tr) < period:
                return None
            atr = float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
        else:
            rets = closes.diff().abs().dropna()
            if len(rets) < period:
                return None
            atr = float(rets.tail(period).mean())
        return atr / ref_close if atr > 0 else None
    except Exception as e:
        logger.debug(f"[ext_gap] ATR computation failed: {e}")
        return None
