"""Intraday (30-min) timing overlay for the Hybrid model.

The daily signal stack (computed on *completed* daily bars) decides DIRECTION and
conviction. This module adds a short-horizon read from 30-minute bars that only
affects *timing*:

  * Entry gate — defer opening a position that would fire against strong intraday
    momentum (e.g. a BUY while the 30-min trend is falling hard). The next 30-min
    tick re-evaluates, so the trade simply waits for a less hostile entry.
  * Exit accelerator (opt-in) — close an open position when the intraday trend
    reverses hard against it.

It never changes the daily-signal direction. Score ∈ [-1, +1]: positive = intraday
up-momentum, negative = down-momentum. Fail-graceful: returns ``None`` on any data
or parse error so the pipeline is never blocked by it.
"""

from __future__ import annotations

import math
from typing import Optional

import yfinance as yf
from loguru import logger

from config import settings

_INTERVAL = "30m"
_PERIOD = "5d"     # ~65 thirty-min bars — enough for the slow EMA + recent context
_FAST = 5          # ~2.5 trading hours
_SLOW = 20         # ~1.5 trading days
_TANH_GAIN = 50.0  # maps an EMA spread of ~1% of price to a ~0.46 score


def compute_intraday_timing(ticker: str) -> Optional[dict]:
    """Return a 30-min momentum/timing read for *ticker*, or ``None`` on failure.

    ``{"score": float[-1,1], "classification": RISING|FALLING|FLAT,
       "last_price": float, "ret_30m": pct}``.
    """
    if not settings.enable_fetch_data:
        return None
    try:
        df = yf.Ticker(ticker).history(period=_PERIOD, interval=_INTERVAL)
    except Exception as e:
        logger.debug(f"[intraday] fetch failed for {ticker}: {e}")
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None

    closes = df["Close"].dropna()
    if len(closes) < _SLOW + 1:
        return None

    last = float(closes.iloc[-1])
    fast = float(closes.ewm(span=_FAST, adjust=False).mean().iloc[-1])
    slow = float(closes.ewm(span=_SLOW, adjust=False).mean().iloc[-1])
    if last <= 0 or slow <= 0:
        return None

    # Fast-vs-slow EMA spread, normalised by price and squashed to [-1, 1].
    score = math.tanh((fast - slow) / slow * _TANH_GAIN)
    prev = float(closes.iloc[-2])
    ret_30m = (last - prev) / prev * 100.0 if prev > 0 else 0.0

    if score > 0.15:
        cls = "RISING"
    elif score < -0.15:
        cls = "FALLING"
    else:
        cls = "FLAT"

    return {
        "score": round(score, 3),
        "classification": cls,
        "last_price": last,
        "ret_30m": round(ret_30m, 3),
    }


def opposes_entry(action: str, timing: Optional[dict], threshold: float) -> bool:
    """True when intraday momentum is strongly against opening *action* now.

    BUY opposed by a falling 30-min trend (score ≤ −threshold); SELL opposed by a
    rising trend (score ≥ +threshold). ``None``/missing timing never opposes.
    """
    if not timing:
        return False
    s = timing.get("score")
    if s is None:
        return False
    if action == "BUY":
        return s <= -abs(threshold)
    if action == "SELL":
        return s >= abs(threshold)
    return False


def reverses_position(action: str, timing: Optional[dict], threshold: float) -> bool:
    """True when intraday momentum has reversed hard against an open *action*.

    Same geometry as ``opposes_entry`` but used for the (opt-in) exit accelerator.
    """
    return opposes_entry(action, timing, threshold)
