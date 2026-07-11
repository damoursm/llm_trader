"""Classic cross-sectional anomalies — 52-week-high proximity, 12-1 momentum,
short-term reversal (2026-07-08, panel-first at weight 0).

Three of the most-replicated return anomalies in the academic literature, added
as PANEL-ONLY methods: scored on every ticker, persisted to the ``signals``
panel (IC / Sim win% / Sim ret%), and trade-attributed via ``_ALL_METHODS`` —
but carrying ZERO weight in ``combined_score`` and excluded from the
coherence/agreement pool until weeks of forward returns say they earn weight
(the same measure-first path the fundamentals factors took). Promotion later =
a ``_BASE_WEIGHTS`` entry + combine/coherence lines in the aggregator.

  hi52         — 52-week-high proximity (George & Hwang 2004, JF): nearness to
                 the 52-week high predicts continuation (anchoring /
                 underreaction) and subsumes much of plain momentum in their
                 head-to-heads. + near the high, − far below it.
  mom_12_1     — 12-1 time-series momentum (Jegadeesh & Titman 1993): trailing
                 ~11-month return SKIPPING the most recent month (which belongs
                 to the reversal effect below). Vol-normalised against the
                 ticker's own return distribution, same idiom as
                 ``price_momentum`` (which covers only 1m/3m).
  st_reversal  — short-term reversal (Lehmann 1990; Jegadeesh 1990): the prior
                 ~1-week return, SIGN-FLIPPED per the score convention
                 (mean-reversion bakes the reversal INTO the sign). Only
                 emitted on liquid names (20-day dollar-volume floor) — on
                 thin names "reversal" is mostly bid-ask bounce.

All three are DAILY-only by construction (a 30-min "52-week high" or weekly
"12-1" would be nonsense), so they are deliberately NOT in
``multi_timeframe.TECHNICAL_METHODS``. Scores ∈ [-1, +1]; 0.0 = no view
(insufficient history / below the liquidity floor / inside the deadband).
Cache-first OHLCV like every sibling scorer; works with ENABLE_FETCH_DATA=false
when the chart caches are warm.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.data.cache import load_ohlcv
from src.data.market_data import get_history

_HI52_WINDOW    = 252   # trading days in the reference year
_HI52_MIN_ROWS  = 200   # below this a "52-week high" is too truncated to anchor on
_HI52_MID_RATIO = 0.80  # ratio at which the score crosses zero
_HI52_SPAN      = 0.20  # ratio units from zero-cross to full score (0.80→0, 1.00→+1)

_M121_SKIP      = 21    # skip the most recent month (belongs to st_reversal)
_M121_LOOKBACK  = 252   # full trailing year required (strict — no truncated windows)
_M121_TANH      = 1.5   # same z→score scale as price_momentum

_REV_WINDOW     = 5     # ~1 trading week
_REV_MIN_ROWS   = 60    # enough bars for a meaningful 5-day return distribution
_REV_TANH       = 1.5
_REV_DEADBAND_Z = 0.25  # |z| below this → no view (a quiet week is not a signal)
_DIST_WINDOW    = 252   # trailing bars for the normalisation distributions
_DVOL_WINDOW    = 20    # 20-day average dollar volume for the liquidity floor


def _get_ohlcv(ticker: str, min_rows: int) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= min_rows:
        return cached
    return get_history(ticker, period="18mo")


def compute_high_52w_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
    """Return (score, ratio_pct) — proximity of the last close to the 52-week high.

    ratio = close / max(High over the trailing 252 bars); score is a linear map
    with the zero-cross at ``_HI52_MID_RATIO`` (0.80): at the high → +1.0,
    at 60% of the high → −1.0. Requires ≥200 bars (an honest reference year);
    returns (0.0, 0.0) otherwise — recent IPOs get no view rather than a
    truncated-window one.
    """
    if df is None:
        df = _get_ohlcv(ticker, _HI52_MIN_ROWS)
    if df is None or df.empty or len(df) < _HI52_MIN_ROWS or "Close" not in df.columns:
        logger.debug(f"[hi52] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, 0.0

    close = pd.to_numeric(df["Close"], errors="coerce")
    highs = (pd.to_numeric(df["High"], errors="coerce")
             if "High" in df.columns else close)
    last_close = float(close.dropna().iloc[-1])
    hi52 = float(highs.tail(_HI52_WINDOW).max())
    if not np.isfinite(last_close) or not np.isfinite(hi52) or hi52 <= 0:
        return 0.0, 0.0

    ratio = last_close / hi52
    score = float(np.clip((ratio - _HI52_MID_RATIO) / _HI52_SPAN, -1.0, 1.0))
    ratio_pct = round(ratio * 100, 2)
    logger.debug(f"[hi52] {ticker}: close={last_close:.2f}  hi52={hi52:.2f}  "
                 f"ratio={ratio_pct:.1f}%  score={score:+.3f}")
    return round(score, 3), ratio_pct


def compute_momentum_12_1_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
    """Return (score, ret_12_1_pct) — vol-normalised trailing-year momentum, skip-month.

    ret_12_1 = close[t−21] / close[t−252] − 1 (the canonical intermediate-horizon
    leg; the skipped month is the short-term-reversal regime). Normalised by the
    ticker's own 21-bar return std × √11 (eleven month-lengths), then
    tanh-mapped — the same self-normalising idiom as ``price_momentum``, so the
    two scores are comparable in scale. Requires a full 252 bars.
    """
    if df is None:
        df = _get_ohlcv(ticker, _M121_LOOKBACK)
    if df is None or df.empty or len(df) < _M121_LOOKBACK or "Close" not in df.columns:
        logger.debug(f"[mom_12_1] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, 0.0

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < _M121_LOOKBACK:
        return 0.0, 0.0
    anchor_old = float(close.iloc[-_M121_LOOKBACK])
    anchor_new = float(close.iloc[-_M121_SKIP])
    if anchor_old <= 0 or not np.isfinite(anchor_old) or not np.isfinite(anchor_new):
        return 0.0, 0.0
    ret_12_1 = anchor_new / anchor_old - 1.0

    monthly = close.pct_change(_M121_SKIP).dropna().tail(_DIST_WINDOW)
    if len(monthly) < 30:
        return 0.0, 0.0
    std_1m = float(monthly.std())
    if std_1m < 1e-8:
        return 0.0, 0.0

    z = ret_12_1 / (std_1m * np.sqrt((_M121_LOOKBACK - _M121_SKIP) / _M121_SKIP))
    if not np.isfinite(z):
        return 0.0, 0.0
    score = float(np.tanh(z / _M121_TANH))
    ret_pct = round(ret_12_1 * 100, 2)
    logger.debug(f"[mom_12_1] {ticker}: 12-1={ret_pct:+.1f}%  z={z:+.2f}  score={score:+.3f}")
    return round(score, 3), ret_pct


def compute_st_reversal_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
    """Return (score, ret_5d_pct) — prior-week return, sign-flipped, liquidity-gated.

    z = ret_5d / std(5-bar returns, trailing year); score = −tanh(z / 1.5), so a
    sharp up-week emits a BEARISH score (the predicted snapback) per the
    score-sign convention. No view when: the 20-day average dollar volume is
    below ``st_reversal_min_dollar_volume`` (bid-ask bounce, not reversal — the
    floor is deliberately far above the trade gate's $5M), volume data is
    missing (fail-closed), or |z| is inside the deadband (a quiet week carries
    no reversal information — and a spurious "view" would pollute
    methods_agreeing / breadth on every trade).
    """
    if df is None:
        df = _get_ohlcv(ticker, _REV_MIN_ROWS)
    if df is None or df.empty or len(df) < _REV_MIN_ROWS or "Close" not in df.columns:
        logger.debug(f"[st_reversal] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, 0.0

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < _REV_WINDOW + 1:
        return 0.0, 0.0
    last = float(close.iloc[-1])
    prev = float(close.iloc[-(_REV_WINDOW + 1)])
    # Zero/garbage prices (bad cache rows on thin names) would make the return —
    # and everything downstream, incl. the z guard (NaN < x is False) — non-finite.
    if prev <= 0 or not np.isfinite(prev) or not np.isfinite(last):
        return 0.0, 0.0
    ret_5d = last / prev - 1.0
    ret_pct = round(ret_5d * 100, 2)

    # Liquidity floor — fail-closed: no/short volume data ⇒ no view.
    if "Volume" not in df.columns:
        return 0.0, ret_pct
    dollar = (pd.to_numeric(df["Volume"], errors="coerce")
              * pd.to_numeric(df["Close"], errors="coerce")).dropna().tail(_DVOL_WINDOW)
    if len(dollar) < _DVOL_WINDOW or float(dollar.mean()) < float(settings.st_reversal_min_dollar_volume):
        logger.debug(f"[st_reversal] {ticker}: below liquidity floor — no view")
        return 0.0, ret_pct

    weekly = close.pct_change(_REV_WINDOW).dropna().tail(_DIST_WINDOW)
    if len(weekly) < 30:
        return 0.0, ret_pct
    std_5d = float(weekly.std())
    if std_5d < 1e-8:
        return 0.0, ret_pct

    z = ret_5d / std_5d
    if not np.isfinite(z) or abs(z) < _REV_DEADBAND_Z:
        return 0.0, ret_pct
    score = float(-np.tanh(z / _REV_TANH))
    logger.debug(f"[st_reversal] {ticker}: 5d={ret_pct:+.1f}%  z={z:+.2f}  score={score:+.3f}")
    return round(score, 3), ret_pct
