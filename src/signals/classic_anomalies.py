"""Classic cross-sectional anomalies — 52-week-high proximity, 12-1 momentum,
short-term reversal (2026-07-08, panel-first at weight 0), plus the 2026-08-10
mean-reversion additions ``rsi2_rev`` and ``dloc_rev`` (selected by the 20-year
full-history battery in ``memory/pivot-horizon-target-2026-08.md`` —
de-correlated cluster winners at daily-IC t ≥ 5 on Gate-4 tradeable names,
stable across 2007-16 / 17-26).

**PROMOTED INTO THE COMBINE 2026-08-11 (user-directed):** all five methods in
this module now carry ``_BASE_WEIGHTS`` entries (mom_12_1 0.15, hi52 0.15,
st_reversal 0.12, rsi2_rev 0.10, dloc_rev 0.10 — the h2-evidence ladder) and
therefore count in coherence / sources_agreeing / the Price-Trend family vote.
The per-side win-rate machinery governs them from here exactly as it governs
every weighted method (all start "unproven" below min_trades → full weight).

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
_REV_MIN_ROWS   = 60    # enough bars for a stable liquidity floor + history guard
# v2 (2026-08-10): FIXED-scale squash. v1 divided ret_5d by the ticker's own
# 5-bar-return std before the tanh — which threw away the cross-sectional
# magnitude information (a 10% down-week in a calm name and a wild one got the
# same score). Measured over 2007→2026 on Gate-4 tradeable names (5.59M rows):
# plain −ret_5d ranks at daily-IC +0.0150 (t +7.2) vs the z-version's +0.0114
# (t +6.2); a fixed-scale tanh is within-day monotone in ret_5d, so it inherits
# the raw ranking exactly while keeping the score in [-1, 1]. 5% ≈ one weekly
# std of a typical tradeable name → tanh(1) ≈ 0.76 at a one-sigma week.
_REV_TANH_RET   = 0.05  # ret_5d at which |score| reaches tanh(1)
_REV_DEADBAND   = 0.01  # |ret_5d| below 1% → no view (a quiet week is not a signal)
_DIST_WINDOW    = 252   # trailing bars for the normalisation distributions
_DVOL_WINDOW    = 20    # 20-day average dollar volume for the liquidity floor

_RSI2_PERIOD    = 2     # Connors RSI(2) — the canonical short-term MR oscillator
_RSI2_MIN_ROWS  = 30
_MR_DEADBAND    = 0.05  # |score| below this → 0.0 (don't pollute methods_agreeing)


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

    v2 (2026-08-10, epoch-registered): score = −tanh(ret_5d / 0.05) — a FIXED
    scale, so a sharp up-week emits a BEARISH score (the predicted snapback) per
    the score-sign convention, and a bigger move means a stronger view across
    names. v1 z-normalised by the ticker's own weekly std first, which discarded
    the cross-sectional magnitude; the 20-year Gate-4 measurement (see
    ``memory/pivot-horizon-target-2026-08.md`` MR battery) has the raw-return
    ranking at daily-IC +0.0150 (t +7.2) vs the z-version's +0.0114 (t +6.2).
    No view when: the 20-day average dollar volume is below
    ``st_reversal_min_dollar_volume`` (bid-ask bounce, not reversal — the floor
    is deliberately far above the trade gate's $5M), volume data is missing
    (fail-closed), or |ret_5d| is inside the 1% deadband (a quiet week carries
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

    if abs(ret_5d) < _REV_DEADBAND:
        return 0.0, ret_pct
    score = float(-np.tanh(ret_5d / _REV_TANH_RET))
    logger.debug(f"[st_reversal] {ticker}: 5d={ret_pct:+.1f}%  score={score:+.3f}")
    return round(score, 3), ret_pct


def _dollar_volume_ok(df: pd.DataFrame, floor: float) -> bool:
    """The st_reversal liquidity floor, shared by every short-horizon MR scorer
    here — below it a 1-5 day "reversal" is mostly bid-ask bounce, i.e. the
    SIGNAL is fake, not merely untradeable. Fail-closed on missing volume."""
    if "Volume" not in df.columns:
        return False
    dollar = (pd.to_numeric(df["Volume"], errors="coerce")
              * pd.to_numeric(df["Close"], errors="coerce")).dropna().tail(_DVOL_WINDOW)
    return len(dollar) >= _DVOL_WINDOW and float(dollar.mean()) >= float(floor)


def compute_rsi2_rev_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
    """Return (score, rsi2) — Connors RSI(2) mean-reversion (2026-08-10, panel-first).

    score = (50 − RSI(2)) / 50 ∈ [−1, +1]: deeply oversold → bullish snapback
    view, overbought → bearish, per the score-sign convention. Wilder smoothing
    (ewm alpha=1/2). Battery basis (20y, Gate-4 tradeable): daily-IC +0.0097
    (t +5.6) on rel-5d, +0.0103 (t +5.9) on the signed-pivot target, and the
    most STABLE short-horizon candidate across date halves (+0.0092 / +0.0102).
    Shares ``st_reversal_min_dollar_volume`` — same bid-ask-bounce argument.
    """
    if df is None:
        df = _get_ohlcv(ticker, _RSI2_MIN_ROWS)
    if df is None or df.empty or len(df) < _RSI2_MIN_ROWS or "Close" not in df.columns:
        logger.debug(f"[rsi2_rev] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, 0.0

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < _RSI2_MIN_ROWS or not np.isfinite(float(close.iloc[-1])):
        return 0.0, 0.0
    if not _dollar_volume_ok(df, settings.st_reversal_min_dollar_volume):
        logger.debug(f"[rsi2_rev] {ticker}: below liquidity floor — no view")
        return 0.0, 0.0

    delta = close.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1.0 / _RSI2_PERIOD, adjust=False).mean()
    dn = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / _RSI2_PERIOD, adjust=False).mean()
    dn_last = float(dn.iloc[-1])
    up_last = float(up.iloc[-1])
    if not (np.isfinite(up_last) and np.isfinite(dn_last)) or (up_last + dn_last) <= 0:
        return 0.0, 0.0
    rsi2 = 100.0 - 100.0 / (1.0 + up_last / dn_last) if dn_last > 0 else 100.0
    score = float(np.clip((50.0 - rsi2) / 50.0, -1.0, 1.0))
    if abs(score) < _MR_DEADBAND:
        return 0.0, round(rsi2, 2)
    logger.debug(f"[rsi2_rev] {ticker}: RSI2={rsi2:.1f}  score={score:+.3f}")
    return round(score, 3), round(rsi2, 2)


def compute_dloc_rev_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
    """Return (score, close_loc_pct) — daily candle-location reversal (2026-08-10).

    loc = (close − low) / (high − low) of the LAST COMPLETED bar; score =
    clip(2·(0.5 − loc), −1, +1): a close pinned to the day's low → +1 (bullish
    bounce view), pinned to the high → −1. The strongest pivot-target candidate
    in the 20y battery on tradeable names (signed-pivot daily-IC +0.0117,
    t +8.1 — the production target's own 1-day horizon), distinct from the
    weekly-reversal cluster (ρ vs ret_5-family < 0.6). rel-5d IC +0.0075
    (t +5.3). Shares the liquidity floor; a zero-range bar (h == l) is no view.
    """
    if df is None:
        df = _get_ohlcv(ticker, _RSI2_MIN_ROWS)
    if df is None or df.empty or len(df) < 2 or "Close" not in df.columns \
            or "High" not in df.columns or "Low" not in df.columns:
        logger.debug(f"[dloc_rev] {ticker}: insufficient data")
        return 0.0, 0.0

    c = pd.to_numeric(df["Close"], errors="coerce")
    h = pd.to_numeric(df["High"], errors="coerce")
    lo = pd.to_numeric(df["Low"], errors="coerce")
    cl, hl, ll = float(c.iloc[-1]), float(h.iloc[-1]), float(lo.iloc[-1])
    if not all(np.isfinite(v) for v in (cl, hl, ll)) or hl <= ll:
        return 0.0, 0.0
    if not _dollar_volume_ok(df, settings.st_reversal_min_dollar_volume):
        logger.debug(f"[dloc_rev] {ticker}: below liquidity floor — no view")
        return 0.0, 0.0

    loc = (cl - ll) / (hl - ll)
    score = float(np.clip(2.0 * (0.5 - loc), -1.0, 1.0))
    if abs(score) < _MR_DEADBAND:
        return 0.0, round(loc * 100, 2)
    logger.debug(f"[dloc_rev] {ticker}: loc={loc*100:.0f}%  score={score:+.3f}")
    return round(score, 3), round(loc * 100, 2)
