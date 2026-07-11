"""IV term-structure slope — front vs back ATM implied vol (2026-07-08).

The single-name IV term structure normally sits in CONTANGO (back-month IV
above front — calm, vol risk premium). BACKWARDATION (front ABOVE back) means the
options market is pricing a near-term event or acute stress: hedging demand is
concentrated at the short end. As a stock-direction hypothesis (what the panel
measures): backwardation → bearish tilt (near-term risk being paid up for),
steep contango → mild bullish (calm). This is a RISK/TIMING read more than an
alpha claim — exactly why it ships panel-first at weight 0 until its IC speaks.

Data comes FREE from the existing GEX chain fetch: ``gamma_exposure`` already
pulls every near expiry's chain per ticker (daily-cached), and now records the
ATM IV of the nearest + farthest processed expiry on each ``GEXSignal``
(``atm_iv_front``/``atm_iv_back`` + DTEs). No new network calls; coverage is
whatever GEX covered (sparse ⇒ 0.0 = no view, which the panel excludes).
Old GEX caches (written before the fields existed) deserialize with the
defaults → no view until the next fresh GEX fetch. Because the GEX window is
≤30 days, this is the SHORT end of the curve — the most event-sensitive part.
"""

from typing import Optional, Tuple

import numpy as np
from loguru import logger

_MIN_DTE_GAP  = 7      # need ≥ a week between front and back for a real slope
_IV_LO, _IV_HI = 0.01, 5.0   # sanity bounds — yfinance chain IVs are junk outside this
_SLOPE_SCALE  = 0.10   # 10 vol pts of backwardation → tanh(1) ≈ 0.76 bearish
_DEADBAND     = 0.02   # |front − back| under 2 vol pts is a flat curve → no view


def _gex_entry(ticker: str, gex_context):
    if gex_context is None or not getattr(gex_context, "signals", None):
        return None
    for sig in gex_context.signals:
        if sig.ticker.upper() == ticker.upper():
            return sig
    return None


def compute_iv_term_score(ticker: str, gex_context) -> Tuple[float, float, str]:
    """Return (score, slope_vol_pts, label) — slope = (front − back) ATM IV in
    vol POINTS (+3.0 = front 3 pts above back = backwardation). label ∈
    BACKWARDATION | CONTANGO | FLAT | NO_DATA. Score is −tanh(slope/scale):
    backwardation bearish, contango mildly bullish. 0.0 = no view."""
    sig = _gex_entry(ticker, gex_context)
    if sig is None:
        return 0.0, 0.0, "NO_DATA"
    front = getattr(sig, "atm_iv_front", None)
    back = getattr(sig, "atm_iv_back", None)
    front_dte = getattr(sig, "front_dte", None) or 0
    back_dte = getattr(sig, "back_dte", None) or 0
    if front is None or back is None:
        return 0.0, 0.0, "NO_DATA"
    front, back = float(front), float(back)
    if not (np.isfinite(front) and np.isfinite(back)):
        return 0.0, 0.0, "NO_DATA"
    if not (_IV_LO < front < _IV_HI and _IV_LO < back < _IV_HI):
        return 0.0, 0.0, "NO_DATA"
    if (back_dte - front_dte) < _MIN_DTE_GAP:
        return 0.0, 0.0, "NO_DATA"

    slope = front - back                       # + = backwardation (front rich)
    slope_pts = round(slope * 100, 2)
    if abs(slope) < _DEADBAND:
        return 0.0, slope_pts, "FLAT"
    label = "BACKWARDATION" if slope > 0 else "CONTANGO"
    score = float(-np.tanh(slope / _SLOPE_SCALE))
    if not np.isfinite(score):
        return 0.0, slope_pts, "NO_DATA"
    logger.debug(f"[iv_term] {ticker}: front={front:.2f}({front_dte}d)  back={back:.2f}({back_dte}d)  "
                 f"slope={slope_pts:+.1f}pts  {label}  score={score:+.3f}")
    return round(score, 3), slope_pts, label
