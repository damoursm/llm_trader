"""news_bear_fresh — bear-news freshness guard (2026-08-15, panel-first, weight 0).

For a BEARISH news read only:

    am    = sign(news) x z3 = -z3       (how much of the bad news the tape has
                                         already priced: 3-session close move in
                                         prior-20d-vol units, sqrt-horizon scaled)
    score = news x clip(1 - am/2, 0, 1.5)

so a name that already fell 2σ with its bad news ABSTAINS (never short into a
hole), a flat tape passes the news through, and bad news the tape has ignored —
or that the price rose against — is boosted up to 1.5x. Bullish and zero news
always abstain: the method is a bear-side specialist by measurement, and a bull
passthrough would only duplicate the `news` method in the panel.

Measured basis (2026-08-15; 39 daily cross-sections 06-17→08-14, Gate-4
population, pivot target — memory/news-interaction-methods-2026-08.md):
within bear-news events the modulated score's daily IC is +0.095 (t +2.96)
versus news alone +0.029 (t +0.8), the guard term alone +0.050 (t +1.2) and raw
3d reversal −0.047 — a genuine PRODUCT effect, holding on the 5d monitor
(+0.102, t +3.59) and flat across the constants plateau (z3/z5 windows x slopes
1.5–3 x caps 1.0/1.5 all t ≥ +2.0; the z1 window is dead and deliberately
excluded — 1-day continuation measured as period-wide market character, not a
news effect). The complementary BULL hypotheses (confirmation boost, priced-in
discount, volume conditioning) all measured as repackaged period momentum or
nothing, and are deliberately NOT shipped. The economic prior is the 20y MR
battery: reversal is pervasively real on tradeable names — shorting an already
-collapsed name fights the bounce.

Contract: score sign = predicted stock direction (always ≤ 0 or abstain 0.0);
|score| = conviction. Fail-soft everywhere — any error abstains.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from loguru import logger

_MIN_BARS = 25     # 20d vol window (shifted) + 3d move + slack
_SLOPE = 2.0       # guard reaches 0 at a 2σ aligned 3-session move
_CAP = 1.5         # an un-fallen / risen-against short is boosted at most 1.5x


def compute_news_bear_fresh(ticker: str, news_score: Optional[float],
                            df=None) -> Tuple[float, float]:
    """→ ``(score, z3)``; score ∈ [−1, 0], 0.0 = abstain.

    ``df`` is the shared completed-bars daily OHLCV frame (same contract as the
    other panel-first OHLCV scorers); ``None`` falls back to the cache. Abstains
    on non-bearish news, missing/short history, or degenerate vol.
    """
    try:
        if news_score is None or float(news_score) >= 0.0:
            return 0.0, 0.0
        if df is None:
            from src.data.cache import load_ohlcv
            df = load_ohlcv(ticker)
        if df is None or len(df) < _MIN_BARS or "Close" not in df.columns:
            return 0.0, 0.0
        close = df["Close"].astype(float)
        vol20 = float(close.pct_change().rolling(20).std().shift(1).iloc[-1])
        r3 = float(close.iloc[-1] / close.iloc[-4] - 1.0)
        if not (vol20 > 0.0) or math.isnan(vol20) or math.isnan(r3):
            return 0.0, 0.0
        z3 = r3 / (vol20 * math.sqrt(3.0))
        am = -z3                                   # aligned prior move for a bear read
        guard = min(max(1.0 - am / _SLOPE, 0.0), _CAP)
        score = max(-1.0, min(0.0, float(news_score) * guard))
        return round(score, 4), round(z3, 3)
    except Exception as e:                          # fail-soft: abstain, never raise
        logger.debug(f"[news_bear_fresh] {ticker}: {e} — abstain")
        return 0.0, 0.0
