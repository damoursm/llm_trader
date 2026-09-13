"""news_bull_fresh — the bull-side mirror of `news_bear_fresh` (2026-09-10,
panel-first, weight 0, user request).

For a BULLISH news read only, symmetric to the bear specialist in every term:

    am    = sign(news) x z3 = +z3       (how much of the good news the tape has
                                         already priced: 3-session close move in
                                         prior-20d-vol units, sqrt-horizon scaled)
    score = news x clip(1 - am/2, 0, 1.5)

so a name that already rose 2σ on its good news ABSTAINS (never chase a name
that has already made the move), a flat tape passes the news through, and good
news the tape has ignored — or that the price fell against — is boosted up to
1.5x. Bearish and zero news always abstain: it is a bull-side specialist, and a
bear passthrough would only duplicate `news_bear_fresh` in the panel.

WHY THIS SHIPS AT WEIGHT 0, AND WHY IT IS NOT OBVIOUSLY GOING TO WORK. Two
independent priors point the wrong way:

1. **The bull side was already tested and failed.** On 2026-08-15 the
   complementary BULL hypotheses — confirmation boost, priced-in discount,
   volume conditioning — all measured as repackaged period momentum or nothing
   (`memory/news-interaction-methods-2026-08.md`), which is why
   `news_bear_fresh` shipped as a bear-ONLY specialist and says so in its own
   docstring. What has changed since is the INPUT, not the idea: that test ran
   on the v4 prompt with the substring relevance filter, where 73-79% of
   sentiment calls returned 0.0 and a short symbol received the whole article
   pool. Under v6 + name relevance the abstention is ~4-8% per call. Re-testing
   a bull hypothesis on a materially better news read is legitimate; expecting
   a different answer on the strength of that alone is not.

2. **The priced-in premise it encodes measured BACKWARDS this month.** The
   guard says "discount a read the tape has already made". Measured on 655
   ticker-days (2026-09-09/10), three independent estimates of how much of a
   move has already traded — volume-surge decay, the realised price move, and
   the model's own stated fraction — ALL say the read is BETTER after the move,
   not worse (`memory/volume-priced-in-2026-09.md`). The z >= 1.0 cohort
   returns +3.76 pp against -0.34 pp for z < 0.5. On that evidence the bull
   guard should be INVERTED rather than mirrored.

That tension is exactly why this is panel-first at weight 0 rather than
weighted, and why `news_bull_fresh_invert` exists as a setting: the panel can
adjudicate the mirror against the inverse on live rows instead of either being
argued for. Note the bear side is NOT symmetric evidence here — its +0.095 IC
(t +2.96) is a measured fact about bear events, and reversal after a collapse
has a 20-year MR prior behind it that has no bull-side twin (an already-risen
name has no equivalent bounce mechanic).

Contract: score sign = predicted stock direction (always >= 0 or abstain 0.0);
|score| = conviction. Fail-soft everywhere — any error abstains.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from loguru import logger

from config.settings import settings

_MIN_BARS = 25     # 20d vol window (shifted) + 3d move + slack
_SLOPE = 2.0       # guard reaches 0 at a 2σ aligned 3-session move
_CAP = 1.5         # an un-risen / fallen-against long is boosted at most 1.5x


def compute_news_bull_fresh(ticker: str, news_score: Optional[float],
                            df=None) -> Tuple[float, float]:
    """→ ``(score, z3)``; score ∈ [0, +1], 0.0 = abstain.

    ``df`` is the shared completed-bars daily OHLCV frame (same contract as the
    other panel-first OHLCV scorers); ``None`` falls back to the cache. Abstains
    on non-bullish news, missing/short history, or degenerate vol.

    ``news_bull_fresh_invert`` flips the guard to ``1 + am/2`` — boosting a read
    the tape has ALREADY confirmed instead of discounting it. Off by default;
    it exists because the 2026-09-09 priced-in work measured that direction to
    be the one the data supports, and the panel is the place to settle it.
    """
    try:
        if news_score is None or float(news_score) <= 0.0:
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
        am = z3                                    # aligned prior move for a bull read
        if bool(getattr(settings, "news_bull_fresh_invert", False)):
            guard = min(max(1.0 + am / _SLOPE, 0.0), _CAP)
        else:
            guard = min(max(1.0 - am / _SLOPE, 0.0), _CAP)
        score = min(1.0, max(0.0, float(news_score) * guard))
        return round(score, 4), round(z3, 3)
    except Exception as e:                          # fail-soft: abstain, never raise
        logger.debug(f"[news_bull_fresh] {ticker}: {e} — abstain")
        return 0.0, 0.0
