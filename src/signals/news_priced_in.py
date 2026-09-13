"""news_unpriced / news_unpriced_all — how much of the news move is LEFT
(2026-09-07, user directive, PANEL-FIRST at weight 0).

The `news` method scores what the story SAYS. These two score what the tape has
already DONE about it, anchored on the story's own start rather than on a fixed
window:

    cluster    = a group of articles with no large time gap inside it
                 (`sentiment.recent_cluster`, applied repeatedly — the freshest
                 group, then the next, and so on)
    anchor     = the last completed daily close at or before the cluster's FIRST
                 article — where the price was when the story began
    z          = sign(news) x (P_now / P_anchor - 1) / (sigma_20 x sqrt(sessions))
                 the realized move IN THE NEWS DIRECTION, in units of the
                 volatility that ruled when the news landed
    unpriced   = clip(1 - z / 2, -1, +1)      z = max(cluster z, trailing-3d z)
    score      = news x unpriced

so with good news and a flat tape the score is the news read at full strength
(the move is still ahead); at a +2 sigma aligned move it is 0 (priced in); past
that it goes NEGATIVE (the tape overshot the story — fade it). Bearish news is
symmetric: an un-fallen name keeps the full bearish read, one that already
collapsed 2 sigma abstains, and one that fell far past the story turns positive.

`news_unpriced` anchors on the MOST RECENT cluster only — the live question,
"has the market reacted to today's story yet". `news_unpriced_all` averages
every cluster's `unpriced`, weighted by how many articles each story drew, so a
name whose older stories are ALSO unpriced scores higher than one where only the
newest is. Weighting that average by RECENCY MASS was tried first and rejected:
the freshest cluster carries ~97% of the mass by construction (measured 1.89 vs
0.05 on a 2-cluster digest), which made the second feature a copy of the first —
and the point of it is to see the stories the first one cannot. The news SIGN is
the digest's single verdict in both (there is one verdict per digest, not one per
cluster); what varies per cluster is the anchor, hence how much of the move has
already happened.

WHAT THE HOUSE ALREADY MEASURED, AND WHY THESE STILL SHIP AT WEIGHT 0
---------------------------------------------------------------------
`news_bear_fresh` is the same idea on the BEAR side with a FIXED 3-session
window, and it measured +0.095 daily IC (t +2.96) within bear events. The
matching BULL hypotheses — "buy good news the price has not moved on" — were
measured on that same 2026-08-15 pass and did NOT hold: they decomposed into
period momentum or nothing (memory/news-interaction-methods-2026-08.md). What is
genuinely new here is the ANCHOR: the story's own start instead of a fixed
3-session look-back, so a 4-day-old catalyst is measured over 4 days and a
30-minute-old one over one session. That is a different quantity, not a re-run —
but the prior is a warning, which is why both land panel-first at weight 0 and
as stacker FEATURES (where the model decides the sign per side) rather than as
weighted methods with a hand-set direction.

Contract: score sign = predicted STOCK direction (+ up), |score| = conviction,
0.0 = abstain. Fail-soft everywhere — any error abstains.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from config.settings import settings

_MIN_BARS = 25          # 20-bar vol window + slack
_SLOPE = 2.0            # unpriced hits 0 at a 2-sigma aligned move (news_bear_fresh's slope)
_CAP = 1.0              # |unpriced| ceiling — an overshoot fades, it does not invert wildly
_MIN_SIGMA = 1e-4       # a degenerate vol series abstains rather than dividing by ~0
_FIXED_SESSIONS = 3     # the trailing window `news_bear_fresh` validated


def cluster_bounds(articles: Sequence, ratio: float = 3.0,
                   floor_hours: float = 24.0,
                   as_of: Optional[datetime] = None,
                   mode: Optional[str] = None,
                   ticker: Optional[str] = None) -> List[Tuple[datetime, float, int]]:
    """Partition a digest into story clusters, freshest first.

    ``[(first_published_at, recency_mass, n_articles), ...]`` — the START of a
    cluster is its OLDEST article, which is when that story began; the mass is
    the same per-article recency weight the sentiment scaler uses, so a cluster
    of stale articles counts for little without being discarded.

    The partition itself lives in `src/analysis/news_clustering.py` so that
    every consumer — this module, the cluster arm, the truncation A/B and the
    offline harnesses — cannot drift apart on what "a cluster" is.
    ``mode=None`` reads `news_cluster_mode` (`"time"` = the legacy relative-gap
    rule, `"content"` = IDF-weighted story grouping). A CALLER that pins a mode
    means it: `news_quiet` pins `"time"` because its measured quantity is the
    age of the last coverage BURST, which a content cluster spanning a week
    would silently redefine.
    """
    from src.analysis.news_clustering import cluster_articles
    from src.analysis.sentiment import _recency_weight
    out: List[Tuple[datetime, float, int]] = []
    for group in cluster_articles(articles, mode=mode, as_of=as_of, ticker=ticker,
                                  ratio=ratio, floor_hours=floor_hours):
        if not group:
            continue
        start = min(a.published_at for a in group)
        mass = float(sum(_recency_weight(a, as_of) for a in group))
        out.append((start, mass, len(group)))
    return out


def _naive_index(df):
    """The frame's index as tz-naive timestamps, computed ONCE per ticker.

    Doing it inside `_anchor_close` re-converted a ~5,000-row index for every
    cluster: measured 6.85 ms per ticker, ~2.7 s of tick on a 400-name universe,
    for a value that cannot change between two clusters of the same frame.
    """
    import pandas as pd
    idx = pd.to_datetime(df.index)
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def _anchor_close(df, when: datetime, idx_naive=None) -> Tuple[Optional[float], Optional[float], int]:
    """``(close_at_or_before(when), sigma_20_at_that_bar, sessions_since)``.

    Causal by construction: the anchor bar is the last COMPLETED bar at or
    before the cluster's first article, and the volatility is measured over the
    20 bars ending there — the vol regime that ruled when the news landed, never
    one that includes the reaction being measured.
    """
    import pandas as pd
    if df is None or len(df) < _MIN_BARS or "Close" not in df.columns:
        return None, None, 0
    try:
        day = pd.Timestamp(when).tz_convert(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)
    except Exception:                                       # noqa: BLE001
        return None, None, 0
    if idx_naive is None:
        idx_naive = _naive_index(df)
    pos = int((idx_naive <= day).sum()) - 1
    if pos < 20:
        return None, None, 0
    close = df["Close"].astype(float)
    anchor = float(close.iloc[pos])
    rets = close.iloc[max(0, pos - 20):pos + 1].pct_change().dropna()
    sigma = float(rets.std())
    sessions = int(len(close) - 1 - pos)
    if anchor <= 0 or not (sigma > _MIN_SIGMA):
        return None, None, 0
    return anchor, sigma, max(1, sessions)


def _z_move(news_sign: float, price_now: float, anchor: float, sigma: float,
            sessions: int) -> float:
    """The realized move IN THE NEWS DIRECTION since ``anchor``, in vol units.
    Larger z = more of the implied move already made."""
    moved = (price_now / anchor) - 1.0
    return news_sign * moved / (sigma * math.sqrt(max(1, sessions)))


def _from_z(z: float) -> float:
    """``clip(1 − z/slope, ±cap)`` — the share of the implied move still ahead."""
    return max(-_CAP, min(_CAP, 1.0 - z / _SLOPE))


def _unpriced(news_sign: float, price_now: float, anchor: float, sigma: float,
              sessions: int) -> float:
    return _from_z(_z_move(news_sign, price_now, anchor, sigma, sessions))


def _trailing_z(df, price_now: float, news_sign: float,
                sessions: int = _FIXED_SESSIONS) -> Optional[float]:
    """The same measurement over a FIXED trailing window, ignoring article times.

    This is the half that catches a reaction which happened BEFORE the article
    was written — "Why X Is Up Today" pieces, post-earnings recaps — where an
    anchor at the article is already past the move and reports ~0 divergence.
    Same 20-bar vol convention as the cluster anchor.
    """
    if df is None or "Close" not in df.columns or len(df) < _MIN_BARS:
        return None
    close = df["Close"].astype(float)
    pos = len(close) - 1 - int(sessions)
    if pos < 20:
        return None
    anchor = float(close.iloc[pos])
    rets = close.iloc[max(0, pos - 20):pos + 1].pct_change().dropna()
    sigma = float(rets.std())
    if anchor <= 0 or not (sigma > _MIN_SIGMA):
        return None
    return _z_move(news_sign, price_now, anchor, sigma, int(sessions))


def compute_news_priced_in(ticker: str, news_score: Optional[float], articles: Sequence,
                           price_now: Optional[float] = None, df=None,
                           as_of: Optional[datetime] = None,
                           ) -> Tuple[float, float, dict]:
    """→ ``(news_unpriced, news_unpriced_all, diag)``; both 0.0 = abstain.

    ``df`` is the shared completed-bars daily frame (same contract as the other
    OHLCV scorers); ``None`` falls back to the cache. ``price_now`` is the live
    mark — the last completed close when it is missing, which only makes the
    measurement staler, never look-ahead.
    """
    try:
        news = float(news_score) if news_score is not None else 0.0
        if news == 0.0 or not articles:
            return 0.0, 0.0, {}
        if df is None:
            from src.data.cache import load_ohlcv
            df = load_ohlcv(ticker)
        if df is None or len(df) < _MIN_BARS or "Close" not in df.columns:
            return 0.0, 0.0, {}
        px = float(price_now) if price_now else float(df["Close"].astype(float).iloc[-1])
        if px <= 0:
            return 0.0, 0.0, {}
        clusters = cluster_bounds(articles, as_of=as_of)
        if not clusters:
            return 0.0, 0.0, {}
        sign = 1.0 if news > 0 else -1.0
        idx_naive = _naive_index(df)                        # once per ticker, not per cluster
        # The fixed-window reading is a property of the TICKER, not of a cluster
        # — computed once and compared against each cluster's own anchor.
        z_fixed = (_trailing_z(df, px, sign)
                   if getattr(settings, "enable_news_unpriced_two_sided", True) else None)
        per: List[Tuple[float, float, float]] = []          # (unpriced, n_articles, mass)
        for start, mass, n in clusters:
            anchor, sigma, sessions = _anchor_close(df, start, idx_naive)
            if anchor is None:
                continue
            z = _z_move(sign, px, anchor, sigma, sessions)
            # MORE priced-in wins: if either reading says the move already
            # happened, it happened. Taking the max cannot make a genuinely
            # un-priced name look priced — both readings would have to be low.
            if z_fixed is not None:
                z = max(z, z_fixed)
            per.append((_from_z(z), float(n), mass))
        if not per:
            return 0.0, 0.0, {}
        fresh = round(max(-1.0, min(1.0, news * per[0][0])), 4)
        wsum = sum(n for _u, n, _m in per)
        agg_u = (sum(u * n for u, n, _m in per) / wsum) if wsum > 0 else per[0][0]
        agg = round(max(-1.0, min(1.0, news * agg_u)), 4)
        diag = {"clusters": len(per), "unpriced_fresh": round(per[0][0], 4),
                "unpriced_all": round(agg_u, 4),
                "fresh_mass": round(per[0][2], 4),
                "cluster_start": clusters[0][0].isoformat()}
        return fresh, agg, diag
    except Exception as e:                                  # noqa: BLE001 - fail-soft
        logger.debug(f"[news_unpriced] {ticker} abstained: {e}")
        return 0.0, 0.0, {}
