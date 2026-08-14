"""news_shock — abnormal news ATTENTION in the direction of the news read.

PANEL-FIRST at weight 0 (2026-08-14, the news-family continuity pass): scored
and persisted every run so its pivot-basis IC accrues, but not in
`_BASE_WEIGHTS`, the combine, coherence, `sources_agreeing` or the family
votes until the panel proves it — the same road squeeze/iv_term/avwap walked.

The thesis (the quantitative half of "news moves prices"): the news LEVEL
(`news`) says WHAT the stories say; this method says HOW LOUD they are
relative to the ticker's own normal. A genuine catalyst produces both a
directional read and an attention spike; a routine story does not spike
attention no matter how positive it reads. Attention alone is direction-less,
so the score is the news score's SIGN times the shock magnitude:

    mass_today = Σ per-article recency weights           (sentiment.attention_mass,
                                                          persisted as signals.news_recency_mass)
    baseline   = per-ticker MEDIAN of trailing daily mass (>= news_shock_min_days
                                                          days with coverage, from
                                                          the signals panel)
    score      = sign(news) · clip(log2(mass_today / baseline) / 3, 0, 1)

log2/3 ⇒ 2× the normal attention → 0.33, 8× → 1.0. Below-baseline attention
is NOT a fade signal — it scores 0 (quiet is the normal state, and "quieter
than usual" carries no direction), so the method fires only on above-baseline
attention that AGREES with a directional news read.

ABSTAINS (0.0) when: the news score is 0 (no direction to amplify), the
ticker has no baseline yet (the mass column is forward-collected — the method
self-activates as history accrues), or the baseline query fails (fail-soft;
a DB hiccup must never fake a signal). Non-replayable (needs the stored
point-in-time attention series), like `news` itself.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

from loguru import logger

from config.settings import settings

# One baseline load per tick is plenty; the panel only grows once per run.
_CACHE: dict = {"ts": 0.0, "baselines": None}
_TTL_SECONDS = 20 * 60

# Full score at 8× the ticker's normal attention (log2(8) = 3).
_LOG2_FULL_SHOCK = 3.0


def load_attention_baselines(force: bool = False) -> Dict[str, float]:
    """``{ticker: median daily news_recency_mass}`` over the trailing window.

    One grouped query over the signals panel: per (ticker, day) the MAX mass
    across that day's runs (a ticker's attention for the day, not per tick),
    days strictly BEFORE today (no self-reference), median over tickers with
    at least ``news_shock_min_days`` covered days. Fail-soft {} — every
    failure mode reads as "no baseline", which makes the method abstain.
    """
    now = time.time()
    if not force and _CACHE["baselines"] is not None and (now - _CACHE["ts"]) < _TTL_SECONDS:
        return _CACHE["baselines"]
    baselines: Dict[str, float] = {}
    try:
        from src.db import repo
        days = max(5, int(getattr(settings, "news_shock_baseline_days", 20)))
        min_days = max(2, int(getattr(settings, "news_shock_min_days", 5)))
        df = repo.fetch_df(f"""
            SELECT ticker, median(day_mass) AS base, count(*) AS n_days
            FROM (
                SELECT ticker, substr(signal_date, 1, 10) AS d,
                       max(news_recency_mass) AS day_mass
                FROM signals
                WHERE signal_date >= (CURRENT_DATE - INTERVAL {days + 5} DAY)::VARCHAR
                  AND signal_date < CURRENT_DATE::VARCHAR
                  AND news_recency_mass IS NOT NULL AND news_recency_mass > 0
                GROUP BY 1, 2
            )
            GROUP BY ticker
            HAVING count(*) >= {min_days}
        """)
        if df is not None and not df.empty:
            baselines = {str(t): float(b) for t, b in zip(df["ticker"], df["base"])
                         if b == b and b > 0}
        logger.info(f"[news_shock] attention baselines for {len(baselines)} ticker(s) "
                    f"({days}d window, >={min_days} covered days)")
    except Exception as e:
        logger.warning(f"[news_shock] baseline load failed (method abstains): {e}")
        baselines = {}
    _CACHE.update(ts=now, baselines=baselines)
    return baselines


def compute_news_shock(news_score: float, mass_today: float,
                       baseline: Optional[float]) -> float:
    """The signed shock score ∈ [-1, +1]; 0.0 on every abstention path."""
    if not news_score or mass_today <= 0 or not baseline or baseline <= 0:
        return 0.0
    ratio = mass_today / baseline
    if ratio <= 1.0:
        return 0.0                       # at/below normal attention — no shock
    magnitude = min(1.0, math.log2(ratio) / _LOG2_FULL_SHOCK)
    return round(math.copysign(magnitude, news_score), 4)


def reset_cache() -> None:
    """Test / asof hook."""
    _CACHE.update(ts=0.0, baselines=None)
