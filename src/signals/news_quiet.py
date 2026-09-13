"""`news_quiet` — the news read once the story has gone QUIET.

MEASURED (2026-09-09, 656 directional ticker-days over 50 days, oriented pivot
return per decision, day-clustered). Splitting on the age of the digest's
FRESHEST news cluster — i.e. how long it has been since anything new was written
about the name:

    freshest cluster >= 48h old   +1.90 pp/decision   t +2.65   halves same
    paired old MINUS fresh        +2.17 pp            t +2.52   halves same
    freshest cluster >= 72h old   +2.29 pp            t +2.45   halves same
    SHORT, >= 48h                 +2.90 pp            t +3.10   halves same
    SHORT, <  48h                 -1.07 pp            t -1.29
    LONG,  >= 48h                 +0.90 pp            t +0.86   halves OPPOSITE
    LONG,  <  48h                 +0.50 pp            t +0.62

It is DIRECTION, not size: hit rate goes 47.9% -> 58.5% across the split while
the median |pivot| barely moves (4.68% -> 5.46%).

READING. While a story is fresh and loud, the tape is crowded and the sentiment
read is worth nothing; once the news flow stops, the residual read predicts the
next pivot. That is consistent with the house's own priors — the news edge lives
on the negative side (`memory/buy-sell-asymmetry-2026-07`) and `news_bear_fresh`
measured +0.095 IC on a FIXED 3-session window, which is the same shape reached
from the other direction.

HOW IT WAS FOUND, and what that costs it. It came out of a REJECTED experiment:
the user's hypothesis was that traded VOLUME since the catalyst measures how
much of a move is already priced (surge, then decay back to baseline = done).
That failed outright — the best discount cell scored t +0.20, and the volume
fraction turned out to track catalyst AGE (rho +0.335) and digest size (-0.279)
rather than the realised move (-0.048). Chasing the confound is what produced
the age split, so **this feature is exploratory, not pre-registered**: age was
reached after the volume hypothesis failed, in the same 50 days, and three
thresholds were tried (24h fails; 48h and 72h pass, monotone in between).
History: `memory/volume-priced-in-2026-09.md`.

SHIPPED ON THE USER'S CALL (2026-09-09), both sides, with the LONG side
explicitly below the house bar. That asymmetry is NOT hand-coded here: the
method is symmetric and `enable_side_adaptive_weights` / `enable_side_winrate_filter`
weight and filter each camp on its OWN accruing record, so if LONG really is the
weaker half the existing machinery discovers it from the ledger instead of from
this docstring.

CONTRACT. `[-1, +1]`, the house sign convention (+ = the stock goes up). The
score is the RAW sentiment verdict on quiet names and 0.0 — an ABSTENTION, so
the name leaves this method's cross-section entirely — on loud ones.

Two deliberate choices:

* **The RAW verdict, not the scaled `news` score.** The scaled one multiplies by
  evidence mass x source diversity, and the quiet cohort has systematically
  fewer articles (median 6 vs 12) — so the scaler shrinks exactly the names this
  method exists to express. Raw is also the quantity the split was measured on.
* **Abstain, never invert, on a loud story.** The measurement says a fresh read
  is worth ~0 (LONG +0.50 pp, SHORT -1.07 pp), not that it is reliably wrong.
  Scoring the negative of a fresh read would be a much stronger claim than the
  evidence supports.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

from loguru import logger

from config.settings import settings
from src.signals.news_priced_in import cluster_bounds

# The freshest cluster must be at least this old for the method to take a view.
# 24h does NOT clear the bar; 48h and 72h both do, so the qualifying range is
# 48-72h. LIVE at 72h since 2026-09-10 (user request): a larger mean on fewer
# names (+2.292 pp / t +2.45 / n 177 against +1.900 / +2.65 / n 210), with the
# SHORT side — where the edge lives — unchanged at +2.94 vs +2.90.
DEFAULT_MIN_AGE_HOURS = 72.0


def _now(as_of: Optional[datetime] = None) -> datetime:
    """The instant ages are measured against — hour-quantised, matching
    `sentiment._clock`, so two runs inside one hour cannot land on opposite
    sides of the threshold for the same article set."""
    base = as_of or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.replace(minute=0, second=0, microsecond=0)


def freshest_cluster_age_hours(articles: Sequence,
                               as_of: Optional[datetime] = None) -> Optional[float]:
    """Hours since the FIRST article of the most recent news cluster.

    Not "hours since the newest article": the unit is the STORY. A cluster is
    the same time-gap grouping `sentiment.recent_cluster` uses (via
    `news_priced_in.cluster_bounds`), so this module and the priced-in family
    cannot disagree about where one story ends and the next begins. None when
    there is no digest to date.
    """
    # PINNED to the time partition, deliberately. This method's +1.90
    # pp/decision was measured with the relative-gap rule, where the freshest
    # cluster's age means "hours since the last BURST of coverage began". Under
    # content clustering a story that broke a week ago and got a fresh article
    # this morning is ONE cluster starting a week ago — so a name with news
    # TODAY would read as quiet, inverting the method. It measures news FLOW,
    # which is a time concept; `news_unpriced` measures a STORY's anchor, which
    # is a content one.
    bounds = cluster_bounds(articles, as_of=as_of, mode="time")
    if not bounds:
        return None
    start = bounds[0][0]
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return max(0.0, (_now(as_of) - start).total_seconds() / 3600.0)


def compute_news_quiet(ticker: str, news_raw: Optional[float], articles: Sequence,
                       as_of: Optional[datetime] = None
                       ) -> Tuple[float, Optional[float]]:
    """``(score, freshest_cluster_age_hours)``.

    ``news_raw`` is the PRE-SCALER sentiment verdict (`news_meta["raw_score"]`).
    Fail-soft: any missing input abstains at 0.0, which for a rank-consumed
    method means the ticker simply does not enter this method's cross-section.
    """
    if not bool(getattr(settings, "enable_news_quiet", True)):
        return 0.0, None
    try:
        if news_raw is None or not articles:
            return 0.0, None
        raw = float(news_raw)
        if raw != raw or raw == 0.0:            # no view to carry forward
            return 0.0, None
        age = freshest_cluster_age_hours(articles, as_of=as_of)
        if age is None:
            return 0.0, None
        floor = float(getattr(settings, "news_quiet_min_age_hours",
                              DEFAULT_MIN_AGE_HOURS) or DEFAULT_MIN_AGE_HOURS)
        if age < floor:
            return 0.0, age                     # story still loud: abstain
        return max(-1.0, min(1.0, raw)), age
    except Exception as e:                                      # noqa: BLE001
        logger.debug(f"[news_quiet] {ticker}: {e}")
        return 0.0, None
