"""Scorer-version epochs — the date a method's implementation last CHANGED
in a way that makes its older stored scores incomparable.

Why this exists
───────────────
Method scores are persisted at the moment they're computed: on every trade
(``trades.method_scores``) and on every scored ticker (the ``signals`` panel).
Those stored numbers are then fed back into LIVE decisions — the win-rate
filters (``aggregator.winrate_filtered_methods`` / ``side_filtered_methods``),
the adaptive weight tilt, and the IC-weight layer.

When a scorer is FIXED, every one of those historical numbers was produced by a
function that no longer exists. Judging the new implementation on them is
simply wrong: on 2026-07-24 ``money_flow`` was found unable to express
direction at all (its contrarian MFI term cancelled its trend-following CMF
term, and its OBV term measured acceleration rather than direction). All 235
closed attributed trades carry scores from that broken version, and new trades
accrue a few per day — so a blended win rate would keep the FIXED scorer
suppressed for months on the BROKEN one's record.

Why not just retrofit the database
──────────────────────────────────
Recomputing the old rows was measured and rejected. Replaying the current
scorer over the cached OHLCV truncated to each signal date reproduces the
stored value exactly for only ~62% of rows (median error 0.001 but max 0.29),
because the cache is not what it was then: it is retroactively split-adjusted,
its per-ticker start dates vary (45 distinct starts among 200 tickers), and
both the OBV and CMF normalisations depend on the FULL series length the run
actually held. Overwriting 273k rows would therefore fabricate a history that
never happened while looking authoritative — the same failure class as the
split-basis and stale-decision-price bugs fixed earlier the same week. The
stored numbers are kept as the true record of what was computed; they are
simply not charged against a different implementation.

Effect
──────
A method with an epoch is evaluated ONLY on trades entered / signals dated on
or after it. Until enough post-epoch evidence accrues it falls below
``winrate_filter_min_trades`` and therefore keeps FULL weight — "unproven", not
"disproven", which is the correct stance for a freshly-fixed scorer.

Why the old rows are not DELETED either
───────────────────────────────────────
Deleting "data from machinery no longer in use" sounds clean but is far too
blunt a cut. Measured against the 276k-row panel: money_flow's superseded
scores are 96.5% of rows, everything predating the buy/sell split combine is
86%, and everything predating the family-agreement/tape confidence factors is
75%. Applied literally the rule would erase most of a five-week,
forward-collected research asset that the IC weighting, predictability sizing,
edge-decay and policy-eval layers all depend on — and it would take the still-
valid parts of each row (forward returns, prices, every UNCHANGED method) with
it. Masking one column keeps the row's other evidence intact.

What belongs here — categorical changes only
────────────────────────────────────────────
Register a change when the output MEANS something different, not when it has
merely been tuned:
  • YES — money_flow: went from unable to express direction (same sign on a
    rising and a falling tape) to direction-aware. The old numbers are not a
    weaker version of the new ones, they are a different quantity.
  • NO — the 2026-07-22 combine (normalised pool → buy−sell difference): the
    same quantity, refined, and measured as a wash at the time (IC5 −0.037 vs
    −0.041). Registering it would blank 86% of the panel to no benefit.
  • NO — the 2026-07-19 confidence factors (family agreement, tape): bounded
    ±12% / ±8% adjustments to the same conviction number.
When in doubt, prefer NOT registering and note the change in CLAUDE.md instead:
a mask is cheap to add later and expensive to have wrongly applied.

Adding an entry
───────────────
Whenever you change what a scorer OUTPUTS categorically, add the INSTANT here
(UTC datetime — deploys land mid-session). Removing an entry re-admits the old
history. ``analysis/signal_panel.build_panel`` applies the mask centrally, so
every panel consumer is protected without touching each one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

# method → the INSTANT the current implementation went live. A deploy lands
# mid-session, so this is a UTC datetime, not a date: on 2026-07-24 five trades
# were entered earlier the same day under the OLD money_flow and a date-granular
# cutoff would have admitted them as evidence for the NEW one.
METHOD_SCORER_EPOCH: dict[str, datetime] = {
    # 2026-07-24 20:01 UTC (scheduler restart): the MFI term now abstains inside
    # a 35-65 neutral band (it used to apply a contrarian reading linearly across
    # the whole range and cancel the equally-weighted, trend-following CMF term),
    # and the OBV term is now scale-normalised sign-preserving instead of a
    # z-score against its own history (which measured acceleration and read a
    # falling tape as bullish). Before this instant the scorer returned the same
    # sign on a rising and a falling tape — tests/test_method_directionality.py.
    "money_flow": datetime(2026, 7, 24, 20, 1, tzinfo=timezone.utc),
}


# ── Confidence-formula epochs (2026-07-27) ───────────────────────────────────
# `confidence` and its six persisted components are NOT method scores, but they
# have the same problem: the formula's inputs changed, so the column mixes
# incompatible values and any analysis of it compares apples to oranges.
#
# Same remedy, same standard: register CATEGORICAL changes only — where the
# number MEANS something different — never refinements. Masked to NaN at
# `signal_panel.build_panel`; the ROW survives, so method scores, prices and
# forward returns remain valid evidence. A retrofit was considered and REJECTED
# (2026-07-27): 78.3% of rows predate the component capture entirely, the
# OHLCV cache is retroactively split-adjusted so the movement/volume/tape
# factors cannot be reproduced as they were, and — decisively — today's weights
# are CALIBRATED FROM THIS PANEL, so rescoring the past with them would inject
# future information into the exact dataset that feeds the IC-weight layer,
# predictability sizing and policy eval. A rescored panel would look
# authoritative and be partly invented.
#
# Registered:
#   2026-07-22 buy/sell split combine. `combined_score` changed from a weighted
#   average over ONE normalised pool to the DIFFERENCE of two camp averages, and
#   `raw_confidence = min(1, |combined|/0.5)` derives straight from it — so the
#   confidence scale itself changed meaning, not merely its level. Measured in
#   the panel: mean confidence 0.15 (07-22) -> 0.50 (07-23), with mean
#   |combined_score| tracking it, confirming the shift is the combine and not
#   the confidence formula.
#
# Deliberately NOT registered (refinements, per the same rule):
#   2026-07-19 family-agreement + tape factors — bounded to +/-12% and +/-8%.
#   2026-07-27 market-relative weighting/filter — changes the weights inside an
#   unchanged formula; the panel's own drift is already visible without masking.
CONFIDENCE_EPOCH: datetime = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)

# Columns the confidence epoch governs: the value plus the six ingredients that
# are only interpretable alongside it.
CONFIDENCE_EPOCH_COLUMNS: tuple = (
    "confidence", "raw_confidence", "coherence_factor", "movement_factor",
    "volume_factor", "family_conf_factor", "tape_conf_factor",
)


def confidence_epoch() -> Optional[date]:
    """The confidence epoch as a DATE, or None when masking is disabled.

    Same mid-day convention as `epoch_for`: returns the day AFTER a mid-day
    change so a date-granular caller excludes the ambiguous partial day."""
    from config.settings import settings
    if not getattr(settings, "enable_confidence_epoch", False):
        return None
    cutoff = CONFIDENCE_EPOCH
    d = cutoff.date()
    return d if (cutoff.hour == 0 and cutoff.minute == 0) else date.fromordinal(d.toordinal() + 1)


def confidence_is_comparable(when) -> bool:
    """Was this row's ``confidence`` produced by the CURRENT formula?

    The ledger-side counterpart of `score_is_comparable`. Calibrations that read
    trade `confidence` (`calibrate_side_threshold`, `calibrate_confidence_sizing`)
    consume the LEDGER, not the panel, so `build_panel`'s masking never reaches
    them — measured 2026-07-27, **75% of the closed trades feeding both carried
    the pre-split confidence scale** (mean 0.873 versus 0.945 after), i.e. both
    were fitting a relationship across two different scales.

    Fail-OPEN: no epoch, an unparseable timestamp, or masking disabled all mean
    comparable, so a malformed date can never silently erase history.
    """
    cutoff = confidence_epoch()
    if cutoff is None or not when:
        return True                    # falsy (None, "") => fail OPEN
    try:
        stamp = str(when)[:10]
        # A value that is not even date-SHAPED cannot be ordered against the
        # cutoff — a naive string compare would rank "not-a-date" above it by
        # accident. Fail open explicitly instead.
        if len(stamp) < 10 or stamp[4] != "-" or stamp[7] != "-":
            return True
        return stamp >= cutoff.isoformat()
    except Exception:
        return True


def epoch_for(method: str) -> Optional[date]:
    """The scorer epoch for *method* as a DATE, for date-granular consumers.

    Returns the day AFTER the change when the epoch falls mid-day, so a caller
    that only has a date (the signals panel stores ``signal_date``) excludes the
    ambiguous partial day rather than half-admitting it. None = never changed.
    """
    cutoff = METHOD_SCORER_EPOCH.get(method)
    if cutoff is None:
        return None
    if cutoff.hour or cutoff.minute or cutoff.second:
        return cutoff.date() + timedelta(days=1)
    return cutoff.date()


def _as_datetime(value) -> Optional[datetime]:
    """Coerce a stored date/datetime/ISO string to an aware UTC datetime."""
    if value is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        text = str(value).strip().replace(" ", "T")
        for parse in (lambda s: datetime.fromisoformat(s),
                      lambda s: datetime.fromisoformat(s[:19]),
                      lambda s: datetime.strptime(s[:10], "%Y-%m-%d")):
            try:
                dt = parse(text)
                break
            except (TypeError, ValueError):
                continue
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def score_is_comparable(method: str, when) -> bool:
    """True when a score stored for *method* at *when* came from the CURRENT
    implementation, so it may be charged against that method's record.

    *when* should be the most precise timestamp available (a trade's
    ``entry_datetime``); a date-only value is compared at midnight UTC, which
    conservatively excludes the whole changeover day.

    Fail-OPEN: a method with no epoch, or an unparseable timestamp, is
    comparable — this gate only ever withholds evidence from a scorer known to
    have changed, and must never discard a method's history over a malformed
    date.
    """
    cutoff = METHOD_SCORER_EPOCH.get(method)
    if cutoff is None:
        return True
    ts = _as_datetime(when)
    if ts is None:
        return True
    return ts >= cutoff
