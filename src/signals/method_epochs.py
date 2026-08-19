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
    # 2026-08-17 01:40 UTC (v3; the shared deploy boundary — see the `news`
    # entry for how it was placed by the runs): the
    # score is CMF ALONE. The 3-year gated battery (754 daily cross-sections /
    # 752k rows / 1,206 tickers, signed pivot target) measured the v2 composite
    # at IC +0.0349 (t +10.2) but CMF alone at +0.0479 (t +11.8, paired t +9.0
    # over the composite, best every year, both sides balanced); the OBV slope
    # term carried nothing (+0.0038, t +0.97, robust to renormalisation) and
    # the contrarian-MFI term was ANTI-predictive (−0.0140, t −2.72; flipped it
    # merely tied no-MFI, paired t −0.3). Same-frame output changes wherever
    # OBV/MFI were active → histories must not pool. money_flow is REPLAYABLE:
    # the panel history is REGENERATED under v3 (replay --write ran with the
    # change; the nightly refactor keeps it current), so the mask only covers
    # what replay cannot restore. Supersedes the 07-24 instant.
    "money_flow": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    # 2026-08-08 16:30 UTC (scheduler restart): ml_ohlcv v1 → v2 — a CATEGORICAL
    # swap, not a retrain. v1 was P(up)−P(down) from a 10-day market-relative
    # classifier, emitted only on the clean-trend+liquid subset (NO_VIEW
    # elsewhere); v2 is 2× a predicted within-day RANK of the signed pivot
    # return, 85 features (76 + 9 leg-state), full universe. Different target,
    # different feature set, different score semantics — the two histories must
    # never pool. Promotion gate: `python -m src.analysis.ml_validate --target
    # pivot` (panel IC +0.0639, t +2.31, edge +2.58pp over 14,610 rows /
    # 2026-08-08); record in memory/pivot-horizon-target-2026-08.md. Ordinary
    # retrains within v2 stay the ml_models registry's job, per the module rule.
    # 2026-08-12 21:00 UTC (deploy-2 restart): the PIVOT DEFINITION moved —
    # close-based zero-threshold zigzag → H/L basis (peaks on highs, troughs on
    # lows, targets to the pivot bar's EXTREME) + the 1% minimum-move threshold
    # (`pivot_min_move_pct`; both user directives, same day). The target AND
    # the 9 leg features changed meaning, so every stored ml_ohlcv score is a
    # different quantity. Serving carries its own second guard: the artifact
    # is stamped `pivot_basis` ("hl1") and a stale-basis artifact ABSTAINS
    # (BASIS_STALE), so between this instant and the retrain landing the
    # method scores nothing rather than something wrong. Supersedes the
    # 2026-08-08 v1→v2 entry (any pre-hl1 row is already pre-this-epoch).
    "ml_ohlcv": datetime(2026, 8, 12, 21, 0, tzinfo=timezone.utc),
    # 2026-08-10 16:20 UTC (scheduler restart; the instant sits between the last
    # v1 run 16:00:14 and the first v2 run 16:38:50 — the restarted scheduler's
    # misfire-grace catch-up tick stamped 16:38:50, EARLIER than the restart
    # command finished, which is why the boundary is placed by the RUNS, not by
    # the wall clock of the restart): st_reversal v1 → v2. v1 scored
    # −tanh(z/1.5) with z = ret_5d / the ticker's OWN weekly std — a
    # self-normalised number whose magnitude meant "how unusual for this name".
    # v2 scores −tanh(ret_5d / 0.05) — a FIXED scale, so magnitude now means
    # "how big the week was" cross-sectionally. Same sign, different meaning →
    # the two histories must not pool. Basis: the 20y Gate-4 MR battery
    # (raw-return ranking IC +0.0150/t +7.2 vs z-version +0.0114/t +6.2; see
    # memory/pivot-horizon-target-2026-08.md).
    "st_reversal": datetime(2026, 8, 10, 16, 20, tzinfo=timezone.utc),
    # 2026-08-11 02:05 UTC (scheduler restart; last v1 run 2026-08-10T20:00 UTC,
    # first v2 run stamped 02:13 — the catch-up tick's generated_at leads the
    # restart wall clock, so the boundary is placed by the RUNS, as always): vwap window 20 → 5 sessions.
    # A 5-session VWAP anchor is a DIFFERENT reference frame than a 20-session
    # one (weekly vs monthly cost basis), so the two score histories must not
    # pool in ledger-based calibrations — and vwap is a WEIGHTED method (0.12),
    # so the win-rate filter / adaptive tilt genuinely consume that history.
    # Panel side: vwap is REPLAYABLE, so the nightly refactor regenerates its
    # panel history under the new window and build_panel prefers the restored
    # cells — the designed epoch+replay combo. Basis: the 62-variant parameter
    # sweep (h2 IC +0.0082 vs +0.0036, paired t +2.16; pivot memory 2026-08-10).
    "vwap": datetime(2026, 8, 11, 2, 5, tzinfo=timezone.utc),
    # 2026-08-14 (scheduler restart, news-family continuity pass): the insider
    # score's evidence weights went CONTINUOUS — log-dollar bucket midpoints
    # (was bucket/8 steps), per-day recency decay on the transaction date (was
    # no time term at all), and a seniority multiplier. Measured motivation:
    # 331 views/run shared 62 distinct values (81% tie mass; one value covered
    # 3,027 panel rows), so the rank transform collapsed most of the method to
    # the median and the shaping deciles were fit on ties. Same direction
    # logic, same /3 normaliser — but the VALUE for the same filing set is a
    # different number, so histories must not pool.
    "insider": datetime(2026, 8, 14, 18, 30, tzinfo=timezone.utc),
    # 2026-08-14 18:30 UTC: the news score's precision scalers went
    # continuous — evidence mass = Σ per-article recency weights replaces the
    # article COUNT in the count-scale (3 fresh articles no longer equal 3
    # stale ones), and the 0.70/0.85/1.0 source-diversity steps became a
    # smooth curve. The raw LLM verdict also gained a two-decimal instruction
    # (cache-salted, so old cached verdicts cannot leak into the new scale).
    # The LLM's round-number quantization (top value 0.7 covered 7.1% of all
    # nonzero rows) multiplied by step scalers left ~38% within-run tie mass.
    # 2026-08-17 01:40 UTC (prompt v3/v4 — the RAW-verdict standard moved, the
    # scalers did not). Measured on 2,265 v2-prompt calls: nonzero verdicts
    # collapsed onto 29 distinct values (98% tie mass) and the MODAL values
    # were v2's own example numbers (0.47/−0.62/0.71 — the "two-decimal"
    # instruction produced two-decimal-LOOKING copies, not variance); and
    # post-surge catalysts kept full magnitude (+0.90 AFTER a reported 75–115%
    # pre-market spike — the measured BUY-side chasing cohort Gate 5 exists
    # for). v3 derives the second decimal from a band-then-placement rubric,
    # adds a priced-in/remaining-move check with the negative-drift asymmetry,
    # tiers company-issued PR and aggregator listicles below independent
    # reporting, and emits rationale BEFORE score. Same article set → a
    # different verdict by design, so the histories must not pool.
    #
    # BOUNDARY — MOVED at deploy time, exactly as the rule demands. It was
    # provisionally written as 2026-08-16 01:00 UTC on the assumption of a
    # restart before the Sunday-night overnight open (Sun 20:30 ET). The
    # restart actually landed at 2026-08-17 01:42 UTC (Sun 21:42 ET), by which
    # time the overnight session had already run TWO OLD-CODE ticks past that
    # provisional instant (runs 2026-08-17_003019 and _013003, 397 signal rows
    # each). Leaving it would have admitted old-formula rows as current — the
    # exact failure the 2026-08-14 confidence-epoch near-miss warned about.
    # Placed by the RUNS, per the standing convention: strictly after the last
    # old-code run (generated_at 2026-08-17T01:30:03Z) and strictly before the
    # first new-code run (the post-restart catch-up tick for the 21:30 ET slot,
    # generated_at 2026-08-17T01:42:5xZ; `run_id`/`generated_at` are
    # `datetime.now(timezone.utc)` at run start, NOT the slot time — verified
    # at pipeline.py:1375). Supersedes the 08-14 instant (any pre-v3 row is
    # already pre-this-epoch). All eight instants registered this weekend share
    # this boundary for the same reason.
    "news": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    # 2026-08-17 01:40 UTC (shared deploy boundary): f_dividend reworked from a
    # stale trailing-year tilt (latest cash vs ~1yr ago, scored EVERY day for a
    # quarter, specials/frequency-mixes included — 18 of its 24 deep-cut panel
    # readings decomposed as data artifacts, and its +0.4 one-row "initiation"
    # measured −0.61%/42.9% win) to an EVENT-windowed, RAISE-ONLY declaration
    # factor: regular-vs-prior-comparable change, tanh(chg×4), linear decay to
    # 0 across ~10 days. Two-year event study (5,804 gated labeled events):
    # the raise side is monotone and replicates in both halves (flat +0.28% →
    # raise +0.42% → big +1.17% → huge +1.98%, drift confined to ~5 sessions,
    # d5→d10 ≈ 0); ALL cuts abstain — the big-cut drift flipped sign across
    # halves (H1 −1.86% / H2 +0.27%), and small cuts were positive in both.
    # A different quantity on every row → histories must not pool.
    # Non-replayable (external feed).
    "f_dividend": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    # 2026-08-17 01:40 UTC (shared deploy boundary), all four trend-context
    # methods at once: the learned ORIENTATION these persisted scores BAKE IN
    # (score = orientation × context strength) was rebased — old: a drift-
    # biased continuation-rate at the fixed horizon shrunk toward a +1
    # CONTINUATION prior, which held every context at +0.28..+0.39 while the
    # panel measured all four DESCENDING on the pivot basis (adx_long daily IC
    # −0.049 t −3.2; the 2026-08-16 decile-direction audit). New: per-day rank
    # IC of the orientation-FREE feature vs the signed pivot target, shrunk
    # toward 0 = abstain — so the same context can now carry the OPPOSITE sign
    # (or none). Sign semantics changed → histories must not pool. Not
    # replayable (the orientation depends on panel state at serve time).
    "kaufman_long": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    "kaufman_short": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    "adx_long": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    "adx_short": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
    # 2026-08-17 01:40 UTC (shared deploy boundary): put_call went CONTINUOUS —
    # the 5-label step map emitted FOUR distinct values across 1,230 panel rows
    # (±0.35/±0.70), collapsing each day's ~24-name cross-section into ≤4
    # rank-tie blocks; the decile curve was tie-block noise and the payoff
    # shaping fit on it flipped out-of-sample (ΔIC −0.269 in the 2026-08-16
    # shape audit). Now tanh(ln(ratio)/scale) with per-side scales anchored to
    # the OLD extreme values at their documented thresholds (2.0 → +0.70,
    # 0.3 → −0.70) — the same quantity with the steps removed, but the VALUE
    # for the same chain is different → histories must not pool. Same disease,
    # same cure, same rule as the news/insider continuity epochs (2026-08-14).
    "put_call": datetime(2026, 8, 17, 1, 40, tzinfo=timezone.utc),
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
#
#   2026-08-14 method RANK basis (user directive): the combine consumes every
#   method's CENTERED WITHIN-RUN RANK instead of its absolute score
#   (`aggregator._rank_transform_run`, `method_score_basis`). Raw scores stay
#   persisted (no scorer epoch — outputs unchanged), but |combined| and every
#   confidence ingredient built on the map (coherence, sources_agreeing,
#   family votes) now live on a different scale: mean |eff score| jumps from
#   ~0.1-0.2 to ~0.5 by construction. Same categorical test as 2026-07-22.
#   Instant = the first scheduler restart with the code; rows before it were
#   produced by the absolute-basis combine. Pinned to the ACTUAL restart, not to
#   the intent: the original placeholder (02:00 UTC) was written before the code
#   was even committed (05:16 UTC), i.e. it sat inside the absolute era, and only
#   the date-granular `confidence_epoch()` ("day after a mid-day change" ->
#   2026-08-15) kept that harmless. Had the restart slipped past 08-15 the mask
#   would have silently admitted absolute-era rows as current-formula.
#
#   The same instant covers the confidence-FORMULA repairs that shipped in the
#   same restart (2026-08-14): confidence is now computed by ONE function
#   (`aggregator._confidence_from`) at all three sites, the cross-sectional
#   overlay RE-DERIVES it from the adjusted score instead of rescaling the
#   already-rounded/capped value, and the sector-alignment multiplier became a
#   persisted component (`sector_conf_factor`) instead of an unrecorded seventh
#   factor. Values move on ~20% of rows (materially where raw_confidence had
#   capped); measured Gate-1 effect on the live panel was 5 of 787 rows at
#   >=0.85 falling below it. Verified post-restart: 418/418 rows reconstruct
#   from their stored components (82/420 before the repairs, 295/420 after the
#   divisor alone). The instant sits a few minutes AFTER the restart that went
#   live (first run stamped 13:30:26 UTC), which costs that one run and nothing
#   else — `confidence_epoch()` is date-granular and returns 2026-08-15, so
#   every 08-14 row is masked either way.
CONFIDENCE_EPOCH: datetime = datetime(2026, 8, 14, 13, 35, tzinfo=timezone.utc)

# Columns the confidence epoch governs: the value plus the six ingredients that
# are only interpretable alongside it.
CONFIDENCE_EPOCH_COLUMNS: tuple = (
    "confidence", "raw_confidence", "coherence_factor", "movement_factor",
    "volume_factor", "family_conf_factor", "tape_conf_factor",
    "sector_conf_factor",
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
