"""Business-cycle sector rotation (`src/data/business_cycle_rotation.py`).

Turns the FRED macro read into a cycle PHASE and a per-sector bias that rides
the synthesis prompt. Pure given its inputs — no network — so the phase decision
table is fully reachable, which matters because it is a chain of `if` branches
whose ORDER is the logic: `RECESSION` must win over everything, an inverted
curve must override the expansion sub-classification, and so on. Reordering two
branches changes the phase without changing any individual condition.

The other property worth pinning is that an absent macro context yields UNKNOWN
rather than a default phase. A phase is a strong claim about the world; deriving
one from missing data would put a confident cycle narrative in the prompt built
on nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data import business_cycle_rotation as bc


def _macro(regime="EXPANSION", yield_curve="NORMAL", inflation="MODERATE",
           unemployment="STABLE", cpi=2.4):
    return SimpleNamespace(regime=regime, yield_curve_signal=yield_curve,
                           inflation_signal=inflation,
                           unemployment_trend=unemployment, cpi_yoy=cpi)


# ── phase classification ────────────────────────────────────────────────────

def test_no_macro_context_is_UNKNOWN_not_a_default_phase():
    phase, evidence = bc._classify_phase(None)
    assert phase == "UNKNOWN"
    assert "no FRED" in evidence


def test_recession_wins_over_every_other_input():
    """The branch order IS the logic: a RECESSION regime must not be
    re-classified by the curve or the inflation reading below it."""
    for yc in ("NORMAL", "INVERTED", "FLAT"):
        for infl in ("LOW", "HIGH", "MODERATE"):
            phase, _e = bc._classify_phase(
                _macro(regime="RECESSION", yield_curve=yc, inflation=infl))
            assert phase == "CONTRACTION"


@pytest.mark.parametrize("yc", ["NORMAL", "INVERTED", "FLAT"])
def test_late_cycle_regime_stays_late_cycle(yc):
    assert bc._classify_phase(_macro(regime="LATE_CYCLE", yield_curve=yc))[0] \
        == "LATE_CYCLE"


def test_a_slowdown_is_late_cycle_only_when_the_curve_is_inverted():
    """The inversion is what separates 'heading down' from 'still expanding' —
    the single most consequential branch in the table."""
    assert bc._classify_phase(
        _macro(regime="SLOWDOWN", yield_curve="INVERTED"))[0] == "LATE_CYCLE"
    assert bc._classify_phase(
        _macro(regime="SLOWDOWN", yield_curve="NORMAL"))[0] == "LATE_EXPANSION"


def test_expansion_is_sub_classified_by_inflation_and_employment():
    early = bc._classify_phase(
        _macro(regime="EXPANSION", inflation="LOW", unemployment="FALLING"))[0]
    mid = bc._classify_phase(
        _macro(regime="EXPANSION", inflation="MODERATE", unemployment="STABLE"))[0]
    late = bc._classify_phase(
        _macro(regime="EXPANSION", inflation="HIGH", unemployment="STABLE"))[0]
    assert (early, mid, late) == ("EARLY_EXPANSION", "MID_EXPANSION", "LATE_EXPANSION")


def test_rising_unemployment_ages_an_expansion_regardless_of_inflation():
    assert bc._classify_phase(
        _macro(regime="EXPANSION", inflation="LOW",
               unemployment="RISING"))[0] == "LATE_EXPANSION"


def test_an_unrecognised_regime_is_UNKNOWN():
    assert bc._classify_phase(_macro(regime="SOMETHING_NEW"))[0] == "UNKNOWN"


def test_evidence_names_every_input_it_used():
    """The narrative is shown to the model; evidence that omits an input it
    actually branched on is a misleading explanation."""
    _phase, ev = bc._classify_phase(
        _macro(regime="EXPANSION", yield_curve="NORMAL",
               inflation="MODERATE", unemployment="FALLING", cpi=3.1))
    for token in ("regime=", "yield_curve=", "inflation=", "unemployment=", "cpi_yoy="):
        assert token in ev


def test_missing_cpi_is_omitted_rather_than_printed_as_none():
    _phase, ev = bc._classify_phase(_macro(cpi=None))
    assert "cpi_yoy" not in ev


def test_a_context_missing_fields_entirely_still_classifies():
    """`getattr` defaults — a partially-populated FRED context must degrade,
    not raise, inside the prompt build."""
    phase, _ev = bc._classify_phase(SimpleNamespace())
    assert phase == "UNKNOWN"


# ── the score → signal ladder ───────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (1.0, "STRONG_LEADER"), (0.55, "STRONG_LEADER"),
    (0.54, "LEADER"), (0.25, "LEADER"),
    (0.24, "NEUTRAL"), (0.0, "NEUTRAL"), (-0.24, "NEUTRAL"),
    (-0.25, "LAGGARD"), (-0.54, "LAGGARD"),
    (-0.55, "STRONG_LAGGARD"), (-1.0, "STRONG_LAGGARD"),
])
def test_score_ladder_boundaries(score, expected):
    assert bc._score_to_signal(score) == expected


def test_the_ladder_is_symmetric_and_monotone():
    """An asymmetric ladder would systematically over- or under-call leaders
    relative to laggards for the same evidence."""
    rank = {"STRONG_LAGGARD": 0, "LAGGARD": 1, "NEUTRAL": 2,
            "LEADER": 3, "STRONG_LEADER": 4}
    xs = [i / 100 for i in range(-100, 101)]
    ranks = [rank[bc._score_to_signal(x)] for x in xs]
    assert ranks == sorted(ranks)
    for x in xs:
        assert rank[bc._score_to_signal(x)] + rank[bc._score_to_signal(-x)] == 4


# ── the phase tables ────────────────────────────────────────────────────────

def test_every_phase_has_a_narrative_and_a_direction():
    """A phase present in the classifier but missing from either table is a
    KeyError inside the prompt build — or worse, a silently empty narrative."""
    phases = {"CONTRACTION", "LATE_CYCLE", "LATE_EXPANSION",
              "MID_EXPANSION", "EARLY_EXPANSION", "UNKNOWN"}
    assert phases <= set(bc._PHASE_NARRATIVES), \
        f"missing narratives: {sorted(phases - set(bc._PHASE_NARRATIVES))}"
    assert phases <= set(bc._PHASE_DIRECTIONS), \
        f"missing directions: {sorted(phases - set(bc._PHASE_DIRECTIONS))}"


def test_every_phase_score_names_a_real_sector_etf():
    """A typo'd key scores 0.0 via `score_map.get(etf, 0.0)` — the sector reads
    NEUTRAL in every phase and nothing reports the dead entry."""
    known = set(bc._SECTOR_NAMES)
    assert known, "sector name table is empty"
    for phase, mapping in bc._PHASE_SCORES.items():
        unknown = set(mapping) - known
        assert not unknown, f"{phase} references unknown sectors: {sorted(unknown)}"


def test_phase_scores_are_bounded():
    for phase, mapping in bc._PHASE_SCORES.items():
        for sector, v in mapping.items():
            assert -1.0 <= float(v) <= 1.0, f"{phase}/{sector} = {v}"


def test_every_phase_has_a_score_table():
    """`_PHASE_SCORES.get(phase, UNKNOWN)` means a missing phase silently falls
    back to the no-bias table — the block keeps rendering with every sector
    neutral, which is indistinguishable from a genuinely flat cycle read."""
    assert set(bc._PHASE_NARRATIVES) <= set(bc._PHASE_SCORES)
    assert "UNKNOWN" in bc._PHASE_SCORES


def test_the_direction_table_is_a_phase_to_label_map():
    valid = {"BULLISH", "BEARISH", "NEUTRAL"}
    assert set(bc._PHASE_DIRECTIONS.values()) <= valid


# ── the public context ──────────────────────────────────────────────────────

def test_context_is_built_from_the_classified_phase():
    ctx = bc.compute_business_cycle_context(_macro(regime="EXPANSION",
                                                   inflation="LOW",
                                                   unemployment="FALLING"), None)
    assert ctx.cycle_phase == "EARLY_EXPANSION"
    assert ctx.cycle_direction == bc._PHASE_DIRECTIONS["EARLY_EXPANSION"]
    assert ctx.summary and ctx.evidence
    # Every sector is scored, sorted best-first, with the leaders extracted.
    assert {b.etf for b in ctx.sector_biases} == set(bc._SECTOR_NAMES)
    scores = [b.cycle_score for b in ctx.sector_biases]
    assert scores == sorted(scores, reverse=True)
    assert "XLF" in ctx.top_cycle_leaders


def test_context_survives_a_missing_macro_context():
    """FRED is one feed among many; losing it must degrade this block, not the
    run."""
    ctx = bc.compute_business_cycle_context(None, None)
    assert ctx.cycle_phase == "UNKNOWN"
    assert ctx.summary
    # UNKNOWN must apply NO tilt at all — a fabricated rotation from missing
    # data is worse than none.
    assert all(b.cycle_score == 0.0 for b in ctx.sector_biases)
    assert ctx.top_cycle_leaders == [] and ctx.weak_cycle_sectors == []


def test_context_survives_a_missing_rotation_context():
    ctx = bc.compute_business_cycle_context(_macro(), None)
    assert ctx.cycle_phase and ctx.summary and ctx.convergence_notes is not None
