"""Cross-sectional ranking overlay (`src/signals/cross_sectional.py`).

An ADDITIVE overlay on `combined_score` (weight 0.20) that measures how far each
ticker deviates from the universe on each method. Two design decisions carry all
the risk, and both are invisible from the output:

* **a zero score is an ABSENT view, not a neutral one.** If a method that did not
  fire were folded into the mean, a ticker would be pulled toward the universe
  average by methods that never looked at it — and in a universe where most
  methods abstain (the normal case) that dominates the result;
* **σ needs a real cross-section.** Fewer than 3 non-zero readings gives a
  meaningless standard deviation, and dividing by it manufactures enormous
  z-scores from noise.

The overlay also rides the inversion/win-rate machinery via `_overlay_factor`,
so its SIGN is load-bearing: a ticker above the universe mean must score
positive.
"""

from __future__ import annotations

import pytest

from src.models import TickerSignal
from src.signals.cross_sectional import (_METHODS_FOR_RANKING, _ZERO_THRESHOLD,
                                         compute_cross_sectional_scores)


def _sig(ticker: str, **scores) -> TickerSignal:
    base = dict(sentiment_score=0.0, technical_score=0.0, rationale="t")
    base.update(scores)
    return TickerSignal(ticker=ticker, direction="NEUTRAL", confidence=0.5, **base)


def _universe(tech_scores: dict) -> list:
    return [_sig(tk, technical_score=v) for tk, v in tech_scores.items()]


# ── the core z-score ────────────────────────────────────────────────────────

def test_the_standout_scores_positive_and_the_laggard_negative():
    """Sign convention: above the universe mean = bullish. The overlay is added
    straight onto `combined_score`, so an inverted sign here silently subtracts
    conviction from the best names."""
    out = compute_cross_sectional_scores(
        _universe({"HIGH": 0.9, "MID": 0.5, "LOW": 0.1}))
    assert out["HIGH"] > 0 > out["LOW"]
    assert abs(out["MID"]) < abs(out["HIGH"])


def test_a_uniform_universe_has_no_standouts():
    """Every ticker identical → σ is 0 → the method is skipped entirely rather
    than producing a divide-by-zero or an all-zero-mean artefact."""
    out = compute_cross_sectional_scores(_universe({"A": 0.5, "B": 0.5, "C": 0.5}))
    assert set(out) == {"A", "B", "C"}
    assert all(v == 0.0 for v in out.values())


def test_scores_are_bounded_to_the_unit_interval():
    out = compute_cross_sectional_scores(
        _universe({"A": 1.0, "B": 0.0001, "C": -1.0, "D": 0.0002, "E": 0.0003}))
    assert all(-1.0 <= v <= 1.0 for v in out.values())


def test_zcap_bounds_a_single_outlier():
    """Without the clip, one extreme reading dominates the average z and the
    overlay stops describing the cross-section."""
    tight = {"A": 0.50, "B": 0.51, "C": 0.52, "OUT": 5.0}
    wide = compute_cross_sectional_scores(_universe(tight), zcap=2.5)
    narrow = compute_cross_sectional_scores(_universe(tight), zcap=0.5)
    assert abs(wide["OUT"]) <= 1.0 and abs(narrow["OUT"]) <= 1.0
    # A smaller cap saturates sooner, so the outlier pins to the boundary.
    assert narrow["OUT"] == pytest.approx(1.0)


# ── zero means "no view" ────────────────────────────────────────────────────

def test_a_zero_score_is_an_absent_view_not_a_neutral_one():
    """The decisive case. If zeros counted as views, ABSTAIN (0.0) would sit
    below the universe mean of a bullish cross-section and score BEARISH — a
    method that never looked at the ticker would push it toward a short."""
    sigs = [_sig("A", technical_score=0.8), _sig("B", technical_score=0.6),
            _sig("C", technical_score=0.7), _sig("QUIET", technical_score=0.0)]
    out = compute_cross_sectional_scores(sigs)
    assert out["QUIET"] == 0.0, "an abstention was scored as a bearish deviation"


def test_a_ticker_is_averaged_only_over_the_methods_that_fired():
    """One strong reading and three abstentions must read as strongly as one
    strong reading alone — not diluted by the silent methods."""
    def _u(extra):
        return [_sig("A", technical_score=0.9, **extra),
                _sig("B", technical_score=0.1, **extra),
                _sig("C", technical_score=0.2, **extra)]

    bare = compute_cross_sectional_scores(_u({}))
    # Adding a method on which everyone abstains must change nothing.
    padded = compute_cross_sectional_scores(_u({"vwap_score": 0.0}))
    assert bare == padded


def test_near_zero_readings_are_treated_as_abstentions():
    below = _ZERO_THRESHOLD / 2
    sigs = [_sig("A", technical_score=0.8), _sig("B", technical_score=0.6),
            _sig("C", technical_score=0.7), _sig("TINY", technical_score=below)]
    assert compute_cross_sectional_scores(sigs)["TINY"] == 0.0


# ── evidence floors ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 2])
def test_a_universe_too_small_to_rank_returns_empty(n):
    """Empty dict, not zeros: the aggregator skips the overlay entirely rather
    than adding a fabricated 0 to every score."""
    sigs = [_sig(f"T{i}", technical_score=0.5) for i in range(n)]
    assert compute_cross_sectional_scores(sigs) == {}


def test_a_method_with_fewer_than_three_readings_is_skipped():
    """σ over two points is not a cross-section. The method contributes nothing
    rather than an enormous z."""
    sigs = [_sig("A", technical_score=0.9, vwap_score=0.9),
            _sig("B", technical_score=0.1, vwap_score=-0.9),
            _sig("C", technical_score=0.2)]                  # vwap abstains
    with_vwap = compute_cross_sectional_scores(sigs)
    tech_only = compute_cross_sectional_scores(
        [_sig("A", technical_score=0.9), _sig("B", technical_score=0.1),
         _sig("C", technical_score=0.2)])
    assert with_vwap == tech_only


def test_all_methods_silent_yields_zero_for_every_ticker():
    """Not an empty dict — the universe WAS big enough to rank, there was simply
    nothing to rank on, and the caller should see a 0 overlay for each name."""
    out = compute_cross_sectional_scores([_sig(f"T{i}") for i in range(4)])
    assert out == {f"T{i}": 0.0 for i in range(4)}


# ── multi-method aggregation ────────────────────────────────────────────────

def test_agreeing_methods_reinforce_and_disagreeing_ones_cancel():
    agree = [_sig("A", technical_score=0.9, momentum_score=0.9),
             _sig("B", technical_score=0.1, momentum_score=0.1),
             _sig("C", technical_score=0.2, momentum_score=0.2)]
    disagree = [_sig("A", technical_score=0.9, momentum_score=0.1),
                _sig("B", technical_score=0.1, momentum_score=0.9),
                _sig("C", technical_score=0.2, momentum_score=0.2)]
    assert compute_cross_sectional_scores(agree)["A"] > \
        compute_cross_sectional_scores(disagree)["A"]


def test_missing_attributes_are_treated_as_no_view():
    """`getattr(s, attr, 0.0)` — a signal object lacking a method attribute
    entirely must not raise inside the scoring loop."""
    from types import SimpleNamespace
    sigs = [SimpleNamespace(ticker=f"T{i}", technical_score=v)
            for i, v in enumerate([0.9, 0.1, 0.2])]
    out = compute_cross_sectional_scores(sigs)
    assert set(out) == {"T0", "T1", "T2"}


# ── the method list ─────────────────────────────────────────────────────────

def test_the_overlay_never_ranks_itself():
    """`cross_sectional` is applied on TOP of the combine; including its own
    score in the z-scored set would be recursive."""
    assert "cross_sectional_score" not in _METHODS_FOR_RANKING


def test_ranked_attributes_all_exist_on_the_signal_model():
    """A renamed field would silently become a permanent abstention for every
    ticker — the overlay would keep running and quietly stop measuring that
    method."""
    fields = set(TickerSignal.model_fields)
    missing = [a for a in _METHODS_FOR_RANKING if a not in fields]
    assert not missing, f"{missing} are ranked but no longer exist on TickerSignal"


def test_ranked_attributes_are_normalised_scores_not_raw_readings():
    """The z-scores assume a common [-1, +1] scale; mixing in a raw indicator
    (a price distance, a percentage) would let one method's units dominate."""
    assert all(a.endswith("_score") for a in _METHODS_FOR_RANKING)
