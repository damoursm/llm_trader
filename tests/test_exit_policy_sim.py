"""Sequential exit-policy simulator (`src/analysis/exit_policy_sim.py`).

The harness that removed the 5-day minimum hold and validated `ml_exit`'s 0.35
threshold. CLAUDE.md instructs re-running it before adding any time-based exit
constraint, which makes its arithmetic load-bearing for a decision that changes
how every arm position closes.

Two things have to be right or the conclusions invert:

* **close-once semantics.** A position is realized at the FIRST day the policy
  fires and the walk stops. Realizing at the last firing day (or at the final
  day regardless) would score every rule as buy-and-hold plus noise;
* **`excess`, not `mean_ret`, is the headline.** On a decaying book the raw
  return of any rule mostly reflects how long it holds, so the fixed-day
  controls trace return-vs-hold with zero information and the excess over that
  curve at the SAME average hold is the only part that is timing skill. A test
  that only checked `mean_ret` would happily bless a rule that just exits early.

The policies are pure state predicates, so they are tested directly; `simulate`
is tested against a hand-built two-position frame where the right answer is
arithmetic rather than statistical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import exit_policy_sim as eps


def _state(**kw) -> dict:
    s = {"days_held": 1, "ex_ret": 0.0, "ex_mfe": 0.0, "ex_mae": 0.0,
         "ex_consensus": 0.0, "ml_exit": 0.0}
    s.update(kw)
    return s


def _frame(rows) -> pd.DataFrame:
    """rows: list of (ticker, entry_date, [(days_held, ex_ret, **extra), ...])."""
    out = []
    for ticker, entry, days in rows:
        for d in days:
            rec = {"ticker": ticker, "entry_date": entry}
            rec.update(d)
            out.append(rec)
    return pd.DataFrame(out)


# ── the policies ────────────────────────────────────────────────────────────

def test_hold_to_end_never_closes():
    """The baseline every other rule is measured against."""
    assert eps.p_hold_to_end(_state(days_held=99, ml_exit=-1.0)) is False


def test_ml_exit_fires_only_on_a_confidently_negative_conviction():
    p = eps.p_ml_exit(threshold=0.35)
    assert p(_state(ml_exit=-0.36)) is True
    assert p(_state(ml_exit=-0.35)) is True            # boundary is inclusive
    assert p(_state(ml_exit=-0.34)) is False
    assert p(_state(ml_exit=+0.9)) is False


def test_ml_exit_treats_a_missing_score_as_hold():
    """No model output is not a reason to close — a fail-soft artifact would
    otherwise flatten the whole book."""
    p = eps.p_ml_exit()
    assert p(_state(ml_exit=None)) is False
    assert p(_state(ml_exit=float("nan"))) is False
    assert p({"days_held": 3}) is False


def test_a_negative_threshold_is_read_as_a_magnitude():
    p = eps.p_ml_exit(threshold=-0.35)
    assert p(_state(ml_exit=-0.5)) is True
    assert p(_state(ml_exit=+0.5)) is False


def test_min_hold_forbids_closing_early():
    """The constraint the harness exists to evaluate: identical rule, only the
    clock differing."""
    free = eps.p_ml_exit(min_hold=0)
    held = eps.p_ml_exit(min_hold=5)
    broken = _state(days_held=1, ml_exit=-0.9)
    assert free(broken) is True
    assert held(broken) is False
    assert held(_state(days_held=5, ml_exit=-0.9)) is True


def test_consensus_mirrors_the_ml_rule_on_its_own_column():
    p = eps.p_consensus(threshold=0.35)
    assert p(_state(ex_consensus=-0.4)) is True
    assert p(_state(ex_consensus=-0.1)) is False
    assert p(_state(ex_consensus=None)) is False
    # It must NOT read the ML column — the two are separate rules under test.
    assert p(_state(ex_consensus=0.0, ml_exit=-0.9)) is False


def test_trailing_arms_before_it_can_trigger():
    p = eps.p_trailing(arm_pct=3.0, give_back=0.5)
    assert p(_state(ex_mfe=2.0, ex_ret=0.0)) is False    # never armed
    assert p(_state(ex_mfe=10.0, ex_ret=6.0)) is False   # gave back < half
    assert p(_state(ex_mfe=10.0, ex_ret=5.0)) is True    # gave back half
    assert p(_state(ex_mfe=10.0, ex_ret=-2.0)) is True


def test_ev_rule_requires_the_expected_gain_to_beat_the_round_trip():
    """The principled alternative to a min-hold: hysteresis from the ECONOMICS,
    so a marginal negative view holds while a broken position can still exit on
    day 1."""
    p = eps.p_ev(cost_pct=1.0)
    wide = {"ex_mfe": 10.0, "ex_mae": -10.0}            # range 20
    assert p(_state(ml_exit=-0.9, **wide)) is True      # 0.9 * 20 = 18 > 1
    assert p(_state(ml_exit=-0.01, **wide)) is False    # 0.2 < 1
    assert p(_state(ml_exit=+0.9, **wide)) is False     # positive view never exits


def test_ev_rule_exits_a_broken_position_on_day_one():
    p = eps.p_ev(cost_pct=1.0)
    assert p(_state(days_held=1, ml_exit=-0.9,
                    ex_mfe=10.0, ex_mae=-10.0)) is True


def test_ev_rule_range_has_a_floor():
    """A position with no excursion yet would otherwise multiply the conviction
    by ~0 and never exit."""
    p = eps.p_ev(cost_pct=0.5)
    assert p(_state(ml_exit=-1.0, ex_mfe=0.0, ex_mae=0.0)) is True


def test_fixed_day_control_ignores_every_signal():
    """The load-bearing control — it must carry ZERO information."""
    p = eps.p_fixed_day(3)
    assert p(_state(days_held=2, ml_exit=-1.0)) is False
    assert p(_state(days_held=3, ml_exit=+1.0)) is True
    assert p(_state(days_held=9, ml_exit=+1.0)) is True


def test_random_control_is_seeded_and_reproducible():
    a = [eps.p_random(0.5, seed=7)(_state()) for _ in range(50)]
    b = [eps.p_random(0.5, seed=7)(_state()) for _ in range(50)]
    assert a == b
    assert eps.p_random(0.0, seed=1)(_state()) is False      # never exits
    assert eps.p_random(1.0, seed=1)(_state()) is True       # always exits


# ── close-once semantics ────────────────────────────────────────────────────

def _two_positions():
    """AAA reverses on day 2; BBB runs the whole way."""
    return _frame([
        ("AAA", "2026-07-01", [
            {"days_held": 1, "ex_ret": 1.0, "ml_exit": +0.9},
            {"days_held": 2, "ex_ret": 5.0, "ml_exit": -0.9},     # first EXIT
            {"days_held": 3, "ex_ret": -8.0, "ml_exit": -0.9},
        ]),
        ("BBB", "2026-07-01", [
            {"days_held": 1, "ex_ret": 1.0, "ml_exit": +0.9},
            {"days_held": 2, "ex_ret": 2.0, "ml_exit": +0.9},
            {"days_held": 3, "ex_ret": 3.0, "ml_exit": +0.9},
        ]),
    ])


def test_a_position_is_realized_at_the_FIRST_firing_day():
    """AAA must book +5.0 (day 2), not −8.0 (day 3). Taking the last firing day
    would make every rule look like buy-and-hold."""
    t = eps.simulate(_two_positions(), {"ml": eps.p_ml_exit()})
    row = t.iloc[0]
    assert row["positions"] == 2
    assert row["mean_ret"] == pytest.approx((5.0 + 3.0) / 2)
    assert row["avg_hold_d"] == pytest.approx((2 + 3) / 2)
    assert row["exit_rate_pct"] == pytest.approx(50.0)


def test_a_policy_that_never_fires_carries_to_the_last_day():
    t = eps.simulate(_two_positions(), {"hold": eps.p_hold_to_end})
    row = t.iloc[0]
    assert row["mean_ret"] == pytest.approx((-8.0 + 3.0) / 2)
    assert row["exit_rate_pct"] == 0.0


def test_costs_are_charged_only_when_the_policy_actually_closes():
    """A rule that churns can beat hold gross and lose net; charging the
    round trip to positions that never exited would erase that distinction."""
    free = eps.simulate(_two_positions(), {"ml": eps.p_ml_exit()}, cost_pct=0.0)
    paid = eps.simulate(_two_positions(), {"ml": eps.p_ml_exit()}, cost_pct=1.0)
    # Only AAA exited, so only AAA pays: the mean drops by cost/2.
    assert paid.iloc[0]["mean_ret"] == pytest.approx(free.iloc[0]["mean_ret"] - 0.5)

    held = eps.simulate(_two_positions(), {"hold": eps.p_hold_to_end}, cost_pct=1.0)
    assert held.iloc[0]["mean_ret"] == pytest.approx(
        eps.simulate(_two_positions(), {"hold": eps.p_hold_to_end}).iloc[0]["mean_ret"])


def test_every_policy_is_scored_on_the_SAME_positions():
    """Comparability is the whole point — a rule must not be able to improve its
    number by dropping positions it has no view on."""
    t = eps.simulate(_two_positions(), {
        "hold": eps.p_hold_to_end,
        "ml": eps.p_ml_exit(),
        "day1": eps.p_fixed_day(1),
    })
    assert set(t["positions"]) == {2}


def test_days_are_walked_in_order_regardless_of_input_order():
    """The frame arrives from a DB query; a policy replayed out of order would
    realize at the wrong day."""
    shuffled = _two_positions().iloc[::-1].reset_index(drop=True)
    t = eps.simulate(shuffled, {"ml": eps.p_ml_exit()})
    assert t.iloc[0]["mean_ret"] == pytest.approx((5.0 + 3.0) / 2)


def test_positions_are_keyed_on_ticker_AND_entry_date():
    """The same ticker held twice is two positions; merging them would splice
    two walks into one."""
    df = _frame([
        ("AAA", "2026-07-01", [{"days_held": 1, "ex_ret": 2.0, "ml_exit": 0.5}]),
        ("AAA", "2026-08-01", [{"days_held": 1, "ex_ret": 4.0, "ml_exit": 0.5}]),
    ])
    assert eps.simulate(df, {"hold": eps.p_hold_to_end}).iloc[0]["positions"] == 2


def test_an_empty_frame_returns_an_empty_table():
    assert eps.simulate(pd.DataFrame(), {"hold": eps.p_hold_to_end}).empty


def test_a_frame_missing_a_required_column_is_refused():
    """Loudly — a silently-skipped column would produce a plausible table built
    from the wrong positions."""
    df = _two_positions().drop(columns=["ex_ret"])
    with pytest.raises(ValueError, match="ex_ret"):
        eps.simulate(df, {"hold": eps.p_hold_to_end})


def test_win_rate_counts_positive_realized_returns():
    t = eps.simulate(_two_positions(), {"ml": eps.p_ml_exit()})
    assert t.iloc[0]["win_pct"] == pytest.approx(100.0)     # +5.0 and +3.0
    h = eps.simulate(_two_positions(), {"hold": eps.p_hold_to_end})
    assert h.iloc[0]["win_pct"] == pytest.approx(50.0)      # -8.0 and +3.0


# ── excess over the matched control ─────────────────────────────────────────

def _control_table():
    """A decaying book: holding longer earns less, with zero information."""
    return pd.DataFrame([
        {"policy": "[control] always exit d1", "avg_hold_d": 1.0, "mean_ret": 3.0},
        {"policy": "[control] always exit d5", "avg_hold_d": 5.0, "mean_ret": 1.0},
        {"policy": "[control] always exit d9", "avg_hold_d": 9.0, "mean_ret": -1.0},
        {"policy": "signal rule", "avg_hold_d": 5.0, "mean_ret": 2.5},
        {"policy": "early rule", "avg_hold_d": 1.0, "mean_ret": 3.0},
    ])


def test_excess_is_measured_against_the_control_at_the_same_hold():
    """The headline. `early rule` earns MORE than `signal rule` in raw return
    and contributes NOTHING — it just holds for a shorter time on a decaying
    book. Ranking on `mean_ret` gets this exactly backwards."""
    t = eps.add_excess_vs_matched_control(_control_table())
    by = t.set_index("policy")["excess"]
    assert by["signal rule"] == pytest.approx(1.5)     # 2.5 vs the d5 control's 1.0
    assert by["early rule"] == pytest.approx(0.0)      # identical to its control
    assert by["signal rule"] > by["early rule"]


def test_excess_interpolates_between_control_points():
    t = eps.add_excess_vs_matched_control(pd.DataFrame([
        {"policy": "[control] always exit d1", "avg_hold_d": 1.0, "mean_ret": 4.0},
        {"policy": "[control] always exit d5", "avg_hold_d": 5.0, "mean_ret": 0.0},
        {"policy": "mid", "avg_hold_d": 3.0, "mean_ret": 2.0},
    ]))
    assert t.set_index("policy")["excess"]["mid"] == pytest.approx(0.0)


def test_the_controls_score_zero_excess_against_themselves():
    t = eps.add_excess_vs_matched_control(_control_table())
    ctrl = t[t["policy"].str.startswith("[control]")]
    assert all(abs(v) < 1e-9 for v in ctrl["excess"])


def test_excess_is_nan_without_controls_rather_than_a_bare_return():
    """No control curve means the number cannot be computed; emitting
    `mean_ret` under the name `excess` would silently restore the exact
    confusion the column exists to prevent."""
    t = eps.add_excess_vs_matched_control(pd.DataFrame([
        {"policy": "signal rule", "avg_hold_d": 5.0, "mean_ret": 2.5}]))
    assert np.isnan(t.iloc[0]["excess"])
