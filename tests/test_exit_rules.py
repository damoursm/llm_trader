"""Per-exit-rule funnel on the pivot target (2026-08-18).

The exit twin of the gate funnel: each rule judged by the positions it CLOSES vs
the ones it lets run, on the oriented REMAINING move to the next H/L pivot.

What these pin is the set of ways a SIMULATED rule can be silently wrong -- each
produces a table that renders normally with meaningless numbers:

  * a threshold compared against the wrong UNIT (ex_giveback is percentage
    points; the live rule compares it to a FRACTION of the peak);
  * an asymmetric rule applied symmetrically (the adverse stop is long 8% /
    short 20%, so a missing direction mis-fires every short);
  * a rule that cannot be reconstructed quietly never firing, which is
    indistinguishable from a rule that never triggers.
"""

import numpy as np
import pandas as pd

from config.settings import settings
from src.analysis import exit_rules as er


def test_rule_order_matches_the_live_evaluation_order():
    """A position is attributed to the FIRST rule that fires, so the order is
    what makes the cohorts partition -- and what each rule is measured on."""
    keys = [k for k, _ in er.EXIT_RULES]
    assert keys == ["macro_regime_exit", "combine_flip", "trailing_stop",
                    "adverse_stop", "mechanical_exit", "horizon_expired"]


def test_unsimulable_rules_are_declared_not_silently_passed():
    """llm_signal_flipped / ml_exit / edge_decay / method_horizon cannot be
    rebuilt from panel state. They must be NAMED, because a rule that never
    fires looks exactly like a rule that cannot run."""
    for r in ("llm_signal_flipped", "ml_exit", "edge_decay", "method_horizon"):
        assert r in er.UNSIMULATED_RULES
    # ...and the proxy that stands in for the LLM rule is labelled as a proxy
    label = dict(er.EXIT_RULES)["combine_flip"]
    assert "PROXY" in label


def test_trailing_stop_compares_giveback_against_a_FRACTION_OF_THE_PEAK():
    """ex_giveback is in percentage POINTS (peak minus current, median ~1.0,
    max ~96). The live rule closes on giving back trailing_give_back_frac (0.5)
    OF THE PEAK. Comparing the pp value against the bare 0.5 would fire on
    almost every position that ever moved."""
    import inspect
    src = inspect.getsource(er.simulate_exit_rules)
    assert "frac * mfe" in src, "give-back must be compared to frac x peak, not frac"
    assert "settings.trailing_arm_pct" in src


def test_adverse_stop_is_asymmetric_by_direction():
    """long 8% / short 20% -- the split runs OPPOSITE the usual prior because
    shorts against this book tend to recover. One shared threshold would close
    every short 12pp early."""
    import inspect
    src = inspect.getsource(er.simulate_exit_rules)
    assert "adverse_stop_pct_long" in src and "adverse_stop_pct_short" in src
    assert "is_long" in src
    assert float(settings.adverse_stop_pct_long) != float(settings.adverse_stop_pct_short)


def test_direction_is_recovered_from_the_entry_day():
    """The exit dataset carries no direction column, so it is joined from the
    panel on the position's ENTRY day -- where its direction was decided."""
    import inspect
    src = inspect.getsource(er.simulate_exit_rules)
    assert 'on=["entry_date", "ticker"]' in src


def test_value_sign_matches_the_gate_table():
    """Both funnels must read the same way: + value = removing that cohort
    helped. Sharing _cohort_stats is what guarantees it."""
    from src.analysis.gate_funnel import _cohort_stats
    assert er._cohort_stats is _cohort_stats
    fired = pd.DataFrame({"oriented": [-2.0] * 10, "sign": [1.0] * 10,
                          "day": [f"2026-08-{i+1:02d}" for i in range(10)]})
    held = pd.DataFrame({"oriented": [+1.0] * 10, "sign": [1.0] * 10,
                         "day": [f"2026-08-{i+1:02d}" for i in range(10)]})
    f, h = _cohort_stats(fired, 0.0), _cohort_stats(held, 0.0)
    assert (h["exc"] - f["exc"]) > 0        # closed the worse cohort => positive


def test_empty_dataset_is_safe(monkeypatch):
    monkeypatch.setattr("src.analysis.ml_exit_dataset.build_exit_dataset",
                        lambda **k: pd.DataFrame())
    out = er.compute_exit_rule_performance(days=5)
    assert out["rows"].empty and out["meta"] == {}
