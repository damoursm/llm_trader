"""Monitor-level exit fixes (2026-07-08) — profit-capture trailing stop + a
mechanical-consensus exit. The scorecard flagged a −58% MFE give-back (winners
round-trip) and that the LLM flip closes positions already ~5.6% down (too late);
these lean on the mechanical side and lock gains earlier.
"""

from config.settings import settings
from src.performance.tracker import _trailing_exit_triggered
from src.analysis.exit_conviction import exit_method_consensus


def _t(mfe, ret):
    return {"max_favorable_excursion": mfe, "return_pct": ret}


def test_trailing_stop_arms_and_captures(monkeypatch):
    monkeypatch.setattr(settings, "enable_trailing_exit", True)
    monkeypatch.setattr(settings, "trailing_arm_pct", 3.0)
    monkeypatch.setattr(settings, "trailing_give_back_frac", 0.5)

    assert not _trailing_exit_triggered(_t(2.0, 0.5))   # peak < arm → not armed
    assert not _trailing_exit_triggered(_t(6.0, 5.0))   # armed, gave back <50% (above trail 3.0) → hold
    assert _trailing_exit_triggered(_t(6.0, 3.0))       # exactly at trail (6×0.5) → exit
    assert _trailing_exit_triggered(_t(6.0, 1.5))       # gave back >50% → exit
    assert _trailing_exit_triggered(_t(6.0, -2.0))      # peaked then fully round-tripped → exit
    # a never-profitable position never arms (MFE stays ≤ 0)
    assert not _trailing_exit_triggered(_t(0.0, -4.0))


def test_trailing_stop_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_trailing_exit", False)
    assert not _trailing_exit_triggered(_t(10.0, 0.0))


def test_mechanical_consensus_direction():
    # exit_method_consensus: + = hold, − = exit. The mechanical exit fires on a
    # confidently negative consensus (money_flow/max_pain say exit).
    assert exit_method_consensus({"money_flow": -0.6, "max_pain": -0.5}) < 0   # → exit side
    assert exit_method_consensus({"money_flow": 0.6, "max_pain": 0.5}) > 0     # → hold side
    assert exit_method_consensus({}) is None                                   # no view
    # the LLM / aggregator / time-overlays are EXCLUDED from the mechanical consensus
    assert exit_method_consensus({"llm_review": -0.9, "aggregator": -0.9}) is None
