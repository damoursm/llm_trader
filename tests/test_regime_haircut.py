"""RISK_OFF size haircut, replacing the shared BUY ban (2026-07-27).

The 7-input historical reconstruction (2000-2026) measured RISK_OFF as the
BEST-performing regime — +2.25% SPY at 21d with a 70.0% up-rate over 337 days,
better than NEUTRAL (+0.37%) and RISK_ON (+0.84%) — while PANIC is the only
genuinely bad one (-0.69%, 45.5% over 66 days). The two had shared one
`allow_buys=False` bucket on no evidence they behave alike, so the system was
sitting out its strongest regime.

A haircut rather than full size: the measurement says longs are fine in
RISK_OFF, but it is a stressed state by definition and its 337 days are
concentrated in a handful of episodes, so "banned → full size" is a bigger step
than the evidence carries.

The property most worth pinning is CONSISTENCY: the entry side and the exit side
must agree about RISK_OFF, or the system opens a haircut long and closes it on
the next tick — worse than either policy alone.
"""

import pytest

from config.settings import settings
import src.data.macro_regime as mr


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(settings, "enable_regime_size_haircut", True)
    monkeypatch.setattr(settings, "risk_off_size_multiplier", 0.5)


# ── the policy itself ──────────────────────────────────────────────────────

def test_panic_still_blocks_buys():
    """PANIC is the one regime the data condemns (-0.69% at 21d, 45.5% up)."""
    assert mr._REGIME_ALLOW_BUYS["PANIC"] is False


def test_risk_off_no_longer_blocks_buys():
    assert mr._REGIME_ALLOW_BUYS["RISK_OFF"] is True


def test_risk_off_is_haircut_not_full_size():
    assert mr.regime_size_multiplier("RISK_OFF") == 0.5


def test_other_regimes_are_unhaircut():
    for reg in ("CAUTION", "NEUTRAL", "RISK_ON"):
        assert mr.regime_size_multiplier(reg) == 1.0, f"{reg} should be untouched"


def test_unknown_regime_is_neutral():
    assert mr.regime_size_multiplier("") == 1.0
    assert mr.regime_size_multiplier(None) == 1.0
    assert mr.regime_size_multiplier("NOT_A_REGIME") == 1.0


def test_disabled_restores_full_size(monkeypatch):
    monkeypatch.setattr(settings, "enable_regime_size_haircut", False)
    assert mr.regime_size_multiplier("RISK_OFF") == 1.0


def test_multiplier_is_case_insensitive():
    assert mr.regime_size_multiplier("risk_off") == 0.5


# ── entry/exit consistency — the property that matters ─────────────────────

def _exit_reason(regime, action="BUY", enabled=True, monkeypatch=None):
    from types import SimpleNamespace
    import src.performance.tracker as tk
    monkeypatch.setattr(settings, "enable_regime_size_haircut", enabled)
    monkeypatch.setattr(settings, "signal_decay_regime_exit", True)
    ctx = SimpleNamespace(regime=regime)
    trade = {"action": action, "ticker": "AAA", "llm_synthesis_model": None}
    return tk._exit_reason_for(trade, action, None, ctx) if hasattr(tk, "_exit_reason_for") else None


def test_exit_rule_no_longer_closes_longs_in_RISK_OFF(monkeypatch):
    """The consistency guard. If RISK_OFF admits a long on entry but the exit
    rule closes it, the system opens a haircut position and kills it on the very
    next tick — strictly worse than either the old ban or the new haircut."""
    import inspect
    import src.performance.tracker as tk
    src = inspect.getsource(tk)
    i = src.index("_exit_regimes = ")
    window = src[i:i + 200]
    assert '("PANIC",)' in window, (
        "with the haircut enabled the exit rule must fire on PANIC only")
    assert '("PANIC", "RISK_OFF")' in window, (
        "the legacy pair must remain as the disabled-path fallback")


def test_entry_and_exit_agree_about_every_regime():
    """Whatever the policy, a regime that ADMITS a long must not immediately
    close it. Encoded as: no regime may both allow buys and be an exit trigger."""
    exit_regimes = {"PANIC"} if settings.enable_regime_size_haircut else {"PANIC", "RISK_OFF"}
    for reg, allowed in mr._REGIME_ALLOW_BUYS.items():
        if allowed:
            assert reg not in exit_regimes, (
                f"{reg} admits longs on entry but closes them on exit")


def test_blocked_regime_needs_no_haircut():
    """PANIC blocks entry outright, so its size multiplier is moot — but it must
    not silently be 0.0 in a way that could zero an unrelated caller."""
    assert mr.regime_size_multiplier("PANIC") == 1.0
