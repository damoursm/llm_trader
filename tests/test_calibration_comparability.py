"""Calibrations may only fit data produced by the CURRENT code (2026-07-27).

Standing rule, stated by the user: *"every calibration should be done on the
same code."* Calibrating on values a superseded implementation produced fits a
relationship that no longer exists.

The panel side was already covered — `build_panel` masks superseded method
scores and (since today) confidence. But the LEDGER side was not: two
calibrations read trade `confidence` directly from the trade ledger, where
`build_panel` never runs. Measured before this gate, **75% of the closed trades
feeding both carried the pre-buy/sell-split confidence scale** (mean 0.873
versus 0.945 after) — so both were fitting across two different scales, and for
`calibrate_confidence_sizing` the 0.07 gap is roughly a whole band's width.

Note what the gate does NOT do: it does not try to CONVERT old values to the new
scale. A retrofit was rejected on evidence (see method_epochs.CONFIDENCE_EPOCH);
excluding is honest, converting would be invention. The Bayesian shrinkage in
both calibrations is what makes the smaller clean sample safe — they go inert on
thin evidence rather than over-fitting it.
"""

from datetime import date

import pytest

from config.settings import settings
import src.signals.method_epochs as me


@pytest.fixture(autouse=True)
def _epoch_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_confidence_epoch", True)


# ── the comparability predicate ────────────────────────────────────────────

def test_pre_epoch_confidence_is_not_comparable():
    assert me.confidence_is_comparable("2026-07-21") is False
    assert me.confidence_is_comparable("2026-08-13T23:59:00+00:00") is False


def test_epoch_day_onward_is_comparable():
    # epoch = 2026-08-14 02:00 UTC (the method RANK basis switch). Mid-day
    # convention: the WHOLE partial day is excluded, comparability starts 08-15.
    assert me.confidence_is_comparable("2026-08-14T03:00:00+00:00") is False
    assert me.confidence_is_comparable("2026-08-15") is True
    assert me.confidence_is_comparable("2026-08-20T10:00:00+00:00") is True


def test_it_fails_OPEN_on_bad_input():
    """A malformed or missing timestamp must never silently erase history —
    the same fail-open contract as `score_is_comparable`."""
    assert me.confidence_is_comparable(None) is True
    assert me.confidence_is_comparable("") is True
    assert me.confidence_is_comparable("not-a-date") is True


def test_disabling_the_epoch_makes_everything_comparable(monkeypatch):
    monkeypatch.setattr(settings, "enable_confidence_epoch", False)
    assert me.confidence_is_comparable("2020-01-01") is True


# ── the two ledger-side calibrations honour it ─────────────────────────────

def _trade(day, conf, ret, action="BUY"):
    # Win/loss is GROSS from the prices (system convention, 2026-08-06).
    sign = 1.0 if action == "BUY" else -1.0
    return {"status": "CLOSED", "action": action, "ticker": "AAA",
            "entry_date": day, "entry_datetime": f"{day}T14:00:00+00:00",
            "confidence": conf, "return_pct": ret,
            "entry_price": 100.0, "exit_price": 100.0 * (1 + sign * ret / 100.0)}


def test_confidence_sizing_excludes_pre_epoch_trades(monkeypatch):
    from src.performance import confidence_sizing as cs
    monkeypatch.setattr(settings, "enable_confidence_recal_sizing", True)
    monkeypatch.setattr(settings, "confidence_recal_min_trades", 1)
    old = [_trade("2026-07-01", 0.90, 5.0) for _ in range(50)]
    new = [_trade("2026-08-20", 0.90, -5.0) for _ in range(3)]
    cal = cs.calibrate_confidence_sizing(old + new)
    assert cal.get("n", 0) == 3, (
        f"expected only the 3 post-epoch trades, got n={cal.get('n')}")


def test_side_threshold_excludes_pre_epoch_trades(monkeypatch):
    import src.performance.tracker as tk
    seen = {}

    def spy(rows):
        seen["n"] = len(rows)
        return None
    monkeypatch.setattr(tk, "_spearman_conf_return", spy)
    monkeypatch.setattr(tk, "_load_trades", lambda: (
        [_trade("2026-07-01", 0.9, 1.0) for _ in range(40)]
        + [_trade("2026-08-20", 0.9, 1.0) for _ in range(7)]))
    tk._SIDE_THRESHOLD_CACHE.clear()
    tk.calibrate_side_threshold("BUY")
    assert seen["n"] == 7, f"expected 7 post-epoch rows, got {seen['n']}"


def test_gate_is_inert_when_the_epoch_is_off(monkeypatch):
    """Turning the epoch off must restore the previous behaviour exactly, so the
    change is reversible."""
    import src.performance.tracker as tk
    monkeypatch.setattr(settings, "enable_confidence_epoch", False)
    seen = {}
    monkeypatch.setattr(tk, "_spearman_conf_return", lambda rows: seen.setdefault("n", len(rows)))
    monkeypatch.setattr(tk, "_load_trades", lambda: (
        [_trade("2026-07-01", 0.9, 1.0) for _ in range(40)]
        + [_trade("2026-07-25", 0.9, 1.0) for _ in range(7)]))
    tk._SIDE_THRESHOLD_CACHE.clear()
    tk.calibrate_side_threshold("BUY")
    assert seen["n"] == 47


def test_method_scores_have_the_same_guarantee():
    """The method-score half of the rule was already in place — assert it so the
    two halves cannot drift apart."""
    from src.signals.method_epochs import score_is_comparable, METHOD_SCORER_EPOCH
    assert "money_flow" in METHOD_SCORER_EPOCH
    assert score_is_comparable("money_flow", "2026-07-01") is False
    assert score_is_comparable("money_flow", "2026-07-26") is True
    # vwap gained an epoch 2026-08-11 (window 20 -> 5): its old rows are now
    # correctly NON-comparable, and a method with no epoch stays always-true.
    assert score_is_comparable("vwap", "2020-01-01") is False
    assert score_is_comparable("vwap", "2026-08-12") is True
    assert score_is_comparable("tech", "2020-01-01") is True
