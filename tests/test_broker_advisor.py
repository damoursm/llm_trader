"""Broker-advisor method (IBKR short-borrow squeeze tilt).

Locks the scoring convention (+ = hard/expensive to short → bullish, fades a SELL;
0 = easy borrow / no data), the fail-soft fetch (off / no gateway → {}), and the
trade-attribution wiring.
"""

import pytest

from config.settings import settings
from src.broker.base import BorrowInfo
from src.signals.broker_advisor import compute_broker_advisor_score, fetch_borrow_context


@pytest.fixture(autouse=True)
def _fixed_params(monkeypatch):
    monkeypatch.setattr(settings, "broker_advisor_max_score", 0.6)
    monkeypatch.setattr(settings, "broker_advisor_expensive_fee_pct", 10.0)
    monkeypatch.setattr(settings, "broker_advisor_hard_shares", 200000.0)


# ── scoring: sign + magnitude convention ──────────────────────────────────

def test_no_data_is_no_view():
    assert compute_broker_advisor_score(None) == 0.0
    assert compute_broker_advisor_score(BorrowInfo(ticker="X")) == 0.0


def test_expensive_fee_is_bullish_capped():
    # Fee path: expensive borrow → positive (bullish squeeze tilt), monotone, capped.
    cheap = compute_broker_advisor_score(BorrowInfo("X", fee_pct=0.5))
    mid = compute_broker_advisor_score(BorrowInfo("X", fee_pct=10.0))   # = expensive_fee_pct
    pricey = compute_broker_advisor_score(BorrowInfo("X", fee_pct=50.0))
    assert 0 < cheap < mid < pricey <= 0.6
    assert mid == pytest.approx(0.6 * 0.7616, abs=1e-3)                 # cap·tanh(1)


def test_scarce_shares_is_bullish_ample_is_no_view():
    # Availability path: 0 shares → full tilt; at the hard line → 0; ample → 0.
    assert compute_broker_advisor_score(BorrowInfo("X", shortable_shares=0.0)) == pytest.approx(0.6)
    assert compute_broker_advisor_score(BorrowInfo("X", shortable_shares=100000.0)) == pytest.approx(0.3)
    assert compute_broker_advisor_score(BorrowInfo("X", shortable_shares=200000.0)) == 0.0
    assert compute_broker_advisor_score(BorrowInfo("X", shortable_shares=5_000_000.0)) == 0.0


def test_fee_path_preferred_over_availability():
    info = BorrowInfo("X", shortable_shares=0.0, fee_pct=10.0)   # both present
    assert compute_broker_advisor_score(info) == pytest.approx(0.6 * 0.7616, abs=1e-3)  # fee wins


def test_not_shortable_flag_is_full_tilt():
    assert compute_broker_advisor_score(BorrowInfo("X", is_shortable=False)) == pytest.approx(0.6)
    assert compute_broker_advisor_score(BorrowInfo("X", is_shortable=True)) == 0.0


def test_score_is_never_negative():
    # The method is a one-sided squeeze tell: bullish or no-view, never bearish.
    for info in (BorrowInfo("X", fee_pct=99.0), BorrowInfo("X", shortable_shares=1.0),
                 BorrowInfo("X", is_shortable=False)):
        assert compute_broker_advisor_score(info) >= 0.0


# ── fetch: gated + fail-soft ───────────────────────────────────────────────

def test_fetch_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "enable_broker_advisor", False)
    assert fetch_borrow_context(["GME"]) == {}


def test_fetch_broker_off_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "enable_broker_advisor", True)
    monkeypatch.setattr(settings, "broker_mode", "off")
    assert fetch_borrow_context(["GME"]) == {}


def test_fetch_uses_broker_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_broker_advisor", True)
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")

    class FakeBroker:
        def is_connected(self): return True
        def connect(self): return True
        def get_short_borrow(self, tickers):
            return {t: BorrowInfo(t, shortable_shares=0.0) for t in tickers}

    out = fetch_borrow_context(["GME", "AMC"], broker=FakeBroker())
    assert set(out) == {"GME", "AMC"}
    assert out["GME"].shortable_shares == 0.0


def test_fetch_is_fail_soft_on_broker_error(monkeypatch):
    monkeypatch.setattr(settings, "enable_broker_advisor", True)
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")

    class BoomBroker:
        def is_connected(self): return True
        def get_short_borrow(self, tickers): raise RuntimeError("gateway down")

    assert fetch_borrow_context(["GME"], broker=BoomBroker()) == {}    # swallowed → {}


# ── integration: registered + trade-attributed ─────────────────────────────

def test_broker_advisor_registered_and_attributed():
    from src.performance.tracker import _ALL_METHODS, _method_scores_from_signal, METHOD_LABELS
    from src.signals.aggregator import _BASE_WEIGHTS
    from src.models import TickerSignal
    assert "broker_advisor" in _ALL_METHODS
    assert "broker_advisor" in _BASE_WEIGHTS
    assert "broker_advisor" in METHOD_LABELS
    sig = TickerSignal(ticker="GME", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="t")
    sig.broker_advisor_score = 0.45
    scores = _method_scores_from_signal("GME", "BULLISH", {"GME": sig})
    assert scores["broker_advisor"] == 0.45
