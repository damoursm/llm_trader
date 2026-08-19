"""Unfilled orders rest across ticks while the price stays near the decision
(2026-08-17).

The problem measured: tick-scoping put an order in the book for
`broker_settle_seconds` (30 s) out of a ~30-minute tick -- about 1.7% presence --
then cancelled it. Fill rate per intended trade was 19.3% overall, and the split
by session showed exactly what that costs:

    rth         56.6%   (ticks overlap continuous trading)
    afterhours  13.9%
    premarket   10.2%
    overnight    4.8%   (liquidity arrives sporadically; a 30 s window misses it)

and 78% of all attempts were submitted off-hours. So the blended rate was
dominated by orders that were never in the book when a counterparty appeared.

The fix keeps an unfilled order WORKING between ticks while the mark is within
`broker_rest_max_drift_bps` of that leg's own decision price. A resting order is
a capped LMT, so it can never fill worse than its own limit; what resting adds is
adverse selection, which the drift bound is there to limit.

These tests pin the guard rails, because a resting order that should have been
cancelled works the book invisibly.
"""

from datetime import datetime, timedelta, timezone

import pytest

from config import settings
from src.broker import reconcile


class _Order:
    def __init__(self, ref):
        self.client_ref = ref


class _Broker:
    """Records cancels so a test can assert an order was (or was not) killed.

    ``working`` is what the broker reports as still live. Resting requires a
    POSITIVE confirmation from here — an order the broker no longer lists is
    dead and must be resubmitted, never rested."""

    def __init__(self, working=("ref-1",), raises=False):
        self.cancelled = []
        self._working = list(working)
        self._raises = raises

    def get_open_orders(self):
        if self._raises:
            raise RuntimeError("broker unreadable")
        return [_Order(r) for r in self._working]

    def get_fills(self):
        return []

    def cancel_order(self, ref):
        self.cancelled.append(ref)
        return True


def _leg(prefix="broker_", status="OPEN", *, entry_price=100.0, exit_price=None,
         submitted_min_ago=45):
    sub = (datetime.now(timezone.utc) - timedelta(minutes=submitted_min_ago)).isoformat()
    t = {
        "ticker": "AAA", "action": "BUY", "status": status,
        "entry_price": entry_price,
        f"{prefix}order_id": "oid-1", f"{prefix}client_ref": "ref-1",
        f"{prefix}status": "Submitted", f"{prefix}submitted_at": sub,
        f"{prefix}fill_qty": 0,
    }
    if exit_price is not None:
        t["exit_price"] = exit_price
    return t


@pytest.fixture(autouse=True)
def _rest_on(monkeypatch):
    monkeypatch.setattr(settings, "broker_tick_scoped_orders", True)
    monkeypatch.setattr(settings, "broker_rest_unfilled_orders", True)
    monkeypatch.setattr(settings, "broker_rest_max_drift_bps", 60.0)
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 90)


def _sync_now():
    """A sync boundary AFTER the order was submitted, so it reads as tick-stale."""
    return datetime.now(timezone.utc)


def _run(broker, trades, monkeypatch, mark):
    monkeypatch.setattr(reconcile, "_live_price", lambda _t: mark)
    # `orders` is the event log every cancel appends to; the resting path never
    # reaches it, which is why only the cancelling tests need it present.
    report: dict = {"orders": []}
    reconcile._cancel_stale_unfilled(broker, trades, report,
                                     positions={}, sync_started=_sync_now())
    return report


# ── the core behaviour ──────────────────────────────────────────────────────

def test_order_rests_when_the_price_has_not_drifted(monkeypatch):
    """THE fix. A previous tick's order stays in the book instead of being
    cancelled — that presence is the whole point."""
    b = _Broker()
    rep = _run(b, [_leg()], monkeypatch, mark=100.3)      # 30 bp < 60 bp bound
    assert b.cancelled == [], "the order was killed despite a stable price"
    assert rep.get("orders_rested") == 1


def test_order_is_cancelled_once_the_price_drifts_too_far(monkeypatch):
    """The bound is what makes resting safe — beyond it we re-decide."""
    b = _Broker()
    rep = _run(b, [_leg()], monkeypatch, mark=101.0)      # 100 bp > 60 bp bound
    assert b.cancelled == ["ref-1"]
    assert not rep.get("orders_rested")


def test_drift_is_symmetric(monkeypatch):
    """A price that ran AWAY and one that came sharply toward us are both
    reasons to re-decide: the first makes the order useless, the second is the
    adverse-selection case resting exposes us to."""
    for mark in (101.0, 99.0):
        b = _Broker()
        _run(b, [_leg()], monkeypatch, mark=mark)
        assert b.cancelled == ["ref-1"], f"mark {mark} should have cancelled"


def test_the_age_ceiling_still_wins(monkeypatch):
    """broker_unfilled_cancel_minutes is an absolute bound — a stable price must
    not let an order rest forever."""
    b = _Broker()
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 30)
    _run(b, [_leg(submitted_min_ago=120)], monkeypatch, mark=100.0)
    assert b.cancelled == ["ref-1"], "an aged-out order rested past its ceiling"


def test_resting_can_be_switched_off(monkeypatch):
    """The strict one-tick lifetime must remain one setting away."""
    b = _Broker()
    monkeypatch.setattr(settings, "broker_rest_unfilled_orders", False)
    _run(b, [_leg()], monkeypatch, mark=100.0)
    assert b.cancelled == ["ref-1"]


# ── each leg is judged against its OWN decision price ───────────────────────

def test_exit_leg_is_judged_against_exit_price_not_entry_price(monkeypatch):
    """An exit's decision price is exit_price. Measuring its drift from
    entry_price would compare against a number that may be days and many
    percent away — re-anchoring every legitimate exit while resting stale ones."""
    b = _Broker()
    # entry far away (100), exit decided at 130, mark 130.2 → 15 bp from the
    # EXIT decision, but 3000 bp from entry.
    leg = _leg(prefix="broker_exit_", status="CLOSED", entry_price=100.0, exit_price=130.0)
    _run(b, [leg], monkeypatch, mark=130.2)
    assert b.cancelled == [], "the exit was judged against the entry price"


def test_exit_still_cancels_on_real_drift(monkeypatch):
    b = _Broker()
    leg = _leg(prefix="broker_exit_", status="CLOSED", entry_price=100.0, exit_price=130.0)
    _run(b, [leg], monkeypatch, mark=132.0)               # ~154 bp from 130
    assert b.cancelled == ["ref-1"]


# ── fail-safe: never rest on unusable inputs ────────────────────────────────

def test_no_live_price_falls_back_to_cancelling(monkeypatch):
    """Unable to measure drift ⇒ unable to justify resting. Cancel, which is
    the previous, known-safe behaviour."""
    b = _Broker()
    _run(b, [_leg()], monkeypatch, mark=None)
    assert b.cancelled == ["ref-1"]


def test_zero_bound_disables_resting(monkeypatch):
    b = _Broker()
    monkeypatch.setattr(settings, "broker_rest_max_drift_bps", 0.0)
    _run(b, [_leg()], monkeypatch, mark=100.0)
    assert b.cancelled == ["ref-1"]


def test_a_DEAD_order_is_never_rested(monkeypatch):
    """THE regression this fix nearly introduced. An off-RTH DAY limit that
    IBKR expired at the session close is gone from the book — the broker no
    longer lists it. Resting it parks the leg forever: it is never resubmitted,
    and for an EXIT the position that must flatten never does. Confirmed-working
    is therefore a precondition for resting, not an afterthought."""
    b = _Broker(working=())                      # broker lists nothing → dead
    _run(b, [_leg()], monkeypatch, mark=100.0)   # price perfectly stable
    assert b.cancelled == ["ref-1"], "a dead order was left resting"


def test_an_unreadable_broker_does_not_rest(monkeypatch):
    """Fail-safe direction. `_order_is_gone` fails OPEN (unknown reads as
    alive) which is right for cancelling and fatal for resting, so the resting
    path uses its own positive check that fails CLOSED."""
    b = _Broker(raises=True)
    _run(b, [_leg()], monkeypatch, mark=100.0)
    assert b.cancelled == ["ref-1"]


def test_partial_fills_are_never_touched(monkeypatch):
    """Pre-existing invariant: cancelling a remainder mid-fill strands a
    mismatched position. Resting must not change that."""
    b = _Broker()
    leg = _leg()
    leg["broker_fill_qty"] = 5
    _run(b, [leg], monkeypatch, mark=100.0)
    assert b.cancelled == []
