"""Whole-share enforcement at the broker boundary (2026-08-04).

IBKR refuses fractional equity over the API — verified with a whatIfOrder dry
run: "Error 10243: Fractional-sized order cannot be placed via API. Please use
desktop version to place this order."

`OrderRequest.quantity` is ANNOTATED `int`, but OrderRequest is a plain
@dataclass, so that annotation is documentation, not a runtime check — a float
would sail through to LimitOrder() and be rejected by the broker. Every caller
happens to floor today (sizing._round_shares, the reconciler's int(abs(...))) and
0 of 4,848 submitted orders were ever fractional, but that is an EMERGENT
invariant. submit_order is the one choke point that can ENFORCE it.
"""

from __future__ import annotations

import pytest

from src.broker.base import OrderRequest
from src.broker.ibkr import IBKRBroker, _whole_shares


@pytest.mark.parametrize("raw,expected", [
    (24.5, 24), (0.5, 0), (3, 3), (3.0, 3), (0.999, 0),
    (-2.7, 0),            # never flips a sign: side carries direction
    ("x", 0), (None, 0), (0, 0),
])
def test_whole_shares_floors_and_never_goes_negative(raw, expected):
    assert _whole_shares(raw) == expected


def _req(qty):
    return OrderRequest(ticker="ADIG", side="SELL", quantity=qty,
                        order_type="LMT", limit_price=22.5, client_ref="t-1")


def test_sub_share_quantity_is_refused_before_touching_the_broker(monkeypatch):
    """A 0.5-share residue must not reach IBKR at all — it can only come back as
    error 10243. It is refused with its own status, not a generic failure."""
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU1")
    called = {"connect": False}
    monkeypatch.setattr(b, "_ensure_connected",
                        lambda: called.__setitem__("connect", True) or True)
    res = b.submit_order(_req(0.5))
    assert res.ok is False
    assert res.status == "SUB_SHARE_QTY"
    assert called["connect"] is False, "should refuse before dialing the broker"


def test_fractional_quantity_is_floored_not_rejected(monkeypatch):
    """24.5 shares is 24 tradeable shares plus an untradeable remainder — submit
    the 24 rather than losing the whole order."""
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU1")
    seen = {}
    monkeypatch.setattr(b, "_ensure_connected", lambda: True)

    def fake_place(contract, order):
        seen["qty"] = order.totalQuantity
        raise RuntimeError("stop here — quantity already captured")
    monkeypatch.setattr(b, "_qualify", lambda t, overnight=False: object())

    class _IB:
        def placeOrder(self, c, o):
            return fake_place(c, o)
    b._ib = _IB()
    b.submit_order(_req(24.5))
    assert seen["qty"] == 24, f"submitted {seen.get('qty')} instead of flooring to 24"


def test_whole_share_orders_pass_through_unchanged(monkeypatch):
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU1")
    seen = {}
    monkeypatch.setattr(b, "_ensure_connected", lambda: True)
    monkeypatch.setattr(b, "_qualify", lambda t, overnight=False: object())

    class _IB:
        def placeOrder(self, c, o):
            seen["qty"] = o.totalQuantity
            raise RuntimeError("stop")
    b._ib = _IB()
    b.submit_order(_req(7))
    assert seen["qty"] == 7
