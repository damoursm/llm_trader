"""IBKRBroker.what_if_short (user directive 2026-10-05: "build the what-if"): IBKR's own margin for a
hypothetical short, priced by IBKR and never transmitted.

What must hold: the order is a DAY limit SELL (ib_async 2.1 fails a what-if on IBKR's 10349 "TIF set
to DAY by preset" message — the first margin probe came back empty for that reason); IBKR's margin
changes, in the account's base currency, become USD rates of the order's value; IBKR's eligibility
refusal (error 201 "No Trading Permission ... Ineligibility reasons") is reported as ``refused`` with
its reason; an unset, failed or otherwise refused answer carries no rates; the request runs under the
what-if's own timeout and the session's timeout is restored; no session = not asked.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.settings import settings
from src.broker.ibkr import IBKRBroker

REFUSAL = ("Order rejected - reason:No Trading Permission, Customer Ineligible; Ineligibility "
           "reasons:<br>No Opening Trades: Small Cap, Subject to Compliance Restriction")


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, h):
        self.handlers.append(h)
        return self

    def __isub__(self, h):
        self.handlers.remove(h)
        return self

    def emit(self, *args):
        for h in list(self.handlers):
            h(*args)


class _FakeIB:
    """The slice of ib_async.IB the what-if touches; ``errors`` are emitted for the order's
    contract before the answer, as IBKR's messages arrive before the request ends."""

    def __init__(self, answer=None, errors=(), raises=None):
        self.answer, self.errors, self.raises = answer, list(errors), raises
        self.errorEvent = _Event()
        self.RequestTimeout = 45.0
        self.orders, self.timeouts = [], []

    def isConnected(self):
        return True

    def qualifyContracts(self, contract):
        return [contract]

    def accountValues(self, account=""):
        return [SimpleNamespace(tag="NetLiquidation", value="991000", currency="CAD")]

    def managedAccounts(self):
        return ["DU1"]

    def sleep(self, seconds):
        return None

    def whatIfOrder(self, contract, order):
        self.orders.append(order)
        self.timeouts.append(self.RequestTimeout)
        if self.raises:
            raise self.raises
        for code, text in self.errors:
            self.errorEvent.emit(7, code, text, contract)
        return self.answer


def _broker(monkeypatch, ib):
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU1")
    b._ib = ib
    monkeypatch.setattr("src.broker.fx.usd_per_unit", lambda ccy: 0.7 if ccy == "CAD" else 1.0)
    monkeypatch.setattr(settings, "sel_short_whatif_timeout_seconds", 10.0)
    return b


def _state(init="1430.0", maint="1000.0", warning=""):
    return SimpleNamespace(initMarginChange=init, maintMarginChange=maint, warningText=warning)


def test_rates_are_usd_multiples_of_the_orders_value_and_the_order_is_a_day_limit_sell(monkeypatch):
    ib = _FakeIB(_state(warning="There is insufficient ABC available for short sale."),
                 errors=[(10349, "Order TIF was set to DAY based on order preset.")])
    out = _broker(monkeypatch, ib).what_if_short("ABC", 50, 10.0)
    # CAD 1,430 / 1,000 on a $500 short at 0.70 USD per CAD
    assert out["init_rate"] == pytest.approx(2.002) and out["maint_rate"] == pytest.approx(1.4)
    assert out["currency"] == "CAD" and out["qty"] == 50 and out["price"] == 10.0
    assert "insufficient" in out["warning"] and "error" not in out and "refused" not in out   # 10349 ignored
    o = ib.orders[0]
    assert (o.action, o.totalQuantity, o.lmtPrice, o.tif) == ("SELL", 50, 10.0, "DAY")
    assert ib.timeouts == [10.0] and ib.RequestTimeout == 45.0                 # its own bound, then restored
    assert ib.errorEvent.handlers == []                                        # the listener is removed


def test_ibkrs_refusal_of_any_opening_short_is_reported_with_its_reason(monkeypatch):
    out = _broker(monkeypatch, _FakeIB([], errors=[(201, REFUSAL)])).what_if_short("ABC", 50, 10.0)
    assert out["refused"] == "No Opening Trades: Small Cap, Subject to Compliance Restriction"
    assert "init_rate" not in out
    close_only = ("Order rejected - reason:No Trading Permission, Customer Ineligible; Ineligibility reasons:"
                  "<br>Margin concern/risk management: For risk management purposes, this product is in close-only")
    out = _broker(monkeypatch, _FakeIB([], errors=[(201, close_only)])).what_if_short("ABC", 50, 10.0)
    assert out["refused"].startswith("Margin concern/risk management")


def test_no_usable_answer_carries_no_rates(monkeypatch):
    other = _broker(monkeypatch, _FakeIB([], errors=[(201, "Order rejected - reason:price out of range")]))
    out = other.what_if_short("ABC", 50, 10.0)
    assert "refused" not in out and "init_rate" not in out and "price out of range" in out["error"]
    unset = _broker(monkeypatch, _FakeIB(_state("1.7976931348623157E308", "1.7976931348623157E308")))
    out = unset.what_if_short("ABC", 50, 10.0)
    assert "init_rate" not in out and out["error"] == "no margin in IBKR's answer"
    ib = _FakeIB(raises=TimeoutError("what-if timed out"))
    out = _broker(monkeypatch, ib).what_if_short("ABC", 50, 10.0)
    assert "init_rate" not in out and "timed out" in out["error"] and ib.RequestTimeout == 45.0


def test_no_session_or_no_order_means_not_asked(monkeypatch):
    b = _broker(monkeypatch, _FakeIB(_state()))
    assert b.what_if_short("ABC", 0, 10.0) is None
    assert b.what_if_short("ABC", 50, 0.0) is None
    assert b.what_if_short("ABC", 50, float("nan")) is None
    monkeypatch.setattr(b, "_ensure_connected", lambda: False)
    assert b.what_if_short("ABC", 50, 10.0) is None


def test_a_refusal_during_a_what_if_logs_as_info_not_as_a_broker_error(monkeypatch):
    """ib_async logs IBKR's 201 through the bridge: during a what-if the refusal is the answer (the entry
    step logs the skipped pick), so it must not read as a broker fault; outside one it stays an ERROR."""
    import logging

    from loguru import logger

    from src.broker import ibkr as ib_mod
    seen = []
    sink = logger.add(lambda m: seen.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    try:
        lg = logging.getLogger("ib_async.wrapper")
        ib_mod._QUIET["whatif"] = True
        lg.error("Error 201, reqId 6: " + REFUSAL)
        ib_mod._QUIET["whatif"] = False
        lg.error("Error 201, reqId 7: " + REFUSAL)
    finally:
        ib_mod._QUIET["whatif"] = False
        logger.remove(sink)
    levels = [lvl for lvl, msg in seen if "Error 201" in msg]
    assert levels == ["INFO", "ERROR"]
    # and the flag is down after a what-if, whatever its outcome
    _broker(monkeypatch, _FakeIB([], errors=[(201, REFUSAL)])).what_if_short("ABC", 50, 10.0)
    assert ib_mod._QUIET["whatif"] is False
