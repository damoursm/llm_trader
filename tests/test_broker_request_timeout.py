"""Broker request timeout — a stuck ib_async API request must not freeze the tick.

Guards the 2026-07-06 fix: the IBKR broker sets ``IB.RequestTimeout`` on its
connection so ``reqExecutions`` / ``reqPositions`` / ``placeOrder`` raise after a
bounded wait instead of hanging the whole scheduler forever.
"""

import sys
import types

from config.settings import settings


def _install_fake_ib_async(monkeypatch):
    """Inject a fake ``ib_async`` module so IBKRBroker._get_ib() constructs a
    stand-in IB (the real dependency isn't in the test interpreter)."""
    class FakeIB:
        def __init__(self):
            self.RequestTimeout = 0.0

        def isConnected(self):
            return False

    mod = types.ModuleType("ib_async")
    mod.IB = FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", mod)


def test_ibkr_sets_request_timeout(monkeypatch):
    monkeypatch.setattr(settings, "broker_request_timeout_seconds", 45.0)
    _install_fake_ib_async(monkeypatch)
    from src.broker.ibkr import IBKRBroker
    ib = IBKRBroker()._get_ib()
    assert ib.RequestTimeout == 45.0          # every request now bounded


def test_ibkr_request_timeout_zero_keeps_legacy(monkeypatch):
    # 0 = the old freeze-forever behaviour (opt-out), applied verbatim.
    monkeypatch.setattr(settings, "broker_request_timeout_seconds", 0.0)
    _install_fake_ib_async(monkeypatch)
    from src.broker.ibkr import IBKRBroker
    ib = IBKRBroker()._get_ib()
    assert ib.RequestTimeout == 0.0


def test_get_market_price_uses_short_timeout_and_restores(monkeypatch):
    """A quote snapshot must run under the SHORT price timeout (so a pre-market
    ticker fails in ~8s, not the full 45s that would march the reconcile — one
    call per open/drift position — into the sync watchdog) and then restore the
    global request bound."""
    monkeypatch.setattr(settings, "broker_request_timeout_seconds", 45.0)
    monkeypatch.setattr(settings, "broker_price_timeout_seconds", 8.0)

    seen = {}

    class FakeTicker:
        last = 12.5
        close = 12.0

        def marketPrice(self):
            return 12.5

    class FakeIB:
        def __init__(self):
            self.RequestTimeout = 45.0

        def isConnected(self):
            return True

        def reqTickers(self, contract):
            seen["timeout_during_call"] = self.RequestTimeout   # bound in effect now
            return [FakeTicker()]

    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker()
    b._ib = FakeIB()                                            # pretend connected
    monkeypatch.setattr(b, "_qualify", lambda t: object())

    px = b.get_market_price("HOOD")
    assert px == 12.5
    assert seen["timeout_during_call"] == 8.0                   # short bound during snapshot
    assert b._ib.RequestTimeout == 45.0                        # restored afterwards


def test_get_market_price_restores_timeout_on_failure(monkeypatch):
    """Even when the snapshot raises (the pre-market case), the global request
    timeout must be restored, not left at the short price bound."""
    monkeypatch.setattr(settings, "broker_request_timeout_seconds", 45.0)
    monkeypatch.setattr(settings, "broker_price_timeout_seconds", 8.0)

    class FakeIB:
        def __init__(self):
            self.RequestTimeout = 45.0

        def isConnected(self):
            return True

        def reqTickers(self, contract):
            raise TimeoutError("no data pre-market")

    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker()
    b._ib = FakeIB()
    monkeypatch.setattr(b, "_qualify", lambda t: object())

    assert b.get_market_price("HOOD") is None
    assert b._ib.RequestTimeout == 45.0                        # restored despite the raise
