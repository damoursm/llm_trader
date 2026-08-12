"""IBKRBroker auto-reconnect (2026-07-11).

A Gateway session that DROPS (restart, network blip, daily re-login) self-heals
at the next broker touchpoint: every public method gates on _ensure_connected,
which redials a disconnected session before failing soft. A failed dial arms a
cooldown (broker_reconnect_cooldown_seconds) that fast-fails further IMPLICIT
revives — so a down/wedged gateway can't charge every price fetch / sync step
the full connect timeout — while EXPLICIT connect() calls (the reconciler's
sync-start retry loop) bypass it, and any successful dial clears it.
"""

from config.settings import settings
from src.broker.base import OrderRequest
from src.broker.ibkr import IBKRBroker


class FakeIB:
    """Stands in for ib_async.IB — injected as broker._ib so _get_ib never
    imports the real dependency."""

    def __init__(self):
        self.connected = False
        self.dials = 0
        self.refuse = False          # True → connect() raises (gateway down)
        self.disconnect_calls = 0

    def isConnected(self):
        return self.connected

    def connect(self, host, port, clientId=None, timeout=None, readonly=False):
        self.dials += 1
        if self.refuse:
            raise ConnectionRefusedError("gateway down")
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    # minimal read surface for the methods exercised below
    def openTrades(self):
        return []

    def reqPositions(self):
        return None

    def positions(self, account=""):
        return []


def _broker(fake: FakeIB) -> IBKRBroker:
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU000000")
    b._ib = fake
    return b


def test_dropped_session_reconnects_on_next_call():
    fake = FakeIB()
    b = _broker(fake)
    assert not b.is_connected()
    assert b.get_open_orders() == []          # touchpoint → automatic redial
    assert fake.dials == 1
    assert b.is_connected()
    assert b.get_open_orders() == []          # already live → no further dial
    assert fake.dials == 1


def test_failed_dial_arms_cooldown_for_implicit_revives(monkeypatch):
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 300.0)
    fake = FakeIB()
    fake.refuse = True
    b = _broker(fake)
    assert b.get_positions() == []            # dial fails → fail-soft
    assert fake.dials == 1
    assert b.get_positions() == []            # inside cooldown → no re-dial
    assert b.get_market_price("AAPL") is None
    assert b.get_fills() == []
    assert fake.dials == 1


def test_explicit_connect_bypasses_cooldown(monkeypatch):
    # The reconciler's sync-start retry loop calls connect() directly and must
    # keep dialing on its own cadence regardless of the implicit throttle.
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 300.0)
    fake = FakeIB()
    fake.refuse = True
    b = _broker(fake)
    assert b.get_open_orders() == []          # arms the cooldown
    assert fake.dials == 1
    assert b.connect() is False
    assert fake.dials == 2


def test_successful_dial_clears_cooldown(monkeypatch):
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 300.0)
    fake = FakeIB()
    fake.refuse = True
    b = _broker(fake)
    assert b.get_open_orders() == []          # failed dial → cooldown armed
    fake.refuse = False
    assert b.connect() is True                # explicit dial succeeds → cleared
    fake.connected = False                    # gateway drops again later
    assert b.get_open_orders() == []          # implicit revive redials immediately
    assert fake.dials == 3
    assert b.is_connected()


def test_zero_cooldown_disables_the_throttle(monkeypatch):
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 0.0)
    fake = FakeIB()
    fake.refuse = True
    b = _broker(fake)
    assert b.get_positions() == []
    assert b.get_positions() == []
    assert fake.dials == 2                    # every touchpoint dials (legacy-ish)


def test_submit_order_reports_disconnected_when_revive_fails(monkeypatch):
    # The status string is what reconcile._is_transient_failure keys off — the
    # submit-retry path then explicitly reconnects and resubmits.
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 300.0)
    fake = FakeIB()
    fake.refuse = True
    b = _broker(fake)
    res = b.submit_order(OrderRequest(ticker="AAPL", side="BUY", quantity=1,
                                      order_type="MKT", client_ref="t-1"))
    assert res.ok is False
    assert res.status == "DISCONNECTED"


def test_deliberate_disconnect_stays_silent_and_reconnects_cleanly():
    fake = FakeIB()
    b = _broker(fake)
    assert b.connect() is True
    b.disconnect()                            # deliberate → no drop warning path
    assert not b.is_connected()
    assert b.get_open_orders() == []          # next touchpoint revives
    assert b.is_connected()


# ── repeated-wedge escalation (2026-08-04) ───────────────────────────────────
# Live failure this reproduces: a wedged gateway ACCEPTED every reconnect and
# then timed out every request, so the forced redial SUCCEEDED ~14 times in a row
# and the `if not ok and wedged` recovery branch never fired. The loop ran
# 07:39→08:00 with auto-restart enabled and did nothing.

def _wedge(b, fake):
    """Put the broker in the alive-but-wedged state: socket reads connected, but
    the request timeout counter is over the threshold."""
    fake.connected = True
    b._consecutive_timeouts = int(settings.broker_wedge_timeout_threshold) + 1


def test_repeated_wedge_fires_recovery_even_when_the_redial_SUCCEEDS(monkeypatch):
    import src.broker.gateway_recovery as gr
    calls = []
    monkeypatch.setattr(gr, "maybe_restart_gateway",
                        lambda reason, wait=True: calls.append(reason) or True)
    monkeypatch.setattr(settings, "broker_wedge_recycle_limit", 3)
    monkeypatch.setattr(settings, "broker_wedge_recycle_window_seconds", 900.0)
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 0.0)
    fake = FakeIB()
    b = _broker(fake)
    for i in range(2):                       # below the limit → no escalation yet
        _wedge(b, fake)
        assert b._ensure_connected() is True  # the dial SUCCEEDS every time
        assert calls == [], f"escalated too early on recycle {i + 1}"
    _wedge(b, fake)                          # third force-recycle inside the window
    assert b._ensure_connected() is True
    assert len(calls) == 1
    assert "wedged-but-alive" in calls[0]


def test_successful_request_clears_the_wedge_history(monkeypatch):
    # Only a real response proves the gateway is back, so it resets the counter —
    # otherwise recycles from a since-resolved episode accumulate toward a
    # spurious restart hours later.
    import src.broker.gateway_recovery as gr
    calls = []
    monkeypatch.setattr(gr, "maybe_restart_gateway",
                        lambda reason, wait=True: calls.append(reason) or True)
    monkeypatch.setattr(settings, "broker_wedge_recycle_limit", 3)
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 0.0)
    fake = FakeIB()
    b = _broker(fake)
    for _ in range(2):
        _wedge(b, fake)
        b._ensure_connected()
    b._note_request(None)                    # a request actually SUCCEEDED
    assert b._wedge_recycle_times == []
    _wedge(b, fake)                          # would have been the 3rd without the reset
    b._ensure_connected()
    assert calls == []


def test_failed_redial_still_fires_recovery_immediately(monkeypatch):
    # The original path must keep working: a wedge whose forced redial FAILS is a
    # dead gateway and escalates on the first occurrence, with no counting.
    import src.broker.gateway_recovery as gr
    calls = []
    monkeypatch.setattr(gr, "maybe_restart_gateway",
                        lambda reason, wait=True: calls.append(reason) or True)
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 0.0)
    fake = FakeIB()
    b = _broker(fake)
    _wedge(b, fake)
    fake.refuse = True
    assert b._ensure_connected() is False
    assert len(calls) == 1 and "forced redial failed" in calls[0]
