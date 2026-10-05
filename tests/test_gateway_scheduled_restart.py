"""The gateway's scheduled daily restart (IBC AutoRestartTime, 11:50 PM ET) — user,
2026-09-29: "Can you solve the broken connection issue? The production needs to be
reliable."

What went wrong: at 23:50 IBC restarts the gateway, closing the API socket. The
client runs no event loop between broker calls, so `isConnected()` kept reading True
and the next sync (23:54) sent real requests on the dead socket — two ERROR lines
(the socket reset, then "cancelPnL: No subscription") before the auto-reconnect.
Worse, a sync DURING the restart minute would exhaust its connect retries and fire
the gateway recovery, which kills the gateway IBC is restarting.

What must hold: the restart time is read from IBC's config (only that line); inside
the window the recovery stands down and the client WAITS for the gateway; an idle
session the gateway closed is probed and redialed quietly before any work; a
dropped P&L subscription is never cancelled; a closed socket logs a WARNING (INFO
while probing), not an ERROR. No network, no processes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.broker import gateway_recovery as gr

ET = ZoneInfo("America/New_York")


@pytest.fixture
def ibc_config(tmp_path, monkeypatch):
    p = tmp_path / "config.ini"
    p.write_text("IbLoginId=someone\nIbPassword=secret\n# AutoRestartTime=01:00 AM (a comment)\n"
                 "AutoRestartTime=11:50 PM\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ibc_config_path", str(p))
    monkeypatch.setattr(settings, "broker_gateway_restart_et", "")
    gr._RESTART_CACHE.clear()
    yield p
    gr._RESTART_CACHE.clear()


def at(h, m, s=0, day=29):
    return datetime(2026, 9, day, h, m, s, tzinfo=ET)


def test_the_restart_time_is_read_from_ibcs_config(ibc_config, monkeypatch):
    assert gr.scheduled_restart_et() == (23, 50)
    ibc_config.write_text("AutoRestartTime=\n", encoding="utf-8")       # IBC's restart switched off
    gr._RESTART_CACHE.clear()
    assert gr.scheduled_restart_et() is None
    monkeypatch.setattr(settings, "broker_gateway_restart_et", "23:05")  # the override wins
    assert gr.scheduled_restart_et() == (23, 5)
    monkeypatch.setattr(settings, "broker_gateway_restart_et", "")
    monkeypatch.setattr(settings, "ibc_config_path", str(ibc_config.parent / "missing.ini"))
    assert gr.scheduled_restart_et() is None


def test_the_window_runs_from_a_minute_before_to_the_window_after(ibc_config):
    assert gr.in_scheduled_restart_window(at(23, 48, 59)) is None
    assert gr.in_scheduled_restart_window(at(23, 49)) == at(23, 55)
    assert gr.in_scheduled_restart_window(at(23, 52)) == at(23, 55)
    assert gr.in_scheduled_restart_window(at(23, 55)) == at(23, 55)
    assert gr.in_scheduled_restart_window(at(23, 55, 1)) is None
    assert gr.in_scheduled_restart_window(at(12, 0)) is None


def test_a_window_can_cross_midnight(monkeypatch):
    monkeypatch.setattr(settings, "broker_gateway_restart_et", "23:58")
    assert gr.in_scheduled_restart_window(at(0, 2, day=30)) == at(0, 3, day=30)
    assert gr.in_scheduled_restart_window(at(0, 4, day=30)) is None


def test_the_recovery_never_kills_the_gateway_ibc_is_restarting(monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(gr.subprocess, "run", lambda *a, **k: calls.append(a[0]) or Done())
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", True)
    monkeypatch.setattr(gr, "_pid_listening_on", lambda port: 4242)
    gr._reset_for_tests()
    monkeypatch.setattr(gr, "in_scheduled_restart_window",
                        lambda now=None: datetime.now(timezone.utc) + timedelta(minutes=3))
    assert gr.maybe_restart_gateway("sync connect retries exhausted", wait=False) is False
    assert calls == []                                            # nothing killed, no second gateway
    monkeypatch.setattr(gr, "in_scheduled_restart_window", lambda now=None: None)
    assert gr.maybe_restart_gateway("sync connect retries exhausted", wait=False) is True
    assert calls and calls[0][0] == "taskkill"                    # outside the window it still recovers


class FakeIB:
    """ib_async's IB, as far as the client's connection handling uses it."""

    def __init__(self, alive=True, dial_fails=0, connected=True):
        self.connected, self.alive, self.dial_fails = connected, alive, dial_fails
        self.dials, self.cancelled, self.RequestTimeout = 0, [], 45.0

    def isConnected(self):
        return self.connected

    def reqCurrentTime(self):
        if not self.alive:
            self.connected = False
            raise ConnectionResetError("[WinError 10053] An established connection was aborted")
        return datetime.now(timezone.utc)

    def disconnect(self):
        self.connected = False

    def connect(self, *a, **k):
        self.dials += 1
        if self.dials <= self.dial_fails:
            raise ConnectionRefusedError("gateway not listening")
        self.connected, self.alive = True, True

    def reqAllOpenOrders(self):
        pass

    def managedAccounts(self):
        return ["DU1"]

    def sleep(self, s):
        pass

    def reqPnL(self, acct):
        if not self.alive:                        # the gateway closes the socket mid-request
            self.connected = False
            raise ConnectionResetError("[WinError 10054] An existing connection was forcibly closed")

        class P:
            dailyPnL, unrealizedPnL, realizedPnL = 1.0, 2.0, 3.0
        return P()

    def cancelPnL(self, acct):
        self.cancelled.append(acct)


def _broker(ib):
    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker()
    b._ib = ib
    b.account = "DU1"
    return b


def test_an_idle_session_the_gateway_closed_is_redialed_before_any_work():
    dead = FakeIB(alive=False)
    b = _broker(dead)
    assert b.connect() is True and dead.dials == 1 and dead.RequestTimeout == 45.0
    live = FakeIB(alive=True)
    assert _broker(live).connect() is True and live.dials == 0      # a live session is kept


def test_during_ibcs_restart_the_client_waits_for_the_gateway(monkeypatch):
    waited = []
    monkeypatch.setattr(gr, "in_scheduled_restart_window",
                        lambda now=None: datetime.now(timezone.utc) + timedelta(minutes=4))
    monkeypatch.setattr(gr, "wait_for_gateway", lambda until, **k: waited.append(until) or True)
    ib = FakeIB(connected=False, dial_fails=1)
    assert _broker(ib).connect() is True and len(waited) == 1 and ib.dials == 2
    monkeypatch.setattr(gr, "in_scheduled_restart_window", lambda now=None: None)
    ib2 = FakeIB(connected=False, dial_fails=1)
    assert _broker(ib2).connect() is False and ib2.dials == 1       # outside it: no wait


def test_a_dropped_pnl_subscription_is_never_cancelled():
    ib = FakeIB()
    b = _broker(ib)
    assert b.get_pnl().daily == 1.0 and ib.cancelled == ["DU1"]      # the normal path cancels
    ib2 = FakeIB()
    b2 = _broker(ib2)
    ib2.alive = False                                                # the socket dies under reqPnL
    assert b2.get_pnl() is None and ib2.cancelled == []


def test_a_closed_socket_is_a_warning_and_quiet_while_probing():
    from loguru import logger
    import src.broker.ibkr as k
    seen = []
    sid = logger.add(lambda m: seen.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    try:
        lg = logging.getLogger("ib_async.client")
        lg.error("[WinError 10053] An established connection was aborted by the software in your host machine")
        k._QUIET["probe"] = True
        lg.error("[WinError 10054] An existing connection was forcibly closed by the remote host")
        k._QUIET["probe"] = False
        lg.error("Error 201, reqId 5: Order rejected - reason: not shortable")
    finally:
        k._QUIET["probe"] = False
        logger.remove(sid)
    got = [lvl for lvl, msg in seen if msg.startswith("[ib_async]")]
    assert got == ["WARNING", "INFO", "ERROR"]                       # a real reject stays an ERROR
