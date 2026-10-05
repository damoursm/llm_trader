"""The email alert for a gateway logged OUT of IBKR (user directive 2026-09-28:
"You can add the alert in the email digest").

2026-09-28 10:58: a manual login to the paper account elsewhere logged the IB
Gateway out ("Existing session detected" → re-login refused, "Unrecognized
Username or Password"). The gateway kept its API port open, so every connect
succeeded and the syncs read healthy for 68 minutes. What must hold: IBC's last
login outcome is read; IBKR's own connectivity codes (1100 lost / 1101-1102
restored) are tracked and reset on a fresh session; either one makes the sync
report carry `gateway_session`, and the broker health verdict goes DOWN with a
message the email banner shows. No network.
"""
from __future__ import annotations

from datetime import datetime, timezone

from config.settings import settings
from src.broker import gateway_recovery as gr


def _ibc_log(tmp_path, lines):
    p = tmp_path / "IBC-3.23.0_GATEWAY-1047_.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_ibc_login_state_reads_the_last_login_outcome(tmp_path):
    ok = "2026-09-28 10:30:13:469 IBC: Login has completed"
    refused = ("2026-09-28 10:59:01:944 IBC: detected dialog entitled: Unrecognized Username or "
               "Password; event=Opened")
    closed = ("2026-09-28 10:59:05:000 IBC: detected dialog entitled: Unrecognized Username or "
              "Password; event=Closed")
    st = gr.ibc_login_state(str(_ibc_log(tmp_path, [ok, "noise", refused, closed])))
    assert st == {"logged_in": False, "at": "2026-09-28 10:59:01",
                  "reason": "login refused: Unrecognized Username or Password", "existing_session": False}
    later = "2026-09-28 12:07:54:000 IBC: Login has completed"
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, [refused, later])))["logged_in"] is True
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, ["nothing here"]))) is None
    assert gr.ibc_login_state(str(tmp_path / "missing")) is None


def test_the_broker_tracks_ibkr_connectivity_and_a_fresh_session_resets_it(monkeypatch):
    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker()
    assert b.ibkr_link_lost_since() is None
    b._on_ib_error(-1, 1100, "Connectivity between IBKR and Trader Workstation has been lost.")
    first = b.ibkr_link_lost_since()
    assert first is not None
    b._on_ib_error(-1, 1100, "again")
    assert b.ibkr_link_lost_since() == first                       # the FIRST loss is kept
    b._on_ib_error(-1, 2104, "market data farm OK")                # unrelated codes change nothing
    assert b.ibkr_link_lost_since() == first
    b._on_ib_error(-1, 1102, "Connectivity restored - data maintained")
    assert b.ibkr_link_lost_since() is None

    class FakeIB:
        def isConnected(self):
            return False

        def disconnect(self):
            pass

        def connect(self, *a, **k):
            pass

        def reqAllOpenOrders(self):
            pass
    b._ib = FakeIB()
    b._link_lost_at = datetime.now(timezone.utc)
    assert b.connect() is True and b.ibkr_link_lost_since() is None


def test_the_sync_report_carries_a_logged_out_gateway(monkeypatch):
    from src.broker import reconcile

    class B:
        def ibkr_link_lost_since(self):
            return None
    monkeypatch.setattr(gr, "ibc_login_state", lambda: {"logged_in": False, "at": "2026-09-28 10:59:01",
                                                         "reason": "login refused: Unrecognized Username or Password"})
    assert reconcile._gateway_session(B()) == {"since": "2026-09-28 10:59:01",
                                               "reason": "login refused: Unrecognized Username or Password",
                                               "auto_recoverable": True}
    monkeypatch.setattr(gr, "ibc_login_state", lambda: {"logged_in": True, "at": "x", "reason": "login completed"})
    assert reconcile._gateway_session(B()) is None

    class Lost(B):
        def ibkr_link_lost_since(self):
            return datetime(2026, 9, 28, 15, 2, 49, tzinfo=timezone.utc)
    got = reconcile._gateway_session(Lost())
    assert got["reason"].startswith("IBKR connectivity lost (error 1100)")


def test_a_logged_out_gateway_makes_the_broker_health_down_with_its_reason():
    from src.pipeline import _assess_broker_health
    report = {"mode": "ibkr_paper", "connected": True, "ok": True, "entries_submitted": 0,
              "exits_submitted": 0, "drift": [], "errors": [],
              "gateway_session": {"since": "2026-09-28 10:59:01",
                                  "reason": "login refused: Unrecognized Username or Password"}}
    h = _assess_broker_health(report)
    assert h["down"] is True and "IB Gateway NOT logged in to IBKR since 2026-09-28 10:59:01" in h["message"]
    assert h["gateway_session"]["reason"].startswith("login refused")
    healthy = dict(report, gateway_session=None)
    assert _assess_broker_health(healthy)["down"] is False


def test_the_email_banner_names_the_logged_out_gateway():
    from jinja2 import Template
    from src.notifications.email_sender import HTML_TEMPLATE
    start = HTML_TEMPLATE.index("{% if broker_health and broker_health.down %}")
    end = HTML_TEMPLATE.index("{% elif broker_health %}", start)
    block = HTML_TEMPLATE[start:end] + "{% endif %}"
    html = Template(block).render(broker_health={
        "down": True, "mode": "ibkr_paper", "message": "IB Gateway NOT logged in", "errors": [],
        "connected": True, "broker_timeouts": 0,
        "gateway_session": {"since": "2026-09-28 10:59:01", "reason": "login refused"}})
    assert "IB Gateway logged out of IBKR" in html and "IBC Gateway" in html and "10:59:01" in html


# ── the automatic restart after IBKR's re-login demand (2026-09-29) ─────────────
# "Re-login is required" -> IBC's Re-login -> "Unrecognized Username or Password":
# 9 of 9 times since August, around IBKR's nightly session reset; a FULL gateway
# restart logs in. After "Existing session detected" (a login elsewhere) it must
# stay alert-only — restarting would kick that session out.

RELOGIN = [
    "2026-09-29 00:25:15:710 IBC: detected dialog entitled: Re-login is required; event=Opened",
    "2026-09-29 00:25:15:710 IBC: Click button: Re-login",
    "2026-09-29 00:25:16:010 IBC: detected dialog entitled: Unrecognized Username or Password; event=Opened",
]
ELSEWHERE = [
    "2026-09-28 10:58:46:454 IBC: detected dialog entitled: Existing session detected; event=Opened",
    "2026-09-28 10:59:01:495 IBC: detected dialog entitled: Re-login is required; event=Opened",
    "2026-09-28 10:59:01:944 IBC: detected dialog entitled: Unrecognized Username or Password; event=Opened",
]


def test_the_login_state_tells_ibkrs_relogin_from_a_login_elsewhere(tmp_path):
    ok = "2026-09-28 12:07:54:000 IBC: Login has completed"
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, [ok] + RELOGIN)))["existing_session"] is False
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, ELSEWHERE)))["existing_session"] is True
    stale = ["2026-09-28 22:00:00:000 IBC: detected dialog entitled: Existing session detected; event=Opened"]
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, stale + RELOGIN)))["existing_session"] is False  # hours before
    assert gr.ibc_login_state(str(_ibc_log(tmp_path, ELSEWHERE + [ok])))["existing_session"] is False


class _Gw:
    def __init__(self):
        self.dials = 0

    def connect(self, force=False):
        self.dials += 1
        return True

    def ibkr_link_lost_since(self):
        return None


def _sync_with_state(monkeypatch, state, restart=True):
    from src.broker import reconcile
    calls = []
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_relogin_restart", restart)
    monkeypatch.setattr(gr, "ibc_login_state", lambda: state)
    monkeypatch.setattr(gr, "maybe_restart_gateway", lambda reason, wait=True: calls.append(reason) or True)
    monkeypatch.setattr(gr, "wait_for_login", lambda after, **k: True)
    b = _Gw()     # no get_account: the sync stops (fail-soft) right after the session block
    report = reconcile.sync(broker=b, trades=[])
    return calls, b, report


def test_the_sync_restarts_the_gateway_after_ibkrs_refused_relogin(monkeypatch):
    refused = {"logged_in": False, "at": "2026-09-29 00:25:16", "reason": "login refused",
               "existing_session": False}
    calls, b, report = _sync_with_state(monkeypatch, refused)
    assert len(calls) == 1 and "re-login was refused" in calls[0]
    assert b.dials == 2 and report["gateway_relogin_restart"] == "logged in"   # redialed the fresh gateway
    elsewhere = dict(refused, existing_session=True)
    calls, _, report = _sync_with_state(monkeypatch, elsewhere)
    assert calls == [] and report["gateway_session"]["auto_recoverable"] is False   # alert only
    calls, _, report = _sync_with_state(monkeypatch, refused, restart=False)
    assert calls == [] and "gateway_relogin_restart" not in report                   # the switch off
    calls, _, _ = _sync_with_state(monkeypatch, {"logged_in": True, "at": "2026-09-29 03:02:40",
                                                  "reason": "login completed", "existing_session": False})
    assert calls == []                                    # logged in: nothing to do


def test_the_restart_reaches_the_tick_report_and_the_email():
    import src.pipeline as pl
    from jinja2 import Template
    from src.notifications.email_sender import HTML_TEMPLATE
    live = {"ok": True, "connected": True, "entries_submitted": 1, "errors": [], "drift": [],
            "gateway_session": None, "gateway_relogin_restart": "logged in"}
    end = {"ok": True, "connected": True, "entries_submitted": 0, "errors": [], "drift": [],
           "gateway_session": None}
    merged = pl._merge_broker_reports(live, end)
    assert merged["gateway_relogin_restart"] == "logged in" and merged["entries_submitted"] == 1
    h = pl._assess_broker_health(dict(merged, mode="ibkr_paper"))
    assert h["down"] is False and h["gateway_relogin_restart"] == "logged in"
    start = HTML_TEMPLATE.index("{% if broker_health and broker_health.down %}")
    stop = HTML_TEMPLATE.index("{% if sel_health and sel_health.down %}", start)
    block = HTML_TEMPLATE[start:stop]
    html = Template(block).render(broker_health=dict(h, slippage=[], retries=0, stale_cancels=0,
                                                     entry_cancels_on_close=0, fills_repaired=0))
    assert "restarted automatically after IBKR's refused re-login (logged in)" in html
    # the red banner: an auto-recoverable refusal says the scheduler handles it; a login
    # elsewhere says to close that session first
    down = {"down": True, "mode": "ibkr_paper", "message": "x", "errors": [], "connected": True,
            "broker_timeouts": 0}
    auto = Template(block).render(broker_health=dict(down, gateway_session={
        "since": "2026-09-29 00:25:16", "reason": "login refused", "auto_recoverable": True}))
    assert "restarts the gateway automatically" in auto and "close that session" not in auto
    manual = Template(block).render(broker_health=dict(down, gateway_session={
        "since": "2026-09-28 10:59:01", "reason": "login refused", "auto_recoverable": False}))
    assert "close that session" in manual and "restarts the gateway automatically" not in manual
