"""Gateway auto-recovery + the overnight-refusal classification (2026-07-14).

Root incident: EVERY overnight-routed order (118 over 2 weeks) was silently
discarded by the gateway's API Precautionary Settings (Error 10329), recorded
as bare 'Cancelled', counted as a REJECT → nightly broker-health CRITICAL. Plus
the standing gap that a dead/wedged GATEWAY needed a manual kill + IBC relaunch.

Covers: reason capture from trade.log, the overnight_deferred tally split, and
maybe_restart_gateway's gating (mode / setting / cooldown) with subprocess
fully faked so no real gateway is ever touched.
"""

from types import SimpleNamespace

import pytest

import src.broker.gateway_recovery as gr
import src.broker.reconcile as rec
from config.settings import settings
from src.broker.base import OrderResult
from src.broker.ibkr import _trade_log_reason


# ── _trade_log_reason ────────────────────────────────────────────────────────

def _entry(msg):
    return SimpleNamespace(message=msg, status="Cancelled", errorCode=10329)


def test_trade_log_reason_last_nonempty():
    trade = SimpleNamespace(log=[_entry(""), _entry("Error 10329, reqId 4: This order "
                                                    "will be directly routed to OVERNIGHT."),
                                 _entry("")])
    assert "10329" in _trade_log_reason(trade)


def test_trade_log_reason_empty_log():
    assert _trade_log_reason(SimpleNamespace(log=[])) is None
    assert _trade_log_reason(SimpleNamespace(log=None)) is None
    assert _trade_log_reason(SimpleNamespace()) is None      # no .log attr at all


# ── _tally_submit_failure: overnight refusals are deferred, not rejects ──────

def _res(status="Cancelled", error=None):
    return OrderResult(ok=False, ticker="NET", side="SELL", requested_qty=5,
                       status=status, error=error)


def test_overnight_cancel_is_deferred_not_reject(monkeypatch):
    monkeypatch.setattr(rec, "_overnight_routing_active", lambda: True)
    report = rec._new_report()
    rec._tally_submit_failure(report, "exit", "NET",
                              _res(error="Cancelled: Error 10329 … Precautionary Settings"))
    assert report["overnight_deferred"] == 1
    assert report["rejects"] == 0
    assert report["errors"] == []          # benign — kept out of the error list


def test_rth_cancel_stays_a_reject(monkeypatch):
    monkeypatch.setattr(rec, "_overnight_routing_active", lambda: False)
    report = rec._new_report()
    rec._tally_submit_failure(report, "exit", "NET", _res())
    assert report["rejects"] == 1
    assert report["overnight_deferred"] == 0
    assert report["errors"]


def test_overnight_hard_error_stays_a_reject(monkeypatch):
    """A genuine reject (margin/permissions → status ERROR) alerts even overnight."""
    monkeypatch.setattr(rec, "_overnight_routing_active", lambda: True)
    report = rec._new_report()
    rec._tally_submit_failure(report, "entry", "NET",
                              _res(status="ERROR", error="insufficient funds"))
    assert report["rejects"] == 1
    assert report["overnight_deferred"] == 0


def test_unresponsive_precedes_deferral(monkeypatch):
    """A gateway timeout during the overnight session is a TIMEOUT, not a deferral."""
    monkeypatch.setattr(rec, "_overnight_routing_active", lambda: True)
    report = rec._new_report()
    rec._tally_submit_failure(report, "exit", "NET",
                              _res(status="ERROR", error="TimeoutError"))
    assert report["broker_timeouts"] == 1
    assert report["overnight_deferred"] == 0
    assert report["rejects"] == 0


# ── maybe_restart_gateway gating ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_recovery_state(monkeypatch):
    gr._reset_for_tests()
    # No test here may ever run a real command.
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    monkeypatch.setattr(gr.time, "sleep", lambda s: None)
    yield calls
    gr._reset_for_tests()


def test_refused_outside_paper(monkeypatch, _fresh_recovery_state):
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", True)
    for mode in ("off", "dry_run", "ibkr_live"):
        monkeypatch.setattr(settings, "broker_mode", mode)
        assert gr.maybe_restart_gateway("test") is False
    assert _fresh_recovery_state == []          # no commands ran


def test_refused_when_disabled(monkeypatch, _fresh_recovery_state):
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", False)
    assert gr.maybe_restart_gateway("test") is False
    assert _fresh_recovery_state == []


def test_paper_restart_kills_and_triggers_task(monkeypatch, _fresh_recovery_state):
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", True)
    monkeypatch.setattr(gr, "_pid_listening_on", lambda port: 4321)
    assert gr.maybe_restart_gateway("wedged", wait=False) is True
    cmds = _fresh_recovery_state
    assert ["taskkill", "/PID", "4321", "/F"] in cmds
    assert any(c[:2] == ["schtasks", "/Run"] for c in cmds)


def test_cooldown_blocks_second_restart(monkeypatch, _fresh_recovery_state):
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", True)
    monkeypatch.setattr(gr, "_pid_listening_on", lambda port: None)
    assert gr.maybe_restart_gateway("first", wait=False) is True
    n_after_first = len(_fresh_recovery_state)
    assert gr.maybe_restart_gateway("second", wait=False) is False   # cooldown
    assert len(_fresh_recovery_state) == n_after_first


def test_wait_polls_port_then_grace(monkeypatch, _fresh_recovery_state):
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_gateway_auto_restart", True)
    seen = {"n": 0}

    def pid(port):
        seen["n"] += 1
        return 4321 if seen["n"] >= 3 else None   # port comes back on the 3rd probe

    monkeypatch.setattr(gr, "_pid_listening_on", pid)
    assert gr.maybe_restart_gateway("connect exhausted", wait=True) is True


def test_email_banner_shows_exact_errors_and_10329_guidance():
    """The REAL email down-banner markup renders the exact per-order errors
    (HTML-escaped) plus the 10329 fix guidance — extracted from HTML_TEMPLATE so
    it tests the shipped template, not a copy."""
    from jinja2 import Template
    import src.notifications.email_sender as es

    tpl = es.HTML_TEMPLATE
    start = "{% if broker_health and broker_health.down %}"
    elif_ = "{% elif broker_health %}"
    i, j = tpl.index(start), tpl.index(elif_)
    # the down-branch body is balanced (its nested if/for all close inside it),
    # so wrapping it in if/endif gives a standalone, renderable template.
    frag = Template(tpl[i:j] + "{% endif %}")

    verdict = {
        "down": True, "mode": "ibkr_paper", "connected": True, "broker_timeouts": 0,
        "message": "1 order(s) rejected",
        "errors": ["exit NET: Cancelled: Error 10329, reqId 4: This order will be "
                   "directly routed to OVERNIGHT. <Precautionary Settings>"],
    }
    html = frag.render(broker_health=verdict)
    assert "Broker execution issue (ibkr_paper)" in html
    assert "1 order(s) rejected" in html
    assert "Error 10329" in html                       # exact reason shown
    assert "&lt;Precautionary Settings&gt;" in html    # untrusted text HTML-escaped
    assert "BypassRedirectOrderWarning=yes" in html    # actionable 10329 fix
    # the wedge hint only appears when there were timeouts
    assert "likely wedged" not in html


def test_email_banner_shows_wedge_hint_on_timeouts():
    from jinja2 import Template
    import src.notifications.email_sender as es
    tpl = es.HTML_TEMPLATE
    i = tpl.index("{% if broker_health and broker_health.down %}")
    j = tpl.index("{% elif broker_health %}")
    frag = Template(tpl[i:j] + "{% endif %}")
    html = frag.render(broker_health={
        "down": True, "mode": "ibkr_paper", "connected": True, "broker_timeouts": 6,
        "message": "broker NOT RESPONDING — 6 request(s) timed out", "errors": [],
    })
    assert "likely wedged" in html
    assert "auto-recovers the gateway" in html


def test_sync_kicks_gateway_when_connect_exhausted(monkeypatch):
    """sync() fires the recovery after its retries, then dials once more."""
    dialed = {"n": 0}

    class _B:
        def connect(self):
            dialed["n"] += 1
            return dialed["n"] >= 3            # succeeds only on the post-restart dial

    monkeypatch.setattr(settings, "broker_connect_retries", 1)   # 2 attempts
    monkeypatch.setattr(settings, "broker_connect_retry_wait_seconds", 1)
    monkeypatch.setattr(rec.time, "sleep", lambda s: None)
    restarts = []
    import src.broker.gateway_recovery as _gr
    monkeypatch.setattr(_gr, "maybe_restart_gateway",
                        lambda reason, wait=True: restarts.append(reason) or True)
    b = _B()
    # get_account etc. don't exist on the fake → sync fails soft AFTER connect;
    # all we assert is the recovery fired and the extra dial happened.
    report = rec.sync(broker=b, trades=[])
    assert restarts == ["sync connect retries exhausted"]
    assert dialed["n"] == 3
    assert report["connected"] is True
