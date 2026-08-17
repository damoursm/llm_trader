"""Online brute-force protection for the public dashboard (2026-08-16).

Context: the dashboard moved to a single public access point — Tailscale Funnel
at https://victushp.tail8e1bf1.ts.net — with the network bypasses removed, so ONE
shared password is the only thing between the internet and live positions/P&L.
Before that change the gate had no rate limit at all: a wrong password cost an
attacker nothing but a round trip.

Two properties carry the defence and neither is observable from normal use:

  1. the limit is checked BEFORE the password comparison, so it bounds GUESSES
     rather than merely successful answers;
  2. it is GLOBAL rather than per-IP, because behind Funnel every request
     arrives from 127.0.0.1 and the only per-client signal (X-Forwarded-For) is
     attacker-controlled — a per-IP bucket would hand out a fresh quota per
     forged header value.
"""

import time

import pytest

from dashboard import app as dash_app
from src.db import repo

# dashboard.data flips the repo read-only process-wide at import, which is right
# for the dashboard and wrong for the rest of the suite.
repo.set_read_only(False)


@pytest.fixture(autouse=True)
def _reset_auth_state():
    """Each test gets a clean bucket (module-level state is shared)."""
    dash_app._auth_failures.clear()
    dash_app._auth_state.update(locked_until=0.0, announced=False)
    yield
    dash_app._auth_failures.clear()
    dash_app._auth_state.update(locked_until=0.0, announced=False)


def _inner(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"ok"]


def _gate():
    return dash_app._basic_auth_middleware(_inner, "viewer", "correct-horse")


def _call(gate, auth: str | None = None, **extra):
    """Drive one request through the gate; returns (status, headers)."""
    import base64 as b64
    env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "victushp.tail8e1bf1.ts.net",
           # a proxy header is what a real Funnel request carries, and it is
           # what stops the bypass check from mistaking it for a local browser
           "HTTP_X_FORWARDED_FOR": "203.0.113.9"}
    env.update(extra)
    if auth is not None:
        env["HTTP_AUTHORIZATION"] = "Basic " + b64.b64encode(auth.encode()).decode()
    seen = {}

    def start_response(status, headers):
        seen["status"] = status
        seen["headers"] = dict(headers)
    gate(env, start_response)
    return seen["status"], seen["headers"]


# ── the basics still hold ────────────────────────────────────────────────────

def test_correct_password_passes():
    status, _ = _call(_gate(), "viewer:correct-horse")
    assert status.startswith("200")


def test_wrong_password_is_denied():
    status, headers = _call(_gate(), "viewer:nope")
    assert status.startswith("401")
    assert "Basic" in headers.get("WWW-Authenticate", "")


# ── the lockout ──────────────────────────────────────────────────────────────

def test_repeated_failures_trip_a_lockout():
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES):
        _call(gate, "viewer:wrong")
    status, headers = _call(gate, "viewer:wrong-again")
    assert status.startswith("429"), "brute force was never throttled"
    assert int(headers.get("Retry-After", 0)) > 0


def test_lockout_bounds_guesses_not_just_answers():
    """THE ordering property. If the limit were checked after the comparison, an
    attacker's guesses would still all be evaluated and the lockout would only
    hide the outcome — useless. A CORRECT password during lockout must also be
    refused, which is what proves no comparison happened."""
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES):
        _call(gate, "viewer:wrong")
    status, _ = _call(gate, "viewer:correct-horse")
    assert status.startswith("429"), "the password was still being compared during lockout"


def test_lockout_is_global_not_per_ip():
    """Behind Funnel every request is 127.0.0.1 and X-Forwarded-For is forged
    freely. Rotating it must not buy a fresh quota."""
    gate = _gate()
    for i in range(dash_app._AUTH_MAX_FAILURES):
        _call(gate, "viewer:wrong", HTTP_X_FORWARDED_FOR=f"203.0.113.{i}")
    status, _ = _call(gate, "viewer:wrong", HTTP_X_FORWARDED_FOR="198.51.100.7")
    assert status.startswith("429"), "rotating X-Forwarded-For evaded the limit"


def test_lockout_expires():
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES):
        _call(gate, "viewer:wrong")
    assert _call(gate, "viewer:correct-horse")[0].startswith("429")
    dash_app._auth_state["locked_until"] = time.time() - 1      # window elapsed
    assert _call(gate, "viewer:correct-horse")[0].startswith("200")


def test_a_success_clears_the_counter():
    """A human who mistypes twice then gets it right must not drift toward a
    lockout on their next visit."""
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES - 1):
        _call(gate, "viewer:wrong")
    assert _call(gate, "viewer:correct-horse")[0].startswith("200")
    assert dash_app._auth_failures == []
    # ...and the budget is genuinely replenished.
    for _ in range(dash_app._AUTH_MAX_FAILURES - 1):
        _call(gate, "viewer:wrong")
    assert _call(gate, "viewer:correct-horse")[0].startswith("200")


def test_credential_less_probe_is_not_counted_as_a_guess():
    """The browser's first request carries no Authorization header — that is the
    normal prelude to the password prompt, not an attempt. Counting it would let
    ordinary page loads (each pulling several sub-resources) trip the lockout."""
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES * 2):
        status, _ = _call(gate, None)
        assert status.startswith("401")
    assert _call(gate, "viewer:correct-horse")[0].startswith("200")


def test_old_failures_fall_out_of_the_window():
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES - 1):
        _call(gate, "viewer:wrong")
    # age them past the sliding window
    dash_app._auth_failures[:] = [t - dash_app._AUTH_FAILURE_WINDOW - 1
                                  for t in dash_app._auth_failures]
    _call(gate, "viewer:wrong")
    assert _call(gate, "viewer:correct-horse")[0].startswith("200")


def test_throttle_response_is_never_cached():
    """A cached 429 would keep locking a viewer out after the window passed."""
    gate = _gate()
    for _ in range(dash_app._AUTH_MAX_FAILURES):
        _call(gate, "viewer:wrong")
    _status, headers = _call(gate, "viewer:wrong")
    assert "no-store" in headers.get("Cache-Control", "")


# ── the shipped posture ──────────────────────────────────────────────────────

def test_guardrails_are_constants_not_settings():
    """These are security guardrails, which this project keeps fixed — a knob
    that weakens a defence is a knob that eventually gets turned."""
    from config import settings
    for name in ("dashboard_auth_max_failures", "dashboard_auth_lockout_seconds",
                 "dashboard_auth_failure_window_seconds"):
        assert not hasattr(settings, name), f"{name} became a tunable setting"
    assert dash_app._AUTH_MAX_FAILURES > 0
    assert dash_app._AUTH_LOCKOUT_SECONDS > 0
