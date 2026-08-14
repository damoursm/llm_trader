"""The dashboard's shared-password gate (added 2026-08-13).

The dashboard shows live positions, P&L, account NAV and fill prices, and is
published on a public URL through an ngrok tunnel. This gate is the only thing
between that URL and the open internet, so it is verified mechanically rather
than by review — the same rule the project applies to unread settings, sinkless
loggers and optional imports guarding a live verdict: a mechanism whose failure
mode is indistinguishable from normal operation must be tested.

Every assertion below is a way the gate could be OPEN while still LOOKING closed.
"""

import base64

import pytest

from dashboard.app import _basic_auth_middleware
from src.db import repo

# Importing dashboard.* flips repo into read-only process-wide (dashboard/data.py
# does it at import, which is correct for the dashboard and wrong for the suite).
# Undo it: this module only needs the middleware, and leaving the flag set would
# make later tests' writes fail in a way that looks unrelated to this file.
repo.set_read_only(False)


_USER = "viewer"
_PASSWORD = "correct-horse-battery-staple"
# Stands in for everything the real inner app would serve: positions, P&L, NAV.
_SECRET = b"INNER-PAYLOAD-POSITIONS-AND-PNL"


def _inner(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [_SECRET]


@pytest.fixture
def gate():
    return _basic_auth_middleware(_inner, _USER, _PASSWORD)


def _call(wsgi, auth_header=None, path="/"):
    """Drive a WSGI callable; return (status, headers, body)."""
    environ = {"REQUEST_METHOD": "GET", "PATH_INFO": path}
    if auth_header is not None:
        environ["HTTP_AUTHORIZATION"] = auth_header
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(wsgi(environ, start_response))
    return captured["status"], dict(captured["headers"]), body


def _basic(user, password):
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_anonymous_request_is_refused(gate):
    status, headers, body = _call(gate)
    assert status.startswith("401")
    assert _SECRET not in body
    # Without WWW-Authenticate a browser never prompts, so the gate would look
    # like a broken dashboard instead of a login.
    assert "Basic" in headers["WWW-Authenticate"]


def test_correct_credentials_pass_through(gate):
    status, _, body = _call(gate, _basic(_USER, _PASSWORD))
    assert status.startswith("200")
    assert body == _SECRET


@pytest.mark.parametrize("header", [
    "Basic " + base64.b64encode(b"viewer:wrong").decode(),          # bad password
    "Basic " + base64.b64encode(b"attacker:" + _PASSWORD.encode()).decode(),  # bad user
    "Basic " + base64.b64encode(_PASSWORD.encode()).decode(),       # password, no user
    "Basic " + base64.b64encode(b"").decode(),                      # empty
    "Basic !!!not-base64!!!",                                       # undecodable
    "Bearer " + base64.b64encode(f"{_USER}:{_PASSWORD}".encode()).decode(),   # wrong scheme
    "",                                                             # empty header
])
def test_bad_credentials_are_refused(gate, header):
    status, _, body = _call(gate, header)
    assert status.startswith("401")
    assert _SECRET not in body


def test_gate_covers_every_path_not_just_the_page(gate):
    """The HTML shell is 4 KB and carries nothing; the DATA rides the callback
    XHRs and the layout endpoint. A gate on '/' alone would leak everything that
    matters while appearing to work in a browser."""
    for path in ("/", "/_dash-layout", "/_dash-dependencies",
                 "/_dash-update-component", "/assets/style.css", "/anything/else"):
        status, _, body = _call(gate, path=path)
        assert status.startswith("401"), f"{path} was not gated"
        assert _SECRET not in body


def test_password_containing_a_colon_still_works():
    """user:pass is colon-delimited, so a colon in the password is the obvious
    place for a naive split() to break the login."""
    pwd = "a:b:c"
    g = _basic_auth_middleware(_inner, _USER, pwd)
    assert _call(g, _basic(_USER, pwd))[0].startswith("200")
    # ...and must not become a way to authenticate as something else.
    assert _call(g, _basic("a", "b:c"))[0].startswith("401")


def test_install_is_driven_by_settings(monkeypatch):
    """The gate must be ON exactly when a password is configured. Both directions
    matter: no-password-but-gated would lock the owner out of a loopback
    dashboard, and password-but-ungated is the public-exposure failure."""
    from dashboard import app as dash_app

    flask_app = dash_app.app.server
    # Flask defines wsgi_app as a CLASS method, so every attribute read returns a
    # fresh bound-method object: compare with == (same __func__ + __self__), never
    # `is`. Installing the gate replaces it with an instance attribute, so the
    # pristine state is restored by removing that attribute, not by re-assigning.
    had_instance_attr = "wsgi_app" in flask_app.__dict__
    original = flask_app.wsgi_app
    try:
        monkeypatch.setattr(dash_app.settings, "dashboard_auth_password", "", raising=False)
        assert dash_app._install_basic_auth() is False
        assert flask_app.wsgi_app == original, "gate installed with no password"

        monkeypatch.setattr(dash_app.settings, "dashboard_auth_password", "s3cret", raising=False)
        monkeypatch.setattr(dash_app.settings, "dashboard_auth_username", _USER, raising=False)
        assert dash_app._install_basic_auth() is True
        assert flask_app.wsgi_app != original, "password set but no gate installed"
        # And the installed gate actually rejects an anonymous caller.
        status, _, _ = _call(flask_app.wsgi_app)
        assert status.startswith("401")
    finally:
        if had_instance_attr:
            flask_app.wsgi_app = original
        else:
            flask_app.__dict__.pop("wsgi_app", None)
