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


def _call(wsgi, auth_header=None, path="/", **environ_extra):
    """Drive a WSGI callable; return (status, headers, body).

    Defaults describe a request arriving over the PUBLIC tunnel — loopback peer
    (ngrok forwards to 127.0.0.1), public Host, forwarded-for stamped by the edge.
    The local-browser case is spelled out explicitly by ``_local_env()``, so a
    test that forgets to say which it means gets the gated one.
    """
    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path,
        "REMOTE_ADDR": "127.0.0.1",
        "HTTP_HOST": "passing-debrief-october.ngrok-free.dev",
        "HTTP_X_FORWARDED_FOR": "203.0.113.7",
        "HTTP_X_FORWARDED_PROTO": "https",
    }
    environ.update(environ_extra)
    # An explicit None REMOVES a default key — that is how a caller says "this
    # request carried no X-Forwarded-For", which is the whole local/public tell.
    environ = {k: v for k, v in environ.items() if v is not None}
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


def _local_env(**over):
    """A browser on this machine: loopback peer, loopback Host, no proxy headers."""
    env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8050",
           "HTTP_X_FORWARDED_FOR": None, "HTTP_X_FORWARDED_PROTO": None}
    env.update(over)
    return env


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


# ─────────────────────────────────────────────────────────────────────────────
# Localhost bypass: no password on this machine, password on the public URL.
#
# The trap these tests exist for: ngrok forwards the public hostname to
# 127.0.0.1, so a public visitor's REMOTE_ADDR is loopback too. A bypass keyed on
# the peer address alone would hand the open internet a free pass while every log
# line, every config value and a naive test still read "gated".
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _default_bypass(monkeypatch):
    """Pin the bypass networks so these tests don't depend on the live .env."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks",
                        "127.0.0.0/8,::1", raising=False)


def test_local_browser_needs_no_password(gate):
    """The actual ask: http://127.0.0.1:8050 opens with no login."""
    status, _, body = _call(gate, **_local_env())
    assert status.startswith("200")
    assert body == _SECRET


@pytest.mark.parametrize("host", ["localhost:8050", "127.0.0.1:8050", "[::1]:8050",
                                  "localhost", "127.0.0.1"])
def test_the_usual_localhost_spellings_all_bypass(gate, host):
    """A user types whichever of these they remember; all are the same machine."""
    remote = "::1" if host.startswith("[") else "127.0.0.1"
    status, _, _ = _call(gate, **_local_env(HTTP_HOST=host, REMOTE_ADDR=remote))
    assert status.startswith("200"), f"Host {host} demanded a password"


def test_ipv4_mapped_ipv6_peer_bypasses(gate):
    """A v4 client on a dual-stack listener shows up as ::ffff:127.0.0.1, which is
    NOT in 127.0.0.0/8 unless the address is unwrapped first."""
    status, _, _ = _call(gate, **_local_env(REMOTE_ADDR="::ffff:127.0.0.1"))
    assert status.startswith("200")


def test_tunnelled_request_is_gated_despite_loopback_peer(gate):
    """THE case. Loopback peer (ngrok forwards to 127.0.0.1) + public Host +
    forwarded-for: this must still demand the password."""
    status, _, body = _call(gate)          # _call's defaults ARE the tunnel case
    assert status.startswith("401")
    assert _SECRET not in body


@pytest.mark.parametrize("header", [
    "HTTP_X_FORWARDED_FOR", "HTTP_X_FORWARDED_PROTO", "HTTP_X_FORWARDED_HOST",
    "HTTP_FORWARDED", "HTTP_X_REAL_IP", "HTTP_VIA", "HTTP_NGROK_TRACE_ID",
    "HTTP_CF_CONNECTING_IP", "HTTP_TRUE_CLIENT_IP",
])
def test_any_proxy_header_defeats_the_bypass(gate, header):
    """Barrier 2, isolated: an otherwise perfectly local-looking request that
    carries ANY proxy marker is treated as remote. A client can ADD one of these
    (locking itself out, harmless); it cannot REMOVE the one the edge stamps on."""
    status, _, body = _call(gate, **_local_env(**{header: "anything"}))
    assert status.startswith("401"), f"{header} did not defeat the bypass"
    assert _SECRET not in body


def test_public_host_defeats_the_bypass_even_with_no_proxy_headers(gate):
    """Barrier 3, isolated. If a future tunnel stopped sending X-Forwarded-*, the
    Host header still names the public domain and still costs a password."""
    status, _, body = _call(gate, **_local_env(HTTP_HOST="passing-debrief-october.ngrok-free.dev"))
    assert status.startswith("401")
    assert _SECRET not in body


def test_missing_host_header_is_gated(gate):
    """HTTP/1.0 with no Host: locality is unprovable, so it is refused."""
    env = _local_env()
    env["HTTP_HOST"] = None
    assert _call(gate, **env)[0].startswith("401")


@pytest.mark.parametrize("peer", ["192.168.1.20", "100.101.102.103", "203.0.113.7", "", "garbage"])
def test_non_local_peer_is_gated(gate, peer):
    """Barrier 1: LAN and tailnet devices are not 'localhost'. They can still get
    in — with the password — which is the documented phone-access flow."""
    status, _, body = _call(gate, **_local_env(REMOTE_ADDR=peer, HTTP_HOST=f"{peer}:8050"))
    assert status.startswith("401")
    assert _SECRET not in body


def test_credentials_still_work_from_a_non_local_peer(gate):
    status, _, body = _call(gate, _basic(_USER, _PASSWORD),
                            **_local_env(REMOTE_ADDR="192.168.1.20", HTTP_HOST="192.168.1.20:8050"))
    assert status.startswith("200")
    assert body == _SECRET


def test_bypass_can_be_switched_off_entirely(gate, monkeypatch):
    """Empty DASHBOARD_AUTH_BYPASS_NETWORKS ⇒ even localhost needs the password."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks", "", raising=False)
    assert _call(gate, **_local_env())[0].startswith("401")


def test_bypass_can_be_widened_to_the_tailnet(gate, monkeypatch):
    """The documented way to make a phone over Tailscale passwordless — and proof
    that widening the bypass does NOT open the tunnel, which still fails on the
    proxy headers and the public Host."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks",
                        "127.0.0.0/8,::1,100.64.0.0/10", raising=False)
    tailnet = _local_env(REMOTE_ADDR="100.101.102.103", HTTP_HOST="100.101.102.103:8050")
    assert _call(gate, **tailnet)[0].startswith("200")
    assert _call(gate)[0].startswith("401"), "widening the bypass opened the tunnel"


# ── Tailscale MagicDNS: the phone addresses the PC by NAME, not by IP ──────────

@pytest.fixture
def tailnet(monkeypatch):
    """The live config: loopback + the tailnet v4/v6 ranges, and the MagicDNS name."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks",
                        "127.0.0.0/8,::1,100.64.0.0/10,fd7a:115c:a1e0::/48", raising=False)
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_hosts",
                        "victushp.tail8e1bf1.ts.net", raising=False)


@pytest.mark.parametrize("peer,host", [
    ("100.90.109.43", "victushp.tail8e1bf1.ts.net:8050"),        # MagicDNS over v4
    ("100.90.109.43", "VictusHP.Tail8e1bf1.TS.NET:8050"),        # browsers may vary case
    ("100.90.109.43", "victushp.tail8e1bf1.ts.net."),            # FQDN trailing dot
    ("100.90.109.43", "100.90.109.43:8050"),                     # by v4 address
    ("fd7a:115c:a1e0::2d3b:6d2c", "[fd7a:115c:a1e0::2d3b:6d2c]:8050"),   # by v6 address
    ("fd7a:115c:a1e0::2d3b:6d2c", "victushp.tail8e1bf1.ts.net:8050"),    # MagicDNS over v6
])
def test_phone_on_the_tailnet_needs_no_password(gate, tailnet, peer, host):
    """Every way a Tailscale phone can address this PC must open without a login —
    a bypass that only covers the raw IPv4 still prompts whenever the Tailscale app
    hands the browser the machine NAME, which is its default presentation."""
    status, _, body = _call(gate, **_local_env(REMOTE_ADDR=peer, HTTP_HOST=host))
    assert status.startswith("200"), f"{peer} via {host} was asked for a password"
    assert body == _SECRET


def test_trusted_hostname_does_not_open_the_tunnel(gate, tailnet):
    """The load-bearing one. Naming a host as trusted must not become a way past
    the gate for tunnelled traffic — which arrives from LOOPBACK, so barrier 1
    cannot stop it and only the proxy headers and the public Host do."""
    assert _call(gate)[0].startswith("401"), "trusting a hostname opened the tunnel"
    # ...and even if the tunnel ever relayed the trusted name as Host, the proxy
    # headers still give it away.
    spoofed = _call(gate, HTTP_HOST="victushp.tail8e1bf1.ts.net:8050")
    assert spoofed[0].startswith("401")
    assert _SECRET not in spoofed[2]


@pytest.mark.parametrize("host", [
    "passing-debrief-october.ngrok-free.dev",   # the real public domain
    "victushp.tail8e1bf1.ts.net.evil.com",      # suffix-looks-right, is not
    "evil.com",
    "victushp.tail8e1bf1.ts.ne",                # near-miss
])
def test_untrusted_hostnames_are_still_gated(gate, tailnet, host):
    """DNS rebinding is the reason the Host check survives the hostname allowlist:
    a page that resolves its own domain to 100.90.109.43 still sends ITS name."""
    status, _, body = _call(gate, **_local_env(REMOTE_ADDR="100.90.109.43", HTTP_HOST=host))
    assert status.startswith("401"), f"{host} was treated as local"
    assert _SECRET not in body


def test_suffix_entry_matches_any_name_below_it(gate, monkeypatch):
    """A leading dot is the rename-proof form: it survives a tailnet or machine
    rename, at the cost of trusting every name under that suffix."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks",
                        "100.64.0.0/10", raising=False)
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_hosts",
                        ".ts.net", raising=False)
    env = _local_env(REMOTE_ADDR="100.90.109.43", HTTP_HOST="anything.tailnet.ts.net:8050")
    assert _call(gate, **env)[0].startswith("200")
    # The suffix must be anchored at a label boundary, not a substring.
    env = _local_env(REMOTE_ADDR="100.90.109.43", HTTP_HOST="evil-ts.net:8050")
    assert _call(gate, **env)[0].startswith("401")


def test_trusted_hostname_still_requires_a_trusted_peer(gate, tailnet):
    """The name is not a password: someone off the tailnet who sends the MagicDNS
    name in Host is still refused, because their peer address gives them away."""
    status, _, _ = _call(gate, **_local_env(REMOTE_ADDR="203.0.113.7",
                                            HTTP_HOST="victushp.tail8e1bf1.ts.net:8050"))
    assert status.startswith("401")


def test_unparseable_bypass_network_grants_nothing(gate, monkeypatch):
    """A typo'd CIDR must not throw (that would 500 the whole dashboard) and must
    not grant access — it is dropped, and a request matching nothing is gated."""
    from dashboard import app as dash_app
    monkeypatch.setattr(dash_app.settings, "dashboard_auth_bypass_networks",
                        "not-a-cidr,127.0.0.1/33", raising=False)
    assert _call(gate, **_local_env())[0].startswith("401")


def test_waitress_is_told_to_keep_the_proxy_headers():
    """The coupling that makes the proxy-header barrier real.

    waitress defaults ``clear_untrusted_proxy_headers=True``, which POPS
    X-Forwarded-* out of the environ before the app runs. Under that default every
    tunnelled request reaches the gate looking exactly like a local one, and the
    only thing left standing between the public URL and live P&L is the Host
    header. Nothing about that failure is visible from inside the app — the unit
    tests above would all still pass — so the wiring is asserted at the source.
    """
    import inspect
    from dashboard import app as dash_app

    src = inspect.getsource(dash_app._serve_once)
    assert "clear_untrusted_proxy_headers=False" in src, (
        "waitress will strip X-Forwarded-* and the localhost bypass will treat "
        "every public visitor as local"
    )


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
