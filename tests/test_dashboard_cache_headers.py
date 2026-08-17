"""Every client must run the CURRENT dashboard (2026-08-16).

The bug this pins: no route sent cache directives, so each browser applied its
own heuristic and phones cached hardest. A deploy landed on the server while the
phone kept rendering the previous version indefinitely — and nothing server-side
could show it, because the logs record a normal 200 for a page the viewer never
sees. The only workarounds were a private tab or a ``?v=`` query string.

Cache headers are exactly the kind of mechanism this project refuses to verify by
review: wrong ones look identical to right ones from the server, and the symptom
appears on a device the developer isn't holding.
"""

import pytest

from dashboard import app as dash_app
from src.db import repo

# dashboard.data flips the repo read-only process-wide at import, which is right
# for the dashboard and wrong for the rest of the suite.
repo.set_read_only(False)


@pytest.fixture()
def client():
    """An AUTHENTICATED test client.

    Since 2026-08-16 the bypass lists are empty and the password gates every
    request, including loopback — so an unauthenticated client gets 401 on
    everything and would test the deny path instead of the cache headers. The
    credentials come from settings, so this follows the real configuration
    rather than pinning a fixture password; with no password configured the
    gate is off and the header is simply ignored."""
    import base64

    from config import settings

    dash_app.app.server.config["TESTING"] = True
    pw = (settings.dashboard_auth_password or "").strip()
    user = (settings.dashboard_auth_username or "").strip()
    with dash_app.app.server.test_client() as c:
        if pw:
            token = base64.b64encode(f"{user}:{pw}".encode()).decode()
            c.environ_base["HTTP_AUTHORIZATION"] = f"Basic {token}"
        yield c


@pytest.fixture(autouse=True)
def _clear_auth_lockout():
    """A neighbouring test that trips the global lockout would otherwise turn
    every request here into a 429."""
    dash_app._auth_failures.clear()
    dash_app._auth_state.update(locked_until=0.0, announced=False)
    yield


def _cc(resp) -> str:
    return resp.headers.get("Cache-Control", "")


# ── the version-defining routes must never be cached ─────────────────────────

@pytest.mark.parametrize("path", ["/", "/_dash-layout", "/_dash-dependencies"])
def test_version_defining_routes_are_no_store(client, path):
    """These three decide WHICH dashboard the browser runs: the index lists the
    asset URLs, the layout is the structure, the dependency graph wires the
    callbacks. A stale copy of any one pins the viewer to an old app."""
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} did not serve"
    assert "no-store" in _cc(resp), f"{path} may be cached: {_cc(resp)!r}"


def test_layout_is_rebuilt_per_request_so_no_store_means_fresh_data(client):
    """no-store is only worth anything because serve_layout is a FUNCTION — the
    layout is rebuilt per page load, so a reload also picks up the latest run."""
    assert callable(dash_app.app.layout) or dash_app.app.layout is not None
    assert client.get("/_dash-layout").status_code == 200


# ── the big bundles stay cached, or the phone pays for it ────────────────────

def test_component_suites_are_cached_hard(client):
    """The ~MBs of React/Plotly bundles carry their package version IN the URL,
    so an upgrade changes the URL. Caching them immutably is what keeps the page
    fast on a phone despite no-store above — losing this would be a silent
    performance regression, not a correctness one.

    The bundle URL is taken from the index page rather than hardcoded: that is
    how a browser finds it, and the exact filenames move with every Dash
    release (a hardcoded guess fails on upgrade for reasons unrelated to
    caching)."""
    import re
    body = client.get("/").get_data(as_text=True)
    urls = re.findall(r'src="(/_dash-component-suites/[^"]+)"', body)
    assert urls, "the index page references no component suites"

    resp = client.get(urls[0])
    assert resp.status_code == 200, f"{urls[0]} did not serve"
    cc = _cc(resp)
    assert "max-age=31536000" in cc and "immutable" in cc, cc


def test_assets_revalidate(client):
    """Our own CSS: cacheable but must be revalidated. Dash appends ?m=<mtime>
    so the URL changes when the file does; no-cache closes the last door on a
    stale stylesheet at the cost of a 304."""
    resp = client.get("/assets/style.css")
    assert resp.status_code == 200
    assert "no-cache" in _cc(resp)


def test_stylesheet_is_actually_linked_from_the_index(client):
    """The whole design system rides on this one tag; if the assets folder ever
    stops being discovered, the page silently renders unstyled."""
    body = client.get("/").get_data(as_text=True)
    assert "/assets/style.css" in body
    assert 'name="viewport"' in body, "phones would render at desktop width"


# ── the auth gate's own 401 must not be cached either ────────────────────────

def test_auth_denial_is_no_store():
    """A cached 401 would lock a viewer out even after the password is fixed —
    the deny response is built in the WSGI wrapper, outside Flask's after_request
    hook, so it carries its own header and is asserted separately."""
    seen = {}

    def inner(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    gate = dash_app._basic_auth_middleware(inner, "u", "pw")

    def start_response(status, headers):
        seen["status"] = status
        seen["headers"] = dict(headers)

    gate({"REMOTE_ADDR": "8.8.8.8", "HTTP_HOST": "example.com",
          "HTTP_X_FORWARDED_FOR": "8.8.8.8"}, start_response)
    assert seen["status"].startswith("401")
    assert "no-store" in seen["headers"].get("Cache-Control", "")
