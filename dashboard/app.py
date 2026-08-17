"""Plotly Dash monitoring dashboard — rationale · method performance · returns.

Single source of truth is DuckDB (read-only here). Launch with:

    python main.py --dashboard
"""

from __future__ import annotations

import base64
import hmac
import ipaddress
import time
from datetime import datetime, timezone

import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from loguru import logger

from config import settings
from dashboard import data, figures
from src.utils import ET

app = Dash(__name__, title="LLM Trader Monitor", suppress_callback_exceptions=True)
server = app.server  # for WSGI deployment if ever needed

_DENY_BODY = b"Authentication required.\n"

# Hostnames that mean "this machine" in a browser's address bar.
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

# Headers that exist ONLY because something proxied the request. Their VALUES are
# never trusted (any client can forge them) — only their PRESENCE is used, and only
# in the safe direction: present ⇒ not local ⇒ password required. Forging one can
# therefore lock a caller out, never let one in.
_PROXY_HEADER_KEYS = frozenset({
    "HTTP_FORWARDED",            # RFC 7239
    "HTTP_X_REAL_IP",
    "HTTP_VIA",
    "HTTP_CF_CONNECTING_IP",     # Cloudflare
    "HTTP_TRUE_CLIENT_IP",
    "HTTP_PROXY_CONNECTION",
})
_PROXY_HEADER_PREFIXES = ("HTTP_X_FORWARDED_", "HTTP_X_ORIGINAL_", "HTTP_NGROK_")

_NETWORK_CACHE: dict[str, tuple] = {}
_HOST_CACHE: dict[str, tuple] = {}


def _bypass_hosts() -> tuple:
    """Parsed ``dashboard_auth_bypass_hosts``, lowercased. Entries starting with
    "." are suffix matches; the rest are exact hostnames."""
    raw = (settings.dashboard_auth_bypass_hosts or "").strip()
    cached = _HOST_CACHE.get(raw)
    if cached is None:
        cached = tuple(
            part.strip().lower().rstrip(".") if not part.strip().startswith(".")
            else part.strip().lower()
            for part in raw.split(",") if part.strip()
        )
        _HOST_CACHE[raw] = cached
    return cached


def _bypass_networks() -> tuple:
    """Parsed ``dashboard_auth_bypass_networks``. Empty tuple ⇒ gate everything.

    Memoised on the raw string rather than at import: the value is read on every
    request, so a bad CIDR must not be able to turn the gate into an exception.
    """
    raw = (settings.dashboard_auth_bypass_networks or "").strip()
    cached = _NETWORK_CACHE.get(raw)
    if cached is not None:
        return cached
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            # Fail CLOSED for this entry: an unparseable CIDR grants nothing.
            logger.warning(f"[dashboard] ignoring unparseable auth-bypass network {part!r}")
    result = tuple(nets)
    _NETWORK_CACHE[raw] = result
    return result


def _ip_in(addr: str, nets: tuple) -> bool:
    """True if ``addr`` parses as an IP inside one of ``nets``."""
    try:
        ip = ipaddress.ip_address((addr or "").strip())
    except ValueError:
        return False
    # A v4 socket reached over a dual-stack listener shows up as ::ffff:127.0.0.1,
    # which is NOT a member of 127.0.0.0/8 until it is unwrapped.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return any(ip in net for net in nets)


def _hostname_only(host: str) -> str:
    """Strip the port (and IPv6 brackets) from a Host header value."""
    host = (host or "").strip()
    if host.startswith("["):                       # [::1]:8050
        end = host.find("]")
        return host[1:end] if end > 0 else host[1:]
    if host.count(":") == 1:                       # 127.0.0.1:8050 — a bare IPv6
        host = host.split(":", 1)[0]               # literal has more than one colon
    return host


def _is_local_request(environ) -> bool:
    """True only when the request provably came from a trusted local browser.

    ⚠ The obvious implementation — ``REMOTE_ADDR is loopback`` — is WRONG here and
    would publish live P&L to the open internet. The ngrok tunnel dials out from
    this PC and forwards the public hostname to 127.0.0.1, so every request from a
    public visitor ALSO arrives with a loopback peer address. Peer address alone
    cannot tell the owner's browser from the whole internet.

    Three independent things must hold, any one of which stops a tunnelled request:

    1. the peer address is in ``dashboard_auth_bypass_networks`` (default: this
       machine) — stops other devices on the LAN/tailnet;
    2. no proxy header is present — the tunnel edge stamps ``X-Forwarded-For``
       and friends on everything it relays, and a client can add such a header
       but cannot remove one;
    3. the ``Host`` header names a trusted address — a loopback/bypass-network IP
       literal, or a name the owner listed in ``dashboard_auth_bypass_hosts``
       (Tailscale MagicDNS). A browser at ``https://<name>.ngrok-free.dev`` sends
       that name through, and it is on nobody's list.

    (2) only works because ``_serve_once`` tells waitress to keep those headers;
    waitress deletes them by default, which would silently collapse this to (1)
    and (3). That coupling is asserted by tests/test_dashboard_auth.py.
    """
    nets = _bypass_networks()
    if not nets:                                   # bypass disabled → gate everything
        return False

    # Key first: environ also holds non-string objects (wsgi.input, wsgi.errors),
    # and truth-testing an arbitrary object before knowing it is a header is how
    # an auth check acquires a way to raise.
    for key in environ:
        if not isinstance(key, str):
            continue
        if key in _PROXY_HEADER_KEYS or key.startswith(_PROXY_HEADER_PREFIXES):
            if environ.get(key):
                return False

    if not _ip_in(environ.get("REMOTE_ADDR", ""), nets):
        return False

    host = _hostname_only(environ.get("HTTP_HOST", "")).lower().rstrip(".")
    if not host:                                   # no Host header ⇒ unprovable ⇒ gated
        return False
    if host in _LOOPBACK_HOSTNAMES:
        return _ip_in("127.0.0.1", nets) or _ip_in("::1", nets)
    if _ip_in(host, nets):                         # a bare IP literal, e.g. 100.90.109.43
        return True
    # A NAME — Tailscale MagicDNS is the reason this exists. Only names the owner
    # listed count; every other name (the public ngrok domain included) is gated.
    for entry in _bypass_hosts():
        if host == entry or (entry.startswith(".") and host.endswith(entry)):
            return True
    return False


# ── online brute-force protection ────────────────────────────────────────────
#
# Security GUARDRAILS, deliberately fixed constants rather than settings (the
# same rule the risk limits follow): a knob that weakens a defence is a knob
# that eventually gets turned.
#
# The limit is GLOBAL, not per-IP, and that is the point. Behind Tailscale
# Funnel every request reaches this process from 127.0.0.1, so a per-IP bucket
# would see one client and protect nothing. The real client address arrives only
# in X-Forwarded-For — a header an attacker sets freely, so keying a rate limit
# on it would hand out a fresh quota per forged value. A global bucket cannot be
# rotated around.
#
# Sizing: 20 failures per 5 minutes is ~5,700 guesses/day. Against the
# 20-character random password this setup ships with, that is not a threat in
# any human timescale; against a weak password nothing here would save you, so
# password strength does the real work and this makes the attempt slow, bounded
# and VISIBLE in the log.
_AUTH_MAX_FAILURES = 20            # failures tolerated inside the window
_AUTH_FAILURE_WINDOW = 300.0       # seconds — the sliding window
_AUTH_LOCKOUT_SECONDS = 300.0      # how long a tripped limit stays tripped

_auth_failures: list = []          # timestamps of recent failed attempts
_auth_lock = __import__("threading").Lock()
_auth_state: dict = {"locked_until": 0.0, "announced": False}


def _auth_locked_for() -> float:
    """Seconds remaining on the lockout, or 0.0 when attempts are allowed."""
    with _auth_lock:
        remaining = _auth_state["locked_until"] - time.time()
        if remaining <= 0 and _auth_state["announced"]:
            _auth_state["announced"] = False
            logger.info("[dashboard] auth lockout expired — accepting attempts again")
        return max(0.0, remaining)


def _auth_record_failure() -> None:
    """Count a failed attempt and trip the lockout once the window fills."""
    now = time.time()
    with _auth_lock:
        _auth_failures.append(now)
        cutoff = now - _AUTH_FAILURE_WINDOW
        while _auth_failures and _auth_failures[0] < cutoff:
            _auth_failures.pop(0)
        if len(_auth_failures) >= _AUTH_MAX_FAILURES and not _auth_state["announced"]:
            _auth_state["locked_until"] = now + _AUTH_LOCKOUT_SECONDS
            _auth_state["announced"] = True
            _auth_failures.clear()
            # CRITICAL, not warning: on a PUBLIC url this is either an attack or
            # a badly broken client, and both are worth waking up to. Logged
            # once per lockout, never per attempt — a crawler would otherwise
            # flood the log, which is why individual 401s stay silent.
            logger.critical(
                f"[dashboard] auth lockout — {_AUTH_MAX_FAILURES} failed password "
                f"attempts within {_AUTH_FAILURE_WINDOW:.0f}s. Rejecting attempts "
                f"for {_AUTH_LOCKOUT_SECONDS:.0f}s. If this was not you, the "
                f"dashboard URL is being probed.")


def _auth_record_success() -> None:
    """A correct password clears the counter — a human who typed it wrong twice
    before getting it right must not drift toward a lockout."""
    with _auth_lock:
        _auth_failures.clear()


def _basic_auth_middleware(inner, username: str, password: str):
    """Wrap a WSGI callable in an HTTP Basic-Auth gate (one shared credential).

    Requests that ``_is_local_request`` proves came from this machine skip the
    prompt. With ``dashboard_auth_bypass_networks`` EMPTY — the shipped
    configuration since 2026-08-16 — that check always returns False, so every
    request presents the password, including this PC's own browser. The bypass
    machinery is kept because it is the difference between "loopback" and "a
    tunnelled request that merely looks like loopback", and turning it back on
    is a one-line config change; leaving it wired means the distinction stays
    tested rather than rotting.

    Failed attempts are rate-limited GLOBALLY (see the guardrails above). The
    ordering matters: the lockout is checked BEFORE the password comparison, so
    a tripped limit caps the number of guesses rather than merely the number of
    successful answers.

    Split out from ``_install_basic_auth`` so the gate can be tested against a
    stub inner app instead of the whole Dash stack — an auth check nobody can
    exercise is exactly the kind of mechanism this project refuses to trust.
    """
    expected = f"{username}:{password}".encode("utf-8")

    def _gate(environ, start_response):
        # A bypass network (none by default) browses without a login; everything
        # else must present the shared password. See _is_local_request for why
        # this is not "is the peer 127.0.0.1" — Funnel and ngrok alike make the
        # whole internet look like 127.0.0.1.
        if _is_local_request(environ):
            return inner(environ, start_response)

        # Ahead of the comparison on purpose: this bounds GUESSES, not answers.
        retry_after = _auth_locked_for()
        if retry_after > 0:
            body = b"Too many failed attempts. Try again shortly.\n"
            start_response("429 Too Many Requests", [
                ("Retry-After", str(int(retry_after) + 1)),
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ])
            return [body]

        header = environ.get("HTTP_AUTHORIZATION", "")
        if header.startswith("Basic "):
            try:
                supplied = base64.b64decode(header[6:].strip())
            except Exception:
                supplied = b""
            # Constant-time: a plain == leaks the password one byte at a time.
            if hmac.compare_digest(supplied, expected):
                _auth_record_success()
                return inner(environ, start_response)
            _auth_record_failure()
        # A missing header is the browser's first, credential-less request — the
        # normal prelude to the password prompt, not a guess, so it is not
        # counted. Only a WRONG credential is.
        #
        # No per-request log here on purpose: a crawler on a public URL would
        # otherwise flood the log with one line per probe. The lockout logs once.
        start_response("401 Unauthorized", [
            ("WWW-Authenticate", 'Basic realm="LLM Trader Monitor", charset="UTF-8"'),
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(_DENY_BODY))),
            ("Cache-Control", "no-store"),
        ])
        return [_DENY_BODY]

    return _gate


def _install_basic_auth() -> bool:
    """Install the gate on the live server. True if the gate is ON.

    Installed at IMPORT time, not in ``run()``, so every entry point is covered —
    including ``server`` being handed to an external WSGI host, which would
    otherwise bypass a gate installed only on our own serve path.

    It wraps ``wsgi_app`` rather than using a Flask ``before_request`` hook so it
    sits in front of EVERYTHING Flask serves: the page, the Dash callback XHRs,
    and the static component bundles. A gate with a hole in it is worse than no
    gate, because it looks closed.

    Empty ``dashboard_auth_password`` = no gate. That is the right default for a
    loopback-only dashboard, and it is why the tunnel launcher probes for a 401
    instead of trusting that this ran.
    """
    password = (settings.dashboard_auth_password or "").strip()
    if not password:
        return False
    username = (settings.dashboard_auth_username or "").strip()
    app.server.wsgi_app = _basic_auth_middleware(app.server.wsgi_app, username, password)
    return True


AUTH_ENABLED = _install_basic_auth()


# Routes whose response defines WHICH VERSION of the app the browser is running.
# The index page lists the asset/bundle URLs; the layout is the page structure;
# the dependency graph wires the callbacks. A stale copy of any one of them
# pins the viewer to an old dashboard.
_VERSION_DEFINING_PREFIXES = ("/_dash-layout", "/_dash-dependencies",
                              "/_reload-hash", "/_favicon.ico")


def _install_cache_headers() -> None:
    """Make every client run the CURRENT dashboard, with no manual refresh.

    Nothing here sent cache directives, so each browser applied its own
    heuristic — and phones cache hardest. The result: a deploy landed on the
    server while a phone kept rendering the previous version indefinitely, with
    no way to tell from the server side (the logs show a normal 200 for a page
    the viewer never sees). Only a private tab or a ``?v=`` query string broke
    it, which is not something anyone should have to remember.

    Three classes, by what the response actually is:

    * **version-defining** (the index page, ``_dash-layout``,
      ``_dash-dependencies``) → ``no-store``. Small (~4 KB each) and fetched
      once per page load, so forbidding the cache outright costs nothing
      measurable and is the only setting that GUARANTEES freshness. This is
      also what keeps the DATA current: ``serve_layout`` is a function, so a
      reload rebuilds it against the latest run.
    * **component suites** (the ~MBs of React/Plotly bundles) → cached for a
      year as ``immutable``. Their URLs already carry the package version, so a
      library upgrade changes the URL; caching them hard is what keeps the page
      fast on a phone despite the above.
    * **assets** (our ``style.css``, the favicon) → ``no-cache``, i.e. "you may
      keep a copy but you must revalidate". Dash already appends ``?m=<mtime>``
      so the URL changes whenever the file does; revalidation is a 304 costing
      a few hundred bytes and removes the last way to be served a stale
      stylesheet.

    Pinned by ``tests/test_dashboard_cache_headers.py`` — a regression here is
    invisible from every server-side surface, which is exactly the class this
    project refuses to leave to review.
    """
    from flask import request

    @app.server.after_request
    def _set_cache_headers(response):
        path = request.path or ""
        if path.startswith("/_dash-component-suites/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-cache"
        elif path == "/" or path.startswith(_VERSION_DEFINING_PREFIXES):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"          # HTTP/1.0 proxies
            response.headers["Expires"] = "0"
        return response


_install_cache_headers()

# The system font stack — one typography for UI, tables and charts (figures.py
# uses the same stack). tabular-nums on cells comes from assets/style.css.
_FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

_TABLE_KW = dict(
    page_size=25,           # rows shown per page (applies to every table)
    sort_action="native",   # click a column header to sort — toggles ascending → descending → off
    sort_mode="multi",      # shift-click additional headers to sort by several columns
    tooltip_delay=200,      # ms before a header tooltip appears
    tooltip_duration=None,  # keep the explanation visible until the mouse leaves
    style_table={"overflowX": "auto"},
    style_cell={
        "fontFamily": _FONT_STACK, "fontSize": 12.5, "padding": "6px 10px",
        "textAlign": "left", "whiteSpace": "normal", "height": "auto",
        "maxWidth": 460, "border": "0", "borderBottom": "1px solid #eef2f7",
        "color": "#1e293b",
    },
    style_header={
        "backgroundColor": "#f8fafc", "fontWeight": "600", "cursor": "help",
        "fontSize": 11.5, "color": "#475569", "textTransform": "uppercase",
        "letterSpacing": "0.03em", "border": "0",
        "borderBottom": "1px solid #e2e8f0",
    },
    style_data={"backgroundColor": "white"},
)


def _kpi(label: str, value: str, color: str = "#111827", tooltip: str = "") -> html.Div:
    """A stat tile. ``tooltip`` (if given) shows as a hover explanation; the label
    gets a dotted underline + help cursor to advertise that it's there. Pass a
    semantic ``color`` (figures.POS/NEG) only when the value carries a sign the
    reader should see at a glance; the default renders in primary ink."""
    label_style = {}
    if tooltip:
        label_style["borderBottom"] = "1px dotted #cbd5e1"
    value_style = {} if color in ("#111827", None) else {"color": color}
    return html.Div(
        [
            html.Div(label, className="kpi-label", style=label_style),
            html.Div(value, className="kpi-value", style=value_style),
        ],
        title=tooltip,
        className="kpi" + (" kpi--help" if tooltip else ""),
    )


def _h3(text: str, tooltip: str = "") -> html.H3:
    """Section heading with an optional hover explanation."""
    return html.H3(text, title=tooltip or None,
                   className="section-h" + (" section-h--help" if tooltip else ""))


def _health_banner():
    """Prominent banners for the latest run: failed data sources (red) and a
    price-provenance alert (amber) when a new trade entered far from the run
    snapshot. Returns an empty Div when all good."""
    blocks = []

    try:
        failures = data.latest_run_failures()
    except Exception as e:
        logger.debug(f"[dashboard] health banner skipped: {e}")
        failures = []
    if failures:
        items = []
        for f in failures:
            lbl = f.get("source_label") or "?"
            err = f.get("error")
            items.append(f"{lbl} — {err}" if err else lbl)
        blocks.append(html.Div(
            [
                html.B(f"⚠ {len(failures)} data source(s) failed in the latest run"),
                html.Div(" · ".join(items), className="banner-detail"),
            ],
            className="banner banner--error",
        ))

    # Feeds that WENT DARK: historically-populated sources whose recent runs are
    # all empty — invisible to the failed-sources banner (they return ok=true)
    # and to any single run's status (event-driven feeds are often legitimately
    # empty once). See data_quality.compute_dark_sources.
    try:
        dark = data.dark_sources()
    except Exception as e:
        logger.debug(f"[dashboard] dark-sources banner skipped: {e}")
        dark = []
    if dark:
        items = [f"{d['source']} ({d['prior_empty_pct']:.0f}% → 100% empty over the "
                 f"last {d['recent_empty']} fetches)" for d in dark]
        blocks.append(html.Div(
            [
                html.B(f"📡 {len(dark)} data feed(s) went dark"),
                html.Div(" · ".join(items) + " — see the Data Quality tab.",
                         className="banner-detail"),
            ],
            className="banner banner--warn",
        ))

    try:
        pp = (data.latest_gate_diag() or {}).get("price_provenance")
    except Exception as e:
        logger.debug(f"[dashboard] provenance banner skipped: {e}")
        pp = None
    if pp and pp.get("down"):
        blocks.append(html.Div(
            [
                html.B("🔔 Price provenance alert"),
                html.Div((pp.get("message") or "") + " — see the Execution tab.",
                         className="banner-detail"),
            ],
            className="banner banner--warn",
        ))

    return html.Div(blocks) if blocks else html.Div()


def _pct(x, signed: bool = False) -> str:
    if x is None:
        return "–"
    try:
        return f"{x:+.2f}%" if signed else f"{x:.1f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_et(iso_str) -> str:
    """ISO 8601 (any timezone) → ``'YYYY-MM-DD HH:MM'`` in US/Eastern.

    Timestamps are persisted in UTC (or with a raw offset); we convert to Eastern
    and drop the ``+00:00`` tail — the column header carries the ``(ET)`` label.
    Returns the input unchanged if it can't be parsed.
    """
    if not iso_str:
        return ""
    try:
        return datetime.fromisoformat(str(iso_str)).astimezone(ET).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso_str)


_INT = Format(precision=0, scheme=Scheme.fixed)
_NUM1 = Format(precision=1, scheme=Scheme.fixed)
_NUM2 = Format(precision=2, scheme=Scheme.fixed)
_NUM3 = Format(precision=3, scheme=Scheme.fixed)
_NUM4 = Format(precision=4, scheme=Scheme.fixed)


def _columns(spec):
    """Build DataTable column dicts from ``(id, label, format, tooltip)`` tuples.

    A non-None format marks the column numeric, so it renders cleanly *and* sorts
    numerically rather than lexicographically ("9" before "10"). The tooltip is
    consumed by ``_header_tooltips``, not here.
    """
    cols = []
    for cid, label, fmt, _tip in spec:
        col = {"id": cid, "name": label}
        if fmt is not None:
            col["type"] = "numeric"
            col["format"] = fmt
        cols.append(col)
    return cols


def _header_tooltips(spec) -> dict:
    """column id → hover explanation, for a DataTable's ``tooltip_header``."""
    return {cid: tip for cid, _label, _fmt, tip in spec if tip}


# Friendly column specs — order here is the on-screen column order.
# Each tuple is (id, header label, numeric format or None, hover explanation).
_REC_COL_SPEC = [
    ("ticker", "Ticker", None, "The stock or ETF symbol."),
    ("action", "Action", None, "The call: BUY, SELL, HOLD or WATCH. Only BUY/SELL are actionable (paper-traded)."),
    ("direction", "Direction", None, "Directional lean behind the call — BULLISH or BEARISH."),
    ("confidence", "Confidence %", _INT, "Model confidence, 0–100%. A BUY/SELL is actionable only above the regime-adjusted threshold — baseline 85% at NEUTRAL (RISK_ON 79 / CAUTION 87 / RISK_OFF 89 / PANIC 95, +6pp off-RTH, +10pp overnight) — AND with ≥2 agreeing signal sources (Gate 1b). Note the SCALE moved with the rank basis: raw confidence now divides by rank_raw_confidence_scale (0.642) rather than the old absolute 0.5, and the ML-combine arm uses its own divisor again, so confidence is comparable WITHIN a combine engine, not across eras."),
    ("time_horizon", "Horizon (LLM)", None, "The LLM's intended holding window (SHORT-TERM / SWING / POSITION). Capped at trade time to the mechanical edge horizon — the LLM may confirm or shorten it, never lengthen."),
    ("target_horizon", "Edge horizon", None, "Horizon synthesis: the cost-aware holding horizon (30m/3h/6h/1d/3d/1w/2w/1m) whose net-of-cost expected gross return is highest, from each method's MEASURED per-horizon IC (sign-aware). Blank when the IC panel is too thin or horizon synthesis is off."),
    ("horizon_net_edge_pct", "Net edge %", _NUM2, "Expected GROSS return at the edge horizon minus the round-trip cost hurdle. Positive = the edge clears costs at that horizon; ≤0 means no horizon is worth trading (prefer WATCH/HOLD)."),
    ("expected_move_pct", "Exp move %", _NUM2, "Expected FAVOURABLE move (magnitude, gross, pre-cost) at the target horizon — how far the name is expected to travel in the position's direction. The 'upside' the selection maximises."),
    ("upside_score", "Upside", _NUM2, "Selection rank key = conviction (probability of moving that way) × expected move (magnitude) × market-alignment factor. The biggest expected favourable mover in the regime's direction ranks highest; counter-regime names are haircut."),
    ("market_aligned", "Market", None, "Is the position aligned with the macro-regime market direction? aligned = with the regime (beta tailwind), counter = against it (haircut), neutral = no strong regime call. The regime layer owns the market direction; selection amplifies the biggest mover in it."),
    ("shadow_target_horizon", "Shadow horizon", None, "SHADOW (not yet live): horizon from the direction-aware, MARKET-NEUTRAL edge curve — each method weighted by its per-side (bull/bear) skill on returns net of SPY. Compare against 'Edge horizon' to see where direction-conditioning + drift-removal change the call."),
    ("shadow_direction", "Shadow dir", None, "SHADOW: the direction the market-neutral curve favours. When it DISAGREES with 'Direction', a method that is anti-predictive on this side has been flipped — the disagreement is the thing to watch before promoting the shadow curve."),
    ("shadow_horizon_net_edge_pct", "Shadow net %", _NUM2, "SHADOW: market-relative (alpha over SPY) net edge at the shadow horizon, after the cost hurdle. Smaller than 'Net edge %' by construction (market drift removed) — that gap is how much of the live edge was just beta/drift."),
    ("actionable", "Actionable", None, "TRUE = passed the confidence + sources-agreeing gate and was paper-traded. FALSE = monitor only."),
    ("dominant_method", "Top Method", None, "The signal method that contributed most to this call (e.g. news, technical, momentum)."),
    ("type", "Type", None, "Asset class — STOCK, ETF or COMMODITY."),
    ("llm_provider", "LLM", None, "Which model synthesised the recommendation (e.g. Claude Haiku, DeepSeek)."),
    ("generated_at", "Generated (ET)", None, "When the recommendation was produced, in US/Eastern time."),
    ("rationale", "Rationale", None, "The model's plain-English reasoning for the call."),
]

_TRADE_COL_SPEC = [
    ("ticker", "Ticker", None, "The stock or ETF symbol."),
    ("action", "Action", None, "BUY (long) or SELL (short) — how the position was opened."),
    ("direction", "Direction", None, "BULLISH (long) or BEARISH (short)."),
    ("entry_dt", "Entry (ET)", None, "When the position was opened, in US/Eastern time."),
    ("session", "Session", None, "US-market session the position was ENTERED in: rth (09:30–16:00 ET), premarket (04:00–09:30), afterhours (16:00–20:00), or overnight (20:00–04:00). Pre-market + after-hours make up the 'extended' session: those entries are sized down and bear the wider extended spread in their return."),
    ("entry_price", "Entry $", _NUM2, "Fill price at entry (the bid-ask spread is applied in the return, not here)."),
    ("filled_qty", "Shares", None, "Shares actually filled at IBKR (real-executions view only)."),
    ("exit_dt", "Exit (ET)", None, "When the position was closed, in US/Eastern time. Blank while still open."),
    ("exit_price", "Exit $", _NUM2, "Fill price at exit. Blank while the position is open."),
    ("held", "Held", None, "Wall-clock holding time: days + hours (e.g. 2d 5h), hours (6h), or minutes (45m) for the freshest entries. Open positions measure entry → now; closed ones entry → exit. Legacy date-only rows fall back to the trading-days count (Nd)."),
    ("target_horizon", "Target horizon", None, "Horizon synthesis: the cost-aware holding horizon the position was opened for (e.g. 6h, 1w), capped to the LLM's call. Drives the matched exit time-stop — once held past this window the position must stay strongly confirmed to keep running. Blank for trades opened before horizon synthesis."),
    ("return_pct", "Return %", _NUM2, "Spread-adjusted % return. For OPEN positions this is the live mark-to-market — 'what if you closed right now'."),
    ("position_size_multiplier", "Size ×", _NUM2, "Capital weight after the whole sizing chain. Confidence contributes a CONTINUOUS ramp capped at 1.5×, not the old 1.0/1.5/2.0 tiers: the legacy ramp's span above 1.0× is compressed by confidence_size_span (0.5) because entry confidence measured nearly uninformative about outcomes, so paying a full 2.0× for it was sizing on noise. Agreement BREADTH is the evidence-backed conviction signal that replaced the surrendered span. Then: expected-edge blend × predictability tilt → regime haircut → correlation haircut → extended/overnight multiplier."),
    ("filled_notional_usd", "Notional $", _NUM2, "Actual dollars at risk: filled shares × average fill price (real-executions view only)."),
    ("status", "Status", None, "OPEN (held, live mark) or CLOSED (realised)."),
    ("exit_reason", "Exit reason", None, "Why the position closed. LIVE rules: llm_signal_flipped (the opener now calls the opposite direction), horizon_expired (held past its target-horizon window without strong re-confirmation — the matched exit), trailing_stop, adverse_stop, macro_regime_exit, ml_exit (ml_arm trades only), method_horizon, edge_decay, intraday_reversal, and the signal-decay backstop for legacy/rule-opened trades. RETIRED rules still present in history: llm_confidence_loss (OFF — the only exit rule the post-exit forward returns condemned: +1.50/+2.24% left behind at 1d/5d, 62% of those exits kept running) and mechanical_exit (OFF since 2026-08-02 — anti-predictive consensus, and it could not fire at its threshold anyway). Blank while open."),
    ("broker_entry", "IBKR entry", None, "Did the entry order really execute at the broker? ✓ filled (shares) · ⏳ working / partial · ↻ re-anchoring (tick-scoped cancel; resubmits at the current mark) · ✕ cancelled · ✗ rejected/failed · – never sent (broker off, duplicate twin, sizing skip, or pre-broker history). Simulated view only — the IBKR view contains only filled orders by construction."),
    ("broker_exit", "IBKR exit", None, "Same for the closing order. ⏳ pending = the ledger closed the trade and the exit goes out on the next sync. Blank while the position is open."),
]

# Entry Performance table — header explanations (table is built inline below).
_METHOD_HEADER_TIPS = {
    "Method": "The signal method (e.g. news sentiment, technical, momentum) — or an LLM engine row: 'Synthesis LLM' made the final BUY/SELL call, 'Sentiment LLM' scored the per-ticker news (run-dominant engine).",
    "Win rate %": "ABSOLUTE. Method rows — solo simulation: for each closed trade, what if ONLY this method had decided the direction? LLM rows — share of the engine's recommended trades (executed or not) currently positive. Cannot separate 'the signal works' from 'the market went up' — compare against Rel win %.",
    "Rel win %": "MARKET-RELATIVE: share of the method's calls where the stock beat SPY IN THE DIRECTION CALLED, over the whole simulated-trade panel (thousands of observations, versus the few hundred attributed ledger trades behind the absolute column). This is the basis METHOD WEIGHTING now uses, because weighting is a signal-quality decision and beta is a confound there. Sizing and P&L stay absolute — the book is outright long/short, so alpha you cannot capture must not size it.",
    "vs base": "Rel win % minus the MEASURED baseline, which is NOT 50%: the cap-weighted index beats its typical constituent, so the median stock is market-relative-negative. The exact bar is re-measured from the panel and printed above this table — it MOVES, so read it there rather than remembering a number. Positive here means the method genuinely adds something over that bar.",
    "Rel n": "Observations behind Rel win % — the whole scored panel, so typically thousands versus the dozens or low hundreds behind the absolute Win rate %.",
    "Trades": "Method rows: closed trades this method had a view on (|score| ≥ 0.10). LLM rows: every BUY/SELL the engine recommended — actionable or not, executed or simulated — deduped to its last call per ticker per day.",
    "Avg return %": "Average % return across those trades.",
}

# Decision-funnel table — header explanations (pipeline stage evaluation).
_STAGE_HEADER_TIPS = {
    "Stage": "One step of the decision pipeline, in execution order: the mechanical Aggregator, the LLM Synthesis stream it feeds, then each actionable gate — Gate 1 regime confidence threshold, Gate 1b agreement floor (≥2 independent sources, mechanical since 2026-07-20), Gate 2 PANIC BUY-block, Gate 3 earnings blackout, Gate 4 liquidity floor (the $5/$5M TRADE floor — a name below it is still fully scored, just never traded), and Gate 5 overextension / anti-chase (BUY-only, defers a BUY that already ran >12% over 5 bars; SELLs are never gated). '→ past Gate k' = the calls still alive after that gate; '✂ Gate k drops' = exactly what that gate discarded. A drop is attributed to the FIRST gate that rejects it, so the rows partition cleanly. Compare a drops row against its survivor row: drops performing WORSE = the gate is filtering losers (working); drops performing BETTER = the gate is throwing away winners.",
    "Trades": "Directional calls in that stage's stream, deduped to the last call per ticker per day. The shrink from row to row is each gate's real selectivity.",
    "Win rate %": "Share of the stage's calls currently positive, scored as pseudo-trades: snapshot price at call time → latest cached close, through the real cost model — every call counts, not just the ones that became ledger trades.",
    "Avg return %": "Average forward % return across the stage's calls on the same pseudo-trade basis. A gate earns its place when this rises from the pre-gate row to the post-gate row.",
}

# Macro Performance table — header explanations (aggregated decision layers).
_MACRO_HEADER_TIPS = {
    "Layer": "The aggregated decision layer being judged: 'LLM Synthesis' = the final BUY/SELL caller (all engines combined; the per-engine split is in the Model Evaluation table below), 'Aggregator' = the mechanical combined signal (the weighted blend of all method scores), or 'Bundle · X' = one method family (e.g. Technical, Options) voting by the sign of its summed scores. Each layer is scored on its OWN full stream of directional calls.",
    "Win rate %": "Share of that layer's directional calls currently positive — counting EVERY call it made (actionable or not, executed or not), not just the trades that survived the gates.",
    "Trades": "Number of directional calls the layer made, deduped to its last call per ticker per day (same rule as the LLM rows below).",
    "Avg return %": "Average forward % return across those calls: snapshot price at the call → latest cached close, net of the modeled one-way cost (so a brand-new call starts slightly negative, like a real position).",
}


# ── LLM model usage (Entry Performance tab → "LLM models used" section) ───────
# Exact model ids per provider. Sources of truth in the code:
#   synthesis Claude   → settings.analyst_model
#   synthesis DeepSeek → claude_analyst._DEEPSEEK_ANALYST_MODEL  ("deepseek-v4-flash")
#   sentiment DeepSeek → sentiment.DEEPSEEK_MODEL                ("deepseek-v4-flash")
#   sentiment Claude   → sentiment.HAIKU_MODEL                   ("claude-haiku-4-5-20251001")
_PROVIDER_LABEL = {
    "anthropic": "Anthropic (Claude)", "deepseek": "DeepSeek", "qwen": "Qwen",
    "rule-based": "Rule-based", "none": "—", "": "—",
}
_SENTIMENT_MODEL = {
    "deepseek": "deepseek-v4-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "qwen": settings.qwen_model,
    "none": "(none — cached / no LLM call)",
}


def _synthesis_model(provider) -> str:
    """Exact model id that produced the final synthesis for a given provider."""
    p = (provider or "").lower()
    if p == "anthropic":
        return settings.analyst_model           # the configured Claude model
    if p == "deepseek":
        return "deepseek-v4-flash"               # DeepSeek V4-Flash analyst fallback
    if p == "qwen":
        return settings.qwen_model                # Qwen3.7-Max — 2026-07-11 primary
    if p == "rule-based":
        return "rule-based (no LLM)"
    return "—"


def _parse_sentiment_summary(summary):
    """'deepseek×40, anthropic×2' → [('deepseek', 40), ('anthropic', 2)]."""
    if not summary:
        return []
    out = []
    for tok in str(summary).split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, sep, cnt = tok.partition("×")
        try:
            n = int(cnt) if sep else 0
        except ValueError:
            n = 0
        out.append((name.strip().lower(), n))
    return out


def _models_used_rows(runs) -> list:
    """Aggregate which exact LLMs ran across all recorded runs (synthesis + sentiment)."""
    from collections import defaultdict
    syn = defaultdict(int)
    sen_runs, sen_calls = defaultdict(int), defaultdict(int)
    for r in runs.itertuples():
        sp = getattr(r, "llm_synthesis_provider", None)
        syn[(_synthesis_model(sp), _PROVIDER_LABEL.get((sp or "").lower(), sp or "—"))] += 1
        for prov, n in _parse_sentiment_summary(getattr(r, "llm_sentiment_provider", None)):
            key = (_SENTIMENT_MODEL.get(prov, prov), _PROVIDER_LABEL.get(prov, prov.title()))
            sen_runs[key] += 1
            sen_calls[key] += n
    rows = []
    for (model, api), n in sorted(syn.items(), key=lambda kv: -kv[1]):
        rows.append({"Role": "Synthesis", "Model": model, "API": api, "Runs": n, "Calls": "—"})
    for key, n in sorted(sen_runs.items(), key=lambda kv: -sen_calls[kv[0]]):
        model, api = key
        rows.append({"Role": "Sentiment", "Model": model, "API": api, "Runs": n, "Calls": sen_calls[key]})
    return rows


_MODELS_HEADER_TIPS = {
    "Role": "Synthesis = the model that made the final BUY/SELL/HOLD/WATCH call. Sentiment = the model that scored per-ticker news.",
    "Model": "The exact model id that ran (including DeepSeek or rule-based fallbacks).",
    "API": "The provider behind the model.",
    "Runs": "How many recorded pipeline runs used this model in this role.",
    "Calls": "Sentiment only — total per-ticker LLM calls made with this model across runs.",
}


def _safe(render):
    """Render a tab body, surfacing data errors inline instead of crashing the page."""
    try:
        return render()
    except FileNotFoundError:
        return html.Div("No database yet. Run the pipeline (or `python -m src.db.migrate`) first.",
                        style={"padding": 20, "color": "#dc2626"})
    except Exception as e:  # keep the dashboard alive on any data hiccup
        logger.warning(f"[dashboard] tab render failed: {e}")
        return html.Div(f"Could not load data: {e}", style={"padding": 20, "color": "#dc2626"})


def _header() -> html.Div:
    """Slim brand bar: identity on the left, the latest-run status chip on the
    right. The chip is deliberately honest about staleness — a run older than
    ~4 h turns it amber, which is how a silently dead scheduler becomes visible
    from the phone without opening a single tab."""
    chip = None
    try:
        info = data.latest_run_info()
    except Exception:
        info = None
    if info:
        started = info.get("started_at")
        stale = False
        try:
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(str(started))).total_seconds() / 3600.0
            stale = age_h > 4.0
        except (TypeError, ValueError):
            pass
        bits = [f"Last run {_fmt_et(started)} ET"]
        regime = info.get("macro_regime") or ""
        mode = info.get("market_mode") or ""
        if regime or mode:
            bits.append(" / ".join(b for b in (regime, mode) if b))
        chip = html.Div(
            [html.Span(className="dot dot--warn" if stale else "dot dot--ok"),
             html.Span("  ·  ".join(bits))],
            className="run-chip" + (" run-chip--stale" if stale else ""),
            title=("The most recent pipeline run. Amber = older than 4 hours — "
                   "check that the scheduler is alive." if stale
                   else "The most recent pipeline run and its macro regime / market mode."),
        )
    return html.Div(
        [
            html.Div([
                html.Div("▲", className="brand-mark"),
                html.Div([
                    html.Div("LLM Trader", className="brand-name"),
                    html.Div("Monitor · DuckDB-backed, read-only", className="brand-sub"),
                ]),
            ], className="brand"),
            chip or html.Div(),
        ],
        className="app-header",
    )


def serve_layout() -> html.Div:
    """Build the page fresh on every load; tab content renders LAZILY.

    Each ``dcc.Tab`` holds its own EMPTY container div; a per-tab callback fills
    that container the first time the tab becomes active and leaves it alone
    afterwards (``PreventUpdate`` when it's not the active tab or already has
    content — so a revisit is instant and nothing recomputes). The initial page
    therefore ships only the chrome + the active tab, instead of building all
    six tabs server-side per load — the difference between a sub-second landing
    and a multi-minute one whenever any tab's data is cold.

    ⚠ Do NOT replace this with a single shared content container swapped on
    ``tabs.value`` — that pattern was tried and every tab showed the first-
    rendered content in the browser even though the server returned the right
    payload. Per-tab containers avoid it structurally: content never moves
    between containers and component ids stay put, exactly like the old
    all-embedded layout, just filled on demand.

    Being a function, the layout is rebuilt per page load, so a long-running
    dashboard always reflects the latest pipeline run without a restart.
    """
    return html.Div(
        className="app-shell",
        children=[
            _header(),
            _health_banner(),
            dcc.Tabs(
                id="tabs", value="rationale",
                className="app-tabs", parent_className="app-tabs-wrap",
                persistence=True, persistence_type="session",  # keep the selected tab across reloads
                children=[
                    dcc.Tab(label=label, value=value,
                            className="app-tab", selected_className="app-tab--selected",
                            children=dcc.Loading(
                                html.Div(id=f"tab-{value}", className="tab-body"),
                                color="#2a78d6"))
                    for value, label, _render in _TAB_SPEC
                ],
            ),
        ],
    )


app.layout = serve_layout


# ── Tab 1: Recommendations & Rationale ─────────────────────────────────────

_RUN_DROPDOWN_LIMIT = 250   # ~2 weeks of 30-min ticks; 1,200+ options made the
                            # dropdown unusable and bloated the tab payload.


def _rationale_tab():
    runs = data.runs_df()
    if runs.empty:
        return html.Div("No runs recorded yet. Run the pipeline first.", style={"padding": 20})
    recent = runs.head(_RUN_DROPDOWN_LIMIT)
    options = [
        {
            "label": f"{_fmt_et(getattr(r, 'started_at', None))} ET"
                     f"   ·   {getattr(r, 'market_mode', None) or '–'} / {getattr(r, 'macro_regime', None) or '–'}"
                     f"   ·   LLM: {getattr(r, 'llm_synthesis_provider', None) or '–'}",
            "value": r.run_id,
        }
        for r in recent.itertuples()
    ]
    trimmed = len(runs) - len(recent)
    return html.Div([
        html.Div(
            [html.Label("Run", title="Pick which pipeline run to inspect. Each entry is one analysis run, shown as its Eastern start time · market mode / macro regime · the LLM used.",
                        className="filter-label"),
             dcc.Dropdown(id="run-select", options=options, value=options[0]["value"],
                          clearable=False, style={"width": 560}),
             html.Span(f"showing the {len(recent)} most recent of {len(runs)} runs",
                       className="filter-note") if trimmed > 0 else html.Span()],
            className="filter-item",
        ),
        dcc.Loading(html.Div(id="rationale-body")),
    ])


@app.callback(Output("rationale-body", "children"), Input("run-select", "value"))
def _rationale_body(run_id):
    src = data.run_sources_df(run_id)
    recs = data.recommendations_df(run_id)

    ok_n = int(src["ok"].sum()) if not src.empty else 0
    chips = [
        html.Span(
            f"{'✓' if bool(r['ok']) else '✗'} {r['source_label']}",
            title=(r.get("error") or ("Succeeded — data fetched this run." if bool(r["ok"]) else "Failed.")),
            style={"display": "inline-block", "margin": "2px 10px 2px 0", "cursor": "help",
                   "color": figures.POS if bool(r["ok"]) else figures.NEG, "fontSize": 13},
        )
        for r in src.to_dict("records")
    ]

    recs_disp = recs.copy()
    if not recs_disp.empty:
        recs_disp["confidence"] = (recs_disp["confidence"].astype(float) * 100).round(0).astype("Int64")
        recs_disp["generated_at"] = recs_disp["generated_at"].map(_fmt_et)
        # Row id = ticker so an active-cell click resolves to the ticker robustly
        # (survives native sort / filter / pagination) for the review-timeline plot.
        recs_disp["id"] = recs_disp["ticker"]

    run_meta = data.run_row(run_id)
    syn = (run_meta["llm_synthesis_provider"] if run_meta is not None else None) or "–"
    sent = (run_meta["llm_sentiment_provider"] if run_meta is not None else None) or "–"

    return html.Div([
        html.Div(f"LLM — synthesis: {syn}   ·   sentiment: {sent}",
                 style={"color": "#374151", "fontSize": 14, "marginBottom": 12}),
        _h3(f"APIs used this run  ·  {ok_n}/{len(src)} succeeded",
            "Each external data source the pipeline called this run — green ✓ succeeded, red ✗ failed. Hover a chip for the error or status."),
        html.Div(chips or "No source records.", style={"marginBottom": 18}),
        _h3(f"Recommendations  ·  {len(recs_disp)} shown",
            "Every BUY/SELL/HOLD/WATCH the model produced this run. Green-tinted rows are actionable (paper-traded). Hover a column header for its definition. Click any row to chart that ticker's hold-review confidence over time below."),
        dash_table.DataTable(
            id="rec-table",
            data=recs_disp.to_dict("records"),
            columns=_columns(_REC_COL_SPEC),
            tooltip_header=_header_tooltips(_REC_COL_SPEC),
            filter_action="native",
            style_data_conditional=[
                {"if": {"filter_query": "{actionable} = true"}, "backgroundColor": "#ecfdf5"},
                {"if": {"filter_query": "{action} = BUY", "column_id": "action"}, "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{action} = SELL", "column_id": "action"}, "color": figures.NEG, "fontWeight": "bold"},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "ticker"}, "fontWeight": "bold"},
                {"if": {"column_id": "rationale"}, "minWidth": 260, "maxWidth": 520},
            ],
            **_TABLE_KW,
        ),
        dcc.Loading(html.Div(id="rec-review-plot", style={"marginTop": 16})),
    ])


@app.callback(Output("rec-review-plot", "children"), Input("rec-table", "active_cell"))
def _rec_review_plot(active_cell):
    """Click a recommendation row → chart that ticker's opener-pinned hold-review
    confidence over time, with price + entry/exit decisions, so deterioration →
    direction-change is visible. ``row_id`` is the ticker (set in _rationale_body)."""
    if not active_cell:
        return html.Div(
            "↑ Click any recommendation row to chart its hold-review confidence over time "
            "(the per-tick re-judgment by the engines that opened it) against price and the "
            "entry/exit decisions.",
            style={"color": "#6b7280", "fontStyle": "italic", "padding": "8px 2px"})
    ticker = active_cell.get("row_id")
    if not ticker:
        return html.Div()
    return _safe(lambda: _review_timeline_section(ticker))


def _review_timeline_section(ticker: str):
    reviews = data.trade_reviews_df(ticker)
    trades = data.trades_for_ticker(ticker)
    if reviews is None or reviews.empty:
        return html.Div(
            f"No hold-review history recorded for {ticker} yet. Only LLM-opened positions accrue "
            "it, and only from this feature's first run onward — it fills in tick by tick while a "
            "position is held.",
            style={"color": "#6b7280", "padding": "8px 2px"})
    return html.Div([
        _h3(f"{ticker} — hold-review confidence over time",
            "Each point is one tick's re-judgment of this position by the SAME synthesis + sentiment "
            "engines that opened it, on fresh news + prices (so it's an apples-to-apples vs the entry "
            "confidence). Marker colour = the review's action (green BUY / red SELL / grey HOLD). "
            "Dashed line = entry confidence; dotted line = the close floor. ⚠ The floor no longer "
            "triggers an llm_confidence_loss exit on its own (that rule is OFF — it was measured to "
            "leave money behind); its live consumer is the ramped horizon_expired test, so the floor "
            "matters only once a position is held past its target horizon. Grey line = price; triangles = entry, "
            "✕ = exit. Watch whether the confidence sliding toward the floor precedes a colour flip."),
        dcc.Graph(figure=figures.confidence_timeline_fig(reviews, trades)),
    ])


# ── Filter toggles (shared shell) ────────────────────────────────────────────

def _filter_row(label: str, tooltip: str, component_id: str, options, default: str) -> html.Div:
    """One labeled segmented control — the shared shell every filter toggle uses.
    Styling lives in assets/style.css (.seg / .seg-item); the input element is
    a real radio so Dash persistence keeps working."""
    return html.Div(
        [
            html.Label(label, title=tooltip, className="filter-label"),
            dcc.RadioItems(
                id=component_id, options=options, value=default, inline=True,
                persistence=True, persistence_type="session",
                className="seg", labelClassName="seg-item", inputClassName="seg-radio",
            ),
        ],
        className="filter-item",
    )


# ── Time-window toggle (shared by the Entry Performance & Returns tabs) ──────
_WINDOW_OPTIONS = [
    {"label": "1 Week", "value": "7"},
    {"label": "1 Month", "value": "30"},
    {"label": "Inception", "value": "all"},
]


def _window_toggle(component_id: str) -> html.Div:
    """A 1-week / 1-month / inception selector. The tab's metrics and plots
    recompute against trades ENTERED within the chosen window ('Inception' = every
    trade ever). Defaults to Inception so the initial view shows the full book."""
    return _filter_row(
        "Window",
        "Filter the metrics and plots in this tab to trades entered in the last week, the last month, or since inception (every trade).",
        component_id, _WINDOW_OPTIONS, "all")


def _window_days(value):
    """RadioItems value → window_days int, or None for inception (all trades)."""
    return None if value in (None, "all") else int(value)


def _window_label(value) -> str:
    """RadioItems value → human label for tile captions."""
    return {"7": "1 week", "30": "1 month"}.get(str(value), "inception")


# ── Trading-session toggle (RTH / pre-market / after-hours / overnight) ───────
_SESSION_OPTIONS = [
    {"label": "All sessions", "value": "all"},
    {"label": "RTH", "value": "rth"},
    {"label": "Pre-market", "value": "premarket"},
    {"label": "After-hours", "value": "afterhours"},
    {"label": "Overnight", "value": "overnight"},
]


_SESSION_TOGGLE_TITLE = (
    "Filter to trades entered during Regular hours (09:30–16:00 ET), Pre-market "
    "(04:00–09:30), After-hours / post-market (16:00–20:00), or Overnight "
    "(20:00–04:00). The bot trades all four sessions (overnight on the Sun–Thu-night "
    "venue calendar).")


def _session_toggle(component_id: str, title: str = _SESSION_TOGGLE_TITLE) -> html.Div:
    """RTH / pre-market / after-hours / overnight selector. Filters the tab's
    metrics and plots to that US-market session (what "session" means per tab is
    stated in ``title`` — trades are filtered by ENTRY session, exit analyses by
    the review/exit moment, panels by signal-generation time). Pre-market and
    After-hours are the two halves of the coarse 'extended' session."""
    return _filter_row("Session", title, component_id, _SESSION_OPTIONS, "all")


def _session_value(value):
    """RadioItems value → session string ('rth'|'premarket'|'afterhours'|'overnight'),
    or None for all. 'premarket'/'afterhours' are the two halves of 'extended'."""
    return None if value in (None, "all") else value


# ── Direction toggle (long / short / both) ───────────────────────────────────
_DIRECTION_OPTIONS = [
    {"label": "Long + Short", "value": "all"},
    {"label": "Long only", "value": "long"},
    {"label": "Short only", "value": "short"},
]


def _direction_toggle(component_id: str) -> html.Div:
    """Long (BUY) / Short (SELL) / both selector. Filters the tab's metrics and
    plots to positions ENTERED in that direction."""
    return _filter_row(
        "Direction",
        "Filter to LONG positions (BUY entries), SHORT positions (SELL entries), or both.",
        component_id, _DIRECTION_OPTIONS, "all")


def _direction_value(value):
    """RadioItems value → 'long' | 'short', or None for both."""
    return None if value in (None, "all") else value


# ── Asset-type toggle (stocks / ETFs / commodities) ──────────────────────────
_ASSET_OPTIONS = [
    {"label": "All types", "value": "all"},
    {"label": "Stocks", "value": "stock"},
    {"label": "ETFs", "value": "etf"},
    {"label": "Commodities", "value": "commodity"},
]


def _asset_toggle(component_id: str) -> html.Div:
    """Instrument-type selector (Stocks / ETFs / Commodities / all). Filters the
    tab's metrics and plots to trades whose instrument ``type`` matches (the same
    STOCK / ETF / COMMODITY label stored at entry)."""
    return _filter_row(
        "Type",
        "Filter to a single instrument type: individual Stocks, ETFs "
        "(sector / factor / index funds), or Commodities (metals, energy, "
        "agriculture ETFs). 'All types' = every instrument.",
        component_id, _ASSET_OPTIONS, "all")


def _asset_value(value):
    """RadioItems value → 'stock' | 'etf' | 'commodity', or None for all types."""
    return None if value in (None, "all") else value


# ── Method-Performance evidence source (gated ledger vs all-scored panel) ────
_METHOD_SOURCE_OPTIONS = [
    {"label": "Ledger (gated trades)", "value": "ledger"},
    {"label": "All scored tickers (simulated)", "value": "panel"},
]


def _method_source_toggle(component_id: str) -> html.Div:
    """The solo-method table's evidence base. Ledger = only the gate-selected
    trades that actually opened (small, selection-biased). All scored tickers =
    every method's implied BUY/SELL on EVERY scored ticker each run (the
    simulated_trades panel), scored on gross forward returns — thousands of
    observations, unbiased by the trading gates."""
    return _filter_row(
        "Source",
        "Ledger (gated trades): solo-method performance over only the trades the gates let through — apples-to-apples with the real book but a small, selection-biased sample. "
        "All scored tickers (simulated): one simulated trade per NEW directional call a method makes (the run it first called the direction — not one per run/day), scored on GROSS forward returns at the pivot basis + 1d/3d/1w/2w/1m — the unbiased directional-predictiveness view. Honors the Window toggle (by signal date), Session (the session the ENTRY was decided in — sessions partition the trades, so All = their sum), and Direction (the side of the method's call — a positive score is its long call).",
        component_id, _METHOD_SOURCE_OPTIONS, "ledger")


# ── Trade-source toggle (simulated ledger vs actual IBKR fills) ──────────────
_SOURCE_OPTIONS = [
    {"label": "Simulated (model)", "value": "sim"},
    {"label": "IBKR (actual fills)", "value": "broker"},
]


def _source_toggle(component_id: str) -> html.Div:
    """Two books, one toggle. Simulated = the strategy ledger (every decision at
    its decision price through the modeled cost stack). IBKR = only orders that
    actually filled, at real fill prices with real commissions."""
    return _filter_row(
        "Trades",
        "Simulated (model): every decision the strategy made, priced at decision time with modeled spread + commission costs — strategy quality, independent of execution. "
        "IBKR (actual fills): only orders that really filled at the broker, at actual average fill prices with the commissions actually charged — execution reality, no modeled costs. "
        "The gap between the two views is the execution gap: slippage, unfilled or expired orders, and sizing rounding.",
        component_id, _SOURCE_OPTIONS, "sim")


def _usd(x, signed: bool = True) -> str:
    if x is None:
        return "–"
    try:
        return f"${x:+,.2f}" if signed else f"${x:,.2f}"
    except (TypeError, ValueError):
        return str(x)


# ── Tab 2: Entry Performance ────────────────────────────────────────────────

_IC_TOOLTIP = (
    "Spearman rank correlation between a method's score and the forward "
    "close-to-close return, at the PIVOT decision basis plus the 1/5/10-day "
    "monitoring grid. Computed over the persisted signals panel — EVERY scored "
    "ticker each run, not just the few that became trades — so it is unbiased by "
    "the trading gates. "
    "Pick ONE method (or a whole family) above; its three sides then appear as "
    "rows: BUY = the method restricted to its positive/bullish scores, SELL = its "
    "negative/bearish scores, ALL = every non-zero score. A genuinely predictive "
    "sell side shows a POSITIVE IC too (a more-negative score ranking a "
    "more-negative return), so the sides are read the same way. "
    "'Sim win %' = simulated solo win rate (share of non-zero scores whose sign "
    "matched the move); 'Sim ret %' = simulated solo return (mean "
    "sign(score)×forward-return — the gross P&L if that method alone decided the "
    "trade). 'IC std' = standard deviation of the PER-DAY IC and 'ICIR' = "
    "mean(daily IC)/std(daily IC) — the IC's reliability: each signal-day counts "
    "once, so they are NOT inflated by same-day cross-sectional correlation the "
    "way a standard error off the raw n would be (|ICIR| ≳ 0.5 is a stable edge, "
    "≈ 0 is noise); both need several signal-days before they populate. A "
    "persistent NEGATIVE IC is sign-inverted (a logic bug); IC ≈ 0 at large n is "
    "dead weight. 'Views' = scored, non-zero observations. "
    "Deliberately independent of the Direction toggle above (which filters the "
    "TRADE-based tables by entry direction) — here both sides are always shown "
    "together, which is the comparison that matters. Run/forward-return based; "
    "n grows every run — judge nothing on a thin panel.")

_IC_HORIZONS = (1, 5, 10)
# The PIVOT pseudo-horizon leads the fixed grid (2026-08-12): "pv" is the
# DECISION basis (filter / IC weights / states), the fixed columns stay for
# monitoring. (suffix, display) pairs drive both the rows and the headers.
_IC_BLOCKS = (("pv", "pivot"),) + tuple((f"{h}d", f"{h}d") for h in _IC_HORIZONS)

# The three sides, in display order: (side key, row label, accent).
_IC_SIDES = (("ic_buy", "▲ Buy (bullish calls)", figures.POS),
             ("ic_sell", "▼ Sell (bearish calls)", figures.NEG),
             ("ic", "● All calls", "#475569"))

# Aggregate score columns — the headline rows, not per-method scores.
_IC_AGGREGATE = ("combined_score", "cmb_buy", "cmb_sell")


def _ic_method_labels() -> dict:
    """method column → human label, including the aggregate rows."""
    from src.performance.tracker import METHOD_LABELS
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined = buy − sell)"
    labels["cmb_buy"] = "Combined BUY side (bull-camp conviction)"
    labels["cmb_sell"] = "Combined SELL side (bear-camp conviction)"
    return labels


def _ic_family_groups() -> dict:
    """``family name -> [method columns]`` covering EVERY panel score column.

    The 7 information families from ``agreement.METHOD_FAMILIES`` are the real
    grouping — they are what the combine votes by, so "how did the Options
    family do" is a question about the system rather than about a reporting
    bucket. The weighted pool is only part of the panel though (timeframe
    variants, the f_* factors and the panel-first methods have no family), so
    the leftovers are grouped by what they ARE. Every column lands in exactly
    one group; ``tests/test_dashboard_method_explorer.py`` pins that, because a
    method silently missing from the dropdown is unreachable in the UI."""
    from src.analysis.signal_panel import PANEL_SCORE_COLUMNS
    from src.signals.agreement import METHOD_FAMILIES
    from src.db.schema import SIGNAL_FUNDAMENTAL_COLUMNS

    known = set(PANEL_SCORE_COLUMNS)
    groups: dict = {"Aggregate (combined score & camps)":
                    [m for m in _IC_AGGREGATE if m in known]}
    claimed = set(groups["Aggregate (combined score & camps)"])

    for family, members in METHOD_FAMILIES.items():
        present = [m for m in members if m in known and m not in claimed]
        if present:
            groups[f"Family · {family}"] = present
            claimed.update(present)

    def _take(name, predicate):
        present = [m for m in PANEL_SCORE_COLUMNS
                   if m not in claimed and predicate(m)]
        if present:
            groups[name] = present
            claimed.update(present)

    _take("Technical · 30-min candles", lambda m: m.endswith("_30m"))
    _take("Technical · Weekly candles", lambda m: m.endswith("_1w"))
    _take("Fundamentals & corporate actions",
          lambda m: m in set(SIGNAL_FUNDAMENTAL_COLUMNS))
    # Whatever is left: panel-first methods at weight 0 and the additive
    # overlays — scored and IC-tracked, but outside the family vote.
    _take("Panel-first & overlays (not in the family vote)", lambda m: True)
    return groups


def _ic_dropdown_options() -> list:
    """Dropdown options: every family, then every individual method.

    Built from STATIC metadata — no database read — because this runs while the
    tab is being rendered. Touching ``data.signal_ic()`` here would re-impose the
    very cost the lazy explorer exists to avoid."""
    labels = _ic_method_labels()
    options = []
    for family, members in _ic_family_groups().items():
        options.append({"label": f"◆  {family}  ({len(members)})",
                        "value": f"fam:{family}"})
    for family, members in _ic_family_groups().items():
        for m in members:
            options.append({"label": f"      {labels.get(m, m)}", "value": f"m:{m}"})
    return options


def _ic_resolve(selection) -> list:
    """Dropdown value → the method columns it covers ([] when nothing picked)."""
    if not selection:
        return []
    if selection.startswith("fam:"):
        return list(_ic_family_groups().get(selection[4:], []))
    if selection.startswith("m:"):
        return [selection[2:]]
    return []


def _ic_rows(res: dict, methods: list, horizons: list) -> list:
    """One row per (method, side) for the selected methods.

    Sides with NO views are dropped, not shown blank. Two real cases produce
    them — a method that has only ever scored one way genuinely has no opposite
    side, and a freshly epoch-registered scorer has its whole history masked to
    NaN — and in both a row of dashes reads as "measured zero" when the truth is
    "nothing to measure". Dropping them lets the caller's empty-state say so."""
    labels = _ic_method_labels()
    by_side = {}
    for side_key, _lbl, _accent in _IC_SIDES:
        df = res.get(side_key)
        if df is None or getattr(df, "empty", True):
            by_side[side_key] = {}
            continue
        by_side[side_key] = {str(r["method"]): r for _, r in df.iterrows()}

    rows = []
    for m in methods:
        for side_key, side_label, _accent in _IC_SIDES:
            r = by_side.get(side_key, {}).get(m)
            if r is None or not int(r["views"]):
                continue
            row = {"method": labels.get(m, m), "side": side_label,
                   "views": int(r["views"])}
            for sfx, _disp in horizons:
                n = r.get(f"n_{sfx}")
                row[f"n_{sfx}"] = int(n) if pd.notna(n) else None
                for key, digits in (("ic", 3), ("icstd", 3), ("icir", 2),
                                    ("hit", 1), ("simret", 2)):
                    v = r.get(f"{key}_{sfx}")
                    row[f"{key}_{sfx}"] = round(float(v), digits) if pd.notna(v) else None
            rows.append(row)
    return rows


def _ic_table(rows: list, horizons: list) -> dash_table.DataTable:
    """The method-explorer table: rows = method × side, columns = the horizon grid."""
    cols = [{"name": "Method", "id": "method"},
            {"name": "Side", "id": "side"},
            {"name": "Views", "id": "views", "type": "numeric", "format": _INT}]
    for sfx, disp in horizons:
        cols += [
            {"name": f"n@{disp}", "id": f"n_{sfx}", "type": "numeric", "format": _INT},
            {"name": f"IC@{disp}", "id": f"ic_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"IC std@{disp}", "id": f"icstd_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"ICIR@{disp}", "id": f"icir_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"Sim win@{disp} %", "id": f"hit_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"Sim ret@{disp} %", "id": f"simret_{sfx}", "type": "numeric", "format": _NUM2},
        ]
    # Sign colouring on every shown horizon (the old table coloured only the
    # longest, which hid a sign flip across the curve — the thing worth seeing).
    cond = [{"if": {"filter_query": '{side} contains "All calls"'},
             "backgroundColor": "#f8fafc"}]
    for sfx, _disp in horizons:
        for c in (f"ic_{sfx}", f"icir_{sfx}", f"simret_{sfx}"):
            cond += [
                {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c},
                 "color": figures.NEG, "fontWeight": "bold"},
            ]
    return dash_table.DataTable(
        data=rows, columns=cols, style_data_conditional=cond,
        style_cell_conditional=[{"if": {"column_id": "method"}, "fontWeight": "600"},
                                {"if": {"column_id": "side"}, "minWidth": 150}],
        **_TABLE_KW)


def _ic_section():
    """The method explorer: pick one method (or one family) and see its stats.

    Replaces the three always-visible buy/sell/all blocks (2026-08-16). Those
    rendered every panel column three times over five categories — ~15 tables of
    ~30 columns — and, worse, forced ``data.signal_ic()`` before the tab could
    paint at all. Here the section renders a dropdown and nothing else; the
    panel is read only once a selection is made, and the buy/sell split the old
    blocks carried is now three ROWS of the selected method.
    """
    return html.Div([
        _h3("Method explorer — signal IC, win rate and simulated return", _IC_TOOLTIP),
        html.Div(
            "Pick a method or a whole family. Nothing is computed until you do — "
            "the panel join behind these numbers is the most expensive query on "
            "the page.",
            className="section-note"),
        html.Div(
            [
                html.Label("Method", title="Choose one method, or a ◆ family to see all of "
                                           "its members at once. Families are the 7 INFORMATION "
                                           "families the combine votes by; the remaining groups "
                                           "cover the timeframe variants, the fundamental factors "
                                           "and the panel-first methods that sit outside the vote.",
                           className="filter-label"),
                dcc.Dropdown(id="ic-select", options=_ic_dropdown_options(), value=None,
                             placeholder="Select a method or family…", clearable=True,
                             persistence=True, persistence_type="session",
                             style={"flex": 2, "minWidth": 340}),
                dcc.Dropdown(id="ic-horizons",
                             options=[{"label": disp, "value": sfx} for sfx, disp in _IC_BLOCKS],
                             value=[sfx for sfx, _d in _IC_BLOCKS], multi=True,
                             placeholder="Horizons…", persistence=True,
                             persistence_type="session",
                             style={"flex": 1, "minWidth": 220, "marginLeft": 8}),
            ],
            className="filter-item filter-item--grow",
        ),
        dcc.Loading(html.Div(id="ic-body")),
    ])


@app.callback(Output("ic-body", "children"),
              Input("ic-select", "value"), Input("ic-horizons", "value"))
def _ic_body(selection, sel_horizons):
    """Render the picked method/family. Returns the placeholder WITHOUT reading
    the panel when nothing is selected — that guard is what keeps opening the tab
    cheap, so it must stay ahead of the data call."""
    methods = _ic_resolve(selection)
    if not methods:
        return html.Div("↑ Pick a method or a family above to see its IC, win rate "
                        "and simulated return.", className="empty-note")
    return _safe(lambda: _ic_body_render(methods, sel_horizons))


def _ic_body_render(methods: list, sel_horizons):
    horizons = [(sfx, disp) for sfx, disp in _IC_BLOCKS
                if not sel_horizons or sfx in sel_horizons] or list(_IC_BLOCKS)
    res = data.signal_ic()
    rows = _ic_rows(res, methods, horizons)
    if not rows:
        return html.Div(
            f"No scored views yet for this selection. The signals panel holds "
            f"{res.get('panel_rows', 0):,} row(s) across {res.get('tickers', 0):,} "
            f"ticker(s), but this method has none that count — either it has not "
            f"scored anything yet, or its scorer was recently changed and the "
            f"superseded history is masked (see METHOD_SCORER_EPOCH). Both refill "
            f"forward, run by run.",
            className="empty-note")
    return html.Div([
        html.Div(f"{len(methods)} method(s) × up to 3 sides · panel: "
                 f"{res.get('panel_rows', 0):,} rows / {res.get('tickers', 0):,} tickers",
                 className="section-note"),
        _ic_table(rows, horizons),
    ])


_SIM_PERF_TOOLTIP = (
    "Each method's simulated solo ENTRIES — one trade per NEW directional call: the run "
    "where the method first called the direction (its first view, a sign flip, or a "
    "re-emerged call after a gap), NOT one per run/day, so a standing call isn't "
    "pseudo-replicated and the Session buckets are a true partition (All sessions = the "
    "sum of the four sessions). Scored on GROSS close-to-close forward returns from the "
    "entry tick (no costs; the question is directional predictiveness, not net P&L). This "
    "is the unbiased counterpart to the ledger solo table: every scored ticker counts, "
    "not only the gate-selected trades that opened. 'Trades' = the method's entry events. Per horizon (pv/1d/3d/1w/2w/1m): "
    "'n@' = joint observations with a forward return, 'IC@' = Spearman rank correlation "
    "between the method's score and the forward return (ranking skill; a persistent "
    "positive IC is real edge, a persistent negative IC is sign-inverted), 'IC std@' / "
    "'ICIR@' = the IC's reliability — stdev and information-ratio (mean/std) of the "
    "PER-DAY IC, where each signal-day counts once so they aren't inflated by same-day "
    "cross-sectional correlation (|ICIR| ≳ 0.5 is stable, ≈ 0 is noise; need several days), "
    "'Win@ %' = "
    "share of the method's solo calls that were directionally right, 'Ret@ %' = mean "
    "signed forward return. Forward returns come from the OHLCV cache, so 1w/2w/1m fill "
    "in only after a post-close cache warm. Use the Horizons/Metrics pickers above to "
    "trim the columns. Run-based; honors the Window toggle by signal date. A method "
    "predictive of direction shows IC > 0 / Win > 50 PERSISTING across horizons; judge "
    "nothing on a thin n.")

# "pv" first — the H/L pivot pseudo-horizon is the decision basis
# (2026-08-13 standardization); the fixed grid stays as monitors. Derived from
# `data.PANEL_HORIZONS` so the columns rendered here and the columns actually
# COMPUTED there can never drift apart (the intraday 30m/3h/6h were dropped
# 2026-08-15 — see that constant for why).
_SIM_HORIZONS = ("pv",) + tuple(data.PANEL_HORIZONS)

# Per-horizon metric columns, in display order (matches the IC table: n, IC, win,
# ret). Each: (header template, id template, numeric format).
_SIM_METRIC_ORDER = ("n", "ic", "icstd", "icir", "win", "ret")
_SIM_METRIC_SPECS = {
    "n":     ("n@{}", "n_{}", _INT),
    "ic":    ("IC@{}", "ic_{}", _NUM2),
    "icstd": ("IC std@{}", "icstd_{}", _NUM2),
    "icir":  ("ICIR@{}", "icir_{}", _NUM2),
    "win":   ("Win@{} %", "win_{}", _NUM2),
    "ret":   ("Ret@{} %", "ret_{}", _NUM2),
}
_SIM_METRIC_LABELS = {"n": "n (obs)", "ic": "IC", "icstd": "IC std", "icir": "ICIR",
                      "win": "Win %", "ret": "Ret %"}


def _sim_column_filters(h_id: str = "sim-horizons", m_id: str = "sim-metrics") -> html.Div:
    """Horizons + metrics multi-selects that trim the columns of every simulated
    table below. Empty selection falls back to all (never an empty table). The
    ids are parameterised so the Method-Performance and Exit-Performance tabs each
    get an independent pair (duplicate component ids would break Dash)."""
    return html.Div(
        [
            html.Label("Columns",
                       title="Pick which horizons and which metrics (n / IC / Win % / Ret %) "
                             "appear in the category tables below. Applies to all "
                             "category tables at once; clearing a picker shows everything.",
                       className="filter-label"),
            dcc.Dropdown(id=h_id,
                         options=[{"label": h, "value": h} for h in _SIM_HORIZONS],
                         value=list(_SIM_HORIZONS), multi=True, placeholder="Horizons…",
                         persistence=True, persistence_type="session",
                         style={"flex": 2, "minWidth": 300}),
            dcc.Dropdown(id=m_id,
                         options=[{"label": _SIM_METRIC_LABELS[m], "value": m} for m in _SIM_METRIC_ORDER],
                         value=list(_SIM_METRIC_ORDER), multi=True, placeholder="Metrics…",
                         persistence=True, persistence_type="session",
                         style={"flex": 1, "minWidth": 210, "marginLeft": 8}),
        ],
        className="filter-item filter-item--grow",
    )


def _sim_perf_table(subset, labels, horizons, metrics):
    """One IC-category's simulated-performance DataTable, limited to the chosen
    horizons × metrics (IC/win/ret/n)."""
    horizons = [h for h in _SIM_HORIZONS if h in horizons] or list(_SIM_HORIZONS)
    metrics = [m for m in _SIM_METRIC_ORDER if m in metrics] or list(_SIM_METRIC_ORDER)
    rows = []
    for _, r in subset.iterrows():
        row = {"method": labels.get(r["method"], r["method"]), "views": int(r["views"])}
        for lbl in horizons:
            for m in metrics:
                cid = _SIM_METRIC_SPECS[m][1].format(lbl)
                v = r.get(cid)
                if m == "n":
                    row[cid] = int(v) if pd.notna(v) else 0
                else:
                    row[cid] = round(float(v), 3) if pd.notna(v) else None
        rows.append(row)
    cols = [{"name": "Method", "id": "method"},
            {"name": "Trades", "id": "views", "type": "numeric", "format": _INT}]
    for lbl in horizons:
        for m in metrics:
            name_t, id_t, fmt = _SIM_METRIC_SPECS[m]
            cols.append({"name": name_t.format(lbl), "id": id_t.format(lbl),
                         "type": "numeric", "format": fmt})
    cond = []
    for lbl in horizons:
        if "win" in metrics:
            wc = f"win_{lbl}"
            cond += [
                {"if": {"filter_query": f"{{{wc}}} >= 50", "column_id": wc},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": f"{{{wc}}} < 50", "column_id": wc},
                 "color": figures.NEG, "fontWeight": "bold"},
            ]
        for mc in ("ic", "icir", "ret"):
            if mc in metrics:
                c = f"{mc}_{lbl}"
                cond += [
                    {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                    {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
                ]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _simulated_perf_section(window_days, session=None, direction=None,
                            sel_horizons=None, sel_metrics=None):
    """Per-method directional win rate + IC + gross return over ALL scored tickers
    (the simulated_trades panel), grouped by the same IC categories. The
    horizons/metrics selections trim every category table's columns. ``session``
    filters by signal-generation session; ``direction`` by the method's own call
    side (a filter caption states any active filter so the basis is unambiguous)."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.signal_panel import IC_CATEGORY_ORDER
    sel_horizons = sel_horizons or list(_SIM_HORIZONS)
    sel_metrics = sel_metrics or list(_SIM_METRIC_ORDER)
    df = data.simulated_method_perf(days=window_days, session=session, direction=direction)
    heading = _h3("Simulated single-method performance — all scored tickers", _SIM_PERF_TOOLTIP)
    filt_bits = []
    if session:
        filt_bits.append(f"session = {session} (signal-generation time)")
    if direction:
        filt_bits.append(f"direction = {direction} calls only")
    filt_note = (html.Div("Filtered: " + " · ".join(filt_bits),
                          style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8})
                 if filt_bits else None)
    if df is None or getattr(df, "empty", True):
        return html.Div([
            heading,
            *( [filt_note] if filt_note is not None else [] ),
            html.Div("No simulated single-method trades with forward returns match "
                     "this window/session/direction yet. They accrue every run; "
                     "materialise existing history with "
                     "`python -m src.analysis.simulated_trades --backfill`.",
                     style={"color": "#6b7280"}),
        ])
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    children = [heading] + ([filt_note] if filt_note is not None else [])
    has_cat = "category" in df.columns
    for category in IC_CATEGORY_ORDER:
        subset = df[df["category"] == category] if has_cat else df
        if subset is None or subset.empty:
            continue
        children.append(html.Div(category, style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#475569"}))
        children.append(_sim_perf_table(subset, labels, sel_horizons, sel_metrics))
        if not has_cat:
            break
    return html.Div(children)


_POLICY_TOOLTIP = (
    "Offline policy evaluation — the COUNTERFACTUAL P&L of alternative sizing policies, "
    "replayed over the signals panel (every scored ticker, not just the trades that opened, "
    "so it's free of the ledger's selection bias). Because forward returns are observable for "
    "EVERY candidate, no importance-sampling is needed — each policy is simply replayed with "
    "known outcomes. All rows share the SAME actionable gate, so they trade the identical set "
    "(same 'Decisions' and equal-weighted 'Avg net %'); they differ ONLY in how they SIZE, "
    "which shows up in 'Cap-wtd net %' (capital-weighted return — where the sizing actually "
    "lands the money). A sizing policy EARNS ITS KEEP only if its cap-weighted return beats "
    "'flat (gate only)'. 'Info ratio' is mean/std of the per-day return (each day counted once). "
    "Net of the same calibrated round-trip cost the ledger charges. Forward-collected + thin "
    "(a 5-day horizon needs 5 sessions of cache past each signal) — directional, not yet "
    "conclusive; watch it thicken.")


def _policy_eval_section():
    """Head-to-head counterfactual of the sizing policies over the unbiased
    signals panel — the standing answer to 'does breadth/confidence sizing earn
    its keep?'. Shown at 1-day and 5-day horizons (short horizons fill in first)."""
    children = [_h3("Sizing policy comparison — counterfactual (offline eval)", _POLICY_TOOLTIP)]
    any_data = False
    for h in (1, 5):
        df = data.policy_comparison(days=90, horizon=h)
        if df is None or getattr(df, "empty", True):
            continue
        any_data = True
        rows, flat = [], None
        for _, r in df.iterrows():
            if str(r["policy"]).startswith("flat"):
                flat = r.get("cap_wtd_ret")
        for _, r in df.iterrows():
            cw = r.get("cap_wtd_ret")
            vs = (round(float(cw) - float(flat), 3)
                  if cw is not None and flat is not None and not str(r["policy"]).startswith("flat")
                  else None)
            rows.append({
                "policy": r["policy"], "decisions": r.get("n_decisions"), "days": r.get("n_days"),
                "win": r.get("win_rate"), "avg": r.get("avg_net_ret"),
                "capwtd": cw, "vs_flat": vs, "ir": r.get("info_ratio"),
            })
        cols = [
            {"name": "Policy", "id": "policy"},
            {"name": "Decisions", "id": "decisions", "type": "numeric", "format": _INT},
            {"name": "Days", "id": "days", "type": "numeric", "format": _INT},
            {"name": "Win %", "id": "win", "type": "numeric", "format": _NUM2},
            {"name": "Avg net %", "id": "avg", "type": "numeric", "format": _NUM2},
            {"name": "Cap-wtd net %", "id": "capwtd", "type": "numeric", "format": _NUM2},
            {"name": "vs flat", "id": "vs_flat", "type": "numeric", "format": _NUM2},
            {"name": "Info ratio", "id": "ir", "type": "numeric", "format": _NUM2},
        ]
        children.append(html.Div(f"{h}-day horizon", style={
            "fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#475569"}))
        children.append(dash_table.DataTable(
            data=rows, columns=cols,
            style_data_conditional=[
                {"if": {"filter_query": "{vs_flat} > 0", "column_id": "vs_flat"},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{vs_flat} < 0", "column_id": "vs_flat"},
                 "color": figures.NEG, "fontWeight": "bold"},
                {"if": {"filter_query": '{policy} contains "flat"'}, "backgroundColor": "#eff6ff"},
            ],
            **_TABLE_KW))
    if not any_data:
        children.append(html.Div(
            "No decidable decisions yet — the signals panel needs forward-return history "
            "(warm it with `python -m src.analysis.signal_panel --refresh`).",
            style={"color": "#6b7280"}))
    return html.Div(children)


_PREDICT_TOOLTIP = (
    "Predictability-feature IC — the measurement behind 'find stocks whose direction is easier to "
    "forecast for swing trading'. For every scored ticker it computes cheap per-stock features from "
    "OHLCV **as of the signal date** (no look-ahead), buckets the whole signals panel into quantiles "
    "of each feature, and reports how well the aggregate combined_score predicted the forward return "
    "INSIDE each bucket. Features: Kaufman trend efficiency (20d — clean move vs chop), ADX (14d — "
    "trend strength), realized volatility (20d), and signal breadth (methods agreeing). Per horizon: "
    "'IC' = Spearman(score, forward return), 'hit %' = directional accuracy, 'sim %' = mean signed "
    "return. The signal is SEPARATION ACROSS BUCKETS, not the level: a feature is a useful "
    "predictability filter iff hit/IC/sim climb Low→High (trend features) or peak in the MID bucket "
    "(volatility). Uses the whole panel (features are OHLCV-derived, not stamp-dependent), so it has "
    "signal now — but it is ~2 weeks of one regime; treat as directional, and lean on the 5-day "
    "swing horizon (10-day n is still thin).")

_PREDICT_EDGE_TOOLTIP = (
    "The punchline: per feature, how much its buckets SEPARATE prediction quality — the "
    "best-minus-worst-bucket spread in hit % and signed-return %, and which bucket is best, per "
    "horizon. A large spread with a sensible best bucket (High for trend efficiency / ADX, Mid for "
    "volatility) means that feature sorts predictable names from unpredictable ones and is worth "
    "promoting to a discovery-prioritisation / sizing tilt (Tier 1). A spread ≈ 0, or a 'best' "
    "bucket that contradicts the hypothesis, means it doesn't.")

_PRED_HORIZONS = ("pv", 1, 5, 10)

_PRICEVOL_TOOLTIP = (
    "Do penny / thin-volume names behave differently from pricier / liquid ones — the question behind "
    "widening the discovery filter to < $1 / < $5M? Two datasets on fixed, interpretable bands (aligned "
    "to the $1/$5 price and $5M/$20M dollar-volume gate thresholds): (1) realized TRADE returns from the "
    "ledger — the strategy's actual P&L, by the trade's entry price and the stock's as-of-entry 20-day "
    "dollar volume (small + selection-biased, read the n on each bar); (2) combined_score across the "
    "UNBIASED signals panel (thousands of rows) with the mean 5-day forward return alongside — the "
    "large-sample view of how conviction and the realized move vary across the price/volume grid. Bars "
    "are green ≥ 0 / red < 0.")


def _pv_row(figs):
    """A responsive flex row of dcc.Graphs (wraps on a narrow screen)."""
    return html.Div(
        [html.Div(dcc.Graph(figure=f), style={"flex": "1 1 440px", "minWidth": 0}) for f in figs],
        style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})


def _price_volume_section():
    """Return & score by stock price and dollar volume — the penny-vs-pricier
    divergence behind the widened discovery filter."""
    res = data.price_volume_perf()
    tr = (res or {}).get("trades") or {}
    sc = (res or {}).get("signals") or {}
    children = [_h3("Return & score by price / dollar-volume", _PRICEVOL_TOOLTIP)]

    # (1) realized trade returns from the ledger
    children.append(html.Div(
        f"Realized trade return — {tr.get('n_trades', 0)} trades "
        f"({tr.get('n_with_dvol', 0)} with a volume read). Small, selection-biased sample — "
        "watch the n on each bar.",
        style={"fontWeight": "bold", "marginTop": 8, "marginBottom": 4, "color": "#475569"}))
    children.append(_pv_row([
        figures.bucket_bar_fig(tr.get("by_price"), "Trade return by stock price", "Avg return %", pct=True),
        figures.bucket_bar_fig(tr.get("by_dvol"), "Trade return by dollar volume", "Avg return %", pct=True),
    ]))

    # (2) combined_score across the unbiased panel + the 5-day forward return
    children.append(html.Div(
        f"Signal conviction & realized move — combined_score and mean 5-day forward return over "
        f"{sc.get('n_rows', 0):,} unbiased signals-panel rows.",
        style={"fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#475569"}))
    children.append(_pv_row([
        figures.bucket_bar_fig(sc.get("by_price"), "Score by stock price", "Avg combined_score"),
        figures.bucket_bar_fig(sc.get("by_dvol"), "Score by dollar volume", "Avg combined_score"),
    ]))
    children.append(_pv_row([
        figures.bucket_bar_fig(sc.get("fwd_by_price"), "Forward 5d return by price (unbiased)", "Avg 5d fwd %", pct=True),
        figures.bucket_bar_fig(sc.get("fwd_by_dvol"), "Forward 5d return by dollar volume (unbiased)", "Avg 5d fwd %", pct=True),
    ]))
    return html.Div(children)


def _predictability_section():
    """Bucketed conditional IC of combined_score by per-stock predictability
    feature — the Tier-0 measurement of which features make our direction call
    more forecastable at a swing horizon."""
    res = data.predictability()
    buckets = res.get("buckets") if isinstance(res, dict) else None
    edges = res.get("edges") if isinstance(res, dict) else None
    heading = _h3("Predictability by stock feature (conditional IC)", _PREDICT_TOOLTIP)
    if buckets is None or getattr(buckets, "empty", True):
        return html.Div([heading, html.Div(
            "No signals-panel rows with forward returns yet — warm forward closes with "
            "`python -m src.analysis.signal_panel --refresh`. Accrues every run.",
            style={"color": "#6b7280"})])

    children = [heading]

    # ── the edge summary (headline) ──
    if edges is not None and not getattr(edges, "empty", True):
        erows = []
        for _, r in edges.iterrows():
            row = {"label": r["label"]}
            for h in _PRED_HORIZONS:
                sfx = "pv" if h == "pv" else f"{h}d"
                hs, hb, ss = (r.get(f"hit_spread_{sfx}"), r.get(f"hit_best_{sfx}"),
                              r.get(f"simret_spread_{sfx}"))
                row[f"hitsp_{sfx}"] = round(float(hs), 2) if pd.notna(hs) else None
                row[f"best_{sfx}"] = hb if (hb is not None and pd.notna(hb)) else "—"
                row[f"simsp_{sfx}"] = round(float(ss), 3) if pd.notna(ss) else None
            erows.append(row)
        ecols = [{"name": "Feature", "id": "label"}]
        for h in _PRED_HORIZONS:
            sfx, disp = ("pv", "pivot") if h == "pv" else (f"{h}d", f"{h}d")
            ecols += [
                {"name": f"Hit spread@{disp} %", "id": f"hitsp_{sfx}", "type": "numeric", "format": _NUM2},
                {"name": f"Best@{disp}", "id": f"best_{sfx}"},
                {"name": f"Sim spread@{disp} %", "id": f"simsp_{sfx}", "type": "numeric", "format": _NUM2},
            ]
        children += [
            html.Div("Feature edge — best-minus-worst bucket separation", style={
                "fontWeight": "bold", "marginTop": 8, "marginBottom": 4, "color": "#475569"}),
            html.Div("Larger spread = the feature sorts predictable from unpredictable names. "
                     "'Best' should be High for trend efficiency / ADX, Mid for volatility.",
                     title=_PREDICT_EDGE_TOOLTIP,
                     style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 6, "cursor": "help"}),
            dash_table.DataTable(data=erows, columns=ecols, **_TABLE_KW),
        ]

    # ── the bucketed detail ──
    brows, cond = [], []
    for _, r in buckets.iterrows():
        is_base = r["feature"] == "(all rows)"
        rng = "all" if pd.isna(r.get("lo")) else f"[{float(r['lo']):g}, {float(r['hi']):g}]"
        row = {"label": "BASELINE (all)" if is_base else r["label"],
               "bucket": "—" if is_base else r["bucket"], "range": rng,
               "n_rows": int(r["n_rows"])}
        for h in _PRED_HORIZONS:
            sfx = "pv" if h == "pv" else f"{h}d"
            n, ic, hit, sim = (r.get(f"n_{sfx}"), r.get(f"ic_{sfx}"),
                               r.get(f"hit_{sfx}"), r.get(f"simret_{sfx}"))
            row[f"n_{sfx}"] = int(n) if pd.notna(n) else 0
            row[f"ic_{sfx}"] = round(float(ic), 3) if pd.notna(ic) else None
            row[f"hit_{sfx}"] = round(float(hit), 1) if pd.notna(hit) else None
            row[f"sim_{sfx}"] = round(float(sim), 2) if pd.notna(sim) else None
        brows.append(row)
    bcols = [{"name": "Feature", "id": "label"}, {"name": "Bucket", "id": "bucket"},
             {"name": "Range", "id": "range"},
             {"name": "Rows", "id": "n_rows", "type": "numeric", "format": _INT}]
    for h in _PRED_HORIZONS:
        sfx, disp = ("pv", "pivot") if h == "pv" else (f"{h}d", f"{h}d")
        bcols += [
            {"name": f"n@{disp}", "id": f"n_{sfx}", "type": "numeric", "format": _INT},
            {"name": f"IC@{disp}", "id": f"ic_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"Hit@{disp} %", "id": f"hit_{sfx}", "type": "numeric", "format": _NUM2},
            {"name": f"Sim@{disp} %", "id": f"sim_{sfx}", "type": "numeric", "format": _NUM2},
        ]
        for c in (f"ic_{sfx}", f"sim_{sfx}"):
            cond += [
                {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
            ]
        hc = f"hit_{sfx}"
        cond += [
            {"if": {"filter_query": f"{{{hc}}} >= 50", "column_id": hc}, "color": figures.POS},
            {"if": {"filter_query": f"{{{hc}}} < 50", "column_id": hc}, "color": figures.NEG},
        ]
    cond.append({"if": {"filter_query": '{bucket} = "—"'}, "backgroundColor": "#eff6ff"})
    children += [
        html.Div("Bucket detail — combined_score prediction quality within each feature bucket",
                 style={"fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#475569"}),
        dash_table.DataTable(data=brows, columns=bcols, style_data_conditional=cond, **_TABLE_KW),
    ]
    return html.Div(children)


_SOURCE_PERF_TOOLTIP = (
    "Discovery-source performance — which parts of the universe-construction funnel actually "
    "surface names that move. The pipeline stamps the FIRST discovery source that surfaced each "
    "ticker (watchlist / trending / screener / macro→holdings / smart_money / sector_etf / "
    "related-company / catalyst …) onto every scored row; this groups the signals panel (every "
    "scored ticker, joined with forward returns — NOT just the gate-selected trades, so it's free "
    "of selection bias) by that stamp. 'Rows' = scored ticker-rows; 'Funnel %' = the source's "
    "slice of the STAMPED funnel (the '(unstamped)' bucket is pre-stamp history — the stamp is "
    "forward-collected from 2026-07-03 — so it is excluded from that denominator and sinks to the "
    "bottom; it still carries forward returns while the freshly-stamped sources' fill in). Per "
    "horizon (1d/5d/10d): 'n@' = rows with a forward return; 'Fwd ret@ %' = mean RAW forward "
    "return of the source's names (discovery quality — do these names tend to rise?, "
    "direction-agnostic); 'Win@ %' = share of moved names that rose; 'IC@' = Spearman correlation "
    "of the aggregate combined_score against the forward return (signal skill ON this source's "
    "names — a persistent NEGATIVE IC means the source's names 'trade but predict backwards'). A "
    "source earns more discovery budget when it combines a large-enough n with a positive Fwd ret "
    "AND a non-negative IC; a big-share source that is flat-return / negative-IC is funnel noise "
    "the near-zero-IC gates then have to sift. Run/forward-return based (ignores the window "
    "toggle); forward-collected — n grows every run, judge nothing on a thin panel.")

_SOURCE_TRADE_TOOLTIP = (
    "Realized trade outcomes grouped by the discovery source that first surfaced the ticker — the "
    "small-n, gate-selected counterpart to the forward-return table above (this is what actually "
    "opened and made or lost money). Direction is baked into the return, so a win is simply "
    "return > 0. Open trades contribute their live mark-to-market. Judge alongside the unbiased "
    "panel view — a handful of trades from one source is anecdote, not evidence.")

_SRC_HORIZONS = ("pv", 1, 5, 10)


def _source_perf_section():
    """Per-discovery-source forward-return performance over the signals panel
    (the unbiased accumulator behind an adaptive discovery budget) plus the
    realized per-source trade outcomes from the ledger."""
    children = [_h3("Discovery source performance (forward returns by provenance)",
                    _SOURCE_PERF_TOOLTIP)]

    perf = data.source_performance()
    if perf is None or getattr(perf, "empty", True):
        children.append(html.Div(
            "No per-source signal rows with forward returns yet — the signals panel accrues "
            "every run (warm forward closes with `python -m src.analysis.signal_panel --refresh`).",
            style={"color": "#6b7280"}))
    else:
        rows = []
        for _, r in perf.iterrows():
            fp = r.get("funnel_pct")
            row = {"source": r["source"], "rows": int(r["rows"]),
                   "funnel_pct": round(float(fp), 1) if pd.notna(fp) else None}
            for h in _SRC_HORIZONS:
                sfx = "pv" if h == "pv" else f"{h}d"
                n, fwd, win, ic = (r.get(f"n_{sfx}"), r.get(f"fwd_{sfx}"),
                                   r.get(f"win_{sfx}"), r.get(f"ic_{sfx}"))
                row[f"n_{sfx}"] = int(n) if pd.notna(n) else 0
                row[f"fwd_{sfx}"] = round(float(fwd), 2) if pd.notna(fwd) else None
                row[f"win_{sfx}"] = round(float(win), 1) if pd.notna(win) else None
                row[f"ic_{sfx}"] = round(float(ic), 3) if pd.notna(ic) else None
            rows.append(row)
        cols = [{"name": "Source", "id": "source"},
                {"name": "Rows", "id": "rows", "type": "numeric", "format": _INT},
                {"name": "Funnel %", "id": "funnel_pct", "type": "numeric", "format": _NUM2}]
        cond = []
        for h in _SRC_HORIZONS:
            sfx, disp = ("pv", "pivot") if h == "pv" else (f"{h}d", f"{h}d")
            cols += [
                {"name": f"n@{disp}", "id": f"n_{sfx}", "type": "numeric", "format": _INT},
                {"name": f"Fwd ret@{disp} %", "id": f"fwd_{sfx}", "type": "numeric", "format": _NUM2},
                {"name": f"Win@{disp} %", "id": f"win_{sfx}", "type": "numeric", "format": _NUM2},
                {"name": f"IC@{disp}", "id": f"ic_{sfx}", "type": "numeric", "format": _NUM2},
            ]
            for c in (f"fwd_{sfx}", f"ic_{sfx}"):
                cond += [
                    {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                    {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
                ]
            wc = f"win_{sfx}"
            cond += [
                {"if": {"filter_query": f"{{{wc}}} >= 50", "column_id": wc}, "color": figures.POS},
                {"if": {"filter_query": f"{{{wc}}} < 50", "column_id": wc}, "color": figures.NEG},
            ]
        children.append(dash_table.DataTable(data=rows, columns=cols,
                                             style_data_conditional=cond, **_TABLE_KW))

    trade_perf = data.source_trade_perf()
    children.append(_h3("Realized trades by discovery source", _SOURCE_TRADE_TOOLTIP))
    if not trade_perf:
        children.append(html.Div("No attributed trades yet.", style={"color": "#6b7280"}))
    else:
        tcols = [
            {"name": "Source", "id": "source"},
            {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
            {"name": "Win %", "id": "win_rate", "type": "numeric", "format": _NUM2},
            {"name": "Avg return %", "id": "avg_return", "type": "numeric", "format": _NUM2},
            {"name": "Median %", "id": "median_return", "type": "numeric", "format": _NUM2},
            {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
            {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
        ]
        tcond = [
            {"if": {"filter_query": "{avg_return} > 0", "column_id": "avg_return"}, "color": figures.POS},
            {"if": {"filter_query": "{avg_return} < 0", "column_id": "avg_return"}, "color": figures.NEG},
            {"if": {"filter_query": "{win_rate} >= 50", "column_id": "win_rate"}, "color": figures.POS},
            {"if": {"filter_query": "{win_rate} < 50", "column_id": "win_rate"}, "color": figures.NEG},
        ]
        children.append(dash_table.DataTable(data=trade_perf, columns=tcols,
                                             style_data_conditional=tcond, **_TABLE_KW))
    return html.Div(children)


_MC_METHODS_TOOLTIP = (
    "Monte Carlo luck-vs-skill — is each method's track record statistically distinguishable "
    "from a coin flip at its sample size? Judged on the GROSS solo win rate (sign(score) × raw "
    "price move, pre-cost) — the exact number the win-rate method filter selects on, same train "
    "split. Two resampling tests per method (2000 sims, fixed seed): BOOTSTRAP resamples the "
    "method's own trades with replacement → the 5–95% CI on its win rate and mean oriented "
    "return (how wide is the evidence); PERMUTATION NULL replaces every direction call with a "
    "fair coin on the same |price moves| → p(luck) = probability a NO-SKILL method would post "
    "at least this win rate by chance (p(ret) = same test on the mean return — a method right "
    "on the BIG moves scores better here than raw hit rate shows). One-sided: p < 0.05 ⇒ "
    "evidence of real skill; p > 0.95 ⇒ reliably WORSE than chance (inversion candidate); "
    "anything between = the record is consistent with noise — a keep/drop decision based on it "
    "is provisional. 'Filter state' shows what the live win-rate filter did with the method. "
    "The selection-bias line above the table runs the WHOLE filter on synthetic coin-flip "
    "methods at the real trade counts: if chance alone would keep about as many methods as the "
    "filter kept, the current kept set is not yet evidence of skill (expect churn as trades "
    "accrue). Small samples move these p-values a lot — re-read as the ledger grows.")


def _mc_overfit_section():
    """Monte Carlo overfitting check — per-method luck-vs-skill + the win-rate
    filter's selection-bias null (src/analysis/monte_carlo.py)."""
    rep = data.monte_carlo_methods()
    rows = rep.get("rows") or []
    heading = _h3("Overfitting check — Monte Carlo luck vs skill", _MC_METHODS_TOOLTIP)
    if not rows:
        return html.Div([heading, html.Div(
            "No closed trades with method attribution yet — accrues with the ledger.",
            style={"color": "#6b7280"})])

    sel = rep.get("selection") or {}
    sel_line = None
    if sel.get("n_judgeable"):
        sel_line = html.Div(
            f"Win-rate filter selection-bias null: at the real per-method trade counts, pure "
            f"chance would keep {sel['kept_null_mean']} ± {sel['kept_null_sd']} of "
            f"{sel['n_judgeable']} judgeable methods (5–95%: {sel['kept_null_lo']}–"
            f"{sel['kept_null_hi']}); the live filter kept {sel['kept_actual']} "
            f"(p ≥ actual = {sel['p_ge_actual']}). → {sel.get('verdict', '')}",
            style={"color": "#475569", "marginBottom": 8})
    elif sel.get("verdict"):
        sel_line = html.Div(sel["verdict"], style={"color": "#6b7280", "marginBottom": 8})

    trows = [{
        "method": r["method"], "state": r.get("filter_state", "—"), "n": r["n"],
        "wr": r["win_rate"], "wr_ci": f"{r['wr_lo']:.0f} – {r['wr_hi']:.0f}",
        "p_luck": r["p_luck"],
        "ret": r["mean_ret"], "ret_ci": f"{r['ret_lo']:.2f} – {r['ret_hi']:.2f}",
        "p_ret": r["p_ret"], "verdict": r["verdict"],
    } for r in rows]
    tcols = [
        {"name": "Method", "id": "method"},
        {"name": "Filter state", "id": "state"},
        {"name": "Trades", "id": "n", "type": "numeric", "format": _INT},
        {"name": "Gross WR %", "id": "wr", "type": "numeric", "format": _NUM1},
        {"name": "WR CI 5–95%", "id": "wr_ci"},
        {"name": "p (luck)", "id": "p_luck", "type": "numeric", "format": _NUM3},
        {"name": "Mean ret %", "id": "ret", "type": "numeric", "format": _NUM2},
        {"name": "Ret CI 5–95%", "id": "ret_ci"},
        {"name": "p (ret)", "id": "p_ret", "type": "numeric", "format": _NUM3},
        {"name": "Verdict", "id": "verdict"},
    ]
    cond = [
        {"if": {"filter_query": "{p_luck} < 0.05", "column_id": "p_luck"}, "color": figures.POS},
        {"if": {"filter_query": "{p_luck} > 0.95", "column_id": "p_luck"}, "color": figures.NEG},
        {"if": {"filter_query": "{p_ret} < 0.05", "column_id": "p_ret"}, "color": figures.POS},
        {"if": {"filter_query": "{p_ret} > 0.95", "column_id": "p_ret"}, "color": figures.NEG},
        {"if": {"filter_query": '{verdict} contains "SKILL"', "column_id": "verdict"},
         "color": figures.POS},
        {"if": {"filter_query": '{verdict} contains "worse"', "column_id": "verdict"},
         "color": figures.NEG},
        {"if": {"filter_query": '{state} = "FILTERED"', "column_id": "state"},
         "color": "#6b7280"},
    ]
    return html.Div([heading] + ([sel_line] if sel_line is not None else []) + [
        dash_table.DataTable(data=trows, columns=tcols, style_data_conditional=cond,
                             **_TABLE_KW),
    ])


_CONF_COMPONENTS_TOOLTIP = (
    "Isolates each multiplier in the confidence formula (confidence = raw × coherence × "
    "movement × volume × family × tape — src/signals/aggregator.py::_score_ticker) to see "
    "which ones actually earn their keep. 'Raw score only' = min(1, |combined_score| / 0.5) "
    "with no multiplier applied; each other row applies exactly ONE factor on top of raw "
    "(capped at 1.0) — in ISOLATION, not stacked cumulatively. 'Live (all combined)' is the "
    "actual confidence the system uses today, shown as the reference row. IC = Spearman rank "
    "correlation between the variant's value and the DIRECTION-ORIENTED forward return "
    "(sign(combined_score) × forward return) — positive means higher readings of that "
    "variant genuinely predict better outcomes; ≈0 means it doesn't discriminate despite "
    "moving the number (this top table deliberately omits win%/return — ungated, those "
    "never depend on the variant's value, only IC does; they'd be redundant at best, "
    "missingness noise at worst). The conviction-band table below splits each variant's OWN "
    "value into Low (0.10–0.35) / Medium (0.35–0.65) / High "
    "(0.65+) — the same cut points tracker._eval_stats uses for per-method calibration — so "
    "you can see whether win rate / return actually RISES with that variant's own conviction "
    "(a well-behaved component shows Low < Medium < High; flat or inverted means it isn't "
    "separating good calls from bad ones). Forward-collected from 2026-07-21 when the factor "
    "columns were added to the signals panel — 0 rows at first, fills in every run.")

_CONF_COMPONENTS_EXIT_NOTE = (
    "Exit-side: the SAME isolation, but over signals-panel rows re-scored on an ALREADY-OPEN "
    "position mid-hold (oriented by the trade's own direction, not the ticker's possibly-"
    "since-drifted current call; forward return measured from the re-read tick, not the "
    "original entry) — does a component's reading, taken WHILE HOLDING, predict what happens "
    "to the position from that point on. No separate capture path: held tickers stay in the "
    "scored universe every tick, so this is a join against the trades ledger's open interval, "
    "not a new signal.")


# "pv" leads (2026-08-13 standardization: pivot target first, fixed as monitors)
_DEFAULT_CONF_HORIZONS = ("pv", 1, 5, 10)


def _conf_component_ic_table(icdf: pd.DataFrame) -> dash_table.DataTable:
    rows = icdf.rename(columns={"label": "Variant"}).to_dict("records")
    cols = [{"name": "Variant", "id": "Variant"}]
    for h in _DEFAULT_CONF_HORIZONS:
        sfx, disp = ("pv", "pivot") if h == "pv" else (f"{h}d", f"{h}d")
        cols += [
            {"name": f"n@{disp}", "id": f"n_{sfx}", "type": "numeric", "format": _INT},
            {"name": f"IC@{disp}", "id": f"ic_{sfx}", "type": "numeric", "format": _NUM3},
            {"name": f"ICIR@{disp}", "id": f"icir_{sfx}", "type": "numeric", "format": _NUM2},
        ]
    cond = [{"if": {"filter_query": '{Variant} = "Live (all combined)"'},
            "backgroundColor": "#eff6ff"}]
    for h in _DEFAULT_CONF_HORIZONS:
        sfx = "pv" if h == "pv" else f"{h}d"
        cond += [
            {"if": {"filter_query": f"{{ic_{sfx}}} > 0.03", "column_id": f"ic_{sfx}"},
             "color": figures.POS},
            {"if": {"filter_query": f"{{ic_{sfx}}} < -0.03", "column_id": f"ic_{sfx}"},
             "color": figures.NEG},
        ]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _conf_component_band_table(banddf: pd.DataFrame) -> dash_table.DataTable:
    rows = banddf.rename(columns={"label": "Variant", "band_label": "Band"}).to_dict("records")
    cols = [{"name": "Variant", "id": "Variant"}, {"name": "Band", "id": "Band"}]
    for h in _DEFAULT_CONF_HORIZONS:
        sfx, disp = ("pv", "pivot") if h == "pv" else (f"{h}d", f"{h}d")
        cols += [
            {"name": f"n@{disp}", "id": f"n_{sfx}", "type": "numeric", "format": _INT},
            {"name": f"Win@{disp} %", "id": f"win_{sfx}", "type": "numeric", "format": _NUM1},
            {"name": f"Ret@{disp} %", "id": f"ret_{sfx}", "type": "numeric", "format": _NUM2},
        ]
    cond = [{"if": {"filter_query": '{Band} = "High (0.65+)"'}, "backgroundColor": "#eff6ff"}]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _confidence_components_section():
    """Entry-side confidence-component isolation (src/analysis/confidence_components.py)."""
    rep = data.confidence_components_entry()
    heading = _h3("Confidence-formula component isolation", _CONF_COMPONENTS_TOOLTIP)
    if not rep.get("has_factors"):
        return html.Div([heading, html.Div(
            "Forward-collecting — the per-factor columns (coherence / movement / volume / "
            "family / tape) were just added to the signals panel; this fills in from the "
            "next pipeline run onward.", style={"color": "#6b7280"})])
    icdf, banddf = rep.get("ic"), rep.get("bands")
    if icdf is None or icdf.empty:
        return html.Div([heading, html.Div(
            f"{rep.get('panel_rows', 0)} signal row(s) with factor data — not enough "
            "forward-return history yet.", style={"color": "#6b7280"})])
    return html.Div([
        heading,
        html.Div(f"{rep['panel_rows']} scored ticker-tick(s) with factor data",
                 style={"color": "#475569", "marginBottom": 8}),
        _conf_component_ic_table(icdf),
        html.Div("By conviction band (does win rate / return rise with THIS variant's own "
                 "conviction level?):",
                 style={"marginTop": 14, "marginBottom": 4, "color": "#475569"}),
        _conf_component_band_table(banddf),
    ])


def _exit_confidence_components_block(session=None, direction=None):
    """Exit-side confidence-component isolation — held-position mid-hold re-reads."""
    rep = data.confidence_components_exit(session=session, direction=direction)
    note = html.Div(_CONF_COMPONENTS_EXIT_NOTE, style={"color": "#6b7280", "marginBottom": 8})
    if not rep.get("has_factors") or not rep.get("panel_rows"):
        return html.Div([note, html.Div(
            "No held-position re-reads with factor data yet — accrues once a position "
            "opened after 2026-07-21 is held past its entry day.",
            style={"color": "#6b7280"})])
    icdf, banddf = rep.get("ic"), rep.get("bands")
    if icdf is None or icdf.empty:
        return html.Div([note, html.Div(
            f"{rep.get('panel_rows', 0)} held re-read(s) — not enough forward-return "
            "history yet.", style={"color": "#6b7280"})])
    return html.Div([
        note,
        html.Div(f"{rep['panel_rows']} held-position re-read(s) with factor data",
                 style={"color": "#475569", "marginBottom": 8}),
        _conf_component_ic_table(icdf),
        html.Div("By conviction band:",
                 style={"marginTop": 14, "marginBottom": 4, "color": "#475569"}),
        _conf_component_band_table(banddf),
    ])


def _methods_tab():
    # The LLM-models-used table is run-based (not trade-windowed), so it lives
    # outside the windowed body. The per-method performance section (bar + table)
    # is filled by _methods_body() against the selected time window.
    runs = data.runs_df()
    model_rows = _models_used_rows(runs) if not runs.empty else []
    models_table = dash_table.DataTable(
        data=model_rows,
        columns=[{"name": c, "id": c} for c in ["Role", "Model", "API", "Runs", "Calls"]],
        tooltip_header=_MODELS_HEADER_TIPS,
        **_TABLE_KW,
    ) if model_rows else html.Div("No runs recorded yet.", style={"color": "#6b7280"})

    return html.Div([
        html.Div([
            _window_toggle("methods-window"),
            _session_toggle("methods-session"),
            _direction_toggle("methods-direction"),
            _asset_toggle("methods-asset"),
            _method_source_toggle("methods-source"),
            _sim_column_filters(),
        ], className="filter-bar"),
        dcc.Loading(html.Div(id="methods-body")),
        _safe(_method_decile_section),
        _safe(_ic_section),
        _safe(_mc_overfit_section),
        _safe(_confidence_components_section),
        _safe(_policy_eval_section),
        _safe(_predictability_section),
        _safe(_price_volume_section),
        _safe(_source_perf_section),
        _h3("LLM models used (synthesis & sentiment)",
            "Which exact LLMs actually ran across all recorded pipeline runs — the final-call 'synthesis' model and the per-ticker 'sentiment' model — including any DeepSeek or rule-based fallbacks. Not affected by the window toggle above (it's run-based, not trade-based). Hover a column header for details."),
        models_table,
    ])


def _method_decile_section():
    """Per-method decile curve on the pivot basis, method picked by dropdown —
    the visual companion to the 2026-08-13 rank directive: the combine now
    consumes each method's within-day RANK, and this is exactly that rank's
    decile payoff. Full-panel (window-toggle independent; the ranks are
    within-day, so mixing windows is safe)."""
    from src.performance.tracker import METHOD_LABELS
    res = data.method_decile_curves()
    methods = sorted((res.get("methods") or {}).keys())
    if not methods:
        return html.Div("No settled pivot rows yet — the decile view fills as "
                        "pivots confirm.", style={"color": "#6b7280"})
    labels = dict(METHOD_LABELS)
    default = "ext_gap" if "ext_gap" in methods else methods[0]
    meta = res.get("meta") or {}
    sub = (f"{meta.get('rows', 0):,} settled panel rows · {meta.get('days', 0)} days "
           f"({meta.get('d0', '')} → {meta.get('d1', '')}) · means winsorized "
           f"{meta.get('winsor', ['', ''])[0]}..{meta.get('winsor', ['', ''])[1]}%")
    return html.Div([
        _h3("Method decile curve — signed pivot return by within-day score rank",
            "Each panel row's method score is ranked WITHIN ITS DAY among that method's non-zero "
            "scores and bucketed into deciles; bars show the decile's mean SIGNED move to the next "
            "H/L pivot (win% and n in the hover). This is the exact consumption the rank basis "
            "(method_score_basis=rank) feeds the combine: an upward slope = the day's stronger "
            "scores genuinely carry more upside; a flat or n-shaped curve = the ranking carries "
            "little. Pick a method below."),
        html.Div(sub, style={"color": "#6b7280", "fontSize": 12, "marginBottom": 6}),
        dcc.Dropdown(id="method-decile-dd",
                     options=[{"label": labels.get(m, m), "value": m} for m in methods],
                     value=default, clearable=False,
                     style={"width": 340, "marginBottom": 8}),
        dcc.Loading(dcc.Graph(id="method-decile-graph",
                              figure=figures.method_decile_fig(
                                  res["methods"][default], default,
                                  labels.get(default, default)))),
    ])


@app.callback(Output("method-decile-graph", "figure"),
              Input("method-decile-dd", "value"))
def _method_decile_update(method):
    from src.performance.tracker import METHOD_LABELS
    res = data.method_decile_curves()
    curve = (res.get("methods") or {}).get(method) or {}
    return figures.method_decile_fig(curve, method or "",
                                     dict(METHOD_LABELS).get(method, method or ""))


@app.callback(Output("methods-body", "children"),
              Input("methods-window", "value"), Input("methods-session", "value"),
              Input("methods-direction", "value"), Input("methods-asset", "value"),
              Input("methods-source", "value"),
              Input("sim-horizons", "value"), Input("sim-metrics", "value"))
def _methods_body(window_value, session_value, direction_value, asset_value, source_value,
                  sim_horizons, sim_metrics):
    if source_value == "panel":
        # All scored tickers — honors the window (by signal date), the session
        # the signal was GENERATED in, the direction of the method's own call
        # (positive score = its long call), and the Horizons/Metrics pickers.
        # (The unbiased signal panel has no instrument-type column, so the Type
        # filter applies only to the ledger-based view below.)
        return _safe(lambda: _simulated_perf_section(_window_days(window_value),
                                                     _session_value(session_value),
                                                     _direction_value(direction_value),
                                                     sim_horizons, sim_metrics))
    return _safe(lambda: _methods_perf_section(_window_days(window_value), _session_value(session_value),
                                               _direction_value(direction_value),
                                               _asset_value(asset_value)))


def _calibration_block(window_days, session, direction=None):
    """Confidence-calibration buckets + slope (item #2) — the formal summary of
    the return-vs-confidence scatter above it."""
    cal = data.confidence_calibration(window_days, session, direction)
    rows = [{"bucket": b["label"], "trades": b["trades"], "win": b["win_rate"],
             "avg": b["avg_return"], "median": b["median_return"], "wtd": b["wtd_avg_return"],
             "best": b["best"], "worst": b["worst"]} for b in (cal.get("buckets") or [])]
    cols = [
        {"name": "Confidence bucket", "id": "bucket"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Win rate %", "id": "win", "type": "numeric", "format": _NUM2},
        {"name": "Avg return %", "id": "avg", "type": "numeric", "format": _NUM2},
        {"name": "Median %", "id": "median", "type": "numeric", "format": _NUM2},
        {"name": "Wtd avg %", "id": "wtd", "type": "numeric", "format": _NUM2},
        {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
        {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
    ]
    table = (dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW) if rows
             else html.Div("No trades with a stored confidence in this window yet.",
                           style={"color": "#6b7280"}))
    return html.Div([
        _h3("Confidence calibration — buckets + slope",
            "The bucketed companion to the scatter above: trades grouped by entry "
            "confidence. A positive slope means confidence carries return-predictive "
            "information worth sizing on; flat/negative means it does not. The measured "
            "answer so far is FLAT, which is why the confidence ramp is compressed to a "
            "1.5× cap instead of the legacy 2.0× — the buckets are points on a continuous "
            "ramp, not discrete tiers. Closed trades at realised return, open at live "
            "mark. Respects the window + session toggles."),
        html.Div(cal.get("verdict", ""),
                 style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        dcc.Graph(figure=figures.calibration_bar_fig(cal)),
        table,
    ])


def _methods_perf_section(window_days, session=None, direction=None, asset_type=None):
    perf = data.performance(window_days=window_days, session=session, direction=direction,
                            asset_type=asset_type)
    solo = perf.get("solo_method_perf") or {}
    labels = perf.get("method_labels") or {}
    order = perf.get("method_order_by_winrate") or list(solo.keys())

    # Market-relative skill — the basis the WEIGHTING now uses. Shown beside the
    # absolute win rate rather than replacing it: they answer different
    # questions (does the signal carry information vs what did it earn), and the
    # gap between them is exactly the beta the absolute number cannot see.
    rel_skill, rel_base = data.market_relative_skill()
    # The baseline is MEASURED and drifts (48.6% when first documented, 49.8%
    # a month later), so it is rendered live rather than quoted in a tooltip —
    # a hardcoded copy of a moving number is stale the day after it is written.
    _rel_baseline_note = lambda: (
        f"'vs base' is measured against the current market-relative baseline of "
        f"{rel_base:.2f}% — the share of the time the median stock beats SPY, NOT 50%. "
        f"The cap-weighted index beats its typical constituent, so 50% would hold "
        f"every method to a bar {50.0 - rel_base:+.2f}pp too high. Re-measured from "
        f"the panel each run.")

    rows = []
    for m in order:
        overall = (solo.get(m) or {}).get("overall") or {}
        if not overall:
            continue
        r = rel_skill.get(m) or {}
        rw = r.get("win_rate")
        rows.append({
            "Method": labels.get(m, m),
            "Win rate %": round(overall["win_rate"], 1) if overall.get("win_rate") is not None else None,
            "Rel win %": rw,
            "vs base": (round(rw - rel_base, 1) if rw is not None else None),
            "Rel n": r.get("trades"),
            "Trades": overall.get("trades", overall.get("n")),
            "Avg return %": round(overall["avg_return"], 2) if overall.get("avg_return") is not None else None,
        })

    # Per-LLM rows — EVERY trade each engine recommended (executed or not):
    # one pseudo-trade per (engine, ticker, day), entered at the recorded
    # recommendation-time price and marked at the latest cached close, so
    # LLM APIs are compared on their full call stream rather than the few
    # recommendations that survived the trading gates.
    llm = perf.get("llm_perf") or {}
    for role, label in (("synthesis", "Synthesis LLM"), ("sentiment", "Sentiment LLM")):
        by_model = llm.get(role) or {}
        for model, st in sorted(by_model.items(), key=lambda kv: -(kv[1].get("trades") or 0)):
            rows.append({
                "Method": f"{label} · {model}",
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    # Held-positions prompt A/B — CONCLUDED: the experiment was retired and ON
    # adopted, so `exit_hold_prompt` is now always True and the OFF cohort is
    # frozen history. Kept because it is the evidence for adopting ON. Each
    # trade's closing run stamped exit_hold_prompt; pre-experiment closes carry
    # no stamp and are excluded from both rows.
    hp = perf.get("hold_prompt_eval") or {}
    for key, label in (("on", "Exit eval · hold-prompt ON (adopted)"),
                       ("off", "Exit eval · hold-prompt OFF (retired cohort)")):
        st = hp.get(key)
        if st and st.get("trades"):
            rows.append({
                "Method": label,
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    # Long-horizon buy arm A/B — outcomes grouped by the per-run coin that
    # replaced combined_buy_score with the learned 5d stacker AND held those buys
    # longer (ml_arm_min_hold_days). The avg hold days rides the label so the
    # hold-lengthening — the whole point — is visible next to the return. ON should
    # show a longer hold; the bet is that captures the 5d+ edge the short book misses.
    lha = perf.get("ml_arm_eval") or {}
    for key, tag in (("on", "ON"), ("off", "OFF")):
        st = lha.get(key)
        if st and st.get("trades"):
            rows.append({
                "Method": f"ML-arm eval · ML combine {tag} ({st.get('avg_days_held', 0)}d hold)",
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    # Blind-synthesis A/B — entry outcomes grouped by whether the entry run hid
    # the aggregator's verdict from the synthesis prompt (ON = the LLM's own
    # independent judgment; OFF = the legacy echo-prone sighted prompt).
    bs = perf.get("blind_synthesis_eval") or {}
    for key, label in (("on", "Entry eval · blind-synthesis ON"),
                       ("off", "Entry eval · blind-synthesis OFF")):
        st = bs.get(key)
        if st and st.get("trades"):
            rows.append({
                "Method": label,
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    # Three-arm prompt bake-off, ledger view — CONCLUDED 2026-08-16, so these are
    # historical cohorts: only SIGHTED still accrues trades. Separate from the
    # boolean above because the dual arm also sets blind=False, so "OFF" merges
    # dual+sighted. The unbiased per-ticker version is the arm block further down,
    # which carries the verdict.
    for key, label in (("dual", "Entry eval · prompt arm DUAL-CASE (retired)"),
                       ("blind", "Entry eval · prompt arm BLIND (retired)"),
                       ("sighted", "Entry eval · prompt arm SIGHTED (live)")):
        st = (perf.get("synth_arm_eval") or {}).get(key)
        if st and st.get("trades"):
            rows.append({
                "Method": label,
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["Method", "Win rate %", "Rel win %", "vs base", "Rel n",
                  "Trades", "Avg return %"]],
        tooltip_header=_METHOD_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Method} contains "LLM"'}, "backgroundColor": "#eef2ff"},
            {"if": {"filter_query": '{Method} contains "hold-prompt"'}, "backgroundColor": "#fdf4ff"},
            {"if": {"filter_query": '{Method} contains "ML combine"'}, "backgroundColor": "#ecfdf5"},
            {"if": {"filter_query": '{Method} contains "blind-synthesis"'}, "backgroundColor": "#fefce8"},
            {"if": {"filter_query": '{Method} contains "prompt arm"'}, "backgroundColor": "#eff6ff"},
        ],
        **_TABLE_KW,
    ) if rows else html.Div("No per-method stats in this window yet.", style={"color": "#6b7280"})

    # ── Macro evaluation — aggregated decision layers (synthesis vs aggregator
    # vs method bundles), each scored on its full directional-call stream so the
    # LLM's confidence and the aggregator's confidence are directly comparable.
    macro_rows = [
        {
            "Layer": r["label"],
            "Win rate %": round(r["win_rate"], 1) if r.get("win_rate") is not None else None,
            "Trades": r.get("trades"),
            "Avg return %": round(r["avg_return"], 2) if r.get("avg_return") is not None else None,
        }
        for r in (perf.get("macro_eval") or [])
    ]
    macro_table = dash_table.DataTable(
        data=macro_rows,
        columns=[{"name": c, "id": c} for c in ["Layer", "Win rate %", "Trades", "Avg return %"]],
        tooltip_header=_MACRO_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Layer} contains "Synthesis"'}, "backgroundColor": "#eef2ff"},
            {"if": {"filter_query": '{Layer} contains "Aggregator"'}, "backgroundColor": "#ecfdf5"},
        ],
        **_TABLE_KW,
    ) if macro_rows else html.Div("No macro-layer stats in this window yet.", style={"color": "#6b7280"})

    # ── Decision funnel — per-stage evaluation (aggregator → synthesis → the
    # four actionable gates), each stage's survivors AND each gate's drops
    # scored on the same pseudo-trade basis so every step's marginal value
    # (filtered losers vs discarded winners) is directly visible.
    stage_rows = [
        {
            "Stage": r["label"],
            "Trades": r.get("trades"),
            "Win rate %": round(r["win_rate"], 1) if r.get("win_rate") is not None else None,
            "Avg return %": round(r["avg_return"], 2) if r.get("avg_return") is not None else None,
        }
        for r in (perf.get("stage_eval") or [])
    ]
    stage_table = dash_table.DataTable(
        data=stage_rows,
        columns=[{"name": c, "id": c} for c in ["Stage", "Trades", "Win rate %", "Avg return %"]],
        tooltip_header=_STAGE_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Stage} contains "✂"'}, "backgroundColor": "#fef2f2"},
            {"if": {"filter_query": '{Stage} contains "ACTIONABLE"'}, "backgroundColor": "#ecfdf5"},
        ],
        **_TABLE_KW,
    ) if stage_rows else html.Div("No stage-funnel stats in this window yet.", style={"color": "#6b7280"})

    return html.Div([
        dcc.Graph(figure=figures.method_winrate_fig(perf)),
        _h3("Return vs entry confidence",
            "Each dot is one trade: its entry confidence (x) against its return (y) — closed trades at their realised return, "
            "open trades (hollow diamonds) at their live mark-to-market; green = win, red = loss. Confidence still gates entry "
            "(Gate 1) and still sizes, but through a ramp deliberately COMPRESSED to a 1.5× cap (confidence_size_span 0.5) "
            "because this very plot measured it nearly uninformative — so a flat line here is the EXPECTED result, not a bug, "
            "and it is the reason breadth rather than confidence carries conviction in the sizing chain. What would be news is a "
            "clearly POSITIVE slope (confidence has started earning its span back) or a clearly NEGATIVE one (it is anti-predictive "
            "and the span should go to 0). ⚠ Mixed-era caution: the rank basis and the ML-combine arm each rescaled raw confidence, "
            "so points from different eras are not on one x-axis. Respects the window + session toggles above."),
        dcc.Graph(figure=figures.confidence_return_fig(perf)),
        _calibration_block(window_days, session, direction),
        _h3("Macro evaluation — decision layers (LLM synthesis vs aggregator vs bundles)",
            "Head-to-head performance of the aggregated decision layers, all scored on the SAME unbiased basis: "
            "every directional call each layer made — not just the few that became trades — entered at the call-time "
            "snapshot price and marked at the latest close through the real cost model. 'LLM Synthesis' is the final "
            "BUY/SELL call (all engines; the per-engine split is in the Model Evaluation table below); 'Aggregator' is "
            "the mechanical combined signal; each 'Bundle' is a method family voting by its summed score. This is the "
            "direct test of whether the LLM's confidence or the aggregator's confidence is the more reliable predictor. "
            "Respects the window + session toggles above."),
        macro_table,
        _h3("Decision funnel — per-stage performance (aggregator → LLM → gates 1-4)",
            "Every step of the decision pipeline evaluated on the SAME pseudo-trade basis: the mechanical "
            "aggregator's directional calls, the LLM synthesis stream, then the four actionable gates in "
            "execution order — Gate 1 regime confidence threshold, Gate 2 PANIC/RISK_OFF BUY-block, Gate 3 "
            "earnings blackout, Gate 4 tradeable-liquidity floor. Each '→ past Gate k' row is the surviving "
            "stream after that gate; the '✂' row under it is exactly what the gate discarded (gate outcomes "
            "are reconstructed exactly from each run's persisted threshold/allow_buys/actionable flags, plus "
            "the per-ticker gate_outcomes stamp on new runs). Read each gate by comparing its two rows: "
            "drops worse than survivors = the gate filters losers; drops better = it discards winners. "
            "The final green row is the actionable set the sizing layer actually received. "
            "Respects the window + session toggles above."),
        stage_table,
        _h3("Model evaluation — signal methods (solo simulation) & LLM engines",
            "Method rows: how each signal method would have performed deciding alone (each closed trade re-simulated as if only that method set the direction). "
            "Highlighted LLM rows: every BUY/SELL the engine recommended — executed or simulated — entered at the recommendation-time price, marked at the latest close, "
            "deduped to the engine's last call per ticker per day. ⚠ The engine A/B is OVER: routing is now 100% deepseek-v4-flash for both synthesis and sentiment "
            "(the pool is a one-model list, so the alternate branches are unreachable). Multi-engine rows are HISTORICAL — the bake-off concluded that no provider "
            "differs at either task across eleven paired tests, making engine choice a cost/latency decision. Rows stamped with the rule-based fallback are not an "
            "LLM at all and rank last by construction. Hover the column headers for details."),
        html.Div(_rel_baseline_note(), className="section-note"),
        table,
        _ticker_perf_block(window_days),
        _arm_eval_block(window_days),
    ])


# ── Per-ticker simulated performance (gate-independent) ────────────────────

_TICKER_PERF_TIPS = {
    "Ticker": "The scored name. Filter this column to find a specific ticker; filter Source to isolate a discovery channel (e.g. watchlist).",
    "Source": "Which discovery source first surfaced the ticker. 'watchlist' = a name you pinned in STOCK_WATCHLIST — always in the universe, never dropped by the discovery liquidity gate.",
    "Scored": "Ticker-days the name was scored at all. Pinned names accrue this every tick regardless of what the gates decide.",
    "View": "Of those, days the combined score carried an actual direction (|score| ≥ 0.02). The rest are no-view days and are EXCLUDED from the returns, not counted as zero.",
    "Avg score": "Mean combined_score. Positive = the system leans bullish on this name overall.",
    "Avg conf": "Mean confidence. Compare against the ~0.85 actionable bar to see how far off being tradeable a name typically is.",
    "Ret pivot %": "Mean oriented return to the next H/L pivot extreme (the standing decision basis) — % from the signal day's close to the next confirmed swing high/low, in the signal's direction.",
    "Hit pivot %": "Share of pivot-settled view-days where the signal's direction matched the sign of the move to the next pivot extreme.",
    "Ret 1d %": "SIMULATED: mean return if the system had taken the signal's own direction each day and held 1 session. A bearish call on a stock that fell counts as a WIN.",
    "Hit 1d %": "Share of 1-day observations where the signal's direction was right.",
    "Ret 5d %": "Same, held 5 sessions — the swing horizon most of the calibration targets.",
    "Hit 5d %": "Share of 5-day observations the signal's direction got right.",
    "Ret 10d %": "Same, held 10 sessions.",
    "Hit 10d %": "Share of 10-day observations the signal's direction got right.",
    "Recs": "Times the LLM produced any recommendation for this ticker (including HOLD/WATCH).",
    "Dir recs": "Of those, how many were directional BUY/SELL calls.",
    "Actionable": "How many survived ALL the gates (confidence, agreement, BUY-block, earnings blackout, liquidity floor, overextension). A good simulated return with Actionable = 0 means the gates are consistently declining this name — that gap is the interesting part.",
    "Trades": "Real trades opened on this ticker in the ledger (gate survivors only).",
    "Open": "Currently open positions on this ticker.",
    "Real ret %": "Mean realized return of those real trades, through the full cost model. Differs from the simulated columns because it only includes gate survivors and pays real costs.",
}

_TICKER_PERF_COLS = ["Ticker", "Source", "Scored", "View", "Avg score", "Avg conf",
                     "Ret pivot %", "Hit pivot %",
                     "Ret 1d %", "Hit 1d %", "Ret 5d %", "Hit 5d %",
                     "Ret 10d %", "Hit 10d %",
                     "Recs", "Dir recs", "Actionable", "Trades", "Open", "Real ret %"]


def _ticker_perf_block(window_days):
    """Every scored ticker's own record, whether or not the gates let it trade."""
    df = data.ticker_perf(days=window_days)
    if df is None or df.empty:
        return html.Div([
            _h3("Per-ticker simulated performance",
                "Every scored ticker's own record, independent of the gates."),
            html.Div("No scored tickers in this window yet.", style={"color": "#6b7280"}),
        ])

    rows = [{
        "Ticker": r["ticker"], "Source": r.get("source"),
        "Scored": r["signal_days"], "View": r["view_days"],
        "Avg score": r["avg_score"], "Avg conf": r.get("avg_conf"),
        "Ret pivot %": r.get("ret_pv"), "Hit pivot %": r.get("hit_pv"),
        "Ret 1d %": r.get("ret_1d"), "Hit 1d %": r.get("hit_1d"),
        "Ret 5d %": r.get("ret_5d"), "Hit 5d %": r.get("hit_5d"),
        "Ret 10d %": r.get("ret_10d"), "Hit 10d %": r.get("hit_10d"),
        "Recs": r.get("recs"), "Dir recs": r.get("dir_recs"),
        "Actionable": r.get("actionable"), "Trades": r.get("trades"),
        "Open": r.get("open"), "Real ret %": r.get("real_ret"),
    } for _, r in df.iterrows()]

    return html.Div([
        _h3("Per-ticker simulated performance — every scored name, gates or not",
            "The plainest question about a name you deliberately pinned: how is the strategy doing on THIS ticker? "
            "Every OTHER performance table aggregates — by method, by discovery source, by feature bucket, by realized "
            "trade — so none of them answer it. The return columns are SIMULATED over every scored ticker-day (if the "
            "system had taken the combined score's direction that day, what did it earn?), which is deliberately "
            "gate-INDEPENDENT: the trade ledger only contains names that survived Gates 1-5, so a pinned ticker that "
            "never clears the confidence bar would be invisible there while still being scored every tick. "
            "The funnel columns on the right (Recs → Dir recs → Actionable → Trades) show where each name actually "
            "stops. Sort by Ret 5d %, or filter Source to 'watchlist' to isolate your pinned names. "
            "Respects the window toggle above."),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in _TICKER_PERF_COLS],
            tooltip_header=_TICKER_PERF_TIPS,
            style_data_conditional=[
                {"if": {"filter_query": '{Source} = "watchlist"'},
                 "backgroundColor": "#eef2ff", "fontWeight": "600"},
                {"if": {"filter_query": "{Ret 5d %} < 0", "column_id": "Ret 5d %"},
                 "color": "#b91c1c"},
                {"if": {"filter_query": "{Ret 5d %} > 0", "column_id": "Ret 5d %"},
                 "color": "#047857"},
                {"if": {"filter_query": "{Actionable} = 0 && {Dir recs} > 0"},
                 "borderLeft": "3px solid #f59e0b"},
            ],
            **_TABLE_KW,
        ),
    ])


# ── Synthesis prompt-arm bake-off — CONCLUDED 2026-08-16 ───────────────────
#
# The dual/blind/sighted experiment ran 2026-07-25 → 2026-08-16 and ANSWERED:
# re-evaluated on the pivot basis with day-clustered statistics, no arm is
# distinguishable. The live pipeline is sighted-only (both shares 0.0,
# enable_shadow_arms off), so these tables are a CLOSED RESULT, not a running
# A/B — the surrounding copy says so, because a table that reads as live invites
# someone to keep waiting for an answer that already arrived.

_ARM_CONCLUDED_TOOLTIP = (
    "CONCLUDED 2026-08-16 — historical, no new rows. The dual-case / blind / sighted bake-off ran "
    "2026-07-25 → 2026-08-16: each tick every arm was asked about every ticker (one live, the rest "
    "shadow), so arms were compared on the same ticker-days with the same engine and context and the "
    "prompt as the only difference. Re-evaluated on the PIVOT basis with day-clustered statistics "
    "(22,534 labeled rows, 16 settled days) NO ARM IS DISTINGUISHABLE: the paired disagreement subset "
    "gives |day-t| < 1 for all three pairs, the nominal winner wins only 38–44% of DAYS, and "
    "dual-vs-blind flips sign between row- and day-weighting. Sighted was kept on tiebreakers — "
    "shortest per-ticker block and the least anti-informative confidence (which feeds sizing) — not on "
    "P&L. The renderer and shadow machinery are kept and tested under explicit opt-in; raise a share "
    "above 0 (and enable_shadow_arms) to re-open.")

_ARM_SUMMARY_TIPS = {
    "Arm": "Which synthesis prompt produced the call. Dual-case = BULL and BEAR cases side by side, each from its own vetted method set. "
           "Blind = the aggregator's verdict hidden. Sighted = the prompt showing the verdict — the arm that remains live.",
    "Calls": "Ticker-days this arm answered while the experiment ran. Every arm was asked about every ticker each tick (one live, the rest shadow), so these are near-identical — that is what makes the paired table below possible. Frozen: no new calls accrue.",
    "Buy %": "Share of its calls that were BUY.",
    "Sell %": "Share of its calls that were SELL.",
    "Flat %": "Share it declined to trade (HOLD/WATCH). Not a failure — the dual-case arm is explicitly told that declining is a valid output, and a decline earns 0 rather than a loss.",
    "Strategy ret %": "Mean forward return treating a decline as 0 (no position, no P&L). This is the arm AS A STRATEGY: declining a loser genuinely beats taking it.",
    "Dir ret %": "Mean forward return over only the calls it actually made directionally — how good its picks were, ignoring how often it picked.",
    "Dir win %": "Share of its directional calls that moved its way.",
    "Conf IC": "Spearman correlation between the arm's own stated confidence and its realized oriented return. Positive = its confidence ranks its own calls; ~0 = confidence carries no information (the established finding here).",
}

_ARM_PAIR_TIPS = {
    "Pair": "The two arms compared head-to-head on ticker-days BOTH answered.",
    "Common": "Ticker-days both arms were asked about — the paired sample.",
    "Agree %": "How often the two arms took the same side (including both declining). A high number means the prompt rarely changes the decision.",
    "Disagree": "Ticker-days the two arms took DIFFERENT sides. This is the real sample size of the experiment — the only rows where the prompt changed anything. Everything else is a shared call neither arm can take credit for.",
    "A ret %": "First arm's mean return on the disagreement rows only.",
    "B ret %": "Second arm's mean return on the same disagreement rows.",
    "Edge %": "A minus B on the disagreement rows. Positive = the first arm was right where they differed. Read this as the arm's value; treat a small Disagree count as no answer yet.",
}


def _arm_eval_block(window_days):
    """Per-ticker comparison of the three synthesis prompt arms.

    Deliberately shows the PAIRED table alongside the per-arm one: the unpaired
    view is the shape of comparison that produced the 2026-07-22 bake-off's
    window artifacts, so it is presented as context rather than as the answer.
    """
    res = data.arm_eval(days=window_days) or {}
    if not res.get("calls"):
        return html.Div([
            _h3("Synthesis prompt arms — CONCLUDED 2026-08-16 (historical)",
                _ARM_CONCLUDED_TOOLTIP),
            html.Div("No arm calls recorded in this window. The experiment is closed and "
                     "no new arm rows accrue (shares 0.0, ENABLE_SHADOW_ARMS off) — widen "
                     "the window to see the accrued history.",
                     className="empty-note"),
        ])

    _hs = res.get("horizons") or [1]
    horizon = "pv" if "pv" in _hs else (5 if 5 in _hs else _hs[0])

    srows = [{
        "Arm": r["label"], "Calls": r["calls"],
        "Buy %": r["buy_pct"], "Sell %": r["sell_pct"], "Flat %": r["flat_pct"],
        "Strategy ret %": r["mean_ret"], "Dir ret %": r["dir_ret"],
        "Dir win %": r["dir_win"],
        "Conf IC": round(r["conf_ic"], 3) if r.get("conf_ic") is not None else None,
    } for r in res["summary"].get(horizon, [])]

    prows = [{
        "Pair": p["pair"], "Common": p["common"], "Agree %": p["agree_pct"],
        "Disagree": p["disagree"], "A ret %": p["a_ret"], "B ret %": p["b_ret"],
        "Edge %": p["edge"],
    } for p in res["pairs"].get(horizon, [])]

    return html.Div([
        _h3("Synthesis prompt arms — CONCLUDED 2026-08-16, historical "
            + ("(pivot target)" if horizon == "pv" else f"({horizon}-day)"),
            _ARM_CONCLUDED_TOOLTIP
            + f" Accrued sample: {res['calls']:,} calls, {res.get('shadow', 0):,} of them shadow."),
        html.Div("Closed experiment — the live pipeline is SIGHTED-only and no new arm rows "
                 "accrue. Kept because the accrued history is the evidence behind that "
                 "decision; the machinery is revivable by raising a share above 0.",
                 className="section-note"),
        dash_table.DataTable(
            data=srows,
            columns=[{"name": c, "id": c} for c in
                     ["Arm", "Calls", "Buy %", "Sell %", "Flat %",
                      "Strategy ret %", "Dir ret %", "Dir win %", "Conf IC"]],
            tooltip_header=_ARM_SUMMARY_TIPS,
            style_data_conditional=[
                {"if": {"filter_query": '{Arm} contains "Dual"'}, "backgroundColor": "#eef2ff"},
            ],
            **_TABLE_KW,
        ) if srows else html.Div("No scored arm calls yet.", style={"color": "#6b7280"}),
        _h3("Head-to-head — where the arms actually disagreed",
            "The row that mattered. Two arms agreeing on a ticker tells you nothing about either one, so the return "
            "columns are computed ONLY over the ticker-days where they took different sides. A high Agree % means the "
            "prompt rarely changes the decision — the blind A/B showed the echo rate moves only 94.5% → 91.4%, "
            "i.e. the model echoes because it reads the same method scores, not because it sees the verdict. "
            "⚠ Read the Edge % column with day-clustered eyes: the FINAL verdict (2026-08-16) is that no pair separates "
            "— |day-t| < 1 for all three, and the nominal winner wins only 38–44% of DAYS. A non-zero Edge % here is "
            "the row-weighted point estimate, which is exactly the statistic that overstated these arms."),
        dash_table.DataTable(
            data=prows,
            columns=[{"name": c, "id": c} for c in
                     ["Pair", "Common", "Agree %", "Disagree",
                      "A ret %", "B ret %", "Edge %"]],
            tooltip_header=_ARM_PAIR_TIPS,
            **_TABLE_KW,
        ) if prows else html.Div("No paired arm calls yet.", style={"color": "#6b7280"}),
    ])


# ── Tab: Exit Performance ──────────────────────────────────────────────────

_EXIT_PERF_TOOLTIP = (
    "Every held position is re-scored each tick with signed HOLD-CONVICTION per exit "
    "method (+ = keep running, − = reverse/exit) — the `exit_signals` panel. The table "
    "shows each method's ACTIVATION EVENTS: the tick the method first turned against the "
    "position (its conviction crossed into negative — 'the exit fired'), attributed to "
    "that tick's session, so a method repeating 'exit' for days counts once and the "
    "Session buckets are a true partition (All sessions = the sum of the four). Each "
    "activation is joined to the position's DIRECTION-ORIENTED forward return from that "
    "tick (a short's forward return is negated) and scored in the direction the method "
    "CALLED (sign(score)×forward), so the usual reading holds: 'Win@ %' > 50 = after the "
    "method said get out, the position usually DID move adversely (the exit was right; "
    "below 50 = it fires too early), 'Ret@ %' positive = the average post-activation move "
    "vindicated the exit, 'IC@' positive = deeper exit-conviction ⇒ more adverse "
    "subsequent move. Per horizon (pv…1m): 'n@' = activations with a forward return, "
    "'IC std@' / 'ICIR@' = the IC's reliability (per-day; needs several days). The "
    "synthesized `llm_review` row is history-backed from `trade_reviews`; the rest "
    "accrue as the panel fills. Forward-collected — judge nothing on a thin n.")

_EXIT_SHADOW_TOOLTIP = (
    "The SIMULATED exit book: every scored ticker (the signals panel) treated as a hypothetical "
    "position held in its own aggregate direction, with each position-independent exit method scored "
    "as a signed hold-conviction (method score × the ticker's direction; aggregator = combined_score). "
    "Shown as ACTIVATION EVENTS — the run where the method's conviction first crossed negative for "
    "that hypothetical position (a direction flip starts a new position), attributed to that run's "
    "session, so the Session buckets are a true partition (All sessions = the sum of the four). Each "
    "activation joins the direction-oriented forward return through the SAME engine as the held book, "
    "but over the WHOLE UNIVERSE instead of only the gate-selected positions we actually held — "
    "escaping the held book's tiny, selection-biased sample. Read like the held table: Win% > 50 / "
    "positive Ret / positive IC = the method's exit calls were vindicated by the subsequent move. "
    "Only the position-INDEPENDENT methods appear here (aggregator + the signal-methods-as-exits); "
    "`horizon` and the synthesized `llm_review` need a real entry, so they exist ONLY in the Held view.")


# ── Exit-perf source toggle (real held book vs simulated universe shadow) ────
_EXIT_SOURCE_OPTIONS = [
    {"label": "Held positions (ledger)", "value": "held"},
    {"label": "All scored tickers (simulated)", "value": "shadow"},
]


def _exit_source_toggle(component_id: str) -> html.Div:
    """The exit-IC evidence base. Held = the exit methods scored on the positions we
    actually held (real book; the only place horizon / llm_review exist). Simulated =
    every scored ticker as a hypothetical position held in its aggregate direction —
    the position-independent methods over the whole universe (large, unbiased)."""
    return _filter_row(
        "Source",
        "Held positions (ledger): each exit method's ACTIVATIONS against the positions we ACTUALLY held — the tick it first said 'get out' (conviction crossed negative), attributed to that tick's session. The real book (small, selection-biased), and the ONLY view with horizon + the synthesized llm_review. "
        "All scored tickers (simulated): the same activation events over EVERY scored ticker treated as a hypothetical position held in its aggregate direction (aggregator + the signal-methods-as-exits) — the large, unbiased sample backfilled from the signals panel. horizon / llm_review are held-only and don't appear here.",
        component_id, _EXIT_SOURCE_OPTIONS, "held")


def _exit_perf_section(window_days, source="held", sel_horizons=None, sel_metrics=None,
                       session=None, direction=None):
    """Per-exit-method IC / win / ret, grouped by the two exit categories. ``source``
    picks the evidence base: 'held' = the real exit_signals + trade_reviews book;
    'shadow' = the simulated universe (all scored tickers). ``session`` filters by
    the session the review happened in (the exit decision's firing moment);
    ``direction`` by the (hypothetical) position's side. A caption states which
    basis + filters are shown so the view is never ambiguous. Reuses the
    simulated-perf table renderer."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.exit_methods import EXIT_CATEGORY_ORDER, EXIT_METHOD_LABELS
    sel_horizons = sel_horizons or list(_SIM_HORIZONS)
    sel_metrics = sel_metrics or list(_SIM_METRIC_ORDER)
    shadow = (source == "shadow")
    if shadow:
        df = data.shadow_exit_method_perf(days=window_days, session=session, direction=direction)
        heading = _h3("Exit-method performance — ALL scored tickers (simulated shadow book)",
                      _EXIT_SHADOW_TOOLTIP)
        caption = ("Source: SIMULATED — every scored ticker as a hypothetical position held in its "
                   "aggregate direction, over the whole universe (backfilled from the signals panel). "
                   "Position-independent methods only; horizon + llm_review are held-only and not shown here.")
        empty_msg = ("No simulated exit returns match this window/session/direction yet — the signals "
                     "panel needs forward-return history "
                     "(warm it with `python -m src.analysis.signal_panel --refresh`).")
    else:
        df = data.exit_method_perf(days=window_days, session=session, direction=direction)
        heading = _h3("Exit-method performance — HELD positions (ledger)", _EXIT_PERF_TOOLTIP)
        caption = ("Source: HELD (ledger) — the exit methods scored on the positions we ACTUALLY held "
                   "each tick. The only view with horizon + the synthesized llm_review; small + "
                   "selection-biased. Switch Source to 'All scored tickers' for the large simulated sample.")
        empty_msg = ("No exit-method forward returns match this window/session/direction yet. The "
                     "synthesized hold-review row populates from `trade_reviews`; the decomposed "
                     "methods accrue once positions are held.")
    if session:
        caption += f" — Session filter: {session} (review moment)."
    if direction:
        caption += f" — Direction filter: {direction} positions only."
    cap = html.Div(caption, style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8})
    if df is None or getattr(df, "empty", True):
        return html.Div([heading, cap, html.Div(empty_msg, style={"color": "#6b7280"})])
    labels = dict(METHOD_LABELS)
    labels.update(EXIT_METHOD_LABELS)
    children = [heading, cap]
    has_cat = "category" in df.columns
    for category in EXIT_CATEGORY_ORDER:
        subset = df[df["category"] == category] if has_cat else df
        if subset is None or subset.empty:
            continue
        children.append(html.Div(category, style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#475569"}))
        children.append(_sim_perf_table(subset, labels, sel_horizons, sel_metrics))
        if not has_cat:
            break
    return html.Div(children)


def _exit_reason_block(session=None, direction=None):
    """Realized outcome per exit RULE (exit_reason) over closed trades — the
    concrete, ledger-based companion to the forward-looking IC table above.
    ``session`` filters by the session the trade EXITED in; ``direction`` by
    the position's side."""
    rows = data.exit_reason_breakdown(session=session, direction=direction)
    if not rows:
        return html.Div("No closed trades match this session/direction yet.",
                        style={"color": "#6b7280"})
    data_rows = [{
        "reason": r["exit_reason"], "trades": r["trades"], "win": r["win_rate"],
        "avg": r["avg_return"], "median": r.get("median_return"),
        "compound": r["compound_return"], "best": r["best"], "worst": r["worst"],
    } for r in rows]
    cols = [
        {"name": "Exit reason", "id": "reason"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Win rate %", "id": "win", "type": "numeric", "format": _NUM2},
        {"name": "Avg return %", "id": "avg", "type": "numeric", "format": _NUM2},
        {"name": "Median %", "id": "median", "type": "numeric", "format": _NUM2},
        {"name": "Compound %", "id": "compound", "type": "numeric", "format": _NUM2},
        {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
        {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
    ]
    return dash_table.DataTable(data=data_rows, columns=cols, **_TABLE_KW)


def _exit_forward_block(session=None, direction=None):
    """Post-exit forward returns — what each CLOSED trade would have earned if
    held 1/3/5/10 more sessions, anchored at the actual exit fill and oriented
    by the position's side. Positive = the exit left money on the table."""
    rep = data.exit_forward(session=session, direction=direction)
    if not rep.get("n"):
        return html.Div(rep.get("verdict") or "No closed trades with post-exit bars yet.",
                        style={"color": "#6b7280"})
    hs = rep["horizons"]

    reason_rows = [{
        "reason": r["exit_reason"], "trades": r["trades"],
        "mean_pv": r.get("mean_pv"), "pos_pv": r.get("pct_pos_pv"),
        **{f"mean_{h}": r.get(f"mean_{h}d") for h in hs},
        **{f"pos_{h}": r.get(f"pct_pos_{h}d") for h in hs},
    } for r in rep["by_reason"] + [{"exit_reason": "ALL exits", **rep["overall"]}]]
    reason_cols = ([{"name": "Exit reason", "id": "reason"},
                    {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
                    {"name": "Mean →pivot %", "id": "mean_pv", "type": "numeric", "format": _NUM2}]
                   + [{"name": f"Mean +{h}d %", "id": f"mean_{h}", "type": "numeric", "format": _NUM2}
                      for h in hs]
                   + [{"name": "%+ →pivot", "id": "pos_pv", "type": "numeric", "format": _NUM1}]
                   + [{"name": f"%+ @{h}d", "id": f"pos_{h}", "type": "numeric", "format": _NUM1}
                      for h in hs])

    trade_rows = [{
        "ticker": r["ticker"], "exit_date": r["exit_date"], "ret": r["return_pct"],
        "fwd_pv": r.get("fwd_pv"),
        **{f"fwd_{h}": r.get(f"fwd_{h}d") for h in hs},
        "reason": r["exit_reason"],
    } for r in rep["per_trade"]]
    trade_cols = ([{"name": "Ticker", "id": "ticker"},
                   {"name": "Exit date", "id": "exit_date"},
                   {"name": "Realized %", "id": "ret", "type": "numeric", "format": _NUM2},
                   {"name": "Fwd →pivot %", "id": "fwd_pv", "type": "numeric", "format": _NUM2}]
                  + [{"name": f"Fwd +{h}d %", "id": f"fwd_{h}", "type": "numeric", "format": _NUM2}
                     for h in hs]
                  + [{"name": "Exit reason", "id": "reason"}])

    pending = (f" · {rep['n_pending']} exit(s) pending forward bars"
               if rep.get("n_pending") else "")
    return html.Div([
        html.Div(f"{rep['verdict']}  ({rep['n']} closed trade(s) with forward bars{pending})",
                 style={"color": "#475569", "marginBottom": 8}),
        dash_table.DataTable(data=reason_rows, columns=reason_cols, **_TABLE_KW),
        html.Div("Per-trade detail (most recent exits first)", style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#475569"}),
        dash_table.DataTable(data=trade_rows, columns=trade_cols, **_TABLE_KW),
    ])


_EXIT_SESSION_TITLE = (
    "Filter the exit analyses to a US-market session — Regular hours (09:30–16:00 ET), "
    "Pre-market (04:00–09:30), After-hours (16:00–20:00), or Overnight (20:00–04:00). "
    "For the exit-method tables this is the session the REVIEW happened in (the moment "
    "the exit decision would fire); for the exit-reason table below it is the session "
    "the trade actually EXITED in.")


_MC_EXITS_TOOLTIP = (
    "Monte Carlo exit-timing test — does each exit rule TIME its exits better than random, or "
    "is its realized outcome just what any exit in the same windows would have gotten? For every "
    "CLOSED trade the feasible exit window is each session close from the first session after "
    "entry through the actual hold + 10 more sessions; the RANDOM-EXIT NULL draws one uniform "
    "random exit per trade per simulation (2000 sims, fixed seed) and records the group's mean "
    "gross oriented return. Both arms anchor entry at the real entry fill and exit at SESSION "
    "CLOSES (the actual arm at the actual exit date's close) so the comparison is apples-to-"
    "apples; gross of costs (both arms pay the same one-way exit cost). 'Percentile' = where the "
    "rule's actual mean lands inside its own random-exit distribution: ≥ 95 ⇒ the rule genuinely "
    "times exits (green); ≤ 5 ⇒ RANDOM exits would have beaten the rule (red — the rule "
    "destroys timing value); anything between ≈ the rule adds no measurable timing skill — its "
    "realized P&L is explained by WHICH trades it closed, not WHEN. p(random ≥ actual) is the "
    "one-sided probability. Trades without enough cached sessions are skipped, never guessed. "
    "Small groups (n < ~10) are noise — judge only as trades accrue.")


def _exit_mc_block(session=None, direction=None):
    """Exit-timing-vs-random Monte Carlo per exit rule (src/analysis/monte_carlo.py)."""
    rep = data.monte_carlo_exits(session=session, direction=direction)
    if not rep.get("n"):
        return html.Div(rep.get("verdict") or "No closed trades with cached exit windows yet.",
                        style={"color": "#6b7280"})
    rows = [{
        "reason": r["reason"], "trades": r["trades"],
        "actual": r["actual_mean"], "null": r["null_mean"],
        "null_ci": f"{r['null_lo']:.2f} – {r['null_hi']:.2f}",
        "pctile": r["percentile"], "p_rand": r["p_random_beats"],
        "verdict": r["verdict"],
    } for r in rep["rows"]]
    cols = [
        {"name": "Exit reason", "id": "reason"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Actual mean %", "id": "actual", "type": "numeric", "format": _NUM2},
        {"name": "Random-exit mean %", "id": "null", "type": "numeric", "format": _NUM2},
        {"name": "Null CI 5–95%", "id": "null_ci"},
        {"name": "Percentile", "id": "pctile", "type": "numeric", "format": _NUM1},
        {"name": "p (random ≥ actual)", "id": "p_rand", "type": "numeric", "format": _NUM3},
        {"name": "Verdict", "id": "verdict"},
    ]
    cond = [
        {"if": {"filter_query": "{pctile} >= 95", "column_id": "pctile"}, "color": figures.POS},
        {"if": {"filter_query": "{pctile} <= 5", "column_id": "pctile"}, "color": figures.NEG},
        {"if": {"filter_query": '{verdict} contains "BETTER"', "column_id": "verdict"},
         "color": figures.POS},
        {"if": {"filter_query": '{verdict} contains "BEATEN"', "column_id": "verdict"},
         "color": figures.NEG},
        {"if": {"filter_query": '{reason} = "ALL exits"'}, "backgroundColor": "#eff6ff"},
    ]
    skipped = (f" · {rep['n_skipped']} trade(s) skipped (no cached window)"
               if rep.get("n_skipped") else "")
    return html.Div([
        html.Div(f"{rep['n']} closed trade(s) in the MC{skipped}",
                 style={"color": "#475569", "marginBottom": 8}),
        dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond,
                             **_TABLE_KW),
    ])


def _exit_perf_tab():
    return html.Div([
        html.Div([
            _window_toggle("exit-window"),
            _session_toggle("exit-session", title=_EXIT_SESSION_TITLE),
            _direction_toggle("exit-direction"),
            _exit_source_toggle("exit-source"),
            _sim_column_filters("exit-horizons", "exit-metrics"),
        ], className="filter-bar"),
        dcc.Loading(html.Div(id="exit-body")),
    ])


@app.callback(Output("exit-body", "children"),
              Input("exit-window", "value"), Input("exit-session", "value"),
              Input("exit-direction", "value"), Input("exit-source", "value"),
              Input("exit-horizons", "value"), Input("exit-metrics", "value"))
def _exit_body(window_value, session_value, direction_value, source_value,
               sim_horizons, sim_metrics):
    session = _session_value(session_value)
    direction = _direction_value(direction_value)
    return html.Div([
        _safe(lambda: _exit_perf_section(_window_days(window_value), source_value,
                                         sim_horizons, sim_metrics,
                                         session=session, direction=direction)),
        _h3("Exit-reason outcomes — realized P&L by exit rule (closed ledger trades)",
            "For every CLOSED trade, the realized return grouped by the exit_reason that "
            "fired. ⚠ This table is CUMULATIVE history, so it still carries rows from two "
            "RETIRED rules — llm_confidence_loss (OFF; the post-exit forward returns "
            "condemned it) and mechanical_exit (OFF since 2026-08-02) — which can no longer "
            "produce new rows. Read those as a record of why they were turned off, not as "
            "live performance. Always the "
            "real ledger (independent of the Source toggle above) — the concrete realized "
            "outcome of each exit rule, companion to the forward-looking IC table. Honors "
            "the Session (session the trade EXITED in) and Direction toggles; the Window "
            "toggle does not apply (closed trades are few). Open trades excluded (no exit "
            "yet)."),
        _safe(lambda: _exit_reason_block(session=session, direction=direction)),
        _h3("Post-exit forward returns — what if we had held longer? (closed ledger trades)",
            "For every CLOSED trade, the oriented return the position would have earned had it "
            "stayed on 1/3/5/10 more trading sessions — anchored at the ACTUAL exit fill "
            "(sign × (close_{exit+N} / exit_price − 1); long +, short −), forward closes from the "
            "daily OHLCV cache (recently-exited tickers stay warmed by the EOD maintenance pass). "
            "Positive = the position kept moving our way after we left it (the exit left money on "
            "the table); negative = the exit dodged a drawdown. Gross of costs (holding defers the "
            "same exit cost rather than adding one). '%+ @Nd' = share of exits still going our way "
            "at that horizon — a rule with a persistently positive mean is firing too early. "
            "Grouped by exit rule; always the real ledger. Honors the Session (exit session) and "
            "Direction toggles; the Window toggle does not apply. Exits without forward bars yet "
            "(closed today / cache gap) are counted as pending, never guessed."),
        _safe(lambda: _exit_forward_block(session=session, direction=direction)),
        _h3("Exit timing vs random exits — Monte Carlo (closed ledger trades)",
            _MC_EXITS_TOOLTIP),
        _safe(lambda: _exit_mc_block(session=session, direction=direction)),
        _h3("Confidence-formula component isolation — held positions", _CONF_COMPONENTS_TOOLTIP),
        _safe(lambda: _exit_confidence_components_block(session=session, direction=direction)),
        _safe(_horizon_edge_section),
        _safe(_exit_policy_eval_section),
    ])


_HORIZON_EDGE_TOOLTIP = (
    "The realized EDGE-DECAY of combined_score by holding horizon — measured tick-by-tick, "
    "ticker-by-ticker over the whole signals panel (every scored ticker at every tick is a "
    "hypothetical entry in its signal's direction), restricted to the ACTIONABLE subset "
    "(confidence ≥ 0.85 — the traded population). This is the ground truth the horizon time-stop "
    "rests on, at thousands of observations where the held-position `horizon` IC can't reach "
    "(only ~5 real positions have ever outlived their window). Per horizon: 'n' = observations, "
    "'IC' = Spearman(combined_score, forward return), 'win %' = directional hit, 'edge %' = mean "
    "sign(score)×forward-return (the P&L of following the signal that long). A peak-then-decay "
    "shape (edge positive early, ≤0 later) justifies a time-stop at the decay point. The measured "
    "**edge window** (last horizon with positive edge) drives the `edge_decay` exit stop, which "
    "evidence-throttled raises the close floor once a position is held past it. Run-based; "
    "forward-collected — long horizons thin until the cache warms; ~one regime so far.")


def _horizon_edge_section():
    """The realized edge-decay curve + the calibrated edge-window that feeds the
    edge-decay time-stop."""
    res = data.horizon_edge_curve()
    curve = res.get("curve") if isinstance(res, dict) else None
    cal = res.get("cal") if isinstance(res, dict) else {}
    heading = _h3("Signal edge by holding horizon (edge-decay time-stop)", _HORIZON_EDGE_TOOLTIP)
    if curve is None or getattr(curve, "empty", True):
        return html.Div([heading, html.Div(
            "No forward-return history yet — accrues every run.", style={"color": "#6b7280"})])
    ed, strength, peak = cal.get("edge_days"), cal.get("strength", 0.0), cal.get("peak_day")
    cap = (f"Measured edge window: {ed} trading day(s) (peak ~{peak}d) · "
           f"time-stop strength {strength:.2f} (evidence-throttled)"
           if ed else "No edge-decay window measured yet (edge not yet observed to turn "
                      "negative) — the stop stays inert until it does.")
    rows = []
    for _, r in curve.iterrows():
        rows.append({"horizon": f"{int(r['horizon'])}d", "n": int(r["n"]),
                     "ic": round(float(r["ic"]), 3) if pd.notna(r["ic"]) else None,
                     "win": round(float(r["win"]), 1) if pd.notna(r["win"]) else None,
                     "edge": round(float(r["edge"]), 3) if pd.notna(r["edge"]) else None})
    cols = [{"name": "Hold", "id": "horizon"},
            {"name": "n", "id": "n", "type": "numeric", "format": _INT},
            {"name": "IC", "id": "ic", "type": "numeric", "format": _NUM2},
            {"name": "Win %", "id": "win", "type": "numeric", "format": _NUM2},
            {"name": "Edge %", "id": "edge", "type": "numeric", "format": _NUM2}]
    cond = []
    for c in ("edge", "ic"):
        cond += [
            {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
            {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
        ]
    cond += [
        {"if": {"filter_query": "{win} >= 50", "column_id": "win"}, "color": figures.POS},
        {"if": {"filter_query": "{win} < 50", "column_id": "win"}, "color": figures.NEG},
    ]
    return html.Div([heading,
                     html.Div(cap, style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8}),
                     dash_table.DataTable(data=rows, columns=cols,
                                          style_data_conditional=cond, **_TABLE_KW)])


_EXIT_POLICY_TOOLTIP = (
    "Offline EXIT policy evaluation — the COUNTERFACTUAL value of alternative CLOSE rules, "
    "the exit-side twin of the entry sizing comparison. Each held position-day is a "
    "close-vs-hold decision; the reward is what the position DID NEXT: holding captures its "
    "oriented forward return, closing captures 0. A good close rule therefore CLOSES the days "
    "whose forward return is about to go negative (cutting losers) and HOLDS the rest. "
    "'avg_fwd_on_close' is the mean oriented forward return of the days each rule closed — "
    "you WANT it negative (you avoided a drop). 'exit_alpha' = held_mean − allhold_mean: how "
    "much better the book you CARRY does than holding everything; > 0 means the rule earns its "
    "keep, and 'always hold' is the 0 baseline. This validates whether an exit-BREADTH or "
    "aggregator rule beats the current LLM-scalar close BEFORE any of it is wired live. "
    "Replayed over the exit_signals panel (every held tick, deduped to last-per-day) + OHLCV "
    "forward returns — but that panel is NEW, so this fills in slowly; judge nothing until it "
    "spans many days (info_ratio populates only after >1 day).")


def _exit_policy_eval_section():
    """Counterfactual close-rule comparison — does exit-breadth / aggregator beat
    the current LLM-scalar close? Shown at 1-day and 5-day horizons."""
    children = [_h3("Close-rule comparison — counterfactual (offline exit eval)",
                    _EXIT_POLICY_TOOLTIP)]
    any_data = False
    for h in (1, 5):
        df = data.exit_policy_comparison(days=90, horizon=h)
        if df is None or getattr(df, "empty", True):
            continue
        any_data = True
        rows = [{
            "policy": r["policy"], "decisions": r.get("n_decisions"), "days": r.get("n_days"),
            "close_pct": r.get("close_rate"), "fwd_on_close": r.get("avg_fwd_on_close"),
            "fwd_on_hold": r.get("avg_fwd_on_hold"), "alpha": r.get("exit_alpha"),
            "ir": r.get("info_ratio"),
        } for _, r in df.iterrows()]
        cols = [
            {"name": "Close rule", "id": "policy"},
            {"name": "Decisions", "id": "decisions", "type": "numeric", "format": _INT},
            {"name": "Days", "id": "days", "type": "numeric", "format": _INT},
            {"name": "Close %", "id": "close_pct", "type": "numeric", "format": _NUM2},
            {"name": "Fwd on close %", "id": "fwd_on_close", "type": "numeric", "format": _NUM2},
            {"name": "Fwd on hold %", "id": "fwd_on_hold", "type": "numeric", "format": _NUM2},
            {"name": "Exit alpha %", "id": "alpha", "type": "numeric", "format": _NUM2},
            {"name": "Info ratio", "id": "ir", "type": "numeric", "format": _NUM2},
        ]
        children.append(html.Div(f"{h}-day horizon", style={
            "fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#475569"}))
        children.append(dash_table.DataTable(
            data=rows, columns=cols,
            style_data_conditional=[
                {"if": {"filter_query": "{alpha} > 0", "column_id": "alpha"},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{alpha} < 0", "column_id": "alpha"},
                 "color": figures.NEG, "fontWeight": "bold"},
                # A good close: forward return on the days it cut is negative.
                {"if": {"filter_query": "{fwd_on_close} < 0", "column_id": "fwd_on_close"},
                 "color": figures.POS},
                {"if": {"filter_query": '{policy} contains "always hold"'},
                 "backgroundColor": "#eff6ff"},
            ],
            **_TABLE_KW))
    if not any_data:
        children.append(html.Div(
            "No decidable exit decisions yet — the exit_signals panel needs forward-return "
            "history (it is newer than the entry panel; accrues as positions are held and the "
            "OHLCV cache warms past each review day).", style={"color": "#6b7280"}))
    return html.Div(children)


# ── Tab 3: Returns ─────────────────────────────────────────────────────────

def _held_disp(t: dict) -> str:
    """Wall-clock holding time as a compact ``2d 5h`` / ``6h`` / ``45m`` string.

    Open positions measure entry → now; closed ones entry → exit. Falls back
    to the trading-days count (``Nd``) for legacy date-only rows. An entry
    timestamp in the future (overnight decisions snap execution to the next
    session open) clamps to 0.
    """
    def _parse(iso):
        dt = datetime.fromisoformat(str(iso))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    try:
        start = _parse(t.get("entry_datetime"))
        end = _parse(t.get("exit_datetime")) if t.get("status") == "CLOSED" \
            else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        d = t.get("days_held")
        return f"{int(d)}d" if d is not None else ""
    minutes = max(0, int((end - start).total_seconds() // 60))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    return f"{hours}h" if hours else f"{mins}m"


def _ibkr_leg_disp(t: dict, prefix: str) -> str:
    """Compact label for one broker order leg ('broker_' / 'broker_exit_'):
    did the order really go through? Raw statuses are mapped to a handful of
    glyph-led states so the column scans at a glance."""
    status = str(t.get(f"{prefix}status") or "").strip()
    if not status:
        if prefix == "broker_exit_" and t.get("broker_order_id") and t.get("status") == "CLOSED":
            return "⏳ pending"     # ledger closed; the exit goes out next sync
        return "" if (prefix == "broker_exit_" and t.get("status") == "OPEN") else "–"
    qty = int(t.get(f"{prefix}fill_qty") or 0)
    req = int(t.get(f"{prefix}requested_qty") or 0)
    if status == "Filled":
        return f"✓ filled {qty}" if qty else "✓ filled"
    if status in ("Submitted", "PreSubmitted", "PendingSubmit"):
        return f"⏳ partial {qty}/{req}" if qty else "⏳ working"
    if status in ("STALE_CANCELLED", "EXPIRED"):
        return "↻ re-anchoring"     # tick-scoped cancel; resubmits at the current mark
    if status == "DUPLICATE_REF_NOT_SUBMITTED":
        return "– duplicate, not sent"
    if status == "NOTHING_TO_CLOSE":
        return "– nothing held"
    if status == "DRYRUN":
        return "dry-run"
    if status.upper().startswith("SKIPPED"):
        return "– " + status.replace("_", " ").lower()
    if status in ("Cancelled", "ApiCancelled", "Inactive"):
        reason = t.get(f"{prefix}cancel_reason")
        return f"✕ cancelled ({reason})" if reason else "✕ cancelled"
    if status in ("RESTORED_NOT_SUBMITTED", "RESTORED_ADOPTED"):
        return "restored"
    return f"✗ {status}"            # rejects / connection failures / raw errors


def _pct4(x) -> str:
    """Percent with up to 3 decimals (one-way fees are small — 0.4% not 0%)."""
    if x is None:
        return "–"
    try:
        return f"{x:.3f}%"
    except (TypeError, ValueError):
        return str(x)


def _trades_table(trades: list, table_id: str | None = None):
    """Render a trade ledger as a DataTable.

    When ``table_id`` is given the table gets that ``id`` and each row gets an
    ``id`` equal to its ticker, so an ``active_cell`` click resolves to the
    ticker robustly (survives native sort / filter / pagination) — used by the
    Returns tab to chart the clicked ticker's confidence-over-time plot below.
    """
    if not trades:
        return html.Div("None.", style={"color": "#6b7280", "marginBottom": 12})
    df = pd.DataFrame(trades)
    # Show entry/exit as Eastern-time date + time (HH:MM); fall back to the
    # date-only field for any legacy row missing the full datetime.
    df["entry_dt"] = [_fmt_et(t.get("entry_datetime")) or (t.get("entry_date") or "") for t in trades]
    df["exit_dt"] = [_fmt_et(t.get("exit_datetime")) or (t.get("exit_date") or "") for t in trades]
    # Entry session, shown at the finer premarket/afterhours grain so the column
    # agrees with the session filter (derived from entry_datetime; date-only
    # legacy rows → rth). The stored coarse 'extended' stamp is split here.
    from src.performance.tracker import _trade_session_fine
    df["session"] = [_trade_session_fine(t) for t in trades]
    df["held"] = [_held_disp(t) for t in trades]
    # IBKR order-status columns — simulated view only: the IBKR view contains
    # filled orders by construction, so the columns would be all-✓ noise there.
    if not trades[0].get("broker_view"):
        df["broker_entry"] = [_ibkr_leg_disp(t, "broker_") for t in trades]
        df["broker_exit"] = [_ibkr_leg_disp(t, "broker_exit_") for t in trades]
    spec = [t for t in _TRADE_COL_SPEC if t[0] in df.columns]
    df = df[[s[0] for s in spec]]
    records = df.to_dict("records")
    extra = {}
    if table_id:
        for rec in records:
            rec["id"] = rec.get("ticker")   # active_cell.row_id → ticker
        extra["id"] = table_id
    return dash_table.DataTable(
        data=records,
        columns=_columns(spec),
        tooltip_header=_header_tooltips(spec),
        filter_action="native",
        style_data_conditional=[
            {"if": {"filter_query": "{return_pct} > 0", "column_id": "return_pct"}, "color": figures.POS},
            {"if": {"filter_query": "{return_pct} < 0", "column_id": "return_pct"}, "color": figures.NEG},
        ],
        **extra,
        **_TABLE_KW,
    )


def _returns_tab():
    return html.Div([
        html.Div([
            _source_toggle("returns-source"),
            _window_toggle("returns-window"),
            _session_toggle("returns-session"),
            _direction_toggle("returns-direction"),
            _asset_toggle("returns-asset"),
        ], className="filter-bar"),
        dcc.Loading(html.Div(id="returns-body")),
    ])


@app.callback(Output("returns-body", "children"),
              Input("returns-window", "value"), Input("returns-session", "value"),
              Input("returns-direction", "value"), Input("returns-asset", "value"),
              Input("returns-source", "value"))
def _returns_body(window_value, session_value, direction_value, asset_value, source_value):
    if (source_value or "sim") == "broker":
        return _safe(lambda: _broker_returns_section(window_value, session_value, direction_value,
                                                     asset_value))
    return _safe(lambda: _returns_section(window_value, session_value, direction_value, asset_value))


@app.callback(Output("returns-review-plot", "children"),
              Input("returns-open-table", "active_cell"),
              Input("returns-closed-table", "active_cell"))
def _returns_review_plot(open_cell, closed_cell):
    """Click any open/closed trade row in the Returns tab → chart that ticker's
    confidence score over the days the position was held (the opener-pinned
    hold-review, left axis) against the stock price (right axis) — the same
    per-ticker plot as the Recommendations tab. ``ctx.triggered_id`` picks
    whichever of the two tables was clicked last; ``row_id`` is the ticker
    (set in ``_trades_table``). Shared by the Simulated and IBKR views (only one
    renders at a time, so the table ids never collide)."""
    from dash import ctx
    cell = closed_cell if ctx.triggered_id == "returns-closed-table" else open_cell
    if not cell:
        return html.Div(
            "↑ Click any open or closed trade row to chart that ticker's confidence "
            "score over the days it was held against the stock price.",
            style={"color": "#6b7280", "fontStyle": "italic", "padding": "8px 2px"})
    ticker = cell.get("row_id")
    if not ticker:
        return html.Div()
    return _safe(lambda: _review_timeline_section(ticker))


def _broker_returns_section(window_value, session_value=None, direction_value=None,
                            asset_value=None):
    """The IBKR view: what actually executed, at actual prices and commissions.
    Dollar P&L leads — real fills have real notionals, so percentages alone
    hide sizing. No modeled costs anywhere in this view."""
    from datetime import date, timedelta
    from src.performance.broker_view import summarize_broker_trades

    trades = data.broker_trades()
    wd = _window_days(window_value)
    if wd:
        cutoff = (date.today() - timedelta(days=wd)).isoformat()
        trades = [t for t in trades if str(t.get("entry_date") or "") >= cutoff]
    sess = _session_value(session_value)
    if sess:
        from src.performance.tracker import _session_matches
        trades = [t for t in trades if _session_matches(t, sess)]
    dirn = _direction_value(direction_value)
    if dirn:
        want = "BUY" if dirn == "long" else "SELL"
        trades = [t for t in trades if (t.get("action") or "").upper() == want]
    atype = _asset_value(asset_value)
    if atype:
        trades = [t for t in trades if (t.get("type") or "STOCK").upper() == atype.upper()]
    if not trades:
        return html.Div(
            "No IBKR fills recorded in this window yet — either broker_mode is "
            "off/dry_run, or no submitted order has filled.",
            style={"color": "#6b7280", "padding": 20})

    equity_usd = data.broker_account_equity_usd()
    s = summarize_broker_trades(trades, account_equity_usd=equity_usd)
    from src.performance.broker_view import avg_one_way_cost_pct_from_legs
    lmt_cost = avg_one_way_cost_pct_from_legs(data.filled_lmt_legs())
    # "Size ×" is a sim concept; in this view the dedicated Shares/Notional
    # columns carry the real sizing, so drop the multiplier from the tables.
    strip = lambda ts: [{k: v for k, v in t.items() if k != "position_size_multiplier"} for t in ts]
    open_trades = strip(sorted((t for t in trades if t["status"] == "OPEN"),
                               key=lambda t: str(t.get("entry_datetime") or ""), reverse=True))
    closed_trades = strip(sorted((t for t in trades if t["status"] == "CLOSED"),
                                 key=lambda t: str(t.get("exit_datetime") or ""), reverse=True))

    cards = html.Div(
        [
            _kpi("Realized P&L", _usd(s.get("realized_pnl_usd")),
                 figures.POS if (s.get("realized_pnl_usd") or 0) >= 0 else figures.NEG,
                 tooltip="Sum over closed fills: signed price move × shares filled, minus the commissions IBKR actually charged. No modeled costs."),
            _kpi("Open P&L", _usd(s.get("unrealized_pnl_usd")),
                 figures.POS if (s.get("unrealized_pnl_usd") or 0) >= 0 else figures.NEG,
                 tooltip="Mark-to-market of positions still held at the broker, vs their actual entry fills, minus entry commissions (exit cost unknown until it happens)."),
            _kpi("Return (wtd)", _pct(s.get("weighted_return"), signed=True),
                 figures.POS if (s.get("weighted_return") or 0) >= 0 else figures.NEG,
                 tooltip="Total % return on actual fills over CLOSED round-trips, each weighted by its REAL filled notional (so sizing counts — more precise than the equal-weighted 'Avg return'). Net of actual commissions."),
            *([_kpi("P&L vs equity", _pct(s.get("account_return_pct"), signed=True),
                    figures.POS if (s.get("account_return_pct") or 0) >= 0 else figures.NEG,
                    tooltip="Cumulative P&L of these trades (realized + open, this window) as a % of your LATEST IBKR account NAV (NetLiquidation), converted to USD — the account-relative impact. Uses the real account equity, not just per-trade fills; approximate (latest NAV vs windowed P&L).")]
              if s.get("account_return_pct") is not None else []),
            _kpi("Win rate", _pct(s.get("win_rate")),
                 tooltip="Share of broker positions with a positive net return on actual fills — CLOSED round-trips at their realized return, still-OPEN positions at their live mark (same open-inclusive convention as the Simulated view, so the two toggle sides are comparable)."),
            _kpi("Avg return", _pct(s.get("avg_return"), signed=True),
                 tooltip="Mean % return on actual fill prices net of actual commissions, EQUAL-weighted across every position (open at its live mark; compare with 'Return (wtd)', which is CLOSED-only and weights by real dollars)."),
            _kpi("Median return", _pct(s.get("median_return"), signed=True),
                 tooltip="Median % return on actual fills, net of actual commissions, across every position (open at its live mark) — the middle one, unaffected by a single outsized win/loss."),
            _kpi("Commissions", _usd(s.get("commissions_usd"), signed=False),
                 tooltip="Total commissions IBKR actually charged on these fills (exit legs counted once filled)."),
            _kpi("Avg 1-way cost (LMT)", _pct4(lmt_cost),
                 tooltip="Average ALL-IN cost per ONE-WAY fill across all real LMT fills, as a % of that leg's notional: real IBKR commission PLUS the execution cost (how far the fill landed from the decision price — captures the bid-ask crossing and any latency drift, positive = adverse). LMT ONLY — market (MKT) fills are excluded, since LMT is what the system uses going forward; drift-flatten cleanups are excluded too. Signed, so a favorable fill can lower it. This is the figure the simulated cost is calibrated to once enough LMT fills accumulate."),
            _kpi("Closed / Open", f"{s.get('closed', 0)} / {s.get('open', 0)}",
                 tooltip="Broker round-trips completed vs positions genuinely still held at the broker (a ledger-closed trade whose exit hasn't filled is still OPEN here)."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    ibkr_pnl = data.broker_account_pnl()
    pnl_block = []
    if ibkr_pnl:
        pnl_block = [
            _h3("IBKR account P&L (live — straight from IBKR)",
                "IBKR's own account P&L via reqPnL — ground truth including ALL fees, FX, and "
                "dividends, which the fill-derived numbers above can only approximate. "
                "ACCOUNT-LEVEL (all positions), NOT per-trade: 'Unrealized' is the current open "
                "P&L; 'Realized today' and 'Daily' are TODAY's figures (they reset each session). "
                "Converted to USD."),
            html.Div(
                [
                    _kpi("IBKR Unrealized", _usd(ibkr_pnl.get("unrealized")),
                         figures.POS if (ibkr_pnl.get("unrealized") or 0) >= 0 else figures.NEG,
                         tooltip="Current open-position P&L across the whole account, straight from IBKR (reqPnL.unrealizedPnL) — includes fees/FX/dividends."),
                    _kpi("IBKR Realized today", _usd(ibkr_pnl.get("realized")),
                         figures.POS if (ibkr_pnl.get("realized") or 0) >= 0 else figures.NEG,
                         tooltip="Today's realized P&L across the account, straight from IBKR (resets each session)."),
                    _kpi("IBKR Daily P&L", _usd(ibkr_pnl.get("daily")),
                         figures.POS if (ibkr_pnl.get("daily") or 0) >= 0 else figures.NEG,
                         tooltip="Today's total account P&L change, straight from IBKR (reqPnL.dailyPnL)."),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
            ),
        ]

    return html.Div([
        cards,
        *pnl_block,
        _h3("Open broker positions",
            "Shares genuinely held at IBKR right now (entry filled; exit not filled yet — even if the simulated ledger already closed the trade), marked at the latest price. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(open_trades, table_id="returns-open-table"),
        _h3("Closed broker round-trips",
            "Entry and exit both filled at IBKR. Returns are computed on the actual average fill prices, net of the commissions actually charged. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(closed_trades, table_id="returns-closed-table"),
        dcc.Loading(html.Div(id="returns-review-plot", style={"marginTop": 16})),
    ])


def _exit_quality_block(window_value, session_value, direction_value=None):
    """MFE/MAE exit-quality (item #5). Sim-ledger only — the excursion fields are
    maintained on the simulated trade, not the broker view."""
    rep = data.exit_quality(_window_days(window_value), _session_value(session_value),
                            _direction_value(direction_value))
    heading = _h3("Exit quality — MFE / MAE",
                  "Where the exit landed inside each trade's own MFE→MAE range. Exit "
                  "placement 1.0 = sold at the peak; 0.0 = cut at the worst point. Capture "
                  "= fraction of the favorable peak kept. Low placement ⇒ exits skew late "
                  "(cutting near the bottom); low capture with healthy MFE ⇒ winners ridden "
                  "back to flat (no profit-taking). Closed trades only; trades with a "
                  "degenerate (entered≈closed) excursion band are excluded.")
    if not rep.get("n"):
        return html.Div([heading, html.Div(rep.get("verdict", ""), style={"color": "#6b7280"})])
    cap = rep.get("avg_capture")
    cards = html.Div(
        [
            _kpi("Avg exit placement", _pct((rep["avg_placement"] or 0) * 100),
                 tooltip="Mean position in the MFE→MAE range (100% = exited at the peak, 0% = at the worst point)."),
            _kpi("Avg capture", _pct(cap * 100) if cap is not None else "–",
                 tooltip="Mean fraction of the favorable peak (MFE) kept at exit."),
            _kpi("Exited near MAE", _pct(rep["pct_exited_near_mae"]),
                 tooltip="Share of trades exited in the bottom 20% of their range — cutting near the worst point."),
            _kpi("Gave back >½ peak", _pct(rep.get("pct_gave_back_most_mfe")),
                 tooltip="Share of trades that kept less than half their favorable peak."),
            _kpi("Trades", str(rep["n"]),
                 tooltip="Analysable closed trades (non-degenerate MFE/MAE band)."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"ticker": r["ticker"], "ret": r["return_pct"], "mfe": r["mfe"], "mae": r["mae"],
             "place": r["exit_placement"], "capture": r["capture"], "giveback": r["give_back"],
             "reason": r["exit_reason"]} for r in rep["per_trade"]]
    cols = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Return %", "id": "ret", "type": "numeric", "format": _NUM2},
        {"name": "MFE %", "id": "mfe", "type": "numeric", "format": _NUM2},
        {"name": "MAE %", "id": "mae", "type": "numeric", "format": _NUM2},
        {"name": "Placement", "id": "place", "type": "numeric", "format": _NUM2},
        {"name": "Capture", "id": "capture", "type": "numeric", "format": _NUM2},
        {"name": "Give-back %", "id": "giveback", "type": "numeric", "format": _NUM2},
        {"name": "Exit reason", "id": "reason"},
    ]
    return html.Div([
        heading,
        html.Div(rep.get("verdict", ""), style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.mfe_capture_fig(rep)),
        dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
    ])


def _returns_section(window_value, session_value=None, direction_value=None, asset_value=None):
    perf = data.performance(window_days=_window_days(window_value), session=_session_value(session_value),
                            direction=_direction_value(direction_value),
                            asset_type=_asset_value(asset_value))
    stats = perf.get("stats") or {}
    pm = perf.get("portfolio_metrics") or {}
    wlabel = _window_label(window_value)

    compound = pm.get("compound_inception", stats.get("compound_return"))
    cards = html.Div(
        [
            _kpi(f"Compound ({wlabel})", _pct(compound, signed=True),
                 figures.POS if (compound or 0) >= 0 else figures.NEG,
                 tooltip="Path-faithful compound return over the selected window: each day's capital-weighted return across active positions, chained over real closing prices. Counts trades ENTERED within the window; open positions are included at their live mark."),
            _kpi("Win rate", _pct(stats.get("win_rate")),
                 tooltip="Share of trades with a positive spread-adjusted return. A flat round-trip is a loss (you pay the bid-ask spread). Open positions count at their live mark."),
            _kpi("Avg return", _pct(stats.get("avg_return"), signed=True),
                 tooltip="Mean per-trade % return, equal-weighted across all trades in the window (open trades at their live mark)."),
            _kpi("Median return", _pct(stats.get("median_return"), signed=True),
                 tooltip="Median per-trade % return in the window — the middle trade, unaffected by a single outsized win/loss (open trades at their live mark). Compare with 'Avg return': a median well below the average means a few big winners are lifting the mean."),
            _kpi("Weighted avg", _pct(stats.get("weighted_avg_return"), signed=True),
                 tooltip="Per-trade % return weighted by position size (the confidence-tier multiplier), so larger positions count more."),
            _kpi("Best", _pct(stats.get("best"), signed=True), figures.POS,
                 tooltip="Best single-trade % return in the window."),
            _kpi("Worst", _pct(stats.get("worst"), signed=True), figures.NEG,
                 tooltip="Worst single-trade % return in the window."),
            _kpi("Avg 1-way cost" + (" (real)" if perf.get("sim_cost_is_real") else ""),
                 _pct4(perf.get("sim_one_way_cost_pct")),
                 tooltip=("All-in one-way cost charged on each simulated leg, as a % of trade value. "
                          + ("CALIBRATED TO REAL IBKR FILLS: the measured average actual cost "
                             "(commission + execution vs decision price) is applied flat to every "
                             "entry and exit, so the simulated returns reflect what execution really costs."
                             if perf.get("sim_cost_is_real")
                             else "MODELED (half-spread + commission) — not enough real IBKR fills yet to "
                                  "calibrate (set by sim_real_fill_costs_min_legs); switches to real fills "
                                  "automatically once they accumulate.")
                          + " Compare with the IBKR view's 'Avg 1-way cost' (measured directly from fills).")),
            _kpi("Closed / Open", f"{stats.get('total_closed', 0)} / {stats.get('total_open', 0)}",
                 tooltip="Number of closed (realised) trades vs. positions currently open, within the selected window."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        dcc.Graph(figure=figures.equity_curve_fig(perf)),
        _h3("Open positions", "Positions currently held, marked to the latest price — the return shown is live mark-to-market ('what if you closed now'). Filtered to the selected entry window. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(perf.get("open_trades") or [], table_id="returns-open-table"),
        _h3("Closed trades", "Realised round-trips, with their final spread-adjusted return. Filtered to the selected entry window. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(perf.get("closed_trades") or [], table_id="returns-closed-table"),
        _exit_quality_block(window_value, session_value, direction_value),
        dcc.Loading(html.Div(id="returns-review-plot", style={"marginTop": 16})),
    ])


# ── Tab 4: Execution (price provenance · broker forensics · tracking error) ──

def _execution_tab():
    return html.Div([
        _safe(_provenance_section),
        _safe(_broker_forensics_section),
        _safe(_tracking_error_section),
    ])


def _provenance_section():
    """Price-provenance detail (item #8): trades in the latest run whose entry
    price diverged from the run snapshot beyond the session band."""
    pp = (data.latest_gate_diag() or {}).get("price_provenance")
    heading = _h3("Price provenance — entry vs snapshot",
                  "Standing guard against the stale-price class: each new trade's recorded "
                  "entry price is compared to the run's analysis snapshot for that ticker. A "
                  "divergence beyond the session band (RTH tight, off-hours wider) is flagged — "
                  "the automatic version of the one-off CRDO fill-vs-snapshot audit. Latest run.")
    if not pp:
        return html.Div([heading, html.Div(
            "No price-provenance record in the latest run (no trades opened, no snapshot, "
            "or the check is disabled).", style={"color": "#6b7280"})])
    flagged = pp.get("flagged") or []
    if not flagged:
        body = html.Div(
            f"✓ All {pp.get('n_checked', 0)} new trade(s) in the latest run entered within "
            "the snapshot band.", style={"color": figures.POS, "fontWeight": "bold"})
    else:
        rows = [{"ticker": f["ticker"], "entry": f["entry_price"], "snap": f["snapshot_price"],
                 "bps": f["bps"], "session": f["session"], "band": f["band"]} for f in flagged]
        cols = [
            {"name": "Ticker", "id": "ticker"},
            {"name": "Entry $", "id": "entry", "type": "numeric", "format": _NUM2},
            {"name": "Snapshot $", "id": "snap", "type": "numeric", "format": _NUM2},
            {"name": "Divergence bp", "id": "bps", "type": "numeric", "format": _NUM2},
            {"name": "Session", "id": "session"},
            {"name": "Band bp", "id": "band", "type": "numeric", "format": _NUM2},
        ]
        body = html.Div([
            html.Div("🔔 " + (pp.get("message") or ""),
                     style={"color": "#92400e", "marginBottom": 8}),
            dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
        ])
    return html.Div([heading, body])


def _broker_forensics_section():
    """Slippage / fill-rate / drift / reject forensics (item #3)."""
    rep = data.broker_forensics()
    heading = _h3("Broker execution forensics",
                  "Over all persisted broker orders: fill rate vs kill rate (the settle-or-kill "
                  "design), fill slippage by session (is the LMT cap achievable?), how often broker "
                  "positions drift from the ledger, and what the broker rejects. broker_mode must be "
                  "on for rows to accrue.")
    if not rep.get("n_orders"):
        return html.Div([heading, html.Div(
            "No broker orders recorded yet (broker_mode off / dry-run, or nothing submitted).",
            style={"color": "#6b7280"})])
    fo, d = rep["fill_outcomes"], rep["drift"]
    cards = html.Div(
        [
            _kpi("Fill rate", _pct(fo.get("fill_rate")),
                 tooltip="Filled orders ÷ terminal orders (still-working and no-op rows excluded)."),
            _kpi("Order events", str(rep["n_orders"]),
                 tooltip="Total persisted broker order / fill-repair events."),
            _kpi("Drift runs", f"{d.get('runs_with_drift', 0)}/{d.get('n_runs', 0)}",
                 tooltip="Reconcile runs where a broker position diverged from the ledger."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    outcome_rows = [{"outcome": k, "count": v}
                    for k, v in sorted(fo.get("counts", {}).items(), key=lambda kv: -kv[1])]
    outcomes_table = dash_table.DataTable(
        data=outcome_rows,
        columns=[{"name": "Fill outcome", "id": "outcome"},
                 {"name": "Count", "id": "count", "type": "numeric", "format": _INT}],
        **_TABLE_KW) if outcome_rows else html.Div()
    reject_df = rep["reject_reasons"]
    reject_rows = reject_df.to_dict("records") if reject_df is not None and not reject_df.empty else []
    reject_table = (dash_table.DataTable(
        data=reject_rows,
        columns=[{"name": "Reject reason", "id": "reason"},
                 {"name": "Count", "id": "n", "type": "numeric", "format": _INT}],
        **_TABLE_KW) if reject_rows
        else html.Div("No rejected / failed orders.", style={"color": "#6b7280", "marginTop": 8}))
    return html.Div([
        heading, cards,
        dcc.Graph(figure=figures.slippage_by_session_fig(rep["slippage_by_session"])),
        _h3("Fill outcomes", "Count of order events by terminal outcome (filled / killed / failed / working / skipped)."),
        outcomes_table,
        _h3("Reject reasons", "Failed / rejected orders grouped by the broker error message."),
        reject_table,
    ])


def _tracking_error_section():
    """Sim-vs-broker tracking error (item #4)."""
    rep = data.tracking_error()
    heading = _h3("Sim-vs-broker tracking error",
                  "The gap between the modeled ledger and actual IBKR fills, per matched trade. A "
                  "line hugging zero = the model tracks reality; a persistent one-sided drift = a "
                  "cost-model / pricing bug (the auto-catch for the stale-price class). Needs filled "
                  "broker orders.")
    if not rep.get("n_matched"):
        return html.Div([heading, html.Div(rep.get("verdict", ""), style={"color": "#6b7280"})])
    o = rep["overall"]
    cards = html.Div(
        [
            _kpi("Matched trades", str(rep["n_matched"]),
                 tooltip="Sim trades whose broker entry actually filled."),
            _kpi("Mean Δreturn", _pct(o["mean_d_return"], signed=True),
                 tooltip="Mean (sim − broker) return; + = the sim is optimistic vs real fills."),
            _kpi("Mean entry gap",
                 f"{o['mean_entry_bps']:+.0f} bp" if o.get("mean_entry_bps") is not None else "–",
                 tooltip="Mean signed entry-price gap (broker − sim) in basis points."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"ticker": r["ticker"], "date": r["entry_date"], "session": r["session"],
             "sim": r["sim_return"], "broker": r["broker_return"], "dret": r["d_return"],
             "bps": r["entry_bps"]} for r in rep["per_trade"]]
    cols = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Entry date", "id": "date"},
        {"name": "Session", "id": "session"},
        {"name": "Sim %", "id": "sim", "type": "numeric", "format": _NUM2},
        {"name": "Broker %", "id": "broker", "type": "numeric", "format": _NUM2},
        {"name": "Δreturn %", "id": "dret", "type": "numeric", "format": _NUM2},
        {"name": "Entry gap bp", "id": "bps", "type": "numeric", "format": _NUM2},
    ]
    return html.Div([
        heading,
        html.Div(rep.get("verdict", ""), style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.tracking_error_fig(rep)),
        dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
    ])


# ── Tab 5: Data Quality (source reliability · per-method coverage) ───────────

_DQ_LOOKBACK_DAYS = 14


def _data_quality_tab():
    return html.Div([
        html.Div(
            f"Source reliability + per-method coverage over the last {_DQ_LOOKBACK_DAYS} days — "
            "catches flaky/slow data sources and feeds that went dark BEFORE they quietly degrade "
            "signals (the failure mode behind every data warning so far). Built from the per-run "
            "run_sources + signals tables.",
            style={"color": "#6b7280", "marginBottom": 12}),
        _safe(_source_reliability_section),
        _safe(_method_coverage_section),
        _safe(_calibrations_section),
    ])


def _calibrations_section():
    """Live view of every SELF-CALIBRATED parameter: the exact value in force
    on the latest run vs its documented prior and the evidence count behind it
    — so a drifting or mis-learning parameter is as visible as a dark feed."""
    cals = (data.latest_gate_diag() or {}).get("calibrations") or []
    heading = _h3(
        "Calibrations — self-adapting parameters (latest run)",
        "Each row is a parameter the system LEARNS from its own data instead of a hardcoded "
        "constant: real-fill trading cost, per-session spread multipliers, the horizon cost "
        "hurdle, the breadth-sizing ramp, … 'Value' is what the latest run actually traded "
        "with; 'Prior' is the documented fallback it shrinks toward when evidence is thin; "
        "'Evidence n' is how many observations back the current value (0 = prior fully in "
        "force). A value drifting far from its prior on strong evidence is the system "
        "learning; on WEAK evidence it deserves a look. Snapshotted per run into gate_diag.")
    if not cals:
        return html.Div([heading, html.Div(
            "No calibration snapshot yet — appears after the next pipeline run on this code.",
            style={"color": "#6b7280"})])
    rows = [{
        "name": c.get("name"), "value": c.get("value"), "prior": c.get("prior"),
        "n": c.get("n_evidence"), "unit": c.get("unit"), "note": c.get("note"),
    } for c in cals]
    cols = [
        {"name": "Parameter", "id": "name"},
        {"name": "Value", "id": "value", "type": "numeric", "format": _NUM4},
        {"name": "Prior", "id": "prior", "type": "numeric", "format": _NUM4},
        {"name": "Evidence n", "id": "n", "type": "numeric", "format": _INT},
        {"name": "Unit", "id": "unit"},
        {"name": "Basis", "id": "note"},
    ]
    return html.Div([heading, dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW)])


def _source_reliability_section():
    rows = data.source_reliability(_DQ_LOOKBACK_DAYS)
    heading = _h3("Source reliability",
                  "Per data source over the window, with FOUR outcomes — ok (returned data), empty "
                  "(ran fine but returned nothing), dead (upstream gone, no free replacement), error "
                  "(raised). 'Empty' is first-class: a source that runs but returns nothing is no longer "
                  "silently counted as ok. An always-on feed that goes empty is flagged ⚠ for "
                  "investigation; event-driven feeds (8-K, earnings…) are expected to be empty sometimes; "
                  "dead feeds (greyed) are acknowledged, not actionable. Sorted worst-first. From run_sources.")
    if not rows:
        return html.Div([heading, html.Div("No run-source records in the window yet.",
                                            style={"color": "#6b7280"})])
    below = [r for r in rows if (r.get("success_rate") or 100.0) < 100.0]
    unexpected = [r for r in rows if r.get("unexpected_empty")]
    dead = [r for r in rows if r.get("known_dead")]
    slowest = max(rows, key=lambda r: r.get("median_s") or 0.0)
    cards = html.Div(
        [
            _kpi("Sources tracked", str(len(rows)),
                 tooltip="Distinct enabled data sources that ran at least once in the window."),
            _kpi("Errored", str(len(below)), figures.NEG if below else figures.POS,
                 tooltip="Sources that raised on at least one run — silent data loss candidates."),
            _kpi("Empty (always-on)", str(len(unexpected)), figures.NEG if unexpected else figures.POS,
                 tooltip="Always-on feeds whose LATEST run returned nothing though they should always "
                         "have data — investigate. Excludes event-driven and dead sources."),
            _kpi("Dead feeds", str(len(dead)), "#9ca3af",
                 tooltip="Sources whose upstream is gone with no free replacement (^TICK delisted; "
                         "congressional Stock Watcher 403). Acknowledged, not actionable — shown so the "
                         "deadness stays VISIBLE rather than masked as ok."),
            _kpi("Slowest (median)", f"{slowest['source']} · {slowest.get('median_s') or 0:.0f}s",
                 tooltip="The source with the highest median fetch time — the biggest tick-budget cost."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    def _status_disp(r):
        return "empty ⚠" if r.get("unexpected_empty") else r.get("last_status", "")

    def _note(r):
        if r.get("unexpected_empty"):
            return "⚠ always-on feed returned nothing — investigate"
        if r.get("known_dead"):
            return "known-dead — upstream gone, no free replacement"
        if r.get("last_error"):
            return r["last_error"]
        if r.get("expected_sparse"):
            return "event-driven (empty sometimes normal)"
        return ""

    table_rows = [{"source": r["source"], "runs": r["runs"], "success": r["success_rate"],
                   "empty_pct": r.get("empty_rate"), "status": _status_disp(r),
                   "median_s": r["median_s"], "p90_s": r["p90_s"], "note": _note(r)}
                  for r in rows]
    cols = [
        {"name": "Source", "id": "source"},
        {"name": "Runs", "id": "runs", "type": "numeric", "format": _INT},
        {"name": "Success %", "id": "success", "type": "numeric", "format": _NUM2},
        {"name": "Empty %", "id": "empty_pct", "type": "numeric", "format": _NUM2},
        {"name": "Last status", "id": "status"},
        {"name": "Median s", "id": "median_s", "type": "numeric", "format": _NUM2},
        {"name": "p90 s", "id": "p90_s", "type": "numeric", "format": _NUM2},
        {"name": "Note", "id": "note"},
    ]
    cond = [
        {"if": {"filter_query": "{success} < 100", "column_id": "success"},
         "color": figures.NEG, "fontWeight": "bold"},
        # An always-on feed that returned nothing this run — the actionable flag.
        {"if": {"filter_query": '{status} contains "⚠"'},
         "backgroundColor": "#3f1d1d", "color": "#fca5a5", "fontWeight": "bold"},
        # Dead feeds: greyed/italic — acknowledged, not an alarm.
        {"if": {"filter_query": '{status} = "dead"'},
         "color": "#9ca3af", "fontStyle": "italic"},
    ]
    return html.Div([
        heading, cards,
        dcc.Graph(figure=figures.source_latency_fig(rows)),
        dash_table.DataTable(data=table_rows, columns=cols, style_data_conditional=cond, **_TABLE_KW),
    ])


def _method_coverage_section():
    from src.performance.tracker import METHOD_LABELS
    cov = data.method_coverage(_DQ_LOOKBACK_DAYS)
    per = cov.get("per_method") or []
    heading = _h3("Per-method data coverage",
                  "For each signal method, the share of scored tickers with a REAL (non-zero) score — a "
                  "method reads 0.0 ('no view') when its data source failed for a ticker, so a feed going "
                  "dark shows as collapsing coverage before it shows as bad performance. Δ = recent minus "
                  "prior coverage; a large negative Δ is the alarm. From the signals panel.")
    if not per:
        return html.Div([heading, html.Div("No signal rows in the window yet.",
                                            style={"color": "#6b7280"})])
    drops = [r for r in per if r.get("delta") is not None and r["delta"] <= -20]
    cards = html.Div(
        [
            _kpi("Methods", str(len(per)),
                 tooltip="Signal methods tracked in the signals panel."),
            _kpi("Signal rows", f"{cov.get('n_rows', 0):,}",
                 tooltip="Total run×ticker rows in the window (the coverage denominator)."),
            _kpi("Coverage drops", str(len(drops)), figures.NEG if drops else figures.POS,
                 tooltip="Methods whose coverage fell ≥20pp recent-vs-prior — a feed that likely went dark."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"method": METHOD_LABELS.get(r["method"], r["method"]), "coverage": r["coverage_pct"],
             "scored": r["n_scored"], "total": r["n_total"], "recent": r["recent_pct"],
             "prior": r["prior_pct"], "delta": r["delta"]} for r in per]
    cols = [
        {"name": "Method", "id": "method"},
        {"name": "Coverage %", "id": "coverage", "type": "numeric", "format": _NUM2},
        {"name": "Scored", "id": "scored", "type": "numeric", "format": _INT},
        {"name": "Of", "id": "total", "type": "numeric", "format": _INT},
        {"name": "Recent %", "id": "recent", "type": "numeric", "format": _NUM2},
        {"name": "Prior %", "id": "prior", "type": "numeric", "format": _NUM2},
        {"name": "Δ (pp)", "id": "delta", "type": "numeric", "format": _NUM2},
    ]
    cond = [{"if": {"filter_query": "{delta} <= -20", "column_id": "delta"},
             "color": figures.NEG, "fontWeight": "bold"}]
    return html.Div([
        heading,
        html.Div("Low coverage is NORMAL for sparse methods (PEAD, extended-gap, options-derived put_call / "
                 "max_pain / OI-skew / IV — they only fire for a subset of tickers). The actionable signal "
                 "is a negative Δ: a method whose coverage dropped means its feed went dark.",
                 style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.method_coverage_fig(cov)),
        dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW),
    ])


# ── Lazy tab hydration ───────────────────────────────────────────────────────
# (value, tab label, renderer). serve_layout builds one EMPTY container per tab
# from this spec; the callbacks below fill each container the FIRST time its tab
# becomes active and never again (sticky — a revisit is instant). Defined after
# the renderers so the spec can reference them directly.
_TAB_SPEC = (
    ("rationale", "Recommendations & Rationale", _rationale_tab),
    ("methods", "Entry Performance", _methods_tab),
    ("exit_perf", "Exit Performance", _exit_perf_tab),
    ("returns", "Returns", _returns_tab),
    ("execution", "Execution", _execution_tab),
    ("data_quality", "Data Quality", _data_quality_tab),
)


def _fill_tab(active, existing, value, render):
    """The per-tab hydration rule (module-level so tests can pin it): render
    only when this tab is the active one AND its container is still empty;
    anything else is a no-op, which is what makes revisits instant and keeps
    nested component state alive across tab switches."""
    if active != value or existing:
        raise PreventUpdate
    return html.Div(_safe(render), className="tab-inner")


def _register_tab_callbacks() -> None:
    """One callback per tab: fill `tab-<value>` when that tab first becomes
    active. On page load Dash fires all six with the current tab value — five
    raise PreventUpdate immediately, one renders. Clicking a new tab renders it
    once; after that its `children` State is non-empty and it is left alone, so
    nested components (toggles, tables) keep their state across tab switches
    exactly as when everything was embedded up front."""
    for value, _label, render in _TAB_SPEC:
        @app.callback(Output(f"tab-{value}", "children"),
                      Input("tabs", "value"),
                      State(f"tab-{value}", "children"))
        def _fill(active, existing, _value=value, _render=render):
            return _fill_tab(active, existing, _value, _render)


_register_tab_callbacks()


def _serve_once(host: str, port: int) -> None:
    """Serve the WSGI app once.

    Prefers ``waitress`` — a production-grade, multi-threaded, cross-platform WSGI
    server (the right choice on Windows, where gunicorn does not run). It stays
    responsive for always-on use, recycles stuck connections, and won't fall over
    the way the Werkzeug development server does. Falls back to the Dash dev server
    only when waitress isn't installed.
    """
    try:
        from waitress import serve
    except ImportError:
        logger.warning(
            "waitress not installed — using the Dash dev server, which is less "
            "robust for always-on use. Install it with `pip install waitress`."
        )
        app.run(host=host, port=port, debug=False)
        return

    # A few worker threads so a slow performance() render can't block the whole UI;
    # channel_timeout reaps connections that go quiet instead of leaking them.
    #
    # clear_untrusted_proxy_headers=False is a SECURITY requirement here, not a
    # relaxation. Waitress defaults it to True, which DELETES X-Forwarded-For /
    # -Proto / -Host / Forwarded from the environ before the app ever sees them.
    # Since the ngrok tunnel forwards to 127.0.0.1, that would leave a public
    # visitor indistinguishable from a browser on this machine, and the localhost
    # bypass in _is_local_request would hand the open internet a free pass —
    # silently, with every log line and test still reading normal. We never trust
    # these headers' values; we only need to SEE that they were sent.
    serve(app.server, host=host, port=port, threads=8, channel_timeout=120,
          clear_untrusted_proxy_headers=False)


def _lan_ipv4() -> str | None:
    """Best-guess primary LAN IPv4 of this machine (the address a phone on the
    same Wi-Fi would use). No traffic is sent — a UDP socket 'connected' to a
    public IP just makes the OS pick the outbound interface. None on failure."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip if not ip.startswith("127.") else None
    except Exception:
        return None
    finally:
        s.close()


def _tailscale_ipv4() -> str | None:
    """This machine's Tailscale IP (100.64.0.0/10) if Tailscale is up — the
    address a phone reaches from ANYWHERE (cellular included) over the private
    tailnet. Tries the CLI on PATH, then the default Windows install path.
    None when Tailscale isn't installed/running."""
    import subprocess
    for exe in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        for line in (out.stdout or "").splitlines():
            ip = line.strip()
            if ip.startswith("100."):
                return ip
        return None
    return None


def run() -> None:
    """Run the dashboard, auto-restarting on an unexpected crash so it stays alive."""
    host, port = settings.dashboard_host, settings.dashboard_port
    logger.info(f"Dashboard starting at http://{host}:{port}  (Ctrl+C to stop)")
    if AUTH_ENABLED:
        logger.info(f"  🔒 Basic-Auth ON — user '{settings.dashboard_auth_username}' "
                    "(shared password from DASHBOARD_AUTH_PASSWORD)")
        nets = _bypass_networks()
        if nets:
            logger.info("  🏠 No password from " + ", ".join(str(n) for n in nets)
                        + " (tunnelled requests are gated even though they arrive "
                          "from loopback — proxy headers + Host give them away)")
            hosts = _bypass_hosts()
            if hosts:
                logger.info("     ...addressed as " + ", ".join(hosts)
                            + ", or by IP. Any OTHER hostname is asked for the password.")
        else:
            logger.info("  🔒 No bypass — every request needs the password, "
                        "including this machine (DASHBOARD_AUTH_BYPASS_NETWORKS is empty)")
    else:
        logger.info("  🔓 Basic-Auth OFF (no DASHBOARD_AUTH_PASSWORD) — anyone who can "
                    "reach this port sees positions and P&L. Required before exposing it publicly.")
    if host in ("0.0.0.0", "::"):
        # Bound to all interfaces → reachable from other devices: the LAN when
        # home, and the Tailscale tailnet from anywhere (incl. cellular).
        lan = _lan_ipv4()
        if lan:
            logger.info(f"  📱 Same Wi-Fi: http://{lan}:{port}")
        ts = _tailscale_ipv4()
        if ts:
            logger.info(f"  🌍 Away from home (Tailscale): http://{ts}:{port}")
        else:
            logger.info(
                "  🌍 For away-from-home access, install Tailscale (see CLAUDE.md → "
                "'Monitoring dashboard') — do NOT port-forward this to the internet."
            )
        logger.info(
            "  Bound to ALL interfaces — the (read-only) dashboard is reachable by "
            f"any device that can route to it. Windows Firewall must allow inbound TCP {port}."
        )

    # Pre-warm the heavy caches off the request path: without this, the first
    # page load after every pipeline run pays the full cold rebuild (~60s) while
    # the browser waits. The warmer refills the same caches a request would have,
    # so it can only ever make a load faster.
    try:
        data.start_cache_warmer()
    except Exception as e:                      # optimisation only — never fatal
        logger.warning(f"[dashboard] cache warmer not started: {e}")

    backoff = 2
    while True:
        try:
            _serve_once(host, port)
            return  # clean shutdown
        except KeyboardInterrupt:
            logger.info("Dashboard stopped.")
            return
        except Exception as e:  # pragma: no cover — last-resort supervisor
            logger.error(f"Dashboard server crashed: {e!r} — restarting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    run()
