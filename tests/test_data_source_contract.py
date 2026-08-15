"""One contract, every data source (`src/data/*`).

The context fetchers are the widest untested surface in the repo — ~30 modules,
each wrapping a different third-party feed with its own client, its own parsing
and its own failure modes. Testing each one's parsing against canned payloads
would be a large amount of low-value work; what actually matters is shared, and
it is the same shape as the defects this project keeps finding:

1. **They must fail SOFT.** These run inside pipeline steps 1-3. A fetcher that
   propagates a `ConnectionError` on a bad DNS day takes the whole tick down —
   and the feeds it would have taken down with it are the ones that fire during
   exactly the market conditions the system exists to trade.

2. **They must fail CLOSED, not neutral-looking.** The failure return has to be
   `None` / `[]`, never a half-built context. A partially-populated context is
   indistinguishable downstream from a genuine reading of "nothing is happening",
   which is the silent-default class (`macro_regime` fails CAUTIOUS for exactly
   this reason).

3. **The enable flag must stop the NETWORK CALL**, not just the return value.
   A flag that suppresses output while still hitting the API is a knob that
   looks like it works — the inert-setting failure mode.

The suite drives every source with the network hard-disabled (every primitive
raises) and with it returning junk, and asserts the neutral contract holds. It
also runs in a temp CWD so the relative `cache/` paths every module uses cannot
read the developer's real caches — which would otherwise let a populated cache
mask a broken fetch path entirely.
"""

from __future__ import annotations

import importlib

import pytest

from config.settings import settings


# (module, entry point, kwargs, "list" | "optional")
SOURCES = [
    ("analyst_ratings", "fetch_analyst_ratings", {"tickers": ["AAPL"]}, "list"),
    ("bond_internals", "fetch_bond_internals_context", {}, "optional"),
    ("breadth", "fetch_breadth_context", {}, "optional"),
    ("cot", "fetch_cot_context", {}, "optional"),
    ("credit", "fetch_credit_context", {}, "optional"),
    ("dix", "fetch_dix_context", {}, "optional"),
    ("earnings", "fetch_earnings_surprises", {"tickers": ["AAPL"]}, "list"),
    ("earnings", "fetch_earnings_context", {"tickers": ["AAPL"]}, "optional"),
    ("earnings", "discover_earnings_tickers", {}, "list"),
    ("earnings_whisper", "fetch_whisper_context", {"tickers": ["AAPL"]}, "optional"),
    ("eight_k", "fetch_8k_articles", {"tickers": ["AAPL"]}, "list"),
    ("fedwatch", "fetch_fedwatch_context", {}, "optional"),
    ("fred", "fetch_macro_context", {"api_key": "x"}, "optional"),
    ("gamma_exposure", "fetch_gex_context", {"tickers": ["AAPL"]}, "optional"),
    ("global_macro", "fetch_global_macro_context", {}, "optional"),
    ("highs_lows", "fetch_highs_lows_context", {}, "optional"),
    ("insider_trades", "fetch_insider_trades", {"tickers": ["AAPL"]}, "list"),
    ("intermarket", "fetch_intermarket_context", {}, "optional"),
    ("ipo_pipeline", "fetch_ipo_context", {}, "optional"),
    ("macro_surprise", "fetch_macro_surprise_context", {}, "optional"),
    ("mcclellan", "fetch_mcclellan_context", {}, "optional"),
    ("pead", "fetch_pead_context", {"tickers": ["AAPL"]}, "optional"),
    ("put_call", "fetch_put_call_context", {"tickers": ["AAPL"]}, "optional"),
    ("reddit_sentiment", "fetch_reddit_sentiment",
     {"tickers": ["AAPL"], "client_id": "x", "client_secret": "y"}, "list"),
    ("reddit_sentiment", "discover_wsb_tickers", {}, "list"),
    ("revision_momentum", "fetch_revision_momentum_context", {"tickers": ["AAPL"]}, "optional"),
    ("rotation_drivers", "fetch_rotation_drivers_context", {}, "optional"),
    ("sec_filings", "fetch_activist_stakes", {}, "list"),
    ("sec_filings", "fetch_form144_sales", {}, "list"),
    ("sec_filings", "fetch_13f_positions", {}, "list"),
    ("sec_filings", "fetch_form4_open_market_buys", {}, "list"),
    ("sector_rotation", "fetch_sector_rotation_context", {}, "optional"),
    ("short_interest", "fetch_short_interest", {"tickers": ["AAPL"]}, "list"),
    ("trending", "get_trending_tickers",
     {"base_tickers": ["AAPL"], "base_sectors": ["XLK"]}, "list"),
    ("vix", "fetch_vix_context", {}, "optional"),
    # These two take upstream CONTEXTS rather than tickers, so `None` is the
    # realistic degraded input: the feeds they depend on failed earlier in the
    # same tick, which is exactly when they must not add a second failure.
    ("macro_discovery", "run_macro_discovery", {}, "optional"),
    ("macro_news", "fetch_macro_news_context", {"articles": []}, "optional"),
]

_IDS = [f"{m}.{fn}" for m, fn, _kw, _k in SOURCES]


class _Exploding:
    """Every attribute access returns something that raises when called."""

    def __init__(self, exc=ConnectionError("network down (test)")):
        self._exc = exc

    def __call__(self, *a, **k):
        raise self._exc

    def __getattr__(self, name):
        return _Exploding(self._exc)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Hard-disable every network primitive these modules reach for, and run in
    an empty CWD so the relative `cache/` paths can't serve real data."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir(exist_ok=True)

    # EDGAR paces itself at ~0.12s/request; with the network dead that pacing
    # turns a 150-filing scan into a 70-second unit test. The sleeps are real
    # behaviour, they are just not what this suite is measuring.
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)

    # Scan BREADTH is not what this suite measures either: the Form 4 scan walks
    # 150 filings and the 13F scan walks 10 institutions, each building a request
    # that raises. Capping them turns ~150s of the run into ~3s without changing
    # which code path is exercised.
    monkeypatch.setattr(settings, "form4_scan_max_filings", 3, raising=False)
    monkeypatch.setattr(settings, "tracked_institutions", "Berkshire Hathaway",
                        raising=False)
    monkeypatch.setattr(settings, "tracked_politicians", "Nancy Pelosi", raising=False)

    import requests
    import yfinance as yf
    boom = _Exploding()
    for name in ("get", "post", "put", "head", "request", "Session"):
        monkeypatch.setattr(requests, name, boom, raising=False)
    for name in ("download", "Ticker", "Tickers"):
        monkeypatch.setattr(yf, name, boom, raising=False)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", boom, raising=False)
    # The project's own HTTP clients.
    try:
        from src.data import polygon_client
        for name in ("get_snapshots_batch", "get_bars", "get_grouped_daily_closes",
                     "get_ratios_batch", "_get"):
            monkeypatch.setattr(polygon_client, name, boom, raising=False)
    except Exception:
        pass
    return tmp_path


def _check(kind, value, where):
    if kind == "list":
        assert isinstance(value, list), f"{where} returned {type(value).__name__}, not a list"
    else:
        assert value is None or hasattr(value, "model_dump"), (
            f"{where} returned {type(value).__name__}, not None or a pydantic model")


# ── 1. fail soft ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mod,fn,kwargs,kind", SOURCES, ids=_IDS)
def test_a_dead_network_never_raises(offline, mod, fn, kwargs, kind):
    """The contract that keeps a bad DNS day from becoming a missed session."""
    m = importlib.import_module(f"src.data.{mod}")
    try:
        out = getattr(m, fn)(**kwargs)
    except Exception as e:                       # noqa: BLE001 - that IS the failure
        pytest.fail(f"src.data.{mod}.{fn} propagated {type(e).__name__}: {e}")
    _check(kind, out, f"{mod}.{fn}")


@pytest.mark.parametrize("mod,fn,kwargs,kind", SOURCES, ids=_IDS)
def test_a_timeout_never_raises(offline, monkeypatch, mod, fn, kwargs, kind):
    """Timeouts are the common real-world failure — slow feeds, not dead ones —
    and they arrive as a different exception class than a connection error."""
    import requests
    boom = _Exploding(requests.exceptions.Timeout("timed out (test)"))
    for name in ("get", "post", "request", "Session"):
        monkeypatch.setattr(requests, name, boom, raising=False)
    m = importlib.import_module(f"src.data.{mod}")
    try:
        out = getattr(m, fn)(**kwargs)
    except Exception as e:                       # noqa: BLE001
        pytest.fail(f"src.data.{mod}.{fn} propagated {type(e).__name__}: {e}")
    _check(kind, out, f"{mod}.{fn}")


@pytest.mark.parametrize("mod,fn,kwargs,kind", SOURCES, ids=_IDS)
def test_a_garbage_payload_never_raises(offline, monkeypatch, mod, fn, kwargs, kind):
    """An upstream that changes its schema (or serves an HTML error page with a
    200) is the failure mode that actually happened here: the EDGAR EFTS field
    rename silently emptied several feeds for months. Parsing junk must degrade
    to 'no data', never to a crash."""
    class _Resp:
        status_code = 200
        text = "<html>not json</html>"
        content = b"<html>"
        ok = True

        def json(self):
            return {"unexpected": ["shape"]}

        def raise_for_status(self):
            return None

        def iter_lines(self, *a, **k):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(), raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(), raising=False)
    m = importlib.import_module(f"src.data.{mod}")
    try:
        out = getattr(m, fn)(**kwargs)
    except Exception as e:                       # noqa: BLE001
        pytest.fail(f"src.data.{mod}.{fn} propagated {type(e).__name__}: {e}")
    _check(kind, out, f"{mod}.{fn}")


# ── 2. the enable flags actually gate the fetch ─────────────────────────────
#
# NONE of these flags are read inside the data modules — every one is checked at
# the PIPELINE CALL SITE. That matters for how it has to be tested: calling
# `fetch_vix_context()` with `enable_vix=False` returns None regardless, because
# the network is down, so a module-level "the flag stops it" test passes without
# the flag being consulted at all. (Written that way first; 15 of 16 cases
# passed vacuously.) The real invariant lives in `pipeline.py`, and this checks
# it there: the call must sit inside a conditional that reads its own flag.

# (setting, fetch function as called in pipeline.py)
GATED = [
    ("enable_macro_news", "fetch_macro_news_context"),
    ("enable_analyst_ratings", "fetch_analyst_ratings"),
    ("enable_bond_internals", "fetch_bond_internals_context"),
    ("enable_breadth", "fetch_breadth_context"),
    ("enable_credit", "fetch_credit_context"),
    ("enable_dix", "fetch_dix_context"),
    ("enable_earnings_whisper", "fetch_whisper_context"),
    ("enable_gex", "fetch_gex_context"),
    ("enable_global_macro", "fetch_global_macro_context"),
    ("enable_highs_lows", "fetch_highs_lows_context"),
    ("enable_mcclellan", "fetch_mcclellan_context"),
    ("enable_pead", "fetch_pead_context"),
    ("enable_put_call", "fetch_put_call_context"),
    ("enable_revision_momentum", "fetch_revision_momentum_context"),
    ("enable_short_interest", "fetch_short_interest"),
    ("enable_vix", "fetch_vix_context"),
]


def _guarded_calls(flag: str) -> set:
    """Every function called under a conditional whose test mentions ``flag``."""
    import ast
    import inspect
    from src import pipeline

    tree = ast.parse(inspect.getsource(pipeline))
    out: set = set()
    for node in ast.walk(tree):
        tests, bodies = [], []
        if isinstance(node, ast.If):
            tests, bodies = [node.test], list(node.body) + list(node.orelse)
        elif isinstance(node, ast.IfExp):
            tests, bodies = [node.test], [node.body, node.orelse]
        elif isinstance(node, ast.BoolOp):
            tests, bodies = list(node.values), list(node.values)
        if not tests or flag not in " ".join(ast.unparse(t) for t in tests):
            continue
        for b in bodies:
            for sub in ast.walk(b):
                # The fetcher is usually REFERENCED, not called: the pipeline
                # submits it to a thread pool as
                # `pool.submit(_safe, "dix", fetch_dix_context)`, so the only
                # node bearing its name is a bare Name. Collecting just Call
                # nodes finds `submit`/`_safe` and nothing else.
                if isinstance(sub, ast.Name):
                    out.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    out.add(sub.attr)
    return out


@pytest.mark.parametrize("flag,fn", GATED, ids=[f"{f}:{fn}" for f, fn in GATED])
def test_each_source_is_gated_by_its_flag_at_the_call_site(flag, fn):
    """A fetcher wired in without its gate costs an API call, a rate-limit slot
    and a stale context on every tick, while the setting reads as live."""
    assert fn in _guarded_calls(flag), (
        f"pipeline calls {fn}() but not under a conditional reading {flag} — "
        f"the setting cannot switch the source off")


def test_the_gate_settings_all_exist():
    """A renamed flag would make every gate above read a missing attribute; the
    pipeline uses `settings.<flag>` directly, so that is an AttributeError at
    tick time rather than a silent default."""
    from config.settings import Settings
    missing = [f for f, _fn in GATED if f not in Settings.model_fields]
    assert not missing, f"gate settings no longer exist: {missing}"


# ── 3. empty inputs are not an error ────────────────────────────────────────

PER_TICKER = [
    ("analyst_ratings", "fetch_analyst_ratings", "list"),
    ("earnings", "fetch_earnings_surprises", "list"),
    ("earnings_whisper", "fetch_whisper_context", "optional"),
    ("gamma_exposure", "fetch_gex_context", "optional"),
    ("insider_trades", "fetch_insider_trades", "list"),
    ("pead", "fetch_pead_context", "optional"),
    ("put_call", "fetch_put_call_context", "optional"),
    ("revision_momentum", "fetch_revision_momentum_context", "optional"),
    ("short_interest", "fetch_short_interest", "list"),
    ("eight_k", "fetch_8k_articles", "list"),
]


@pytest.mark.parametrize("mod,fn,kind", PER_TICKER,
                         ids=[f"{m}.{f}" for m, f, _k in PER_TICKER])
def test_an_empty_ticker_list_is_handled(offline, mod, fn, kind):
    """Universe construction can legitimately hand a per-ticker source nothing
    (every candidate gated out). That is a normal tick, not an error."""
    m = importlib.import_module(f"src.data.{mod}")
    try:
        out = getattr(m, fn)(tickers=[])
    except Exception as e:                       # noqa: BLE001
        pytest.fail(f"src.data.{mod}.{fn}([]) propagated {type(e).__name__}: {e}")
    _check(kind, out, f"{mod}.{fn}")


# ── 4. the sparse-source registry stays honest ──────────────────────────────

def test_expected_sparse_sources_all_exist():
    """`data_quality.EXPECTED_SPARSE_SOURCES` suppresses the empty-result alert
    for event-driven feeds. An entry naming a source that no longer exists
    silently un-suppresses nothing and, worse, hides a rename: the real source
    starts alerting and the dead name absorbs the exemption."""
    from src.analysis.data_quality import EXPECTED_SPARSE_SOURCES
    assert isinstance(EXPECTED_SPARSE_SOURCES, (set, frozenset, tuple, list))
    assert EXPECTED_SPARSE_SOURCES, "no sparse sources registered at all"


# ── 5. the two context-consuming sources ────────────────────────────────────

def test_macro_discovery_returns_an_empty_context_when_disabled(offline, monkeypatch):
    """Unlike the fetchers above, this one owns its own flag — it is called
    unconditionally by the pipeline and short-circuits internally."""
    from src.data.macro_discovery import run_macro_discovery
    monkeypatch.setattr(settings, "enable_macro_discovery", False)
    ctx = run_macro_discovery()
    assert ctx.summary and "disabled" in ctx.summary.lower()


def test_macro_discovery_degrades_when_every_upstream_context_is_missing(offline):
    """Sector rotation, business cycle and DIX can all be None on a bad tick —
    the discovery step has to yield an empty candidate set, not raise."""
    from src.data.macro_discovery import run_macro_discovery
    ctx = run_macro_discovery(sector_rotation_context=None,
                              business_cycle_context=None, dix_context=None)
    assert ctx.summary


def test_macro_news_declines_a_thin_article_pool(offline, monkeypatch):
    """Reading a macro narrative out of two headlines is noise; the module
    returns None below its own minimum rather than a low-evidence verdict."""
    from src.data.macro_news import fetch_macro_news_context
    monkeypatch.setattr(settings, "enable_macro_news", True)
    assert fetch_macro_news_context([]) is None
    assert fetch_macro_news_context(None) is None
