"""Shared pytest configuration for the llm_trader test suite.

Adds the project root to ``sys.path`` so test files can ``from src.performance...
import ...`` without installing the package.  Run via:

    python -m pytest tests/

from the project root (the recommended invocation — it auto-discovers and
respects the path injection below).
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point settings.db_path at a throwaway file for EVERY test.

    Post-mortem (2026-06-10): two legacy tests called tracker._save_trades —
    which is a FULL-REPLACE of the production DuckDB trades table — and wiped
    the real trade ledger at data/llm_trader.db (the tests predated the
    JSON→DuckDB cutover and monkeypatched only the obsolete TRADES_FILE path).
    This fixture makes that whole accident class impossible: no test can reach
    the production database, no matter what it calls.
    """
    from config.settings import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test_llm_trader.db"))


@pytest.fixture(autouse=True)
def _spread_only_costs(monkeypatch):
    """Pin the sim commission model to 'none' for the whole suite.

    The long-standing spread/NAV/tracker hand-math tests assert spread-only
    numbers (and must not depend on the developer's .env). Commission-specific
    tests opt back in by setting ``settings.commission_model`` explicitly.
    """
    from config.settings import settings

    monkeypatch.setattr(settings, "commission_model", "none")


@pytest.fixture(autouse=True)
def _no_gateway_auto_restart(monkeypatch):
    """Disable the IB Gateway auto-restart for the whole suite.

    gateway_recovery.maybe_restart_gateway runs REAL ``taskkill`` / ``schtasks``
    commands — a reconcile/wedge test driving a connect failure must never kill
    the developer's actual gateway. The recovery tests opt back in explicitly
    (and monkeypatch subprocess)."""
    from config.settings import settings

    monkeypatch.setattr(settings, "broker_gateway_auto_restart", False)


@pytest.fixture(autouse=True)
def _isolated_wedge_history(tmp_path, monkeypatch):
    """Point the broker's persisted repeated-wedge history at a throwaway file.

    IBKRBroker writes cache/broker_wedge_recycles.json on every force-recycle so
    the count survives a watchdog relaunch (2026-09-03) — any wedge test would
    otherwise leave stamps in the PRODUCTION file, and the live scheduler would
    inherit them as evidence at its next start."""
    from src.broker import ibkr as _ibkr

    monkeypatch.setattr(_ibkr, "WEDGE_HISTORY_PATH", tmp_path / "broker_wedge_recycles.json")


@pytest.fixture(autouse=True)
def _no_shadow_sentiment(monkeypatch):
    """The shadow sentiment pass is OFF for every test unless one asks for it.

    In production it fires a background LLM call for every scored ticker. In a
    test that means a real thread calling a monkeypatched client AFTER the
    assertion ran — which is how it first showed up: a fallback test counted two
    DeepSeek calls where it expected one, the second being the shadow racing the
    assertion. `tests/test_sentiment_shadow.py` re-enables it explicitly and
    runs the pass inline."""
    from config.settings import settings

    # Passing-mention abstention OFF for every other test: fixtures use
    # placeholder headlines that never name the ticker, so the filter fires
    # on all of them (15 unrelated tests). The tests that DO exercise it turn
    # it back on explicitly.
    monkeypatch.setattr(settings, "enable_passing_mention_abstention", False,
                        raising=False)
    monkeypatch.setattr(settings, "enable_sentiment_shadow", False)
    # HOSTED ENGINES ON for the suite (2026-09-18). Production runs LOCAL-ONLY
    # (`ENABLE_HOSTED_SENTIMENT_ENGINES=false` in .env) because both hosted
    # accounts are unfunded, but the multi-engine machinery — try-order, pins,
    # per-call fallback, provider attribution — is KEPT and revivable the moment
    # one is re-funded, so the suite keeps exercising it. The revivable-dead-
    # branch convention, same as the decommissioned prompt arms. The tests that
    # assert LOCAL-ONLY behaviour turn it back off explicitly.
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", True,
                        raising=False)


@pytest.fixture(autouse=True)
def _no_shadow_synthesis(monkeypatch):
    """The engine-shadow synthesis branch is OFF for every test unless one asks
    for it — the same reason as the sentiment shadow above: in production it
    starts a daemon thread that calls `generate_recommendations` AGAIN with a
    pinned engine after the live pass returned, so any test that stubs the
    engines and counts calls would see a second, racing call.
    `tests/test_engine_shadow.py` re-enables it explicitly and joins the
    thread."""
    from config.settings import settings

    monkeypatch.setattr(settings, "enable_synthesis_shadow", False)


@pytest.fixture(autouse=True)
def _no_catalyst_repair(monkeypatch):
    """The catalyst-repair pass is OFF for every test unless one asks for it.
    In production every PRIMARY sentiment verdict may hand its digest to a
    background thread that calls the LOCAL LLM up to three more times, so a
    test that counts engine calls — or one that runs with no local server —
    would otherwise see a racing, failing specialist. ``tests/test_catalyst_
    repair.py`` re-enables it explicitly with a stubbed specialist and drains
    the pool before asserting."""
    from config.settings import settings

    monkeypatch.setattr(settings, "enable_catalyst_repair", False)


@pytest.fixture(autouse=True)
def _no_polygon_network(monkeypatch):
    """Keep the LIVE Polygon API out of the unit suite.

    With a real POLYGON_API_KEY in the environment, two paths reach the network
    from ordinary tests — found twice on 2026-08-31:

    * ``reconcile._quote_for``'s NBBO fallback pulled a real book for the
      ticker "TEST" (an actual listing), so the broker-reliability tests'
      mid-based-cap assertions drifted with the market;
    * ``tracker._fetch_price`` began leading with Polygon, so the IBKR
      fallback tests got a real quote instead of their stub.

    Both are neutralised here rather than at each call site, so a NEW test that
    happens to touch a price path is protected by default. Tests of these
    layers opt back in explicitly and monkeypatch the client function."""
    from config.settings import settings

    monkeypatch.setattr(settings, "enable_polygon_quotes", False)
    monkeypatch.setattr("src.data.polygon_client.get_last_price",
                        lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _absolute_score_basis(monkeypatch):
    """Pin ``method_score_basis="absolute"`` for the whole suite.

    The long-standing build_signals integration tests assert the combine's
    ARITHMETIC (camp averages, inversion swaps, filter exclusion) on known
    absolute inputs over 1-3-ticker fixtures. Under the live "rank" default a
    cross-section that thin makes every method ABSTAIN (score 0 — the
    2026-08-13 no-fallback rule), which is correct in production and useless
    as a test fixture. The two-phase continuation path runs either way, so the
    refactor stays covered; the rank transform itself is pinned by
    tests/test_method_rank_basis.py, which opts back in explicitly."""
    from config.settings import settings

    monkeypatch.setattr(settings, "method_score_basis", "absolute")


@pytest.fixture(autouse=True)
def _default_llm_primary(monkeypatch):
    """Pin the LLM routing/thinking settings to their code defaults for the whole
    suite so engine-routing / hold-review-pinning / thinking tests don't inherit a
    .env override (the developer's .env sets ``llm_primary_provider='qwen'`` and
    ``llm_max_thinking=true``, which coerce pins to qwen and flip bulk sentiment/
    macro-news to thinking). Tests that exercise those opt back in explicitly (their
    per-test monkeypatch runs after this one)."""
    from config.settings import settings

    monkeypatch.setattr(settings, "llm_primary_provider", "deepseek")
    # The hold-review suites exercise the LLM exit path, which the 2026-09-04
    # directive switched OFF in production (.env: zero synthesis calls in the
    # trading path). The machinery is kept and must stay tested — one flag
    # restores it — so the suite pins it ON and production stays off.
    monkeypatch.setattr(settings, "enable_llm_hold_review", True)
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    # Same shape for the two exits the 2026-09-05 directive downgraded to
    # SHADOW-ONLY and for the ml_exit arm coupling it removed: production
    # ships the new behaviour, the suites keep exercising the old machinery
    # (which is kept, not deleted), and the tests that assert the NEW
    # behaviour set these flags themselves.
    monkeypatch.setattr(settings, "signal_decay_exits_shadow_only", False)
    monkeypatch.setattr(settings, "ml_exit_all_positions", False)
    # The shadow-sentiment suites assert on WHICH tickers get a pair, so they
    # must not inherit production's sampling (.env: engine pinned to deepseek,
    # 50% of calls). Pin the full-coverage defaults; the sampling tests set
    # the share themselves.
    monkeypatch.setattr(settings, "sentiment_shadow_share", 1.0)
    monkeypatch.setattr(settings, "sentiment_shadow_engine", "auto")
    monkeypatch.setattr(settings, "llm_max_thinking", False)
    # DeepSeek-only sentiment for the legacy suites (the .env sets a 10% Qwen share);
    # the sentiment-routing tests opt into a Qwen share explicitly.
    monkeypatch.setattr(settings, "sentiment_qwen_share", 0.0)
    # Same for the self-hosted engine: the .env runs a live A/B at
    # SENTIMENT_LOCAL_SHARE=0.30 (2026-09-03), and `reset_sentiment_providers`
    # samples the shares SEQUENTIALLY, so without this pin any test that expects
    # its DeepSeek stub to be called fails on ~30% of runs (it did — the
    # news-event cache round-trip went red in a full-suite run after passing
    # alone). Code defaults here; test_local_llm / test_llm_fallback_chains /
    # test_sentiment_routing opt in explicitly.
    monkeypatch.setattr(settings, "enable_local_llm", False)
    monkeypatch.setattr(settings, "sentiment_local_share", 0.0)
    # Deterministic prompts/exits for the legacy suites: no random blind-arm
    # synthesis, single-review LLM exits. The A/B + confirmation tests opt in.
    monkeypatch.setattr(settings, "blind_synthesis_share", 0.0)
    monkeypatch.setattr(settings, "enable_llm_exit_confirmation", False)
    # Long-horizon buy arm OFF for the legacy suites: the .env now A/B-tests it at
    # 0.5, which would randomly swap combined_buy_score for the ml_buy stacker and
    # make aggregator/pipeline tests non-deterministic. The arm tests opt in.
    monkeypatch.setattr(settings, "ml_combine_arm_share", 0.0)
    monkeypatch.setattr(settings, "enable_ml_combine", False)
    # ML exit model OFF for the legacy suites: it only closes arm trades (pinned
    # off above) and fail-softs without an artifact, but pinning it keeps
    # build_exit_scores from trying to load a model during unrelated exit tests.
    monkeypatch.setattr(settings, "enable_ml_exit_model", False)
    # Pin the Qwen route to DashScope-direct defaults — the developer's .env
    # points at OpenRouter (different model id + thinking dialect + explicit
    # cache markers). OpenRouter-route tests monkeypatch these explicitly.
    monkeypatch.setattr(settings, "qwen_base_url",
                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(settings, "qwen_model", "qwen3.7-max")


@pytest.fixture(autouse=True)
def _offline_company_names(tmp_path, monkeypatch):
    """Keep ``src.data.company_names`` off the network and off the real cache
    for EVERY test. Its name resolver reaches SEC (bulk registrant list) and
    Polygon (per-symbol reference) lazily from the news relevance filter and the
    feed tag confirmation, so any test that builds a digest would otherwise
    download ~11k names into ``cache/company_names.json``. Here the state starts
    EMPTY (no symbol has a name → only the symbol/alias evidence tiers fire);
    a test that needs names seeds them with ``company_names._seed_for_tests``."""
    from src.data import company_names as cn

    cn._reset_for_tests()
    monkeypatch.setattr(cn, "_CACHE_PATH", tmp_path / "company_names.json")
    monkeypatch.setattr(cn, "_refresh_sec",
                        lambda st: st.__setitem__("sec_loaded_at", cn._iso(cn._now())))
    monkeypatch.setattr(cn, "_polygon_name", lambda ticker: None)
    monkeypatch.setattr(cn, "_polygon_industry", lambda ticker: None)
    yield
    cn._reset_for_tests()


@pytest.fixture(autouse=True)
def _isolated_sentiment_cache(tmp_path, monkeypatch):
    """Point the sentiment LLM cache at a throwaway file + drop the in-memory
    copy for EVERY test. The cache is a process-global keyed by (ticker, engine,
    article set) persisted under cache/ — without this, a test's canned LLM
    response could leak into another test (or into the production cache file)."""
    from src.analysis import sentiment

    monkeypatch.setattr(sentiment, "_sent_cache_path",
                        lambda: tmp_path / "sentiment_llm.json")
    sentiment._reset_sentiment_cache_for_tests()
    yield
    sentiment._reset_sentiment_cache_for_tests()


@pytest.fixture(autouse=True)
def _reset_winrate_filter_cache():
    """Drop the aggregator's process-global win-rate filter cache before AND after
    every test. winrate_filtered_methods() caches its computed drop-set for
    ic_weight_cache_seconds (1800s) so build_signals' hold-review pool doesn't
    recompute it each call; without this reset a test that mocks
    compute_solo_method_performance to a non-empty ledger could leak its drop-set
    into a later test (the autouse _isolated_db points every test at an EMPTY DB, so
    the correct value is frozenset() unless a test explicitly mocks otherwise)."""
    from src.signals.aggregator import reset_winrate_filter_cache

    reset_winrate_filter_cache()
    yield
    reset_winrate_filter_cache()


@pytest.fixture(autouse=True)
def _no_real_cost_override():
    """Reset the process-global real-fill cost override (flat + per-session)
    and the calibration registry before AND after every test. They're
    module-globals installed by calibrate_sim_costs / the calibrated
    computations; without this reset one test's calibration could leak into
    another's hand-computed spread/NAV assertions."""
    from src.performance import spread
    from src.performance.calibration import reset_calibrations
    from src.analysis.exit_floor_calibration import reset_cache as _reset_exit_floor
    from src.analysis.threshold_calibration import reset_cache as _reset_threshold
    from src.performance.edge_sizing import reset_cache as _reset_edge

    def _reset_all():
        spread.set_real_cost_override(None)
        spread.set_cost_attribution(None, None, None)   # per-trade cost lookups
        # Parsed-OHLCV memo: keyed on (path, mtime, size), so a test that
        # rewrites the same fixture file within one mtime tick at an identical
        # size could otherwise read the previous test's frame.
        try:
            from src.data.cache import clear_ohlcv_parse_cache
            clear_ohlcv_parse_cache()
        except Exception:
            pass
        # Memoised signal panels — keyed on the latest run_id, which a test that
        # writes its own DB would otherwise share with the previous test.
        try:
            from src.analysis.signal_panel import reset_panel_cache
            reset_panel_cache()
        except Exception:
            pass
        # Per-direction horizon ramp — measured from the ledger, so a test that
        # writes its own trades must not inherit the previous one's multiplier.
        try:
            from src.performance.tracker import reset_horizon_ramp_cache
            reset_horizon_ramp_cache()
        except Exception:
            pass
        reset_calibrations()
        _reset_exit_floor()
        _reset_threshold()
        _reset_edge()

    _reset_all()
    yield
    _reset_all()
