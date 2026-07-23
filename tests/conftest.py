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
def _default_llm_primary(monkeypatch):
    """Pin the LLM routing/thinking settings to their code defaults for the whole
    suite so engine-routing / hold-review-pinning / thinking tests don't inherit a
    .env override (the developer's .env sets ``llm_primary_provider='qwen'`` and
    ``llm_max_thinking=true``, which coerce pins to qwen and flip bulk sentiment/
    macro-news to thinking). Tests that exercise those opt back in explicitly (their
    per-test monkeypatch runs after this one)."""
    from config.settings import settings

    monkeypatch.setattr(settings, "llm_primary_provider", "deepseek")
    monkeypatch.setattr(settings, "llm_max_thinking", False)
    # DeepSeek-only sentiment for the legacy suites (the .env sets a 10% Qwen share);
    # the sentiment-routing tests opt into a Qwen share explicitly.
    monkeypatch.setattr(settings, "sentiment_qwen_share", 0.0)
    # Deterministic prompts/exits for the legacy suites: no random blind-arm
    # synthesis, single-review LLM exits. The A/B + confirmation tests opt in.
    monkeypatch.setattr(settings, "blind_synthesis_share", 0.0)
    monkeypatch.setattr(settings, "enable_llm_exit_confirmation", False)
    # Pin the Qwen route to DashScope-direct defaults — the developer's .env
    # points at OpenRouter (different model id + thinking dialect + explicit
    # cache markers). OpenRouter-route tests monkeypatch these explicitly.
    monkeypatch.setattr(settings, "qwen_base_url",
                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(settings, "qwen_model", "qwen3.7-max")


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
        reset_calibrations()
        _reset_exit_floor()
        _reset_threshold()
        _reset_edge()

    _reset_all()
    yield
    _reset_all()
