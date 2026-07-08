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
