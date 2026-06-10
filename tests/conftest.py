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
