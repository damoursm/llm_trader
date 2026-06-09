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
def _spread_only_costs(monkeypatch):
    """Pin the sim commission model to 'none' for the whole suite.

    The long-standing spread/NAV/tracker hand-math tests assert spread-only
    numbers (and must not depend on the developer's .env). Commission-specific
    tests opt back in by setting ``settings.commission_model`` explicitly.
    """
    from config.settings import settings

    monkeypatch.setattr(settings, "commission_model", "none")
