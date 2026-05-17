"""Shared pytest configuration for the llm_trader test suite.

Adds the project root to ``sys.path`` so test files can ``from src.performance...
import ...`` without installing the package.  Run via:

    python -m pytest tests/

from the project root (the recommended invocation — it auto-discovers and
respects the path injection below).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
