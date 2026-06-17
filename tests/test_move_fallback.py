"""Regression test for the MOVE stale-cache fallback (#5).

^MOVE is intermittently unavailable on yfinance; on a bad day the live fetch
failed on EVERY run (the cache is only written on success), so the Macro Regime
Filter lost its bond-vol read ~11% of runs. fetch_move_context now carries
forward the most recent cached level instead of returning None.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

import src.data.move as move
from src.models import MOVEContext
from config import settings


def _ctx(report_date, level, signal="NORMAL"):
    return MOVEContext(move=level, signal=signal, direction="NEUTRAL", source="^MOVE",
                       report_date=report_date, summary=f"MOVE Index at {level} ({signal}).")


@pytest.fixture
def failing_live(tmp_path, monkeypatch):
    """Isolated cache dir + both live ^MOVE paths failing."""
    monkeypatch.setattr(move, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(move.yf, "download", lambda *a, **k: pd.DataFrame())   # empty history
    monkeypatch.setattr(move, "_fetch_current_price", lambda: None)            # fast_info None
    return tmp_path


def test_move_carries_forward_recent_cache(failing_live):
    today = date(2026, 6, 16)
    move._save_cache(_ctx(today - timedelta(days=1), 118.0, "HIGH"))   # yesterday's good read
    ctx = move.fetch_move_context(today)
    assert ctx is not None
    assert ctx.move == 118.0 and ctx.signal == "HIGH"                  # carried forward verbatim
    assert "stale" in ctx.summary.lower()                             # flagged stale
    assert "carried forward" in ctx.source.lower()
    # today's cache must NOT be written, so the next run still retries a fresh fetch
    assert not (failing_live / f"move_{today.isoformat()}.json").exists()


def test_move_carries_forward_within_window_only(failing_live):
    today = date(2026, 6, 16)
    move._save_cache(_ctx(today - timedelta(days=30), 118.0, "HIGH"))  # 30d old → outside 7d window
    assert move.fetch_move_context(today) is None


def test_move_no_cache_returns_none(failing_live):
    assert move.fetch_move_context(date(2026, 6, 16)) is None          # live fails + no prior cache


def test_move_prefers_today_cache(failing_live):
    today = date(2026, 6, 16)
    move._save_cache(_ctx(today, 88.0, "NORMAL"))                      # today already cached
    ctx = move.fetch_move_context(today)
    assert ctx is not None and ctx.move == 88.0
    assert "stale" not in ctx.summary.lower()                         # fresh, not carried forward
