"""OHLCV cache writes silently going stale on Windows (2026-08-04).

`save_ohlcv` writes to a temp file then `os.replace`s it into place. On Windows
that raises [WinError 5] Access is denied whenever ANY process has the
destination open — there is no atomic overwrite-through-open-handle as on POSIX.
The old code logged a warning and carried on, so the cache stayed STALE with no
further signal.

Measured: 22 failures in one day, ALL of them SPY / XLK / XLV / XLY — the
market-relative benchmark and the sector ETFs. Those are the most-read files, so
the collision rate is highest exactly where staleness does the most damage.
"""

from __future__ import annotations

import os
import pandas as pd
import pytest

from src.data import cache


def _df(v: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [v]}, index=pd.to_datetime(["2026-08-04"]))


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_ohlcv_dir", lambda interval="1d": tmp_path)
    monkeypatch.setattr(cache, "_ohlcv_path",
                        lambda t, interval="1d": tmp_path / f"{t}.json")


def test_replace_retries_until_the_reader_lets_go(monkeypatch):
    """The blocking reader is always transient (a dashboard accessor or another
    scorer thread), so a short backoff must clear it — not surrender."""
    real_replace, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:                      # first 2 attempts: reader holds it
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", flaky)
    monkeypatch.setattr(cache, "_REPLACE_BASE_DELAY", 0.001)
    cache.save_ohlcv("SPY", _df(1.0))
    assert calls["n"] == 3                      # retried, then succeeded
    assert (cache._ohlcv_path("SPY")).exists()  # and the write actually landed


def test_persistent_blocker_gives_up_cleanly_without_leaking_temp_files(monkeypatch):
    def always_denied(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(cache.os, "replace", always_denied)
    monkeypatch.setattr(cache, "_REPLACE_BASE_DELAY", 0.001)
    cache.save_ohlcv("SPY", _df(1.0))           # must not raise
    leftovers = [p for p in cache._ohlcv_dir().iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"temp files leaked: {leftovers}"


def test_a_clean_write_does_not_retry(monkeypatch):
    calls = {"n": 0}
    real = os.replace

    def counted(src, dst):
        calls["n"] += 1
        return real(src, dst)

    monkeypatch.setattr(cache.os, "replace", counted)
    cache.save_ohlcv("SPY", _df(2.0))
    assert calls["n"] == 1                      # no cost on the happy path
