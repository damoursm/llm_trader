"""Dead / delisted names must not be retried on every tick (2026-09-29).

The opportunity screener's pool is the curated list plus every cached OHLCV file,
read cache-first with no staleness check: acquired names whose cache froze at the
takeover price (CRNX, APGE, FBRX, ATAI — 12-22 sessions behind SPY) read as fresh
52-week highs, were injected into the universe on every tick, and failed every
snapshot ("No data for CRNX — skipping"); thin caches of dead names (EDAP, BXDIF)
spent the warm-up budget on an empty fetch every tick. And the discovery
liquidity gate re-fetched a name with no bars at all (FGRS) on every tick.

What must hold: a cache more than ``screen_max_stale_sessions`` BENCHMARK
sessions behind is never screened and (outside the curated list) never fetched;
staleness is counted against SPY's own bars, so an outage that freezes every
cache judges nothing stale; a curated name is refreshed once and screened only if
current. The gate remembers a name whose fetch came back empty for the rest of
the day — only when the same run's other fetches returned data. No network.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.data import liquidity as liq
from src.data import screener as sc

D = pd.Timestamp("2026-09-28")
DATES = pd.bdate_range(end=D, periods=320)


def _frame(dates, jump=True, price=50.0, vol=5e6):
    """A liquid frame whose LAST bar is a new 52-week high on 5x volume (a setup)."""
    n = len(dates)
    close = np.linspace(price * 0.8, price, n)
    volume = np.full(n, vol)
    if jump:
        close[-1] = price * 1.2
        volume[-1] = vol * 5
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": volume}, index=dates)


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setattr(settings, "enable_opportunity_screener", True)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "screen_max_fetch_per_run", 30)
    frames = {"SPY": _frame(DATES, jump=False, price=700.0)}
    fetches = []
    fetched_frames = {}
    monkeypatch.setattr(sc, "load_ohlcv", lambda t: frames.get(t))
    monkeypatch.setattr(sc, "get_history", lambda t, period=None: fetches.append(t) or fetched_frames.get(t))

    def run(pool):
        monkeypatch.setattr(sc, "_candidate_pool", lambda: list(pool))
        return [h.ticker for h in sc.run_screener().hits]
    return frames, fetched_frames, fetches, run


def test_a_stale_cache_is_never_screened(screen):
    frames, _, fetches, run = screen
    frames["LIVE"] = _frame(DATES)                 # current
    frames["DEAD"] = _frame(DATES[:-15])           # froze 15 sessions ago (CRNX: 19)
    frames["LAG1"] = _frame(DATES[:-1])            # one session behind: still screened
    hits = run(["DEAD", "LIVE", "LAG1"])
    assert "LIVE" in hits and "LAG1" in hits
    assert "DEAD" not in hits and "DEAD" not in fetches


def test_a_thin_dead_cache_is_not_refetched_every_tick(screen):
    frames, _, fetches, run = screen
    frames["EDAPX"] = _frame(DATES[-95:-83])       # 12 bars, 83 sessions old (EDAP)
    frames["THIN"] = _frame(DATES[-30:])           # thin but current: warm-up as before
    run(["EDAPX", "THIN"])
    assert "EDAPX" not in fetches
    assert "THIN" in fetches


def test_a_curated_stale_name_is_refreshed_and_screened_only_if_current(screen):
    frames, fetched, fetches, run = screen
    frames["AAPL"] = _frame(DATES[:-20])
    fetched["AAPL"] = _frame(DATES)                # the refresh brings it current
    assert "AAPL" in run(["AAPL"]) and fetches == ["AAPL"]
    fetches.clear()
    fetched["AAPL"] = _frame(DATES[:-20])          # the refresh changes nothing: dead
    assert "AAPL" not in run(["AAPL"]) and fetches == ["AAPL"]


def test_an_outage_that_freezes_every_cache_judges_nothing_stale(screen):
    frames, _, _, run = screen
    old = DATES[:-10]
    frames["SPY"] = _frame(old, jump=False, price=700.0)   # the benchmark froze too
    frames["LIVE"] = _frame(old)
    assert "LIVE" in run(["LIVE"])


def test_no_benchmark_means_no_staleness_verdict(screen):
    frames, _, _, run = screen
    del frames["SPY"]
    frames["DEAD"] = _frame(DATES[:-15])
    assert "DEAD" in run(["DEAD"])                 # cannot tell: the old behaviour


def test_sessions_behind_counts_benchmark_sessions():
    ref = sc._ref_sessions(_frame(DATES, jump=False))
    assert sc._sessions_behind(_frame(DATES), ref) == 0
    assert sc._sessions_behind(_frame(DATES[:-19]), ref) == 19
    assert sc._sessions_behind(None, ref) is None and sc._sessions_behind(_frame(DATES), None) is None


# ── the discovery liquidity gate: an empty fetch is not retried the same day ──

@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(settings, "enable_discovery_liquidity_gate", True)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "enable_security_type_filter", False)
    monkeypatch.setattr(settings, "liquidity_gate_fetch_workers", 1, raising=False)
    monkeypatch.setattr(liq, "_NO_DATA", {})
    monkeypatch.setattr(liq, "is_valid_ticker", lambda t: True)
    monkeypatch.setattr(liq, "load_ohlcv", lambda t: None)          # every name is cold
    fetches = []
    data = {}
    monkeypatch.setattr(liq, "get_history", lambda t, period=None: fetches.append(t) or data.get(t))
    return data, fetches


def test_the_gate_does_not_refetch_a_name_with_no_bars_the_same_day(gate):
    data, fetches = gate
    data["GOOD"] = _frame(DATES[-40:])
    for _ in range(3):                                           # three ticks
        kept = liq.apply_liquidity_gate(["FGRSX", "GOOD"], budget={"n": 10})
        assert kept == ["GOOD"]
    assert fetches.count("FGRSX") == 1 and fetches.count("GOOD") == 3


def test_an_outage_marks_nothing(gate):
    _, fetches = gate                                            # every fetch empty
    for _ in range(2):
        assert liq.apply_liquidity_gate(["AAA", "BBB"], budget={"n": 10}) == []
    assert fetches.count("AAA") == 2 and fetches.count("BBB") == 2
    assert liq._NO_DATA == {}


def test_the_memory_expires_at_the_day_boundary(gate, monkeypatch):
    data, fetches = gate
    data["GOOD"] = _frame(DATES[-40:])
    liq.apply_liquidity_gate(["FGRSX", "GOOD"], budget={"n": 10})
    liq._NO_DATA["FGRSX"] = date.today() - timedelta(days=1)     # remembered yesterday
    liq.apply_liquidity_gate(["FGRSX", "GOOD"], budget={"n": 10})
    assert fetches.count("FGRSX") == 2
