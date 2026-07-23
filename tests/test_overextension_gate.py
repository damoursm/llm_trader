"""Overextension / anti-chase gate (actionable-filter Gate 5, 2026-07-22).

The BUY-vs-SELL forensics found the BUY side's failure mechanism is CHASING:
40% of BUY calls landed in the top quintile of trailing-5d gainers (2x the
universe share) and hit 38% at 5d (median −2.4% vs SPY) — recent big gainers
mean-revert while the combined score peaks exactly then. Blocking actionable
BUYs whose trailing 5-bar run-up exceeds 12% removed a cohort measured at a
32.5% hit rate / −4.1% median and lifted the kept cohort to 43.4%.

Invariants under test:
  • blocks a BUY above the run-up threshold, passes one at/below it
  • SELL is NEVER gated (fading spikes / riding crashes is measured edge —
    the call site applies the predicate to BUYs only)
  • fail-OPEN: no cache / short history / bad closes → not overextended
  • flag off → never blocks
  • the funnel classifier recognises the "overextended" stamp
"""

import pandas as pd
import pytest

from config.settings import settings
from src.pipeline import _is_overextended, _recent_runup_pct


def _bars(closes):
    idx = pd.date_range("2026-07-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


@pytest.fixture
def cache(monkeypatch):
    store = {}
    import src.data.cache as cache_mod
    monkeypatch.setattr(cache_mod, "load_ohlcv", lambda tk, **kw: store.get(tk))
    monkeypatch.setattr(settings, "enable_overextension_gate", True)
    monkeypatch.setattr(settings, "overextension_runup_pct", 12.0)
    monkeypatch.setattr(settings, "overextension_lookback_bars", 5)
    return store


def test_runup_math(cache):
    # 100 → 120 over the last 5 bars = +20%
    cache["CHASE"] = _bars([90, 95, 100, 104, 108, 112, 116, 120])
    assert _recent_runup_pct("CHASE") == pytest.approx((120 - 100) / 100 * 100)


def test_blocks_buy_beyond_threshold(cache):
    cache["CHASE"] = _bars([100, 100, 100, 105, 110, 115, 118, 120])   # +20% / 5 bars
    assert _is_overextended("CHASE") is True


def test_allows_buy_at_or_below_threshold(cache):
    cache["CALM"] = _bars([100, 100, 100, 101, 102, 103, 104, 105])    # +5% / 5 bars
    assert _is_overextended("CALM") is False
    # exactly AT the threshold is not "more than" — passes
    cache["EDGE"] = _bars([100, 100, 100, 103, 106, 109, 111, 112])    # +12.0%
    assert _is_overextended("EDGE") is False


def test_pullback_requalifies(cache):
    """The gate defers, not bans: once the name cools below the bar it passes."""
    cache["COOL"] = _bars([100, 105, 110, 118, 122, 118, 112, 108])    # +8% vs 5 bars ago...
    # trailing 5: last=108 vs closes[-6]=110 → −1.8% — cooled, passes
    assert _is_overextended("COOL") is False


def test_crash_is_not_overextended(cache):
    """A big DROP is not a run-up — the gate is one-sided by construction."""
    cache["CRSH"] = _bars([100, 100, 100, 95, 88, 80, 75, 70])         # −30%
    assert _is_overextended("CRSH") is False


def test_fail_open_no_cache(cache):
    assert _recent_runup_pct("MISSING") is None
    assert _is_overextended("MISSING") is False


def test_fail_open_short_history(cache):
    cache["NEW"] = _bars([100, 108, 118])          # < lookback+1 bars
    assert _recent_runup_pct("NEW") is None
    assert _is_overextended("NEW") is False


def test_fail_open_bad_closes(cache):
    cache["BAD"] = _bars([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0])   # prev close 0
    assert _is_overextended("BAD") is False


def test_flag_off_never_blocks(cache, monkeypatch):
    cache["CHASE"] = _bars([100, 100, 100, 110, 120, 130, 140, 150])
    monkeypatch.setattr(settings, "enable_overextension_gate", False)
    assert _is_overextended("CHASE") is False


def test_lookback_setting_respected(cache, monkeypatch):
    """+15% happened over 5 bars but only +6% over the last 3 — a 3-bar lookback
    must read the shorter window."""
    monkeypatch.setattr(settings, "overextension_lookback_bars", 3)
    cache["T"] = _bars([100, 100, 100, 104, 109, 109, 112, 115.5])
    # 3-bar: 115.5 vs closes[-4]=109 → +5.96%
    assert _recent_runup_pct("T") == pytest.approx((115.5 - 109) / 109 * 100)
    assert _is_overextended("T") is False


def test_funnel_recognises_overextended_stamp():
    """The stage-eval classifier must route the new stamp to Gate 5, not let it
    fall through to the legacy Gate 3/4 count-attribution."""
    from src.performance.tracker import _classify_stage_outcome, _STAGE_OUTCOME_G5
    call = {"ticker": "CHASE", "confidence": 0.95, "action": "BUY", "actionable": False}
    ctx = {"threshold": 0.85, "allow_buys": True,
           "outcomes": {"CHASE": "overextended"}, "eb": 0, "ut": 5}
    assert _classify_stage_outcome(call, ctx) == _STAGE_OUTCOME_G5
