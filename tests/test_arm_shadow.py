"""Shadow synthesis arms (2026-07-25).

Each tick the live arm's recommendations drive the run and the OTHER arms are
asked the same question about the same tickers, so the arms become comparable
per ticker-day. These tests pin the properties that make a shadow call safe to
run in production and honest to compare against:

  * the live arm is REUSED, never re-asked (a second call would cost money and
    could answer differently, breaking the pairing);
  * shadow calls are pinned to the engine the live arm used, so the ARM is the
    only difference — and pinning is also what suppresses the run/sentiment
    provenance writes that would otherwise mis-stamp the run;
  * a shadow failure is invisible to the run.

All fakes, no network.
"""

import threading

import pytest

from config.settings import settings
from src.analysis import arm_shadow


class _Rec:
    def __init__(self, ticker, action="BUY"):
        self.ticker, self.action = ticker, action
        self.direction, self.confidence = "BULLISH", 0.9


class _Sig:
    def __init__(self, ticker, price=100.0):
        self.ticker, self.price = ticker, price


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_shadow_arms", True)


def _start(generate, live_arm="dual", live_recs=None, **kw):
    return arm_shadow.maybe_start(
        signals=[_Sig("AAA")], live_arm=live_arm,
        live_recs=live_recs if live_recs is not None else [_Rec("AAA")],
        synth_kwargs={}, generate=generate, **kw)


# ── what gets asked ────────────────────────────────────────────────────────

def test_live_arm_is_reused_not_reasked():
    """Re-asking the live arm would double its cost and could return a
    different answer than the one that actually drove the run."""
    asked = []

    def gen(signals, **kw):
        asked.append((kw.get("dual_case"), kw.get("blind_synthesis")))
        return [_Rec("AAA")]

    rows = _start(gen, live_arm="dual").rows(timeout=10)
    assert (True, False) not in asked, "the live (dual) arm must not be re-asked"
    assert sorted(asked) == [(False, False), (False, True)]
    assert {r["arm"] for r in rows} == {"dual", "blind", "sighted"}
    assert [r["live"] for r in rows].count(True) == 1


def test_shadow_calls_are_pinned_to_the_live_engine():
    """The arm must be the only difference — an unpinned call could land on a
    different engine and confound the comparison. Pinning also suppresses the
    provenance writes, so a shadow can never mis-stamp the run."""
    engines = []

    def gen(signals, **kw):
        engines.append(kw.get("force_engine"))
        return [_Rec("AAA")]

    _start(gen, force_engine="qwen").rows(timeout=10)
    assert engines == ["qwen", "qwen"]


def test_each_live_arm_asks_the_other_two():
    for live in arm_shadow.ARMS:
        asked = []

        def gen(signals, _a=asked, **kw):
            _a.append(arm_shadow.live_arm_name(kw.get("dual_case"),
                                               kw.get("blind_synthesis")))
            return [_Rec("AAA")]

        rows = _start(gen, live_arm=live).rows(timeout=10)
        assert live not in asked
        assert set(asked) == set(arm_shadow.ARMS) - {live}
        assert {r["arm"] for r in rows} == set(arm_shadow.ARMS)


# ── failure containment ────────────────────────────────────────────────────

def test_one_failing_arm_does_not_lose_the_others():
    def gen(signals, **kw):
        if kw.get("blind_synthesis"):
            raise RuntimeError("engine exploded")
        return [_Rec("AAA")]

    rows = _start(gen, live_arm="dual").rows(timeout=10)
    assert {r["arm"] for r in rows} == {"dual", "sighted"}


def test_empty_result_records_nothing_for_that_arm():
    """A forced-engine failure returns [] — that must record no call rather
    than a fabricated one."""
    rows = _start(lambda signals, **kw: []).rows(timeout=10)
    assert {r["arm"] for r in rows} == {"dual"}


def test_join_timeout_still_persists_the_live_arm():
    stuck = threading.Event()

    def gen(signals, **kw):
        stuck.wait(30)
        return [_Rec("AAA")]

    try:
        rows = _start(gen).rows(timeout=0.2)
        assert [r["arm"] for r in rows] == ["dual"]
        assert rows[0]["live"] is True
    finally:
        stuck.set()


# ── gating ─────────────────────────────────────────────────────────────────

def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "enable_shadow_arms", False)
    assert _start(lambda signals, **kw: [_Rec("AAA")]) is None


def test_no_live_recommendations_returns_none():
    """Nothing to pair against — an LLM outage run must not fire shadow calls."""
    assert _start(lambda signals, **kw: [_Rec("AAA")], live_recs=[]) is None


def test_live_arm_name_matches_pipeline_precedence():
    """dual supersedes blind, mirroring the pipeline's own flip."""
    assert arm_shadow.live_arm_name(True, False) == "dual"
    assert arm_shadow.live_arm_name(True, True) == "dual"
    assert arm_shadow.live_arm_name(False, True) == "blind"
    assert arm_shadow.live_arm_name(False, False) == "sighted"
