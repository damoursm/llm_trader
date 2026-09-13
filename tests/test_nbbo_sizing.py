"""NBBO-based position sizing (src/performance/nbbo_sizing.py).

The properties worth pinning are the ones a future edit could break silently:
the two curves' SHAPES (a long step past the wide cut, a short boost that then
decreases monotonically forever), CONTINUITY across both boundaries (the whole
reason the smoothing bands exist), and the fail-soft contract — no measured
book, a disabled flag or a raising lookup must all leave sizing byte-identical
at 1.0 rather than quietly tilting on a proxy.
"""
import importlib

import pytest

from config.settings import settings
from src.performance import nbbo_sizing as ns


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Pin the curve parameters so a settings change can't silently rewrite the
    expectations below — the shape is what is under test, not today's numbers."""
    monkeypatch.setattr(settings, "enable_nbbo_sizing", True, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_long_wide_bps", 20.0, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_long_wide_mult", 0.35, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_short_knee_bps", 5.0, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_short_tight_mult", 1.5, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_short_decay_bps", 15.0, raising=False)
    monkeypatch.setattr(settings, "nbbo_size_floor_mult", 0.20, raising=False)


# ── LONG curve ──────────────────────────────────────────────────────────────
def test_long_is_full_size_up_to_the_cut():
    for bps in (0.1, 1.0, 5.0, 12.0, 19.9, 20.0):
        assert ns.long_multiplier(bps) == 1.0


def test_long_over_the_cut_is_sized_a_lot_less_than_under_it():
    """The directive: over 20 bp sized a lot less than under 20 bp."""
    under = ns.long_multiplier(19.0)
    over = ns.long_multiplier(30.0)
    assert under == 1.0
    assert over == pytest.approx(0.35)
    assert over <= 0.5 * under          # "a lot less", not a nudge


def test_long_curve_is_monotone_non_increasing_and_continuous():
    xs = [i * 0.05 for i in range(0, 1400)]          # 0 .. 70 bp
    ys = [ns.long_multiplier(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))
    assert max(abs(b - a) for a, b in zip(ys, ys[1:])) < 0.05   # no cliff


def test_long_ramp_sits_on_the_lenient_side_of_the_boundary():
    """A name a hair over the cut must not lose 3x its size to quote noise."""
    assert ns.long_multiplier(20.1) > 0.95
    assert ns.long_multiplier(24.0) == pytest.approx(0.35)


# ── SHORT curve ─────────────────────────────────────────────────────────────
def test_short_under_the_knee_is_sized_up():
    for bps in (0.1, 1.0, 3.0, 4.0):
        assert ns.short_multiplier(bps) == pytest.approx(1.5)
    assert ns.short_multiplier(4.5) > 1.0            # inside the smoothing band
    assert ns.short_multiplier(4.99) > 1.0


def test_short_is_monotonically_decreasing_at_and_above_the_knee():
    """The directive: monotonically decreasing weight for >= 5 bp."""
    xs = [5.0 + i * 0.1 for i in range(0, 1000)]     # 5 .. 105 bp
    ys = [ns.short_multiplier(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))
    assert ys[0] == pytest.approx(1.0)               # continuous through the knee
    strict = [b < a for a, b in zip(ys, ys[1:])]
    assert sum(strict) > 0.5 * len(strict)           # genuinely decreasing, not a step


def test_short_curve_is_monotone_over_its_whole_domain_and_continuous():
    xs = [i * 0.05 for i in range(0, 2000)]          # 0 .. 100 bp
    ys = [ns.short_multiplier(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))
    assert max(abs(b - a) for a, b in zip(ys, ys[1:])) < 0.05


def test_short_never_falls_below_the_floor():
    assert ns.short_multiplier(10_000.0) == pytest.approx(0.20)
    assert ns.short_multiplier(1e9) >= 0.20


def test_the_two_sides_get_different_treatment_at_the_same_book():
    """A 3 bp book sizes a short UP and leaves a long alone; a 40 bp book cuts
    the long hard and the short to roughly a quarter. One shared curve could
    not do both — that asymmetry is the whole point."""
    assert ns.short_multiplier(3.0) > ns.long_multiplier(3.0) == 1.0
    assert ns.long_multiplier(40.0) == pytest.approx(0.35)
    assert ns.short_multiplier(40.0) < 0.35


# ── the wired entry point ───────────────────────────────────────────────────
def test_multiplier_reads_the_quoted_book_per_side(monkeypatch):
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps",
                        lambda tk, session=None: {"bps": 2.0, "raw_bps": 2.0, "estimator": "nbbo"})
    m, d = ns.nbbo_size_multiplier("AAA", "SELL")
    assert m == pytest.approx(1.5) and d["side"] == "short" and d["estimator"] == "nbbo"
    m, d = ns.nbbo_size_multiplier("AAA", "BUY")
    assert m == 1.0 and d["side"] == "long"


def test_wide_book_cuts_the_long(monkeypatch):
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps",
                        lambda tk, session=None: {"bps": 35.0, "raw_bps": 35.0, "estimator": "nbbo"})
    m, d = ns.nbbo_size_multiplier("AAA", "BUY")
    assert m == pytest.approx(0.35) and d["bps"] == 35.0


def test_no_measured_book_means_no_tilt(monkeypatch):
    """The CS/class proxies measured no separation at any threshold, so their
    absence must read as 'no view', never as a wide book."""
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps",
                        lambda tk, session=None: None)
    for action in ("BUY", "SELL"):
        m, d = ns.nbbo_size_multiplier("AAA", action)
        assert m == 1.0 and d["reason"] == "no measured book"


def test_lookup_failure_is_fail_soft(monkeypatch):
    def _boom(tk, session=None):
        raise RuntimeError("quote feed down")
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps", _boom)
    m, _ = ns.nbbo_size_multiplier("AAA", "SELL")
    assert m == 1.0


def test_flag_off_is_exactly_neutral(monkeypatch):
    monkeypatch.setattr(settings, "enable_nbbo_sizing", False, raising=False)
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps",
                        lambda tk, session=None: {"bps": 1.0, "raw_bps": 1.0, "estimator": "nbbo"})
    for action in ("BUY", "SELL"):
        m, d = ns.nbbo_size_multiplier("AAA", action)
        assert m == 1.0 and d["reason"] == "disabled"


def test_boost_is_capped_at_the_tight_multiplier(monkeypatch):
    """Nothing downstream should ever see a multiplier above the configured
    boost, whatever a mis-set decay or floor would otherwise produce."""
    monkeypatch.setattr(settings, "nbbo_size_short_tight_mult", 1.5, raising=False)
    monkeypatch.setattr("src.performance.liquidity_forecast.quoted_halfspread_bps",
                        lambda tk, session=None: {"bps": 0.01, "raw_bps": 0.01, "estimator": "nbbo"})
    m, _ = ns.nbbo_size_multiplier("AAA", "SELL")
    assert m <= 1.5


# ── the session basis ───────────────────────────────────────────────────────
def test_quoted_halfspread_divides_out_the_session_widening(monkeypatch):
    """An overnight book must not be charged twice: the session haircut already
    prices it, so the threshold has to mean the same thing in every session."""
    lf = importlib.import_module("src.performance.liquidity_forecast")
    monkeypatch.setattr(settings, "enable_liquidity_forecast", True, raising=False)
    monkeypatch.setattr(lf, "_live_spread", lambda tk: 50.0)
    rth = lf.quoted_halfspread_bps("AAA", session="rth")
    overnight = lf.quoted_halfspread_bps("AAA", session="overnight")
    assert rth["bps"] == pytest.approx(50.0)
    assert overnight["raw_bps"] == pytest.approx(50.0)
    assert overnight["bps"] == pytest.approx(50.0 / float(settings.spread_overnight_multiplier))
    assert overnight["bps"] < rth["bps"]


@pytest.mark.parametrize("estimator", ["cs", "ar", "ibkr"])
def test_quoted_halfspread_takes_only_the_live_nbbo(monkeypatch, estimator):
    """Nothing but the live book may feed a LEVEL-based curve.

    CS/AR are volatility proxies rather than quotes. The IBKR sweep IS a
    measured book but sits at 0.78x the real one on this basis (769 tickers,
    RTH, 2026-09-02) — it disagrees about which side of the 4/5 bp line a name
    is on for 15-17% of names, which would over-boost shorts and under-cut longs
    with no visible symptom. A regression that reinstates either as a fallback
    must fail here.
    """
    lf = importlib.import_module("src.performance.liquidity_forecast")
    monkeypatch.setattr(settings, "enable_liquidity_forecast", True, raising=False)
    monkeypatch.setattr(lf, "_live_spread", lambda tk: None)
    monkeypatch.setattr(lf, "_structural",
                        lambda tk: {"struct_half_bps": 9.0, "estimator": estimator,
                                    "price": 50.0, "adv": 1e7})
    assert lf.quoted_halfspread_bps("AAA") is None


def test_quoted_halfspread_returns_the_live_book_unconverted(monkeypatch):
    lf = importlib.import_module("src.performance.liquidity_forecast")
    monkeypatch.setattr(settings, "enable_liquidity_forecast", True, raising=False)
    monkeypatch.setattr(lf, "_live_spread", lambda tk: 7.5)
    got = lf.quoted_halfspread_bps("AAA", session="rth")
    assert got == {"bps": 7.5, "raw_bps": 7.5, "estimator": "nbbo"}
