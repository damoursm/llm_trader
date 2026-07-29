"""Inversion of ADDITIVE OVERLAY methods (2026-07-25).

The pooled methods invert via `weights[m] = -weights[m]`, which only reaches
entries of `_BASE_WEIGHTS`. Overlays -- `cross_sectional`, the `f_*` factors,
the `kaufman_*`/`adx_*` trend methods -- are added to `combined_score` OUTSIDE
that pool, so naming one in `inverted_methods` was ACCEPTED BY CONFIG AND THEN
SILENTLY DID NOTHING. That is the dangerous failure mode: it looks configured.

Why `cross_sectional` specifically: 40.8% gross win over 174 attributed trades,
permutation p(luck)=0.996 -- the worst-measuring method in the system, on the
LARGEST sample (100% ticker coverage) -- while keeping its full 0.20 additive
weight, because the win-rate filter also iterates `_BASE_WEIGHTS` only. `tech`,
at a strictly better 41.6% / p=0.988, was dropped outright.

All synthetic, no network.
"""

import pytest

from config.settings import settings
import src.signals.aggregator as agg


@pytest.fixture(autouse=True)
def _clean():
    """Inversion is read from settings each call; reset around every test."""
    before = settings.inverted_methods
    agg.reset_winrate_filter_cache()
    yield
    settings.inverted_methods = before
    agg.reset_winrate_filter_cache()


# ── the hook itself ────────────────────────────────────────────────────────

def test_overlay_sign_flips_only_for_an_inverted_method():
    settings.inverted_methods = "cross_sectional"
    assert agg._overlay_sign("cross_sectional") == -1.0
    assert agg._overlay_sign("news") == 1.0


def test_overlay_sign_is_case_and_space_insensitive():
    settings.inverted_methods = " Cross_Sectional , insider "
    assert agg._overlay_sign("CROSS_SECTIONAL") == -1.0
    assert agg._overlay_sign("insider") == -1.0


def test_no_inversion_configured_is_a_no_op():
    settings.inverted_methods = ""
    assert agg._overlay_sign("cross_sectional") == 1.0


# ── the property that was broken ───────────────────────────────────────────

def test_cross_sectional_overlay_actually_reverses_the_combine():
    """The regression. A bullish cross-sectional score must now PUSH THE
    COMBINE DOWN when the method is inverted.

    Against the un-fixed code this test fails: `cs_w` was read straight from
    settings, so the overlay added `+0.20 x cs` regardless of `inverted_methods`
    and the two branches below were identical.
    """
    cs = 0.8
    base = 0.10

    settings.inverted_methods = ""
    normal = base + float(settings.cross_sectional_weight) * agg._overlay_sign("cross_sectional") * cs

    settings.inverted_methods = "cross_sectional"
    inverted = base + float(settings.cross_sectional_weight) * agg._overlay_sign("cross_sectional") * cs

    assert normal > base, "un-inverted: a bullish cs score should raise the combine"
    assert inverted < base, "INVERTED: the same bullish cs score must LOWER it"
    assert normal != inverted, "the inversion must actually change the arithmetic"
    # Symmetric about the un-adjusted combine.
    assert (normal - base) == pytest.approx(-(inverted - base))


def test_inverting_one_overlay_leaves_the_others_alone():
    settings.inverted_methods = "cross_sectional"
    for other in ("f_value", "f_split", "kaufman_long", "adx_short"):
        assert agg._overlay_sign(other) == 1.0


def test_pooled_and_overlay_inversion_read_the_same_config():
    """One switch governs both paths — a method named once is inverted wherever
    it is applied, so the two mechanisms can never disagree about a method."""
    settings.inverted_methods = "insider,cross_sectional"
    inv = agg._inverted_methods()
    assert "insider" in inv and "cross_sectional" in inv
    assert agg._overlay_sign("cross_sectional") == -1.0
    assert agg._overlay_sign("insider") == -1.0


# ── overlays are now FILTERED too, not just invertible (2026-07-26) ────────

def test_overlay_factor_zeroes_a_filtered_overlay(monkeypatch):
    """The gap that made retiring the manual pins unsafe.

    An overlay is applied outside the normalised pool, so the win-rate filter
    used to skip it entirely — `cross_sectional` sat at its full 0.20 additive
    weight on a 40.8% win rate with nothing able to drop it. Un-inverting it
    would then have PROMOTED it from "backwards" to "full weight, unprotected",
    strictly worse than either state.
    """
    monkeypatch.setattr(agg, "winrate_filtered_methods",
                        lambda: frozenset({"cross_sectional"}))
    assert agg._overlay_factor("cross_sectional") == 0.0


def test_overlay_factor_carries_the_inversion_when_not_filtered(monkeypatch):
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())
    settings.inverted_methods = "cross_sectional"
    assert agg._overlay_factor("cross_sectional") == -1.0
    settings.inverted_methods = ""
    assert agg._overlay_factor("cross_sectional") == 1.0


def test_filter_wins_over_inversion(monkeypatch):
    """A method bad on BOTH sides is noise: dropping beats flipping."""
    monkeypatch.setattr(agg, "winrate_filtered_methods",
                        lambda: frozenset({"cross_sectional"}))
    settings.inverted_methods = "cross_sectional"
    assert agg._overlay_factor("cross_sectional") == 0.0


def test_overlay_factor_fails_soft(monkeypatch):
    """A broken filter lookup must never silently zero a live weight."""
    def boom(): raise RuntimeError("filter unavailable")
    monkeypatch.setattr(agg, "winrate_filtered_methods", boom)
    settings.inverted_methods = ""
    assert agg._overlay_factor("cross_sectional") == 1.0


def test_the_filter_now_judges_invertible_overlays(monkeypatch):
    """Before 2026-07-26 the filter iterated _BASE_WEIGHTS only, so an overlay
    could never be dropped however bad its record."""
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    monkeypatch.setattr(settings, "winrate_filter_threshold", 0.50)
    monkeypatch.setattr(settings, "winrate_filter_min_trades", 10)
    monkeypatch.setattr(settings, "enable_oos_validation", False)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)
    monkeypatch.setattr(
        tracker, "compute_solo_method_gross_winrate",
        lambda split=None, effective=False, **kw: {
            "cross_sectional": {"trades": 174, "win_rate": 40.8}})
    agg.reset_winrate_filter_cache()
    assert "cross_sectional" in agg.winrate_filtered_methods()
