"""Per-SIDE win-rate filter + weighting (2026-07-24).

A method's bullish and bearish calls are separate skills — the audit measured
most methods near 40% on their BUY-side views and 47-59% on their SELL-side
ones. With the buy/sell split combine each camp is therefore filtered and
weighted on ITS OWN record: a method whose shorts work and whose longs don't is
kept for the bearish camp and dropped from the bullish one.

Covers the per-side gross win rate (tracker), the selection logic and weight
multipliers (aggregator), the empty-camp guard, and the wiring into the combine.
All fakes, no network.
"""

import pytest

from config.settings import settings
import src.signals.aggregator as agg
import src.performance.tracker as tracker


def _side_perf(**methods):
    """name -> {"buy": (n, wr), "sell": (n, wr)} → the per-side dict shape."""
    out = {"buy": {}, "sell": {}}
    for m, sides in methods.items():
        for side, (n, wr) in sides.items():
            out[side][m] = {"trades": n, "win_rate": wr}
    return out


def _patch_sides(monkeypatch, per_side, overall=None):
    """Patch the gross-winrate lookup to answer per side."""
    def _fake(split=None, side=None, **kw):
        if side in ("buy", "sell"):
            return per_side.get(side, {})
        return overall or {}
    monkeypatch.setattr(tracker, "compute_solo_method_gross_winrate", _fake)
    agg.reset_winrate_filter_cache()


def _base(monkeypatch):
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    monkeypatch.setattr(settings, "enable_side_winrate_filter", True)
    monkeypatch.setattr(settings, "enable_side_adaptive_weights", True)
    monkeypatch.setattr(settings, "winrate_filter_threshold", 0.50)
    monkeypatch.setattr(settings, "winrate_filter_min_trades", 10)
    monkeypatch.setattr(settings, "side_weight_prior_n", 10)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_oos_validation", False)
    # 2026-08-11 promotions score on the fixture caches and would pollute the
    # hand-built method worlds these tests construct - off in this file's base.
    for _f in ("enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
               "enable_rsi2_rev", "enable_dloc_rev", "enable_ml_ohlcv"):
        monkeypatch.setattr(settings, _f, False)
    agg.reset_winrate_filter_cache()


# ── the per-side gross win rate itself ──────────────────────────────────────

def test_gross_winrate_splits_by_side(monkeypatch):
    """side="buy" counts only bullish views, "sell" only bearish ones."""
    trades = [
        # method's view is +0.8 (bullish) and the stock ROSE → buy-side win
        {"status": "CLOSED", "entry_price": 100.0, "exit_price": 110.0,
         "method_scores": {"news": 0.8}},
        # bullish view, stock FELL → buy-side loss
        {"status": "CLOSED", "entry_price": 100.0, "exit_price": 90.0,
         "method_scores": {"news": 0.8}},
        # bearish view, stock FELL → sell-side win
        {"status": "CLOSED", "entry_price": 100.0, "exit_price": 90.0,
         "method_scores": {"news": -0.8}},
    ]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    buy = tracker.compute_solo_method_gross_winrate(side="buy")["news"]
    sell = tracker.compute_solo_method_gross_winrate(side="sell")["news"]
    both = tracker.compute_solo_method_gross_winrate()["news"]
    assert buy == {"trades": 2, "win_rate": 50.0}
    assert sell == {"trades": 1, "win_rate": 100.0}
    assert both == {"trades": 3, "win_rate": pytest.approx(66.7)}


# ── selection logic ─────────────────────────────────────────────────────────

def test_drops_only_the_failing_side(monkeypatch):
    """The headline case: longs are a coin-flip loser, shorts work. The method
    must leave the bullish camp and stay in the bearish one."""
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(
        news={"buy": (110, 46.4), "sell": (38, 55.3)},
    ))
    assert "news" in agg.side_filtered_methods("buy")
    assert "news" not in agg.side_filtered_methods("sell")


def test_respects_min_trades_per_side(monkeypatch):
    """A thin side is not evidence — 30% over 5 views keeps full weight."""
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(news={"buy": (5, 30.0), "sell": (40, 60.0)}))
    assert agg.side_filtered_methods("buy") == frozenset()


def test_exactly_50_is_kept(monkeypatch):
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(news={"buy": (20, 50.0), "sell": (20, 50.0)}))
    assert agg.side_filtered_methods("buy") == frozenset()
    assert agg.side_filtered_methods("sell") == frozenset()


def test_inverted_methods_exempt_per_side(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    _patch_sides(monkeypatch, _side_perf(
        insider={"buy": (77, 37.7), "sell": (77, 51.9)},
        news={"buy": (110, 40.0), "sell": (38, 60.0)},
    ))
    assert "insider" not in agg.side_filtered_methods("buy")   # sign already corrected
    assert "news" in agg.side_filtered_methods("buy")


def test_flag_off_is_noop(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setattr(settings, "enable_side_winrate_filter", False)
    _patch_sides(monkeypatch, _side_perf(news={"buy": (110, 10.0), "sell": (38, 10.0)}))
    assert agg.side_filtered_methods("buy") == frozenset()


# ── per-side weight multipliers ─────────────────────────────────────────────

def test_side_weights_reward_the_stronger_side(monkeypatch):
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(vwap={"buy": (72, 51.4), "sell": (163, 58.9)}))
    buy = agg.side_weight_multipliers("buy")["vwap"]
    sell = agg.side_weight_multipliers("sell")["vwap"]
    assert sell > buy > 0.9, f"the stronger side should weigh more (buy={buy}, sell={sell})"


def test_side_weights_shrink_small_samples_toward_neutral(monkeypatch):
    """A 3-view 100% side must be pulled hard toward 1.0 — nowhere near the
    2.0× a literal reading of 100% would earn. (Note the shrinkage does NOT
    rank it below a 200-view 60% method: with prior_n=10 they land at 1.23×
    and 1.19×. Both are bounded and close to neutral, which is the point.)"""
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(
        news={"buy": (3, 100.0), "sell": (10, 50.0)},
        vwap={"buy": (200, 60.0), "sell": (10, 50.0)},
    ))
    mults = agg.side_weight_multipliers("buy")
    assert mults["news"] < 1.4, "a 3-view 100% side must not earn a large boost"
    assert 1.0 < mults["vwap"] < 1.4
    # A 50%-over-many-views side sits at exactly neutral.
    assert agg.side_weight_multipliers("sell")["vwap"] == pytest.approx(1.0, abs=0.02)


def test_zero_percent_side_is_not_read_as_a_coin_flip(monkeypatch):
    """Regression: ``rec.get("win_rate", 50.0) or 50.0`` rewrote a genuine 0.0%
    win rate as 50% (0.0 is falsy), so the worst possible method escaped the
    filter AND got a neutral weight."""
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(news={"buy": (20, 0.0), "sell": (20, 60.0)}))
    assert "news" in agg.side_filtered_methods("buy"), "0% must be filtered, not exempt"
    assert agg.side_weight_multipliers("buy")["news"] < 1.0, "0% must not weigh 1.0x"


def test_side_weights_no_record_is_neutral(monkeypatch):
    _base(monkeypatch)
    _patch_sides(monkeypatch, _side_perf(news={"buy": (20, 60.0), "sell": (20, 60.0)}))
    assert agg.side_weight_multipliers("buy")["vwap"] == 1.0     # no record → inert


def test_side_weights_respect_clamps(monkeypatch):
    _base(monkeypatch)
    monkeypatch.setattr(settings, "side_weight_min_multiplier", 0.8)
    monkeypatch.setattr(settings, "side_weight_max_multiplier", 1.2)
    _patch_sides(monkeypatch, _side_perf(
        news={"buy": (500, 100.0), "sell": (500, 0.0)},
    ))
    assert agg.side_weight_multipliers("buy")["news"] == 1.2
    assert agg.side_weight_multipliers("sell")["news"] == 0.8


# ── the guard: a filter must never empty a camp ─────────────────────────────

def test_empty_camp_guard_keeps_the_book_two_sided(monkeypatch):
    """If every live method fails on one side, suppress that camp's filter for
    the run — an always-zero camp would make the book structurally
    single-direction, a far bigger bet than the filter is entitled to place."""
    from tests.test_winrate_filter import _setup_build, _perf, _patch_perf
    _base(monkeypatch)
    _setup_build(monkeypatch, massive_score=0.8)
    # Global filter sees a healthy method; the per-side view fails EVERY method
    # on the buy side.
    _patch_sides(
        monkeypatch,
        _side_perf(massive={"buy": (20, 10.0), "sell": (20, 90.0)},
                   news={"buy": (20, 10.0), "sell": (20, 90.0)}),
        overall=_perf(massive=(20, 60.0), news=(20, 60.0)),
    )
    sigs = agg.build_signals(["AAA", "BBB"], [])
    # massive = +0.8 is a BULLISH view; with the guard the bullish camp survives.
    assert sigs and all(s.combined_buy_score > 0 for s in sigs)


# ── stream separation: the per-side filter must reach coherence too ─────────
#
# The combine drops a method's BULLISH contribution when it is buy-filtered.
# coherence_factor / sources_agreeing / family agreement must see the SAME
# book, or a method judged sub-coin-flip on this side still inflates the
# agreement count (helping pass Gate 1b) and the confidence factor while
# contributing nothing to the score. Found 2026-07-25 during the stream audit —
# the code comment claimed the invariant held, but only the GLOBAL filter was
# applied there; the per-side sets were used solely in the combine loop.

def test_side_filtered_method_is_dropped_from_agreement_and_coherence(monkeypatch):
    """A SECOND, unfiltered bullish method must drive the direction — otherwise
    combined collapses to 0, sources_agreeing is 0 either way, and the test
    passes without exercising the fix at all (verified: it did)."""
    from tests.test_winrate_filter import _setup_build, _perf
    _base(monkeypatch)
    _setup_build(monkeypatch, massive_score=0.8)
    # news is bullish and NOT filtered -> it alone must carry the call.
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.6, "bullish"))
    # massive is healthy OVERALL (global filter keeps it) but its BUY side is
    # sub-coin-flip, so the bullish camp must exclude it.
    _patch_sides(
        monkeypatch,
        _side_perf(massive={"buy": (40, 20.0), "sell": (40, 80.0)},
                   news={"buy": (40, 70.0), "sell": (40, 70.0)}),
        overall=_perf(massive=(40, 60.0), news=(40, 70.0)),
    )
    sigs = agg.build_signals(["AAA", "BBB"], [])
    assert sigs
    for s in sigs:
        assert s.combined_buy_score > 0, "news should carry the bullish camp"
        # massive = +0.8 is a BULLISH view and is buy-filtered: it contributed
        # nothing to the score, so it must not count as an agreeing source
        # either (that count gates Gate 1b) nor prop up coherence.
        assert s.sources_agreeing == 1, (
            f"expected only news to agree, got {s.sources_agreeing} — a buy-filtered "
            "method is still counted toward sources_agreeing/coherence")
        assert s.massive_score == 0.8, "raw score still recorded for the panel"


def test_side_filter_leaves_the_other_side_agreement_intact(monkeypatch):
    """The mirror: a SELL-filtered method must still count on a bullish call."""
    from tests.test_winrate_filter import _setup_build, _perf
    _base(monkeypatch)
    _setup_build(monkeypatch, massive_score=0.8)
    _patch_sides(
        monkeypatch,
        _side_perf(massive={"buy": (40, 80.0), "sell": (40, 20.0)}),
        overall=_perf(massive=(40, 60.0)),
    )
    sigs = agg.build_signals(["AAA", "BBB"], [])
    assert sigs and all(s.combined_buy_score > 0 for s in sigs)
    assert all(s.sources_agreeing >= 1 for s in sigs), (
        "a SELL-filtered method must still count on its BULLISH call")
