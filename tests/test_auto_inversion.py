"""Automatic method inversion + post-inversion verification (2026-07-25).

Before this, inversion was a hand-edited `.env` string and an inverted method got
a BLANKET exemption from the win-rate filter — so nothing ever checked that
flipping a method actually produced a better-than-chance one. An inversion could
sit at full weight forever on the strength of the raw record that justified
flipping it.

Three rules are pinned here:
  * inverting requires the raw record to be significantly worse than chance —
    a PLAIN per-method test on that method's full history, with NO
    multiple-comparison correction (2026-07-26): the methods are ~0.72
    correlated over 246k scored ticker-days, so a correction computed from the
    test COUNT over-corrects on an independence assumption that does not hold;
  * the finding must REPLICATE on the independent panel arm before anything is
    inverted — different data, ~1000x the observations, a different statistic.
    That is what replaces the correction, and it handles correlated methods by
    construction;
  * an inverted method must then clear the ordinary >50% bar ON ITS CORRECTED
    RECORD to stay in the combine — ties count as losses BOTH ways, so
    `flipped != 100 - raw` and a mostly-tied method can fail in both directions.
    That method is noise, not backwards information.

All synthetic, no network, no production ledger.
"""

import pytest

from config.settings import settings
import src.signals.aggregator as agg
import src.performance.tracker as tk


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_auto_inversion", True)
    monkeypatch.setattr(settings, "inversion_min_trades", 30)
    monkeypatch.setattr(settings, "inversion_alpha", 0.05)
    # Arm A isolated by default; replication has its own tests at the end.
    monkeypatch.setattr(settings, "inversion_require_replication", False)
    tk._AUTO_INVERSION_CACHE.clear()
    agg.reset_winrate_filter_cache()
    yield
    tk._AUTO_INVERSION_CACHE.clear()
    agg.reset_winrate_filter_cache()


def _fake_winrates(monkeypatch, table, ties=0.0):
    """table: {method: (win_rate_pct, n)}."""
    def fake(split=None, side=None, effective=False):
        out = {}
        for m, (w, n) in table.items():
            wr = (100.0 - w - ties) if (effective and m in agg._inverted_methods()) else w
            out[m] = {"trades": n, "win_rate": round(wr, 1)}
        return out
    monkeypatch.setattr(tk, "compute_solo_method_gross_winrate", fake)
    monkeypatch.setattr(tk, "_tie_share", lambda method, split: ties)
    return fake


# ── the statistic ──────────────────────────────────────────────────────────

def test_binomial_tail_is_exact():
    """Sanity-anchor the p-value against hand-computable cases."""
    assert tk._binom_p_below_half(0, 1) == pytest.approx(0.5)
    assert tk._binom_p_below_half(1, 2) == pytest.approx(0.75)
    assert tk._binom_p_below_half(5, 10) == pytest.approx(0.623, abs=1e-3)
    # A coin-flip record is never evidence of being backwards.
    assert tk._binom_p_below_half(50, 100) > 0.4
    # A strongly one-sided record is.
    assert tk._binom_p_below_half(30, 100) < 1e-4


def test_a_coinflip_method_is_never_inverted(monkeypatch):
    _fake_winrates(monkeypatch, {"news": (50.0, 200)})
    assert tk.calibrate_method_inversion() == {}


def test_only_methods_the_combine_USES_are_candidates(monkeypatch):
    """A name that is not a weighted method or an invertible overlay can never
    be auto-inverted — inverting something the combine never reads is a no-op
    dressed as a decision."""
    _fake_winrates(monkeypatch, {"not_a_method": (5.0, 500)})
    assert tk.calibrate_method_inversion() == {}


# ── bar 1: plain per-method significance (no correction) ───────────────────

def test_significance_is_plain_and_per_method(monkeypatch):
    """No multiple-comparison correction, deliberately (2026-07-26).

    A correction computed from the COUNT of tests is indefensible here: the
    methods are ~0.72 correlated over 246k scored ticker-days (momentum vs
    market_momentum is 0.98 — the same signal twice), so Bonferroni
    over-corrects on an independence assumption that does not hold, while plain
    alpha reports ONE phenomenon as N findings. The count is not the right
    denominator, so a method's verdict must not depend on how many OTHER
    methods happened to be judged alongside it.
    """
    marginal = {"tech": (41.0, 120)}                    # p ~ 0.02
    _fake_winrates(monkeypatch, marginal)
    tk._AUTO_INVERSION_CACHE.clear()
    alone = "tech" in tk.calibrate_method_inversion()

    many = dict(marginal)
    many.update({m: (50.0, 100) for m in agg._BASE_WEIGHTS if m != "tech"})
    _fake_winrates(monkeypatch, many)
    tk._AUTO_INVERSION_CACHE.clear()
    crowded = "tech" in tk.calibrate_method_inversion()

    assert alone is True, "a p~0.02 record should qualify on the ledger arm"
    assert crowded == alone, (
        "the verdict changed because OTHER methods were judged alongside it — "
        "that is the correction this design deliberately removed")


def test_thin_samples_are_never_inverted(monkeypatch):
    """Below the evidence floor a method is UNPROVEN, not backwards."""
    _fake_winrates(monkeypatch, {"pead": (10.0, 12)})      # awful but tiny
    assert tk.calibrate_method_inversion() == {}


def test_an_overwhelming_record_does_invert(monkeypatch):
    _fake_winrates(monkeypatch, {"tech": (25.0, 200)})
    out = tk.calibrate_method_inversion()
    assert "tech" in out
    assert out["tech"]["effective_win_rate"] == pytest.approx(75.0)


# ── bar 2: the flipped method must actually be good ────────────────────────

def test_a_mostly_tied_method_is_not_inverted(monkeypatch):
    """The case that makes bar 2 more than a formality: ties count as losses on
    BOTH sides, so a method can be significantly worse than chance raw AND still
    below 50% once flipped. It is noise — leave it to the ordinary filter."""
    # 30% raw with 45% ties → flipped = 100 - 30 - 45 = 25% — still terrible.
    _fake_winrates(monkeypatch, {"money_flow": (30.0, 200)}, ties=45.0)
    assert tk.calibrate_method_inversion() == {}, (
        "flipping a mostly-tied method does not produce a >50% method")


def test_flip_must_clear_the_same_threshold_the_filter_uses(monkeypatch):
    _fake_winrates(monkeypatch, {"vwap": (30.0, 200)}, ties=25.0)   # flipped = 45%
    assert tk.calibrate_method_inversion() == {}
    _fake_winrates(monkeypatch, {"vwap": (30.0, 200)}, ties=5.0)    # flipped = 65%
    tk._AUTO_INVERSION_CACHE.clear()
    assert "vwap" in tk.calibrate_method_inversion()


# ── manual pins ────────────────────────────────────────────────────────────

def test_manual_pin_wins_even_with_no_statistical_support(monkeypatch):
    """A human override must survive the auto-detector declining to invert —
    that is how the three live inversions keep working below the bar."""
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    _fake_winrates(monkeypatch, {"insider": (50.0, 200)})
    assert tk.calibrate_method_inversion() == {}
    assert "insider" in agg._inverted_methods()


def test_auto_and_manual_are_unioned(monkeypatch):
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    _fake_winrates(monkeypatch, {"tech": (25.0, 200), "insider": (50.0, 200)})
    assert agg._inverted_methods() >= {"insider", "tech"}


def test_disabling_auto_leaves_only_the_pins(monkeypatch):
    monkeypatch.setattr(settings, "enable_auto_inversion", False)
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    _fake_winrates(monkeypatch, {"tech": (25.0, 200)})
    assert agg._inverted_methods() == frozenset({"insider"})


def test_auto_detection_failure_falls_back_to_pins(monkeypatch):
    """Fail-soft: a broken detector must not silently un-invert a live method."""
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    def boom(*a, **k): raise RuntimeError("ledger unavailable")
    monkeypatch.setattr(tk, "calibrate_method_inversion", boom)
    assert agg._inverted_methods() == frozenset({"insider"})


# ── the filter no longer exempts an inverted method ────────────────────────

def test_inverted_method_is_judged_on_its_CORRECTED_record(monkeypatch):
    """The exemption this replaces: an inverted method used to be skipped by the
    filter entirely. It must now earn its place on the flipped record."""
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    # raw 40% → corrected 60%: passes on merit, not by exemption.
    _fake_winrates(monkeypatch, {m: (40.0, 200) for m in agg._BASE_WEIGHTS})
    agg.reset_winrate_filter_cache()
    assert "insider" not in agg.winrate_filtered_methods()


def test_inverted_method_still_below_50_after_flipping_is_DROPPED(monkeypatch):
    """The requirement in full: inversion is not a permanent exemption. If the
    corrected record is still sub-coin-flip, the method leaves the combine."""
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)
    monkeypatch.setattr(settings, "enable_winrate_method_filter", True)
    # raw 30% with 45% ties → corrected 25%: bad both ways.
    _fake_winrates(monkeypatch, {m: (30.0, 200) for m in agg._BASE_WEIGHTS}, ties=45.0)
    agg.reset_winrate_filter_cache()
    assert "insider" in agg.winrate_filtered_methods()


# ── inversion is PER-METHOD, never per-family ──────────────────────────────

def test_a_family_name_is_not_an_inversion_target(monkeypatch):
    """Inversion applies to individual methods only. A FAMILY
    (agreement.METHOD_FAMILIES) is a grouping used for breadth voting, not a
    signal with a sign, so it has nothing to invert — and must not silently
    look configured."""
    from src.signals.agreement import METHOD_FAMILIES
    invertible = set(agg._BASE_WEIGHTS) | set(agg.INVERTIBLE_OVERLAYS)
    assert not (set(METHOD_FAMILIES) & invertible), (
        "no family name may collide with an invertible method name")

    monkeypatch.setattr(settings, "inverted_methods", "Rel-Strength")
    agg._INVERSION_NAME_WARNED.clear()
    names = agg._manual_inverted_methods()
    # Parsed, but inert: it flips no weight and no overlay.
    assert not (names & invertible)
    assert agg._overlay_sign("cross_sectional") == 1.0


def _captured_warnings(fn):
    """Collect loguru WARNINGs emitted by `fn` (pytest's caplog only sees the
    stdlib logger; this project logs through loguru)."""
    from loguru import logger
    msgs: list = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink)
    return msgs


def test_unknown_inversion_name_is_warned_not_silently_ignored(monkeypatch):
    """A typo used to SILENTLY un-invert a live method — the same
    'looks configured, does nothing' failure as the overlay bug."""
    monkeypatch.setattr(settings, "inverted_methods", "insidr")
    agg._INVERSION_NAME_WARNED.clear()
    msgs = _captured_warnings(agg._manual_inverted_methods)
    assert any("insidr" in m and "not an invertible method" in m for m in msgs),         "an unrecognised name must warn"


def test_a_family_name_warns_that_inversion_is_per_method(monkeypatch):
    monkeypatch.setattr(settings, "inverted_methods", "Rel-Strength")
    agg._INVERSION_NAME_WARNED.clear()
    msgs = _captured_warnings(agg._manual_inverted_methods)
    assert any("FAMILY" in m and "per-method" in m for m in msgs),         "a family name should say so, not just 'unknown'"


def test_valid_names_produce_no_warning(monkeypatch):
    monkeypatch.setattr(settings, "inverted_methods", "insider,cross_sectional")
    agg._INVERSION_NAME_WARNED.clear()
    msgs = _captured_warnings(agg._manual_inverted_methods)
    assert not [m for m in msgs if "not an invertible method" in m]


def test_inverting_one_member_does_not_invert_its_family(monkeypatch):
    """Inverting `insider` corrects ITS vote inside Smart-Money; the family's
    other member (`broker_advisor`) is untouched, and no family-level sign
    exists to flip."""
    from src.signals.agreement import METHOD_FAMILIES
    monkeypatch.setattr(settings, "inverted_methods", "insider")
    monkeypatch.setattr(settings, "enable_auto_inversion", False)
    inv = agg._inverted_methods()
    smart = {m.lower() for m in METHOD_FAMILIES["Smart-Money"]}
    assert "insider" in inv
    assert smart - inv == {"broker_advisor"}, (
        "only the named member is inverted, not the whole family")


# ── replication: a ledger finding must reproduce on the panel ──────────────

def _fake_panel(monkeypatch, qualifying):
    monkeypatch.setattr(tk, "panel_inversion_evidence",
                        lambda: {m: {"qualifies": m in qualifying,
                                     "excess": {"1d": -0.5}, "t": {"1d": 3.0}}
                                 for m in agg._BASE_WEIGHTS})


def test_a_ledger_finding_that_does_not_reproduce_is_NOT_inverted(monkeypatch):
    """The situation measured live on 2026-07-26: SIX methods cleared the plain
    ledger test (p 0.0055-0.0165) and none reproduced on the panel. Correlated
    at +0.72, they are one unconfirmed phenomenon counted six times — which no
    alpha setting could distinguish, and replication does by construction."""
    monkeypatch.setattr(settings, "inversion_require_replication", True)
    _fake_winrates(monkeypatch, {"tech": (25.0, 200)})
    _fake_panel(monkeypatch, qualifying=set())
    tk._AUTO_INVERSION_CACHE.clear()
    assert tk.calibrate_method_inversion() == {}


def test_a_finding_confirmed_by_BOTH_arms_is_inverted(monkeypatch):
    monkeypatch.setattr(settings, "inversion_require_replication", True)
    _fake_winrates(monkeypatch, {"tech": (25.0, 200)})
    _fake_panel(monkeypatch, qualifying={"tech"})
    tk._AUTO_INVERSION_CACHE.clear()
    out = tk.calibrate_method_inversion()
    assert "tech" in out
    assert out["tech"]["arm"] == "ledger+panel"


def test_panel_alone_is_not_enough(monkeypatch):
    """Symmetric: the panel confirming something the ledger never flagged is
    not two independent pieces of evidence either."""
    monkeypatch.setattr(settings, "inversion_require_replication", True)
    _fake_winrates(monkeypatch, {"tech": (50.0, 200)})
    _fake_panel(monkeypatch, qualifying={"tech"})
    tk._AUTO_INVERSION_CACHE.clear()
    assert tk.calibrate_method_inversion() == {}


def test_replication_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inversion_require_replication", False)
    _fake_winrates(monkeypatch, {"tech": (25.0, 200)})
    _fake_panel(monkeypatch, qualifying=set())
    tk._AUTO_INVERSION_CACHE.clear()
    assert "tech" in tk.calibrate_method_inversion()
