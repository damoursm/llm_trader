"""The exit side must consume the CURRENT entry strategy (2026-08-18).

User directive: "make sure the latest models and strategies are used for exit."
Each coupling below is one that would fail INVISIBLY — the exit keeps running,
just against yesterday's strategy — so each is pinned mechanically:

  1. the exit `aggregator` signal is THIS tick's live combined_score (rank
     basis, shaping, stacker swap included by construction);
  2. every entry method reaches the exit re-score: a method added to
     `_ALL_METHODS` that never appears on the exit side would silently be
     entry-only;
  3. the ML exit model's feature set IS the stacker's SIGNED set
     (`EXIT_METHODS is STACKER_SIGNED_FEATURES`) — one list, so a stacker
     feature decision propagates to the exit model at its next retrain rather
     than forking, while the unsigned context features stay out because every
     exit feature is multiplied by the position's direction;
  4. the ml_exit artifact's stamped bases match what serving feeds it
     (label = pivot, combine = absolute), and BOTH feeders (dataset builder and
     live twin) prefer the abs shadow — the documented basis-invariance that
     keeps a trained model's features stable across combine changes.
"""

from types import SimpleNamespace

from src.performance.tracker import _ALL_METHODS, _method_scores_from_signal


def _sig(**over):
    """A TickerSignal stand-in carrying the fields the exit re-score reads
    DIRECTLY (attribute access, not getattr-with-default)."""
    base = dict(combined_score=0.0, direction="BULLISH",
                sentiment_score=0.0, technical_score=0.0, insider_score=0.0,
                put_call_score=0.0, max_pain_score=0.0, oi_skew_score=0.0,
                vwap_score=0.0, pattern_score=0.0, momentum_score=0.0,
                sector_momentum_score=0.0, money_flow_score=0.0,
                trend_strength_score=0.0, pead_score=0.0, iv_rank_score=0.0)
    base.update(over)
    return SimpleNamespace(**base)


def test_exit_aggregator_reads_the_live_combined_score():
    """The `aggregator` exit score must be the SAME TickerSignal the entry side
    built this tick — that is what makes the exit follow every entry-side
    change (rank basis, shaping, stacker swap) with zero extra wiring."""
    from src.analysis.exit_methods import build_exit_scores
    sig = _sig(combined_score=0.42)
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
             "entry_date": "2026-08-01"}
    scores = build_exit_scores(trade, None, {"AAA": sig}, None)
    assert scores.get("aggregator") == 0.42          # long: oriented = as-is
    trade_short = dict(trade, action="SELL", direction="BEARISH")
    scores_s = build_exit_scores(trade_short, None, {"AAA": sig}, None)
    assert scores_s.get("aggregator") == -0.42       # short: sign-oriented


def test_every_entry_method_reaches_the_exit_rescore():
    """`_method_scores_from_signal` is the exit side's window onto the entry
    methods. Its key set must equal `_ALL_METHODS` exactly: a missing key means
    a method that exists at entry but silently never at exit."""
    empty = _method_scores_from_signal("AAA", "BULLISH", None)
    assert set(empty) == set(_ALL_METHODS), (
        set(_ALL_METHODS) ^ set(empty) or "sets differ")
    # ...and with a real-ish signal the same keys come back (getattr defaults
    # cover fields the signal object lacks).
    sig = _sig(sentiment_score=0.1, technical_score=0.2)
    got = _method_scores_from_signal("AAA", "BULLISH", {"AAA": sig})
    assert set(got) == set(_ALL_METHODS)


def test_exit_model_features_are_the_stackers_signed_set_by_identity():
    """Still ONE list, narrowed 2026-09-07: the exit model derives from
    STACKER_SIGNED_FEATURES, not from the full live set.

    Every exit feature is oriented by the position's direction (`ex_<m>` =
    score x sign), so the unsigned context features added on 2026-09-07 — the
    article count, the recency mass, the market-state widths and the catalyst
    one-hot — cannot ride along: orienting a magnitude by a sign that means
    nothing there is not a feature, it is noise with a plausible name. Equality
    + source-derivation are asserted so the two sets can never fork, and the
    duplicate check still guards the additive-derivation bug."""
    import inspect

    from src.analysis.ml_exit_dataset import EXIT_METHODS
    from src.analysis.ml_stacker import (STACKER_CONTEXT_FEATURES,
                                         STACKER_SIGNED_FEATURES)
    assert list(EXIT_METHODS) == list(STACKER_SIGNED_FEATURES)
    assert len(set(EXIT_METHODS)) == len(EXIT_METHODS), "duplicate feature"
    assert not (set(EXIT_METHODS) & set(STACKER_CONTEXT_FEATURES))
    src = inspect.getsource(__import__("src.analysis.ml_exit_dataset",
                                       fromlist=["x"]))
    assert "EXIT_METHODS: List[str] = list(STACKER_SIGNED_FEATURES)" in src, (
        "EXIT_METHODS must be DERIVED from STACKER_SIGNED_FEATURES, not a copy "
        "that can drift")


def test_ml_exit_artifact_bases_match_serving():
    """The artifact must stamp label_basis=pv (the pivot standard) and
    combine_basis=absolute (what live_exit_features actually feeds). A mismatch
    means the weekly retrain and the serving path have diverged. Skipped when no
    artifact exists (fresh clone / CI) — absence is already fail-soft."""
    import pytest

    from src.analysis.ml_exit_dataset import _load_exit_artifact
    art = _load_exit_artifact()
    if art is None:
        pytest.skip("no ml_exit artifact on this machine")
    cfg = art.get("config") or {}
    assert cfg.get("label_basis") == "pv", cfg
    assert cfg.get("combine_basis") == "absolute", cfg


def test_both_ex_combine_feeders_prefer_the_abs_shadow():
    """Basis invariance holds only if the DATASET builder and the LIVE twin use
    the same preference (abs shadow first, combined_score fallback). One side
    switching alone shifts the trained model's feature distribution mid-hold —
    the exact failure the shadow column exists to prevent."""
    import inspect

    from src.analysis import ml_exit_dataset as m
    for fn in (m.build_exit_dataset, m.live_exit_features):
        src = inspect.getsource(fn)
        assert "combined_score_abs" in src, f"{fn.__name__} lost the abs preference"
