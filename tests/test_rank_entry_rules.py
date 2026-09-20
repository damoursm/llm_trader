"""The 2026-09-18 entry stack: the gap-cluster rule, the own-history freshness
filter, the shadow arm and ml_ohlcv as the sole combine.

The first test is the one that matters most: with every new flag at its default
the selector is byte-identical to the shipped rank rule, because all four pieces
land on a LIVE trading path and are meant to be inert until a deliberate cutover.
"""

from __future__ import annotations

import pytest

from config import settings
from src.signals import rank_entry, score_history


class _Sig:
    """The slice of TickerSignal the selector touches."""

    def __init__(self, ticker, combined, ml=None, direction="NEUTRAL"):
        self.ticker = ticker
        self.combined_score = combined
        self.ml_ohlcv_score = combined if ml is None else ml
        self.direction = direction
        self.confidence = 0.5
        self.sources_agreeing = 3
        self.rationale = "x"
        self.price = 10.0


def _signals(scores, directions=None):
    out = []
    for i, v in enumerate(scores):
        d = directions[i] if directions else ("BULLISH" if v > 0.1 else
                                              "BEARISH" if v < -0.1 else "NEUTRAL")
        out.append(_Sig(f"T{i:03d}", v, direction=d))
    return out


def _actions(recs):
    return {r.ticker: r.action for r in recs if r.action in ("BUY", "SELL")}


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_rule", "topk", raising=False)
    monkeypatch.setattr(settings, "rank_entry_score", "combined", raising=False)
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", False, raising=False)
    monkeypatch.setattr(settings, "ml_ohlcv_sole_combine", False, raising=False)
    monkeypatch.setattr(settings, "gate1_rank_cap", 3, raising=False)
    score_history.reset_cache()
    yield
    score_history.reset_cache()


# ── the inertness contract ───────────────────────────────────────────────────

def test_defaults_reproduce_the_shipped_rank_rule():
    """Flags off ⇒ band ∧ top-K by combined_score, exactly as before."""
    sigs = _signals([0.9, 0.8, 0.7, 0.6, 0.05, -0.6, -0.7, -0.8, -0.9])
    got = _actions(rank_entry.build_rank_recommendations(sigs))
    assert got == {"T000": "BUY", "T001": "BUY", "T002": "BUY",
                   "T008": "SELL", "T007": "SELL", "T006": "SELL"}


def test_shipped_rule_still_needs_the_band():
    """A name with the top score but no fired band is not bought."""
    sigs = _signals([0.9, 0.8], directions=["NEUTRAL", "BULLISH"])
    assert _actions(rank_entry.build_rank_recommendations(sigs)) == {"T001": "BUY"}


# ── the gap cluster ──────────────────────────────────────────────────────────

def test_gap_cluster_takes_the_names_above_the_hole(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    # Two names far above a tight, evenly spaced pack: one big hole at rank 2.
    scores = [0.90, 0.88] + [0.30 - 0.001 * i for i in range(40)]
    got = _actions(rank_entry.build_rank_recommendations(_signals(scores)))
    assert got.get("T000") == "BUY" and got.get("T001") == "BUY"
    assert "T002" not in got


def test_gap_cluster_abstains_without_a_hole(monkeypatch):
    """No cluster ⇒ no trade. This is the rule's OWN abstention, which is why it
    does not consult the direction band — a test pins that it really abstains."""
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    scores = [0.50 - 0.01 * i for i in range(40)]         # perfectly even spacing
    assert _actions(rank_entry.build_rank_recommendations(_signals(scores))) == {}


def test_gap_cluster_ignores_the_band(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    scores = [0.90, 0.88] + [0.30 - 0.001 * i for i in range(40)]
    sigs = _signals(scores)
    for s in sigs:                                        # every band off
        s.direction = "NEUTRAL"
    assert _actions(rank_entry.build_rank_recommendations(sigs)).get("T000") == "BUY"


def test_gap_cluster_is_mirrored_at_the_bottom(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    scores = [0.30 - 0.001 * i for i in range(40)] + [-0.88, -0.90]
    got = _actions(rank_entry.build_rank_recommendations(_signals(scores)))
    assert got.get("T041") == "SELL" and got.get("T040") == "SELL"


def test_gap_multiple_is_honoured(monkeypatch):
    """The SAME hole must pass at 10x and fail at 40x, so the setting is shown to
    be what decides rather than the shape of the fixture."""
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    # Even 0.001 spacing with one 0.020 hole after the first two names: the local
    # spacing around it is 0.001, so the ratio is 20.
    scores = [0.900, 0.899] + [0.879 - 0.001 * i for i in range(40)]
    monkeypatch.setattr(settings, "rank_entry_gap_multiple", 10.0, raising=False)
    assert set(_actions(rank_entry.build_rank_recommendations(_signals(scores)))) == {"T000", "T001"}
    monkeypatch.setattr(settings, "rank_entry_gap_multiple", 40.0, raising=False)
    assert _actions(rank_entry.build_rank_recommendations(_signals(scores))) == {}


# ── the own-history freshness filter ─────────────────────────────────────────

def _standings(monkeypatch, table):
    monkeypatch.setattr(score_history, "load_standings", lambda column="ml_ohlcv": table)


def test_freshness_drops_a_name_that_is_not_a_new_high(monkeypatch):
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    # T000 has been higher before; T001 has not.
    _standings(monkeypatch, {"T000": (50, 0.95, -0.5), "T001": (50, 0.10, -0.5)})
    sigs = _signals([0.90, 0.80], directions=["BULLISH", "BULLISH"])
    assert _actions(rank_entry.build_rank_recommendations(sigs)) == {"T001": "BUY"}


def test_freshness_keeps_a_name_with_no_standing(monkeypatch):
    """Fewer than `rank_entry_fresh_min_history` prior scores is 'never rated',
    not 'rated badly' — the cohort measured best in nine of eleven rules."""
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    _standings(monkeypatch, {"T000": (3, 0.95, -0.5)})     # thin history
    sigs = _signals([0.90], directions=["BULLISH"])
    assert _actions(rank_entry.build_rank_recommendations(sigs)) == {"T000": "BUY"}


def test_freshness_is_mirrored_for_shorts(monkeypatch):
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    _standings(monkeypatch, {"T000": (50, 0.5, -0.95), "T001": (50, 0.5, -0.10)})
    sigs = _signals([-0.90, -0.80], directions=["BEARISH", "BEARISH"])
    got = _actions(rank_entry.build_rank_recommendations(sigs))
    assert got == {"T001": "SELL"}                         # T000 has been lower before


def test_standings_never_straddle_a_scorer_epoch(monkeypatch):
    """The window must start at the scorer epoch when that is later than the
    plain look-back. ml_ohlcv's scale moves with every retrain (|0.0375| on the
    daily 1% artifact against |0.097| on the 30-minute 5% one), so a window
    spanning a retrain reads nearly every name as a new high and the filter
    quietly becomes a no-op at exactly the moment it is most trusted."""
    import datetime as _dt
    import pandas as pd
    seen = {}

    class _Repo:
        @staticmethod
        def fetch_df(sql, params=None):
            if "DISTINCT signal_date" in sql:
                return pd.DataFrame({"signal_date": ["2026-08-01", "2026-08-02"]})
            seen["start"] = params[0]
            return pd.DataFrame({"ticker": ["A"], "n": [40], "hi": [0.5], "lo": [-0.5]})

    import src.db
    monkeypatch.setattr(src.db, "repo", _Repo)      # `from src.db import repo` reads the attribute
    monkeypatch.setattr("src.signals.method_epochs.epoch_for",
                        lambda m: _dt.date(2026, 9, 18), raising=False)
    monkeypatch.setattr(score_history, "_today", lambda: "2026-09-30")
    score_history.reset_cache()
    score_history.load_standings("ml_ohlcv")
    assert seen["start"] == "2026-09-18"        # the epoch, not the 2026-08-01 look-back


def test_freshness_fails_open(monkeypatch):
    """A broken standings query must revert to the unfiltered cut, never empty
    the book: the unfiltered cut is the measured fallback, an empty one is not."""
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    monkeypatch.setattr(score_history, "load_standings",
                        lambda column="ml_ohlcv": {})
    sigs = _signals([0.90, 0.80], directions=["BULLISH", "BULLISH"])
    assert len(_actions(rank_entry.build_rank_recommendations(sigs))) == 2


# ── which score the rule ranks on ────────────────────────────────────────────

def test_ranking_on_ml_ohlcv_uses_that_column(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_score", "ml_ohlcv", raising=False)
    monkeypatch.setattr(settings, "gate1_rank_cap", 1, raising=False)
    sigs = [_Sig("A", combined=0.9, ml=0.1, direction="BULLISH"),
            _Sig("B", combined=0.1, ml=0.9, direction="BULLISH")]
    assert _actions(rank_entry.build_rank_recommendations(sigs)) == {"B": "BUY"}


def test_panel_column_follows_the_score():
    from src.signals.rank_entry import _panel_column, _score_column
    assert (_score_column(), _panel_column()) == ("combined", "combined_score")


# ── the shadow arm ───────────────────────────────────────────────────────────

def test_shadow_rule_returns_only_actionable_picks(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_shadow_rule", "top_pct", raising=False)
    monkeypatch.setattr(settings, "rank_entry_shadow_pct", 0.05, raising=False)
    recs = rank_entry.build_shadow_recommendations(_signals([0.1 * i for i in range(-20, 20)]))
    assert recs and all(r.action in ("BUY", "SELL") for r in recs)
    assert len([r for r in recs if r.action == "BUY"]) == 2      # 5% of 40


def test_shadow_does_not_disturb_the_live_selection(monkeypatch):
    """The shadow must be a pure function of the cross-section — computing it
    cannot change what the live rule picked."""
    sigs = _signals([0.9, 0.8, 0.7, -0.7, -0.8, -0.9])
    before = _actions(rank_entry.build_rank_recommendations(sigs))
    rank_entry.build_shadow_recommendations(sigs)
    assert _actions(rank_entry.build_rank_recommendations(sigs)) == before


# ── the A/B between two selection rules ──────────────────────────────────────

def test_ab_off_means_arm_a_decides_every_run():
    assert settings.rank_entry_ab_share == 0.0
    assert all(rank_entry.resolve_arm(f"run_{i}") == "topk" for i in range(50))


def test_ab_is_deterministic_on_the_run_id(monkeypatch):
    """A retried run must land on the SAME arm, or the sample over-represents
    exactly the runs that failed once."""
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.5, raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    first = [rank_entry.resolve_arm(f"2026-09-21_{i:06d}") for i in range(40)]
    again = [rank_entry.resolve_arm(f"2026-09-21_{i:06d}") for i in range(40)]
    assert first == again


def test_ab_splits_runs_near_the_share(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.5, raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule_b", "top_pct", raising=False)
    arms = [rank_entry.resolve_arm(f"2026-09-21_{i:06d}") for i in range(400)]
    share_b = arms.count("top_pct") / len(arms)
    assert 0.40 < share_b < 0.60, share_b
    assert set(arms) == {"gap_cluster", "top_pct"}


def test_each_arm_stamps_its_own_model(monkeypatch):
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    assert rank_entry.model_stamp("gap_cluster") == "rank-gap10f"
    assert rank_entry.model_stamp("top_pct") == "rank-top5f"
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", False, raising=False)
    assert rank_entry.model_stamp("gap_cluster") == "rank-gap10"
    assert rank_entry.model_stamp("topk") == rank_entry.RANK_MODEL


def test_shadow_auto_records_the_arm_that_did_not_decide(monkeypatch):
    """Every run yields a matched pair on one cross-section, which is what makes
    the two rules comparable at all."""
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.5, raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule_b", "top_pct", raising=False)
    monkeypatch.setattr(settings, "rank_entry_shadow_rule", "auto", raising=False)
    for i in range(30):
        rid = f"2026-09-21_{i:06d}"
        assert rank_entry.shadow_rule_for(rid) != rank_entry.resolve_arm(rid)


def test_ab_arm_actually_changes_what_is_picked(monkeypatch):
    """The flip must reach SELECTION, not just the stamp."""
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule_b", "top_pct", raising=False)
    monkeypatch.setattr(settings, "rank_entry_shadow_pct", 0.05, raising=False)
    scores = [0.50 - 0.01 * i for i in range(40)]         # evenly spaced: no cluster
    sigs = _signals(scores)
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.0, raising=False)
    assert _actions(rank_entry.build_rank_recommendations(sigs, run_id="r1")) == {}
    monkeypatch.setattr(settings, "rank_entry_ab_share", 1.0, raising=False)
    assert len(_actions(rank_entry.build_rank_recommendations(sigs, run_id="r1"))) == 4


# ── union routing: both rules decide every run ───────────────────────────────

@pytest.fixture
def _union(monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_union", True, raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule", "gap_cluster", raising=False)
    monkeypatch.setattr(settings, "rank_entry_rule_b", "top_pct", raising=False)
    monkeypatch.setattr(settings, "rank_entry_shadow_pct", 0.05, raising=False)
    return None


def _union_fixture():
    """A cross-section with a clear top cluster, so the two rules disagree."""
    return _signals([0.90, 0.88] + [0.30 - 0.001 * i for i in range(58)])


def test_union_takes_both_rules_every_run(_union):
    assert rank_entry.active_rules("r1") == ["gap_cluster", "top_pct"]
    got = _actions(rank_entry.build_rank_recommendations(_union_fixture(), run_id="r1"))
    # gap cluster: the 2 names above the hole; top 5% of 60: 3 a side
    assert {"T000", "T001", "T002"} <= set(got)
    assert got["T000"] == "BUY" and got["T059"] == "SELL"


def test_union_is_a_superset_of_each_rule(_union, monkeypatch):
    sigs = _union_fixture()
    union = set(_actions(rank_entry.build_rank_recommendations(sigs, run_id="r1")))
    monkeypatch.setattr(settings, "rank_entry_union", False, raising=False)
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.0, raising=False)
    a = set(_actions(rank_entry.build_rank_recommendations(sigs, run_id="r1")))
    monkeypatch.setattr(settings, "rank_entry_ab_share", 1.0, raising=False)
    b = set(_actions(rank_entry.build_rank_recommendations(sigs, run_id="r1")))
    assert a and b and a | b == union


def test_union_overrides_the_ab_share(_union, monkeypatch):
    monkeypatch.setattr(settings, "rank_entry_ab_share", 0.5, raising=False)
    assert all(len(rank_entry.active_rules(f"r{i}")) == 2 for i in range(20))


def test_a_name_both_rules_pick_is_one_recommendation(_union):
    recs = rank_entry.build_rank_recommendations(_union_fixture(), run_id="r1")
    tickers = [r.ticker for r in recs if r.action in ("BUY", "SELL")]
    assert len(tickers) == len(set(tickers)), "a shared pick must not be emitted twice"


def test_shared_picks_name_both_rules_in_the_rationale(_union):
    recs = {r.ticker: r for r in rank_entry.build_rank_recommendations(_union_fixture(), run_id="r1")}
    assert "Gap cluster + Top percentile" in recs["T000"].rationale


def test_union_stamp_names_both_rules(_union, monkeypatch):
    monkeypatch.setattr(settings, "enable_rank_entry_freshness", True, raising=False)
    assert rank_entry.model_stamp(rank_entry.active_rules("r1")) == "rank-gap10f+rank-top5f"


def test_picks_for_rule_isolates_one_rule(_union):
    sigs = _union_fixture()
    a = {r.ticker for r in rank_entry.picks_for_rule(sigs, rule="gap_cluster")}
    b = {r.ticker for r in rank_entry.picks_for_rule(sigs, rule="top_pct")}
    assert a and b and a != b and a <= set(_actions(
        rank_entry.build_rank_recommendations(sigs, run_id="r1")))


# ── ml_ohlcv as the sole combine ─────────────────────────────────────────────

def test_sole_combine_bands_and_divisor_are_this_score_s_own():
    from src.signals.aggregator import _direction_bands, _raw_confidence_scale
    lo, sh = _direction_bands("ml_ohlcv")
    assert (lo, sh) == (settings.ml_ohlcv_diff_threshold_long,
                        settings.ml_ohlcv_diff_threshold_short)
    assert _raw_confidence_scale("ml_ohlcv") == settings.ml_ohlcv_raw_confidence_scale
    # and the other combines are untouched
    assert _direction_bands("ml") != (lo, sh)


def test_sole_combine_leaves_the_absolute_twin_alone():
    """`combined_score_abs` is what ml_exit reads as `ex_combine`, and it exists
    to be BASIS-INVARIANT so the exit model's feature does not step when the
    combine changes shape. Sole-combine must not write the ml_ohlcv score there:
    |combined_score_abs| averages 0.20-0.25 against that score's 0.07-0.11, so it
    would shrink the series ~2.5x on the model that closes every position."""
    import ast, io
    src = io.open("src/signals/aggregator.py", encoding="utf-8").read()
    tree = ast.parse(src)
    # no assignment to the absolute camps may sit inside the sole-combine branch
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "_sole"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                names = {t.id for t in ast.walk(sub) if isinstance(t, ast.Name)
                         and isinstance(t.ctx, ast.Store)}
                assert not names & {"_abs_buy", "_abs_sell", "_abs_combined"}, (
                    f"sole-combine branch assigns the absolute twin: {names}")
    # and the absolute total is computed unconditionally
    assert "_abs_combined = _apply_overlays(_abs_buy - _abs_sell)" in src


def test_sole_combine_is_off_by_default():
    assert settings.ml_ohlcv_sole_combine is False
    assert settings.rank_entry_rule == "topk"
    assert settings.enable_rank_entry_freshness is False
    assert settings.enable_rank_entry_shadow is False


# ── the 30-minute serving guard ──────────────────────────────────────────────

def test_a_30m_artifact_never_reaches_the_daily_path(monkeypatch):
    """The dispatch is on the ARTIFACT's own stamp. Without it a 30-minute model
    is fed daily features and scores silently wrong, which is the failure that
    looks like a working deploy."""
    from src.signals import ml_model
    seen = {}
    monkeypatch.setattr(ml_model, "_load_artifact",
                        lambda: {"config": {"feature_bars": "30m", "target": "pivot_rank"},
                                 "features": [], "model": None})
    monkeypatch.setattr(ml_model, "_score_30m",
                        lambda t, a: (seen.setdefault("hit", t), (0.42, "OK"))[1])
    assert ml_model.compute_ml_score("AAPL") == (0.42, "OK")
    assert seen["hit"] == "AAPL"


def test_a_daily_artifact_still_takes_the_daily_path(monkeypatch):
    from src.signals import ml_model
    monkeypatch.setattr(ml_model, "_load_artifact",
                        lambda: {"config": {"target": "pivot_rank"}, "features": [], "model": None})
    monkeypatch.setattr(ml_model, "_score_30m",
                        lambda t, a: pytest.fail("daily artifact took the 30m path"))
    ml_model.compute_ml_score("AAPL")        # may abstain; must not raise or dispatch
