"""The tick persists the pre-combine market state (2026-09-25): `atr_pct`,
`bb_width_pct`, `vol_ratio` and `tape_score` land in `signals` under the replay's
column names — the values the stackers read when they serve, NULL (never NaN)
when not computed — and the replay restore still prefers a replayed value."""
import inspect
import math

import pandas as pd
import pytest

from config import settings
from src.analysis.technical import TechnicalResult
from src.db import repo
from src.db.schema import REPLAYABLE_CONTEXT_COLUMNS, SIGNAL_MARKET_STATE_COLUMNS
from src.signals import aggregator as agg
from src.signals.agreement import TapeCheck


def _row(**kw):
    r = {"ticker": "AAA", "type": "STOCK", "direction": "bullish",
         "combined_score": 0.2, "confidence": 0.7, "price": 100.0,
         "scores": {"news": 0.2}}
    r.update(kw)
    return r


def test_the_columns_carry_the_replay_context_names():
    """One name per quantity: the panel's replay merge overwrites a same-named
    column, so a live column under another name would sit beside it unmerged."""
    assert set(SIGNAL_MARKET_STATE_COLUMNS) <= set(REPLAYABLE_CONTEXT_COLUMNS)


def test_insert_signals_roundtrips_the_values_and_stores_null_not_nan():
    repo.insert_signals("run-ms", "2026-09-25T14:00:00+00:00", "2026-09-25", [
        _row(ticker="AAA", atr_pct=0.0312, bb_width_pct=0.081, vol_ratio=1.42,
             tape_score=-0.35),
        _row(ticker="BBB")])
    df = repo.fetch_df(
        "SELECT ticker, atr_pct, bb_width_pct, vol_ratio, tape_score, "
        "atr_pct IS NULL AS a0, bb_width_pct IS NULL AS b0, vol_ratio IS NULL AS v0, "
        "tape_score IS NULL AS t0 FROM signals ORDER BY ticker")
    a, b = df.iloc[0], df.iloc[1]
    assert (a["atr_pct"], a["bb_width_pct"], a["vol_ratio"], a["tape_score"]) == \
        pytest.approx((0.0312, 0.081, 1.42, -0.35))
    assert bool(b["a0"]) and bool(b["b0"]) and bool(b["v0"]) and bool(b["t0"])


def test_ctx6_rounds_like_the_replay_and_refuses_non_finite():
    assert agg._ctx6(0.123456789) == 0.123457
    assert agg._ctx6(-0.35) == -0.35
    for bad in (None, float("nan"), float("inf"), float("-inf"), "x"):
        assert agg._ctx6(bad) is None


def _setup(monkeypatch, tech=True, tape=True):
    from tests.test_news_events import _OFF
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", False)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(settings, "enable_technical_analysis", tech)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "enable_tape_confirmation", tape)
    monkeypatch.setattr(agg, "_ML_ARM_OVERRIDE", False)
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.3, "n"))
    monkeypatch.setattr(agg, "compute_technical_score",
                        lambda ticker, df=None: TechnicalResult(
                            score=0.2, vol_ratio=1.2345678, atr_pct=0.0412345678,
                            bb_width_pct=0.0912345678))
    monkeypatch.setattr(agg, "compute_tape_confirmation",
                        lambda ticker, df=None: TapeCheck(score=0.4444444444,
                                                          label="BULLISH_TAPE"))


def test_the_tick_fills_them_from_its_own_technical_pass_and_tape_check(monkeypatch):
    _setup(monkeypatch)
    s = agg.build_signals(["FAKEAAA"], articles=[], snapshots=[])[0]
    assert s.atr_pct == 0.041235 and s.bb_width_pct == 0.091235
    assert s.vol_ratio == 1.234568 and s.tape_score == 0.444444


def test_a_pass_that_did_not_run_leaves_them_null(monkeypatch):
    _setup(monkeypatch, tech=False, tape=False)
    s = agg.build_signals(["FAKEAAA"], articles=[], snapshots=[])[0]
    assert (s.atr_pct, s.bb_width_pct, s.vol_ratio, s.tape_score) == (None, None, None, None)


def test_the_raw_tape_is_kept_when_only_the_stackers_ask_for_it(monkeypatch):
    """`tape_confirmation_score` is flag-scoped for the confidence factor;
    `tape_score` is the check the stackers read, which runs for the ML arm even
    with tape confirmation off."""
    _setup(monkeypatch, tape=False)
    monkeypatch.setattr(agg, "_ML_ARM_OVERRIDE", True)
    s = agg.build_signals(["FAKEAAA"], articles=[], snapshots=[])[0]
    assert s.tape_score == 0.444444 and s.tape_confirmation_score == 0.0


def test_the_pipeline_persists_every_column():
    """A column the pipeline stops writing fails HERE instead of reading NULL
    forever (the inert-mechanism rule)."""
    import src.pipeline as p
    src = inspect.getsource(p._persist_run)
    for c in SIGNAL_MARKET_STATE_COLUMNS:
        assert f'"{c}": getattr(s, "{c}", None)' in src


def test_a_replayed_value_still_overwrites_the_live_one(monkeypatch):
    """Where the EOD replay reached a run, the panel keeps the replayed value;
    the live value only fills the rows it has not reached."""
    from src.analysis import replay
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-09-24"], "ticker": ["AAA"],
        "generated_at": ["2026-09-24T19:00:00"], "atr_pct": [0.05],
        "tape_score": [0.2]}))
    df = pd.DataFrame({
        "signal_date": ["2026-09-24", "2026-09-25"], "ticker": ["AAA", "AAA"],
        "generated_at": ["2026-09-24T19:00:00", "2026-09-25T14:00:00"],
        "atr_pct": [0.04, 0.03], "tape_score": [0.1, math.nan]})
    out, restored = replay.restore_replayed(df, methods=("atr_pct", "tape_score"))
    assert out["atr_pct"].tolist() == pytest.approx([0.05, 0.03])
    assert out.loc[0, "tape_score"] == pytest.approx(0.2)
    assert math.isnan(out.loc[1, "tape_score"])
    assert restored["atr_pct"].tolist() == [True, False]


def test_stacker_training_warns_on_thin_coverage_not_on_absence():
    """The columns now always exist, so a missing replay shows only as thin
    COVERAGE — which is what the training guard has to judge."""
    from loguru import logger

    from src.analysis import ml_stacker as ms
    full = pd.DataFrame({c: [0.1, 0.2, 0.3, 0.4, 0.5] for c in SIGNAL_MARKET_STATE_COLUMNS})
    assert ms._warn_thin_market_state(full) == []
    thin = full.copy()
    thin.loc[:2, "tape_score"] = math.nan                          # 40% populated
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        assert ms._warn_thin_market_state(thin) == ["tape_score"]
        assert ms._warn_thin_market_state(full.drop(columns=["vol_ratio"])) == ["vol_ratio"]
    finally:
        logger.remove(sink)
    assert any("tape_score is only 40% populated" in m for m in msgs)
