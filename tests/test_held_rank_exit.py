"""held_rank exit signal (2026-08-22, panel-first).

The position's CURRENT aggregate score ranked within its OWN tick history since
entry -- "is this position's signal now at its weakest since we got in". Pinned
here: the sign convention, the symmetry of the mid-rank mapping, the abstain
floor, the abs-basis pool, the OPEN-only pool filter, and the pipeline ordering
the live pool depends on (monitor BEFORE this tick's signals persist).
"""

from types import SimpleNamespace

import pytest

from config.settings import settings
from src.analysis.exit_methods import (EXIT_DECISION_METHODS, EXIT_METHOD_LABELS,
                                       _abs_combine, _held_rank_pools,
                                       build_exit_scores, held_rank_score,
                                       reset_cache)


def _sig(**over):
    base = dict(combined_score=0.05, combined_score_abs=0.05, direction="BULLISH",
                sentiment_score=0.0, technical_score=0.0, insider_score=0.0,
                put_call_score=0.0, max_pain_score=0.0, oi_skew_score=0.0,
                vwap_score=0.0, pattern_score=0.0, momentum_score=0.0,
                sector_momentum_score=0.0, money_flow_score=0.0,
                trend_strength_score=0.0, pead_score=0.0, iv_rank_score=0.0)
    base.update(over)
    return SimpleNamespace(**base)


# -- the score itself ---------------------------------------------------------

def test_midrank_is_symmetric():
    """The naive rank/n mapping scored the pool minimum -0.6 but the maximum
    +1.0 -- a bullish bias the persisted panel would inherit. Mid-rank is
    symmetric: min -(n-1)/n, max +(n-1)/n, all-ties exactly 0."""
    pool = [0.1, 0.2, 0.3, 0.4]
    assert held_rank_score(pool, 0.5) == pytest.approx(0.8)
    assert held_rank_score(pool, 0.05) == pytest.approx(-0.8)
    assert held_rank_score([0.2] * 4, 0.2) == 0.0


def test_abstains_below_min_pool():
    """A rank among two observations is a coin -- 0.0 (no view), never a guess."""
    assert held_rank_score([0.1], 0.5) == 0.0
    assert held_rank_score([0.1, 0.2, 0.3], 0.5, min_ticks=5) == 0.0
    assert held_rank_score([0.1, 0.2, 0.3, 0.4], 0.5, min_ticks=5) != 0.0


def test_nan_now_abstains():
    assert held_rank_score([0.1, 0.2, 0.3, 0.4], float("nan")) == 0.0


def test_short_orientation_flips_the_rank(monkeypatch):
    """For a SHORT, a more-negative combine is a STRONGER signal. The caller
    orients pool and now by dir_sign, so the same raw history that reads
    'weakest since entry' for a long reads 'strongest' for a short."""
    monkeypatch.setattr(settings, "held_rank_min_ticks", 4)
    raw = [0.4, 0.3, 0.2, 0.1]          # combine declining since entry
    now = 0.05                           # today: weakest raw
    long_score = held_rank_score([v * 1.0 for v in raw], now * 1.0)
    short_score = held_rank_score([v * -1.0 for v in raw], now * -1.0)
    assert long_score < 0 < short_score
    assert long_score == pytest.approx(-short_score)


# -- pool construction --------------------------------------------------------

def test_abs_combine_prefers_the_shadow():
    """One quantity per pool: a held window can span an ml-arm flip, and the
    stacker scale (~0.06) mixed with the weighted scale (~0.30) is not a rank."""
    assert _abs_combine(SimpleNamespace(combined_score_abs=0.3, combined_score=0.06)) == 0.3
    assert _abs_combine(SimpleNamespace(combined_score_abs=None, combined_score=0.06)) == 0.06
    assert _abs_combine(None) is None


def test_pool_query_reads_the_abs_column():
    import inspect
    src = inspect.getsource(_held_rank_pools)
    assert "combined_score_abs" in src


def test_pool_filters_to_open_trades(monkeypatch):
    """The monitor passes the FULL ledger. Without the OPEN filter the query
    widens to every closed trade since June -- hundreds of tickers."""
    captured = {}

    def fake_fetch(sql, params=None):
        captured["params"] = params
        import pandas as pd
        return pd.DataFrame()

    monkeypatch.setattr("src.db.repo.fetch_df", fake_fetch)
    reset_cache()
    trades = [
        {"ticker": "OPEN1", "status": "OPEN", "entry_date": "2026-08-20"},
        {"ticker": "CLOSED1", "status": "CLOSED", "entry_date": "2026-06-01"},
    ]
    _held_rank_pools(trades, force=True)
    assert "OPEN1" in (captured.get("params") or [])
    assert "CLOSED1" not in (captured.get("params") or [])
    # min_entry comes from the OPEN trade, not the June-closed one
    assert "2026-08-20" in (captured.get("params") or [])
    reset_cache()


# -- wiring -------------------------------------------------------------------

def test_registered_as_held_only_decision_method():
    """EXIT_DECISION_METHODS membership keeps it OUT of the universe shadow
    book (it needs an entry date) and in the decision category on the dashboard."""
    assert "held_rank" in EXIT_DECISION_METHODS
    assert "held_rank" in EXIT_METHOD_LABELS


def test_excluded_from_the_consensus():
    """Panel-first probation: it must not nudge the confidence floor until the
    panel proves it."""
    from src.analysis.exit_conviction import _CONSENSUS_SKIP
    assert "held_rank" in _CONSENSUS_SKIP


def test_build_exit_scores_emits_it_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "held_rank_min_ticks", 3)
    reset_cache()
    import pandas as pd

    def fake_fetch(sql, params=None):
        return pd.DataFrame({
            "ticker": ["AAA"] * 4,
            "generated_at": [f"2026-08-2{i}T10:00:00+00:00" for i in range(4)],
            "combined_score_abs": [0.30, 0.25, 0.20, 0.15],
            "combined_score": [0.30, 0.25, 0.20, 0.15],
        })
    monkeypatch.setattr("src.db.repo.fetch_df", fake_fetch)
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
             "status": "OPEN", "entry_datetime": "2026-08-20T00:00:00+00:00",
             "entry_date": "2026-08-20"}
    scores = build_exit_scores(trade, None, {"AAA": _sig(combined_score_abs=0.05)},
                               None, _hr_all_trades=[trade])
    reset_cache()
    assert "held_rank" in scores
    assert scores["held_rank"] < 0          # today's 0.05 is the weakest since entry


def test_disabled_flag_suppresses_it(monkeypatch):
    monkeypatch.setattr(settings, "enable_held_rank_exit_signal", False)
    reset_cache()
    trade = {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
             "status": "OPEN", "entry_date": "2026-08-20"}
    scores = build_exit_scores(trade, None, {"AAA": _sig()}, None,
                               _hr_all_trades=[trade])
    assert "held_rank" not in scores
    reset_cache()


def test_monitor_runs_before_signals_persist():
    """The live pool assumes THIS tick's signals rows are NOT yet in the DB
    (today's value is appended from memory, exactly once). That holds because
    monitor_open_positions runs before _persist_run in run_pipeline -- pinned
    here, since a reorder would silently double-count today's score."""
    src = open("src/pipeline.py", encoding="utf-8").read()
    assert src.index("monitor_open_positions(", src.index("def run_pipeline")) < \
        src.rindex("_persist_run(")


# -- the standing invariant behind the pool (user directive, 2026-08-23) ------

def test_held_tickers_are_always_pinned_into_the_scored_universe():
    """EVERY exit surface assumes a held ticker is scored every tick: the
    aggregator exit score, the per-method re-scores, ml_exit's features and the
    held_rank pool all read this tick's TickerSignal for the held name. The
    pipeline guarantees it in two places, both asserted here at the source --
    a regression is invisible (exit methods silently abstain, pools thin out)
    until a position is mismanaged.

    Empirical state when pinned: 42/42 open positions at 100% per-run scoring
    coverage since entry."""
    src = open("src/pipeline.py", encoding="utf-8").read()
    # 1. open-trade tickers are unioned into the universe every tick
    assert '_mark_source(new_from_trades, "open_position_pin")' in src, (
        "open positions are no longer pinned into the universe")
    # 2. ...and exempt from the discovery liquidity gate, so a held name that
    #    fell below the floors keeps being scored (it just can't open NEW trades)
    prot = src[src.index("_protected = {"):src.index("_discovered =")]
    assert "open_trade_tickers" in prot, (
        "open positions are no longer protected from the discovery gate")
    # 3. the pin happens BEFORE the gate consumes the protected set
    assert src.index('open_position_pin') < src.index("_protected = {")
