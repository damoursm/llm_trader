"""Offline EXIT policy evaluation — counterfactual close-rule value.

Verifies the close-vs-hold replay: reward = oriented fwd if hold else 0, the
exit_alpha (held_mean − allhold_mean) that measures whether a close rule keeps
the better positions, avg_fwd_on_close (negative = cutting losers), the
exit-breadth and always-hold policies, dedup-to-last-per-day, and determinism.
"""

from datetime import date

import pandas as pd
import pytest

import src.analysis.exit_policy_eval as xpe


def _mk_panel(rows):
    """rows: (position_id, signal_date, ticker, entry_direction, fwd, scores_dict)."""
    out = []
    for pid, sd, tk, dirn, fwd, scores in rows:
        d = {"position_id": pid, "signal_date": sd, "ticker": tk,
             "entry_direction": dirn, "fwd": fwd}
        d.update(scores)
        out.append(d)
    return pd.DataFrame(out)


# ── reward semantics + exit_alpha ─────────────────────────────────────────────

def test_hold_captures_fwd_close_captures_zero():
    # One winner (+4) one loser (−4). A policy that closes the loser and holds
    # the winner keeps only +4 → held_mean +4 vs allhold_mean 0 → alpha +4.
    panel = _mk_panel([
        ("p1", "2026-06-01", "WIN", "BULLISH", +4.0, {"aggregator": +0.5}),
        ("p2", "2026-06-01", "LOS", "BULLISH", -4.0, {"aggregator": -0.5}),
    ])
    r = xpe.evaluate_exit_policy(panel, xpe.exit_aggregator)   # close when aggregator < 0
    assert r["n_decisions"] == 2
    assert r["close_rate"] == pytest.approx(50.0)
    assert r["avg_fwd_on_close"] == pytest.approx(-4.0)        # it cut the loser
    assert r["avg_fwd_on_hold"] == pytest.approx(+4.0)
    assert r["allhold_mean"] == pytest.approx(0.0)
    assert r["exit_alpha"] == pytest.approx(+4.0)             # earns its keep


def test_always_hold_is_zero_alpha_baseline():
    panel = _mk_panel([
        ("p1", "2026-06-01", "A", "BULLISH", +3.0, {"aggregator": -0.9}),
        ("p2", "2026-06-02", "B", "BULLISH", -5.0, {"aggregator": -0.9}),
    ])
    r = xpe.evaluate_exit_policy(panel, xpe.exit_always_hold)
    assert r["close_rate"] == pytest.approx(0.0)
    assert r["avg_fwd_on_close"] is None
    assert r["exit_alpha"] == pytest.approx(0.0)              # holding everything = the baseline
    assert r["avg_fwd_on_hold"] == pytest.approx(r["allhold_mean"])


def test_a_bad_exit_that_cuts_winners_has_negative_alpha():
    # Policy closes the WINNER (aggregator says exit on the good one) → keeps the
    # loser → held_mean below allhold_mean → negative alpha.
    panel = _mk_panel([
        ("p1", "2026-06-01", "WIN", "BULLISH", +6.0, {"aggregator": -0.5}),   # closed
        ("p2", "2026-06-01", "LOS", "BULLISH", -2.0, {"aggregator": +0.5}),   # held
    ])
    r = xpe.evaluate_exit_policy(panel, xpe.exit_aggregator)
    assert r["avg_fwd_on_close"] == pytest.approx(+6.0)       # cut a winner (bad)
    assert r["exit_alpha"] == pytest.approx(-2.0 - 2.0)       # held −2 vs allhold +2


# ── short orientation + exit-breadth ─────────────────────────────────────────

def test_short_position_orientation():
    # A SHORT held while the stock ROSE +5 → oriented fwd = −5 (bad to hold).
    # scores are already position-oriented; a majority-exit breadth vote closes it.
    panel = _mk_panel([
        ("p1", "2026-06-01", "SH", "BEARISH", -5.0,
         {"news": -0.5, "momentum": -0.5, "tech": -0.5, "vwap": +0.2}),   # 3/4 exit
    ])
    r = xpe.evaluate_exit_policy(panel, xpe.make_exit_breadth(0.5))
    assert r["close_rate"] == pytest.approx(100.0)            # 75% > 50% → close
    assert r["avg_fwd_on_close"] == pytest.approx(-5.0)       # correctly cut the losing short


def test_exit_breadth_threshold_gates():
    # 2 of 4 signal methods vote exit = 50% — not > 0.5 → hold; > 0.4 → close.
    scores = {"news": -0.5, "momentum": -0.5, "tech": +0.5, "vwap": +0.5}
    panel = _mk_panel([("p1", "2026-06-01", "X", "BULLISH", 1.0, scores)])
    assert xpe.evaluate_exit_policy(panel, xpe.make_exit_breadth(0.50))["close_rate"] == 0.0
    assert xpe.evaluate_exit_policy(panel, xpe.make_exit_breadth(0.40))["close_rate"] == 100.0
    # Decision-layer/excursion methods are excluded from the breadth vote.
    assert xpe._exit_breadth_frac({"aggregator": -1, "mfe": -1, "news": +0.5}) == pytest.approx(0.0)


# ── panel build: dedup to last-per-day + forward return + drop missing ───────

def test_build_panel_dedups_last_review_and_orients(monkeypatch):
    import src.analysis.exit_policy_eval as m
    # Two ticks same position-day; the LAST (14:30) must win. Forward +10%.
    exit_rows = pd.DataFrame([
        {"reviewed_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "AAA", "position_id": "p", "entry_direction": "BULLISH",
         "method": "aggregator", "score": +0.9},
        {"reviewed_at": "2026-06-01T14:30:00+00:00", "signal_date": "2026-06-01",
         "ticker": "AAA", "position_id": "p", "entry_direction": "BULLISH",
         "method": "aggregator", "score": -0.9},                       # latest
    ])
    monkeypatch.setattr("src.analysis.exit_panel._load_exit_signals", lambda days=None: exit_rows)
    d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
    monkeypatch.setattr("src.analysis.simulated_trades._daily_series",
                        lambda tk: ([d0, d1], {d0: 100.0, d1: 110.0}))
    panel = m.build_exit_decision_panel(horizon=1)
    assert len(panel) == 1
    assert panel.iloc[0]["aggregator"] == pytest.approx(-0.9)          # last tick kept
    assert panel.iloc[0]["fwd"] == pytest.approx(10.0)                 # +10% oriented long


def test_build_panel_drops_rows_without_forward_return(monkeypatch):
    import src.analysis.exit_policy_eval as m
    exit_rows = pd.DataFrame([
        {"reviewed_at": "2026-06-01T14:00:00+00:00", "signal_date": "2026-06-01",
         "ticker": "NOFWD", "position_id": "p", "entry_direction": "BULLISH",
         "method": "aggregator", "score": 0.5},
    ])
    monkeypatch.setattr("src.analysis.exit_panel._load_exit_signals", lambda days=None: exit_rows)
    monkeypatch.setattr("src.analysis.simulated_trades._daily_series", lambda tk: ([], {}))
    assert m.build_exit_decision_panel(horizon=5).empty


# ── compare + determinism + empty ────────────────────────────────────────────

def test_compare_ranks_by_alpha_and_is_deterministic():
    panel = _mk_panel([
        ("p1", "2026-06-01", "A", "BULLISH", -3.0, {"aggregator": -0.8, "news": -0.5, "tech": -0.5}),
        ("p2", "2026-06-02", "B", "BULLISH", +2.0, {"aggregator": +0.8, "news": +0.5, "tech": +0.5}),
    ])
    out = {n: xpe.evaluate_exit_policy(panel, p) for n, p in xpe.DEFAULT_EXIT_POLICIES.items()}
    # aggregator<0 closes the −3 loser, holds the +2 winner → positive alpha.
    assert out["aggregator < 0"]["exit_alpha"] > 0
    assert out["always hold (baseline)"]["exit_alpha"] == pytest.approx(0.0)
    again = xpe.evaluate_exit_policy(panel, xpe.exit_aggregator)
    assert again == out["aggregator < 0"]                              # deterministic


def test_empty_is_safe():
    assert xpe.evaluate_exit_policy(pd.DataFrame(), xpe.exit_always_hold) == {"n_decisions": 0, "n_days": 0}
    assert xpe.build_exit_decision_panel(exit_df=pd.DataFrame()).empty
    assert xpe.compare_exit_policies(exit_df=pd.DataFrame()).empty
