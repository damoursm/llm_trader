"""Per-discovery-source performance: funnel share + forward-return + realized trades.

Guards the evidence behind an adaptive discovery budget — that each source's
forward-return stats (mean move, up-share, combined_score IC) and its realized
trade outcomes are computed correctly and grouped by the ``universe_source``
stamp, with the missing-stamp rows kept visible rather than folded into a source.
"""

import pandas as pd
import pytest

import src.analysis.source_performance as spf


def _panel(rows, horizon=1, day="2026-01-02"):
    """Build a joined-panel DataFrame (one row per scored ticker) from
    ``(source, combined_score, fwd_ret)`` tuples."""
    return pd.DataFrame([
        {"signal_date": day, "universe_source": src, "combined_score": sc,
         f"fwd_ret_{horizon}d": fw}
        for src, sc, fw in rows
    ])


def test_source_label_normalises_missing():
    assert spf._source_label("smart_money") == "smart_money"
    assert spf._source_label("  trending  ") == "trending"
    assert spf._source_label(None) == spf._UNSTAMPED
    assert spf._source_label(float("nan")) == spf._UNSTAMPED
    assert spf._source_label("") == spf._UNSTAMPED
    assert spf._source_label("nan") == spf._UNSTAMPED
    assert spf._source_label("None") == spf._UNSTAMPED


def test_funnel_share_and_forward_return():
    panel = _panel([
        ("smart_money", 0.5, -5.0),
        ("smart_money", 0.3, -2.0),
        ("smart_money", 0.2, 1.0),
        ("trending", 0.8, 10.0),
        ("trending", 0.4, 4.0),
    ])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=2)
    # Biggest funnel contributor first.
    assert list(out["source"]) == ["smart_money", "trending"]
    sm = out[out["source"] == "smart_money"].iloc[0]
    tr = out[out["source"] == "trending"].iloc[0]

    assert sm["rows"] == 3 and sm["funnel_pct"] == 60.0
    assert tr["rows"] == 2 and tr["funnel_pct"] == 40.0
    # Mean RAW forward return — discovery quality, direction-agnostic.
    assert sm["fwd_1d"] == pytest.approx(-2.0)
    assert tr["fwd_1d"] == pytest.approx(7.0)
    # Up-share (share of moved names that rose).
    assert sm["win_1d"] == pytest.approx(100.0 / 3, abs=0.1)   # 1 of 3
    assert tr["win_1d"] == pytest.approx(100.0)


def test_ic_detects_backwards_predicting_source():
    # smart_money: higher combined_score → LOWER forward return ⇒ IC = -1 (the
    # 'they trade but predict backwards' case). trending: aligned ⇒ IC = +1.
    panel = _panel([
        ("smart_money", 0.5, -5.0),
        ("smart_money", 0.3, -2.0),
        ("smart_money", 0.2, 1.0),
        ("trending", 0.8, 10.0),
        ("trending", 0.4, 4.0),
    ])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=2)
    assert out[out["source"] == "smart_money"].iloc[0]["ic_1d"] == pytest.approx(-1.0)
    assert out[out["source"] == "trending"].iloc[0]["ic_1d"] == pytest.approx(1.0)


def test_simret_signs_by_combined_score():
    # A short call (score < 0) on a name that fell is a WIN → positive simret.
    panel = _panel([
        ("catalyst", 0.8, 10.0),    # long, rose  → +10
        ("catalyst", -0.5, -6.0),   # short, fell → +6
    ])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=2)
    row = out.iloc[0]
    assert row["simret_1d"] == pytest.approx(8.0)   # mean(+10, +6)


def test_unstamped_rows_kept_separate():
    panel = _panel([
        ("trending", 0.5, 3.0),
        ("smart_money", 0.4, 2.0),
        (None, 0.4, 2.0),
        (float("nan"), 0.3, 1.0),
    ])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=1)
    assert spf._UNSTAMPED in set(out["source"])
    unstamped = out[out["source"] == spf._UNSTAMPED].iloc[0]
    assert unstamped["rows"] == 2                 # both the None and the NaN stamp
    # Pre-stamp history is NOT part of the funnel: no funnel share, sorted last.
    # (a mixed None/float column becomes NaN under DataFrame construction).
    assert pd.isna(unstamped["funnel_pct"])
    assert out.iloc[-1]["source"] == spf._UNSTAMPED
    # Funnel share is over STAMPED rows only (2), not the grand total (4).
    assert out[out["source"] == "trending"].iloc[0]["funnel_pct"] == 50.0
    assert out[out["source"] == "smart_money"].iloc[0]["funnel_pct"] == 50.0


def test_min_n_gates_stats_but_reports_count():
    panel = _panel([("trending", 0.5, 3.0), ("trending", 0.4, 2.0)])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=10)
    row = out.iloc[0]
    assert row["n_1d"] == 2                        # count still reported
    assert pd.isna(row["fwd_1d"]) and pd.isna(row["ic_1d"])   # stats gated off


def test_missing_forward_return_not_counted():
    panel = pd.DataFrame([
        {"signal_date": "2026-01-02", "universe_source": "trending",
         "combined_score": 0.5, "fwd_ret_1d": 3.0},
        {"signal_date": "2026-01-02", "universe_source": "trending",
         "combined_score": 0.4, "fwd_ret_1d": None},   # cache didn't reach the forward bar
    ])
    out = spf.compute_source_performance(panel, horizons=(1,), min_n=1)
    assert out.iloc[0]["n_1d"] == 1                 # only the row with a forward return


def test_empty_or_unstamped_panel_returns_empty():
    assert spf.compute_source_performance(pd.DataFrame(), horizons=(1,)).empty
    # A panel with no universe_source column at all → empty (nothing to group).
    no_col = pd.DataFrame([{"signal_date": "2026-01-02", "combined_score": 0.5,
                            "fwd_ret_1d": 3.0}])
    assert spf.compute_source_performance(no_col, horizons=(1,)).empty


# ── realized-trade view ──────────────────────────────────────────────────────

def test_source_trade_perf_groups_and_scores():
    trades = [
        {"universe_source": "smart_money", "return_pct": -3.0},
        {"universe_source": "smart_money", "return_pct": 2.0},
        {"universe_source": "trending", "return_pct": 5.0},
        {"universe_source": "trending", "return_pct": None},   # no return → skipped
        {"universe_source": None, "return_pct": 1.0},
    ]
    out = spf.compute_source_trade_perf(trades)
    by = {r["source"]: r for r in out}

    assert out[0]["source"] == "smart_money"          # most trades first
    assert by["smart_money"]["trades"] == 2
    assert by["smart_money"]["win_rate"] == 50.0
    assert by["smart_money"]["avg_return"] == pytest.approx(-0.5)
    assert by["smart_money"]["worst"] == -3.0 and by["smart_money"]["best"] == 2.0

    assert by["trending"]["trades"] == 1              # the None-return row dropped
    assert by["trending"]["win_rate"] == 100.0
    assert spf._UNSTAMPED in by                       # None stamp kept visible


def test_source_trade_perf_empty():
    assert spf.compute_source_trade_perf([]) == []
    assert spf.compute_source_trade_perf([{"universe_source": "x"}]) == []   # no return_pct
