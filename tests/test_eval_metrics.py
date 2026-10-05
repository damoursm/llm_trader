"""The house model metrics (`src/analysis/eval_metrics.py`): IC to the next H/L
pivot with its day-clustered t, top/bottom 5%, 3% and 1% returns, on two disjoint
test sets — and the selection objective the next models train on (2026-09-25)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import eval_metrics as em


def _panel(days=12, n=100, noise=0.5, start="2026-07-01", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.date_range(start, periods=days, freq="D")):
        y = rng.normal(size=n)
        s = y + noise * rng.normal(size=n)
        for j in range(n):
            rows.append({"signal_date": d.strftime("%Y-%m-%d"), "ticker": f"T{j:03d}",
                         "score": s[j], "fwd_ret_pivot": y[j]})
    return pd.DataFrame(rows)


def test_a_score_that_reads_the_label_scores_high_ic_and_ordered_tails():
    out = em.model_metrics(_panel(), "score")
    assert out["days"] == 12 and out["rows"] == 1200
    assert out["ic"]["mean"] > 0.7 and out["ic"]["t"] > 10
    top5, bot5 = out["top_5%"]["mean"], out["bottom_5%"]["mean"]
    top3, bot3 = out["top_3%"]["mean"], out["bottom_3%"]["mean"]
    top1, bot1 = out["top_1%"]["mean"], out["bottom_1%"]["mean"]
    assert top1 > top3 > top5 > out["universe_mean"]["mean"] > bot5 > bot3 > bot1
    assert out["top_5%"]["names_per_day"] == 5 and out["top_3%"]["names_per_day"] == 3
    assert out["top_1%"]["names_per_day"] == 1 and out["bottom_1%"]["names_per_day"] == 1
    assert not math.isnan(out["ic"]["half1"]) and not math.isnan(out["ic"]["half2"])


def test_tail_size_is_ceil_of_the_share_and_thin_days_do_not_count():
    df = _panel(days=3, n=41)
    out = em.model_metrics(df, "score")
    assert out["top_5%"]["names_per_day"] == 3 and out["top_3%"]["names_per_day"] == 2  # ceil(2.05), ceil(1.23)
    assert out["top_1%"]["names_per_day"] == 1                                            # ceil(0.41)
    thin = _panel(days=2, n=10, start="2026-07-10")
    assert em.model_metrics(thin, "score")["days"] == 0          # < MIN_DAY_ROWS
    both = pd.concat([df, thin])
    assert em.model_metrics(both, "score")["days"] == 3


def test_rows_without_a_score_or_label_are_dropped_and_ties_break_on_the_ticker():
    df = _panel(days=3, n=30)
    df.loc[df.index[:5], "fwd_ret_pivot"] = np.nan
    df.loc[df.index[5:8], "score"] = np.nan
    assert em.model_metrics(df, "score")["rows"] == 90 - 8
    tie = pd.DataFrame({"signal_date": ["2026-07-01"] * 25, "ticker": [f"Z{i:02d}" for i in range(25)],
                        "score": [1.0] * 5 + list(range(20)), "fwd_ret_pivot": list(range(25))})
    tie.loc[:4, "score"] = 100.0                                  # five tied at the top
    out = em.model_metrics(tie, "score", min_rows=20)
    assert out["top_5%"]["mean"] == 0.5                           # ceil(1.25)=2 -> Z00, Z01 (labels 0, 1)


def test_the_two_test_sets_are_disjoint_at_sept_27_28():
    dates = ["2026-06-16", "2026-06-17", "2026-09-27", "2026-09-28", "2026-10-15"]
    df = pd.DataFrame({"signal_date": dates, "x": range(5)})
    parts = em.split_test_sets(df)
    assert list(parts["set1_history"]["signal_date"]) == ["2026-06-17", "2026-09-27"]
    assert list(parts["set2_live_all_source"]["signal_date"]) == ["2026-09-28", "2026-10-15"]


def test_by_test_set_reports_each_set_separately():
    df = pd.concat([_panel(days=5, start="2026-09-20"), _panel(days=5, start="2026-09-28", seed=1)])
    out = em.by_test_set(df, "score")
    assert out["set1_history"]["days"] == 5 and out["set1_history"]["last_day"] == "2026-09-24"
    assert out["set2_live_all_source"]["days"] == 5
    assert out["set2_live_all_source"]["first_day"] == "2026-09-28"


# ── the selection objective (user directive 2026-09-25) ─────────────────────

def test_return_per_day_divides_by_sessions_and_floors_at_one_day():
    r = em.return_per_day([2.0, 2.0, 2.0, -3.0, 2.0], [1, 13, 26, 39, np.nan])
    assert r[0] == 2.0            # one bar away: floored at a day, not 13x
    assert r[1] == 2.0 and r[2] == 1.0 and r[3] == -1.0
    assert math.isnan(r[4])


def test_own_history_reads_prior_days_only_and_counts_every_run():
    days = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"]
    rows = [{"signal_date": d, "ticker": "A", "s": float(i + 1)} for i, d in enumerate(days)]
    rows.append({"signal_date": "2026-07-03", "ticker": "A", "s": 99.0})   # a later run the same day
    rows += [{"signal_date": d, "ticker": "B", "s": 0.0} for d in days]
    df = pd.DataFrame(rows)
    st = em.own_history_standing(df, "s", window_days=2)
    a = st[df["ticker"] == "A"].reset_index(drop=True)
    assert a.loc[0, "n_prior"] == 0 and math.isnan(a.loc[0, "prior_max"])
    assert (a.loc[2, "n_prior"], a.loc[2, "prior_max"], a.loc[2, "prior_min"]) == (2, 2.0, 1.0)
    assert a.loc[4, "prior_max"] == 2.0            # the same day's earlier/later runs never count
    assert (a.loc[3, "n_prior"], a.loc[3, "prior_max"]) == (3, 99.0)    # last 2 dates: 07-02, 07-03 (2 runs)


def _runs(scores_by_run, labels=None, bars=None):
    """{run_id: {ticker: score}} -> rows; signal_date = the run id's date part."""
    rows = []
    for rid, sc in scores_by_run.items():
        for tk, s in sc.items():
            rows.append({"run_id": rid, "signal_date": rid[:10], "ticker": tk, "score": s,
                         "fwd_ret_pivot": (labels or {}).get((rid, tk), 1.0),
                         "bars_ahead": (bars or {}).get((rid, tk), 13)})
    return pd.DataFrame(rows)


def test_selection_takes_the_top_or_bottom_one_and_breaks_ties_on_the_ticker():
    sc = {"2026-07-01_140000": {"A": 0.9, "B": 0.9, "C": 0.1, "D": -0.5},
          "2026-07-02_140000": {"A": 0.2, "B": 0.1, "C": 0.8, "D": -0.9}}
    df = _runs(sc, labels={("2026-07-01_140000", "A"): 2.0, ("2026-07-02_140000", "C"): 4.0,
                           ("2026-07-01_140000", "D"): -1.0, ("2026-07-02_140000", "D"): -3.0},
               bars={("2026-07-02_140000", "C"): 26})
    kw = dict(min_run_rows=3, own_history=False)
    lng = em.selection_objective(df, "score", "long", **kw)
    assert lng["entries"] == 2                                   # A (tie with B, A first), then C
    assert lng["objective"]["mean"] == pytest.approx((2.0 + 4.0 / 2) / 2)
    sht = em.selection_objective(df, "score", "short", **kw)
    assert sht["objective"]["mean"] == pytest.approx((1.0 + 3.0) / 2)   # a fall is a short's gain


def test_the_own_history_rule_drops_a_stale_top_pick_and_keeps_a_novice():
    days = [f"2026-07-{d:02d}" for d in (1, 2, 6, 7)]
    sc = {}
    for d in days[:3]:
        sc[f"{d}_140000"] = {"OLD": 0.5, "X": 0.0, "Y": -0.1}
    sc[f"{days[3]}_140000"] = {"OLD": 0.5, "X": 0.0, "Y": -0.1}        # OLD tops again, NOT a new high
    sc[f"{days[3]}_150000"] = {"OLD": 0.4, "NEW": 0.6, "Y": -0.1}      # NEW has no standing
    df = _runs(sc)
    out = em.selection_objective(df, "score", "long", min_run_rows=3, own_min_history=2)
    e = em._selection(df, "score", "long", min_run_rows=3, own_min_history=2)[0]
    taken = e[e["_entry"]]
    assert list(zip(taken["_d"], taken["ticker"])) == [("2026-07-01", "OLD"), ("2026-07-02", "OLD"),
                                                       ("2026-07-07", "NEW")]
    assert out["entries"] == 3                                   # 07-06: OLD has 2 priors, 0.5 is not > 0.5


def test_one_entry_per_name_per_day_at_its_first_run():
    sc = {"2026-07-01_140000": {"A": 0.9, "B": 0.1, "C": 0.0},
          "2026-07-01_143000": {"A": 0.95, "B": 0.1, "C": 0.0}}
    df = _runs(sc, labels={("2026-07-01_140000", "A"): 1.0, ("2026-07-01_143000", "A"): 5.0})
    out = em.selection_objective(df, "score", "long", min_run_rows=3, own_history=False)
    assert out["entries"] == 1 and out["objective"]["mean"] == pytest.approx(1.0)


def test_a_top_pick_without_a_label_is_not_replaced_by_the_next_name():
    df = _runs({"2026-07-01_140000": {"A": 0.9, "B": 0.5, "C": 0.0}},
               labels={("2026-07-01_140000", "A"): np.nan, ("2026-07-01_140000", "B"): 9.0})
    out = em.selection_objective(df, "score", "long", min_run_rows=3, own_history=False)
    assert out["entries"] == 1 and out["entries_labeled"] == 0
    assert math.isnan(out["objective"]["mean"])                  # never B's +9


def test_selection_by_test_set_carries_set1_history_into_set2():
    sc = {f"2026-09-{d}_140000": {"A": 0.9, "B": 0.1, "C": 0.0} for d in (23, 24, 25, 26, 27)}
    sc["2026-09-28_140000"] = {"A": 0.8, "B": 0.1, "C": 0.0}          # top, but below its set-1 highs
    df = _runs(sc)
    out = em.selection_by_test_set(df, "score", "long", min_run_rows=3, own_min_history=2)
    assert out["set2_live_all_source"]["days"] == 1
    assert out["set2_live_all_source"]["entries"] == 0          # set-1 history makes 0.8 stale
    assert out["set1_history"]["entries"] == 2                  # 09-23/24 as a novice; then no new high


def test_objective_defaults_match_the_live_freshness_rule():
    from config.settings import Settings
    f = Settings.model_fields
    assert em.OWN_WINDOW_DAYS == f["rank_entry_fresh_window_days"].default
    assert em.OWN_MIN_HISTORY == f["rank_entry_fresh_min_history"].default


def test_selection_entries_carry_the_frames_other_returns():
    df = _runs({"2026-07-01_140000": {"A": 0.9, "B": 0.5, "C": 0.0}})
    df["f1d"] = [0.7, -0.2, 0.1]
    e = em.selection_entries(df, "score", "long", min_run_rows=3, own_history=False)
    assert list(e["ticker"]) == ["A"] and e["f1d"].iloc[0] == pytest.approx(0.7)


def test_selection_entries_keep_the_frames_row_labels():
    """A caller joins entries back to the frame (the same-run control); the
    labels must be the frame's own, never a renumbering."""
    df = _runs({"2026-07-01_140000": {"A": 0.1, "B": 0.5, "C": 0.9},
                "2026-07-02_140000": {"A": 0.8, "B": 0.1, "C": 0.2}})
    df.index = [100, 101, 102, 103, 104, 105]
    e = em.selection_entries(df, "score", "long", min_run_rows=3, own_history=False)
    assert list(e.index) == [102, 103]
    assert (df.loc[e.index, "ticker"] == e["ticker"]).all()
