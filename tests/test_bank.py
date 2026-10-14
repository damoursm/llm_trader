"""The bank of models on the pooled history (src/backtest/bank.py; user directive 2026-10-14: "check for the full
dataset (A0+A+B) ... enough trades made so that the results are significant, reliable, and transferable")."""
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import bank as BK
from src.backtest import periods


def _trades(n, mean, sd=2.0, start="2018-01-10", end="2026-09-01", book="x", seed=0, days=3.0):
    rng = np.random.default_rng(seed)
    ent = pd.to_datetime(np.sort(rng.integers(pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9,
                                               size=n)), unit="s").normalize()
    net = rng.normal(mean, sd, size=n)
    t = pd.DataFrame({"book": book, "ticker": "T", "entry_day": ent, "net": net, "gross": net, "days": days,
                      "decision": [d.date() for d in ent]})
    t["year"] = pd.DatetimeIndex(t.entry_day).year
    t["set"] = [periods.set_of(d, extra=True) for d in t.decision]
    t["month"] = pd.DatetimeIndex(t.entry_day).to_period("M").astype(str)
    return t


def test_the_pool_is_a0_a_b_weighted_by_their_years():
    w = periods.pool_weights()
    assert periods.POOLED == ("A0", "A", "B") and abs(sum(w.values()) - 1.0) < 1e-12
    assert periods.pool_span() == ("2018-01-02", "2026-09-24")
    assert w["A"] > w["A0"] > w["B"] and abs(w["B"] - periods.years("B") / sum(map(periods.years, "A0 A B".split()))) < 1e-12


def test_a_strong_arm_qualifies_a_thin_one_is_insufficient_a_noisy_one_fails():
    strong = _trades(200, 1.5, 3.0)
    c = BK.card(strong, "x", slice_=0.1)
    q = BK.qualify(c)
    assert q["verdict"] == "QUALIFIES", q
    assert c["overall"]["rpd_lo95"] > 0 and c["overall"]["months"] >= BK.MIN_EPISODES
    thin = _trades(60, 1.5, 3.0)
    q = BK.qualify(BK.card(thin, "x", slice_=0.1))
    assert q["verdict"] == "INSUFFICIENT" and q["checks"]["significant"], q
    noisy = _trades(300, 0.05, 6.0, seed=3)
    q = BK.qualify(BK.card(noisy, "x", slice_=0.1))
    assert q["verdict"] == "FAILS" and q["checks"]["significant"] is False, q


def test_one_year_carrying_the_edge_fails_and_so_does_an_edge_in_one_half():
    a = _trades(40, 12.0, 2.0, start="2019-01-05", end="2019-12-20", seed=1)
    b = _trades(160, -0.6, 1.0, start="2020-01-05", end="2026-09-01", seed=2)
    t = pd.concat([a, b], ignore_index=True).sort_values("entry_day").reset_index(drop=True)
    c = BK.card(t, "x", slice_=0.1)
    q = BK.qualify(c)
    assert c["overall"]["avg_trade"] > 0 and q["checks"]["not_one_year"] is False and q["verdict"] == "FAILS"
    assert c["loyo"]["2019"]["avg_trade"] < 0
    early = _trades(100, -1.0, 1.0, end="2021-12-31", seed=4)
    late = _trades(100, 3.0, 1.0, start="2022-01-05", seed=5)
    q = BK.qualify(BK.card(pd.concat([early, late], ignore_index=True), "x", slice_=0.1))
    assert q["checks"]["halves"] is False and q["verdict"] == "FAILS"


def test_the_longest_history_supplies_the_count_only_when_the_pool_is_short():
    pooled = _trades(17, 5.0, 8.0, seed=6)
    long = {"by_year": {str(y): {"n": 4, "avg_trade": 2.5, "sd_trade": 6.0, "avg_days": 5.0} for y in range(1990, 2015)},
            "source": "test"}
    c = BK.card(pooled, "x", long=long, slice_=0.25)
    q = BK.qualify(c)
    assert q["basis"] == "longest" and c["longest"]["n"] == 17 + 100 and c["longest"]["episodes"] >= 25
    assert q["verdict"] == "QUALIFIES", q
    bad = {"by_year": {str(y): {"n": 4, "avg_trade": -2.0, "sd_trade": 6.0, "avg_days": 5.0} for y in range(1990, 2015)}}
    assert BK.qualify(BK.card(pooled, "x", long=bad, slice_=0.25))["verdict"] == "FAILS"
    many = _trades(200, 1.5, 3.0)                          # enough on the pool: the earlier years are not read
    assert BK.qualify(BK.card(many, "x", long=bad, slice_=0.1))["basis"] == "pooled"
    years_2018_on = {str(y): {"n": 4, "avg_trade": -50.0, "sd_trade": 1.0} for y in range(2018, 2021)}
    c2 = BK.card(pooled, "x", long={"by_year": {**long["by_year"], **years_2018_on}}, slice_=0.25)
    assert c2["longest"]["n"] == 117                       # LONG's years inside the pool are never counted twice


def test_a_filter_is_justified_by_enough_significantly_losing_removed_trades():
    good = _trades(150, 2.0, 2.0, seed=7)
    bad = _trades(45, -3.0, 2.0, seed=8)
    t = pd.concat([good, bad], ignore_index=True)
    removed = np.r_[np.zeros(len(good), bool), np.ones(len(bad), bool)]
    f = BK.filter_test(t, removed)
    assert f["justified_by_trades"] and f["rpd_change"] > 0
    few = BK.filter_test(t.iloc[:170], removed[:170])
    assert not few["justified_by_trades"] and not few["enough_removed"]
    neutral = _trades(45, 0.0, 4.0, seed=9)
    t2 = pd.concat([good, neutral], ignore_index=True)
    assert not BK.filter_test(t2, removed)["justified_by_trades"]


def test_the_walk_forward_decides_each_year_on_the_years_before_it_only():
    t = _trades(300, 1.0, 2.0, seed=10)
    seen = []

    def choose(past):
        seen.append(int(past.year.max()) if len(past) else None)
        cut = float(past.net.median()) if len(past) else 0.0

        def rule(df):
            return df.net.to_numpy() > cut - 100            # keep everything (a threshold far below)
        rule.label = f"cut {cut:.2f}"
        return rule
    wf = BK.walk_forward(t, choose, first_year=2020)
    years = sorted(int(y) for y in wf["chosen"])
    assert years[0] == 2020 and seen[:len(years)] == [y - 1 for y in years]
    assert wf["out_of_time"]["n"] == int((t.year >= 2020).sum()) and wf["decay"] is not None


def test_the_arm_alone_compounds_its_realized_pnl_at_its_slice():
    t = pd.DataFrame({"entry_day": pd.to_datetime(["2018-02-01", "2018-03-01"]), "days": [5.0, 5.0],
                      "net": [10.0, -10.0]})
    a = BK.alone(t, 0.5, "2018-01-02", "2019-01-02")
    assert abs(a["end_equity"] - 10_000 * 1.05 * 0.95) < 1e-6
    assert abs(a["maxdd"] - 5.0) < 1e-9


def _nav(lo, hi, rate):
    days = pd.bdate_range(lo, hi)
    yrs = (days - days[0]).days / 365.25
    return {"days": [d.date().isoformat() for d in days], "values": list((1.0 + rate) ** yrs)}


def test_the_combination_chains_the_sets_and_compounds_a_month_two_sets_share():
    runs = {}
    rates = {"A0": 0.10, "A": 0.30, "B": 0.60}
    for s in periods.POOLED:
        S = periods.get(s)
        runs[s] = {k: {"growth": rates[s], "maxdd": 10.0 + k + (5 if s == "B" else 0), "calls": ["x"] if (s, k) == ("B", 1) else [],
                       "nav": _nav(S.lo, S.hi, rates[s]), "pos": []} for k in range(3)}
    c = BK.combination(runs)
    w = periods.pool_weights()
    lg = sum(w[s] * 100 * math.log1p(rates[s]) for s in periods.POOLED)
    assert abs(c["growth"]["log_points"] - lg) < 1e-9 and c["seeds"] == 3
    assert c["maxdd"]["worst"] == 17.0 and c["margin_calls"]["draws_with_one"] == 1
    assert c["months"]["positive_share"] > 0.95 and c["months"]["longest_under_water_months"] == 0
    m = BK._chain([BK._months(BK.nav_of(runs[s][0], periods.get(s).lo)[0]) for s in periods.POOLED])
    jun = pd.Period("2024-06", "M")
    a_part = BK._months(BK.nav_of(runs["A"][0], periods.get("A").lo)[0])[jun]
    b_part = BK._months(BK.nav_of(runs["B"][0], periods.get("B").lo)[0])[jun]
    assert abs(m[jun] - ((1 + a_part) * (1 + b_part) - 1)) < 1e-12
    assert m.index.is_monotonic_increasing and m.index.is_unique


def test_a_draw_without_a_nav_reads_its_positions_realized_at_their_exits():
    ns = pd.Timestamp("2018-03-05 15:00", tz="America/New_York").value
    rec = {"pos": [{"strategy": "dip", "exit_ns": ns, "pnl": 500.0}]}
    s, basis = BK.nav_of(rec, "2018-01-02")
    assert basis == "realized" and abs(s.iloc[-1] - 1.05) < 1e-12


def test_the_month_bootstrap_is_seeded_and_brackets_the_estimate():
    t = _trades(150, 1.0, 2.0, seed=11)
    a, b = BK.month_boot(t, B=500), BK.month_boot(t, B=500)
    assert np.array_equal(a, b)
    q = BK.quality(t)
    assert q["rpd_lo95"] < q["return_per_day"] < q["rpd_hi95"]


@pytest.mark.parametrize("label,expect", [({"n": 40, "months": 12, "rpd_lo95": 0.1, "rpd_hi95": 0.5}, "EARNS"),
                                          ({"n": 40, "months": 12, "rpd_lo95": -0.5, "rpd_hi95": -0.1}, "LOSES"),
                                          ({"n": 40, "months": 12, "rpd_lo95": -0.1, "rpd_hi95": 0.3}, "unclear"),
                                          ({"n": 20, "months": 12, "rpd_lo95": 0.1, "rpd_hi95": 0.5}, None)])
def test_a_context_cell_is_read_only_from_enough_trades(label, expect):
    assert BK._cell_label(label) == expect


# ── context activation (PREREG108) ───────────────────────────────────────────

def _ctx_trades(rows):
    """rows: (decision day, cell, net %, hold days)."""
    t = pd.DataFrame(rows, columns=["day", "cell", "net", "days"])
    t["day"] = pd.to_datetime(t.day)
    t["exit"] = t.day + pd.to_timedelta(t.days, unit="D")
    t["dmonth"] = t.day.dt.to_period("M").astype(str)
    t["w"] = 1.0
    return t


def _history(n_months=30, bad_cell="below/high", bad=-3.0, good=2.0):
    rows = []
    for k, d in enumerate(pd.date_range("2018-01-05", periods=n_months, freq="MS") + pd.Timedelta(days=4)):
        for j in range(4):
            wig = 0.5 * ((j % 2) * 2 - 1) + 0.4 * np.sin(k + j)          # months differ: a finite clustered t
            rows.append((d + pd.Timedelta(days=j), "above/mid", good + wig, 3.0))
            rows.append((d + pd.Timedelta(days=j), bad_cell, bad + wig, 3.0))
    return _ctx_trades(rows)


def test_a_cell_that_lost_is_switched_off_by_every_rule_and_a_proven_one_stays_on():
    t = _history(bad=-1.0)                                  # the arm earns overall (+0.5 a trade), one cell loses
    asof = pd.Timestamp("2020-08-01")
    for rule in BK.ACTIVATION_RULES:
        mp = BK.activation_map(t, asof, rule)
        assert not mp["burn_in"] and "below/high" not in mp["active"], rule
        assert "above/mid" in mp["active"], rule
    # a cell never seen: K1 follows the arm's own average (positive here: on), K2 needs proof (off), K3 stays on
    assert "above/low" in BK.activation_map(t, asof, "shrunk")["active"]
    assert "above/low" not in BK.activation_map(t, asof, "proven")["active"]
    assert "above/low" in BK.activation_map(t, asof, "optout")["active"]


def test_the_map_reads_only_trades_closed_before_its_month_and_starts_after_a_year():
    t = _history(bad=1.0)                                               # every cell earns
    assert BK.activation_map(t, "2018-11-01")["burn_in"]                # < a year of closed history: all on
    late = _ctx_trades([(pd.Timestamp("2020-07-20") + pd.Timedelta(days=j), "above/high", -50.0, 40.0)
                        for j in range(60)])                             # decided before, closed after the asof
    both = pd.concat([t, late], ignore_index=True)
    mp = BK.activation_map(both, "2020-08-01", "shrunk")
    assert "above/high" in mp["active"] and mp["cells"]["above/high"]["n"] == 0
    assert "above/high" not in BK.activation_map(both, "2020-10-01", "shrunk")["active"]


def test_activated_keeps_a_trade_only_where_its_month_s_map_holds_its_cell():
    t = _history(n_months=36)
    sched = BK.schedule(t, "shrunk", months=sorted(set(t.dmonth)))
    keep = BK.activated(t, sched)
    early = t.day < pd.Timestamp("2019-01-10")
    assert keep[early.to_numpy()].all()                                 # the burn-in year takes everything
    late_bad = ((t.cell == "below/high") & (t.day > pd.Timestamp("2019-06-01"))).to_numpy()
    assert not keep[late_bad].any() and keep[(t.cell == "above/mid").to_numpy()].all()
