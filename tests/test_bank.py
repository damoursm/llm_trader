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


def test_the_arm_alone_skips_entries_past_its_open_cap():
    t = pd.DataFrame({"entry_day": pd.to_datetime(["2018-02-01", "2018-02-02", "2018-02-03"]),
                      "days": [10.0, 10.0, 10.0], "net": [10.0, 10.0, 10.0]})
    full = BK.alone(t, 0.5, "2018-01-02", "2019-01-02")
    capped = BK.alone(t, 0.5, "2018-01-02", "2019-01-02", max_open=1)
    assert full["skipped"] == 0 and capped["skipped"] == 2
    assert abs(capped["end_equity"] - 10_000 * 1.05) < 1e-6
    assert full["end_equity"] > capped["end_equity"]


# ── the bar and the re-judging (user directive 2026-10-15: "Adopt the re-judging rule and test long arms against the
# market's average daily rise. The idea is to beat the market but I'm also ok with using 3x for trades we are
# confident the market will go up (or down).") ──

def _flat_bench(daily_pct, lo="1999-01-01", hi="2026-12-31"):
    """A benchmark rising ``daily_pct`` every weekday."""
    days = pd.bdate_range(lo, hi)
    return pd.Series(daily_pct, index=days, name=f"flat{daily_pct}")


def _monthly(months, per_month, values, seed=0, days=3.0, sd=0.0):
    """Trades on the first days of ``months`` ("YYYY-MM"), ``per_month`` each, month k's trades around ``values[k]``."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, m in enumerate(months):
        first = pd.Period(m, "M").to_timestamp()
        for j in range(per_month):
            v = float(values[k % len(values)]) + (rng.normal(0.0, sd) if sd else 0.0)
            rows.append({"entry_day": first + pd.Timedelta(days=j), "net": v, "gross": v, "days": days})
    t = pd.DataFrame(rows)
    t["book"], t["ticker"] = "x", "T"
    t["decision"] = [d.date() for d in t.entry_day]
    t["year"] = pd.DatetimeIndex(t.entry_day).year
    t["set"] = [periods.set_of(d, extra=True) for d in t.decision]
    t["month"] = pd.DatetimeIndex(t.entry_day).to_period("M").astype(str)
    return t


def _pool_months(per_year=4):
    """``per_year`` months in each pooled year (2018 .. 2026)."""
    return [f"{y}-{m:02d}" for y in range(2018, 2027) for m in (2, 4, 6, 8)[:per_year]]


def test_the_bar_is_the_benchmark_s_average_day_over_the_days_a_session_spans():
    b = _flat_bench(0.1)
    r = BK.bench_rpd(b, "2018-01-02", "2026-09-24")
    gap = 7.0 / 5.0                                                     # weekdays: a session spans 1.4 days
    assert abs(r - BK.per_day(0.1, gap)) < 2e-4, r
    up = pd.Series(np.where(b.index.year == 2020, 1.0, 0.1), index=b.index)
    assert BK.bench_rpd(up, "2018-01-02", "2026-09-24") > r
    assert abs(BK.bench_rpd(up, "2018-01-02", "2026-09-24", exclude_years=[2020]) - r) < 2e-4


def test_a_long_arm_must_beat_its_bar_a_short_arm_only_zero():
    t = _trades(400, 0.6, 1.0, seed=12)                                 # ~ +0.2 %/day, held 3 days
    alone = BK.qualify(BK.card(t, "x", slice_=0.1))
    assert alone["verdict"] == "QUALIFIES" and alone["bar"] == 0.0 and alone["bar_spec"] is None, alone
    c = BK.card(t, "x", slice_=0.1, bench=_flat_bench(0.5))            # the market +0.5% a weekday: ~0.36 %/day
    q = BK.qualify(c)
    assert c["bench"]["rpd"] > c["overall"]["rpd_hi95"]
    assert q["verdict"] == "FAILS" and not q["checks"]["significant"] and not q["checks"]["halves"], q
    assert "average day" in q["why"][0]
    low = BK.qualify(BK.card(t, "x", slice_=0.1, bench=_flat_bench(0.05)))
    assert low["verdict"] == "QUALIFIES" and 0.0 < low["bar"] < 0.05, low


def test_every_long_live_arm_has_a_bar_and_no_short_arm_has_one():
    from src.backtest import reliability as REL
    longs = {a for a, s in REL.SIDE.items() if s == "long"}
    assert longs and longs == set(BK.BENCH), (longs, BK.BENCH)
    # PREREG123 (user 2026-10-16): the market's average day, whatever the arm holds — the crash book's 3x fund too
    assert set(BK.BENCH.values()) == {"SPY"}, BK.BENCH


def test_an_arm_failing_only_significance_is_re_judged_on_its_longest_history():
    pool = _monthly(_pool_months(), 3, [4.0, -3.6])                    # +0.2 a trade every year; months +-12
    c = BK.card(pool, "x", slice_=0.1)
    q = BK.qualify(c)
    assert q["verdict"] == "FAILS" and [k for k, v in q["checks"].items() if v is False] == ["significant"], q
    early = _monthly([f"{y}-{m:02d}" for y in range(2000, 2018) for m in range(1, 13)], 2, [1.0], sd=0.5, seed=13)
    c = BK.card(pool, "x", slice_=0.1, long_trades=early)
    q = BK.qualify(c)
    assert q["verdict"] == "QUALIFIES" and q["basis"] == "longest (re-judged)", q
    assert q["pooled_checks"]["significant"] is False and c["longest"]["n"] == len(pool) + len(early)
    assert c["longest"]["episodes"] == 36 + 216 and c["longest"]["years_before"][0] == "2000"
    bad_half = pd.concat([_monthly(_pool_months()[:18], 6, [-1.0]), _monthly(_pool_months()[18:], 6, [3.0])])
    q = BK.qualify(BK.card(bad_half, "x", slice_=0.1, long_trades=early))
    assert q["basis"] == "pooled" and q["checks"]["halves"] is False, q   # another check failed: no re-judging


def test_the_longest_history_s_significance_counts_each_month_once():
    pool = _monthly(_pool_months(), 3, [4.0, -3.6])
    months = [f"{y}-{m:02d}" for y in range(2000, 2018) for m in range(1, 13)]
    early = _monthly(months, 20, [5.0, -4.6])                            # +0.2 a trade; each month all one way
    x = np.r_[early.net.to_numpy(), pool.net.to_numpy()]
    assert x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) > 1.96       # trade by trade it would look significant
    c = BK.card(pool, "x", slice_=0.1, long_trades=early)
    q = BK.qualify(c)
    assert q["basis"] == "longest (re-judged)" and q["verdict"] == "FAILS" and not q["checks"]["significant"], q
    assert c["longest"]["rpd_lo95"] < 0


def test_yearly_summaries_are_judged_clustered_and_against_the_bar():
    pooled = _trades(17, 5.0, 8.0, seed=6)
    long = {"by_year": {str(y): {"n": 4, "avg_trade": 2.5 + (0.8 if y % 2 else -0.8), "sd_trade": 6.0,
                                 "avg_days": 5.0} for y in range(1990, 2015)}}
    c = BK.card(pooled, "x", long=long, slice_=0.25)
    assert BK.qualify(c)["verdict"] == "QUALIFIES" and c["longest"]["bar"] == 0.0
    assert c["longest"]["t"] <= c["longest"]["t_trades_vs_0"] + 1e-9   # the larger standard error decides
    hot = BK.card(pooled, "x", long=long, slice_=0.25, bench=_flat_bench(1.5, lo="1989-01-01"))
    q = BK.qualify(hot)
    assert hot["longest"]["bar_trade"] > hot["longest"]["mean"] and q["verdict"] == "FAILS", q
    assert not q["checks"]["significant"] and q["basis"] == "longest"


def test_the_benchmark_takes_the_carried_history_before_the_daily_cache(monkeypatch):
    late = pd.Series([0.2, 0.2], index=pd.to_datetime(["2010-02-12", "2010-02-16"]))
    carried = {"TQQQ": {"days": ["2009-12-31", "2010-02-11", "2010-02-12"], "ret": [1.0, 2.0, 9.0]}}
    BK.bench_daily.cache_clear()
    monkeypatch.setattr(BK, "_cache_returns", lambda sym: late)
    monkeypatch.setattr(BK, "_carried_bench", lambda: carried)
    try:
        s = BK.bench_daily("TQQQ")
        assert list(s.index.strftime("%Y-%m-%d")) == ["2009-12-31", "2010-02-11", "2010-02-12", "2010-02-16"]
        assert list(s.to_numpy()) == [1.0, 2.0, 0.2, 0.2]               # the cache wins where both hold a day
    finally:
        BK.bench_daily.cache_clear()


def test_load_runs_names_every_book_by_its_live_name(tmp_path):
    """PREREG165 / 167 built the capitulation book as "capb": the live configuration's draws (`LIVE_RUNS`) must
    read as the live arm "cap" everywhere they are loaded (2026-10-20: the cap trades read 0 under the other name)."""
    import pickle
    rec = {"growth": 0.5, "maxdd": 20.0, "pos": [{"strategy": "capb", "pnl": 1.0, "exit_ns": 1},
                                                  {"strategy": "dip", "pnl": 2.0, "exit_ns": 2}],
           "calls": [{"strategy": "capb", "ns": 3}], "label": "X"}
    for st in ("A0", "A", "B"):
        with open(tmp_path / f"{st}_X_s0.pkl", "wb") as fh:
            pickle.dump(rec, fh)
    runs = BK.load_runs(tmp_path, "X")
    assert set(runs) == {"A0", "A", "B"}
    assert [p["strategy"] for p in runs["B"][0]["pos"]] == ["cap", "dip"]
    assert runs["B"][0]["calls"] == [{"strategy": "cap", "ns": 3}]
    assert rec["pos"][0]["strategy"] == "capb"                                  # the record loaded is a copy
    assert BK.canonical_book("capb") == "cap" and BK.canonical_book("vol") == "vol"
    assert set(BK.ARMS) == {"vol", "etf", "fall", "dip", "crash", "cap"} and "cap" in BK.live_slices()

def test_a_live_arms_earlier_trades_are_read_as_its_longest_history(tmp_path):
    """`bank.LONG_TRADES` (the capitulation book: PREREG165's untouched 2006-2017 trades) feeds the re-judging; a
    missing file or an arm without a table reads None, never an error."""
    import pickle
    assert set(BK.LONG_TRADES) <= set(BK.ARMS)
    frame = pd.DataFrame({"entry_day": pd.to_datetime(["2008-02-07", "2009-03-10"]), "net": [9.1, -4.8],
                          "gross": [9.12, -4.78], "days": [3.7, 7.0], "tk": ["AAPL", "MSFT"]})
    path, key, _ = BK.LONG_TRADES["cap"]
    (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / path, "wb") as fh:
        pickle.dump({key: frame}, fh)
    t = BK.long_trades("cap", root=tmp_path)
    assert list(t.columns) == ["entry_day", "net", "gross", "days"] and len(t) == 2
    assert "PREREG165" in t.attrs["source"]
    assert BK.long_trades("vol", root=tmp_path) is None
    assert BK.long_trades("cap", root=tmp_path / "missing") is None

