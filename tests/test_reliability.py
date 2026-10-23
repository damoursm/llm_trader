"""The reliability card, the plateau rule and the bundle (user 2026-10-12, items 1, 2 and 5)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import reliability as R
from src.backtest.account import per_day

ET = "America/New_York"


def _ns(s: str) -> int:
    return int(pd.Timestamp(s, tz=ET).value)


def _pos(strategy, tkn, entry, exit, entry_px, exit_px, shares, pnl, **kw):
    d = {"strategy": strategy, "tkn": tkn, "entry_ns": _ns(entry), "exit_ns": _ns(exit), "entry_px": entry_px,
         "exit_px": exit_px, "shares": shares, "pnl": pnl}
    d.update(kw)
    return d


def test_the_plateau_rule_takes_the_middle_of_the_run_and_flags_a_peak():
    pts = [(1, 10.0), (2, 19.0), (3, 20.0), (4, 19.5), (5, 18.5), (6, 9.0)]
    p = R.plateau(pts, tol=0.1)
    assert p["best"] == 3 and p["plateau"] == [2, 3, 4, 5] and p["center"] == 4 and p["on_peak"] is False
    q = R.plateau([(1, 5.0), (2, 20.0), (3, 6.0)], tol=0.1)
    assert q["best"] == 2 and q["on_peak"] is True and q["center"] == 2
    assert R.plateau([], tol=0.1)["best"] is None
    r = R.plateau([(1, 3.0), (2, float("nan")), (3, 3.1), (4, 3.05)], tol=0.1)
    assert r["plateau"] == [3, 4] and r["on_peak"] is False
    lo = R.plateau([(1, 2.0), (2, 1.0), (3, 1.05), (4, 3.0)], tol=0.1, higher_is_better=False)
    assert lo["best"] == 2 and lo["plateau"] == [2, 3]


def test_trades_from_positions_reads_the_engines_records_and_a_studys_compact_copy():
    pos = [_pos("vol", "AAA", "2025-03-03 10:00", "2025-03-05 10:00", 10.0, 9.0, 100, 95.0),     # a short: 10 -> 9
           _pos("dip", "BBB", "2025-03-03 09:30", "2025-03-04 09:30", 50.0, 51.0, 20, 18.0),
           {"strategy": "fall", "tkn": "CCC", "entry_ns": _ns("2025-03-03 10:00"), "exit_ns": _ns("2025-03-04 10:00"),
            "ret": 0.02, "pnl": 40.0},
           {"strategy": "vol", "tkn": "OPEN", "entry_ns": _ns("2025-03-03 10:00"), "exit_ns": None, "ret": 0.0, "pnl": 0.0}]
    t = R.trades_from_positions(pos)
    assert list(t.book) == ["vol", "dip", "fall"]
    a = t.iloc[0]
    assert a.gross == pytest.approx(10.0) and a.net == pytest.approx(9.5) and a.days == pytest.approx(2.0)
    b = t.iloc[1]
    assert b.gross == pytest.approx(2.0) and b.net == pytest.approx(1.8)
    c = t.iloc[2]
    assert math.isnan(c.gross) and c.net == pytest.approx(2.0)


def test_stats_follow_the_standing_definitions():
    t = pd.DataFrame({"book": "vol", "ticker": ["A", "B", "C"], "entry": pd.to_datetime(["2025-01-02"] * 3),
                      "exit": pd.to_datetime(["2025-01-04"] * 3), "days": [2.0, 2.0, 5.0], "net": [3.0, -1.0, 1.0],
                      "gross": [3.2, -0.8, -0.1], "pnl": [30.0, -10.0, 10.0]})
    s = R.stats(t)
    assert s["n"] == 3 and s["avg_trade"] == pytest.approx(1.0) and s["avg_days"] == pytest.approx(3.0)
    assert s["return_per_day"] == pytest.approx(per_day(1.0, 3.0))
    assert s["gross_win_rate"] == pytest.approx(1 / 3) and s["win_basis"] == "gross"     # the gross direction decides
    t2 = t.assign(gross=np.nan)
    assert R.stats(t2)["gross_win_rate"] == pytest.approx(2 / 3) and R.stats(t2)["win_basis"] == "net"
    assert R.stats(t.iloc[:0]) == {"n": 0}


def _spy(n=600, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({"Close": c}, index=idx)


def test_regimes_label_trend_and_volatility_terciles_and_apply_as_of():
    spy = _spy()
    tab = R.regime_table(spy, trend_sessions=50, vol_sessions=20)
    assert set(tab.trend.dropna().unique()) <= {"above", "below"} and tab.trend.iloc[:49].isna().all()
    v = tab.vol.dropna()
    counts = v.value_counts(normalize=True)
    assert all(abs(counts[k] - 1 / 3) < 0.05 for k in ("low", "mid", "high"))
    r = R.regime_of([spy.index[100] + pd.Timedelta(hours=10), spy.index[0] - pd.Timedelta(days=30)], tab)
    assert r.trend.iloc[0] == tab.trend.iloc[100] and r.trend.iloc[1] is None


def test_the_card_groups_by_year_set_and_regime_and_names_where_the_arm_loses():
    spy = _spy(n=1500)
    tab = R.regime_table(spy, trend_sessions=50, vol_sessions=20)
    rows = []
    for k in range(40):
        day = pd.Timestamp("2021-03-01", tz=ET) + pd.Timedelta(days=7 * k)
        rows.append({"book": "vol", "ticker": f"T{k}", "entry": day, "exit": day + pd.Timedelta(days=3), "days": 3.0,
                     "net": -2.0 if k < 15 else 4.0, "gross": -2.0 if k < 15 else 4.0, "pnl": 0.0})
    t = pd.DataFrame(rows)
    c = R.card(t, "vol", tab)
    assert c["arm"] == "vol" and c["side"] == "short" and c["spec"]["parameters"]
    assert c["overall"]["n"] == 40 and set(c["by_year"]) == {"2021"}
    assert "A" in c["by_set"] and c["by_set"]["A"]["n"] == 40                 # 2021 decisions: borrow-data set A
    assert set(c["by_trend"]) <= {"above", "below"} and set(c["by_vol"]) <= {"low", "mid", "high"}
    assert c["loses_in"] == [] or all("=" in x for x in c["loses_in"])
    t2 = t.assign(net=-1.0)
    assert "by_year=2021" in R.card(t2, "vol", tab)["loses_in"]


def test_sensitivity_reports_the_change_in_return_per_day_overall_and_by_set():
    day = pd.Timestamp("2025-03-03 10:00", tz=ET)
    base = pd.DataFrame([{"book": "vol", "ticker": "A", "entry": day, "exit": day + pd.Timedelta(days=2), "days": 2.0,
                          "net": 4.0, "gross": 4.0, "pnl": 0.0}] * 12)
    worse = base.assign(net=2.0)
    s = R.sensitivity(base, {"costs_x2": worse}, "vol")
    assert s["variants"]["costs_x2"]["rpd_change"] == pytest.approx(per_day(2.0, 2.0) - per_day(4.0, 2.0))
    assert s["variants"]["costs_x2"]["rpd_change_by_set"]["B"] < 0


def test_the_bundle_measures_activity_overlap_and_the_co_activation_headroom():
    pos = [_pos("vol", "A", "2025-03-03 10:00", "2025-03-07 10:00", 10.0, 9.0, 100, 100.0),     # notional 1000
           _pos("dip", "B", "2025-03-05 09:30", "2025-03-12 09:30", 50.0, 51.0, 20, 20.0),      # notional 1000
           _pos("crash", "TQQQ", "2025-03-20 09:30", "2025-03-25 09:30", 40.0, 44.0, 25, 100.0)]
    sessions = pd.bdate_range("2025-03-03", "2025-03-28", tz=ET)
    b = R.bundle(pos, start=10_000.0, sessions=sessions)
    assert set(b["arms"]) == {"vol", "dip", "crash"}
    assert b["arms"]["vol"]["trades"] == 1 and 0 < b["arms"]["vol"]["active_share"] < 1
    assert b["overlap"]["vol"]["dip"] == b["overlap"]["dip"]["vol"] > 0 and b["overlap"]["vol"]["crash"] == 0
    assert abs(sum(a["pnl_share"] for a in b["arms"].values()) - 1.0) < 1e-9
    # the peak requirement: 03-05 / 03-06, the short (1000 x 2.86) and the long (1000 x 0.5) open on 10,000 equity
    h = b["headroom"]
    assert h["req_peak"] == pytest.approx(3360 / 10_000, rel=1e-6) and h["peak_session"] in ("2025-03-05", "2025-03-06")
    assert sorted(h["active_at_peak"]) == ["dip", "vol"] and h["slice_multiple"] == pytest.approx(10_000 / 3360, rel=1e-6)
    assert R.bundle([])["headroom"] is None


def test_arm_alone_accepts_the_simulators_strategy_list():
    """`mimic.live_strategies` returns a list of Strategy objects (the first build crashed on `.items()`)."""
    from src.backtest import account as A
    from src.backtest import mimic as M
    strategies = M.live_strategies(M.Mimic())
    assert isinstance(strategies, list)
    assert set(R._by_name(strategies)) >= {"vol", "dip"}
    assert R.arm_alone([], strategies, M.live_rules(M.Mimic()), A.default_settle()).empty


def test_every_live_arm_has_a_spec_naming_real_settings_and_no_copied_number():
    """2026-10-21: the specs named "125%", "15 sessions", "the 5-session average" after the deploys changed them —
    a spec names its settings; their values are the arm's fingerprint (`arm_settings`)."""
    import re
    from config.settings import settings
    from src.backtest import live_calibration as LC
    from src.backtest import reliability as REL
    assert set(REL.ARMS) == set(REL.SPEC) == set(REL.SIDE) == set(REL.INIT_RATE) == set(LC.ARM_KEYS)
    fields = type(settings).model_fields
    for arm, sp in REL.SPEC.items():
        names = sp["parameters"] + sp["switches"] + sp["sizing"]
        assert names and all(n in fields for n in names), (arm, [n for n in names if n not in fields])
        for key in ("mechanism", "context", "horizon"):
            assert not re.search(r"\d+(\.\d+)?\s*(%|sessions?\b|x\b)", sp[key]), (arm, key, sp[key])
        assert REL.arm_settings(arm) == {n: getattr(settings, n) for n in sorted(set(names))}
        assert re.fullmatch(r"[0-9a-f]{12}", REL.arm_fingerprint(arm))
