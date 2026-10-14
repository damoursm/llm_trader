"""PREREG105's rebuilt sets A2 / B2 in the simulator (`src/backtest/rebuild.py`, `periods.REBUILT_SETS`).

What must hold: (1) THE EXISTING SETS AND DEFAULTS ARE BIT-IDENTICAL — A2 / B2 are asked for by name only, every other
periods function reads A / B / C / A0 / D as before, and `mimic.live_books` given the rebuilt components runs its
default path (no set-A0 code, the default crash call) and builds what it builds with any other component; (2) the
construction rules: the data's end is frozen once per construction (an override, an extension); every row of a
rebuilt part is in its class's store precision; two series are "the same" only bar for bar and value for value; the
traded-price factor is set A0's construction — the jump test for a split inside set A0's data, measurement against
the unadjusted close (snapped to the table's ratios) for every later split, set A0's own measurement for the rows up
to its data end; (3) the rebuilt components read the rebuilt files (the re-ranked ETF picks by their sidecar, the
extended bars in the extended stores' precision, the dip panel's own bars) and guard against another store's arrays;
(4) the harness swaps a study's components, V6's flags, mimic's lend / fee arrays, the crash book's bars and the halt
files for a block, and restores every one."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import a0 as A0
from src.backtest import halts as H
from src.backtest import mimic as M
from src.backtest import periods as P
from src.backtest import rebuild as RB
from src.backtest.strategies import crash as CRASH
from src.backtest.strategies import dm as DM


def dn(s) -> int:
    return int((pd.Timestamp(s) - pd.Timestamp("1970-01-01")).days)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("PREREG105_SANDBOX", str(tmp_path))
    monkeypatch.delenv("PREREG105_DATA_END", raising=False)
    return tmp_path


def split_factor(sday, c, splits):
    """`volengine.split_factor` (the research engine's jump test), verbatim."""
    f = np.ones(len(c))
    for ex_dn, frm, to in splits:
        nb = int(np.searchsorted(sday, ex_dn))
        if nb == 0 or nb >= len(c):
            continue
        jump = c[nb] / c[nb - 1]
        if abs(np.log(jump) - np.log(frm / to)) >= np.log(1.5):
            f[:nb] *= to / frm
    return f


# ── (1) the existing sets and defaults are untouched ─────────────────────────

def test_a2_and_b2_are_sets_a_and_b_asked_for_by_name_only():
    a2, b2 = P.get("A2"), P.get("B2")
    assert (a2.lo, a2.hi, a2.borrow) == (P.get("A").lo, P.get("A").hi, P.get("A").borrow)
    assert (b2.lo, b2.hi, b2.borrow) == (P.get("B").lo, P.get("B").hi, P.get("B").borrow)
    assert [s.key for s in P.SETS] == ["A", "B", "C"] and [s.key for s in P.EXTRA_SETS] == ["A0"]
    assert [s.key for s in P.REBUILT_SETS] == ["A2", "B2"]
    for extra in (False, True):
        assert P.set_of("2022-06-01", extra) == "A" and P.set_of("2025-06-02", extra) == "B"
        assert [k for k, _, _ in P.windows("2026-09-24", extra)] == (["A0"] if extra else []) + ["A", "B"]
    assert P.year_parts("2021-02-01", "2026-09-24")[3][:2] == ("2024 A", "A")
    with pytest.raises(KeyError):
        P.get("A1")
    with pytest.raises(KeyError):
        P.get("C2")


def _halts_tests():
    p = Path(__file__).with_name("test_backtest_halts.py")
    spec = importlib.util.spec_from_file_location("_halts_stub_rb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_books_with_rebuilt_components_runs_the_default_path_and_builds_the_same_books(monkeypatch):
    T = _halts_tests()

    def boom(*a, **k):
        raise AssertionError("set A0's code ran on a rebuilt window")
    for f in ("check_window", "check_components", "check_v6", "vol_engine", "etf_arm", "dip_panel", "ext_bars"):
        monkeypatch.setattr(A0, f, boom)
    real_case = T._stub_case

    def case():
        eng, etf, costs, p_lend, dip = real_case()
        eng.period = etf.period = RB.PERIOD                  # the rebuilt components' marker
        return eng, etf, costs, p_lend, dip
    monkeypatch.setattr(T, "_stub_case", case)
    books, log, eng, etf = T._live_books(monkeypatch, M.Mimic(seed=3))
    assert T._fields(books) == T.BEFORE and log == T.BEFORE_LOG


# ── (2) the construction rules ───────────────────────────────────────────────

def test_the_data_end_is_frozen_once_and_moved_only_by_an_extension(sandbox, monkeypatch):
    monkeypatch.setenv("PREREG105_DATA_END", "2026-10-13")
    assert not RB.spans_path().exists() and RB.data_end() == "2026-10-13" and RB.spans()["fixed"] is False
    sp = RB.fix_spans()
    assert sp["data_end"] == sp["store_hi"] == "2026-10-13" and RB.spans_path().exists()
    monkeypatch.setenv("PREREG105_DATA_END", "2026-11-02")
    assert RB.data_end() == "2026-10-13"                  # frozen: the environment no longer moves it
    assert RB.fix_spans()["data_end"] == "2026-10-13"     # fixing again keeps it
    sp = RB.fix_spans("2026-10-20")                       # an extension
    assert RB.data_end() == RB.store_hi() == "2026-10-20" and sp["history"] == ["2026-10-13"]
    assert RB.chunks() == (2021, 2022, 2023, 2024, 2025, 2026) and RB.dip_hi() == "2026-10-20"
    with pytest.raises(ValueError):
        RB.fix_spans("2026-09-01")                        # before set B's last decision


def test_a_production_store_is_reused_only_to_the_reuse_cut(sandbox, monkeypatch):
    """Polygon still revises the latest weeks: a store's bars are reused to 30 NYSE sessions before the data's end."""
    monkeypatch.setenv("PREREG105_DATA_END", "2026-10-13")
    assert RB.reuse_to() == "2026-08-31"                  # 30 sessions back (Labor Day 2026-09-07 skipped)
    assert RB.REUSE_OVERLAP >= 1 and RB.CHECK_LO < RB.EARLY_END < RB.LATE_FROM


def test_the_rebuilt_windows(sandbox, monkeypatch):
    monkeypatch.setenv("PREREG105_DATA_END", "2026-10-09")
    assert RB.window("A2") == (P.get("A").lo, P.get("A").hi) and RB.window("B2") == (P.get("B").lo, P.get("B").hi)
    assert RB.window("D", now="2026-10-14 17:00") == ("2026-09-25", "2026-10-09")     # set D cut at the data's end
    assert RB.window("D2", now="2026-10-01 12:00") == ("2026-09-25", "2026-09-30")
    RB.check_window("2021-01-04", "2026-10-09")
    with pytest.raises(ValueError):
        RB.check_window("2020-12-31", "2021-03-01")
    with pytest.raises(ValueError):
        RB.check_window("2026-01-02", "2026-10-13")


def _bars(ts, v):
    return pd.DataFrame({"ts": pd.to_datetime(ts), "open": [1.123456789] * len(ts), "high": [1.2] * len(ts),
                         "low": [1.0] * len(ts), "close": [1.1] * len(ts), "volume": v, "vwap": [1.1] * len(ts)})


def test_every_row_is_in_its_classs_store_precision(monkeypatch):
    monkeypatch.setattr(RB, "raw_volume_names", lambda: frozenset({"AMC"}))
    ts = ["2025-06-02 14:30", "2026-02-20 14:30", "2026-02-23 14:30"]           # before / on Polygon's switch
    v = [10.6, 20.6, 30.6]
    d = RB.store_precision(_bars(ts, v), "XYZ@D")
    assert d["close"][0] == float(np.float32(1.1)) and list(d["volume"]) == [10.0, 20.0, 30.0]
    x = RB.store_precision(_bars(ts, v), "XYZ@X")
    assert x["close"][0] == 1.1 and list(x["volume"]) == [10.0, 20.0, 30.0]
    e = RB.store_precision(_bars(ts, v), "XYZ@E")
    assert e["close"][0] == 1.1 and list(e["volume"]) == v
    p = RB.store_precision(_bars(ts, v), "XYZ")
    assert p["close"][0] == 1.1 and list(p["volume"]) == [10.0, 20.0, 30.6]   # Polygon's fractions from 2026-02-23
    r = RB.store_precision(_bars(ts, v), "AMC")
    assert list(r["volume"]) == v                                                # a name the deep store kept raw
    # set A0's precision on the rows A0 covers (the reused early segment): the same rule
    df = pd.DataFrame({"Open": [1.123456789], "High": [1.2], "Low": [1.0], "Close": [1.1], "Volume": [10.6]})
    assert A0.in_precision(df.copy(), "XYZ")["Volume"][0] == RB.store_precision(_bars(ts[:1], [10.6]), "XYZ")["volume"][0]


def test_two_series_are_the_same_bar_for_bar_and_value_for_value():
    a = _bars(["2021-03-01 14:30", "2021-03-01 15:00"], [5.0, np.nan])
    assert RB.same_rows(a, a.copy())[0]
    b = a.copy()
    b.loc[1, "close"] = 1.2
    ok, why = RB.same_rows(a, b)
    assert not ok and why["column"] == "close" and why["rows"] == 1
    ok, why = RB.same_rows(a, a.iloc[:1])
    assert not ok and why["only_a"] == 1
    assert RB.et_days(pd.Series(pd.to_datetime(["2021-03-02 01:00", "2021-03-01 23:30"]))).tolist() == \
        [dn("2021-03-01"), dn("2021-03-01")]              # 20:00 and 18:30 ET: the session of 03-01


def test_a_reused_store_is_the_same_series_rth_exactly_and_extended_hours_in_their_precision():
    """The deep store keeps its extended-hours bars in float32 (and whole shares): a fresh fetch's float64 extended
    bars are the same series when they agree in that precision; the regular-hours bars must agree exactly."""
    ts = ["2021-03-01 13:00", "2021-03-01 14:30", "2021-03-01 21:00"]          # pre, rth, post (naive UTC)
    a = _bars(ts, [100.7, 200.0, 300.2]).assign(session=["pre", "rth", "post"])
    b = a.copy()
    b.loc[[0, 2], "open"] = float(np.float32(1.123456789))                   # the stored float32 extended bars
    b.loc[[0, 2], "volume"] = [100.0, 300.0]
    assert RB.same_series(a, b) == (True, {"bars": 3, "rth": 1})
    c = b.copy()
    c.loc[1, "open"] = float(np.float32(1.123456789))                        # a float32 REGULAR-hours bar: different
    ok, why = RB.same_series(a, c)
    assert not ok and why["part"] == "rth" and why["column"] == "open"
    d = b.copy()
    d.loc[2, "close"] = 1.15
    ok, why = RB.same_series(a, d)
    assert not ok and why["part"] == "extended"
    assert RB.ext_precision(a)["volume"].tolist() == [100.0, 200.0, 300.0]


def _series(days, closes):
    sd = np.array([dn(d) for d in days for _ in range(2)], np.int64)
    c = np.array([x for x in closes for _ in range(2)], float)
    return sd, c


def test_the_factor_is_set_a0s_construction_extended():
    days = [d.date().isoformat() for d in pd.bdate_range("2021-06-14", "2021-07-30")]
    closes = [10.0] * len(days)
    sd, c = _series(days, closes)                        # stored (adjusted) closes: no jump anywhere
    early_split = (dn("2021-06-21"), 1.0, 2.0)           # 2:1 inside set A0's data: the jump test applies it
    small = (dn("2021-07-12"), 1.0, 1.1)                 # a 10% stock split after it: only measurement can see it
    big = (dn("2021-07-26"), 1.0, 3.0)                   # 3:1 later still
    unadj = {}
    for d in days:                                       # the unadjusted close a trader saw: the splits after d
        f = 1.0
        for ex, frm, to in (small, big):
            if ex > dn(d):
                f *= to / frm
        unadj[d] = 10.0 * f
    rec = RB.factor_record(sd, c, [early_split, small, big], unadj.get)
    assert rec["K"] == dn("2021-06-30") and rec["jump"] == [list(early_split)]
    (lo0, k0, f0, h0), (lo1, hi1, f1, h1), (lo2, hi2, f2, h2), (lo3, hi3, f3, h3) = rec["segments"]
    assert (lo0, k0, f0, h0) == (dn("2021-06-14"), dn("2021-06-30"), pytest.approx(3.3), "table")   # A0's own
    assert (lo1, hi1, f1) == (dn("2021-07-01"), dn("2021-07-09"), pytest.approx(3.3))   # after K, before the 10%
    assert (lo2, hi2, f2, h2) == (dn("2021-07-12"), dn("2021-07-23"), pytest.approx(3.0), "table")
    assert (lo3, hi3, f3, h3) == (dn("2021-07-26"), dn("2021-07-30"), 1.0, "none")
    f = RB.row_factors(sd, c, rec, split_factor)
    assert f[sd < dn("2021-06-21")] == pytest.approx(2.0 * 3.3)                         # the jump split too
    assert f[(sd >= dn("2021-06-21")) & (sd < dn("2021-07-12"))] == pytest.approx(3.3)
    assert split_factor(sd, c, [small])[0] == 1.0        # the jump test alone misses the 10% split


def test_the_k_segment_is_measured_as_set_a0_measured_it():
    """Set A0's `post_factor`: the median ratio over the last 5 sessions <= its data end, snapped to the products of
    the splits after it — the rebuild's K segment, whatever comes after K."""
    days = [d.date().isoformat() for d in pd.bdate_range("2021-06-01", "2021-08-31")]
    sd, c = _series(days, [20.0] * len(days))
    post = (dn("2021-08-02"), 1.0, 4.0)
    ratio = {d: (80.4 if d <= "2021-06-30" else 40.0) for d in days}       # an unlisted adjustment after K too
    rec = RB.factor_record(sd, c, [post], lambda d: ratio[d])
    k = rec["segments"][0]
    assert k[1] == dn("2021-06-30") and k[2] == 4.0 and k[3] == "table"   # 80.4 / 20 = 4.02: snapped
    after = rec["segments"][1]
    assert after[0] == dn("2021-07-01") and after[3] == "measured" and after[2] == pytest.approx(2.0)
    nc = RB.factor_record(sd, c, [post], lambda d: None)
    assert nc["segments"][0][2:] == [4.0, "no_close"]    # no close: every split after applied (set A0's fallback)


def test_a_name_without_a0_rows_is_measured_segment_by_segment():
    days = [d.date().isoformat() for d in pd.bdate_range("2023-03-01", "2023-03-31")]
    sd, c = _series(days, [5.0] * len(days))
    rev = (dn("2023-03-15"), 10.0, 1.0)                  # a 1-for-10 reverse split
    unadj = {d: (0.5 if d < "2023-03-15" else 5.0) for d in days}
    rec = RB.factor_record(sd, c, [rev], unadj.get)
    assert rec["K"] is None and rec["jump"] == []
    assert [s[2:] for s in rec["segments"]] == [[pytest.approx(0.1), "table"], [1.0, "none"]]
    f = RB.row_factors(sd, c, rec, split_factor)
    assert (c * f)[sd < dn("2023-03-15")] == pytest.approx(0.5)


# ── (3) the rebuilt components ───────────────────────────────────────────────

def _events(n=3, tkn=("UVXY@E", "SPY@E", "QQQ@E")):
    return pd.DataFrame({"tkn": list(tkn)[:n], "dn": [dn("2023-03-06")] * n, "pos": list(range(n))},
                        index=[10, 11, 12][:n])


def test_the_rebuilt_etf_arm_reads_its_own_re_ranked_picks(sandbox):
    from src.backtest.strategies import etf as ETF
    cls = RB._etf_arm_class()
    arm = object.__new__(cls)
    arm.E, arm._ex = _events(), {}
    key = frozenset({"UVXY", "VXX"})
    with pytest.raises(FileNotFoundError):
        arm.events(key)
    d = RB.research_out() / "etf"
    d.mkdir(parents=True)
    re_ranked = _events(3, ("UVXY@E", "IWM@E", "QQQ@E"))
    re_ranked.to_pickle(RB.etf_events_path("novix"))
    meta = {"period": RB.PERIOD, "excluded": sorted(key), "source_sig": ETF.events_signature(arm.E)}
    RB.etf_events_path("novix").with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    got = arm.events(key)
    assert list(got.tkn) == ["IWM@E", "QQQ@E"]            # re-ranked, the excluded products never returned
    assert arm.events(None) is arm.E and arm.events_source(key).endswith("vol_events_ab2_novix_arm.pkl")
    assert RB.excluded_sidecar(frozenset({"UVXY"})) is None                 # another exclusion: not this file
    arm2 = object.__new__(cls)
    arm2.E, arm2._ex = _events(2), {}
    with pytest.raises(ValueError, match="another version"):
        arm2.events(key)
    meta["period"] = "A0"                                 # another period's sidecar is never read
    RB.etf_events_path("novix").with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    assert RB.excluded_sidecar(key) is None


def _part(path, ts, sessions):
    from src.data.deep import write_parquet
    df = pd.DataFrame({"ticker": "AAA", "ts": pd.DatetimeIndex(ts), "open": [10.123456789] * len(ts),
                       "high": [10.3] * len(ts), "low": [10.0] * len(ts), "close": [10.2] * len(ts),
                       "volume": [100.7] * len(ts), "vwap": [10.1] * len(ts), "trades": [3] * len(ts),
                       "session": sessions})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, path)


def test_the_rebuilt_extended_bars_and_dip_bars_read_the_rebuilt_parts(sandbox):
    _part(RB.parts_dir() / "AAA.parquet", ["2023-03-01 13:00", "2023-03-01 14:30"], ["pre", "rth"])
    b = RB.ext_bars()
    assert isinstance(b, M.ExtBars) and b.period == RB.PERIOD
    assert b.path("AAA@D") == RB.parts_dir() / "AAA.parquet" and b.path("BBB") is None
    x = b.get("AAA")
    assert x["o"][0] == float(np.float32(10.123456789)) and list(x["v"]) == [100.0, 100.0]
    assert list(x["rth"]) == [False, True]
    fr = RB.DipPanelR.bars_fn("AAA@D")
    assert len(fr) == 1 and fr.index[0] == pd.Timestamp("2023-03-01 14:30") and fr["Open"].iloc[0] == 10.123456789
    assert RB.DipPanelR({}, []).period == RB.PERIOD


def test_the_rebuilt_engine_refuses_another_stores_arrays():
    cls = RB._vol_engine_class()
    eng = object.__new__(cls)
    eng.st = types.SimpleNamespace(C=pd.DataFrame({"x": range(5)}))
    eng.fee_fallback = np.zeros(5)
    assert len(eng.fee_fallback) == 5
    eng.fee_fallback = None
    with pytest.raises(ValueError, match="fee fill"):
        eng.fee_fallback = np.zeros(7)
    assert cls.period == RB.PERIOD != "A0"
    mod = types.ModuleType("_v6_rb")
    mod.NFHI = np.zeros(7, bool)
    sys.modules["_v6_rb"] = mod
    try:
        eng.Real = type("Real6", (), {"__module__": "_v6_rb"})
        with pytest.raises(ValueError, match="pressure flags"):
            RB.check_v6(eng)
        mod.NFHI = np.zeros(5, bool)
        RB.check_v6(eng)
    finally:
        del sys.modules["_v6_rb"]
    eng.Real = type("Real", (), {})
    RB.check_v6(eng)                                      # no V6: nothing to check


# ── (4) the harness ──────────────────────────────────────────────────────────

def test_the_harness_swaps_and_restores_every_component(monkeypatch):
    eng = types.SimpleNamespace(period=RB.PERIOD, st=types.SimpleNamespace(C=pd.DataFrame({"x": range(4)})))
    monkeypatch.setattr(RB, "vol_engine", lambda: eng)
    monkeypatch.setattr(RB, "etf_arm", lambda: "etf2")
    monkeypatch.setattr(RB, "dip_panel", lambda: "panel2")
    monkeypatch.setattr(RB, "ext_bars", lambda extra_dir=None: "bars2")
    monkeypatch.setattr(RB, "settle", lambda: "settle2")
    monkeypatch.setattr(RB, "dip_pieces", lambda prm=None: ["dip2"])
    monkeypatch.setattr(RB, "pressure_hi", lambda n: np.ones(n, bool))
    v6 = types.ModuleType("run71")
    v6.NFHI, v6.NF = np.zeros(9, bool), np.zeros(9)
    monkeypatch.setitem(sys.modules, "run71", v6)
    keep = (M.lend_p_path, M.fee_pred_path, CRASH._daily, DM.bars_30m)
    H.set_halt_paths(["h2019.csv"])                       # a study left set A0's halt files installed
    W = {"eng": "eng_ab", "etf": "etf_ab", "panel": "p_ab", "dip": "d_ab", "bars": "b_ab", "costs": "costs"}
    before = dict(W)                                      # no "settle": the harness adds it, and takes it away
    try:
        with RB.harness(W) as w:
            assert w is W and W["eng"] is eng and (W["etf"], W["panel"], W["bars"], W["settle"], W["dip"]) == (
                "etf2", "panel2", "bars2", "settle2", ["dip2"])
            assert W["costs"] == "costs" and v6.NFHI.tolist() == [True] * 4
            assert M.lend_p_path("v2") == RB.store_dir() / "lend_p_v2.npy"
            assert M.fee_pred_path("v4") == RB.store_dir() / "fee_pred_v4.npy"
            assert CRASH._daily is RB.daily and DM.bars_30m is RB.bars_rth_frame
            assert H.halt_paths() is None                 # the existing NYSE files (2021 on)
        assert W == before and len(v6.NFHI) == 9 and not v6.NFHI.any()
        assert (M.lend_p_path, M.fee_pred_path, CRASH._daily, DM.bars_30m) == keep
        assert H.halt_paths() == ["h2019.csv"]
    finally:
        H.set_halt_paths(None)


def test_the_harness_refuses_v6_without_the_rebuilt_flags(monkeypatch):
    eng = types.SimpleNamespace(period=RB.PERIOD, st=types.SimpleNamespace(C=pd.DataFrame({"x": range(4)})))
    for f, v in (("vol_engine", lambda: eng), ("etf_arm", lambda: "e"), ("dip_panel", lambda: "p"),
                 ("ext_bars", lambda extra_dir=None: "b"), ("settle", lambda: "s"), ("dip_pieces", lambda prm=None: []),
                 ("pressure_hi", lambda n: None)):
        monkeypatch.setattr(RB, f, v)
    v6 = types.ModuleType("run71")
    v6.NFHI, v6.NF = np.zeros(9, bool), np.zeros(9)
    monkeypatch.setitem(sys.modules, "run71", v6)
    W = {"costs": "c", "eng": "eng_ab"}
    with pytest.raises(FileNotFoundError, match="pressure flags"):
        with RB.harness(W):
            pass
    assert W == {"costs": "c", "eng": "eng_ab"} and len(v6.NFHI) == 9           # nothing swapped, nothing added


def test_the_lend_and_fee_arrays_are_named_as_mimics():
    assert RB.lend_p_path("v1").name == M.LEND_P_FILES["v1"] and RB.lend_p_path("v2").name == M.LEND_P_FILES["v2"]
    assert RB.fee_pred_path("v4").name == M.FEE_PRED_FILES["v4"]
    assert RB.store_dir() != M.lend_p_path().parent       # never the existing store's
