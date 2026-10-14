"""PREREG101's set A0 in the simulator (`src/backtest/a0.py` + the A0 switches of the adapters, `mimic`, `halts`).

What must hold: (1) THE DEFAULTS AND SETS A / B ARE BIT-IDENTICAL — no A0 code runs for a window on/after the deep
store's first session with the default components (other studies import this code while they run and pair their
draws with earlier runs); (2) a window before the deep store takes set A0's components and refuses another period's
(a default engine would silently build nothing); (3) each A0 component reads A0's data: the calendars continue A0's
sessions with the default ones, the fees take IBKR's API bar where it covers the day and IBKR's archived file
otherwise, the dip adapter reads an A0 panel's own bars and names, the crash book decides on A0's QQQ, the extended
bars come from A0's parts in the extended stores' precision, the halt model reads the configured halt files; (4) the
harness swaps a study's components (and V6's flags) for a block and restores them."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import a0 as A0
from src.backtest import halts as H
from src.backtest import mimic as M
from src.backtest.strategies import crash as CRASH
from src.backtest.strategies import dip as DIP

ET = "America/New_York"


def dn(s) -> int:
    return int((pd.Timestamp(s) - pd.Timestamp("1970-01-01")).days)


def _halts_tests():
    """The stub live combination of `test_backtest_halts` (its pieces captured before PREREG99's change)."""
    p = Path(__file__).with_name("test_backtest_halts.py")
    spec = importlib.util.spec_from_file_location("_halts_stub", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── (1) the defaults are untouched ───────────────────────────────────────────

def test_live_books_on_set_a_runs_no_a0_code_and_builds_what_it_built_before(monkeypatch):
    T = _halts_tests()

    def boom(*a, **k):
        raise AssertionError("set A0's code ran on a set-A window")
    for f in ("check_window", "check_components", "check_v6", "vol_engine", "etf_arm", "dip_panel", "ext_bars"):
        monkeypatch.setattr(A0, f, boom)
    real_pieces = CRASH.pieces

    def crash_pieces(lo, hi, *a, **k):
        assert "period" not in k, "the crash book took a period on a set-A window"
        return real_pieces(lo, hi, *a, **k)
    monkeypatch.setattr(CRASH, "pieces", crash_pieces)
    books, log, eng, etf = T._live_books(monkeypatch, M.Mimic(seed=3))
    assert T._fields(books) == T.BEFORE and log == T.BEFORE_LOG
    assert eng.calls == [("turn", "fall", 0.5, 5, True, True),
                         ("pieces", "vol", {"entry_at_pick": True, "rule201": False}, 6)]


def test_the_default_components_carry_no_period():
    from src.backtest.strategies.etf import EtfArm
    from src.backtest.strategies.vol import VolEngine
    assert VolEngine.period is None and EtfArm.period is None and M.ExtBars.period is None
    assert M.ExtBars().period is None
    with pytest.raises(ValueError):
        VolEngine(period="B")
    with pytest.raises(ValueError):
        EtfArm(period="B")
    with pytest.raises(ValueError):
        M.ExtBars(period="B")


def test_the_default_extended_bars_read_the_deep_and_delisted_stores(tmp_path, monkeypatch):
    full, dl = tmp_path / "full", tmp_path / "delisted"
    full.mkdir(), dl.mkdir()
    (full / "AAA.parquet").write_text("x"), (dl / "BBB.parquet").write_text("x")
    monkeypatch.setattr(M, "FULL", full)
    monkeypatch.setattr(M, "DELISTED", dl)
    b = M.ExtBars()
    assert b.path("AAA") == full / "AAA.parquet" and b.path("BBB@D") == dl / "BBB.parquet"
    assert b.path("BBB") is None and b.path("AAA@D") is None


def test_lend_all_reads_the_lend_model_unless_the_engine_is_a0s(monkeypatch):
    T = _halts_tests()
    eng, _, _, p, _ = T._stub_case()
    loaded = []
    monkeypatch.setattr(M, "lend_p_path", lambda model="v2": loaded.append(model) or "unused")
    monkeypatch.setattr(np, "load", lambda path, *a, **k: np.full(len(eng.st.C), 0.25))
    a = M.lend_all(eng, M.Mimic(seed=1, house=False))
    assert loaded == ["v2"]
    eng.a0_lend_p = np.zeros(len(eng.st.C))
    loaded.clear()
    b = M.lend_all(eng, M.Mimic(seed=1, house=False))
    assert loaded == [] and np.array_equal(a, b)          # 2023 rows: before the archive's end, the draw unread


def test_the_halt_model_reads_every_years_file_unless_configured(monkeypatch):
    seen = []
    monkeypatch.setattr(H, "halt_book", lambda paths=None: seen.append(paths) or H.HaltBook(
        pd.DataFrame(columns=["sym", "start", "resume", "reason", "exchange"])))
    monkeypatch.setattr(H, "production_slots", lambda: (570, 600))
    monkeypatch.setattr(H, "_MODELS", {})
    assert H.halt_paths() is None
    cal = np.array([dn("2023-03-01"), dn("2023-03-02")], np.int64)
    H.default_model(cal)
    assert seen == [None]
    try:
        H.set_halt_paths(["a.csv", "b.csv"])
        assert H.halt_paths() == ["a.csv", "b.csv"]
        H.default_model(cal)
        assert seen[-1] == ["a.csv", "b.csv"]
    finally:
        H.set_halt_paths(None)
    assert H.halt_paths() is None


# ── (2) a window before the deep store takes A0's components ─────────────────

def test_a_window_before_the_deep_store_is_set_a0s():
    assert A0.in_a0("2018-01-02") and A0.in_a0("2021-01-29") and not A0.in_a0("2021-02-01")
    A0.check_window("2018-01-02", "2021-01-29")
    A0.check_window("2021-02-01", "2021-04-30")
    with pytest.raises(ValueError):
        A0.check_window("2017-12-29", "2018-06-01")
    with pytest.raises(ValueError):
        A0.check_window("2020-06-01", "2021-06-30")


def test_live_books_refuses_another_periods_components_on_an_a0_window(monkeypatch):
    T = _halts_tests()
    eng, etf, costs, p_lend, dip = T._stub_case()
    with pytest.raises(ValueError, match="A0's components"):
        M.live_books("2018-03-01", "2018-03-14", M.Mimic(seed=0), eng=eng, etf=etf, panel=None, bars=M.ExtBars(),
                     costs=costs, p_lend=p_lend, settle=lambda d: int(d) + 2, dip=[])


def test_an_a0_engine_builds_the_crash_book_from_a0s_data(monkeypatch):
    T = _halts_tests()
    monkeypatch.setattr(M, "house_mix", lambda: (("X", 2.0, 2.86),))
    monkeypatch.setattr(settings, "enable_crash_long", True, raising=False)
    eng, etf, costs, p_lend, dip = T._stub_case()
    eng.period = etf.period = "A0"
    bars = M.ExtBars()
    bars.period = "A0"
    panel = A0.DipPanel({}, [])
    seen = {}
    monkeypatch.setattr(A0, "dip_is_a0", lambda d: True)
    monkeypatch.setattr(CRASH, "pieces", lambda lo, hi, **k: seen.setdefault("crash", (lo, hi, k)) and [])
    a0_settle = A0.settles_a0
    monkeypatch.setattr(A0, "settles_a0", lambda f: True)
    books = M.live_books("2021-02-01", "2021-04-30", M.Mimic(seed=0, ext=False), eng=eng, etf=etf, panel=panel,
                         bars=bars, costs=costs, p_lend=p_lend, settle=lambda d: int(d) + 2, dip=dip)
    assert seen["crash"] == ("2021-02-01", "2021-04-30", {"period": "A0"}) and "crash" in books
    assert a0_settle(lambda d: int(d) + 2)                 # a calendar-day stub: 2018-01-02 + 2 = 2018-01-04
    assert not a0_settle(lambda d: dn("2021-01-06"))       # the default calendar: no session before 2021


def test_v6_on_an_a0_engine_must_carry_a0s_flags(monkeypatch):
    mod = types.ModuleType("_v6_stub")
    mod.NFHI = np.zeros(7, bool)
    monkeypatch.setitem(sys.modules, "_v6_stub", mod)
    Real6 = type("Real6", (), {"__module__": "_v6_stub"})
    eng = types.SimpleNamespace(Real=Real6, st=types.SimpleNamespace(C=pd.DataFrame({"x": range(5)})))
    with pytest.raises(ValueError, match="pressure flags"):
        A0.check_v6(eng)
    mod.NFHI = np.zeros(5, bool)
    A0.check_v6(eng)
    A0.check_v6(types.SimpleNamespace(Real=type("Real", (), {}), st=eng.st))     # no V6: nothing to check


# ── (3) the A0 components read A0's data ─────────────────────────────────────

def test_a0s_calendars_continue_with_the_default_ones():
    a0 = np.array([dn("2021-06-28"), dn("2021-06-29"), dn("2021-06-30")], np.int64)
    tail = np.array([dn("2021-06-29"), dn("2021-06-30"), dn("2021-07-01"), dn("2021-07-02")], np.int64)
    assert list(A0.extend(a0, tail)) == [dn("2021-06-28"), dn("2021-06-29"), dn("2021-06-30"), dn("2021-07-01"),
                                         dn("2021-07-02")]
    assert list(A0.extend(np.zeros(0, np.int64), tail)) == list(tail)


def _fake_realfee(monkeypatch, parts):
    mod = types.ModuleType("realfee")

    class RealFees:
        def __init__(self, parts=None, field="fee", archive=None):
            self.parts, self.field, self.archive, self.cache = parts, field, archive, {}
    mod.RealFees, mod.PARTS, mod.ARCHIVE_STALE = RealFees, str(parts), 7
    monkeypatch.setitem(sys.modules, "realfee", mod)
    monkeypatch.setattr(A0, "_research_path", lambda: None)


def test_a0s_fees_take_ibkrs_api_bar_where_it_covers_the_day_else_ibkrs_archived_file(tmp_path, monkeypatch):
    api, arch = tmp_path / "api", tmp_path / "arch"
    api.mkdir(), arch.mkdir()
    pd.DataFrame({"date": ["2019-03-04", "2019-03-05"], "fee": [1.5, 1.6]}).to_csv(api / "AAA.csv", index=False)
    pd.DataFrame({"date": ["2019-02-25", "2019-02-26", "2019-03-04"], "fee": [30.0, 31.0, 99.0]}).to_csv(
        arch / "AAA.csv", index=False)
    pd.DataFrame({"date": ["2018-05-01"], "fee": [12.0]}).to_csv(arch / "DEAD.csv", index=False)
    _fake_realfee(monkeypatch, api)
    monkeypatch.setattr(A0, "archive_fee_dir", lambda: arch)
    rf = A0.real_fees()
    days = np.array([dn(d) for d in ("2019-02-26", "2019-03-01", "2019-03-04", "2019-03-08", "2019-03-20")])
    # before the API's history: the archive (carried 3 days); the API's own days; past the API's week: the archive's
    # last row is older than a week too -> the fallback
    assert list(rf.daily("AAA", days, 9.75)) == [31.0, 31.0, 1.5, 1.6, 9.75]
    assert list(rf.daily("DEAD@D", np.array([dn("2018-05-02"), dn("2018-05-20")]), 9.75)) == [12.0, 9.75]
    assert list(rf.daily("NONE@E", np.array([dn("2018-05-02")]), 4.0)) == [4.0]
    assert rf.covered("AAA", dn("2019-02-26")) and not rf.covered("NONE", dn("2019-02-26"))


def test_the_a0_extended_bars_come_from_a0s_parts_in_the_extended_stores_precision(tmp_path, monkeypatch):
    parts = tmp_path / "parts"
    parts.mkdir()
    ts = pd.DatetimeIndex(["2018-03-01 13:00", "2018-03-01 14:30"])           # 08:00 and 09:30 ET (naive UTC)
    df = pd.DataFrame({"ticker": "AAA", "ts": ts, "open": [10.123456789, 10.2], "high": [10.3, 10.4],
                       "low": [10.0, 10.1], "close": [10.2, 10.3], "volume": [100.7, 200.2], "vwap": [10.1, 10.2],
                       "trades": [3, 4], "session": ["pre", "rth"]})
    from src.data.deep import write_parquet
    write_parquet(df, parts / "AAA.parquet")
    monkeypatch.setattr(A0, "parts_dir", lambda: parts)
    b = M.ExtBars(period="A0")
    assert b.period == "A0" and b.path("AAA@D") == parts / "AAA.parquet" and b.path("AAA") == parts / "AAA.parquet"
    x = b.get("AAA")
    assert x["o"][0] == float(np.float32(10.123456789)) and list(x["v"]) == [100.0, 200.0]
    assert list(x["rth"]) == [False, True]


def test_a0s_bars_keep_each_classs_store_precision(monkeypatch):
    monkeypatch.setattr(A0, "raw_volume_names", lambda: frozenset({"AMC"}))
    df = pd.DataFrame({"Open": [1.123456789], "High": [1.2], "Low": [1.0], "Close": [1.1], "Volume": [10.6]})
    d = A0.in_precision(df.copy(), "XYZ@D")
    assert d["Close"][0] == float(np.float32(1.1)) and d["Volume"][0] == 10.0      # the delisted store's parquet
    x = A0.in_precision(df.copy(), "XYZ@X")
    assert x["Close"][0] == 1.1 and x["Volume"][0] == 10.0                         # the outside store: whole shares
    p = A0.in_precision(df.copy(), "XYZ")
    assert p["Close"][0] == 1.1 and p["Volume"][0] == 10.0                         # the deep store: float64, whole shares
    r = A0.in_precision(df.copy(), "AMC")
    assert r["Volume"][0] == 10.6                                                  # ... or Polygon's, as it kept them
    e = A0.in_precision(df.copy(), "XYZ@E")
    assert e["Close"][0] == 1.1 and e["Volume"][0] == 10.6                         # the ETF store: Polygon's as served


def _daily(days, closes):
    return pd.DataFrame({"o": closes, "h": [c * 1.01 for c in closes], "l": [c * 0.99 for c in closes], "c": closes,
                         "v": [2e7] * len(closes)}, index=pd.DatetimeIndex(days))


def test_the_dip_adapter_reads_an_a0_panels_names_and_bars(monkeypatch):
    days = pd.bdate_range("2017-01-02", periods=260)
    closes = list(np.linspace(100, 150, 250)) + [150, 149, 140, 130, 125, 150, 151, 152, 153, 154]
    panel = A0.DipPanel({"MEGA@D": _daily(days, closes), "FUND@E": _daily(days, closes)}, ["MEGA@D"])
    monkeypatch.setattr(DIP, "bars_30m", lambda n: (_ for _ in ()).throw(AssertionError("the deep store was read")))
    seen = []

    def bars_fn(name):
        seen.append(name)
        idx = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} 14:30") for d in days])
        return pd.DataFrame({"Open": closes, "High": closes, "Low": [c * 0.97 for c in closes], "Close": closes,
                             "Volume": [1e6] * len(closes)}, index=idx)
    panel.bars_fn = bars_fn
    assert DIP.eligible_names(panel) == ["MEGA@D"]
    P = DIP.order_pieces(DIP.DipParams(min_dv=1e8), str(days[240].date()), str(days[-5].date()), panel=panel)
    assert seen and set(seen) == {"MEGA@D"} and P and {p["tkn"] for p in P} == {"MEGA@D"}


def test_the_crash_book_decides_on_a0s_qqq_and_holds_a0s_tqqq(monkeypatch):
    days = pd.bdate_range("2017-01-02", periods=260)
    q = _daily(days, [100.0] * 260)
    seen = {}
    monkeypatch.setattr(A0, "daily", lambda s: seen.setdefault("daily", s) and q)
    monkeypatch.setattr(A0, "bars_rth_frame", lambda s: seen.setdefault("bars", s) and pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.DatetimeIndex(["2018-01-02 14:30"])))
    from src.backtest.strategies import dm as DM
    monkeypatch.setattr(DM, "daily", lambda s: pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                                                             "Volume": [1.0]}, index=pd.DatetimeIndex(["2018-01-02"])))
    monkeypatch.setattr(DM, "bars_30m", lambda s: (_ for _ in ()).throw(AssertionError("the deep store was read")))
    calls = []
    monkeypatch.setattr(CRASH, "trades", lambda lo, hi, daily=None, sym=None: calls.append((daily is q, sym)) or [])
    assert CRASH.pieces("2018-01-02", "2018-12-31", sym="QQQ", trade_sym="TQQQ", period="A0") == []
    assert seen == {"daily": "QQQ", "bars": "TQQQ"} and calls == [(True, "QQQ")]
    with pytest.raises(ValueError):
        CRASH.pieces("2018-01-02", "2018-12-31", period="B")


def test_dip_pieces_of_another_period_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(A0, "parts_dir", lambda: tmp_path)
    (tmp_path / "MEGA.parquet").write_text("x")
    a0 = [{"tkn": "MEGA", "pick_day": dn("2019-05-01")}]
    assert A0.dip_is_a0(a0) and A0.dip_is_a0([])
    assert not A0.dip_is_a0(a0 + [{"tkn": "MEGA", "pick_day": dn("2021-11-01")}])     # the default book's
    assert not A0.dip_is_a0([{"tkn": "OTHER", "pick_day": dn("2019-05-01")}])          # a name without A0 bars


def test_the_reliability_cards_of_set_a0_use_a0s_components_and_never_touch_the_live_reference(tmp_path, monkeypatch):
    from src.backtest import live_calibration as LC
    from src.backtest import reliability as R
    from src.backtest.strategies import etf as ETF
    from src.backtest.strategies import vol as VOL
    seen = []
    for f in ("vol_engine", "etf_arm", "dip_panel", "ext_bars"):
        monkeypatch.setattr(A0, f, lambda *a, _f=f, **k: seen.append(_f) or _f)
    monkeypatch.setattr(A0, "settle", lambda: "settle0")

    def boom(*a, **k):
        raise AssertionError("a default component was loaded for set A0")
    monkeypatch.setattr(VOL, "VolEngine", boom)
    monkeypatch.setattr(ETF, "EtfArm", boom)
    monkeypatch.setattr(DIP, "build_panel", boom)
    monkeypatch.setattr(M, "LedgerCosts", lambda snap: "costs")
    calls = []
    monkeypatch.setattr(M, "live_books", lambda lo, hi, mim, **k: calls.append((lo, hi, k["eng"], k["settle"])) or {})
    monkeypatch.setattr(R, "regime_table", lambda: None)
    monkeypatch.setattr(R, "out_dir", lambda day=None: tmp_path / "cards" / "2026-10-14")
    monkeypatch.setattr(LC, "out_dir", lambda: tmp_path / "live_calibration")
    out = R.build(("A0",), sensitivities=False, log=lambda *a: None)
    assert out.name == "2026-10-14_A0" and (out / "bundle_arms_alone.json").exists()
    assert calls == [("2018-01-02", "2021-01-29", "vol_engine", "settle0")]
    assert sorted(seen) == ["dip_panel", "etf_arm", "ext_bars", "vol_engine"]
    assert not (tmp_path / "live_calibration").exists()              # set B's reference is production's: untouched


# ── (4) the harness ──────────────────────────────────────────────────────────

def test_the_harness_swaps_the_components_and_v6s_flags_for_a_block(monkeypatch):
    eng = types.SimpleNamespace(period="A0", st=types.SimpleNamespace(C=pd.DataFrame({"x": range(4)})))
    monkeypatch.setattr(A0, "vol_engine", lambda: eng)
    monkeypatch.setattr(A0, "etf_arm", lambda: "etf0")
    monkeypatch.setattr(A0, "dip_panel", lambda: "panel0")
    monkeypatch.setattr(A0, "ext_bars", lambda extra_dir=None: "bars0")
    monkeypatch.setattr(A0, "settle", lambda: "settle0")
    monkeypatch.setattr(A0, "dip_pieces", lambda prm=None: ["dip0"])
    monkeypatch.setattr(A0, "pressure_hi", lambda n: np.ones(n, bool))
    monkeypatch.setattr(A0, "halt_files", lambda: ["h2019.csv"])
    v6 = types.ModuleType("run71")
    v6.NFHI, v6.NF = np.zeros(9, bool), np.zeros(9)
    monkeypatch.setitem(sys.modules, "run71", v6)
    W = {"eng": "eng_ab", "etf": "etf_ab", "panel": "p_ab", "dip": "d_ab", "bars": "b_ab", "settle": "s_ab",
         "costs": "costs"}
    keep = dict(W)
    with A0.harness(W) as w:
        assert w is W and W["eng"] is eng and (W["etf"], W["panel"], W["bars"], W["settle"], W["dip"]) == (
            "etf0", "panel0", "bars0", "settle0", ["dip0"])
        assert W["costs"] == "costs" and v6.NFHI.tolist() == [True] * 4 and H.halt_paths() == ["h2019.csv"]
    assert W == keep and len(v6.NFHI) == 9 and not v6.NFHI.any() and H.halt_paths() is None
