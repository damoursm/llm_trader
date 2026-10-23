"""AUDIT151's corrections as the simulator's defaults (2026-10-19; the user: "Yes please do 1 and 2") — each pinned on
small synthetic inputs: lendability AT THE TRADE (``lend_intra``), the legs' REAL quoted spreads (``quote_costs``),
the dip book's attainable open fills (``dip_open``), the dividends (``dividends``), the ETF arm's stock-era tickers
(``etf_reused``), production's halt guard (``halts``); `Mimic.off()` and `mimic.PRE151` keep the construction before."""
from __future__ import annotations

import pickle
import types

import numpy as np
import pandas as pd
import pytest

from src.backtest import mimic as M

ET = "America/New_York"


def ns(s):
    return int(pd.Timestamp(s, tz=ET).value)


# ── the defaults ──────────────────────────────────────────────────────────────

def test_every_correction_is_on_by_default_and_off_in_the_earlier_constructions():
    m, o, pre = M.Mimic(), M.Mimic.off(), M.Mimic(**M.PRE151)
    assert (m.halts, m.lend_intra, m.etf_reused, m.quote_costs, m.dip_open, m.dividends) == \
        ("guard", M.LEND_INTRA, False, True, True, True)
    for x in (o, pre):
        assert (x.halts, x.lend_intra, x.etf_reused, x.quote_costs, x.dip_open, x.dividends) == \
            (None, None, True, False, False, False)
    assert M.LEND_INTRA == (0.65, 0.92, 0.98, 0.80, 0.964)
    assert M.etf_exclusion(m) == M.vix_etps() | M.REUSED_ETF and M.etf_exclusion(pre) == M.vix_etps()
    fees = {s.name: s.fees for s in M.live_strategies(m)}
    assert all(fees[n] == "engine" for n in ("vol", "etf", "fall", "dip")) and M.quoted(m) and not M.quoted(pre)
    assert {s.name: s.fees for s in M.live_strategies(pre)}["vol"] is None
    for bad in ((0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 1.5), (0.5, 0.5, 0.5, 0.5, 0.0)):
        with pytest.raises(ValueError):
            M.Mimic(lend_intra=bad)
    with pytest.raises(ValueError):
        M.Mimic(dividends=1)


# ── lendability at the trade ──────────────────────────────────────────────────

def _eng(days, pools):
    n = len(days)
    C = pd.DataFrame({"dn": [M.dnum(d) for d in days], "tkn": [f"T{i}" for i in range(n)],
                      "bod": np.zeros(n, int), "actual": np.full(n, 10.0)})
    eng = types.SimpleNamespace(st=types.SimpleNamespace(C=C))

    def lend_mask(at="pick"):
        eng._B_pick = (np.ones(n, bool), np.asarray(pools, float), np.zeros(n), np.zeros(n))
        return np.ones(n, bool)
    eng.lend_mask = lend_mask
    return eng


def test_an_archive_days_lendable_short_survives_by_its_morning_pool_and_a_set_b_draw_by_its_own_rate():
    n = 3000
    days = ["2023-05-01"] * n                                             # set A: IBKR's archived morning file
    small = _eng(days, np.full(n, 1e5))
    big = _eng(days, np.full(n, 5e6))
    kw = {"house": False}
    one = np.ones(n)
    s = M.lend_all(small, M.Mimic(seed=0, **kw), p_lend=one).mean()
    b = M.lend_all(big, M.Mimic(seed=0, **kw), p_lend=one).mean()
    assert abs(s - 0.65) < 0.03 and abs(b - 0.98) < 0.015                 # by the pool bucket
    assert M.lend_all(small, M.Mimic(seed=0, lend_intra=None, **kw), p_lend=one).all()   # the morning file alone
    again = M.lend_all(_eng(days, np.full(n, 1e5)), M.Mimic(seed=0, **kw), p_lend=one)
    assert (again == M.lend_all(_eng(days, np.full(n, 1e5)), M.Mimic(seed=0, **kw), p_lend=one)).all()   # keyed
    setb = _eng(["2025-03-03"] * n, np.full(n, np.nan))
    p = np.full(n, 1.0)                                                     # every draw lendable on the morning model
    got = M.lend_all(setb, M.Mimic(seed=0, lend_persist=0.0, lend_day_lam=0.0, **kw), p_lend=p).mean()
    assert abs(got - 0.80) < 0.03
    own = _eng(["2026-10-01"] * 50, np.full(50, 1e5))                       # our own per-tick archive: untouched
    from src.data.deep import borrow_history as BH
    orig = BH.lendable_at
    BH.lendable_at = lambda *a, **k: (True, {"available": 1e4})
    try:
        assert M.lend_all(own, M.Mimic(seed=0, **kw), p_lend=np.ones(50)).all()
    finally:
        BH.lendable_at = orig


def test_a_turn_or_an_etf_check_on_an_archive_day_is_judged_at_the_trade(monkeypatch):
    from src.data.deep import borrow_history as BH
    monkeypatch.setattr(BH, "lendable_at", lambda t, when, price, min_usd=None, **kw: (True, {"available": 100.0}))
    f = M.lendable_at_trade(M.Mimic(seed=0))
    when = pd.Timestamp("2023-05-01 15:00", tz="UTC")
    kept = np.mean([f(f"N{i}", when, 10.0)[0] for i in range(3000)])          # a $1k pool: the smallest bucket
    assert abs(kept - 0.65) < 0.03
    assert all(f(f"N{i}", pd.Timestamp("2025-03-03 15:00", tz="UTC"), 10.0)[0] for i in range(200))   # set B: the draw
    assert M.lendable_at_trade(M.Mimic(lend_intra=None)) is BH.lendable_at
    monkeypatch.setattr(BH, "lendable_at", lambda *a, **k: (False, {}))
    assert f("X", when, 10.0)[0] is False                                    # never lendable where the file was not


# ── the legs' real quoted spreads ─────────────────────────────────────────────

def _store(tmp_path):
    d = tmp_path / "studies" / "audit151" / "quote_costs"
    d.mkdir(parents=True)
    rows = [("vol", "AAA", "in", ns("2023-03-06 11:00"), 30.0), ("vol", "AAA", "out", ns("2023-03-08 10:00"), 900.0),
            ("vol", "ZZZ", "in", 1, 10.0), ("vol", "ZZZ", "out", 2, 50.0), ("vol", "ZZY", "in", 3, None),
            ("dip", "MEGA", "in", ns("2023-03-06 09:30") + 2, 4.0), ("dip", "MEGA", "out", ns("2023-03-10 09:30"), 6.0),
            ("crash", "TQQQ", "in", 5, 2.0), ("crash", "TQQQ", "out", 6, 2.0)]
    pd.DataFrame(rows, columns=["arm", "tkn", "leg", "ns", "hs_bp"]).assign(at="x").to_csv(d / "quotes.csv",
                                                                                          index=False)
    from src.backtest import quotes as Q
    return Q.QuoteTable(str(tmp_path))


def test_a_market_leg_pays_its_own_quote_a_missing_one_its_arms_median_and_a_resting_limit_none(tmp_path):
    qt = _store(tmp_path)
    assert qt.med[("vol", "in")] == 20.0 and qt.med[("vol", "out")] == 475.0       # NaN rows never count
    short = {"strategy": "vol", "tkn": "AAA", "ens": ns("2023-03-06 11:00"), "xns": ns("2023-03-08 10:00")}
    q = qt.apply(dict(short, hs_in=23.0, hs_out=23.0), short, -1, True)
    assert (q["hs_in"], q["hs_out"]) == (30.0, 400.0) and "q_fb" not in q           # capped at the broker's 400 bp
    miss = dict(short, tkn="BBB")
    q = qt.apply(dict(miss, hs_in=23.0, hs_out=23.0), miss, -1, True)
    assert (q["hs_in"], q["hs_out"], q["q_fb"]) == (20.0, 400.0, 3)              # the arm's medians, flagged
    dip = {"strategy": "dip", "tkn": "MEGA", "ord_k": 0, "ens": ns("2023-03-06 09:30") + 2,
           "xns": ns("2023-03-10 09:30")}
    q = qt.apply(dict(dip, hs_in=1.0, hs_out=1.0), dip, 1, False)
    assert (q["hs_in"], q["hs_out"]) == (4.0, 6.0)                                   # an open fill crosses
    touch = dict(dip, ens=ns("2023-03-06 10:30"))
    q = qt.apply(dict(touch, hs_in=1.0, hs_out=1.0), touch, 1, False)
    assert q["hs_in"] == 0.0 and q["hs_out"] == 6.0                                  # a touched resting limit: none
    crash = {"strategy": "crash", "tkn": "TQQQ", "ens": 5, "xns": 6}
    assert qt.apply(dict(crash, hs_in=9.0, hs_out=9.0), crash, 1, True)["hs_in"] == 2.0


def test_the_traded_legs_that_fell_back_are_listed_for_the_fetch():
    from src.backtest import quotes as Q
    P = [{"strategy": "vol", "tkn": "BBB", "ens": 100, "xns": 200, "q_fb": 3, "q_ens": 100, "q_xns": 200},
         {"strategy": "vol", "tkn": "CCC", "ens": 300, "xns": 400, "q_fb": 2, "q_ens": 300, "q_xns": 400},
         {"strategy": "dip", "tkn": "MEGA", "ens": 950, "xns": 990, "q_fb": 1, "q_ens": 900, "q_xns": 990}]
    pos = [{"strategy": "vol", "tkn": "BBB", "entry_ns": 100}, {"strategy": "dip", "tkn": "MEGA", "entry_ns": 950}]
    got = Q.traded_misses(P, pos)
    assert got == [{"arm": "vol", "tkn": "BBB", "leg": "in", "ns": 100}, {"arm": "vol", "tkn": "BBB", "leg": "out",
                                                                          "ns": 200},
                   {"arm": "dip", "tkn": "MEGA", "leg": "in", "ns": 900}]            # the store's key, not the repriced


# ── the dip book's attainable open fills ──────────────────────────────────────

def test_an_open_fill_takes_ibkrs_attainable_price_or_waits_for_its_touch(tmp_path, monkeypatch):
    from config import settings
    from src.backtest.strategies import dip as DIP
    d = tmp_path / "studies" / "audit151" / "dip_open"
    d.mkdir(parents=True)
    pd.DataFrame([("MEGA", "2023-03-06", 98.0, "arrival", 97.6, None, True),
                  ("MEGA", "2023-03-07", 98.0, "touch", 98.0, "10:41:00", True),
                  ("ODD", "2023-03-06", 50.0, "arrival", 49.0, None, False)],
                 columns=["tkn", "day", "limit", "a30.how", "a30.price", "a30.touch_min", "terms_ok"]).to_csv(
        d / "fills_all.csv", index=False)
    monkeypatch.setattr(settings, "backtest_research_dir", str(tmp_path))
    DIP._OPEN.clear()
    pt = np.array([ns("2023-03-06 10:00"), ns("2023-03-06 10:30"), ns("2023-03-06 11:00")], np.int64)
    p = {"strategy": "dip", "tkn": "MEGA", "ord_lim": 98.0, "ens": ns("2023-03-06 09:30") + 2, "e": 97.5,
         "pt": pt, "pc": np.array([97.0, 98.0, 99.0])}
    log = {}
    q = DIP.reprice_open(p, log)
    assert q["e"] == 97.6 and q["ens"] == ns("2023-03-06 09:30:30") and len(q["pt"]) == 3
    t = dict(p, ens=ns("2023-03-07 09:30") + 2, pt=pt + 86400 * 10 ** 9)
    q = DIP.reprice_open(t, log)
    assert q["e"] == 98.0 and q["ens"] == ns("2023-03-07 11:00") and len(q["pt"]) == 0 and len(q["pc"]) == 0
    other = dict(p, ord_lim=97.0)                                            # another order's limit: not this row
    assert DIP.reprice_open(other, log) is other
    odd = dict(p, tkn="ODD", ord_lim=50.0)
    assert DIP.reprice_open(odd, log) is odd                                 # the stores' terms disagree: the print
    touch_fill = dict(p, ens=ns("2023-03-06 10:30"))
    assert DIP.reprice_open(touch_fill, log) is touch_fill                   # only an open fill is repriced
    assert log == {"dip_open_arrival": 1, "dip_open_touch": 1, "dip_open_unknown": 1, "dip_open_terms": 1}
    DIP._OPEN.clear()


# ── the dividends ─────────────────────────────────────────────────────────────

def test_a_short_pays_and_a_long_receives_every_ex_date_it_held_through(tmp_path, monkeypatch):
    from config import settings
    from src.backtest import dividends as DV
    st = int(pd.Timestamp("2023-03-07 20:00", tz=ET).tz_convert("UTC").value)
    tab = {"ABC": {"ex": ["2023-03-08"], "cash": [0.50], "start_ns": [st], "prev": ["2023-03-07"], "s": [1.0]}}
    d = tmp_path / "t.pkl"
    pickle.dump(tab, open(d, "wb"))
    monkeypatch.setattr(DV, "TABLE", d)
    monkeypatch.setattr(settings, "backtest_research_dir", str(tmp_path))
    DV._T.clear()
    monkeypatch.setattr(DV, "raw_close", lambda day, sym: None)
    pt = np.array([ns("2023-03-07 15:00"), ns("2023-03-08 10:00")], np.int64)
    short = {"strategy": "vol", "tkn": "ABC", "ens": ns("2023-03-06 11:00"), "xns": ns("2023-03-09 10:00"),
             "e": 20.0, "x": 18.0, "pt": pt, "pc": np.array([19.0, 18.5]), "ph": np.array([19.2, 18.8]),
             "pl": np.array([18.9, 18.2])}
    log = []
    q = DV.adjust(short, log)
    assert q["x"] == 18.5 and list(q["pc"]) == [19.0, 19.0] and q["_div"] == 0.5 and len(log) == 1
    long_ = dict(short, strategy="dip")
    assert DV.adjust(long_)["x"] == 18.5                                   # a long's value grows by what it is owed
    late = dict(short, ens=st + 1)                                          # entered at/after the instant
    assert DV.adjust(late) is late
    early = dict(short, xns=st - 1)                                         # out before it
    assert DV.adjust(early) is early
    assert DV.adjust(dict(short, strategy="research_x")) is not None
    DV._T.clear()
