"""PREREG227 — the parity audit's simulator corrections (2026-10-27): S1 the capitulation and fear books' open fills
at IBKR's attainable 09:30:30 price (a fill without a price that session none, one the table lacks stamped 09:30:30),
S2 the crash book's re-entry (a candidate every qualifying session), S3 the vol arm's fallback walked to the account
(the pass loop's chains, the engine's ``alt_of``, one name a day), S4 every long book's dividends, S5 the insider,
capitulation and fear books' legs at their quotes (a fill moved to a later touch without the open's spread), S6 the
account's start ($50,000, production's) read back by every NAV, S7 the copied parameters read from settings, S8 the
capitulation and fear signals keeping an exit past the data on set D. Synthetic data only — no research store, no
network."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import account as A
from src.backtest import halts as H
from src.backtest import mimic as M

ET = "America/New_York"


def ns(t: str) -> int:
    return pd.Timestamp(t, tz=ET).value


# ── the switches ─────────────────────────────────────────────────────────────

def test_the_corrections_are_on_by_default_and_pre227_restores_the_construction_before():
    m = M.Mimic()
    assert (m.open_order, m.crash_reenter, m.vol_walk, m.div_all, m.quote_all) == (True,) * 5
    assert m.start == float(settings.sel_short_account_initial)          # production's ($50,000; the tests pin 10,000)
    o = M.Mimic(**M.PRE227)
    assert (o.open_order, o.crash_reenter, o.vol_walk, o.div_all, o.quote_all, o.start) == (False,) * 5 + (10_000.0,)
    f = M.Mimic.off()
    assert (f.open_order, f.vol_walk, f.start) == (False, False, 10_000.0)
    assert M.live_rules(m).start == m.start and M.live_rules(o).start == 10_000.0
    assert M.live_rules(M.Mimic(start=50_000.0)).start == 50_000.0
    for bad in ({"start": 0}, {"start": True}, {"start": float("nan")}, {"vol_walk": 1}, {"quote_all": "yes"}):
        with pytest.raises(ValueError):
            M.Mimic(**bad)


def test_the_copied_parameters_are_productions_settings():
    """S7: the numbers the simulator held are read from settings — equal to what it held."""
    from src.backtest.strategies import etf as E
    from src.backtest.strategies import vol as V
    old = {"top_k": 1, "fallback": 5, "fresh_W": 20, "fresh_k": 1, "min_atr": 0.0, "runup_L": 5, "min_runup": 0.0,
           "max_dtc": 1.0, "min_rvol": 1.58, "min_lend_usd": 1e4, "max_fee": None, "max_pool_drop": None, "cap": 4.0,
           "volnorm_on": False, "volnorm": 0.5}
    assert {k: V.LIVE[k] for k in old} == old
    assert all(type(V.LIVE[k]) is type(v) for k, v in old.items())
    assert V.LIVE["cap"] == float(settings.sel_short_vol_cover_multiple)
    assert V.LIVE["fallback"] == int(settings.sel_short_vol_fallback_ranks)
    assert (V.FALL["cap"], V.FALL["volnorm_on"], V.FALL["volnorm"]) == (6.0, True, 0.7)
    assert V.FALL["cap"] == float(settings.sel_short_cover_multiple)
    assert V.FALL["volnorm"] == float(settings.sel_short_fall_volnorm_ratio)
    assert M.etf_rule_kw() == {"cover": 6.0, "max_dtc": 1.0, "min_rvol": 1.58}
    assert E.VOLNORM == 0.5 == float(settings.sel_short_volnorm_ratio)
    assert M.Mimic().fall_turn == (0.5, 5) == M.Mimic().etf_turn
    assert M.Mimic().fall_turn == (float(settings.sel_short_turn_atr), int(settings.sel_short_turn_sessions))


def test_the_turn_reads_productions_numbers(monkeypatch):
    monkeypatch.setattr(settings, "sel_short_turn_atr", 0.75)
    assert M.Mimic().fall_turn == (0.75, 5) == M.Mimic().etf_turn


# ── S6: the account's start ──────────────────────────────────────────────────

def _long(tkn="AAA", e=100.0, x=110.0, t0="2023-03-06 09:30", t1="2023-03-08 09:30", strategy="crash", **kw):
    p = {"strategy": strategy, "tkn": tkn, "ens": ns(t0), "xns": ns(t1), "e": e, "x": x, "hs_in": 0.0, "hs_out": 0.0,
         "dv20": 1e12, "pt": np.array([ns(t0) + 30 * 60 * 10 ** 9], np.int64), "pc": np.array([e])}
    p.update(kw)
    return p


def test_a_run_records_its_start_and_every_nav_reads_it():
    from src.backtest import bank as BK
    from src.backtest import study as STU
    S = [A.Strategy("crash", 1, slices=2.0, max_open=1, quality=0.2)]
    days = pd.bdate_range("2023-03-06", "2023-03-09")
    for start in (10_000.0, 50_000.0):
        res = A.simulate([_long()], S, A.Rules(start=start), keep_curve=True)
        assert res["start"] == start
        nav = STU.nav_daily(res, days)
        assert nav[-1] == pytest.approx(res["final"] / start)
        m = A.metrics(res, "2023-03-06", "2024-03-06")
        yrs = 366 / 365.25
        assert m["final"] == res["final"] and m["growth"] == pytest.approx((res["final"] / start) ** (1 / yrs) - 1)
        assert A.metrics(res, "2023-03-06", "2024-03-06", start=start * 2)["growth"] < 0    # an explicit start wins
    assert A.run_start({}) == 10_000.0 and A.run_start({"start": 50_000.0}) == 50_000.0
    assert A.run_start({"start": 50_000.0}, 7.0) == 7.0
    assert BK.live_start() == float(settings.sel_short_account_initial)
    rec = {"pos": [{"exit_ns": ns("2023-03-08 09:30"), "pnl": 500.0}]}
    s, how = BK.nav_of(rec, "2023-03-06")                     # a run without a start began at $10,000
    assert how == "realized" and s.iloc[-1] == pytest.approx(1.05)
    s, _ = BK.nav_of(dict(rec, start=50_000.0), "2023-03-06")
    assert s.iloc[-1] == pytest.approx(1.01)


# ── S2: the crash book buys again ────────────────────────────────────────────

def _qqq():
    days = pd.bdate_range("2022-01-03", periods=270)
    c = np.r_[np.linspace(100, 120, 255), [110, 104, 99, 95, 92, 90, 89, 93, 97, 101, 104, 106, 108, 109, 110]]
    return pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": 1e6}, index=days)


def test_a_candidate_every_qualifying_session_the_first_ones_as_before():
    from src.backtest.strategies import crash as CR
    d = _qqq()
    lo, hi = d.index[250], d.index[-1]
    one = CR.trades(lo, hi, daily=d, sym="QQQ")
    every = CR.trades(lo, hi, daily=d, sym="QQQ", every=True)
    assert len(one) >= 1 and len(every) > len(one)
    assert all(t in every for t in one)                       # the non-overlapping ones are among them, unchanged
    assert len({t["entry"] for t in every}) == len(every)     # one per qualifying session


def test_a_buy_the_account_cannot_fund_is_bought_at_the_next_qualifying_open():
    S = [A.Strategy("crash", 1, slices=1.0, max_open=1, quality=0.2)]
    first = _long(e=1e9, x=1e9, t0="2023-03-06 09:30", t1="2023-03-10 09:30")      # no share fits
    second = _long(e=100.0, x=105.0, t0="2023-03-07 09:30", t1="2023-03-10 09:30")
    res = A.simulate([first, second], S, A.Rules(start=50_000.0))
    assert [(p["entry_ns"], p["shares"]) for p in res["positions"]] == [(ns("2023-03-07 09:30"), 500)]
    held = _long(e=100.0, x=101.0, t0="2023-03-06 09:30", t1="2023-03-10 09:30")    # held: the overlap is rejected
    res = A.simulate([held, second], S, A.Rules(start=50_000.0))
    assert [p["entry_ns"] for p in res["positions"]] == [ns("2023-03-06 09:30")]


# ── S3: the vol arm's fallback walked to the account ─────────────────────────

def _short(tkn, rank, **kw):
    t0 = "2023-03-06 11:30"
    p = {"strategy": "vol", "tkn": tkn, "ens": ns(t0), "xns": ns("2023-03-07 11:30"), "e": 10.0, "x": 9.0,
         "hs_in": 0.0, "hs_out": 0.0, "dv20": 1e12, "pt": np.array([ns("2023-03-06 12:00")], np.int64),
         "pc": np.array([9.5]), "rank": float(rank), "group": tkn}
    p.update(kw)
    return p


def _strats():
    vol = A.Strategy("vol", -1, slices=20.0, quality=1.3, book="sel_short", group_cap=2, group_scope=("vol",))
    dip = A.Strategy("dip", 1, slices=2.0, max_open=2, quality=0.13, book="dip_long", preemptible=True)
    return [vol, dip]


def test_the_walk_opens_the_next_backup_when_the_account_refuses_one():
    held = _long(tkn="B1", strategy="dip", e=10.0, x=10.0, t0="2023-03-03 09:30", t1="2023-03-10 09:30")
    chain = [_short("B1", 0.000, alt_of="w1_0", alt_k=0, walk_dedupe=True),           # another book holds B1
             _short("B2", 0.001, alt_of="w1_0", alt_k=1, walk_dedupe=True),
             _short("B3", 0.002, alt_of="w1_0", alt_k=2, walk_dedupe=True)]
    res = A.simulate([held] + chain, _strats(), A.Rules(start=50_000.0, one_book_per_name=True))
    got = sorted(p["tkn"] for p in res["positions"] if p["strategy"] == "vol")
    assert got == ["B2"]                                      # the walk reached B2; B3 settled behind it
    assert [w["tkn"] for w in res["alt_walks"]] == ["B2"]
    hi = [dict(c) for c in chain]
    hi[2]["alt_hi"] = True                                    # a HIGH member: still the extras loop's after B2 opened
    res = A.simulate([held] + hi, _strats(), A.Rules(start=50_000.0, one_book_per_name=True))
    assert sorted(p["tkn"] for p in res["positions"] if p["strategy"] == "vol") == ["B2", "B3"]
    free = A.simulate(chain, _strats(), A.Rules(start=50_000.0, one_book_per_name=True))
    assert [p["tkn"] for p in free["positions"]] == ["B1"] and free["alt_walks"] == []


def test_the_walk_without_chains_changes_nothing():
    chain = [_short("B1", 0.0), _short("B2", 1.0)]
    res = A.simulate(chain, _strats(), A.Rules(start=50_000.0, one_book_per_name=True))
    assert sorted(p["tkn"] for p in res["positions"]) == ["B1", "B2"] and "alt_walks" not in res


def test_a_backup_or_extra_of_a_name_the_arm_opened_that_session_is_skipped():
    a = _short("B1", 0.0)
    later = _short("B1", 0.0, ens=ns("2023-03-06 13:00"), pt=np.array([ns("2023-03-06 13:30")], np.int64),
                   walk_dedupe=True)
    res = A.simulate([a, later], _strats(), A.Rules(start=50_000.0))
    assert len(res["positions"]) == 1
    nxt = dict(later, ens=ns("2023-03-07 10:00"), xns=ns("2023-03-08 11:30"),
               pt=np.array([ns("2023-03-07 10:30")], np.int64))
    res = A.simulate([a, nxt], _strats(), A.Rules(start=50_000.0))
    assert len(res["positions"]) == 2                         # the next session: a new day


SLOTS = (240, 300, 360, 420, 450, 480, 510, 540) + tuple(range(570, 961, 30)) + (990, 1020, 1050, 1080, 1140, 1190)
CAL = np.array([int((pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days)
                for d in pd.bdate_range("2021-03-01", "2021-03-31")], np.int64)


def _cand(sym, T, lend=True, hi=False, src="P", row=None):
    return H.Cand(src=src, j=0, pos=0, row=row if row is not None else abs(hash(sym)) % 10 ** 6, sym=sym,
                  tk=abs(hash(sym)) % 10 ** 6, T=T, close=10.0, target=8.0, lend0=lend, day=H.day_of(T), hi=hi)


def test_the_pass_loop_records_the_backups_the_walk_would_reach():
    m = H.NoHalts(H.Schedule(CAL, SLOTS, 75.0))
    T = ns("2021-03-10 11:30")
    run = H.Run(_cand("PICK", T, lend=False, src="L", row=1),
                [_cand("B0", T, lend=False, row=10), _cand("B1", T, row=11), _cand("B2", T, row=12),
                 _cand("B3", T, hi=True, row=13), _cand("B4", T, lend=False, row=14)])
    old = H.VolMachine(m, "guard", True, True, lambda c, at, px: c.lend0).run([run])
    assert [(t.cand.sym, t.role) for t in old.trades] == [("B1", "fallback"), ("B3", "extra")] and not old.chain
    new = H.VolMachine(m, "guard", True, True, lambda c, at, px: c.lend0, walk=True).run([run])
    assert [(t.cand.sym, t.role) for t in new.trades] == [("B1", "fallback"), ("B2", "alt"), ("B3", "extra")]
    assert new.chain == {11: ("w1_0", 0, False), 12: ("w1_0", 1, False), 13: ("w1_0", 2, True)}
    lendable = H.Run(_cand("PICK", T, lend=True, src="L", row=1), list(run.backups))
    vm = H.VolMachine(m, "guard", True, True, lambda c, at, px: c.lend0, walk=True).run([lendable])
    assert [(t.cand.sym, t.role) for t in vm.trades] == [("PICK", "pick"), ("B3", "extra")] and not vm.chain


def test_the_chain_is_ranked_together_at_its_heads_place():
    P = [_short("ZZZ", 0, alt_of="w9_0", alt_k=0), _short("AAA", 0, alt_of="w9_0", alt_k=1), _short("MMM", 0)]
    M.rank_shorts(P)
    r = {p["tkn"]: p["rank"] for p in P}
    assert r["ZZZ"] < r["AAA"] < r["ZZZ"] + 1 and r["MMM"] < r["ZZZ"]


# ── S1: the capitulation and fear books' open fills ──────────────────────────

def _fills(tmp_path, monkeypatch, rows, ext=True):
    from src.backtest.strategies import dip as DIP
    d = tmp_path / ("studies/prereg227/open_fills" if ext else "studies/audit151/dip_open")
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=["tkn", "day", "limit", "a30.how", "a30.price", "a30.touch_min", "terms_ok"]).to_csv(
        d / ("fills_all227.csv" if ext else "fills_all.csv"), index=False)
    monkeypatch.setattr(settings, "backtest_research_dir", str(tmp_path))
    DIP._OPEN.clear()
    return DIP


def test_the_capitulation_and_fear_open_fills_are_repriced_like_the_dip_books(tmp_path, monkeypatch):
    DIP = _fills(tmp_path, monkeypatch, [("MEGA", "2023-03-06", 98.0, "arrival", 97.6, None, True),
                                         ("MEGA", "2023-03-07", 98.0, "touch", 98.0, "10:41:00", True),
                                         ("MEGA", "2023-03-08", 98.0, "none", None, None, True)])
    pt = np.array([ns("2023-03-06 10:00"), ns("2023-03-06 10:30"), ns("2023-03-06 11:00")], np.int64)
    p = {"strategy": "fear", "tkn": "MEGA", "ord_lim": 98.0, "ens": ns("2023-03-06 09:30") + 2, "e": 97.5,
         "pt": pt, "pc": np.array([97.0, 98.0, 99.0]), "hs_in": 4.0}
    assert DIP.reprice_open(p) is p                                     # before PREREG227: the dip book's only
    kw = {"books": ("dip", "cap", "fear"), "none_drops": True, "stamp_unknown": True, "touch_free": True}
    log: dict = {}
    q = DIP.reprice_open(p, log, **kw)
    assert q["e"] == 97.6 and q["ens"] == ns("2023-03-06 09:30:30") and q["hs_in"] == 4.0
    t = dict(p, strategy="cap", ens=ns("2023-03-07 09:30") + 2, pt=pt + 86400 * 10 ** 9)
    q = DIP.reprice_open(t, log, **kw)
    assert q["ens"] == ns("2023-03-07 11:00") and q["e"] == 98.0 and q["hs_in"] == 0.0     # rested: no spread
    gone = dict(p, ens=ns("2023-03-08 09:30") + 2)
    assert DIP.reprice_open(gone, log, **kw) is None                    # never reached its limit after arriving
    assert DIP.reprice_open(gone, log, books=("dip", "cap", "fear")) is gone
    lack = dict(p, ens=ns("2023-03-09 09:30") + 2)
    q = DIP.reprice_open(lack, log, **kw)
    assert q["ens"] == ns("2023-03-09 09:30:30") and q["e"] == 97.5     # the print, at production's instant
    assert log["dip_open_none_dropped"] == 1 and log["dip_open_stamped"] == 1
    DIP._OPEN.clear()


def test_the_extended_table_is_read_once_it_exists(tmp_path, monkeypatch):
    DIP = _fills(tmp_path, monkeypatch, [("MEGA", "2023-03-06", 98.0, "arrival", 97.0, None, True)], ext=False)
    assert DIP.open_fills()[("MEGA", "2023-03-06", 98.0)][1] == 97.0
    _fills(tmp_path, monkeypatch, [("MEGA", "2023-03-06", 98.0, "arrival", 97.0, None, True),
                                   ("CAPX", "2023-03-06", 50.0, "arrival", 49.5, None, True)])
    assert ("CAPX", "2023-03-06", 50.0) in DIP.open_fills()
    own = tmp_path / "studies" / "prereg188" / "data"                    # a study's own table still wins (set D's)
    own.mkdir(parents=True)
    pd.DataFrame([("SETD", "2026-10-01", 10.0, "arrival", 9.9, None, True)],
                 columns=["tkn", "day", "limit", "a30.how", "a30.price", "a30.touch_min", "terms_ok"]).to_csv(
        own / "fills188.csv", index=False)
    monkeypatch.setattr(DIP, "OPEN_FILLS", DIP.Path("studies") / "prereg188" / "data" / "fills188.csv")
    assert set(DIP.open_fills()) == {("SETD", "2026-10-01", 10.0)}
    DIP._OPEN.clear()


def test_at_one_open_the_dip_book_fills_first_then_the_capitulation_book_then_the_fear_book():
    def order(strategy, prio):
        return {"strategy": strategy, "tkn": strategy.upper(), "ens": ns("2023-03-06 09:30:30"),
                "xns": ns("2023-03-08 09:30"), "e": 100.0, "x": 101.0, "hs_in": 0.0, "hs_out": 0.0, "dv20": 1e12,
                "pt": np.array([ns("2023-03-06 10:00")], np.int64), "pc": np.array([100.0]), "rank": 0.0,
                "oid": f"{strategy}|1", "ord_place": [ns("2023-03-06 09:30") + 1], "ord_lim": 100.0,
                "ord_end": ns("2023-03-06 16:00") + 1, "ord_k": 0}
    S = [A.Strategy(n, 1, slices=1.0, max_open=1, priority=M.PRIORITY[n], quality=M.QUALITY[n], orders="fill",
                    preemptible=n != "dip", yield_whole=True) for n in ("dip", "cap", "fear")]
    res = A.simulate([order("fear", 9), order("cap", 7), order("dip", 6)], S,
                     A.Rules(start=50_000.0, fills_make_room=True, fill_room_exact=True))
    filled = [(o["tkn"], o["n"]) for o in res["orders"] if o["what"] == "filled"]
    assert filled[0][0] == "DIP" and [t for t, _ in filled] == ["DIP", "CAP", "FEAR"][:len(filled)]


# ── S4 / S5: dividends and quotes ────────────────────────────────────────────

def test_every_long_book_receives_its_dividends(tmp_path, monkeypatch):
    from src.backtest import dividends as DV
    st = int(pd.Timestamp("2023-03-07 20:00", tz=ET).tz_convert("UTC").value)
    tab = {"ABC": {"ex": ["2023-03-08"], "cash": [0.50], "start_ns": [st], "prev": ["2023-03-07"], "s": [1.0]}}
    d = tmp_path / "t.pkl"
    pickle.dump(tab, open(d, "wb"))
    monkeypatch.setattr(DV, "TABLE", d)
    monkeypatch.setattr(settings, "backtest_research_dir", str(tmp_path))
    DV._T.clear()
    monkeypatch.setattr(DV, "raw_close", lambda day, sym: None)
    base = {"tkn": "ABC", "ens": ns("2023-03-06 11:00"), "xns": ns("2023-03-09 10:00"), "e": 20.0, "x": 18.0,
            "pt": np.array([ns("2023-03-08 10:00")], np.int64), "pc": np.array([18.5])}
    for book in ("cap", "ins", "fear"):
        p = dict(base, strategy=book)
        assert DV.adjust(p) is p                                         # AUDIT151's construction: no dividend
        assert DV.adjust(p, longs=DV.ALL_LONG)["x"] == 18.5
    assert DV.ALL_LONG == {"dip", "crash", "cap", "ins", "fear"}
    DV._T.clear()


def test_the_insider_capitulation_and_fear_legs_pay_their_quotes(tmp_path):
    d = tmp_path / "studies" / "audit151" / "quote_costs"
    d.mkdir(parents=True)
    rows = [("dip", "MEGA", "in", ns("2023-03-06 09:30") + 2, 4.0), ("dip", "MEGA", "out", ns("2023-03-10 09:30"), 6.0),
            ("ins", "SMOL", "in", ns("2023-03-06 09:30"), 30.0), ("ins", "SMOL", "out", ns("2023-06-01 09:30"), 40.0)]
    pd.DataFrame(rows, columns=["arm", "tkn", "leg", "ns", "hs_bp"]).assign(at="x").to_csv(d / "quotes.csv", index=False)
    from src.backtest import quotes as Q
    qt = Q.QuoteTable(str(tmp_path))
    cap = {"strategy": "cap", "tkn": "MEGA", "ord_k": 0, "ens": ns("2023-03-06 09:30") + 2, "xns": ns("2023-03-10 09:30")}
    assert qt.apply(dict(cap, hs_in=1.0, hs_out=1.0), cap, 1, False)["hs_in"] == 1.0      # before: the flat 1 bp
    q = qt.apply(dict(cap, hs_in=1.0, hs_out=1.0), cap, 1, False, limit_arms=("dip", "cap", "fear"))
    assert (q["hs_in"], q["hs_out"]) == (4.0, 6.0)
    ins = {"strategy": "ins", "tkn": "SMOL", "ens": ns("2023-03-06 09:30"), "xns": ns("2023-06-01 09:30")}
    assert qt.apply(dict(ins, hs_in=23.0, hs_out=23.0), ins, 1, True)["hs_in"] == 23.0    # before: the ledger's
    q = qt.apply(dict(ins, hs_in=23.0, hs_out=23.0), ins, 1, True, market_arms=("crash", "ins"))
    assert (q["hs_in"], q["hs_out"]) == (30.0, 40.0) and "q_fb" not in q


# ── S8: the capitulation and fear signals on set D ───────────────────────────

def test_a_capitulation_signal_whose_exit_lies_past_the_data_is_kept_on_set_d(monkeypatch):
    from src.backtest.strategies import cap as CAP
    from src.backtest.strategies import dip as DIP
    days = pd.bdate_range("2022-01-03", periods=240)
    c = np.r_[np.linspace(200, 150, 236), [140, 130, 120, 110]]          # under its average, RSI2 low at the end
    panel = {"BIG": pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": 1e8}, index=days)}
    monkeypatch.setattr(CAP, "gate_days", lambda theta=None: pd.DatetimeIndex(days))
    prm = CAP.params()
    fn = CAP.signals_fn()
    S = fn(panel, prm, days[225], days[-1], names=["BIG"])
    assert len(S) and not S["last_close"].any()                          # sets A0 / A / B: the exit inside the data
    monkeypatch.setattr(DIP, "OPEN_END", True)
    S2 = fn(panel, prm, days[225], days[-1], names=["BIG"])
    assert len(S2) > len(S) and S2["last_close"].any()
