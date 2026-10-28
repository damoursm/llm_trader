"""The backtest made to mimic production (src/backtest/mimic.py, PREREG62; user 2026-10-09: "Apply all of the simulator
fixes. It needs to mimic production as best as possible."): production's off-grid ticks on the stored extended bars,
the shorts' exits there, the price alignment, the live entry order, the ledger's leg costs, the engine's per-strategy
fee model. Synthetic bars and pieces only — no research store, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import account as A
from src.backtest import mimic as M

ET = "America/New_York"
WINDOWS = "04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-19:00@60,19:50-19:50"


@pytest.fixture(autouse=True)
def production_slots(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "trade", raising=False)
    monkeypatch.setattr(settings, "overnight_hours_mode", "off", raising=False)
    monkeypatch.setattr(settings, "extended_windows", WINDOWS, raising=False)
    monkeypatch.setattr(settings, "intraday_session_start", "09:30", raising=False)
    monkeypatch.setattr(settings, "intraday_session_end", "16:00", raising=False)
    M.off_ticks.cache_clear()
    yield
    M.off_ticks.cache_clear()


def ns(day, hhmm):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET).value


def day_bars(days, price=lambda day, m: 10.0, drop=()):
    """Every 30-minute bar 04:00-20:00 ET of ``days`` (ExtBars' arrays); ``price(day, start minute)`` the close, the
    open / high / low around it; ``drop``: (day, "HH:MM" bar start) left out (no trade)."""
    rows = []
    for d in days:
        for m in range(240, 1200, 30):
            hhmm = f"{m // 60:02d}:{m % 60:02d}"
            if (d, hhmm) in drop:
                continue
            c = float(price(d, m))
            rows.append((ns(d, hhmm), c, c * 1.01, c * 0.99, c, 570 <= m < 960,
                         (pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days))
    a = np.array([r[0] for r in rows], np.int64)
    return {"start": a, "end": a + M.BAR_NS, "o": np.array([r[1] for r in rows]), "h": np.array([r[2] for r in rows]),
            "l": np.array([r[3] for r in rows]), "c": np.array([r[4] for r in rows]),
            "rth": np.array([r[5] for r in rows]), "day": np.array([r[6] for r in rows], np.int64)}


def bars_of(name, b):
    eb = M.ExtBars()
    eb._m[name] = b
    return eb


# ── production's off-grid ticks ──────────────────────────────────────────────

def test_the_off_grid_ticks_are_the_schedulers_extended_slots_and_the_open():
    got = [f"{m // 60:02d}:{m % 60:02d}/{k}" for m, k in M.off_ticks()]
    assert got == ["04:00/1", "05:00/1", "06:00/1", "07:00/1", "07:30/1", "08:00/1", "08:30/1", "09:00/1", "09:30/0",
                   "16:30/1", "17:00/1", "17:30/1", "18:00/1", "19:00/1", "19:50/1"]


def test_a_tick_reads_the_bar_ending_at_it_the_open_tick_the_opening_print():
    D = "2024-01-09"
    b = day_bars([D], price=lambda d, m: 100.0 + m / 60.0)
    T, px, hi, lo = M.tick_marks(b, ns(D, "00:00"), ns(D, "23:59"))
    got = [str(pd.Timestamp(t, unit="ns", tz="UTC").tz_convert(ET).time())[:5] for t in T]
    # 04:00 has no bar ending at it (the first bar ends 04:30); 19:50 reads the bar ending 19:30
    assert got == ["05:00", "06:00", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30",
                   "16:30", "17:00", "17:30", "18:00", "19:00", "19:50"]
    assert px[got.index("05:00")] == pytest.approx(100.0 + 270 / 60.0)      # the 04:30-05:00 bar's close
    assert px[got.index("09:30")] == pytest.approx(100.0 + 570 / 60.0)      # the 09:30 bar's OPEN
    assert px[got.index("19:50")] == pytest.approx(100.0 + 1140 / 60.0)     # the 19:00-19:30 bar's close


def test_a_tick_without_a_fresh_bar_is_skipped_and_ticks_stay_strictly_inside_the_trade():
    D = "2024-01-09"
    b = day_bars([D], drop={(D, "08:30"), (D, "09:30")})
    T, *_ = M.tick_marks(b, ns(D, "07:00"), ns(D, "17:00"))
    got = [str(pd.Timestamp(t, unit="ns", tz="UTC").tz_convert(ET).time())[:5] for t in T]
    assert got == ["07:30", "08:00", "08:30", "16:30"]        # no 09:00 (its bar is missing), no 09:30 open


# ── the shorts' exits at those ticks ─────────────────────────────────────────

def short_piece(D0, D1, e=10.0, lvl=8.0, x=7.5, apk=20.0, pa=None, **kw):
    pt = np.array([ns(D0, "15:30"), ns(D0, "16:00"), ns(D1, "10:00")], np.int64)
    pc = np.array([10.0, 10.0, x])
    p = {"strategy": "vol", "tkn": "AAA", "ens": ns(D0, "15:00"), "xns": ns(D1, "10:00"), "e": e, "x": x,
         "hs_in": 20.0, "hs_out": 20.0, "dv20": 1e7, "pt": pt, "pc": pc, "ph": pc * 1.01,
         "pa": np.asarray(pa if pa is not None else [20.0, 20.0, 18.0], float), "apk": apk, "lvl": lvl, "vn": 0.5,
         "cover_mult": 6.0, "why": "target", "d0": 0, "cum": np.cumsum(np.full(60, 1.0)), "bfac": 1.0,
         "full_ps": 99.0, "pick_day": 0}
    p.update(kw)
    return p


def test_a_pre_market_tick_at_the_target_covers_there_with_its_borrow_recomputed():
    D0, D1 = "2024-01-09", "2024-01-11"
    path = {("2024-01-10", 1110): 9.0, ("2024-01-11", 390): 7.9}       # 18:30-19:00 and 06:30-07:00 bars

    def price(d, m):
        if d == D0:
            return 10.0
        if d == D1 and 570 <= m < 960:
            return 7.5                                       # the regular session the old exit read
        return path.get((d, m), 9.5 if m < 570 or m >= 960 else 9.6)
    b = day_bars([D0, "2024-01-10", D1], price=price)
    p = short_piece(D0, D1)
    p["pt"] = np.r_[p["pt"][:2], ns("2024-01-10", "10:00"), ns(D1, "10:00")]
    p["pc"] = np.array([10.0, 10.0, 9.6, 7.5])
    p["ph"], p["pa"] = p["pc"] * 1.01, np.array([20.0, 20.0, 19.0, 18.0])
    q, why = M.extend_short(p, bars_of("AAA", b), settle=lambda d: d + 1)
    assert why == "exit" and q["why"] == "ext_target"
    assert q["xns"] == ns(D1, "07:00") and q["x"] == pytest.approx(7.9)
    assert all(t < q["xns"] for t in q["pt"]) and list(q["pt"]) == sorted(q["pt"])
    assert ns("2024-01-10", "19:00") in set(q["pt"])                     # the evening's ticks are marks
    d_out = int(A.day_numbers([q["xns"]])[0])
    assert q["d_out"] == d_out and q["full_ps"] == pytest.approx(float(p["cum"][min(d_out + 1 - 1, 59)]))
    assert p["xns"] == ns(D1, "10:00")                                   # the input piece is untouched


def test_the_volatility_exit_fires_off_the_grid_only_in_profit_on_the_latest_regular_hours_atr():
    D0, D1 = "2024-01-09", "2024-01-10"
    b = day_bars([D0, D1], price=lambda d, m: 10.0 if m < 960 or d == D1 else 9.9)
    # the ATR% halved at the 16:00 bar (8 <= 0.5 x 20) while the close (10.0) was not in profit: the 16:30 tick's
    # 9.9 is -> covered there
    p = short_piece(D0, D1, lvl=None, pa=[20.0, 8.0, 8.0], x=10.0)
    q, why = M.extend_short(p, bars_of("AAA", b), settle=lambda d: d + 1)
    assert why == "exit" and q["why"] == "ext_volnorm" and q["xns"] == ns(D0, "16:30")
    p2 = short_piece(D0, D1, lvl=None, pa=[20.0, 15.0, 15.0], x=10.0)      # ATR% not halved: no exit
    q2, why2 = M.extend_short(p2, bars_of("AAA", b), settle=lambda d: d + 1)
    assert why2 == "marks" and q2["xns"] == p2["xns"]


def test_the_cover_fires_off_the_grid_at_six_times_the_entry():
    D0, D1 = "2024-01-09", "2024-01-10"
    b = day_bars([D0, D1], price=lambda d, m: 61.0 if (d == D1 and m == 450) else 10.0)   # the 07:30-08:00 bar
    q, why = M.extend_short(short_piece(D0, D1, lvl=None, x=10.0), bars_of("AAA", b), settle=lambda d: d + 1)
    assert why == "exit" and q["why"] == "ext_cap" and q["xns"] == ns(D1, "08:00") and q["x"] == pytest.approx(61.0)


def test_the_stored_bars_are_aligned_to_the_pieces_prices():
    D0, D1 = "2024-01-09", "2024-01-10"
    p = short_piece(D0, D1)
    b2 = day_bars([D0, D1], price=lambda d, m: 20.0)                     # stored at twice the traded price
    p["pc"] = np.array([10.0, 10.0, 10.0])
    assert M.price_scale(p, b2) == pytest.approx(0.5)
    p["pc"] = np.array([10.0, 10.0, 12.0])                               # the two sides disagree by 20%
    assert M.price_scale(p, b2) is None


# ── the live order, the draws, the configuration ─────────────────────────────

def test_a_bars_shorts_enter_by_ticker_then_arm():
    P = [{"strategy": "vol", "tkn": "BBB@X"}, {"strategy": "fall", "tkn": "AAA"}, {"strategy": "vol", "tkn": "AAA@D"},
         {"strategy": "etf", "tkn": "AAB@E"}]
    M.rank_shorts(P)
    order = [f"{p['tkn']}/{p['strategy']}" for p in sorted(P, key=lambda p: p["rank"])]
    assert order == ["AAA@D/vol", "AAA/fall", "AAB@E/etf", "BBB@X/vol"]


def test_draws_are_deterministic_and_house_rates_come_from_the_quoted_names(monkeypatch):
    assert M.u01("lend", "AAA", 19000, 0) == M.u01("lend", "AAA", 19000, 0) != M.u01("lend", "AAA", 19000, 1)
    u = np.array([M.u01("x", k) for k in range(20000)])
    assert 0.0 <= u.min() and u.max() < 1.0 and abs((u < 0.07).mean() - 0.07) < 0.01
    monkeypatch.setattr(M, "house_mix", lambda: (("X", 2.0, 2.86), ("Y", 41.1, 45.2)))
    seen = {M.house_rates(f"T{k}@X", 19000, 0) for k in range(200)}
    assert seen == {(2.0, 2.86), (41.1, 45.2)}
    assert M.house_rates("T1@X", 19000, 0) == M.house_rates("T1", 19000, 0)      # the traded symbol decides


def test_the_configuration_switches():
    assert M.Mimic().label() == "MIMIC" and M.Mimic.off().label() == "OLD"
    with pytest.raises(ValueError):
        M.Mimic(r201=False)                                  # entry at the pick bar needs Rule 201 off
    assert M.short_extra(M.Mimic()) == {"entry_at_pick": True, "rule201": False}
    assert M.short_extra(M.Mimic().without("entry")) == {"rule201": False}
    assert M.short_extra(M.Mimic.off()) == {}
    r, r0 = M.live_rules(M.Mimic()), M.live_rules(M.Mimic.off())
    assert (r.trigger, r.fees, r.mark_net, r.one_book_per_name) == ("high", "ledger", True, True)
    assert (r0.trigger, r0.fees, r0.mark_net, r0.liquidation) == ("high", "engine", False, "all")
    r1 = M.live_rules(M.Mimic(**M.PRE174))                   # the margin calls before PREREG174: at the tick marks
    assert (r1.trigger, r1.liquidation, r1.sem, r1.regt_close) == ("close", "all", 0.0, False)
    S = {s.name: s for s in M.live_strategies(M.Mimic())}
    assert {S[n].priority for n in ("vol", "etf", "fall")} == {0} and S["dip"].fees == "engine"
    assert S["vol"].shrink_budget == pytest.approx(float(settings.sel_short_account_house_init))
    S0 = {s.name: s for s in M.live_strategies(M.Mimic.off())}
    assert (S0["vol"].priority, S0["etf"].priority, S0["fall"].priority) == (0, 1, 2) and S0["vol"].shrink_budget is None



def test_margin_calls_are_ibkrs_liquidation_by_default():
    """PREREG174 (the user 2026-10-21: the backtest does what IBKR does at a margin call): the default judges the margin
    on the bars' extremes and liquidates as IBKR documents it — the most recent positions first, in 100-share lots,
    only back to the maintenance requirement —, strict (no Soft Edge Margin), with the end-of-day Reg T floor."""
    r = M.live_rules(M.Mimic())
    assert (r.trigger, r.liquidation, r.liq_order, r.liq_lot, r.sem, r.regt_close, r.liq_fill, r.liq_slip) == (
        "high", "ibkr", "recent", 100, 0.0, True, "path", pytest.approx(0.10))
    assert M.live_rules(M.Mimic(margin_fill="extreme")).liq_fill == "extreme"
    assert M.live_rules(M.Mimic(margin_check="ticks")).trigger == "close"
    assert M.live_rules(M.Mimic(margin_sem=0.10)).sem == pytest.approx(0.10)
    assert M.live_rules(M.Mimic(margin_order="requirement")).liq_order == "requirement"
    assert M.live_rules(M.Mimic(margin_regt=False)).regt_close is False
    # with the earlier rule the trigger switch decides where it is judged, as before
    assert M.live_rules(M.Mimic(margin="all")).trigger == "close"
    assert M.live_rules(M.Mimic(margin="all").without("trigger")).trigger == "high"
    for bad in ({"margin": "some"}, {"margin_check": "intrabar"}, {"margin_order": "worst"}, {"margin_sem": 1.0},
                {"margin_sem": True}, {"margin_regt": 1}, {"margin_fill": "worst"}, {"margin_slip": 2.0}):
        with pytest.raises(ValueError):
            M.Mimic(**bad)

# ── the ledger's costs ───────────────────────────────────────────────────────

@pytest.fixture
def clean_costs():
    from src.performance import spread
    yield spread
    spread.set_real_cost_override(None)
    spread.set_cost_attribution(None, None, None)


def test_a_leg_costs_the_ledgers_session_and_liquidity_class_average(monkeypatch, clean_costs):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True, raising=False)
    monkeypatch.setattr(settings, "sim_real_fill_min_price", 1.0, raising=False)
    snap = {"frac": 0.0022, "by_session": {"rth": 0.0015, "extended": 0.0029, "overnight": 0.0055},
            "session_costs": {"rth": 0.0020, "premarket": 0.0034, "afterhours": 0.0029},
            "session_bucket": {"rth|1|1": 0.0018, "premarket|0|0": 0.0041}}
    lc = M.LedgerCosts(snap)
    D = "2024-01-09"
    assert lc.bp("SELL", 50.0, ns(D, "11:00"), 3e7) == pytest.approx(18.0)        # rth, $20-100, >= $20M
    assert lc.bp("BUY", 10.0, ns(D, "07:00"), 1e6) == pytest.approx(41.0)         # premarket, < $20, < $20M
    assert lc.bp("BUY", 10.0, ns(D, "17:00"), 1e6) == pytest.approx(29.0)         # afterhours: the session's mean
    q = lc.apply({"e": 50.0, "x": 10.0, "ens": ns(D, "11:00"), "xns": ns(D, "07:00"), "dv20": 3e7}, -1)
    assert q["hs_in"] == pytest.approx(18.0) and q["hs_mark"] == pytest.approx(18.0)


def test_the_engine_charges_each_strategy_its_own_cost_model():
    T0 = pd.Timestamp("2024-01-08 15:00", tz="UTC").value
    DAY = 86400 * 10**9

    def pc(strategy, tkn, e, x, hs):
        return {"strategy": strategy, "tkn": tkn, "ens": T0, "xns": T0 + 2 * DAY, "e": e, "x": x, "hs_in": hs,
                "hs_out": hs, "dv20": 1e9, "pt": [T0 + DAY], "pc": [(e + x) / 2]}
    S = A.vol_short("S")
    L = A.Strategy("L", 1, slices=10, fees="engine")
    r = A.simulate([pc("S", "AAA", 50.0, 45.0, 20.0), pc("L", "BBB", 100.0, 110.0, 1.0)], [S, L],
                   A.Rules(fees="ledger"), settle=lambda d: int(d) + 1)
    pos = {p["tkn"]: p for p in r["positions"]}
    n = pos["AAA"]["shares"]
    from src.performance.spread import _pct_return                # the live account's P&L: shares x entry x return
    ret = _pct_return("SELL", 50.0, 45.0, entry_cost=20e-4, exit_cost=20e-4) / 100.0
    assert pos["AAA"]["pnl"] == pytest.approx(n * 50.0 * ret)     # ledger: hs only, no commission or sale fees
    m = pos["BBB"]["shares"]
    cin = m * 100.0 * 1.0 / 1e4 + A.comm(m, 100.0)
    cout = m * 110.0 * 1.0 / 1e4 + A.comm(m, 110.0) + A.sale_fees(m, 110.0)
    # engine fees (exact commission and sale fees), the dollars booked as the ledger does under ledger rules
    assert pos["BBB"]["pnl"] == pytest.approx((m * 10.0 - cin - cout) / (1 + cin / (m * 100.0)))
    with pytest.raises(ValueError):
        A.Strategy("X", 1, fees="cheap")


def test_the_borrow_models_are_the_default_and_the_earlier_fills_stay_reachable():
    """PREREG67 / PREREG94: the lendability draws read the LightGBM model (v2) linked across sessions and days, and a
    short without an IBKR fee pays F4's prediction by default; PREREG67's construction and the earlier simulator stay
    reachable; an unknown model is refused."""
    from src.backtest import mimic as M
    m = M.Mimic()
    assert m.lend_model == "v2" and m.fee_model is True and m.fee_version == "v4"
    assert (m.lend_persist, m.lend_day_lam, m.lend_day_r) == (0.7, 0.15, 0.5)
    v2 = M.Mimic.borrow_v2(seed=3)
    assert (v2.lend_persist, v2.lend_day_lam, v2.fee_version, v2.seed) == (0.0, 0.0, "v2", 3)
    old = M.Mimic.off()
    assert old.lend_model == "v1" and old.fee_model is False and old.lend_persist == 0.0
    assert M.lend_p_path("v1").name == "lend_p.npy" and M.lend_p_path("v2").name == "lend_p_v2.npy"
    assert M.fee_pred_path().name == "fee_pred_v4.npy" and M.fee_pred_path("v2").name == "fee_pred_v2.npy"
    for bad in ({"lend_model": "v3"}, {"fee_version": "v9"}, {"lend_persist": 1.0}, {"lend_day_lam": -0.1}):
        with pytest.raises(ValueError):
            M.Mimic(**bad)


def test_a_lendability_shift_moves_every_draw_the_same_way():
    """PREREG68: the level of the lendability model is uncertain year to year (5-8 points); ``lend_shift`` moves the
    log-odds of every post-archive draw, so a sensitivity can run the account at a lower or higher lendable share."""
    import types
    from src.backtest import mimic as M
    n = 4000
    C = pd.DataFrame({"dn": np.full(n, M.dnum("2025-03-03")), "tkn": [f"T{i}" for i in range(n)],
                      "bod": np.zeros(n, int), "actual": np.full(n, 10.0)})
    eng = types.SimpleNamespace(st=types.SimpleNamespace(C=C), lend_mask=lambda at="pick": np.ones(n, bool))
    p = np.full(n, 0.5)
    indep = {"house": False, "lend_persist": 0.0, "lend_day_lam": 0.0,     # one day: no market-wide factor here
             "lend_intra": None}                                            # the draw model alone (no AUDIT151 loss)
    base = M.lend_all(eng, M.Mimic(**indep), p_lend=p).mean()
    lo = M.lend_all(eng, M.Mimic(lend_shift=-1.0, **indep), p_lend=p).mean()
    hi = M.lend_all(eng, M.Mimic(lend_shift=+1.0, **indep), p_lend=p).mean()
    assert lo < base < hi
    assert abs(base - 0.5) < 0.03 and abs(hi - 1 / (1 + np.exp(-1.0))) < 0.03



def _cal_eng(days):
    import types
    return types.SimpleNamespace(C4=types.SimpleNamespace(CAL=np.asarray(days, np.int64)))


def test_independent_draws_reproduce_prereg67():
    """lend_persist = lend_day_lam = 0: the draws are PREREG67's u01("lend", name, day, seed)."""
    draw = M.lend_draw_fn(_cal_eng(range(19000, 19050)), M.Mimic.borrow_v2(seed=4))
    for sym, d in (("AAA", 19001), ("BBB", 19030), ("CCC", 19999)):
        assert draw(sym, d) == M.u01("lend", sym, d, 4)


def test_linked_draws_keep_each_picks_probability_and_link_a_names_sessions():
    """PREREG94: each pick keeps its probability (the latent is uniform) while a name's lendability persists from one
    session to the next and the names move together day to day."""
    days = list(range(19000, 19300))
    draw = M.lend_draw_fn(_cal_eng(days), M.Mimic(seed=1, draws="iid"))
    names = [f"N{i}" for i in range(120)]
    U = np.array([[draw(n, d) for d in days] for n in names])                      # names x days
    assert abs((U < 0.3).mean() - 0.3) < 0.03 and abs((U < 0.7).mean() - 0.7) < 0.03
    L = U < 0.6
    agree = (L[:, 1:] == L[:, :-1]).mean()
    assert agree > 0.6 * 0.6 + 0.4 * 0.4 + 0.08                                    # independent draws: 0.52
    share = L.mean(axis=0)                                                          # the day's lendable share
    assert share.var() > 1.5 * 0.6 * 0.4 / len(names)                               # names move together
    ind = M.lend_draw_fn(_cal_eng(days), M.Mimic.borrow_v2(seed=1))
    Li = np.array([[ind(n, d) for d in days] for n in names]) < 0.6
    assert (Li[:, 1:] == Li[:, :-1]).mean() < agree and Li.mean(axis=0).var() < share.var()


def test_independent_draws_are_unchanged_by_the_stratified_option():
    """PREREG95: draws="iid" keeps every earlier draw bit for bit; stratified draws are the default from 2026-10-13
    (user: "Apply the best decisions you're recommending"), the earlier constructions stay independent."""
    assert M.Mimic().draws == "strat" and M.Mimic().strat_n == 5
    assert M.Mimic(draws="iid").strat_n is None
    assert M.Mimic.off().draws == "iid" and M.Mimic.borrow_v2().draws == "iid"
    assert M.Mimic.borrow_v2(draws="strat").draws == "strat"                         # an explicit choice wins
    for kind, sym, d, s in (("refuse", "AAA", 19001, 0), ("house", "BBB", 20100, 7), ("lend", "CCC", 20500, 3)):
        assert M.draw_u(kind, sym, d, s) == M.u01(kind, sym, d, s)
    assert M.refused("AAA@D", 19001, 2) == (M.u01("refuse", "AAA", 19001, 2) < M.REFUSAL_RATE)
    for bad in ({"draws": "lhs"}, {"draws": "strat", "n_draws": 1}):
        with pytest.raises(ValueError):
            M.Mimic(**bad)


def test_refusals_are_drawn_by_name_from_the_versioned_table_by_default(monkeypatch, tmp_path):
    """PREREG193 (the user 2026-10-24: "Make the switch right away"): `Mimic()` refuses by NAME — the refusal table's
    probability at the row (else the name's latest earlier row, else the stand-in) against ONE uniform per name —;
    `Mimic(**PRE193)` keeps the random 7% per (name, session), the rule a study's patch of `refused` decides."""
    rows = pd.DataFrame({"sym": ["AAA", "AAA", "BBB"], "dn": [20000, 20010, 20005], "p": [0.30, 0.90, 0.02]})
    monkeypatch.setattr(M, "REFUSAL_TABLES", {"v1": "t.pkl", "vt": "t.pkl"})
    monkeypatch.setattr(M, "refusal_path", lambda version="v1": tmp_path / "t.pkl")
    pd.to_pickle({"rows": rows, "p_default": 0.005}, tmp_path / "t.pkl")
    M.refusal_table.cache_clear()
    try:
        assert M.Mimic().refusal == "model" and M.Mimic(**M.PRE193).refusal == "rate"
        assert M.Mimic.off().refusal == "rate"
        assert M.refusal_p("AAA", 20000) == 0.30 and M.refusal_p("AAA", 20009) == 0.30       # the latest earlier row
        assert M.refusal_p("AAA", 20010) == 0.90 and M.refusal_p("AAA", 19999) == 0.005       # before its first row
        assert M.refusal_p("ZZZ", 20000) == 0.005                                             # no row: the stand-in
        mim = M.Mimic(seed=3)
        u = M.draw_u("refuse", "AAA", 0, 3, mim.strat_n)                                      # one uniform per name
        assert M.is_refused(mim, "AAA@D", 20000) == (u < 0.30)
        assert M.is_refused(mim, "AAA", 20010) == (u < 0.90)
        rate = M.Mimic(seed=3, **M.PRE193)
        assert M.is_refused(rate, "AAA", 20000) == M.refused("AAA", 20000, 3, n=rate.strat_n)
        monkeypatch.setattr(M, "refused", lambda *a, **k: True)                              # a study's patch
        assert M.is_refused(rate, "AAA", 20000) is True and M.is_refused(mim, "BBB", 20005) == (
            M.draw_u("refuse", "BBB", 0, 3, mim.strat_n) < 0.02)
        hit = [M.is_refused(M.Mimic(seed=s), "AAA", 20010) for s in range(200)]
        assert 0.8 < sum(hit) / 200 < 0.98                                                    # ~ its 90%
        for bad in ({"refusal": "always"}, {"refusal_table": "v9"}):
            with pytest.raises(ValueError):
                M.Mimic(**bad)
    finally:
        M.refusal_table.cache_clear()


def test_the_installed_refusal_table_is_readable_and_covers_the_stores():
    """The deployed table (v1, PREREG193's REFLIST probabilities): one row per (name, session), probabilities in [0, 1],
    its stand-in REFCAL's — skipped where the research store is absent."""
    path = M.refusal_path(M.REFUSAL_TABLE)
    if not path.exists():
        pytest.skip(f"no refusal table at {path}")
    obj = pd.read_pickle(path)
    rows = obj["rows"]
    assert len(rows) > 30000 and not rows.duplicated(["sym", "dn"]).any()
    assert rows["p"].between(0.0, 1.0).all() and 0.0 < float(obj["p_default"]) < 0.05
    assert 0.015 < float(rows["p"].mean()) < 0.04


def test_stratified_draws_put_one_draw_of_the_block_in_each_stratum():
    """PREREG95: within a block of n consecutive seeds every decision's n uniforms fall one per stratum, so a 7%
    refusal hits each (name, session) in its share of the block's draws; each draw alone stays uniform."""
    n = 20
    for key in (("refuse", "AAA", 19001), ("house", "BBB", 20100), ("lend", "CCC", 20500)):
        for block in (0, 3):
            u = [M.draw_u(*key, block * n + j, n) for j in range(n)]
            assert sorted(int(x * n) for x in u) == list(range(n))
    hits = [sum(M.draw_u("refuse", f"T{i}", 20000, j, n) < M.REFUSAL_RATE for j in range(n)) for i in range(300)]
    assert set(hits) <= {1, 2}                                     # 0.07 x 20 = 1.4
    one = np.array([M.draw_u("refuse", f"T{i}", 20000 + i % 7, 5, n) for i in range(4000)])
    assert abs(one.mean() - 0.5) < 0.02 and abs((one < 0.07).mean() - 0.07) < 0.015


def test_stratified_lend_draws_keep_each_draws_law_and_balance_the_block():
    """PREREG95 (Amendment 4): under stratified draws each single draw has the independent draws' law — uniform at
    each pick, a name's persistence and the day factor as PREREG94's — while the block's draws are balanced: the
    variance of a pick's lendable count over the block sits well below the binomial's (ranking the latent among the
    block, which balanced it exactly, shifted set B's growth +12 points and is gone)."""
    days = list(range(19000, 19200))
    n = 10
    draws = [M.lend_draw_fn(_cal_eng(days), M.Mimic(seed=10 + j, draws="strat", n_draws=n)) for j in range(n)]
    names = [f"N{i}" for i in range(80)]
    U = np.array([[[dr(nm, d) for d in days] for nm in names] for dr in draws])        # draws x names x days
    for p in (0.3, 0.6, 0.83):
        assert abs((U < p).mean() - p) < 0.02
        assert (U < p).sum(axis=0).var() < 0.75 * n * p * (1 - p)                     # balanced over the block
    L = U[4] < 0.6                                                                       # one draw: names x days
    agree = (L[:, 1:] == L[:, :-1]).mean()
    assert agree > 0.6 * 0.6 + 0.4 * 0.4 + 0.06                                        # independent: 0.52
    assert L.mean(axis=0).var() > 1.3 * 0.6 * 0.4 / len(names)                         # the day factor
    ref = M.lend_draw_fn(_cal_eng(days), M.Mimic(seed=14, draws="iid"))                              # the same law as iid's
    Lr = np.array([[ref(nm, d) for d in days] for nm in names]) < 0.6
    assert abs((Lr[:, 1:] == Lr[:, :-1]).mean() - agree) < 0.03
    off = M.lend_draw_fn(_cal_eng(days), M.Mimic(seed=10, draws="strat", n_draws=n))
    assert off("ZZZ", 25000) == M.draw_u("lend", "ZZZ", 25000, 10, n)                  # a day off the calendar
