"""The ETF dual-momentum adapter (`src/backtest/strategies/dm.py`): the live decision on daily closes, holdings bought
and sold at the month's first open, a held pick kept as one piece, cash months, 30-minute marks (daily where the store
has none), the decision blind to later closes, and the book in one account with the dip book (the long side is
shared)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import account as A
from src.backtest.strategies import dm as DM

ET = "America/New_York"
DAYS = pd.bdate_range("2023-01-02", "2025-06-30")


def frame(growth: float, seed: int = 0) -> pd.DataFrame:
    c = 100.0 * (1 + growth) ** np.arange(len(DAYS))
    return pd.DataFrame({"Open": c * 0.999, "High": c * 1.002, "Low": c * 0.997, "Close": c, "Volume": 2e7},
                        index=DAYS)


def data(**g) -> dict:
    base = {"QQQ": 0.001, "SPY": 0.0005, "IWM": 0.0, "TLT": -0.0005, "GLD": 0.0002}
    base.update(g)
    return {f: frame(v) for f, v in base.items()}


NOBARS = lambda f: None                                            # noqa: E731  (daily marks)
RF = lambda a, b: 0.03                                             # noqa: E731


def test_the_month_holds_the_top_fund_above_tbills_and_cash_below():
    plan = DM.schedule("2024-03-01", "2024-05-31", data=data(), tbill=RF)
    assert [m["trade_day"] for m in plan] == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-01"),
                                              pd.Timestamp("2024-05-01")]
    assert all(m["pick"] == "QQQ" for m in plan)
    assert plan[0]["decided_at"] == pd.Timestamp("2024-02-29")      # the last session before the month began
    flat = {f: 0.00005 for f in DM.FUNDS}                           # ~1.3% a year: under the 3% T-bill
    assert all(m["pick"] is None for m in DM.schedule("2024-03-01", "2024-05-31", data=data(**flat), tbill=RF))


def test_a_new_pick_sells_at_the_months_first_open_and_buys_the_new_fund():
    d = data()
    # GLD overtakes QQQ from April 2024 on
    g = d["GLD"]["Close"].copy()
    late = DAYS >= pd.Timestamp("2024-01-02")
    g[late] = g[late] * (1.004 ** np.arange(late.sum()))
    d["GLD"] = d["GLD"].assign(Close=g, Open=g * 0.999, Low=g * 0.997, High=g * 1.002)
    plan = DM.schedule("2024-02-01", "2024-06-28", data=d, tbill=RF)
    picks = [m["pick"] for m in plan]
    assert picks[0] == "QQQ" and picks[-1] == "GLD"
    P = DM.pieces("2024-02-01", "2024-06-28", data=d, tbill=RF, bars=NOBARS)
    assert [p["tkn"] for p in P] == ["QQQ", "GLD"]
    switch = next(m["trade_day"] for m in plan if m["pick"] == "GLD")
    q, gl = P
    assert q["xns"] == gl["ens"] == pd.Timestamp(f"{switch.date()} 09:30", tz=ET).value
    assert q["x"] == pytest.approx(d["QQQ"].loc[switch, "Open"]) and gl["e"] == pytest.approx(d["GLD"].loc[switch, "Open"])
    assert gl["xns"] == pd.Timestamp("2024-06-28 16:00", tz=ET).value          # still held at the end: sold at its close
    assert gl["x"] == pytest.approx(d["GLD"].loc["2024-06-28", "Close"])


def test_an_unchanged_pick_is_one_piece_and_a_cash_month_holds_nothing():
    P = DM.pieces("2024-02-01", "2024-06-28", data=data(), tbill=RF, bars=NOBARS)
    assert len(P) == 1 and P[0]["tkn"] == "QQQ"
    assert P[0]["ens"] == pd.Timestamp("2024-02-01 09:30", tz=ET).value
    # daily marks: one 16:00 close a session, from the entry session to the end
    assert len(P[0]["pt"]) == len(DAYS[(DAYS >= "2024-02-01") & (DAYS <= "2024-06-28")])
    flat = {f: 0.00005 for f in DM.FUNDS}
    assert DM.pieces("2024-02-01", "2024-06-28", data=data(**flat), tbill=RF, bars=NOBARS) == []


def test_a_holding_is_marked_on_its_30_minute_bars_inside_its_hold():
    d = data()
    days = DAYS[(DAYS >= "2024-02-01") & (DAYS <= "2024-03-29")]
    idx, rows = [], []
    for t in days:
        for k in range(13):
            s = pd.Timestamp(f"{t.date()} 09:30", tz=ET) + pd.Timedelta(minutes=30 * k)
            idx.append(s.tz_convert("UTC").tz_localize(None))
            c = float(d["QQQ"].loc[t, "Close"]) * (1 + 0.0001 * k)
            rows.append((c, c * 1.001, c * 0.998, c, 1e5))
    b = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))
    P = DM.pieces("2024-02-01", "2024-03-29", data=d, tbill=RF, bars=lambda f: b if f == "QQQ" else None)
    (p,) = P
    assert p["e"] == pytest.approx(b["Open"].iloc[0])
    assert len(p["pt"]) == 13 * len(days) and p["pt"][0] == pd.Timestamp("2024-02-01 10:00", tz=ET).value
    assert p["ens"] < p["pt"][0] and p["pt"][-1] <= p["xns"]
    assert p["x"] == pytest.approx(b["Close"].iloc[-1])
    assert p["hs_in"] == 1.0                                       # 20M shares x ~$100: $1B+ a day, 1 bp


def test_the_decision_never_reads_a_close_after_its_month_end():
    d = data()
    base = DM.schedule("2024-02-01", "2024-06-28", data=d, tbill=RF)
    cut = pd.Timestamp("2024-04-30")                               # May's decision close
    wild = {f: v.copy() for f, v in d.items()}
    for f in wild:
        wild[f].loc[wild[f].index > cut, ["Open", "High", "Low", "Close"]] *= 7.0 if f == "TLT" else 0.2
    again = DM.schedule("2024-02-01", "2024-06-28", data=wild, tbill=RF)
    keep = [m for m in base if m["decided_at"] <= cut]
    assert [(m["pick"], m["scores"]) for m in again[:len(keep)]] == [(m["pick"], m["scores"]) for m in keep]
    assert again[-1]["pick"] == "TLT" != base[-1]["pick"]           # the rewrite reaches June's decision (the leak test)


def test_in_one_account_the_dual_momentum_book_and_the_dip_book_share_the_long_side():
    t0 = pd.Timestamp("2024-03-01 09:30", tz=ET).value
    day = 86400 * 10**9

    def piece(strategy, tkn, e, n_days, t=t0):
        return {"strategy": strategy, "tkn": tkn, "ens": t, "xns": t + n_days * day, "e": e, "x": e,
                "hs_in": 1.0, "hs_out": 1.0, "dv20": 1e10, "pt": np.array([t + k * day for k in range(1, n_days)]),
                "pc": np.full(n_days - 1, e), "pl": np.full(n_days - 1, e)}
    P = [piece("dm", "QQQ", 400.0, 20), piece("dip", "AAPL", 200.0, 5, t0 + day)]
    r = A.simulate(P, [A.dm_long(priority=2), A.dip_long(priority=3)], A.Rules(), keep_skips=True)
    dm = next(p for p in r["positions"] if p["strategy"] == "dm")
    assert dm["shares"] * 400.0 > 9_800.0                           # the whole account in the fund
    assert not any(p["strategy"] == "dip" for p in r["positions"])  # no long room left for the dip book
    assert [s["reason"] for s in r["skips"]] == ["side_cap"]


def test_after_a_margin_call_the_book_rebuys_its_fund_at_the_next_open():
    d = data()
    P_dm = DM.pieces("2024-03-01", "2024-05-31", data=d, tbill=RF, bars=NOBARS)
    (h,) = P_dm                                                    # QQQ, March 1 .. May 31 close
    # the fund takes the whole account; a 1/24 short beside it squeezes 10x on March 12: with the fund's 25%
    # maintenance the account is margin-called that day (equity ~$6,300 against ~$10,700 required), not ruined
    t0 = pd.Timestamp("2024-03-01 09:30", tz=ET).value
    sq = pd.Timestamp("2024-03-12 16:00", tz=ET).value
    short = {"strategy": "S", "tkn": "ZZZ", "ens": t0, "xns": pd.Timestamp("2024-03-20 09:30", tz=ET).value,
             "e": 10.0, "x": 10.0, "hs_in": 1.0, "hs_out": 1.0, "dv20": 1e9,
             "pt": np.array([sq]), "pc": np.array([100.0]), "pl": np.array([100.0])}
    st = [A.dm_long(priority=0), A.vol_short("S", slices=24, priority=1)]
    first = A.simulate(P_dm + [short], st, A.Rules())
    assert first["calls"] and [p["how"] for p in first["positions"] if p["strategy"] == "dm"] == ["margin_call"]
    extra = DM.after_calls(first, P_dm, end="2024-05-31", data=d, bars=NOBARS)
    assert len(extra) == 1 and extra[0]["tkn"] == "QQQ"
    assert extra[0]["ens"] == pd.Timestamp("2024-03-13 09:30", tz=ET).value and extra[0]["xns"] == h["xns"]
    res = DM.simulate_rejoining(P_dm + [short], st, A.Rules(), end="2024-05-31", data=d, bars=NOBARS)
    dm = [p for p in res["positions"] if p["strategy"] == "dm"]
    assert [p["how"] for p in dm][0] == "margin_call" and len(dm) == 2
    assert dm[1]["entry_ns"] == extra[0]["ens"] and dm[1]["exit_ns"] == h["xns"]


def test_open_marks_price_each_later_session_at_its_open():
    d = data()
    (p,) = DM.pieces("2024-03-01", "2024-03-08", data=d, tbill=RF, bars=NOBARS, open_marks=True)
    days = DAYS[(DAYS >= "2024-03-01") & (DAYS <= "2024-03-08")]
    assert len(p["pt"]) == 2 * len(days) - 1                       # a close a session + an open after the first
    assert np.all(np.diff(p["pt"]) > 0)
    k = list(p["pt"]).index(pd.Timestamp("2024-03-04 09:30", tz=ET).value)
    assert p["pc"][k] == pytest.approx(d["QQQ"].loc["2024-03-04", "Open"])
    (q,) = DM.pieces("2024-03-01", "2024-03-08", data=d, tbill=RF, bars=NOBARS)
    assert len(q["pt"]) == len(days)                               # off by default
