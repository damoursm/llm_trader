"""PREREG121 — the simulator's overnight session (`Mimic(overnight=...)`, IBKR's OVERNIGHT venue from the night of
2023-07-16): production's overnight slots read the IBKR bar ending at them; a name whose overnight prices disagree with
the store's 19:30-20:00 close is another security; "marks" marks every position there, "exits" also judges the short
arms' exits there (an exit needs the bar's dollar volume); before the venue existed nothing changes. Synthetic bars
only — no research store, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import mimic as M
from tests.test_backtest_mimic import WINDOWS, bars_of, day_bars, ns

NIGHTS = "20:30-23:30@60,01:00-03:00@60,03:30-03:30"


@pytest.fixture(autouse=True)
def production_slots(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "trade", raising=False)
    monkeypatch.setattr(settings, "overnight_hours_mode", "trade", raising=False)
    monkeypatch.setattr(settings, "extended_windows", WINDOWS, raising=False)
    monkeypatch.setattr(settings, "overnight_windows", NIGHTS, raising=False)
    monkeypatch.setattr(settings, "intraday_session_start", "09:30", raising=False)
    monkeypatch.setattr(settings, "intraday_session_end", "16:00", raising=False)
    M.off_ticks.cache_clear()
    M.night_slots.cache_clear()
    yield
    M.off_ticks.cache_clear()
    M.night_slots.cache_clear()


def write_night(tmp_path, sym, nights, price=lambda night, m: 10.0, volume=1000.0):
    """IBKR overnight bars (bar STARTS 20:00 .. 03:30 ET, UTC-naive ``ts``) of the nights starting on ``nights``."""
    import duckdb
    rows = []
    for d in nights:
        for m in list(range(1200, 1440, 30)) + list(range(0, 240, 30)):
            day = pd.Timestamp(d) + (pd.Timedelta(days=1) if m < 240 else pd.Timedelta(0))
            t = pd.Timestamp(f"{day.date()} {m // 60:02d}:{m % 60:02d}", tz="America/New_York")
            c = float(price(d, m))
            rows.append((t.tz_convert("UTC").tz_localize(None), c, c * 1.01, c * 0.99, c, float(volume)))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    out = tmp_path / "bars30m"
    out.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"copy (select * from df) to '{(out / f'{sym}.parquet').as_posix()}' (format parquet)")
    con.close()
    return out


def test_the_overnight_slots_are_productions():
    got = [f"{m // 60:02d}:{m % 60:02d}" for m in M.night_slots()]
    assert got == ["01:00", "02:00", "03:00", "03:30", "20:30", "21:30", "22:30", "23:30"]


def test_a_night_reads_the_ibkr_bar_ending_at_each_slot_in_the_stores_terms(tmp_path):
    D = "2024-01-09"
    store = day_bars([D, "2024-01-10"])                                  # the store's closes: 10.0
    d = write_night(tmp_path, "AAA", [D], price=lambda n, m: 10.0 + (m if m >= 1200 else m + 1440) / 1e4)
    nb = M.NightBars(bars_of("AAA", store), d=str(d))
    n = nb.get("AAA")
    assert n is not None and "AAA" not in nb.rejected
    T, px, hi, lo, usd = M.night_marks(n, ns(D, "16:00"), ns("2024-01-10", "09:30"))
    got = [str(pd.Timestamp(t, unit="ns", tz="UTC").tz_convert("America/New_York").time())[:5] for t in T]
    assert got == ["20:30", "21:30", "22:30", "23:30", "01:00", "02:00", "03:00", "03:30"]
    k = 10.0 + 1200 / 1e4                                                 # the store-check ratio (the 20:00 bar's open)
    assert px[0] == pytest.approx((10.0 + 1200 / 1e4) / (k / 10.0))      # the 20:00-20:30 bar's close, rescaled
    assert np.all(usd == pytest.approx(1000.0 * np.array(
        [10.0 + (m if m >= 1200 else m + 1440) / 1e4 for m in (1200, 1260, 1320, 1380, 1470, 1530, 1590, 1620)])))
    T2, *_ = M.night_marks(n, ns(D, "21:30"), ns(D, "23:30"))            # strictly inside the trade
    assert [str(pd.Timestamp(t, unit="ns", tz="UTC").tz_convert("America/New_York").time())[:5] for t in T2] == ["22:30"]


def test_a_name_whose_nights_disagree_with_the_store_is_another_security(tmp_path):
    D = "2024-01-09"
    store = day_bars([D, "2024-01-10"])
    d = write_night(tmp_path, "AAA", [D], price=lambda n, m: 20.0)       # twice the store's price every night
    nb = M.NightBars(bars_of("AAA", store), d=str(d))
    assert nb.get("AAA") is None and "another security" in nb.rejected["AAA"]


def test_nights_of_another_price_regime_are_dropped_not_rescaled(tmp_path, monkeypatch):
    """A corporate action one source adjusted and the other did not (WDC's spin-off): IBKR's nights at 0.75x the store's
    before it, 1.0x after. One scale per name (1.0, the majority) read the early nights as a -25% dip every night; the
    regime check drops them, and the first nights after the change until most of the last five checks agree."""
    days = [str(d.date()) for d in pd.bdate_range("2024-01-08", periods=12)]
    store = day_bars(days + ["2024-01-24"])                               # the store's closes: 10.0
    early = set(days[:4])
    d = write_night(tmp_path, "AAA", days, price=lambda n, m: 7.5 if n in early else 10.0)
    nb = M.NightBars(bars_of("AAA", store), d=str(d))
    n = nb.get("AAA")
    assert n is not None and nb.dropped["AAA"] == 6                      # the 4 early nights + 2 of the transition
    kept = sorted({str((pd.Timestamp(t, unit="ns", tz="UTC").tz_convert("America/New_York") - pd.Timedelta(hours=8))
                       .date()) for t in n["start"]})
    assert kept == days[6:]
    assert np.allclose(n["c"], 10.0)                                       # nothing left at the other regime's prices
    # a single scale would have kept every night: the early ones at 7.5 in the store's terms
    monkeypatch.setattr(M.NightBars, "_regime_nights",
                        staticmethod(lambda start, et, first, ok, rel: np.ones(len(start), bool)))
    nb2 = M.NightBars(bars_of("AAA", store), d=str(d))
    assert np.isclose(nb2.get("AAA")["c"].min(), 7.5)


def test_a_stale_check_drops_the_night(tmp_path):
    """A night whose newest check is older than `NIGHT_REGIME_DAYS` has no reliable scale: dropped."""
    days = ["2024-01-09", "2024-03-05"]                                   # eight weeks apart
    store = day_bars(["2024-01-09", "2024-01-10"])                        # only the first night can be checked
    d = write_night(tmp_path, "AAA", days)
    n = M.NightBars(bars_of("AAA", store), d=str(d)).get("AAA")
    assert n is not None
    first = pd.Timestamp(n["start"].min(), unit="ns", tz="UTC").tz_convert("America/New_York")
    last = pd.Timestamp(n["start"].max(), unit="ns", tz="UTC").tz_convert("America/New_York")
    assert str(first.date()) == "2024-01-09" and str(last.date()) == "2024-01-10"


def test_nothing_before_the_venue_existed(tmp_path):
    D = "2023-07-10"                                                      # a week before the venue's first night
    store = day_bars([D, "2023-07-11"])
    d = write_night(tmp_path, "AAA", [D])
    assert M.NightBars(bars_of("AAA", store), d=str(d)).get("AAA") is None


def overnight_short(D0, D1, e=10.0, lvl=8.0):
    pt = np.array([ns(D0, "15:30"), ns(D0, "16:00"), ns(D1, "10:00")], np.int64)
    pc = np.array([10.0, 10.0, 7.5])
    return {"strategy": "vol", "tkn": "AAA", "ens": ns(D0, "15:00"), "xns": ns(D1, "10:00"), "e": e, "x": 7.5,
            "hs_in": 20.0, "hs_out": 20.0, "dv20": 1e7, "pt": pt, "pc": pc, "ph": pc * 1.01,
            "pa": np.array([20.0, 20.0, 18.0]), "apk": 20.0, "lvl": lvl, "vn": 0.5, "cover_mult": 6.0,
            "why": "target", "d0": 0, "cum": np.cumsum(np.full(60, 1.0)), "bfac": 1.0, "full_ps": 99.0, "pick_day": 0}


def night_path(n, m):
    return 10.0 if m in (1200,) else (7.9 if m == 1260 else 9.0)          # the 21:00-21:30 bar crosses the target


def test_exits_cover_a_short_overnight_at_the_target_when_the_bar_traded_enough(tmp_path):
    D0, D1 = "2024-01-09", "2024-01-10"
    store = day_bars([D0, D1], price=lambda d, m: 10.0 if d == D0 or m < 570 else 7.5)
    eb = bars_of("AAA", store)
    d = write_night(tmp_path, "AAA", [D0], price=night_path, volume=1000.0)          # $7,900 traded at 21:30
    q, why = M.night_short(overnight_short(D0, D1), M.NightBars(eb, d=str(d)), eb, lambda x: x + 1, "exits")
    assert why == "night_exit" and q["why"] == "night_target"
    assert q["xns"] == ns(D0, "21:30") and q["x"] == pytest.approx(7.9)
    assert all(t < q["xns"] for t in q["pt"]) and ns(D0, "20:30") in set(q["pt"])
    (tmp_path / "thin").mkdir()
    thin = write_night(tmp_path / "thin", "AAA", [D0], price=night_path, volume=10.0)
    q2, why2 = M.night_short(overnight_short(D0, D1), M.NightBars(eb, d=str(thin)), eb, lambda x: x + 1, "exits")
    assert why2 == "night_marks" and q2["xns"] == ns(D1, "10:00")          # $79 traded: no overnight exit
    q3, why3 = M.night_short(overnight_short(D0, D1), M.NightBars(eb, d=str(d)), eb, lambda x: x + 1, "marks")
    assert why3 == "night_marks" and q3["xns"] == ns(D1, "10:00") and ns(D0, "21:30") in set(q3["pt"])


def test_a_long_gets_the_overnight_marks_only(tmp_path):
    D0, D1 = "2024-01-09", "2024-01-10"
    store = day_bars([D0, D1])
    eb = bars_of("AAA", store)
    d = write_night(tmp_path, "AAA", [D0], price=lambda n, m: 9.0)
    p = {"strategy": "dip", "tkn": "AAA", "ens": ns(D0, "10:00"), "xns": ns(D1, "09:30"), "e": 10.0, "x": 10.0,
         "pt": np.array([ns(D0, "15:30"), ns(D0, "16:00")], np.int64), "pc": np.array([10.0, 10.0]), "ord_k": 0}
    q, why = M.night_long(p, M.NightBars(eb, d=str(d)), eb)
    assert why == "night_marks" and q["xns"] == p["xns"] and len(q["pt"]) == 2 + 8
    assert list(q["pt"]) == sorted(q["pt"])


def test_the_mode_is_checked_and_off_by_default():
    assert M.Mimic().overnight == "off" and M.Mimic.off().overnight == "off"
    with pytest.raises(ValueError):
        M.Mimic(overnight="all")


# ── X1 / C1: the after-hours tick acting on the close ────────────────────────

def long_piece(D0, D_exit, e=10.0, x=11.0):
    pt = np.array([ns(D0, "10:30"), ns(D0, "16:00"), ns(D_exit, "09:30")], np.int64)
    return {"strategy": "dip", "tkn": "AAA", "ens": ns(D0, "10:00"), "xns": ns(D_exit, "09:30"), "e": e, "x": x,
            "hs_in": 1.0, "hs_out": 1.0, "dv20": 2e9, "pt": pt, "pc": np.array([10.0, 10.0, 10.0]),
            "pl": np.array([10.0, 10.0, 10.0]), "ord_k": 0, "days": 1.0}


def test_x1_sells_at_the_after_hours_tick_after_the_triggering_close():
    D0, D1 = "2024-01-09", "2024-01-10"
    store = day_bars([D0, D1], price=lambda d, m: 10.5 if (d == D0 and m == 960) else 10.0)   # 16:00-16:30 bar: 10.5
    q, why = M.pm_exit(long_piece(D0, D1), bars_of("AAA", store), cost_bp=lambda side, px, t, adv: 29.0)
    assert why == "pm_exit" and q["xns"] == ns(D0, "16:30") and q["x"] == pytest.approx(10.5)
    assert q["hs_out"] == 29.0 and all(t < q["xns"] for t in q["pt"])
    gone = day_bars([D0, D1], drop={(D0, "16:00")})                                             # no after-hours print
    assert M.pm_exit(long_piece(D0, D1), bars_of("AAA", gone))[1] == "no_pm_bar"
    p = long_piece(D0, D1)
    p["xns"] = ns(D1, "16:00")                                                                  # a last close, no open
    assert M.pm_exit(p, bars_of("AAA", store))[1] == "not_an_open"


def test_c1_buys_the_crash_fund_at_its_signal_sessions_after_hours_tick():
    D0, D1, D2 = "2024-01-09", "2024-01-10", "2024-01-11"
    store = day_bars([D0, D1, D2], price=lambda d, m: 9.5 if (d == D0 and m == 960) else 10.0)
    p = long_piece(D1, D2)
    p.update(strategy="crash", ens=ns(D1, "09:30"), e=10.0)
    p["pt"] = np.array([ns(D1, "10:00"), ns(D1, "16:00"), ns(D2, "09:30")], np.int64)
    q, why = M.pm_entry(p, bars_of("AAA", store))
    assert why == "pm_entry" and q["ens"] == ns(D0, "16:30") and q["e"] == pytest.approx(9.5)
    assert q["xns"] == p["xns"] and p["ens"] == ns(D1, "09:30")                                 # the input untouched


# ── E1 / E2: the dip book's limit orders outside regular hours ───────────────

class _Panel(dict):
    bars_fn = None


def _ext_from(rth: pd.DataFrame, touch=None, usd_volume=1000.0):
    """ExtBars' arrays: the regular-hours bars ``rth`` plus flat pre-market (04:00-09:30) and after-hours (16:00-20:00)
    bars at the previous regular close; ``touch``: {(day, "HH:MM"): (low, volume)} for a bar to reach down."""
    touch = touch or {}
    rows = []
    starts = pd.DatetimeIndex(rth.index).tz_localize("UTC").tz_convert("America/New_York")
    by_day = {}
    for t, row in zip(starts, rth.itertuples()):
        by_day.setdefault(t.normalize(), []).append((t, row))
    last = None
    for day in sorted(by_day):
        for m in list(range(240, 570, 30)) + list(range(960, 1200, 30)):
            hhmm = f"{m // 60:02d}:{m % 60:02d}"
            ref = last if m < 570 else by_day[day][-1][1].Close
            if ref is None:
                continue
            lo_, vol, *cl = touch.get((str(day.date()), hhmm), (ref, usd_volume))
            t = pd.Timestamp(f"{day.date()} {hhmm}", tz="America/New_York").tz_convert("UTC").value
            rows.append((t, ref, ref, lo_, cl[0] if cl else ref, vol, False))
        for t, row in by_day[day]:
            rows.append((t.tz_convert("UTC").value, row.Open, row.High, row.Low, row.Close, row.Volume, True))
        last = by_day[day][-1][1].Close
    rows.sort()
    a = np.array([r[0] for r in rows], np.int64)
    return {"start": a, "end": a + M.BAR_NS, "o": np.array([r[1] for r in rows]), "h": np.array([r[2] for r in rows]),
            "l": np.array([r[3] for r in rows]), "c": np.array([r[4] for r in rows]), "v": np.array([r[5] for r in rows]),
            "rth": np.array([r[6] for r in rows])}


class _Ext:
    def __init__(self, m):
        self.m = m

    def get(self, name):
        return self.m.get(name)


def _setup(rebound: bool = False):
    from src.backtest.strategies import dip as DIP
    from src.signals import dip_long as dl
    from src.signals import sel_short as ss
    from tests.test_dip_filter import DAY, _bars_of, _daily
    days = ss.sessions_before(DAY, 300)
    daily = _daily(days, seed=1, drop_last=0, dips=(250,))
    if rebound:                                   # the session after the signal closes 8% up, never near the limit
        k = 252
        prev = float(daily["c"].iloc[k - 1])
        daily.iloc[k, daily.columns.get_loc("o")] = prev * 1.02
        daily.iloc[k, daily.columns.get_loc("l")] = prev * 1.01
        daily.iloc[k, daily.columns.get_loc("c")] = prev * 1.08
        daily.iloc[k, daily.columns.get_loc("h")] = prev * 1.09
    rth = _bars_of(daily)
    panel = _Panel({"AAA": dl.daily_bars(rth)})
    panel.bars_fn = lambda tk: rth
    lo, hi = days[230], days[-2]
    S = DIP.signals(panel, DIP.LIVE, lo, hi, ["AAA"])
    r = S.iloc[0]
    d = panel["AAA"]
    i = int(d.index.get_loc(r.sig))
    lim = dl.limit_price(r.close, dl.atr14(d.iloc[:i + 1]))
    return DIP, panel, rth, lo, hi, r, d, i, lim


def test_e1_fills_a_dip_order_in_the_pre_market_when_the_bar_traded_enough():
    DIP, panel, rth, lo, hi, r, d, i, lim = _setup()
    D = str(d.index[i + 1].date())
    xb = _ext_from(rth, touch={(D, "07:00"): (lim - 0.01, 1000.0)})
    P = [p for p in DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=panel, ext=_Ext({"AAA": xb}), names=["AAA"])
         if p["oid"] == f"AAA|{D}" and p["ord_k"] >= 0]
    first = min(P, key=lambda p: p["ens"])
    assert first["fill_src"] == "ext" and first["ens"] == ns(D, "07:30") and first["e"] == pytest.approx(lim)
    assert first["ord_place"][0] == ns(str(d.index[i].date()), "16:30") + 1     # placed after the signal close
    orig = [p for p in DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA"]) if p["oid"] == f"AAA|{D}"]
    assert all(p["ens"] >= first["ens"] for p in orig if p["ord_k"] >= 0)
    thin = _ext_from(rth, touch={(D, "07:00"): (lim - 0.01, 10.0)})             # ~$1,000 traded: no fill there
    P2 = [p for p in DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=panel, ext=_Ext({"AAA": thin}), names=["AAA"])
          if p["oid"] == f"AAA|{D}" and p["ord_k"] >= 0]
    assert all(p["ens"] != ns(D, "07:30") for p in P2)


def test_e1_an_after_hours_fill_counts_its_exit_from_the_next_close():
    DIP, panel, rth, lo, hi, r, d, i, lim = _setup(rebound=True)
    D = str(d.index[i + 1].date())
    xb = _ext_from(rth, touch={(D, "17:00"): (lim - 0.01, 1000.0)})
    rth_touch = any(p["ord_k"] >= 0 for p in DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA"])
                    if p["oid"] == f"AAA|{D}" and pd.Timestamp(p["ens"], unit="ns", tz="UTC").tz_convert(
                        "America/New_York").date() == pd.Timestamp(D).date())
    P = [p for p in DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=panel, ext=_Ext({"AAA": xb}), names=["AAA"])
         if p["oid"] == f"AAA|{D}" and p["ens"] == ns(D, "17:30")]
    if rth_touch:
        pytest.skip("the synthetic day's regular session already reaches the limit")
    assert len(P) == 1 and P[0]["fill_src"] == "ext"
    assert all(t > ns(D, "17:30") for t in P[0]["pt"])                           # marked from the next session
    assert pd.Timestamp(P[0]["pt"][0], unit="ns", tz="UTC").tz_convert("America/New_York").date() > pd.Timestamp(D).date()
    # D's own close (+8%, above its 5-session average) is past at the fill: the exit is judged from D+1's close on,
    # never D+1's open (which counting from D would give)
    assert P[0]["xns"] > ns(str(d.index[i + 2].date()), "09:30")


def test_the_late_print_bar_is_the_trfs_report_window():
    """Overnight trades executed 00:00-04:00 are stamped when the FINRA TRF opens: 08:00 ET to 2026-03-27, 04:00 ET
    from 2026-03-30 (PREREG121 amendment 2)."""
    def bar(day, hhmm):
        return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC").value
    got = M.late_print_bar([bar("2023-08-01", "08:00"), bar("2026-03-27", "08:00"), bar("2026-03-27", "04:00"),
                            bar("2026-03-30", "04:00"), bar("2026-03-30", "08:00"), bar("2026-06-17", "04:30")])
    assert got.tolist() == [True, True, False, True, False, False]


def test_e1_the_late_print_bar_fills_only_by_its_close():
    """The TRF's late prints set that bar's low: a low under the limit is the night's price, never a fill; its close
    under the limit fills at the close, at the bar's end."""
    DIP, panel, rth, lo, hi, r, d, i, lim = _setup()
    D = str(d.index[i + 1].date())
    late = "04:00" if pd.Timestamp(D) >= M.TRF_04 else "08:00"
    end = (pd.Timestamp(f"{D} {late}") + pd.Timedelta(minutes=30)).strftime("%H:%M")
    rth_or_later = lambda P: all(p["ens"] > ns(D, end) for p in P)               # noqa: E731

    def pieces(touch):
        return [p for p in DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=panel, ext=_Ext({"AAA": _ext_from(rth, touch)}),
                                                names=["AAA"]) if p["oid"] == f"AAA|{D}" and p["ord_k"] >= 0]
    assert rth_or_later(pieces({(D, late): (lim - 0.01, 1000.0)}))               # a stale low: no fill there
    other = "05:00" if late == "04:00" else "07:00"
    first = min(pieces({(D, other): (lim - 0.01, 1000.0)}), key=lambda p: p["ens"])  # any other bar's low fills
    assert first["e"] == pytest.approx(lim)
    P = pieces({(D, late): (lim - 0.01, 1000.0, lim - 0.02)})
    first = min(P, key=lambda p: p["ens"])
    assert first["ens"] == ns(D, end) and first["e"] == pytest.approx(lim - 0.02)


def test_e1_without_extended_bars_is_the_regular_hours_book():
    DIP, panel, rth, lo, hi, r, d, i, lim = _setup()
    a = DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA"])
    b = DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=panel, ext=_Ext({}), names=["AAA"])
    key = lambda P: sorted((p["oid"], p["ord_k"], p["ens"], p["xns"], p["e"], p["x"]) for p in P)   # noqa: E731
    assert key(a) == key(b)


def test_e2_mode_is_checked():
    assert M.Mimic().dip_ext_orders == "off"
    with pytest.raises(ValueError):
        M.Mimic(dip_ext_orders="night")
