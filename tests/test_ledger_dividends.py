"""Dividends in the ledger (AUDIT151, 2026-10-19 — `src/performance/dividends.py`): a live book's long receives and
its short pays every ex-date it held through (opened before 20:00 ET of the session before the ex-date, still held
then), Polygon's cash in the trade's terms, frozen on the trade as a fraction of its entry when first due; the return,
the daily NAV and so the live account carry it."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from config.settings import settings
from src.performance import dividends as dv
from src.performance.spread import _pct_return

UTC = timezone.utc


@pytest.fixture
def table(monkeypatch):
    """A live table (the module's own cache bypassed) and no split after any ex-date unless a test says so."""
    monkeypatch.setattr(settings, "enable_ledger_dividends", True)
    t = {"MSFT": (["2026-11-19"], [0.98]), "TQQQ": (["2026-12-22"], [0.12])}
    monkeypatch.setattr(dv, "table", lambda: t)
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda tk, a, b, rows=None: 1.0)
    return t


def _trade(**kw):
    t = {"ticker": "MSFT", "action": "BUY", "type": "STOCK", "status": "OPEN", "entry_price": 400.0,
         "entry_mechanism": "dip_long", "entry_datetime": "2026-11-17T15:00:00+00:00"}
    t.update(kw)
    return t


NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
START = "2026-11-19T01:00:00+00:00"          # 20:00 ET of 2026-11-18, the session before the 11-19 ex-date


def test_the_loader_sums_usd_cash_per_ex_date_and_maps_class_shares(tmp_path, monkeypatch):
    import duckdb
    monkeypatch.setattr(settings, "enable_ledger_dividends", True)
    p = tmp_path / "divs.parquet"
    rows = [("AAPL", "2026-11-12", 0.27, "USD", "CD"), ("AAPL", "2026-11-12", 0.27, "USD", "CD"),   # one record twice
            ("AAPL", "2026-11-12", 0.10, "USD", "SC"),                                             # a special: summed
            ("AAPL", "2026-08-10", 0.26, "USD", "CD"), ("XYZ", "2026-11-12", 0.50, "CAD", "CD"),
            ("ZZZ", "2026-11-12", 0.0, "USD", "CD"), ("OLD", "2019-03-01", 1.0, "USD", "CD"),
            ("BRK.B", "2026-11-13", 0.05, "USD", "CD")]
    con = duckdb.connect()
    con.execute("CREATE TABLE d (ticker VARCHAR, ex_dividend_date VARCHAR, cash_amount DOUBLE, currency VARCHAR, "
                "dividend_type VARCHAR)")
    con.executemany("INSERT INTO d VALUES (?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY d TO '{p.as_posix()}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setattr(dv, "DIV_PATH", p)
    dv.reset()
    t = dv.table()
    assert t["AAPL"] == (["2026-08-10", "2026-11-12"], [0.26, pytest.approx(0.37)])
    assert "XYZ" not in t and "ZZZ" not in t and "OLD" not in t          # not USD, no cash, before FROM
    assert dv._symbol("BRK-B") == "BRK.B" and t["BRK.B"][1] == [0.05]


def test_a_long_receives_and_a_short_pays_from_the_entitlement_instant(table):
    long_ = _trade()
    assert dv.update(long_, "2026-11-19T00:59:00+00:00", now=NOW) == 0 and dv.frac_through(long_) == 0.0
    assert dv.update(long_, START, now=NOW) == 1
    ev = long_["dividends"][0]
    assert ev["ex"] == "2026-11-19" and ev["at"] == START and ev["frac"] == pytest.approx(0.98 / 400.0)
    assert dv.update(long_, NOW.isoformat(), now=NOW) == 0                  # recorded once
    base = _pct_return("BUY", 400.0, 410.0, entry_cost=0.0, exit_cost=0.0)
    got = _pct_return("BUY", 400.0, 410.0, entry_cost=0.0, exit_cost=0.0, dividend=dv.frac_through(long_))
    assert got - base == pytest.approx(0.245)                               # 0.98 / 400 = +0.245%
    short = _trade(action="SELL", entry_mechanism="sel_short")
    dv.update(short, NOW.isoformat(), now=NOW)
    s_base = _pct_return("SELL", 400.0, 390.0, entry_cost=0.0, exit_cost=0.0)
    s_got = _pct_return("SELL", 400.0, 390.0, entry_cost=0.0, exit_cost=0.0, dividend=dv.frac_through(short))
    assert s_got - s_base == pytest.approx(-0.245)                          # the payment in lieu
    assert dv.frac_through(long_, "2026-11-18T23:00:00+00:00") == 0.0       # not due by an earlier instant


def test_entered_after_the_instant_or_closed_before_it_is_not_entitled(table):
    late = _trade(entry_datetime="2026-11-19T01:30:00+00:00")               # 20:30 ET: dated to the ex-date
    assert dv.update(late, NOW.isoformat(), now=NOW) == 0
    early_exit = _trade(status="CLOSED", exit_datetime="2026-11-18T20:59:00+00:00")   # sold 15:59 ET the day before
    assert dv.update(early_exit, NOW.isoformat(), now=NOW) == 0
    held = _trade(status="CLOSED", exit_datetime="2026-11-19T14:31:00+00:00")         # sold on the ex-date
    assert dv.update(held, NOW.isoformat(), now=NOW) == 1
    same_day = _trade(entry_datetime="2026-11-18T19:00:00+00:00")           # bought 14:00 ET the day before: entitled
    assert dv.update(same_day, NOW.isoformat(), now=NOW) == 1


def test_a_split_after_the_ex_date_brings_the_cash_into_the_trades_terms(table, monkeypatch):
    from src.data import intraday_store
    seen = []

    def k(tk, after, through, rows=None):
        seen.append((tk, after, through))
        return 0.5                                                          # a 2-for-1 split after the ex-date
    monkeypatch.setattr(intraday_store, "split_factor_between", k)
    t = _trade(entry_price=200.0)                                           # the entry already in today's terms
    dv.update(t, NOW.isoformat(), now=NOW)
    assert seen == [("MSFT", date(2026, 11, 19), date(2026, 11, 25))]
    assert t["dividends"][0]["cash_ps"] == pytest.approx(0.49) and t["dividends"][0]["frac"] == pytest.approx(0.49 / 200)
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda *a, **k: None)
    t2 = _trade()
    assert dv.update(t2, NOW.isoformat(), now=NOW) == 0                     # unreadable split data: the event waits


def test_only_the_live_books_and_only_when_switched_on(table, monkeypatch):
    legacy = _trade(entry_mechanism="rank_rule")
    assert dv.update(legacy, NOW.isoformat(), now=NOW) == 0 and dv.frac_through(legacy) == 0.0
    t = _trade()
    dv.update(t, NOW.isoformat(), now=NOW)
    monkeypatch.setattr(settings, "enable_ledger_dividends", False)
    assert dv.frac_through(t) == 0.0 and dv.update(_trade(), NOW.isoformat(), now=NOW) == 0


def test_never_an_instant_that_has_not_happened(table):
    t = _trade()
    assert dv.update(t, "2026-12-31T00:00:00+00:00", now=datetime(2026, 11, 18, 12, 0, tzinfo=UTC)) == 0


def test_the_tracker_marks_and_closes_with_it_and_a_late_record_reaches_a_recent_close(table, monkeypatch):
    from src.performance import tracker
    t = _trade()
    monkeypatch.setattr(dv, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NOW),
                                                       "fromisoformat": staticmethod(datetime.fromisoformat),
                                                       "combine": staticmethod(datetime.combine)}))
    assert tracker._dividend(t, NOW.isoformat(), record=True) == pytest.approx(0.98 / 400)
    closed = _trade(status="CLOSED", exit_datetime="2026-11-20T15:00:00+00:00")
    table["MSFT"] = (["2026-11-19"], [0.98])
    assert dv.finalize_closed([closed], now=NOW) == 1                       # the record arrived after the exit
    old = _trade(status="CLOSED", exit_datetime="2026-11-10T15:00:00+00:00",
                 entry_datetime="2026-11-03T15:00:00+00:00")
    table["MSFT"] = (["2026-11-05"], [0.5])
    assert dv.finalize_closed([old], now=NOW) == 0                          # exited 15 days ago: final


def test_the_daily_nav_books_it_on_the_ex_date(table, monkeypatch):
    from src.performance import daily_nav
    closes = {date(2026, 11, 17): 400.0, date(2026, 11, 18): 401.0, date(2026, 11, 19): 400.0,
              date(2026, 11, 20): 402.0}
    monkeypatch.setattr(daily_nav, "_load_close_series", lambda tk: closes)
    t = _trade(status="CLOSED", entry_date="2026-11-17", exit_date="2026-11-20", exit_price=402.0,
               exit_datetime="2026-11-20T20:00:00+00:00")
    base = dict(daily_nav._daily_returns_for_trade(dict(t), date(2026, 11, 25)) and
                {d: r for d, r, _ in daily_nav._daily_returns_for_trade(dict(t), date(2026, 11, 25))})
    dv.update(t, t["exit_datetime"], now=NOW)
    got = {d: r for d, r, _ in daily_nav._daily_returns_for_trade(t, date(2026, 11, 25))}
    assert got[date(2026, 11, 19)] - base[date(2026, 11, 19)] == pytest.approx(0.98 / 400)
    assert all(got[d] == pytest.approx(base[d]) for d in got if d != date(2026, 11, 19))
