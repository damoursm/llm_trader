"""IBKR's borrow fee day by day (`src/performance/borrow_fees.py`, `src/broker/flex.py`; user directive 2026-10-06: "we
want the real exact borrow fees we would have in live trading a real account and we want that data to be as complete
as possible").

What must hold: IBKR's calendar — an overnight-session execution belongs to the next session, T+1 skips weekends, NYSE
holidays and the settlement-only Columbus / Veterans days, and the charged days run from the short's settlement to the
cover's, weekends included; the collateral is roundup(1.02 x the prior close); each day's rate comes from IBKR in
order (its charge on our account, the day's figure in its history, our archive's file of the day, the last one
carried over a day none holds, the file in force for a day not over, then the entry stamp and the default); the
schedule stored on the trade is what the ledger, the daily NAV and the backfill gate read; IBKR's Flex statement is
parsed by attribute and stored per account period without doubling a day; a held name's month of IBKR's bars merges
into its history.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from config.settings import settings
from src.performance import borrow_fees as bf

ET = bf.ET


def _et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_trade_dates_follow_ibkrs_sessions():
    assert bf.trade_date(_et(2026, 10, 6, 14)) == date(2026, 10, 6)          # regular hours
    assert bf.trade_date(_et(2026, 10, 6, 5)) == date(2026, 10, 6)           # pre-market
    assert bf.trade_date(_et(2026, 10, 6, 19, 30)) == date(2026, 10, 6)      # after hours
    assert bf.trade_date(_et(2026, 10, 4, 20, 30)) == date(2026, 10, 5)      # Sunday's overnight session: Monday
    assert bf.trade_date(_et(2026, 10, 6, 21)) == date(2026, 10, 7)          # Tuesday 21:00: Wednesday's
    assert bf.trade_date(_et(2026, 10, 7, 1)) == date(2026, 10, 7)           # Wednesday 01:00: Wednesday's
    assert bf.trade_date(_et(2026, 10, 3, 12)) == date(2026, 10, 5)          # a Saturday: Monday
    assert bf.trade_date("2026-10-06T18:00:00+00:00") == date(2026, 10, 6)   # ISO, UTC
    assert bf.trade_date("2026-10-07T00:30:00") == date(2026, 10, 7)         # naive = UTC: 20:30 ET on the 6th


def test_settlement_skips_weekends_holidays_and_the_settlement_only_days():
    assert bf.settlement_date(date(2026, 10, 6)) == date(2026, 10, 7)
    assert bf.settlement_date(date(2026, 10, 2)) == date(2026, 10, 5)        # Friday: Monday
    assert bf.settlement_date(date(2026, 10, 9)) == date(2026, 10, 13)       # Columbus Day 10-12 trades, never settles
    assert bf.settlement_date(date(2026, 10, 12)) == date(2026, 10, 13)
    assert bf.settlement_date(date(2026, 11, 10)) == date(2026, 11, 12)      # Veterans Day, Wednesday 11-11
    assert bf.settlement_date(date(2026, 11, 25)) == date(2026, 11, 27)      # Thanksgiving
    assert bf.settlement_date(date(2024, 5, 24)) == date(2024, 5, 29)        # still T+2, over Memorial Day
    assert bf.settlement_date(date(2024, 5, 28)) == date(2024, 5, 29)        # T+1 from 2024-05-28


def test_the_charged_days_run_from_the_shorts_settlement_to_the_covers():
    thu_to_mon = bf.charged_days(_et(2026, 10, 1, 10), _et(2026, 10, 5, 11))
    assert thu_to_mon == [date(2026, 10, 2), date(2026, 10, 3), date(2026, 10, 4), date(2026, 10, 5)]
    assert bf.charged_days(_et(2026, 10, 2, 10), _et(2026, 10, 5, 11)) == [date(2026, 10, 5)]   # settles Monday
    assert bf.charged_days(_et(2026, 10, 6, 10), _et(2026, 10, 6, 15)) == []                    # a day trade


def test_the_collateral_is_102pct_of_the_prior_close_rounded_up_to_the_dollar():
    assert bf.collateral_price(5.10) == 6.0
    assert bf.collateral_price(9.80) == 10.0           # 9.996
    assert bf.collateral_price(10.0) == 11.0           # 10.2
    assert bf.collateral_price(50.0) == 51.0           # exactly 51: not 52
    assert bf.collateral_price(0.40) == 1.0


@pytest.fixture
def ibkr(monkeypatch):
    """A fake IBKR: its daily-rate history, our archive's files by day, the file in force now, closes, charges."""
    st = {"rates": [], "file": {}, "now_file": None, "closes": [], "charged": {}}
    monkeypatch.setattr(settings, "enable_ibkr_borrow_schedule", True)
    monkeypatch.setattr(settings, "enable_short_borrow_cost", True)
    monkeypatch.setattr(bf, "_rates", lambda t: st["rates"])

    def file_rate(t, when):
        e = when.astimezone(ET)
        if e.hour == 23 and e.minute == 59:                     # the end-of-day lookup of a past day
            f = st["file"].get(e.date())
            return (f, e.date()) if f is not None else None
        return (st["now_file"], e.date()) if st["now_file"] is not None else None
    monkeypatch.setattr(bf, "_file_rate", file_rate)
    monkeypatch.setattr(bf, "_closes", lambda t: st["closes"])
    monkeypatch.setattr(bf, "_charged", lambda: st["charged"])
    monkeypatch.setattr(bf, "charged_version", lambda: "v1")
    return st


def test_each_days_rate_comes_from_ibkr_in_order(ibkr):
    now = _et(2026, 10, 6, 14)
    ibkr["rates"] = [(date(2026, 10, 1), 50.0, "ibkr_api"), (date(2026, 10, 2), 80.0, "own_archive")]
    assert bf.day_rate("XYZ", date(2026, 10, 2), now) == (80.0, "own_archive")
    assert bf.day_rate("XYZ", date(2026, 10, 3), now) == (80.0, "own_archive_carried")      # Saturday: Friday's
    ibkr["file"][date(2026, 10, 5)] = 120.0                                                  # not summarised yet
    assert bf.day_rate("XYZ", date(2026, 10, 5), now) == (120.0, "own_archive")
    ibkr["now_file"] = 150.0
    assert bf.day_rate("XYZ", date(2026, 10, 6), now) == (150.0, "file_now")                # today: provisional
    ibkr.update(rates=[], file={}, now_file=None)
    assert bf.day_rate("XYZ", date(2026, 10, 2), now, {"borrow_fee_pct": 33.0}) == (33.0, "entry_stamp")
    assert bf.day_rate("XYZ", date(2026, 10, 2), now) == (float(settings.short_borrow_annual_pct), "default")


def test_rows_after_now_never_reach_the_schedule(ibkr):
    """A replay at an earlier instant charges what was known then: a later day's row changes nothing."""
    now = _et(2026, 10, 6, 14)
    ibkr["rates"] = [(date(2026, 10, 1), 36.0, "own_archive")]
    ibkr["closes"] = [(date(2026, 10, 1), 10.0)]
    trade = _short()
    before = bf.schedule(trade, "2026-10-06T15:00:00+00:00", now)
    ibkr["rates"] = ibkr["rates"] + [(date(2026, 10, 7), 900.0, "own_archive"), (date(2026, 10, 8), 900.0, "ibkr_api")]
    ibkr["closes"] = ibkr["closes"] + [(date(2026, 10, 6), 50.0), (date(2026, 10, 7), 80.0)]
    assert bf.schedule(trade, "2026-10-06T15:00:00+00:00", now) == before


def _short(**kw):
    t = {"trade_id": "t1", "ticker": "XYZ", "action": "SELL", "type": "STOCK", "status": "OPEN",
         "entry_datetime": "2026-10-01T14:00:00+00:00", "entry_date": "2026-10-01", "entry_price": 10.0,
         "sel_account_shares": 100, "position_size_multiplier": 1.0}
    t.update(kw)
    return t


def test_a_shorts_schedule_is_charged_day_by_day_and_stored(ibkr):
    ibkr["rates"] = [(date(2026, 10, 1), 36.0, "own_archive")]                 # 36 %/yr = 0.1% of collateral a day
    ibkr["closes"] = [(date(2026, 10, 1), 9.80), (date(2026, 10, 2), 14.50), (date(2026, 10, 5), 10.0)]
    trade = _short()
    frac = bf.cost_fraction(trade, "2026-10-06T15:00:00+00:00", now=_et(2026, 10, 6, 11))
    # a Thursday short settles Friday; a cover now (Tuesday) would settle Wednesday: Fri, Sat, Sun, Mon, Tue
    assert [r[0] for r in trade["borrow_days"]] == ["2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"]
    px = [r[3] for r in trade["borrow_days"]]
    assert px == [10.0, 15.0, 15.0, 15.0, 11.0]          # on Thursday's 9.80, Friday's 14.50 (x3), Monday's 10.00
    fee_ps = sum(p * 36.0 / 100.0 / 360.0 for p in px)
    assert frac == pytest.approx(fee_ps / 10.0)
    assert trade["borrow_fee_usd"] == pytest.approx(fee_ps * 100, abs=1e-4)
    assert trade["borrow_rate_now"] == 36.0 and trade["borrow_final"] is False
    assert trade["borrow_through"] == "2026-10-06"


def test_ibkrs_charge_on_our_account_wins_over_the_formula(ibkr):
    ibkr["rates"] = [(date(2026, 10, 1), 36.0, "own_archive")]
    ibkr["closes"] = [(date(2026, 10, 1), 9.80), (date(2026, 10, 2), 14.50)]
    ibkr["charged"] = {("XYZ", date(2026, 10, 5)): {"fee_ps": 0.05, "rate": 120.0, "price": 15.0}}
    trade = _short()
    bf.cost_fraction(trade, "2026-10-06T15:00:00+00:00", now=_et(2026, 10, 6, 11))
    day = {r[0]: r for r in trade["borrow_days"]}
    assert day["2026-10-05"] == ["2026-10-05", "ibkr_charged", 120.0, 15.0, 0.05]
    assert day["2026-10-02"][1] == "own_archive_carried"             # no row for Friday: Thursday's


def test_a_closed_short_without_a_schedule_waits_for_the_backfill(ibkr, monkeypatch):
    ibkr["rates"] = [(date(2026, 10, 1), 36.0, "own_archive")]
    ibkr["closes"] = [(date(2026, 10, 1), 9.80), (date(2026, 10, 2), 14.50)]
    trade = _short(status="CLOSED", exit_datetime="2026-10-05T15:00:00+00:00", exit_price=9.0)
    assert bf.cost_fraction(trade) is None and "borrow_days" not in trade           # the old flat carry stays
    monkeypatch.setattr(settings, "borrow_backfill_closed", True)
    assert bf.cost_fraction(trade, now=_et(2026, 10, 7, 10)) > 0
    assert trade["borrow_final"] is True and trade["borrow_through"] == "2026-10-05"
    trade["borrow_fee_ps"] = 1.0                       # a final stored schedule is read, never recomputed
    assert bf.cost_fraction(trade) == pytest.approx(0.1)


def test_the_ledger_charges_the_schedule_and_the_hypotheticals_the_flat_carry(ibkr):
    from src.performance import tracker
    ibkr["rates"] = [(date(2026, 10, 1), 36.0, "own_archive")]
    ibkr["closes"] = [(date(2026, 10, 1), 9.80), (date(2026, 10, 2), 14.50), (date(2026, 10, 5), 10.0)]
    trade = _short()
    f = tracker._borrow_cost(trade, "2026-10-06T15:00:00+00:00")
    assert f == pytest.approx(sum(r[4] for r in trade["borrow_days"]) / 10.0)
    flat = tracker._borrow_cost(_short(), "2026-10-06T15:00:00+00:00", schedule=False)
    days = (datetime(2026, 10, 6, 15, tzinfo=timezone.utc) - datetime(2026, 10, 1, 14, tzinfo=timezone.utc)).total_seconds() / 86400
    assert flat == pytest.approx(float(settings.short_borrow_annual_pct) / 100.0 * days / 365.0)


def test_the_daily_nav_charges_each_stored_day_in_its_interval(ibkr, monkeypatch):
    from src.performance import daily_nav
    closes = {date(2026, 10, 1): 10.0, date(2026, 10, 2): 10.0, date(2026, 10, 5): 10.0}
    monkeypatch.setattr(daily_nav, "_load_close_series", lambda t: closes)
    sched = [[f"2026-10-0{d}", "own_archive", 36.0, 11.0, 0.011] for d in (2, 3, 4, 5)]
    with_fee = daily_nav._daily_returns_for_trade(_short(borrow_days=sched), date(2026, 10, 5))
    monkeypatch.setattr(settings, "enable_short_borrow_cost", False)
    without = daily_nav._daily_returns_for_trade(_short(borrow_days=sched), date(2026, 10, 5))
    assert len(with_fee) == len(without) == 2
    assert with_fee[0][1] == pytest.approx(without[0][1] - 0.011 / 10.0)          # Friday's day
    assert with_fee[1][1] == pytest.approx(without[1][1] - 0.033 / 10.0)          # Sat, Sun, Mon


SAMPLE = """<FlexQueryResponse queryName="borrow" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="DU1234567" fromDate="20260901" toDate="20261005" period="Last365CalendarDays">
<BorrowFeesDetails>
<BorrowFeesDetail accountId="DU1234567" currency="USD" fxRateToBase="1" symbol="BRK B" description="BERKSHIRE B"
 conid="72063691" valueDate="20260925" quantity="-10" price="490" value="-4900" borrowFeeRate="0.25" borrowFee="-0.034" />
<BorrowFeesDetail accountId="DU1234567" currency="USD" fxRateToBase="1" symbol="XYZ" description="XYZ CORP"
 conid="123" valueDate="20260926" quantity="-100" price="15" value="-1500" borrowFeeRate="120" borrowFee="-5" />
</BorrowFeesDetails>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>"""


def test_the_flex_statement_parses_by_attribute():
    from src.broker import flex
    rows, periods = flex.parse_borrow_fees(SAMPLE)
    assert [r["ticker"] for r in rows] == ["BRK-B", "XYZ"] and rows[0]["symbol"] == "BRK B"
    assert rows[1]["value_date"] == date(2026, 9, 26) and rows[1]["fee"] == -5.0 and rows[1]["conid"] == 123
    assert periods == [{"account": "DU1234567", "start": date(2026, 9, 1), "end": date(2026, 10, 5)}]


def test_the_flex_fetch_stores_ibkrs_charges_once_and_the_ledger_reads_them(monkeypatch, tmp_path):
    from src.broker import flex
    monkeypatch.setattr(settings, "ibkr_flex_token", "tok")
    monkeypatch.setattr(settings, "ibkr_flex_query_id", "42")
    monkeypatch.setattr(flex.time, "sleep", lambda s: None)
    calls = []

    def fake_get(url, params, timeout=60.0):
        calls.append((url, dict(params)))
        if url == flex.SEND_URL:
            return ("<FlexStatementResponse><Status>Success</Status><ReferenceCode>777</ReferenceCode>"
                    "<Url>https://example.test/GetStatement</Url></FlexStatementResponse>")
        if sum(1 for u, _ in calls if u != flex.SEND_URL) == 1:
            return ("<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
                    "<ErrorMessage>Statement generation in progress.</ErrorMessage></FlexStatementResponse>")
        return SAMPLE
    monkeypatch.setattr(flex, "_get", fake_get)
    out = flex.fetch_borrow_fees()
    assert out["rows"] == 2 and out["stored"] == 2
    assert calls[1] == ("https://example.test/GetStatement", {"t": "tok", "q": "777", "v": "3"})
    assert list((tmp_path / "ibkr_flex").glob("*.xml"))                       # the raw statement is kept
    calls.clear()
    flex.fetch_borrow_fees()                                                    # a re-fetch never doubles a day
    from src.db import repo
    assert len(repo.load_broker_borrow_fees()) == 2
    got = bf._charged()
    assert got[("XYZ", date(2026, 9, 26))]["fee_ps"] == pytest.approx(0.05)


def test_the_flex_fetch_waits_for_a_token_and_reports_ibkrs_refusal(monkeypatch):
    from src.broker import flex
    assert flex.fetch_borrow_fees() == {"skipped": "no Flex token / query id"}
    monkeypatch.setattr(settings, "ibkr_flex_token", "tok")
    monkeypatch.setattr(settings, "ibkr_flex_query_id", "42")
    monkeypatch.setattr(flex, "_get", lambda url, params, timeout=60.0: (
        "<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012</ErrorCode>"
        "<ErrorMessage>Token has expired.</ErrorMessage></FlexStatementResponse>"))
    with pytest.raises(flex.FlexError) as e:
        flex.fetch_borrow_fees()
    assert e.value.code == "1012"


def test_a_held_names_month_merges_into_its_history(tmp_path):
    from src.data.deep import borrow_history as bh
    p = tmp_path / "XYZ.csv"
    p.write_text("date,open_fee,high_fee,low_fee,fee\n2026-09-01,1,1,1,1\n2026-10-01,2,2,2,2\n", encoding="utf-8")
    bh._merge_part(p, ["2026-10-01,3,3,3,3", "2026-10-02,4,4,4,4"])
    assert p.read_text(encoding="utf-8").splitlines() == [
        "date,open_fee,high_fee,low_fee,fee", "2026-09-01,1,1,1,1", "2026-10-01,3,3,3,3", "2026-10-02,4,4,4,4"]


def test_a_rate_is_carried_a_week_at_most_unless_seen_at_the_entry(ibkr):
    now = _et(2026, 10, 6, 14)
    t = {"entry_datetime": "2026-09-29T14:00:00+00:00"}
    ibkr["rates"] = [(date(2024, 6, 21), 7.0, "community_archive"), (date(2026, 9, 28), 50.0, "ibkr_api")]
    assert bf.day_rate("XYZ", date(2026, 10, 2), now, t) == (50.0, "ibkr_api_carried")          # four days back
    ibkr["rates"] = [(date(2024, 6, 21), 7.0, "community_archive")]                            # two years back: no
    assert bf.day_rate("XYZ", date(2026, 10, 2), now, dict(t, borrow_fee_pct=12.0)) == (12.0, "entry_stamp")
    ibkr["rates"] = [(date(2026, 9, 25), 40.0, "own_archive")]                                 # seen at the entry
    held = {"entry_datetime": "2026-09-25T14:00:00+00:00"}
    assert bf.day_rate("XYZ", date(2026, 10, 20), _et(2026, 10, 21, 10), held) == (40.0, "own_archive_carried")


def test_a_name_without_daily_closes_takes_its_30_minute_session_closes(monkeypatch):
    """The collateral needs each session's close: the daily cache, else the deep store's last regular-hours bar."""
    import pandas as pd
    from src.data import intraday_store
    from src.performance import daily_nav
    monkeypatch.setattr(daily_nav, "_load_close_series", lambda t: {})
    idx = pd.DatetimeIndex(["2026-10-01 13:30", "2026-10-01 19:30", "2026-10-02 13:30", "2026-10-02 19:30"])  # UTC
    df = pd.DataFrame({"Close": [9.0, 9.8, 14.0, 14.5]}, index=idx)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda t: df)
    bf.reset()
    assert bf._closes("XYZ") == [(date(2026, 10, 1), 9.8), (date(2026, 10, 2), 14.5)]
    assert bf.prior_close("XYZ", date(2026, 10, 3)) == 14.5


def test_an_unreachable_statement_host_falls_back_to_the_request_host(monkeypatch):
    """IBKR's SendRequest answer named gdcdyn.interactivebrokers.com, whose lookups fail intermittently here
    (2026-10-06): the host is retried first, then the statement is asked from the SendRequest host, the same path."""
    import httpx
    from src.broker import flex
    monkeypatch.setattr(flex.time, "sleep", lambda s: None)
    asked = []

    def fake_get(url, params, timeout=60.0):
        asked.append(url)
        if "gdcdyn" in url:
            raise httpx.ConnectError("[Errno 11002] getaddrinfo failed")
        return SAMPLE
    monkeypatch.setattr(flex, "_get", fake_get)
    got = flex.get_statement("tok", "777", "https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement")
    gdc = "https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"
    assert got == SAMPLE and asked == [gdc] * 4 + [flex.GET_URL]


def test_a_statement_still_generating_is_collected_first_on_the_next_run(monkeypatch, tmp_path):
    """IBKR: never re-initiate a statement still being generated, keep retrieving it — the next run asks for the
    pending one before any new request."""
    from src.broker import flex
    monkeypatch.setattr(settings, "ibkr_flex_token", "tok")
    monkeypatch.setattr(settings, "ibkr_flex_query_id", "42")
    monkeypatch.setattr(flex.time, "sleep", lambda s: None)
    state = {"ready": False, "sends": 0}

    def fake_get(url, params, timeout=60.0):
        if url == flex.SEND_URL:
            state["sends"] += 1
            return ("<FlexStatementResponse><Status>Success</Status><ReferenceCode>555</ReferenceCode>"
                    "<Url>https://example.test/GetStatement</Url></FlexStatementResponse>")
        if not state["ready"]:
            return ("<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
                    "<ErrorMessage>Statement generation in progress.</ErrorMessage></FlexStatementResponse>")
        return SAMPLE
    monkeypatch.setattr(flex, "_get", fake_get)
    with pytest.raises(flex.FlexError):
        flex.fetch_borrow_fees()                                            # gives up: the ref is kept
    assert json.loads((tmp_path / "ibkr_flex" / flex.PENDING).read_text())["ref"] == "555"
    state["ready"] = True
    out = flex.fetch_borrow_fees()
    assert out["rows"] == 2 and state["sends"] == 1                        # collected, never re-requested
    assert not (tmp_path / "ibkr_flex" / flex.PENDING).exists()
