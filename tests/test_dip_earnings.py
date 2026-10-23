"""PREREG137's DEMOTE_E in production (user 2026-10-17: "Fully deploy the earnings dip demotion in live"; from the
2026-10-19 session): in the dip book's daily top pick a signal whose company filed an 8-K Item 2.02 (its earnings
release) after 16:00 ET of the session three before the signal session and by D's 09:30 ranks after every other signal
of its day (`src/signals/dip_earnings.py`, `dip_long.rank_key`). What must hold: the deployed values; the window is the
study's (`build137.py` / `sim137.py`); only an 8-K / 8-K/A naming Item 2.02 counts, its acceptance inside (lo, hi];
the store is read first, EDGAR's current filings for a signal it does not flag, each outcome its status; an earnings dip
ranks last — a day of earnings dips still buys its best —, the ranking is the simulator's (`strategies.dip.rank_pieces`
with ``demote``, by score and by RSI2); the orders carry the flag and compete on it; fail-open (unread filings demote
nothing; a banner problem when it is the pick); off = the day as before; the simulator's construction
(`mimic.earnings_demoted`) is production's and `Mimic()`'s default follows production. The look-ahead guard is in
`tests/test_no_lookahead.py`; the parity with the study's 386 flagged orders and its DMTE draws in
`memory/dip-earnings-dips-2026-10.md`."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings, settings
from src.data import deep
from src.performance import tracker
from src.signals import dip_earnings as DE
from src.signals import dip_long as dl
from src.signals import sel_short as ss
from tests.test_dip_filter import DAY, PREV, _bars, _lsig, at
from tests.test_dip_top_pick import _day, _model, _q1, _tsig, q1_day, store  # noqa: F401 — fixtures

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _et(d: date, h: int, m: int, s: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET)


FILING_COLUMNS = ["ticker", "cik", "accession", "filing_date", "report_date", "acceptance", "act", "form",
                  "file_number", "items", "size", "is_xbrl", "is_inline_xbrl", "primary_doc"]


def _filings(tk: str, rows, cik: str = "0000000042") -> pd.DataFrame:
    """A company's filings as the store holds them: ``rows`` = (acceptance (aware), form, items)."""
    return pd.DataFrame([{"ticker": tk, "cik": cik, "accession": f"{cik}-26-{k:06d}",
                          "filing_date": a.astimezone(ET).date().isoformat(), "report_date": "2026-09-30",
                          "acceptance": pd.Timestamp(a).tz_convert("UTC").tz_localize(None), "act": "34", "form": f,
                          "file_number": "001-00001", "items": it, "size": 1000, "is_xbrl": 0, "is_inline_xbrl": 0,
                          "primary_doc": "x.htm"} for k, (a, f, it) in enumerate(rows)], columns=FILING_COLUMNS)


OLD_FILING = (datetime(2026, 9, 1, 16, 30, tzinfo=ET), "8-K", "5.02")   # an officer change, weeks before


@pytest.fixture
def parts(monkeypatch, tmp_path):
    """The demotion on, EDGAR off, the store's parts in a temp dir: ``put(tk, rows)`` writes a name's part."""
    d = tmp_path / "sec_parts"
    d.mkdir()
    monkeypatch.setattr(DE, "parts_dir", lambda: d)
    monkeypatch.setattr(DE, "store_through", lambda: "2026-10-08T12:33:37+00:00")
    monkeypatch.setattr(DE, "accepted_at", lambda cik, acc, budget=None: None)   # no index page: the JSON's instant
    for k, v in {"enable_dip_long_earnings_demote": True, "dip_long_earnings_demote_sessions": 3,
                 "dip_long_earnings_edgar": False}.items():
        monkeypatch.setattr(settings, k, v)

    def put(tk, rows, cik="0000000042"):
        deep.write_parquet(_filings(tk, rows, cik), d / f"{tk}.parquet")
    return put


def test_the_deployed_values():
    f = Settings.model_fields
    assert (f["enable_dip_long_earnings_demote"].default, f["dip_long_earnings_demote_sessions"].default,
            f["dip_long_earnings_edgar"].default) == (True, 3, True)


# ── the window and the releases ──────────────────────────────────────────────

def test_the_window_is_the_studys():
    """16:00 ET of the session three before the signal session (the one before D) .. 09:30 ET of D — sim137.py's
    ``sess[sig_i - 3] + 16h`` and ``D + 9h30``, in its own construction, across weekends, a holiday and a DST change."""
    from src.performance.market_calendar import is_market_day
    sess = pd.DatetimeIndex([x for x in pd.date_range("2025-12-01", "2026-12-31") if is_market_day(x.date())])
    pos = {x: i for i, x in enumerate(sess)}
    for d in (DAY, date(2026, 10, 19), date(2026, 11, 2), date(2026, 11, 27), date(2026, 1, 2), date(2026, 7, 6)):
        i = pos[pd.Timestamp(d)]
        for n in (1, 3, 5):
            lo = (sess[i - 1 - n] + pd.Timedelta(hours=16)).tz_localize("America/New_York").value
            hi = (pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York").value
            assert DE.window_ns(d, n) == (lo, hi), (d, n)
    lo, hi = DE.window(DAY)                                          # Thursday 10-08: from Friday 10-02's close
    assert (lo, hi) == (_et(date(2026, 10, 2), 16, 0).astimezone(UTC), _et(DAY, 9, 30).astimezone(UTC))
    assert DE.window(date(2026, 11, 2))[1] == datetime(2026, 11, 2, 14, 30, tzinfo=UTC)   # EST after the change
    with pytest.raises(ValueError):
        DE.window(DAY, 0)


def test_only_an_8k_naming_item_202_is_a_release():
    a = _et(PREV, 16, 5)
    f = _filings("AAA", [(a, "8-K", "2.02,9.01"), (a, "8-K/A", "2.02"), (a, "8-K", "1.01,2.02"), (a, "8-K", "5.02"),
                         (a, "10-Q", "2.02"), (a, "8-K", "12.02"), (a, "8-K12B", "2.02"), (a, "8-K", "")])
    f.loc[7, "items"] = None
    got = DE.releases(f)
    assert list(got["form"]) == ["8-K", "8-K/A", "8-K"] and list(got["items"]) == ["2.02,9.01", "2.02", "1.01,2.02"]
    assert (got["ns"] == pd.Timestamp(a).value).all()
    f2 = f.copy()
    f2["acceptance"] = pd.to_datetime(f2["acceptance"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    assert list(DE.releases(f2)["ns"]) == list(got["ns"])            # an aware column: the same instants
    f.loc[0, "acceptance"] = pd.NaT
    assert len(DE.releases(f)) == 2                                  # no acceptance: left out
    assert DE.releases(None).empty and DE.releases(pd.DataFrame()).empty


def test_the_window_counts_after_its_open_and_through_its_close():
    lo, hi = DE.window_ns(DAY)
    rel = pd.DataFrame({"accession": ["a", "b", "c", "d"], "cik": "1", "form": "8-K", "items": "2.02",
                        "ns": [lo, lo + 1, hi, hi + 1]})
    assert DE.in_window(rel.iloc[[0]], lo, hi) is None                # at the open (16:00 sharp): before the window
    assert DE.in_window(rel.iloc[[1]], lo, hi)["accession"] == "b"
    assert DE.in_window(rel.iloc[[2]], lo, hi)["accession"] == "c"    # 09:30 sharp: in
    assert DE.in_window(rel.iloc[[3]], lo, hi) is None                # after 09:30: out
    got = DE.in_window(rel, lo, hi)
    assert (got["accession"], got["n_in_window"]) == ("c", 2)        # the latest of those inside
    assert DE.released(rel["ns"].to_numpy(), lo, hi) and not DE.released(np.array([lo, hi + 1]), lo, hi)


# ── the two sources ──────────────────────────────────────────────────────────

def test_the_store_decides_first_then_edgar(parts, monkeypatch):
    inside, outside = _et(PREV, 16, 30), _et(date(2026, 10, 1), 16, 30)
    parts("AAA", [(inside, "8-K", "2.02,9.01")])
    parts("BBB", [(outside, "8-K", "2.02,9.01"), (inside, "8-K", "5.02")])
    calls = []

    def edgar(tk, cik, budget=None):
        calls.append((tk, cik))
        return _filings(tk, [(_et(DAY, 8, 50), "8-K", "2.02")], cik)  # filed after the pre-open pass, before 09:30
    monkeypatch.setattr(DE, "edgar_filings", edgar)
    r = DE.check("AAA", DAY, edgar=True)
    assert (r["earnings_dip"], r["status"], r["source"], r["edgar_asked"]) == (True, "flagged", "store", False)
    assert r["release"]["acceptance"] == pd.Timestamp(inside).tz_convert("UTC").isoformat() and calls == []
    r = DE.check("BBB", DAY, edgar=True)                              # the store holds none in the window: EDGAR now
    assert (r["earnings_dip"], r["status"], r["source"]) == (True, "flagged", "edgar")
    assert calls == [("BBB", "0000000042")]                          # the part's CIK
    assert DE.check("BBB", DAY, edgar=False)["status"] == "store_only"
    monkeypatch.setattr(DE, "edgar_filings", lambda tk, cik, budget=None: _filings(tk, [(outside, "8-K", "2.02")], cik))
    r = DE.check("BBB", DAY, edgar=True)
    assert (r["earnings_dip"], r["status"], r["source"]) == (False, "clear", "edgar")


def test_an_edgar_release_near_the_window_is_retimed_from_its_index_page(parts, monkeypatch):
    """EDGAR's JSON instant drifts (2026-10-17: +4 h / +8 h on re-reads), so a release it answers within 12 h of the
    window takes its index page's "Accepted" instant: one the JSON puts after 09:30 but accepted at 09:00 is an
    earnings dip, one the JSON puts inside but accepted at 13:00 is not; a far release costs no index request; an
    unreadable page keeps the JSON's instant (noted)."""
    parts("BBB", [OLD_FILING])
    hi = _et(DAY, 9, 30)
    json_rows = [(hi + timedelta(hours=3, minutes=30), "8-K", "2.02"),       # the JSON says 13:00, accepted 09:00
                 (_et(date(2026, 9, 10), 16, 5), "8-K", "2.02")]              # four weeks before: never re-timed
    monkeypatch.setattr(DE, "edgar_filings", lambda tk, cik, budget=None: _filings(tk, json_rows, cik))
    asked = []
    pages = {"0000000042-26-000000": pd.Timestamp(_et(DAY, 9, 0)).value}

    def page(cik, acc, budget=None):
        asked.append(acc)
        return pages.get(acc)
    monkeypatch.setattr(DE, "accepted_at", page)
    r = DE.check("BBB", DAY, edgar=True)
    assert (r["earnings_dip"], r["status"], r["release"]["acceptance_source"]) == (True, "flagged", "index")
    assert r["release"]["acceptance"] == pd.Timestamp(_et(DAY, 9, 0)).tz_convert("UTC").isoformat()
    assert asked == ["0000000042-26-000000"]                         # the far release cost no request
    json_rows[0] = (hi - timedelta(minutes=30), "8-K", "2.02")        # the JSON says 09:00, accepted 13:00
    pages["0000000042-26-000000"] = pd.Timestamp(_et(DAY, 13, 0)).value
    r = DE.check("BBB", DAY, edgar=True)
    assert (r["earnings_dip"], r["status"]) == (False, "clear")
    pages.clear()                                                     # the index page unreadable: the JSON's instant
    r = DE.check("BBB", DAY, edgar=True)
    assert (r["earnings_dip"], r["release"]["acceptance_source"]) == (True, "json")


def test_the_index_page_is_read_as_eastern_time(monkeypatch):
    class Page:
        status_code = 200
        text = ('<div class="infoHead">Accepted</div>\n        <div class="info">2026-10-16 06:58:21</div>')
    seen = []
    monkeypatch.setattr(DE, "_get", lambda url, budget=None: seen.append(url) or Page())
    assert DE.accepted_at("0000086312", "0000086312-26-000180") == pd.Timestamp("2026-10-16 10:58:21", tz="UTC").value
    assert seen == ["https://www.sec.gov/Archives/edgar/data/86312/000008631226000180/0000086312-26-000180-index.htm"]
    Page.text = "<html>no such block</html>"
    assert DE.accepted_at("0000086312", "0000086312-26-000180") is None

    def down(url, budget=None):
        raise TimeoutError("no answer")
    monkeypatch.setattr(DE, "_get", down)
    assert DE.accepted_at("0000086312", "0000086312-26-000180") is None


def test_every_edgar_request_is_one_bounded_call(monkeypatch):
    """No retry and no Retry-After sleep on the opening tick: ONE request through the store's limiter, at most
    `EDGAR_TIMEOUT_SECONDS`, never past the day's budget — a spent budget sends nothing."""
    import requests
    seen = []

    class Answer:
        status_code = 503
        text = ""
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: seen.append((url, timeout)) or Answer())
    with pytest.raises(RuntimeError, match="503"):
        DE.edgar_filings("BBB", "0000000042", DE.Budget(30.0))
    assert len(seen) == 1 and seen[0][0].endswith("/CIK0000000042.json") and seen[0][1] == DE.EDGAR_TIMEOUT_SECONDS
    seen.clear()
    with pytest.raises(RuntimeError, match="503"):
        DE.edgar_filings("BBB", "0000000042", DE.Budget(3.0))
    assert 2.0 < seen[0][1] <= 3.0                                     # the budget's remainder caps the timeout
    seen.clear()
    with pytest.raises(DE.BudgetSpent):
        DE.edgar_filings("BBB", "0000000042", DE.Budget(0.5))
    assert seen == []


def test_every_outcome_has_its_status(parts, monkeypatch):
    parts("BBB", [(_et(date(2026, 10, 1), 16, 30), "8-K", "2.02")])

    def down(tk, cik, budget=None):
        raise RuntimeError("submissions HTTP 503")
    monkeypatch.setattr(DE, "edgar_filings", down)
    r = DE.check("BBB", DAY, edgar=True)                              # EDGAR down: the store's answer, noted
    assert (r["earnings_dip"], r["status"], r["edgar_asked"]) == (False, "store_only", True) and "503" in r["error"]
    for ask in (False, True):                                         # no part: never an earnings dip (the study's)
        r = DE.check("NOPART", DAY, edgar=ask)
        assert (r["earnings_dip"], r["status"], r["edgar_asked"]) == (False, "no_filings", False)
    parts("NOCIK", [OLD_FILING], cik="")
    r = DE.check("NOCIK", DAY, edgar=True)                            # a part without a CIK: the store's answer
    assert (r["status"], r["edgar_asked"]) == ("store_only", False) and "no CIK" in r["error"]

    def unreadable(tk):
        raise OSError("part locked")
    monkeypatch.setattr(DE, "store_filings", unreadable)
    monkeypatch.setattr(DE, "edgar_filings", lambda tk, cik, budget=None: pytest.fail("asked EDGAR without a CIK"))
    for ask in (False, True):
        r = DE.check("BBB", DAY, edgar=ask)                           # an unreadable part: unknown, not demoted
        assert (r["earnings_dip"], r["status"]) == (None, "unknown") and "locked" in r["error"]


def test_the_days_edgar_budget_is_bounded(parts, monkeypatch):
    for tk in ("AAA", "BBB"):
        parts(tk, [OLD_FILING])
    monkeypatch.setattr(settings, "dip_long_earnings_edgar", True)
    monkeypatch.setattr(DE, "EDGAR_BUDGET_SECONDS", -1.0)              # spent before the first request
    monkeypatch.setattr(DE, "edgar_filings", lambda tk, cik, budget=None: pytest.fail("asked EDGAR past the budget"))
    sig = [{"ticker": "AAA"}, {"ticker": "BBB"}]
    b = DE.flag_signals(sig, DAY)
    assert [s["earnings_status"] for s in sig] == ["store_only", "store_only"]
    assert all("budget" in s["earnings_error"] for s in sig) and b["edgar"] == {"on": True, "asked": 0,
                                                                               "failed": ["AAA", "BBB"]}


# ── the day's signals ────────────────────────────────────────────────────────

def _scores(monkeypatch, store, **sc):
    _q1(monkeypatch)
    return _model(monkeypatch, store, sc or {"AAA": 0.4, "BBB": 3.0, "CCC": 1.2})


def test_an_earnings_dip_is_passed_over_for_the_next_best(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    parts("BBB", [(_et(PREV, 16, 5), "8-K", "2.02,9.01")])            # BBB, the best score, reported last night
    parts("CCC", [(_et(PREV, 16, 5), "8-K", "5.02")])                 # an officer change: not a release
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["CCC"]          # the next best score
    assert [(s["ticker"], s["top_rank"], s["earnings_dip"]) for s in out["not_top"]] == [("AAA", 2, False),
                                                                                        ("BBB", 3, True)]
    e = out["top"]["earnings"]
    assert (e["flagged"], e["checked"], e["sessions"], e["unknown"]) == (["BBB"], 3, 3, [])
    assert e["statuses"] == {"BBB": "flagged", "CCC": "store_only", "AAA": "no_filings"}   # best-ranked first
    rel = [s for s in out["not_top"] if s["ticker"] == "BBB"][0]["earnings_release"]
    assert (rel["items"], rel["source"]) == ("2.02,9.01", "store")
    assert out["top"]["picked"] == [{"ticker": "CCC", "score": 1.2, "rsi2": store["CCC"]["rsi2"],
                                     "earnings_dip": False, "earnings_status": "store_only"}]
    disk = json.loads(dl.signals_path(DAY).read_text(encoding="utf-8"))
    assert disk["top"]["earnings"]["flagged"] == ["BBB"] and disk["signals"][0]["ticker"] == "CCC"
    monkeypatch.setattr(settings, "enable_dip_long", True)
    h = dl.health(at(DAY, 12, 0))
    assert h["problems"] == [] and any("top pick CCC" in n and "earnings dip(s) ranked last: BBB" in n
                                       for n in h["notes"])


def test_a_day_of_earnings_dips_buys_its_best(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    for tk in ("AAA", "BBB", "CCC"):
        parts(tk, [(_et(date(2026, 10, 5), 7, 0), "8-K", "2.02")])
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and out["signals"][0]["earnings_dip"] is True
    assert [s["ticker"] for s in out["not_top"]] == ["CCC", "AAA"]   # by score among the earnings dips
    monkeypatch.setattr(settings, "enable_dip_long", True)
    assert any("top pick BBB (score +3.00) — an earnings dip" in n for n in dl.health(at(DAY, 12, 0))["notes"])


def test_a_release_outside_the_window_changes_nothing(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    parts("BBB", [(_et(date(2026, 10, 2), 15, 59), "8-K", "2.02"),   # before the window opens (10-02 16:00)
                  (_et(DAY, 9, 31), "8-K", "2.02"), (_et(date(2026, 10, 9), 8, 0), "8-K", "2.02")])
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and out["top"]["earnings"]["flagged"] == []


def test_with_the_rule_off_the_day_is_as_before(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    parts("BBB", [(_et(PREV, 16, 5), "8-K", "2.02")])
    monkeypatch.setattr(settings, "enable_dip_long_earnings_demote", False)
    monkeypatch.setattr(DE, "flag_signals", lambda sig, d: pytest.fail("flagged with the rule off"))
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and "earnings" not in out["top"]
    assert not any(k.startswith("earnings") for s in out["signals"] + out["not_top"] for k in s)
    assert out["top"]["picked"] == [{"ticker": "BBB", "score": 3.0, "rsi2": store["BBB"]["rsi2"]}]


def test_unread_filings_are_not_demoted_and_raise_the_banner_only_for_the_pick(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    real = DE.store_filings

    def store_of(tk):
        if tk == "BBB":
            raise OSError("part locked")
        return real(tk)
    monkeypatch.setattr(DE, "store_filings", store_of)
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and out["signals"][0]["earnings_dip"] is None
    assert out["top"]["earnings"]["unknown"] == ["BBB"]
    monkeypatch.setattr(settings, "enable_dip_long", True)
    h = dl.health(at(DAY, 12, 0))
    assert any("pick BBB's filings could not be read" in p for p in h["problems"])
    _scores(monkeypatch, store, AAA=5.0, BBB=3.0, CCC=1.2)            # the unread name is not the pick: no problem
    out = dl.compute_signals(DAY)
    assert out["signals"][0]["ticker"] == "AAA" and out["top"]["earnings"]["unknown"] == ["BBB"]
    assert dl.health(at(DAY, 12, 0))["problems"] == []


def test_edgar_down_for_the_pick_is_a_note(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    for tk in ("AAA", "BBB", "CCC"):
        parts(tk, [OLD_FILING])
    monkeypatch.setattr(settings, "dip_long_earnings_edgar", True)

    def down(tk, cik, budget=None):
        raise RuntimeError("submissions HTTP None")
    monkeypatch.setattr(DE, "edgar_filings", down)
    out = dl.compute_signals(DAY)
    assert out["signals"][0]["ticker"] == "BBB" and out["top"]["earnings"]["edgar"]["failed"] == ["BBB", "CCC", "AAA"]
    monkeypatch.setattr(settings, "enable_dip_long", True)
    h = dl.health(at(DAY, 12, 0))
    assert h["problems"] == [] and any("EDGAR did not answer for today's pick BBB" in n and "2026-10-08T12:33" in n
                                       for n in h["notes"])


def test_the_check_failing_as_a_whole_demotes_nothing(monkeypatch, store, parts):
    _scores(monkeypatch, store)
    parts("BBB", [(_et(PREV, 16, 5), "8-K", "2.02")])

    def boom(sig, d):
        for s in sig:
            s["earnings_dip"] = True                                  # a partial stamp before the crash
        raise RuntimeError("boom")
    monkeypatch.setattr(DE, "flag_signals", boom)
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and "boom" in out["top"]["earnings"]["error"]
    assert not any(k.startswith("earnings") for s in out["signals"] + out["not_top"] for k in s)
    monkeypatch.setattr(settings, "enable_dip_long", True)
    assert any("earnings-dip check failed" in p for p in dl.health(at(DAY, 12, 0))["problems"])


# ── the ranking: the simulator's ─────────────────────────────────────────────

def test_the_ranking_with_the_demotion_is_the_simulators():
    """`strategies.dip.rank_pieces(by, topk=1, demote=...)` and production's `rank_key` choose the same order — by
    score and by RSI2, ties included, any set of earnings dips (none, some, all)."""
    from src.backtest.strategies import dip as DIP
    rng = np.random.default_rng(11)
    names = ["T1", "T10", "T2", "AB", "ABC", "B"]
    day = "2026-10-19"
    for _ in range(600):
        tks = list(rng.choice(names, size=int(rng.integers(1, 6)), replace=False))
        rsi = {tk: float(rng.choice([0.5, 1.5, 4.0])) for tk in tks}
        sc = {tk: float(rng.choice([-1.0, 0.5, 2.0])) for tk in tks if rng.random() < 0.6}
        dem = {tk for tk in tks if rng.random() < 0.4}
        pieces = [{"oid": f"{tk}|{day}", "rank": rsi[tk], "tkn": tk} for tk in tks]
        sig = [dict({"ticker": tk, "rsi2": rsi[tk], "earnings_dip": tk in dem},
                    **({"filter_score": sc[tk]} if tk in sc else {})) for tk in tks]
        for by in ("score", "rsi2"):
            out = DIP.rank_pieces(pieces, {f"{tk}|{day}": v for tk, v in sc.items()}, by=by, topk=1,
                                  demote={f"{tk}|{day}" for tk in dem})
            assert [p["oid"].split("|")[0] for p in out] == [min(sig, key=lambda s: dl.rank_key(s, by))["ticker"]]
        # the fill order (every piece's rank): the same order as production's key
        full = DIP.rank_pieces(pieces, {f"{tk}|{day}": v for tk, v in sc.items()}, by="score",
                               demote={f"{tk}|{day}" for tk in dem})
        by_sim = [p["oid"].split("|")[0] for p in sorted(full, key=lambda p: (p["rank"], p["oid"]))]
        assert by_sim == [s["ticker"] for s in sorted(sig, key=lambda s: dl.rank_key(s, "score"))]


def test_an_earnings_dip_ranks_last_in_either_ranking():
    sig = [{"ticker": "A", "rsi2": 1.0, "filter_score": 0.5}, {"ticker": "B", "rsi2": 8.0, "filter_score": 2.0,
                                                              "earnings_dip": True},
           {"ticker": "C", "rsi2": 0.5}, {"ticker": "D", "rsi2": 0.1, "earnings_dip": True},
           {"ticker": "E", "rsi2": 0.2, "earnings_dip": False}, {"ticker": "F", "rsi2": 0.3, "earnings_dip": None}]
    assert [s["ticker"] for s in sorted(sig, key=lambda s: dl.rank_key(s, "score"))] == ["A", "E", "F", "C", "B", "D"]
    assert [s["ticker"] for s in sorted(sig, key=lambda s: dl.rank_key(s, "rsi2"))] == ["E", "F", "C", "A", "D", "B"]


# ── the orders ───────────────────────────────────────────────────────────────

def test_the_orders_carry_the_flag_and_the_last_slot_goes_to_the_other_dip(q1_day):
    """An earlier day's pending pick that is an earnings dip (its day had no other) and today's pick compete for the
    last slot at one opening tick: today's is placed whatever its score; the trade an earnings dip opens stamps it."""
    days, feed = q1_day
    d0 = ss.sessions_before(DAY, 1)[0]
    held = []
    for k, tk in enumerate(("W1", "W2", "W3", "W4", "W5")):
        o = dl.make_order(d0, _lsig(tk, 5.0, 20.0, 1.0, 0.0), k + 2, PREV.isoformat())
        o.update(status="working", qty=1, placed_at=at(d0, 9, 30).isoformat(), placed_session=d0.isoformat(),
                 ibkr_qty=1)
        held.append(o)
    earn = dl.make_order(d0, _lsig("ERN", 1.0, 40.0, 1.0, 0.0), 1, PREV.isoformat())
    earn.update(dip_top=1, filter_score=5.0, earnings_dip=True, earnings_status="flagged",
                earnings_release={"accession": "0000000042-26-000001", "items": "2.02"})
    dl.save_orders(d0, held + [earn])
    top = dict(_tsig("LOW", 9.0, 30.0, 1.0, 0.1, 1), earnings_dip=False, earnings_status="clear")
    days[DAY] = _day(top, [dict(_tsig("ERN2", 2.0, 30.0, 1.0, 9.0, 2), earnings_dip=True, earnings_status="flagged")])
    tracker.record_dip_long_trades(now=at(DAY, 9, 30))
    o = {x["ticker"]: x for x in dl.load_orders(DAY)}
    assert o["LOW"]["status"] == "working" and o["LOW"]["earnings_dip"] is False
    assert o["ERN2"]["status"] == "not_top" and o["ERN2"]["earnings_dip"] is True
    assert [e for e in o["ERN2"]["events"] if e["what"] == "not_top"][0]["earnings_dip"] is True
    e0 = {x["ticker"]: x for x in dl.load_orders(d0)}["ERN"]
    assert e0["status"] == "pending" and e0["events"][-1]["what"] == "no_slot"     # the earnings dip waits
    t_hit = datetime(2026, 10, 8, 14, 30, tzinfo=UTC)
    e0.update(status="working", qty=3, placed_at=at(DAY, 9, 30).isoformat(), placed_session=DAY.isoformat())
    tracker._fill_dip_orders([(t_hit, d0, e0, 39.0, "touch")], [], t_hit, None, set())
    t = [x for x in tracker._load_trades() if x.get("ticker") == "ERN"][0]
    assert (t["dip_earnings_dip"], t["dip_earnings_release"]) == (True, "0000000042-26-000001")


# ── the simulator follows production ─────────────────────────────────────────

def test_the_simulator_follows_production(monkeypatch):
    from src.backtest import mimic as M
    _q1(monkeypatch)
    monkeypatch.setattr(settings, "enable_dip_long_earnings_demote", True)
    monkeypatch.setattr(settings, "dip_long_earnings_demote_sessions", 3)
    assert M.Mimic().dip_earnings == 3 and M.Mimic.off().dip_earnings is None
    monkeypatch.setattr(settings, "dip_long_earnings_demote_sessions", 1)
    assert M.Mimic().dip_earnings == 1
    monkeypatch.setattr(settings, "enable_dip_long_earnings_demote", False)
    assert M.Mimic().dip_earnings is None
    monkeypatch.setattr(settings, "enable_dip_long_earnings_demote", True)
    monkeypatch.setattr(settings, "dip_long_daily_top", 0)            # production flags inside the daily top pick only
    assert M.Mimic().dip_earnings is None
    for bad in (0, -1, True, 2.0, "3"):
        with pytest.raises(ValueError):
            M.Mimic(dip_earnings=bad)


def test_the_simulators_construction_is_productions(monkeypatch, tmp_path):
    """`mimic.earnings_demoted` on the consolidated tables flags exactly the orders production's `check` flags on the
    same companies' parts — the window, the item, the form; a delisted ``@D`` name reads ``XYZ_DELISTED``, set A0's
    symbol its mapped ticker."""
    from src.backtest import mimic as M
    monkeypatch.setattr(deep, "DEEP_DIR", tmp_path / "deep")
    monkeypatch.setattr(DE, "DELISTED_FILINGS", tmp_path / "deep_delisted" / "sec_filings.parquet")
    monkeypatch.setattr(DE, "_TABLE", {})
    monkeypatch.setattr(M, "_EARNINGS", {})
    monkeypatch.setattr(settings, "dip_long_earnings_edgar", False)
    rng = np.random.default_rng(5)
    days = [DAY, date(2026, 10, 19), date(2026, 3, 2), date(2025, 11, 28)]
    rows: dict = {}
    for tk in ("AAA", "BBB", "CCC", "OLD_DELISTED", "NEW"):
        rr = []
        for d in days:
            lo, hi = DE.window(d)
            span = int((hi - lo).total_seconds() // 60)
            for _ in range(2):
                a = lo + timedelta(minutes=int(rng.integers(-2 * 1440, span + 2 * 1440)))
                rr.append((a, str(rng.choice(["8-K", "8-K/A", "10-Q"])), str(rng.choice(["2.02,9.01", "5.02", "2.02"]))))
        rows[tk] = rr
    live = {tk: _filings(tk, rr) for tk, rr in rows.items() if not tk.endswith("_DELISTED")}
    deep.write_parquet(pd.concat(live.values(), ignore_index=True), tmp_path / "deep" / "sec_filings.parquet")
    deep.write_parquet(_filings("OLD_DELISTED", rows["OLD_DELISTED"]), DE.DELISTED_FILINGS)
    for tk, f in dict(live, OLD_DELISTED=_filings("OLD_DELISTED", rows["OLD_DELISTED"])).items():
        deep.write_parquet(f, deep.DEEP_DIR / "sec_filings" / "parts" / f"{tk}.parquet")
    monkeypatch.setattr(M, "_a0map", lambda d=None: {"OLDSYM": "NEW"})
    oids_ab = [f"{tk}|{d}" for tk in ("AAA", "BBB", "CCC", "OLD@D", "ZZZ") for d in days]
    oids_a0 = [f"{tk}|{d}" for tk in ("OLDSYM", "AAA") for d in days]
    got = M.earnings_demoted(oids_ab, 3) | M.earnings_demoted(oids_a0, 3, a0=True)
    company = {"OLD@D": "OLD_DELISTED", "OLDSYM": "NEW"}
    want = {o for o in oids_ab + oids_a0
            if DE.check(company.get(o.split("|")[0], o.split("|")[0]), date.fromisoformat(o.split("|")[1]),
                        edgar=False)["earnings_dip"] is True}
    assert got == want and 0 < len(want) < len(oids_ab) + len(oids_a0)
    assert not any(o.startswith("ZZZ|") for o in got)                # a name without filings is never demoted
