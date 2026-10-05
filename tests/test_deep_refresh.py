"""Deep history store — the nightly INCREMENTAL refresh (``src/data/deep/refresh.py``).

Offline: every test writes into its own DEEP_DIR and stubs the one fetch it
exercises. What is pinned is the machinery a silent failure would hide behind:
the merge keeps history and lets the newer row win, a tail starts from the
PART's newest key (never the manifest), a budget stop leaves a family DUE, the
insider tail never asks for today, the ALFRED merge keeps the true first print,
and the scheduler slot fires once a day including weekends.
"""
from __future__ import annotations

import time
from datetime import date, datetime, time as _t, timedelta, timezone

import pandas as pd
import pytest

from src.data import deep
from src.data.deep import form4_live
from src.data.deep import refresh as R


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(deep, "DEEP_DIR", tmp_path / "deep")
    (tmp_path / "deep").mkdir()
    yield


def _part(family: str, key: str, df: pd.DataFrame) -> None:
    deep.write_parquet(df, deep.family_dir(family) / "parts" / f"{key}.parquet")


def _read(family: str, key: str) -> pd.DataFrame:
    return deep.read_parquet(deep.family_dir(family) / "parts" / f"{key}.parquet")


# ── merge_part ───────────────────────────────────────────────────────────────

def test_merge_part_appends_only_rows_the_part_does_not_hold():
    _part("bars", "A", pd.DataFrame({"ticker": ["A", "A"],
                                     "ts": pd.to_datetime(["2026-09-17 13:30", "2026-09-17 14:00"]),
                                     "close": [1.0, 2.0]}))
    new = pd.DataFrame({"ticker": ["A", "A"],
                        "ts": pd.to_datetime(["2026-09-17 14:00", "2026-09-18 13:30"]),
                        "close": [2.5, 3.0]})
    added, total = R.merge_part("bars", "A", new, ["ts"], "ts")
    assert len(added) == 1 and total == 3
    back = _read("bars", "A")
    assert list(back["close"]) == [1.0, 2.0, 3.0]      # the stored bar is kept verbatim
    # an empty tail touches nothing
    added, total = R.merge_part("bars", "A", pd.DataFrame(), ["ts"], "ts")
    assert len(added) == 0 and total == 3


def test_merge_part_never_dedupes_the_stored_rows():
    """The ingest stores a provider's payload verbatim, exact duplicates
    included (Quiver returns identical awards as separate rows); a tail must
    not clean what it happens to touch, or the store is inconsistent between
    tickers (the first refresh shrank quiver_contracts by 6%)."""
    dup = {"ticker": "A", "Date": "2026-09-01", "Description": "X", "Agency": "DOD", "Amount": 10.0,
           "action_date": "2026-09-01"}
    _part("quiver_contracts", "A", pd.DataFrame([dup, dup]))
    new_row = dict(dup, Date="2026-09-22", action_date="2026-09-22")
    added, total = R.merge_part("quiver_contracts", "A", pd.DataFrame([dup, new_row]),
                                R.QUIVER_IDENTITY["quiver_contracts"], "Date")
    assert len(added) == 1 and total == 3                 # both stored duplicates survive
    # a part written for the first time is the payload verbatim, duplicates included
    added, total = R.merge_part("quiver_contracts", "B", pd.DataFrame([dup, dup]),
                                R.QUIVER_IDENTITY["quiver_contracts"], "Date")
    assert len(added) == 2 and total == 2


def test_merge_part_upsert_replaces_matching_rows_and_keeps_rows_the_refetch_lacks():
    """Quiver's DPI endpoint dropped its pre-2021 rows between August and
    September 2026: a refetch must bring the revised values without losing a
    stored day the endpoint no longer serves."""
    _part("dpi", "A", pd.DataFrame({"ticker": ["A", "A"], "Date": ["2020-01-02", "2026-09-18"],
                                    "DPI": [0.40, 0.41]}))
    new = pd.DataFrame({"ticker": ["A", "A"], "Date": ["2026-09-18", "2026-09-22"], "DPI": [0.45, 0.46]})
    added, total = R.merge_part("dpi", "A", new, ["Date"], "Date", mode="upsert")
    assert len(added) == 1 and total == 3
    back = _read("dpi", "A").set_index("Date")["DPI"]
    assert back["2020-01-02"] == 0.40 and back["2026-09-18"] == 0.45 and back["2026-09-22"] == 0.46
    with pytest.raises(ValueError):
        R.merge_part("dpi", "A", new, ["Date"], "Date", mode="overwrite")


def test_merge_part_identity_is_dtype_proof():
    _part("q", "A", pd.DataFrame({"ticker": ["A"], "Date": ["2026-09-01"], "Amount": [5000]}))
    new = pd.DataFrame({"ticker": ["A"], "Date": ["2026-09-01"], "Amount": [5000.0]})
    added, total = R.merge_part("q", "A", new, ["ticker", "Date", "Amount"], "Date")
    assert len(added) == 0 and total == 1


def test_quiver_congress_repeat_sighting_adds_nothing_even_when_returns_moved():
    row = {"ticker": "A", "Representative": "R", "BioGuideID": "B1", "ReportDate": "2026-09-10",
           "TransactionDate": "2026-09-01", "Transaction": "Purchase", "Range": "$1,001 - $15,000",
           "House": "Senate", "ExcessReturn": 1.2}
    _part("quiver_congress", "A", pd.DataFrame([row]))
    again = dict(row, ExcessReturn=3.4)                 # returns since the trade move every day
    added, total = R.merge_part("quiver_congress", "A", pd.DataFrame([again]),
                                R.QUIVER_IDENTITY["quiver_congress"], "ReportDate")
    assert len(added) == 0 and total == 1
    # the stored row is untouched; the weekly refetch is what brings the new return
    assert _read("quiver_congress", "A")["ExcessReturn"].iloc[0] == 1.2


# ── run_tails ────────────────────────────────────────────────────────────────

def test_run_tails_starts_from_the_parts_newest_key_minus_overlap_stalest_first():
    _part("f", "A", pd.DataFrame({"ticker": ["A"], "date": ["2026-09-15"], "v": [1]}))
    _part("f", "C", pd.DataFrame({"ticker": ["C"], "date": ["2026-09-10"], "v": [1]}))
    order, since_of = [], {}

    def fetch(k, since):
        order.append(k)
        since_of[k] = since
        return pd.DataFrame({"ticker": [k], "date": [since], "v": [9]})

    r = R.run_tails("f", ["A", "B", "C"], fetch, key_col="date", dedupe_on=["date"], sort_by="date",
                    start_default="2020-01-01", overlap_days=2, workers=1)
    assert since_of == {"A": "2026-09-13", "B": "2020-01-01", "C": "2026-09-08"}
    assert order == ["B", "C", "A"]                     # no part first, then the stalest
    assert r["ok"] == 3 and r["new_rows"] == 3 and not r["budget_stop"]
    assert len(_read("f", "A")) == 2                    # appended, history kept


def test_run_tails_budget_stop_submits_nothing_and_reports_it():
    calls = []
    r = R.run_tails("f", ["A"], lambda k, s: calls.append(k), key_col="date", dedupe_on=["date"],
                    sort_by="date", start_default="2020-01-01", deadline=time.time() - 1)
    assert calls == [] and r["ok"] == 0 and r["budget_stop"]


def test_run_tails_records_a_failed_key_and_keeps_going():
    def fetch(k, since):
        if k == "BAD":
            raise RuntimeError("nope")
        return pd.DataFrame({"ticker": [k], "date": ["2026-09-22"]})

    r = R.run_tails("f", ["BAD", "OK"], fetch, key_col="date", dedupe_on=["date"], sort_by="date",
                    start_default="2020-01-01", workers=1)
    assert r["ok"] == 1 and r["failed"] == 1
    assert "BAD" in deep.Manifest("f").failed


# ── run_keys(force=True) ─────────────────────────────────────────────────────

def test_run_keys_force_refetches_everything_stalest_first():
    calls = []

    def fn(k):
        calls.append(k)
        return pd.DataFrame({"k": [k]})

    deep.run_keys("fam", ["A", "B"], fn, workers=1)
    m = deep.Manifest("fam")
    m.done["A"]["at"] = "2026-09-23T00:00:10+00:00"
    m.done["B"]["at"] = "2026-09-23T00:00:05+00:00"
    m.save()
    calls.clear()
    deep.run_keys("fam", ["A", "B"], fn, workers=1)            # nothing pending: the ingest semantics
    assert calls == []
    deep.run_keys("fam", ["A", "B"], fn, workers=1, force=True)
    assert calls == ["B", "A"]                                 # stalest first


# ── cadence + state ──────────────────────────────────────────────────────────

def test_due_by_cadence():
    now = datetime(2026, 9, 23, 5, tzinfo=timezone.utc)
    assert R.due("x", R.DAILY, {}, now)
    st = {"x": {"at": (now - timedelta(hours=19)).isoformat()}}
    assert not R.due("x", R.DAILY, st, now)
    st = {"x": {"at": (now - timedelta(hours=21)).isoformat()}}
    assert R.due("x", R.DAILY, st, now)
    assert not R.due("x", R.WEEKLY, st, now)
    assert R.due("x", R.DAILY, {"x": {"at": "garbage"}}, now)


def test_run_marks_only_the_families_that_completed(monkeypatch):
    def ok(**_):
        return {"family": "ok", "new_rows": 3}

    def cut(**_):
        return {"family": "cut", "budget_stop": True}

    def boom(**_):
        raise RuntimeError("provider down")

    monkeypatch.setattr(R, "FAMILIES", [("ok", R.DAILY), ("cut", R.DAILY), ("boom", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["ok", "cut", "boom"])
    monkeypatch.setattr(R, "REFRESHERS", {"ok": ok, "cut": cut, "boom": boom})
    s = R.run(force=True, budget_seconds=0)
    assert s["completed"] == ["ok"] and s["not_finished"] == ["cut"] and s["failed"] == ["boom"]
    st = R.load_state()
    assert "ok" in st and "cut" not in st and "boom" not in st
    # next night: 'ok' is not due, the cut and the failed one are
    s2 = R.run(budget_seconds=0)
    assert s2["not_due"] == ["ok"] and set(s2["ran"]) == {"cut", "boom"}


def test_run_summary_does_not_mistake_a_failed_count_for_a_failure(monkeypatch):
    """Every runner reports a per-key ``failed`` COUNT; the first full run
    listed 15 healthy families as failed on that key and exited 2."""
    monkeypatch.setattr(R, "FAMILIES", [("x", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["x"])
    monkeypatch.setattr(R, "REFRESHERS", {"x": lambda **k: {"family": "x", "ok": 3, "failed": 2, "new_rows": 1}})
    s = R.run(force=True, budget_seconds=0)
    assert s["completed"] == ["x"] and s["failed"] == [] and "x" in R.load_state()


def test_run_refuses_to_start_while_another_refresh_holds_the_lock(monkeypatch):
    called = []
    monkeypatch.setattr(R, "FAMILIES", [("x", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["x"])
    monkeypatch.setattr(R, "REFRESHERS", {"x": lambda **k: called.append(1) or {"family": "x"}})
    held = R._try_lock(deep.DEEP_DIR / R.LOCK_NAME)
    assert held is not None
    try:
        assert R._try_lock(deep.DEEP_DIR / R.LOCK_NAME) is None      # a second handle is refused
        s = R.run(force=True, budget_seconds=0)
        assert s.get("skipped") and called == []
    finally:
        R._unlock(held)
    s = R.run(force=True, budget_seconds=0)                            # released: runs, and releases again
    assert not s.get("skipped") and called == [1]
    again = R._try_lock(deep.DEEP_DIR / R.LOCK_NAME)
    assert again is not None
    R._unlock(again)


def test_run_refuses_an_unknown_family():
    with pytest.raises(SystemExit):
        R.run(["no_such_family"], budget_seconds=0)


# ── ALFRED incremental merge ─────────────────────────────────────────────────

def test_merge_vintage_update_keeps_the_true_first_print_and_takes_the_new_end():
    cols = ["series_id", "date", "realtime_start", "realtime_end", "value", "vintage"]
    existing = pd.DataFrame([["S", "2026-09-01", "2026-09-02", "9999-12-31", 1.0, False],
                             ["S", "2026-09-02", "2026-09-03", "9999-12-31", 2.0, True]], columns=cols)
    # the window [2026-09-18, ∞) clamps every older value's realtime_start to
    # its own start, reports 2.0 as superseded on 09-20, and brings the
    # revision 2.5 and a new observation
    new = pd.DataFrame([["S", "2026-09-01", "2026-09-18", "9999-12-31", 1.0, True],
                        ["S", "2026-09-02", "2026-09-18", "2026-09-20", 2.0, True],
                        ["S", "2026-09-02", "2026-09-20", "9999-12-31", 2.5, True],
                        ["S", "2026-09-03", "2026-09-19", "9999-12-31", 3.0, True]], columns=cols)
    out = R.merge_vintage_update(existing, new).set_index(["date", "value"])
    assert len(out) == 4
    assert out.loc[("2026-09-01", 1.0), "realtime_start"] == "2026-09-02"
    assert not out.loc[("2026-09-01", 1.0), "vintage"]       # the stored flag survives the window
    assert out.loc[("2026-09-03", 3.0), "vintage"]           # a first sighting takes the new flag
    assert out.loc[("2026-09-02", 2.0), "realtime_start"] == "2026-09-03"
    assert out.loc[("2026-09-02", 2.0), "realtime_end"] == "2026-09-20"
    assert out.loc[("2026-09-02", 2.5), "realtime_start"] == "2026-09-20"
    assert out.loc[("2026-09-03", 3.0), "realtime_start"] == "2026-09-19"
    # an empty window leaves the table as it was
    assert len(R.merge_vintage_update(existing, pd.DataFrame())) == 2


# ── Quiver live top-up ───────────────────────────────────────────────────────

def test_split_live_rows_shapes_like_the_historical_fetch():
    rows = [{"Ticker": "aapl", "Date": "2026-09-22", "Amount": 5, "Client": "X", "Registrant": "Y",
             "Issue": "I", "Specific_Issue": "S"},
            {"Ticker": "ZZZ", "Date": "2026-09-22", "Amount": 1, "Client": "X", "Registrant": "Y",
             "Issue": "I", "Specific_Issue": "S"}]
    per = R.split_live_rows(rows, ["AAPL"])
    assert list(per) == ["AAPL"]
    df = per["AAPL"]
    assert list(df.columns)[0] == "ticker" and "Ticker" not in df.columns
    assert df["ticker"].iloc[0] == "AAPL"
    assert R.split_live_rows([], ["AAPL"]) == {}


# ── insider tail: the seam, never today, recheck, prune ──────────────────────

def _bulk(seam: str) -> None:
    deep.write_parquet(pd.DataFrame({"ticker": ["A"], "filing_date": [seam], "quarter": ["2026q1"]}),
                       deep.DEEP_DIR / "form345.parquet")


def test_live_form345_start_is_the_seam_then_the_day_after_the_newest_part_with_a_recheck():
    today = date(2026, 9, 23)
    assert R.live_form345_start(today) is None                       # nothing at all
    _bulk("2026-03-31")
    assert R.live_form345_start(today) == date(2026, 4, 1)           # no live parts: the seam
    _part("form345_live", "2026-09-18", pd.DataFrame({"x": [1]}))
    assert R.live_form345_start(today, recheck_days=0) == date(2026, 9, 19)
    assert R.live_form345_start(today, recheck_days=5) == date(2026, 9, 18)
    # the recheck never reaches back past the seam
    assert R.live_form345_start(date(2026, 4, 2), recheck_days=30) == date(2026, 4, 1)


def test_run_days_skips_existing_parts_never_writes_an_empty_day_and_consolidates(monkeypatch):
    asked = []

    def fake_fetch_day(d, ciks=None, workers=4):
        asked.append(d)
        if d == date(2026, 9, 21):                        # no index that day
            return pd.DataFrame()
        return pd.DataFrame({"accession": [f"a-{d}"], "filing_date": [d.isoformat()]})

    monkeypatch.setattr(form4_live, "fetch_day", fake_fetch_day)
    _part("form345_live", "2026-09-18", pd.DataFrame({"accession": ["old"], "filing_date": ["2026-09-18"]}))
    r = form4_live.run_days(date(2026, 9, 17), date(2026, 9, 22))
    # weekdays only (09-19/20 is a weekend), the day on disk is skipped, the
    # index-less day leaves no part so it is asked again next time
    assert asked == [date(2026, 9, 17), date(2026, 9, 21), date(2026, 9, 22)]
    parts = deep.family_dir("form345_live") / "parts"
    assert sorted(p.stem for p in parts.glob("*.parquet")) == ["2026-09-17", "2026-09-18", "2026-09-22"]
    assert r["days"] == 2 and r["skipped"] == 1 and r["empty_days"] == 1 and not r["budget_stop"]
    assert len(deep.read_parquet(deep.DEEP_DIR / "form345_live.parquet")) == 3


def test_run_days_stops_at_the_deadline(monkeypatch):
    monkeypatch.setattr(form4_live, "fetch_day", lambda d, ciks=None, workers=4: pd.DataFrame({"a": [1]}))
    r = form4_live.run_days(date(2026, 9, 21), date(2026, 9, 22), deadline=time.time() - 1)
    assert r["days"] == 0 and r["budget_stop"]


def test_refresh_form345_live_never_asks_for_today(monkeypatch):
    _bulk("2026-03-31")
    seen = {}

    def fake_run_days(start, end, **kw):
        seen["start"], seen["end"] = start, end
        return {"days": 0, "budget_stop": False}

    monkeypatch.setattr(form4_live, "run_days", fake_run_days)
    R.refresh_form345_live()
    assert seen["end"] == date.today() - timedelta(days=1)
    assert seen["start"] <= seen["end"]


def test_prune_live_parts_covered_by_a_newly_published_bulk_quarter():
    _bulk("2026-06-30")
    for d in ("2026-06-29", "2026-06-30", "2026-07-01"):
        _part("form345_live", d, pd.DataFrame({"accession": [d]}))
    assert R.prune_live_parts_covered_by_bulk() == 2
    parts = deep.family_dir("form345_live") / "parts"
    assert [p.stem for p in parts.glob("*.parquet")] == ["2026-07-01"]
    assert R.prune_live_parts_covered_by_bulk() == 0


# ── sec_filings → companyfacts trigger ───────────────────────────────────────

def test_sec_filings_refresh_flags_tickers_with_new_xbrl_filings(monkeypatch):
    from src.data.deep import sec
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A", "B"])
    monkeypatch.setattr(sec, "cik_map", lambda: {"A": "0000000001", "B": "0000000002"})
    _part("sec_filings", "A", pd.DataFrame({"ticker": ["A"], "cik": ["0000000001"], "accession": ["old"],
                                            "filing_date": ["2026-09-10"], "form": ["8-K"], "is_xbrl": [0]}))

    def fake_fetch(cik, tk, recent_only=False):
        assert recent_only
        new = {"A": ("new-10q", "10-Q", 1), "B": ("b-8k", "8-K", 0)}[tk]
        return pd.DataFrame({"ticker": [tk], "cik": [cik], "accession": [new[0]],
                             "filing_date": ["2026-09-22"], "form": [new[1]], "is_xbrl": [new[2]]})

    monkeypatch.setattr(sec, "fetch_filings", fake_fetch)
    r = R.refresh_sec_filings(workers=1)
    assert r["xbrl_tickers"] == ["A"]                   # B's new filing carries no XBRL
    assert len(_read("sec_filings", "A")) == 2 and len(_read("sec_filings", "B")) == 1


def test_companyfacts_stale_tickers_compares_financial_xbrl_filings_to_facts():
    def filings(tk, rows):
        _part("sec_filings", tk, pd.DataFrame({"ticker": [tk] * len(rows),
                                               "accession": [f"{tk}-{i}" for i in range(len(rows))],
                                               "filing_date": [r[0] for r in rows],
                                               "form": [r[1] for r in rows],
                                               "is_xbrl": [r[2] for r in rows]}))

    filings("A", [("2026-09-22", "10-Q", 1)])                 # 10-Q after its facts -> stale
    filings("B", [("2026-09-22", "8-K", 1)])                  # an 8-K cover page is not a fact
    filings("C", [("2026-06-30", "10-K", 1)])                 # facts already cover it
    filings("D", [("2026-09-22", "10-Q", 0)])                 # no XBRL instance
    filings("E", [("2026-09-22", "10-K/A", 1)])               # never had facts -> stale
    for tk in ("A", "B", "C", "D"):
        _part("companyfacts", tk, pd.DataFrame({"ticker": [tk], "filed": ["2026-08-05"], "val": [1.0]}))
    assert R.companyfacts_stale_tickers() == ["A", "E"]


def test_refresh_companyfacts_refetches_only_the_stale_names(monkeypatch):
    from src.data.deep import sec
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A", "C"])
    monkeypatch.setattr(sec, "cik_map", lambda: {"A": "0000000001", "C": "0000000003"})
    monkeypatch.setattr(R, "companyfacts_stale_tickers", lambda: ["A"])
    man = deep.Manifest("companyfacts")
    for tk in ("A", "C"):
        man.mark_done(tk, 1)
    man.save()
    fetched = []

    def fake_facts(tk):
        fetched.append(tk)
        return pd.DataFrame({"ticker": [tk], "filed": ["2026-09-22"], "val": [2.0]})

    monkeypatch.setattr(sec, "facts_for_ticker", fake_facts)
    r = R.refresh_companyfacts(workers=1)
    assert fetched == ["A"] and r["ok"] == 1


def test_refresh_companyfacts_keeps_the_stored_part_when_the_refetch_shrinks(monkeypatch):
    from src.data.deep import sec
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A"])
    monkeypatch.setattr(sec, "cik_map", lambda: {"A": "0000000001"})
    monkeypatch.setattr(R, "companyfacts_stale_tickers", lambda: ["A"])
    _part("companyfacts", "A", pd.DataFrame({"ticker": ["A"] * 100, "filed": ["2026-08-05"] * 100,
                                             "val": [1.0] * 100}))
    man = deep.Manifest("companyfacts")
    man.mark_done("A", 100)
    man.save()
    monkeypatch.setattr(sec, "facts_for_ticker",
                        lambda tk: pd.DataFrame({"ticker": [tk], "filed": ["2026-09-22"], "val": [2.0]}))
    r = R.refresh_companyfacts(workers=1)
    assert r["ok"] == 1 and len(_read("companyfacts", "A")) == 100


# ── refetch families never lose history ──────────────────────────────────────

def test_refresh_yf_merges_the_three_tables_and_reports_a_budget_stop(monkeypatch):
    from src.data.deep import yf_deep
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A"])
    _part("yf_earnings", "A", pd.DataFrame({"ticker": ["A"], "event_ts": pd.to_datetime(["2016-01-01"]),
                                            "eps_estimate": [1.0], "eps_reported": [1.1], "surprise_pct": [10.0]}))

    def fake(tk):
        return {"yf_earnings": pd.DataFrame({"ticker": [tk], "event_ts": pd.to_datetime(["2026-10-01"]),
                                             "eps_estimate": [2.0], "eps_reported": [float("nan")],
                                             "surprise_pct": [float("nan")]}),
                "yf_shares": pd.DataFrame({"ticker": [tk], "date": ["2026-09-22"], "shares": [100.0]})}

    monkeypatch.setattr(yf_deep, "fetch_ticker", fake)
    r = R.refresh_yf()
    assert r["ok"] == 1 and r["new_rows"] == 2 and not r["budget_stop"]
    assert len(_read("yf_earnings", "A")) == 2          # the 2016 event the window no longer carries survives
    assert len(_read("yf_shares", "A")) == 1
    r2 = R.refresh_yf(deadline=time.time() - 1)
    assert r2["ok"] == 0 and r2["budget_stop"]


def test_refresh_context_keeps_rows_a_refetch_lacks(monkeypatch):
    from src.data.deep import context
    deep.write_parquet(pd.DataFrame({"symbol": ["SPY", "SPY"], "date": ["2026-09-18", "2026-09-19"],
                                     "close": [1.0, 2.0]}), deep.DEEP_DIR / "market_daily.parquet")
    monkeypatch.setattr(context, "market_daily",
                        lambda: pd.DataFrame({"symbol": ["SPY"], "date": ["2026-09-22"], "close": [3.0]}))
    monkeypatch.setattr(context, "fama_french", lambda: pd.DataFrame())
    monkeypatch.setattr(context, "dix", lambda: pd.DataFrame())
    monkeypatch.setattr(context, "cot_tff_year",
                        lambda y: pd.DataFrame({"report_date": ["2026-09-15"], "year": [y]}))
    monkeypatch.setattr(R, "refresh_fred", lambda: 0)
    out = R.refresh_context()
    assert out["market_daily"] == 3 and out["fama_french"] == 0 and out["cot_tff"] == 1
    assert len(deep.read_parquet(deep.DEEP_DIR / "market_daily.parquet")) == 3


def test_refresh_quiver_dpi_upserts_and_history_survives(monkeypatch):
    from src.data.deep import quiver_deep
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A"])
    _part("quiver_dpi", "A", pd.DataFrame({"ticker": ["A"], "Date": ["2019-05-06"], "DPI": [0.3]}))
    monkeypatch.setattr(quiver_deep, "fetch",
                        lambda fam, tk: pd.DataFrame({"ticker": [tk], "Date": ["2026-09-22"], "DPI": [0.5]}))
    r = R.refresh_quiver_dpi()
    assert r["ok"] == 1 and r["new_rows"] == 1
    assert sorted(_read("quiver_dpi", "A")["Date"]) == ["2019-05-06", "2026-09-22"]


# ── status on an empty store ─────────────────────────────────────────────────

def test_newest_dates_is_none_everywhere_on_an_empty_store():
    out = R.newest_dates()
    assert out and all(v is None for v in out.values())


# ── the scheduler slot ───────────────────────────────────────────────────────

def test_scheduler_deep_refresh_slot_fires_once_per_day_including_weekends(monkeypatch):
    from config.settings import settings
    from src.scheduler import runner

    monkeypatch.setattr(settings, "enable_deep_refresh", True)
    at = _t(23, 45)
    now = datetime(2026, 9, 23, 23, 50)
    assert runner._should_run_deep_refresh(now, None, at)
    assert not runner._should_run_deep_refresh(now, now.date(), at)          # once per date
    assert not runner._should_run_deep_refresh(datetime(2026, 9, 23, 23, 0), None, at)
    assert runner._should_run_deep_refresh(datetime(2026, 9, 26, 23, 50), None, at)   # Saturday
    monkeypatch.setattr(settings, "enable_deep_refresh", False)
    assert not runner._should_run_deep_refresh(now, None, at)


def test_scheduler_preopen_slot_fires_on_market_days_only(monkeypatch):
    from config.settings import settings
    from src.scheduler import runner

    monkeypatch.setattr(settings, "enable_deep_preopen", True)
    at = _t(8, 30)
    wed = datetime(2026, 9, 23, 8, 31)
    assert runner._should_run_deep_preopen(wed, None, at)
    assert not runner._should_run_deep_preopen(wed, wed.date(), at)                    # once per day
    assert not runner._should_run_deep_preopen(datetime(2026, 9, 23, 8, 29), None, at)
    assert not runner._should_run_deep_preopen(datetime(2026, 9, 26, 8, 31), None, at)  # Saturday
    # a (re)start later in the day skips it: the slot state is in memory, so a
    # mid-session relaunch would otherwise launch the ~45-min fetch during trading
    assert runner._should_run_deep_preopen(datetime(2026, 9, 23, 9, 29), None, at)
    assert not runner._should_run_deep_preopen(datetime(2026, 9, 23, 9, 30), None, at)
    assert not runner._should_run_deep_preopen(datetime(2026, 9, 23, 15, 30), None, at)
    monkeypatch.setattr(settings, "enable_deep_preopen", False)
    assert not runner._should_run_deep_preopen(wed, None, at)


def test_preopen_profile_passes_through_to_the_subprocess(tmp_path, monkeypatch):
    import subprocess
    from config.settings import settings
    from src.scheduler import runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "deep_preopen_budget_seconds", 77)
    seen = {}

    class FakePopen:
        def __init__(self, args, **kw):
            seen["args"] = args
            kw["stdout"].write("deep refresh: 5s | completed [] | failed [] | not finished [] | not due []\n")
            self.pid = 4242

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    runner._deep_refresh_work("preopen")
    assert seen["args"][1:] == ["-m", "src.data.deep.refresh", "--profile", "preopen", "--budget-seconds", "77"]
    assert (tmp_path / runner.DEEP_PREOPEN_CONSOLE).exists()


def test_run_preopen_forces_the_fast_families_then_builds_the_snapshot(monkeypatch):
    from src.analysis import deep_features as dfe
    ran, built = [], []
    monkeypatch.setattr(R, "REFRESHERS", {f: (lambda f=f, **k: ran.append(f) or {"family": f})
                                          for f in R.FAMILY_NAMES})
    monkeypatch.setattr(dfe, "build_session_snapshot", lambda d, workers=6: built.append(d) or 7)
    monkeypatch.setattr(R, "extend_bars_30m", lambda d: built.append(("extend", d)) or {"extended": 1})
    monkeypatch.setattr(R, "today_session_day", lambda: 20_000)
    s = R.run_preopen(budget_seconds=0)
    assert sorted(ran) == sorted(R.PREOPEN_FAMILIES)          # forced, every one
    for lane in set(R.PREOPEN_LANES.values()):                # FAMILIES order inside a lane
        fams = [f for f in R.PREOPEN_FAMILIES if R.PREOPEN_LANES[f] == lane]
        assert [f for f in ran if f in fams] == fams
    # the 30-minute store through the previous session FIRST: the snapshot's
    # price-dependent features are read on that grid
    assert built == [("extend", 20_000), 20_000, 19_999]      # today, and yesterday's missing snapshot
    assert s["snapshots"] == {"20000": 7, "19999": 7}
    # the fast families are stamped, so the nightly finds them not due
    st = R.load_state()
    assert all(f in st for f in R.PREOPEN_FAMILIES)


def test_run_preopen_catches_up_the_previous_SESSION_not_the_previous_day(monkeypatch):
    """On a Monday the pre-market ticks score Friday's last bar: Friday's
    snapshot is the one to backfill, never Sunday's."""
    from src.analysis import deep_features as dfe
    built = []
    monkeypatch.setattr(R, "REFRESHERS", {f: (lambda f=f, **k: {"family": f}) for f in R.FAMILY_NAMES})
    monkeypatch.setattr(dfe, "build_session_snapshot", lambda d, workers=6: built.append(d) or 7)
    monkeypatch.setattr(R, "extend_bars_30m", lambda d: {})
    monkeypatch.setattr(R, "today_session_day", lambda: 20_003)          # Monday 2024-10-07
    R.run_preopen(budget_seconds=0)
    assert built == [20_003, 20_000]                                     # Friday 2024-10-04


def test_scheduler_runs_the_refresh_with_its_output_on_a_file_never_a_pipe(tmp_path, monkeypatch):
    """A pipe dies with the scheduler (the broker watchdog kills it inside
    the refresh window on most nights); a file handle survives it."""
    import subprocess
    from config.settings import settings
    from src.scheduler import runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "deep_refresh_budget_seconds", 123)
    seen = {}

    class FakePopen:
        def __init__(self, args, **kw):
            seen["args"], seen["kw"] = args, kw
            kw["stdout"].write("some noise\ndeep refresh: 5s | completed ['x'] | failed [] | "
                               "not finished [] | not due []\n")
            self.pid = 4242

        def wait(self, timeout=None):
            seen["timeout"] = timeout
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    runner._deep_refresh_work()
    assert seen["args"][1:] == ["-m", "src.data.deep.refresh", "--profile", "nightly", "--budget-seconds", "123"]
    assert hasattr(seen["kw"]["stdout"], "fileno") and seen["kw"]["stderr"] is subprocess.STDOUT
    assert "capture_output" not in seen["kw"]
    assert seen["kw"]["env"]["PYTHONIOENCODING"] == "utf-8"       # the file is read back as utf-8
    assert seen["timeout"] == 123 + 900
    assert "deep refresh: 5s" in (tmp_path / runner.DEEP_REFRESH_CONSOLE).read_text(encoding="utf-8")


def test_extend_bars_30m_reaches_the_previous_session_not_today(monkeypatch):
    """The pre-open extension asks for the store through the session BEFORE the
    snapshot's day (Friday's on a Monday), never today's (no bar has traded)."""
    from src.data import intraday_store
    seen = {}
    monkeypatch.setattr(intraday_store, "extend_deep_30m",
                        lambda tickers, **kw: seen.update(kw, n=len(list(tickers))) or {})
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA", "BBB"])
    R.extend_bars_30m(20_003)                                            # Monday 2024-10-07
    assert seen["today"] == date(2024, 10, 4) and seen["min_age_days"] == 0 and seen["n"] == 2


def test_today_session_day_runs_unpatched():
    """Every other pre-open test patches `today_session_day` out — which is how
    a NameError inside it (numpy never imported) shipped and killed every
    pre-open snapshot from 2026-09-23 to 09-25. Call it for real."""
    import numpy as np
    d = R.today_session_day()
    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    assert isinstance(d, int)
    assert d == int(now_et.to_datetime64().astype("datetime64[D]").astype(np.int64))


# ── the pre-open families in concurrent lanes (2026-09-29) ───────────────────
# 09-28 the seven families ran one after another for 47 min (sec_filings alone
# 16: EDGAR's 0.3 s spacing) and the scheduler's kill caught the snapshot build.

def test_lanes_run_side_by_side_and_keep_their_own_order(monkeypatch):
    import threading
    both = threading.Barrier(2, timeout=5)      # only passes when the two lanes overlap
    log = []

    def fam(name, meet=False):
        def _f(**_):
            log.append(("start", name))
            if meet:
                both.wait()                     # BrokenBarrierError when run one after another
            log.append(("end", name))
            return {"family": name, "new_rows": 1}
        return _f

    monkeypatch.setattr(R, "FAMILIES", [("a1", R.DAILY), ("b1", R.DAILY), ("a2", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["a1", "b1", "a2"])
    monkeypatch.setattr(R, "REFRESHERS", {"a1": fam("a1", meet=True), "b1": fam("b1", meet=True),
                                          "a2": fam("a2")})
    s = R.run(force=True, budget_seconds=0, lanes={"a1": "a", "a2": "a", "b1": "b"})
    assert set(s["completed"]) == {"a1", "b1", "a2"} and s["failed"] == []
    assert log.index(("end", "a1")) < log.index(("start", "a2"))      # a lane is sequential
    st = R.load_state()
    assert all(f in st for f in ("a1", "b1", "a2"))                   # every one stamped
    # without lanes the same families run one after another: the barrier breaks
    both.reset()
    log.clear()
    s = R.run(force=True, budget_seconds=0)
    assert "a1" in s["failed"]


def test_preopen_lanes_cover_every_preopen_family_and_split_the_limiters():
    assert set(R.PREOPEN_LANES) == set(R.PREOPEN_FAMILIES)
    lanes = R.PREOPEN_LANES
    assert lanes["form345_live"] == lanes["sec_filings"]              # one EDGAR limiter = one lane
    assert len(set(lanes.values())) >= 3


def test_the_deep_slots_fire_on_their_own_timer_once_per_day(monkeypatch):
    from config.settings import settings
    from src.scheduler import runner
    monkeypatch.setattr(settings, "enable_deep_preopen", True)
    monkeypatch.setattr(settings, "enable_deep_refresh", True)
    fired = []
    monkeypatch.setattr(runner, "_run_deep_preopen", lambda: fired.append("preopen"))
    monkeypatch.setattr(runner, "_run_deep_refresh", lambda: fired.append("refresh"))
    st = {}
    for hhmm in ((8, 29), (8, 30), (8, 31), (8, 45)):          # a tick running 08:30-08:51 changes nothing
        runner._deep_slots_tick(datetime(2026, 9, 29, *hhmm), st, _t(23, 45), _t(8, 30))
    assert fired == ["preopen"]
    runner._deep_slots_tick(datetime(2026, 9, 29, 23, 45, 10), st, _t(23, 45), _t(8, 30))
    runner._deep_slots_tick(datetime(2026, 9, 29, 23, 50), st, _t(23, 45), _t(8, 30))
    assert fired == ["preopen", "refresh"]


def test_the_poll_loop_no_longer_triggers_the_deep_slots():
    import inspect
    from src.scheduler import runner
    src = inspect.getsource(runner.start_scheduler)
    assert "_start_deep_slot_timer(" in src
    assert "_should_run_deep_preopen(" not in src and "_should_run_deep_refresh(" not in src


def test_run_preopen_runs_its_families_in_the_lanes(monkeypatch):
    seen = {}
    monkeypatch.setattr(R, "run", lambda fams, **kw: seen.update(kw, fams=list(fams)) or {"skipped": "x"})
    R.run_preopen(budget_seconds=0)
    assert seen["fams"] == R.PREOPEN_FAMILIES and seen["lanes"] is R.PREOPEN_LANES and seen["force"] is True


def test_a_refresh_killed_at_its_budget_takes_its_process_tree_down(tmp_path, monkeypatch):
    """2026-09-28: the pre-open run killed at its budget left the snapshot
    build's six multiprocessing workers alive for 22 hours (Windows never kills
    a killed parent's children)."""
    import os
    import subprocess
    from config.settings import settings
    from src.scheduler import runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "deep_preopen_budget_seconds", 1)
    monkeypatch.setattr(os, "name", "nt")
    killed, ran = [], []

    class FakePopen:
        def __init__(self, args, **kw):
            self.pid = 777

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("refresh", timeout)

        def kill(self):
            killed.append(self.pid)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: ran.append(list(args)))
    runner._deep_refresh_work("preopen")                     # never raises
    assert ["taskkill", "/PID", "777", "/T", "/F"] in ran      # the whole tree
    assert killed == [777]



# ══ the fast refresh without skipping a row (2026-09-29) ═══════════════════════
# user: "as fast as possible but without hurting the prediction power. So without
# skipping data ingestion". Every faster path stores exactly what the per-ticker
# path stores (parity measured on the live store: short volume / short interest
# 64/64 frames identical, 0 rows the bulk would add; EDGAR's live feed held every
# one of 1,076 universe filings in its span) and falls back to it when unsure.

def test_run_tails_serves_the_bulk_keys_and_fetches_the_rest_one_by_one(monkeypatch):
    fetched = []

    def one(tk, since):
        fetched.append((tk, since))
        return pd.DataFrame({"date": ["2026-09-28"], "v": [2.0]})

    def bulk(since_by):
        assert set(since_by) == {"A", "B", "C"}
        return {"A": pd.DataFrame({"date": ["2026-09-28"], "v": [1.0]}),
                "B": pd.DataFrame()}                     # served: no new rows for B

    r = R.run_tails("fam", ["A", "B", "C"], one, key_col="date", dedupe_on=["date"], sort_by="date",
                    start_default="2024-01-01", workers=2, bulk=bulk)
    assert [k for k, _ in fetched] == ["C"]              # only the key the bulk left out
    assert r["ok"] == 3 and r["new_rows"] == 2
    assert float(_read("fam", "A")["v"].iloc[0]) == 1.0

    def broken(since_by):
        raise RuntimeError("provider down")
    fetched.clear()
    R.run_tails("fam2", ["A", "B"], one, key_col="date", dedupe_on=["date"], sort_by="date",
                start_default="2024-01-01", workers=1, bulk=broken)
    assert sorted(k for k, _ in fetched) == ["A", "B"]   # a failed bulk -> every key on its own


def test_bulk_by_symbol_matches_exactly_and_cuts_each_key_at_its_own_tail(monkeypatch):
    from datetime import date as _date, timedelta as _td
    from src.data.deep import polygon_deep as pdp
    today = _date.today()
    d = lambda n: (today - _td(days=n)).isoformat()      # noqa: E731
    rows = [{"ticker": "BCPC", "date": d(2), "v": 1}, {"ticker": "BCpC", "date": d(2), "v": 99},
            {"ticker": "BRK.B", "date": d(1), "v": 5}, {"ticker": "BRK.B", "date": d(3), "v": 4},
            {"ticker": "ZZZ", "date": d(1), "v": 7}]
    seen = {}

    def fake_all(path, params, max_pages=200):
        seen.update(params)
        return rows, True
    monkeypatch.setattr(pdp, "_paginate_all", fake_all)
    frame = lambda res, tk: pd.DataFrame(res).assign(ticker=tk.upper()) if res else pd.DataFrame()  # noqa: E731
    since = {"BCPC": d(5), "BRK-B": d(2), "BRK.B": d(5), "OLD": d(400), "NEW": ""}
    out = pdp._bulk_by_symbol("/x", "date", since, 10, frame)
    assert set(out) == {"BCPC", "BRK-B", "BRK.B"}         # OLD and NEW fall back to their own calls
    assert seen["date.gte"] == d(5)                        # the oldest SERVED tail
    assert out["BCPC"]["v"].tolist() == [1]                # never the preferred BCpC
    assert out["BRK-B"]["v"].tolist() == [5]               # cut at its OWN since (d2)
    assert sorted(out["BRK.B"]["v"].tolist()) == [4, 5]    # both spellings of BRK.B get rows
    monkeypatch.setattr(pdp, "_paginate_all", lambda *a, **k: (rows, False))
    assert pdp._bulk_by_symbol("/x", "date", since, 10, frame) == {}   # incomplete -> nobody


def test_paginate_all_says_when_it_did_not_reach_the_end(monkeypatch):
    from src.data.deep import polygon_deep as pdp
    from src.data import polygon_client as pc
    monkeypatch.setattr(pdp, "_get", lambda path, params: None)
    assert pdp._paginate_all("/x", {}) == ([], False)                     # the first call failed
    monkeypatch.setattr(pdp, "_get", lambda path, params: {"results": [1], "next_url": "u"})
    monkeypatch.setattr(pc, "_follow_next_url", lambda u: (_ for _ in ()).throw(OSError("reset")))
    assert pdp._paginate_all("/x", {}) == ([1], False)                    # a cursor page failed
    monkeypatch.setattr(pc, "_follow_next_url", lambda u: {"results": [2]})
    assert pdp._paginate_all("/x", {}) == ([1, 2], True)                  # reached the end
    monkeypatch.setattr(pc, "_follow_next_url", lambda u: {"results": [2], "next_url": "u"})
    assert pdp._paginate_all("/x", {}, max_pages=3)[1] is False           # pages ran out


def test_a_failed_polygon_call_is_retried_once(monkeypatch):
    from src.data.deep import polygon_deep as pdp
    from src.data import polygon_client as pc
    calls = []
    monkeypatch.setattr(pdp.time, "sleep", lambda s: None)
    monkeypatch.setattr(pc, "_get", lambda path, params: calls.append(1) or (None if len(calls) == 1 else {"ok": 1}))
    assert pdp._get("/x", {}) == {"ok": 1} and len(calls) == 2


def _feed_page(entries):
    return "<feed>" + "".join(
        f"<entry><title>8-K - CO ({cik}) (Filer)</title>"
        f'<link href="https://www.sec.gov/Archives/edgar/data/{int(cik)}/000000000026000001/x-index.htm"/>'
        f"<updated>{ts}</updated></entry>" for cik, ts in entries) + "</feed>"


def test_the_live_feed_returns_every_filer_back_to_the_instant_or_none(monkeypatch):
    from datetime import datetime as _dt, timezone as _tz
    from types import SimpleNamespace
    from src.data.deep import sec
    pages = [_feed_page([("0000000001", "2026-09-29T08:10:00-04:00"), ("0000000002", "2026-09-29T07:00:00-04:00")]),
             _feed_page([("0000000003", "2026-09-29T06:05:00-04:00"), ("0000000004", "2026-09-28T23:00:00-04:00")])]
    monkeypatch.setattr(sec, "http_get", lambda url, **k: SimpleNamespace(
        status_code=200, text=pages[int(url.split("start=")[1].split("&")[0]) // 100]))
    since = _dt(2026, 9, 29, 10, 0, tzinfo=_tz.utc)                        # 06:00 ET
    assert sec.current_filer_ciks(since) == {"0000000001", "0000000002", "0000000003"}
    # nobody filed since the instant: an EMPTY set (proven), not "unknown"
    assert sec.current_filer_ciks(_dt(2026, 9, 29, 13, 0, tzinfo=_tz.utc)) == set()
    # the listing ends before reaching further back -> cannot prove coverage
    monkeypatch.setattr(sec, "http_get", lambda url, **k: SimpleNamespace(
        status_code=200, text=pages[0] if "start=0&" in url else "<feed></feed>"))
    assert sec.current_filer_ciks(since) is None
    monkeypatch.setattr(sec, "http_get", lambda url, **k: SimpleNamespace(status_code=503, text=""))
    assert sec.current_filer_ciks(since) is None


def test_the_preopen_sec_pass_rereads_only_the_live_feeds_filers(monkeypatch):
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from src.data.deep import sec
    monkeypatch.setattr(deep, "deep_universe", lambda refresh=False: ["A", "B", "C"])
    monkeypatch.setattr(sec, "cik_map", lambda: {"A": "0000000001", "B": "0000000002", "C": "0000000003"})
    fetched = []

    def fake_fetch(cik, tk, recent_only=False):
        fetched.append(tk)
        return pd.DataFrame({"ticker": [tk], "cik": [cik], "accession": [f"{tk}-1"],
                             "filing_date": ["2026-09-29"], "form": ["8-K"], "is_xbrl": [0]})
    monkeypatch.setattr(sec, "fetch_filings", fake_fetch)
    done = _dt.now(_tz.utc) - _td(hours=9)
    R.save_state({"sec_filings": {"at": done.isoformat(), "seconds": 600.0}})
    asked = {}
    monkeypatch.setattr(sec, "current_filer_ciks", lambda since: asked.update(since=since) or {"0000000002"})
    s = R.run(["sec_filings"], force=True, budget_seconds=0, profile="preopen")
    assert fetched == ["B"]                                                # only the feed's filer
    assert asked["since"] == done - _td(seconds=600 + 15 * 60)             # the full pass's START, less 15 min
    st = R.load_state()
    assert "sec_filings_incremental" in st and st["sec_filings"]["at"] == done.isoformat()   # anchor untouched
    assert s["completed"] == ["sec_filings"]
    # no provable coverage -> the full pass, stamped as one
    fetched.clear()
    monkeypatch.setattr(sec, "current_filer_ciks", lambda since: None)
    R.run(["sec_filings"], force=True, budget_seconds=0, profile="preopen")
    assert sorted(fetched) == ["A", "B", "C"] and R.load_state()["sec_filings"]["at"] != done.isoformat()
    # the nightly always takes the full pass
    fetched.clear()
    monkeypatch.setattr(sec, "current_filer_ciks", lambda since: (_ for _ in ()).throw(AssertionError("no feed at night")))
    R.run(["sec_filings"], force=True, budget_seconds=0, profile="nightly")
    assert sorted(fetched) == ["A", "B", "C"]


def test_full_pass_anchor_is_the_last_full_passs_start():
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    assert R.full_pass_anchor("sec_filings") is None
    at = _dt(2026, 9, 29, 4, 30, tzinfo=_tz.utc)
    R.save_state({"sec_filings": {"at": at.isoformat(), "seconds": 660.0}})
    assert R.full_pass_anchor("sec_filings") == at - _td(seconds=660 + 900)


def test_the_nightly_takes_the_full_sec_pass_even_after_a_preopen_stamp():
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    now = _dt(2026, 9, 29, 3, 45, tzinfo=_tz.utc)                          # 23:45 ET
    st = {"sec_filings": {"at": (now - _td(hours=15)).isoformat()}}         # a pre-open full pass, 08:49
    assert R.due("sec_filings", dict(R.FAMILIES)["sec_filings"], st, now)


def test_nightly_lanes_run_universe_first_and_keep_the_dependencies_in_one_lane():
    lanes = R.NIGHTLY_LANES
    assert set(R.FAMILY_NAMES) <= set(lanes), set(R.FAMILY_NAMES) - set(lanes)
    assert lanes["universe"] == ""
    assert lanes["sec_filings"] == lanes["companyfacts"] == lanes["form345_live"]
    assert lanes["delisted"] == lanes["bars30m_full"] and lanes["wiki"] not in (lanes["yf"], lanes["quiver_dpi"])


def test_a_lane_of_its_own_first_family_runs_before_the_lanes_start(monkeypatch):
    import time as _time
    log = []

    def fam(f):
        def _f(**_):
            log.append(("start", f))
            if f == "u":
                _time.sleep(0.3)
            log.append(("end", f))
            return {"family": f}
        return _f
    monkeypatch.setattr(R, "FAMILIES", [("u", R.DAILY), ("a", R.DAILY), ("b", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["u", "a", "b"])
    monkeypatch.setattr(R, "REFRESHERS", {f: fam(f) for f in "uab"})
    R.run(force=True, budget_seconds=0, lanes={"u": "", "a": "x", "b": "y"})
    u_end = log.index(("end", "u"))
    assert log.index(("start", "a")) > u_end and log.index(("start", "b")) > u_end


def test_the_profile_and_per_family_workers_reach_the_refreshers(monkeypatch):
    seen = {}
    monkeypatch.setattr(R, "FAMILIES", [("a", R.DAILY), ("b", R.DAILY)])
    monkeypatch.setattr(R, "FAMILY_NAMES", ["a", "b"])
    monkeypatch.setattr(R, "REFRESHERS", {f: (lambda f=f, **k: seen.update({f: k}) or {"family": f}) for f in "ab"})
    R.run(force=True, budget_seconds=0, workers=6, profile="preopen", family_workers={"b": 10})
    assert seen["a"]["profile"] == "preopen" and seen["a"]["workers"] == 6 and seen["b"]["workers"] == 10


def test_the_preopen_extends_the_30m_store_beside_the_families(monkeypatch):
    import threading
    from src.analysis import deep_features as dfe
    together = threading.Barrier(2, timeout=5)    # passes only while the extension AND a family run
    order = []

    def family(**_):
        if not order.count("families"):
            together.wait()
        order.append("families")
        return {"family": "f"}

    def extend(d):
        together.wait()
        order.append("extend")
        return {"extended": 1}
    monkeypatch.setattr(R, "PREOPEN_LANES", {f: "one" for f in R.PREOPEN_FAMILIES})
    monkeypatch.setattr(R, "REFRESHERS", {f: family for f in R.FAMILY_NAMES})
    monkeypatch.setattr(R, "extend_bars_30m", extend)
    monkeypatch.setattr(dfe, "build_session_snapshot", lambda d, workers=6: order.append(("snapshot", workers)) or 1)
    monkeypatch.setattr(R, "today_session_day", lambda: 20_000)
    monkeypatch.setattr(dfe, "snapshot_path", lambda d: type("P", (), {"exists": lambda self: True})())
    s = R.run_preopen(budget_seconds=0)
    assert "extend" in order and order[-1] == ("snapshot", 10)             # after BOTH, with 10 workers
    assert s["extend_30m"] == {"extended": 1}


def test_the_nightly_cli_runs_in_lanes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(R, "run", lambda fams, **kw: seen.update(kw) or {"seconds": 1, "completed": [], "failed": [],
                                                                        "not_finished": [], "not_due": []})
    monkeypatch.setattr(R, "_below_normal_priority", lambda: None)
    R.main([])
    assert seen["lanes"] is R.NIGHTLY_LANES and seen["profile"] == "nightly"


def test_fred_updates_run_three_at_a_time_in_series_order(monkeypatch):
    import time as _time
    from config.settings import settings
    from src.data.deep import context
    monkeypatch.setattr(settings, "fred_api_key", "k")
    monkeypatch.setattr(context, "FRED_SERIES", ["S1", "S2", "S3", "S4"])
    deep.write_parquet(pd.DataFrame({"series": ["S1"], "date": ["2026-01-01"], "realtime_start": ["2026-09-20"],
                                     "value": [1.0]}), deep.DEEP_DIR / "fred_vintages.parquet")
    got = []

    def upd(sid, key, since):
        _time.sleep({"S1": 0.3, "S2": 0.1, "S3": 0.2, "S4": 0.0}[sid])
        return pd.DataFrame({"series": [sid], "date": ["2026-09-01"], "realtime_start": ["2026-09-28"], "value": [2.0]})
    monkeypatch.setattr(context, "fred_series_update", upd)
    monkeypatch.setattr(R, "merge_vintage_update", lambda old, new: got.append(list(new["series"])) or old)
    R.refresh_fred()
    assert got == [["S1", "S2", "S3", "S4"]]                               # series order, whatever finished first
