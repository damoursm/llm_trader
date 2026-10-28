"""The data feeds (2026-10-25, `src/data/feeds/`): r/wallstreetbets from Arctic Shift, the OCC's options volume, the
FDA's Drugs@FDA export — fetched with injected answers (no network), archived once, read point in time; the runner's
lock, cadence and failure record; the scheduler's timer slot; the health notes."""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from datetime import date, datetime, timedelta, timezone

import pytest

from config import settings


class R:
    def __init__(self, status=200, js=None, text="", content=b""):
        self.status_code, self._js, self.text, self.content = status, js, text, content

    def json(self):
        return self._js


def _post(i, ts, title="x"):
    return {"id": f"p{i}", "created_utc": ts, "title": title, "score": 1, "num_comments": 0, "link_flair_text": None,
            "author": "someone"}


# ── WallStreetBets ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_wsb_fetch_pages_merge_and_resume():
    from src.data.feeds import wsb
    t0 = int(datetime(2026, 10, 20, 12, tzinfo=timezone.utc).timestamp())
    posts = [_post(i, t0 + i) for i in range(105)]
    calls = []

    def get(url, params=None):
        calls.append(dict(params))
        rows = [p for p in posts if p["created_utc"] >= params["after"] and p["created_utc"] < params["before"]]
        return R(js={"data": rows[:100]})
    st = {"after_utc": t0}
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    msg = wsb.run(st, now=now, http_get=get)
    assert "105 posts" in msg and st["after_utc"] == t0 + 104
    assert calls[0]["after"] == t0 - int(settings.feeds_wsb_overlap_minutes * 60)
    rows = wsb.read_day(date(2026, 10, 20))
    assert len(rows) == 105 and "author" not in rows[0]                # the author is never kept
    wsb.run(st, now=now, http_get=get)                                 # the overlap re-reads: merged by id
    assert len(wsb.read_day(date(2026, 10, 20))) == 105


def test_wsb_empty_store_starts_two_days_back():
    from src.data.feeds import wsb
    seen = {}

    def get(url, params=None):
        seen.update(params)
        return R(js={"data": []})
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    st = {}
    wsb.run(st, now=now, http_get=get)
    assert seen["after"] == int(now.timestamp()) - 2 * 86400 - int(settings.feeds_wsb_overlap_minutes * 60)


def test_wsb_error_answer_raises():
    from src.data.feeds import wsb
    with pytest.raises(RuntimeError):
        wsb.run({"after_utc": 0}, now=datetime(2026, 10, 20, tzinfo=timezone.utc),
                http_get=lambda url, params=None: R(js={"data": None, "error": "Timeout. Maybe slow down a bit"}))
    with pytest.raises(RuntimeError):
        wsb.run({"after_utc": 0}, now=datetime(2026, 10, 20, tzinfo=timezone.utc),
                http_get=lambda url, params=None: R(status=503))


def test_wsb_mentions_rules_and_missing_days():
    from src.data.feeds import wsb
    t = int(datetime(2026, 10, 19, 15, tzinfo=timezone.utc).timestamp())
    titles = ["GME to the moon", "$ON breakout", "ON the way up", "S&P is falling", "$S beats", "buy AMC and GME GME"]
    wsb.write_day(date(2026, 10, 19), [_post(i, t + i, tt) for i, tt in enumerate(titles)])
    wsb.write_day(date(2026, 10, 20), [_post(99, t + 86400, "GME again")])
    assert wsb.mentions("GME", date(2026, 10, 19), date(2026, 10, 20)) == 3     # once per title
    assert wsb.mentions("ON", date(2026, 10, 19), date(2026, 10, 19)) == 1      # a word: the cashtag only
    assert wsb.mentions("S", date(2026, 10, 19), date(2026, 10, 19)) == 1       # a single letter: the cashtag only
    assert wsb.mentions("AMC", date(2026, 10, 19), date(2026, 10, 19)) == 1
    assert wsb.mentions("GME", date(2026, 10, 18), date(2026, 10, 19)) is None  # a day not in the store


# ── WallStreetBets when the archive fails (2026-10-27): retries, the pre-cut loop, Reddit's API by post id ──────────

class FakeReddit:
    """Reddit's API as `wsb.fill` reads it: /r/all/new's newest ids, /api/info's posts by id (any subreddit)."""

    def __init__(self, posts, head):
        self.posts, self.head, self.asked = {p["id"]: p for p in posts}, head, []

    def get(self, path, params):
        from src.data.feeds import wsb
        if path == "/r/all/new":
            return {"data": {"children": [{"data": {"id": wsb.b36(self.head)}}]}}
        ids = [x[3:] for x in params["id"].split(",")]
        self.asked += ids
        return {"data": {"children": [{"data": self.posts[i]} for i in ids if i in self.posts]}}


def _reddit_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_feeds_wsb_reddit_fill", True)
    monkeypatch.setattr(settings, "reddit_client_id", "an-app")
    monkeypatch.setattr(settings, "reddit_client_secret", "its-secret")


def test_wsb_retries_keep_the_served_pages_and_resume(monkeypatch):
    from src.data.feeds import wsb
    monkeypatch.setattr(settings, "feeds_wsb_retries", 2)
    waits = []
    monkeypatch.setattr(wsb, "_sleep", waits.append)
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    monkeypatch.setattr(wsb, "_clock", lambda: now)
    t0 = int(datetime(2026, 10, 20, 12, tzinfo=timezone.utc).timestamp())
    posts = [_post(i, t0 + i) for i in range(150)]
    fail = {"n": 1}

    def get(url, params=None):
        if params["after"] > t0 and fail["n"]:                       # the second page fails once
            fail["n"] -= 1
            return R(status=530)
        return R(js={"data": [p for p in posts if params["after"] <= p["created_utc"] < params["before"]][:100]})
    st = {"after_utc": t0}
    wsb.run(st, now=now, http_get=get)
    assert waits == [settings.feeds_wsb_retry_seconds]                  # one retry, the first page's posts kept
    assert len(wsb.read_day(date(2026, 10, 20))) == 150 and st["after_utc"] == t0 + 149
    assert st["archive_error"] is None and st["archive_fail_since"] is None


def test_wsb_fill_reads_reddit_by_id_when_the_archive_fails_and_the_archive_copy_wins(monkeypatch):
    from src.data.feeds import wsb
    _reddit_on(monkeypatch)
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    monkeypatch.setattr(wsb, "_clock", lambda: now)
    t0 = int(datetime(2026, 10, 20, 12, tzinfo=timezone.utc).timestamp())
    base = int("1xb8f7v", 36)
    wsb.write_day(date(2026, 10, 20), [dict(_post(0, t0), id=wsb.b36(base))])          # the archive's newest post
    other = [{"id": wsb.b36(base + k), "created_utc": float(t0 + 20 * k), "title": f"t{k}", "subreddit": "AskReddit"}
             for k in range(1, 250)]
    ours = [{"id": wsb.b36(base + k), "created_utc": float(t0 + 20 * k), "title": "GME to the moon",
             "subreddit": "wallstreetbets", "score": 1, "num_comments": 0, "link_flair_text": None,
             "removed_by_category": "automod_filtered", "author": "someone"} for k in (7, 120, 240)]
    by = {p["id"]: p for p in other}
    by.update({p["id"]: p for p in ours})
    api = FakeReddit(list(by.values()), head=base + 249)
    st = {"after_utc": t0, "after_id": base}
    msg = wsb.run(st, now=now, http_get=lambda url, params=None: R(status=530), api=api)
    rows = [p for p in wsb.read_day(date(2026, 10, 20)) if p.get("src") == "reddit"]
    assert sorted(p["id"] for p in rows) == sorted(p["id"] for p in ours)       # WSB's only, the removed post too
    assert all("author" not in p for p in rows)
    assert len(api.asked) == 249 and st["fill_next_id"] == base + 250           # every id after the stored newest
    assert st["after_utc"] == t0 and st["archive_fail_since"]                    # the archive's resume point unmoved
    assert "Reddit's API 3 post(s)" in msg
    # the archive answers again: its copies replace the fallback's; the outage's span is due for a re-read
    arch = [dict(_post(0, int(p["created_utc"])), id=p["id"], title=p["title"]) for p in ours]
    later = now + timedelta(minutes=30)
    monkeypatch.setattr(wsb, "_clock", lambda: later)
    wsb.run(st, now=later, api=api, http_get=lambda url, params=None: R(
        js={"data": [p for p in arch if params["after"] <= p["created_utc"] < params["before"]]}))
    rows = wsb.read_day(date(2026, 10, 20))
    assert len(rows) == 4 and not any(p.get("src") for p in rows)
    assert st["archive_fail_since"] is None and len(st["rereads"]) == 1
    due = datetime.fromisoformat(st["rereads"][0][2])
    assert due == later + timedelta(hours=settings.feeds_wsb_reread_hours)


def test_wsb_a_fallback_copy_never_replaces_the_archives():
    from src.data.feeds import wsb
    t = int(datetime(2026, 10, 20, 12, tzinfo=timezone.utc).timestamp())
    wsb.write_day(date(2026, 10, 20), [_post(1, t, "archive title")])
    wsb.write_day(date(2026, 10, 20), [dict(_post(1, t, "fallback title"), src="reddit"),
                                       dict(_post(2, t, "only here"), src="reddit")])
    rows = {p["id"]: p for p in wsb.read_day(date(2026, 10, 20))}
    assert rows["p1"]["title"] == "archive title" and "src" not in rows["p1"] and rows["p2"]["src"] == "reddit"


def test_wsb_before_the_cut_the_archive_is_asked_every_minute_until_it_answers(monkeypatch):
    from src.data.feeds import wsb
    from src.utils import ET
    monkeypatch.setattr(settings, "feeds_wsb_precut_lead_minutes", 20.0)
    monkeypatch.setattr(settings, "feeds_wsb_precut_tail_minutes", 27.0)
    clk = {"t": datetime(2026, 10, 27, 8, 45, tzinfo=ET).astimezone(timezone.utc)}     # a Tuesday, before 09:00

    def sleep(s):
        clk["t"] += timedelta(seconds=s)
    monkeypatch.setattr(wsb, "_clock", lambda: clk["t"])
    monkeypatch.setattr(wsb, "_sleep", sleep)
    n = {"calls": 0}

    def get(url, params=None):
        n["calls"] += 1
        return R(status=530) if n["calls"] <= 4 else R(js={"data": []})
    st = {"after_utc": int(clk["t"].timestamp()) - 600}
    wsb.run(st, now=clk["t"], http_get=get)
    assert n["calls"] == 5 and clk["t"] == datetime(2026, 10, 27, 8, 49, tzinfo=ET).astimezone(timezone.utc)
    clk["t"] = datetime(2026, 10, 27, 10, 0, tzinfo=ET).astimezone(timezone.utc)       # outside the window: no loop
    with pytest.raises(RuntimeError, match="no Reddit app"):
        wsb.run(st, now=clk["t"], http_get=lambda url, params=None: R(status=530))
    assert st["archive_fail_since"]
    assert not wsb.in_precut(datetime(2026, 10, 31, 8, 50, tzinfo=ET))                 # a Saturday: no cut


def test_wsb_fills_before_the_cut_when_the_archive_answers_but_is_behind(monkeypatch):
    from src.data.feeds import wsb
    from src.utils import ET
    _reddit_on(monkeypatch)
    monkeypatch.setattr(settings, "feeds_wsb_precut_lead_minutes", 20.0)
    monkeypatch.setattr(settings, "feeds_wsb_precut_tail_minutes", 27.0)
    now = datetime(2026, 10, 27, 8, 50, tzinfo=ET).astimezone(timezone.utc)
    monkeypatch.setattr(wsb, "_clock", lambda: now)
    base = int("1xb8f7v", 36)
    t = int(now.timestamp()) - 600
    api = FakeReddit([{"id": wsb.b36(base + 5), "created_utc": float(t), "title": "$NVDA", "subreddit": "wallstreetbets"}],
                     head=base + 50)
    st = {"after_utc": int(now.timestamp()) - 90 * 60, "after_id": base}               # 90 minutes behind
    msg = wsb.run(st, now=now, http_get=lambda url, params=None: R(js={"data": []}), api=api)
    assert "before the cut" in msg and st["fill_next_id"] == base + 51 and st["fill_newest_utc"] == t


def test_reddit_api_refused_credentials_and_its_rate_limit(monkeypatch):
    from src.data.feeds import wsb
    _reddit_on(monkeypatch)

    class Client:
        def __init__(self, token_status=200):
            self.token_status = token_status

        def post(self, url, data=None, auth=None, headers=None):
            return R(status=self.token_status, js={"access_token": "tok"} if self.token_status == 200 else {})

        def get(self, url, params=None, headers=None):
            r = R(js={"data": {"children": []}})
            r.headers = {"x-ratelimit-remaining": "1", "x-ratelimit-reset": "42"}
            return r
    with pytest.raises(RuntimeError, match="credentials were refused"):
        wsb.RedditAPI(client=Client(401)).get("/r/all/new", {})
    waits = []
    monkeypatch.setattr(wsb, "_sleep", waits.append)
    wsb.RedditAPI(client=Client()).get("/r/all/new", {})
    assert 43.0 in waits                                                 # one request left: wait for the reset


# ── the OCC ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _occ_text(rows):
    from src.data.feeds import occ
    return occ.HEADER + "\n" + "\n".join(f"{q},{u},{u},{a},{p},CBOE,10/23/2026" for q, u, a, p in rows) + "\n"


def test_occ_check_aggregate_and_archive_once():
    from src.data.feeds import occ
    text = _occ_text([(1000, "GME", "C", "C"), (500, "GME", "C", "C"), (300, "GME", "F", "P"), (2000, "AAPL", "C", "C")])
    assert occ.check(text) == 4
    a = occ.aggregate(text)
    assert a[("GME", "C", "C")] == 1500 and a[("GME", "F", "P")] == 300
    with pytest.raises(ValueError):
        occ.check("<html>maintenance</html>")
    d = date(2026, 10, 23)
    occ._write(d, text)
    assert occ.path_for(d).exists() and occ.read_agg(d)[("AAPL", "C", "C")] == 2000


def test_occ_due_window_calendar_and_retry(monkeypatch):
    from src.data.feeds import occ
    monkeypatch.setattr(settings, "feeds_occ_history_days", 7)
    now = datetime(2026, 10, 21, 22, tzinfo=timezone.utc)            # Wednesday 18:00 ET: Wednesday is complete
    due = occ.due(now, {})
    assert due == [date(2026, 10, 14), date(2026, 10, 15), date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 20),
                   date(2026, 10, 21)]
    occ._write(date(2026, 10, 20), _occ_text([(5, "X", "C", "C")]))
    st = {"retry": {"2026-10-21": (now - timedelta(minutes=10)).isoformat()}}
    assert date(2026, 10, 20) not in occ.due(now, st) and date(2026, 10, 21) not in occ.due(now, st)
    st = {"retry": {"2026-10-21": (now - timedelta(hours=2)).isoformat()}}
    assert date(2026, 10, 21) in occ.due(now, st)
    early = datetime(2026, 10, 21, 18, tzinfo=timezone.utc)          # 14:00 ET: Wednesday not complete
    assert occ.last_completed(early) == date(2026, 10, 20)


def test_occ_run_archives_published_and_waits_for_the_rest(monkeypatch):
    from src.data.feeds import occ
    monkeypatch.setattr(settings, "feeds_occ_history_days", 3)
    now = datetime(2026, 10, 21, 22, tzinfo=timezone.utc)

    def get(url, params=None):
        if params["reportDate"] == "20261021":
            return R(text=occ.NONE_TEXT)
        return R(text=_occ_text([(7, "GME", "C", "C")]))
    st = {}
    msg = occ.run(st, now=now, http_get=get)
    assert occ.path_for(date(2026, 10, 20)).exists() and not occ.path_for(date(2026, 10, 21)).exists()
    assert "2026-10-21" in st["retry"] and "not published yet" in msg


def test_occ_a_session_past_its_latest_waits_like_an_unpublished_one(monkeypatch):
    from src.data.feeds import occ
    monkeypatch.setattr(settings, "feeds_occ_history_days", 3)
    now = datetime(2026, 10, 21, 22, tzinfo=timezone.utc)
    later = "Report date cannot be greater than 10/20/2026\r\n"           # the OCC's answer at 16:10 ET 2026-10-26

    def get(url, params=None):
        return R(text=later if params["reportDate"] == "20261021" else _occ_text([(7, "GME", "C", "C")]))
    st = {}
    msg = occ.run(st, now=now, http_get=get)                             # no ValueError: asked again later
    assert occ.path_for(date(2026, 10, 20)).exists() and not occ.path_for(date(2026, 10, 21)).exists()
    assert "2026-10-21" in st["retry"] and "1 not published yet" in msg


def test_occ_sessions_skip_a_closure_and_volume_reads():
    from src.data.feeds import occ
    occ._write(date(2026, 10, 19), _occ_text([(5000, "GME", "C", "C")]))
    occ._write(date(2026, 10, 20), _occ_text([(3, "SPY", "C", "C")]))       # a closure's near-empty file
    occ._write(date(2026, 10, 21), _occ_text([(4000, "GME", "C", "C"), (900, "GME", "M", "C")]))
    days = occ.sessions(date(2026, 10, 22), 5)
    assert days == [date(2026, 10, 19), date(2026, 10, 21)]
    assert occ.volume("GME", days) == [5000, 4000] and occ.volume("BRK.B", days) == [0, 0]


# ── the FDA ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _fda_zip():
    res = {"results": [
        {"application_number": "NDA123456", "sponsor_name": "CITIUS PHARMS", "submissions": [
            {"submission_type": "ORIG", "submission_number": "1", "submission_status": "AP",
             "submission_status_date": "20250102", "submission_class_code": "TYPE 1", "review_priority": "PRIORITY"},
            {"submission_type": "SUPPL", "submission_number": "5", "submission_status": "AP",
             "submission_status_date": "20250601"},
            {"submission_type": "ORIG", "submission_number": "2", "submission_status": "TA",
             "submission_status_date": "20240101"}]},
        {"application_number": "ANDA200000", "sponsor_name": "GENERIC CO", "submissions": [
            {"submission_type": "ORIG", "submission_number": "1", "submission_status": "AP",
             "submission_status_date": "20250103"}]}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("drug-drugsfda-0001-of-0001.json", json.dumps(res))
    return buf.getvalue()


def test_fda_index_export_and_read():
    from src.data.feeds import fda
    raw = _fda_zip()
    hits = []

    def get(url, params=None):
        hits.append(url)
        if url == settings.feeds_fda_index_url:
            return R(js={"results": {"drug": {"drugsfda": {"export_date": "2026-10-23",
                                                            "partitions": [{"file": "https://x/drugsfda.zip"}]}}}})
        return R(content=raw)
    st = {}
    assert "3 approvals" in fda.run(st, http_get=get)
    assert st["export_date"] == "2026-10-23" and len(hits) == 2
    assert "already held" in fda.run(st, http_get=get) and len(hits) == 3      # the index only
    rows = fda.read(start="20250101", end="20251231")
    assert [r["application"] for r in rows] == ["NDA123456"] and rows[0]["class_code"] == "TYPE 1"
    assert len(fda.read(kinds=None, original_only=False)) == 3


def test_fda_keeps_the_newest_exports():
    from src.data.feeds import fda
    raw = _fda_zip()
    for k in range(6):
        fda._write(f"2026-10-{10 + k:02d}", raw)
    kept = sorted(p.name for p in fda.root().glob("drugsfda_*.json.zip"))
    assert kept == [f"drugsfda_2026-10-{10 + k:02d}.json.zip" for k in range(2, 6)]


def test_occ_import_skips_closures(tmp_path):
    from src.data.feeds import occ
    src = tmp_path / "research_raw"
    src.mkdir()
    for d in ("2025-01-08", "2025-01-09", "2025-01-10"):             # 2025-01-09: the market closed
        with gzip.open(src / f"{d}.csv.gz", "wt", encoding="utf-8") as fh:
            fh.write(_occ_text([(2000, "GME", "C", "C")]))
    assert occ.import_dir(src) == 2 and not occ.path_for(date(2025, 1, 9)).exists()


# ── the runner, the timer, the health ───────────────────────────────────────────────────────────────────────────────

def test_runner_records_failures_and_checks_fda_once_a_day(monkeypatch):
    from src.data import feeds as FD
    from src.data.feeds import __main__ as M
    monkeypatch.setattr(settings, "enable_data_feeds", True)
    for src in ("efts", "cboe", "plans", "gdelt"):                       # the census's feeds: their own test below
        monkeypatch.setattr(settings, f"enable_feeds_{src}", False)
    calls = []

    class Ok:
        @staticmethod
        def run(st, now=None, http_get=None, deadline=None):
            calls.append("ok")
            return "fine"

    class Bad:
        @staticmethod
        def run(st, now=None, http_get=None, deadline=None):
            raise RuntimeError("http 503")
    monkeypatch.setattr(M, "_module", lambda src: Bad if src == "occ" else Ok)
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    r = M.run(now=now)
    assert len(r["ran"]) == 2 and r["failed"] == ["occ RuntimeError: http 503"]
    st = FD.load_state()
    assert st["occ"]["last_error"].startswith("RuntimeError") and st["wsb"]["last_ok"] == now.isoformat()
    r = M.run(now=now + timedelta(hours=1))
    assert "fda: checked recently" in r["skipped"] and len(r["ran"]) == 1
    monkeypatch.setattr(settings, "enable_feeds_wsb", False)
    assert "wsb: off" in M.run(now=now + timedelta(hours=2))["skipped"]


def test_runner_lock_keeps_one_run():
    from src.data import feeds as FD
    from src.data.feeds import __main__ as M
    with FD.run_lock() as got:
        assert got
        assert M.run()["skipped"] == ["locked"]


def test_scheduler_feeds_slot():
    from src.scheduler.runner import _feeds_due
    t = datetime(2026, 10, 26, 12, 12)
    assert _feeds_due(t, None, 30.0)
    assert not _feeds_due(datetime(2026, 10, 26, 12, 5), None, 30.0)          # a slot's first minutes
    assert not _feeds_due(datetime(2026, 10, 26, 12, 31), None, 30.0)
    assert not _feeds_due(datetime(2026, 10, 26, 12, 20), t, 30.0)            # too soon after the last
    assert _feeds_due(datetime(2026, 10, 26, 12, 41), t, 30.0)


def test_health_notes_only_when_on(monkeypatch):
    from src.data import feeds as FD
    now = datetime(2026, 10, 20, 14, tzinfo=timezone.utc)
    FD.save_state({"wsb": {"last_ok": (now - timedelta(hours=5)).isoformat()},
                   "occ": {"last_ok": now.isoformat()}, "fda": {"last_ok": now.isoformat()}})
    assert FD.health(now) == {"problems": [], "notes": []}                     # off in the baseline
    monkeypatch.setattr(settings, "enable_data_feeds", True)
    h = FD.health(now)
    assert h["problems"] == [] and any("wsb" in n and "5 h" in n for n in h["notes"])


def test_health_notes_an_archive_outage_and_how_its_gap_is_filled(monkeypatch):
    from src.data import feeds as FD
    monkeypatch.setattr(settings, "enable_data_feeds", True)
    now = datetime(2026, 10, 26, 14, tzinfo=timezone.utc)
    FD.save_state({"wsb": {"last_ok": now.isoformat(), "archive_fail_since": "2026-10-26T12:10:00+00:00",
                           "archive_error": "RuntimeError: http 530"},
                   "occ": {"last_ok": now.isoformat()}, "fda": {"last_ok": now.isoformat()}})
    notes = [x for x in FD.health(now)["notes"] if "Arctic Shift failing since 2026-10-26T12:10" in x]
    assert notes and "http 530" in notes[0] and "no Reddit app configured" in notes[0]
    _reddit_on(monkeypatch)
    st = FD.load_state()
    st["wsb"]["fill_reach_utc"] = int(now.timestamp())
    FD.save_state(st)
    assert any("Reddit's API filled the gap through 14:00 UTC" in x for x in FD.health(now)["notes"])
    st["wsb"]["fill_error"] = "RuntimeError: reddit token http 401 (the app's credentials were refused)"
    FD.save_state(st)
    assert any("Reddit's API failed too" in x and "401" in x for x in FD.health(now)["notes"])


# ── the free-data census's feeds (2026-10-26 evening): EDGAR full-text search, Cboe, the 10b5-1 notes, GDELT ────────

def _et(y, m, d, hh, mm=0):
    from zoneinfo import ZoneInfo
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("America/New_York"))


def test_sec_quiet_windows_wrap_midnight(monkeypatch):
    from src.data.feeds import sec
    monkeypatch.setattr(settings, "feeds_sec_quiet_et", "23:30-00:45,08:15-09:45")
    assert sec.quiet(_et(2026, 10, 26, 23, 40)) and sec.quiet(_et(2026, 10, 27, 0, 30))
    assert sec.quiet(_et(2026, 10, 27, 9, 0)) and not sec.quiet(_et(2026, 10, 27, 0, 50))
    assert not sec.quiet(_et(2026, 10, 27, 12, 0)) and not sec.quiet(_et(2026, 10, 27, 9, 45))


def _efts_hit(i, day, cik="0001935979", tk="BHVN"):
    return {"_id": f"0001935979-26-00008{i}:doc{i}.htm",
            "_source": {"ciks": [cik], "display_names": [f"Biohaven Ltd.  ({tk})  (CIK {cik})"], "file_date": day,
                        "form": "8-K", "sics": ["2834"], "items": ["7.01", "9.01"], "adsh": f"0001935979-26-00008{i}"}}


def test_efts_hits_kept_once_first_seen_kept_and_pdufa_documents_dated(monkeypatch):
    from src.data.feeds import efts
    calls = []
    pdufa = [_efts_hit(1, "2026-10-22"), _efts_hit(2, "2026-10-23")]

    def get(url, params=None):
        calls.append(url)
        if url == settings.feeds_efts_url:
            page = pdufa if params["q"] == '"PDUFA"' and params["from"] == 0 else []
            return R(js={"hits": {"total": {"value": len(page)}, "hits": page}})
        return R(content=b"<html><p>The FDA set a PDUFA target action date of March&nbsp;15, 2027 for the drug.</p>"
                         b"</html>")
    st = {}
    now = datetime(2026, 10, 26, 22, tzinfo=timezone.utc)
    msg = efts.run(st, now=now, http_get=get)
    assert "pdufa 2 hit(s), 2 new, 2 document(s)" in msg and st["windows"]["pdufa"] == "2026-10-26"
    rows = efts.hits("pdufa")
    assert len(rows) == 2 and rows[0]["seen_utc"] == now.isoformat() and rows[0]["items"] == ["7.01", "9.01"]
    d = efts.read_dates("pdufa")
    assert len(d) == 2 and d[0]["exact"] == ["March 15, 2027"] and d[0]["mentions"] == 1
    n_docs = sum(1 for c in calls if "Archives" in c)
    efts.run(st, now=now + timedelta(hours=7), http_get=get)              # the overlap re-reads: kept once
    assert sum(1 for c in calls if "Archives" in c) == n_docs             # no document fetched twice
    assert [r["seen_utc"] for r in efts.hits("pdufa")] == [now.isoformat()] * 2
    starts = [c for c in calls if c == settings.feeds_efts_url]
    assert len(starts) == 2 * len(efts.QUERIES)


def test_efts_skip_reason_quiet_window_and_cadence(monkeypatch):
    from src.data.feeds import efts, plans
    monkeypatch.setattr(settings, "feeds_sec_quiet_et", "23:30-00:45,08:15-09:45")
    assert efts.skip_reason({}, _et(2026, 10, 27, 8, 40)) == "inside production's SEC lanes"
    assert plans.skip_reason({}, _et(2026, 10, 27, 0, 10)) == "inside production's SEC lanes"
    st = {"last_ok": _et(2026, 10, 27, 12, 10).isoformat()}
    assert efts.skip_reason(st, _et(2026, 10, 27, 16, 10)) == "checked recently"
    assert efts.skip_reason(st, _et(2026, 10, 27, 18, 40)) is None


def test_efts_import_a_research_download(tmp_path):
    from src.data.feeds import efts
    src = tmp_path / "pdufa"
    (src / "hits").mkdir(parents=True)
    h = _efts_hit(3, "2003-05-02")
    (src / "hits" / "2003.json").write_text(json.dumps([{"id": h["_id"], **h["_source"]}]), encoding="utf-8")
    (src / "docs").mkdir()
    name = efts.doc_path("pdufa", h["_source"]["adsh"], "doc3.htm").name
    (src / "docs" / name).write_bytes(gzip.compress(b"PDUFA date of the second quarter of 2004"))
    assert efts.import_dir("pdufa", src) == 1
    r = efts.hits("pdufa")[0]
    assert r["imported"] and r["seen_utc"] is None and r["file_date"] == "2003-05-02"
    assert efts.read_dates("pdufa")[0]["quarter"] == ["second quarter of 2004"]


def _cboe_js(total):
    return json.dumps({"ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": str(total)},
                                  {"name": "EQUITY PUT/CALL RATIO", "value": "0.61"}],
                       "EQUITY OPTIONS": [{"name": "VOLUME", "call": 100, "put": 61, "total": 161},
                                          {"name": "OPEN INTEREST", "call": 900, "put": 800, "total": 1700}]}).encode()


def test_cboe_archives_sessions_and_waits_for_the_unposted(monkeypatch):
    from src.data.feeds import cboe
    monkeypatch.setattr(settings, "feeds_cboe_history_days", 2)
    now = datetime(2026, 10, 23, 22, tzinfo=timezone.utc)               # Friday 18:00 ET: Friday is complete
    asked = []

    def get(url, params=None):
        asked.append(url)
        if "2026-10-23" in url:
            return R(status=403)                                         # not posted yet
        return R(content=_cboe_js(0.93 if "2026-10-21" in url else 1.05))
    st = {}
    msg = cboe.run(st, now=now, http_get=get)
    assert msg == "2 session(s) archived 2026-10-22; 1 not posted yet" and "2026-10-23" in st["retry"]
    assert cboe.path_for(date(2026, 10, 21)).exists() and not cboe.path_for(date(2026, 10, 23)).exists()
    assert cboe.ratios(before=date(2026, 10, 23)) == [("2026-10-21", 0.93), ("2026-10-22", 1.05)]
    assert cboe.ratios(before=date(2026, 10, 22)) == [("2026-10-21", 0.93)]          # point in time
    n = len(asked)
    cboe.run(st, now=now + timedelta(minutes=20), http_get=get)          # the retry waits its hour
    assert len(asked) == n
    with pytest.raises(ValueError):
        cboe.parse({"rows": []})


def test_cboe_history_csv_rows():
    from src.data.feeds import cboe
    text = ("Volume and Put/Call Ratio data is compiled for the convenience of site visitors, 2006\n"
            "DATE,CALLS,PUTS,TOTAL,P/C Ratio\n10/03/2019, 2097645, 2882751, 4980396, 1.37\n"
            "10/04/2019, 2175006, 2289715, 4464721, 1.05\n")
    assert cboe.parse_history(text) == [("2019-10-03", 2097645.0, 2882751.0, 4980396.0, 1.37),
                                        ("2019-10-04", 2175006.0, 2289715.0, 4464721.0, 1.05)]


def _notes_zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("sub.tsv", "adsh\tcik\tname\tform\taccepted\n0001-26-1\t123\tACME CORP\t10-Q\t2026-11-03 16:07:00.0\n"
                              "0001-26-2\t456\tOTHER INC\t10-Q\t2026-11-04 09:00:00.0\n")
        z.writestr("txt.tsv", "adsh\ttag\tversion\tddate\tdimh\tvalue\n"
                              "0001-26-1\tTrdArrIndName\tecd/2024\t20260930\t0xabc\tJane Roe\n"
                              "0001-26-2\tRevenues\tus-gaap/2024\t20260930\t0x00000000\tlots\n")
        z.writestr("num.tsv", "adsh\ttag\tversion\tddate\tdimh\tvalue\n"
                              "0001-26-1\tTrdArrSecuritiesAggAvailAmt\tecd/2024\t20260930\t0xabc\t50000\n")
        z.writestr("dim.tsv", "dimhash\tsegments\tsegt\n0xabc\tIndividual=JaneRoe;\t0\n0xdef\tOther=x;\t0\n")
    return b.getvalue()


def test_plans_names_order_parse_and_one_file_a_run(monkeypatch):
    from src.data.feeds import plans
    assert sorted(["2025_10", "2023q3", "2026_01", "2025q3"], key=plans.name_key) == \
        ["2023q3", "2025q3", "2025_10", "2026_01"]
    monkeypatch.setattr(settings, "feeds_plans_start", "2026_10")
    page = "".join(f'<a href="/files/dera/data/financial-statement-notes-data-sets/{n}_notes.zip">x</a>'
                   for n in ("2026_09", "2026_10", "2026_11"))

    def get(url, params=None):
        return R(text=page) if url == settings.feeds_plans_list_url else R(content=_notes_zip())
    st = {}
    now = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)
    assert plans.run(st, now=now, http_get=get) == "2026_10: 2 trading-arrangement facts in 1 filings; 1 more due"
    f = plans.read("facts")
    assert {r["tag"] for r in f} == {"TrdArrIndName", "TrdArrSecuritiesAggAvailAmt"} and f[0]["file"] == "2026_10"
    assert plans.read("dims")[0]["segments"] == "Individual=JaneRoe;"
    assert plans.read("subs")[0]["accepted"].startswith("2026-11-03 16:07")
    assert plans.run(st, now=now, http_get=get).startswith("2026_11: ")
    assert plans.run(st, now=now, http_get=get) == "nothing new (newest listed 2026_11)"
    with pytest.raises(ValueError):
        plans.parse_zip(zipfile.ZipFile(io.BytesIO(_notes_zip_without_sub())))


def _notes_zip_without_sub():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("txt.tsv", "adsh\ttag\n")
    return b.getvalue()


def _gkg_zip(rows):
    hdr = "DATE\tNUMARTS\tCOUNTS\tTHEMES\tLOCATIONS\tPERSONS\tORGANIZATIONS\tTONE\tCAMEOEVENTIDS\tSOURCES\tSOURCEURLS\n"
    body = "".join(f"20261020\t{n}\t\t{th}\t\t\t\t{tone},1,3,4,20,1\t\t\t\n" for n, th, tone in rows)
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("20261020.gkg.csv", hdr + body)
    return b.getvalue()


def test_gdelt_aggregates_themes_checks_md5_and_records_a_missing_day(monkeypatch):
    import hashlib
    from src.data.feeds import gdelt
    monkeypatch.setattr(settings, "feeds_gdelt_start", "2026-10-20")
    raw = _gkg_zip([(3, "ECON_STOCKMARKET;ECON_INFLATION;", -2.5), (1, "ECON_STOCKMARKET;TAX_FNCACT;", 1.0)])
    sums = {"20261020.gkg.csv.zip": hashlib.md5(raw).hexdigest()}

    def get(url, params=None):
        if url == settings.feeds_gdelt_md5_url:
            return R(text="".join(f"{v}  {k}\n" for k, v in sums.items()))
        return R(content=raw) if "20261020" in url else R(status=404)
    st = {}
    now = datetime(2026, 10, 22, 9, tzinfo=timezone.utc)                # the 21st is posted ~07:00 UTC on the 22nd
    assert gdelt.due(now, st) == [date(2026, 10, 20), date(2026, 10, 21)]
    msg = gdelt.run(st, now=now, http_get=get)
    assert msg.startswith("1 day(s) aggregated: 2026-10-20 (4 articles)") and not st.get("missing")
    assert gdelt.shares(date(2026, 10, 20), ["ECON_STOCKMARKET", "ECON_INFLATION", "X"]) == \
        {"ECON_STOCKMARKET": 1.0, "ECON_INFLATION": 0.75, "X": 0.0}
    totals, themes = gdelt.aggregate(raw)
    assert totals["articles"] == 4 and themes["ECON_INFLATION"] == [3, -7.5] and themes["TAX_FNCACT"] == [1, 1.0]
    later = datetime(2026, 10, 25, 9, tzinfo=timezone.utc)               # still not posted three days on: missing
    assert "missing 2026-10-21" in gdelt.run(st, now=later, http_get=get)
    assert date(2026, 10, 21) not in gdelt.due(later, st)
    sums["20261023.gkg.csv.zip"] = "0" * 32

    def bad(url, params=None):
        if url == settings.feeds_gdelt_md5_url:
            return R(text="".join(f"{v}  {k}\n" for k, v in sums.items()))
        return R(content=raw)
    with pytest.raises(ValueError):
        gdelt.run(st, now=later, http_get=bad, start=date(2026, 10, 23), end=date(2026, 10, 23))


def test_runner_runs_the_census_feeds_and_skips_by_their_reason(monkeypatch):
    from src.data import feeds as FD
    from src.data.feeds import __main__ as M, efts
    monkeypatch.setattr(settings, "enable_data_feeds", True)
    monkeypatch.setattr(settings, "feeds_sec_quiet_et", "08:15-09:45")
    ran = []

    class Ok:
        @staticmethod
        def run(st, now=None, http_get=None, deadline=None):
            ran.append(1)
            return "fine"
    real = M._module

    def mod(src):
        if src in ("efts", "plans"):
            return real(src)
        return Ok
    monkeypatch.setattr(M, "_module", mod)
    monkeypatch.setattr(efts, "run", lambda st, now=None, http_get=None, deadline=None: "efts fine")
    r = M.run(only=["efts", "plans", "cboe", "gdelt"], now=_et(2026, 10, 27, 8, 40))
    assert "efts: inside production's SEC lanes" in r["skipped"] and "plans: inside production's SEC lanes" in r["skipped"]
    assert len(ran) == 2
    r = M.run(only=["efts"], now=_et(2026, 10, 27, 12, 10))
    assert r["ran"] == ["efts efts fine"] and FD.load_state()["efts"]["last_ok"]
    assert "efts: checked recently" in M.run(only=["efts"], now=_et(2026, 10, 27, 12, 40))["skipped"]


def test_seeding_commands_wait_for_the_feeds_lock(monkeypatch):
    from src.data import feeds as FD
    from src.data.feeds import __main__ as M, cboe
    monkeypatch.setattr(cboe, "import_history", lambda http_get=None: 7)
    with FD.run_lock() as got:                                           # a timed run holds the stores
        assert got and M.main(["--import-cboe-history"]) == 1
    assert M.main(["--import-cboe-history"]) == 0


def test_health_notes_a_stale_census_feed(monkeypatch):
    from src.data import feeds as FD
    monkeypatch.setattr(settings, "enable_data_feeds", True)
    now = datetime(2026, 10, 28, 14, tzinfo=timezone.utc)
    ok = now.isoformat()
    FD.save_state({"wsb": {"last_ok": ok}, "occ": {"last_ok": ok}, "fda": {"last_ok": ok}, "efts": {"last_ok": ok},
                   "plans": {"last_ok": ok}, "gdelt": {"last_ok": ok},
                   "cboe": {"last_ok": (now - timedelta(hours=40)).isoformat(), "last_error": "RuntimeError: http 500"}})
    notes = FD.health(now)["notes"]
    assert any(n.startswith("data feed cboe: last good fetch 40 h ago") and "http 500" in n for n in notes)
    monkeypatch.setattr(settings, "enable_feeds_cboe", False)
    assert not any("cboe" in n for n in FD.health(now)["notes"])
