"""r/wallstreetbets submissions from the Arctic Shift archive, gaps filled from Reddit's own API (the data feeds,
`src.data.feeds`).

Store: ``feeds_wsb_dir``/<UTC day>.jsonl.gz — one post a line: id, created_utc, title, score, num_comments,
link_flair_text (the author is never kept); a day's file is rewritten atomically with its posts merged by id. A run
fetches from the state's ``after_utc`` minus ``feeds_wsb_overlap_minutes`` (late captures) to now, 100 posts a
request, paced; ``after_utc`` then moves to the newest post seen. An empty store starts at ``feeds_wsb_start`` (else
2 days back); history is imported (`import_dir`, PREREG208's download).

WHEN THE ARCHIVE FAILS (from 2026-10-27; on 2026-10-26 Arctic Shift answered 422 / 500 / 530 from 08:10 ET, the dip
book's WallStreetBets window was stale at its 09:00 cut and its preference stood down): the pages it served are kept
and it is asked again ``feeds_wsb_retries`` times, ``feeds_wsb_retry_seconds`` apart (doubling); inside the dip book's
PRE-CUT window of a market day (``feeds_wsb_precut_lead_minutes`` before ``dip_long_wsb_cut_et`` to
``feeds_wsb_precut_tail_minutes`` after it) it is asked every ``feeds_wsb_precut_retry_seconds`` while the store is
behind, until the run's budget ends. With an app registered under the user's Reddit account (``reddit_client_id`` /
``reddit_client_secret``, application-only OAuth; ``enable_feeds_wsb_reddit_fill``), the gap is FILLED from Reddit
itself (`fill`): Reddit numbers every post in one sequence (about 1.45 million a day in October 2026), so every id
from the newest one stored to the site's newest is read through ``/api/info``, 100 a call, and r/wallstreetbets'
posts are kept — the ones its moderators removed too, as the archive captures them (the subreddit's own listing holds
about a sixth: 91 of the archive's 553 posts of 2026-10-21 .. 26). A filled post carries ``src`` "reddit"; the
archive's copy of a post replaces it, a fallback copy never replaces the archive's, and the archive's own resume
point never moves past its own posts. The archive also fills when, inside the pre-cut window, its newest post is more
than ``dip_long_wsb_max_feed_lag_minutes`` old (the dip book's own staleness test). Once the archive answers again,
the outage's span is read from it again ``feeds_wsb_reread_hours`` later (posts it captures late).

Mentions (`mentions`): a ticker counted once per title in which it appears as an upper-case whole word or a cashtag
($TICKER); an English or WSB word, or a single letter, only as a cashtag (PREREG208's rule). Point in time: a count
of day D reads only the posts created that UTC day.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from config import settings
from src.data import feeds as FD

FIELDS = ("id", "created_utc", "title", "score", "num_comments", "link_flair_text")
WORDS = set("""A I AI ALL AM AN AND ANY ARE AT ATH BE BIG BUY BY CALL CALLS CAN CEO CFO CPI DD DO DTE EOD EPS ETF EU FDA
FED FOMC FOR FUN GDP GO GOOD HAS HE HOLD HOT IF IN IPO IS IT ITM IV JOB LOL LOVE MAN ME MOON MY NEW NO NOT NOW OF OK ON
ONE OR OTM OUT PE PLAY PT PUT PUTS RUN SALE SEC SELL SO SOS TA THE TO TWO UK UP US USA WE WSB YOLO YOU APE APES GAIN
LOSS TLDR EOW ER QE IRS SPAC DEAL REAL HUGE BEST NEXT WOW WIN EVER GET OPEN CASH FREE LIFE NICE BRO TECH""".split())
_RX = re.compile(r"(?<![A-Za-z0-9])(\$?)([A-Z]{1,5})(?![A-Za-z0-9])")
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"
_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

_sleep = time.sleep          # the retries' waits (tests replace them)
_clock = FD.now_utc          # the live clock of the retries and the pre-cut loop


def root() -> Path:
    return Path(settings.feeds_wsb_dir)


def path_for(d: date) -> Path:
    return root() / f"{d.isoformat()}.jsonl.gz"


def read_day(d: date) -> List[dict]:
    p = path_for(d)
    if not p.exists():
        return []
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def write_day(d: date, posts: Iterable[dict]) -> int:
    """Merge ``posts`` into day ``d``'s file by id (a newer copy of a post replaces the older, but a fallback's copy —
    ``src`` set — never replaces the archive's); returns the count."""
    have = {p.get("id"): p for p in read_day(d)}
    for p in posts:
        if not p.get("id"):
            continue
        new = {k: p.get(k) for k in FIELDS}
        if p.get("src"):
            old = have.get(p["id"])
            if old is not None and not old.get("src"):
                continue
            new["src"] = p["src"]
        have[p["id"]] = new
    rows = sorted(have.values(), key=lambda x: (int(x.get("created_utc") or 0), str(x.get("id"))))
    f = path_for(d)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, f)
    return len(rows)


def _store(posts: List[dict]) -> int:
    """Write ``posts`` into their UTC days; returns the number of days touched."""
    by_day: Dict[date, List[dict]] = defaultdict(list)
    for p in posts:
        c = p.get("created_utc")
        if c is None:
            continue
        by_day[datetime.fromtimestamp(int(c), timezone.utc).date()].append(p)
    for d, ps in by_day.items():
        write_day(d, ps)
    return len(by_day)


def b36(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n, 36)
        s = _B36[r] + s
    return s or "0"


def _id_int(pid) -> Optional[int]:
    try:
        return int(str(pid), 36)
    except (TypeError, ValueError):
        return None


# ── the archive ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _page(after: int, before: int, http_get: Callable) -> List[dict]:
    r = http_get(str(settings.feeds_wsb_url), params={"subreddit": str(settings.feeds_wsb_subreddit), "after": after,
                                                      "before": before, "limit": 100, "fields": ",".join(FIELDS),
                                                      "sort": "asc"})
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    js = r.json()
    if js.get("error"):
        raise RuntimeError(str(js.get("error"))[:80])
    data = js.get("data")
    if not isinstance(data, list):
        raise ValueError("not Arctic Shift's answer (no data list)")
    return data


def _default_get():
    import httpx
    client = httpx.Client(timeout=120.0, headers=FD.UA)
    return client.get


def _fetch(after: int, before: int, get: Callable, deadline: Optional[float]) -> Tuple[List[dict], Optional[Exception]]:
    """Every post created in (after, before) seconds UTC, in order, and the error that stopped the pages (None: all
    served; the pages served before it are kept)."""
    out, seen, cur = [], set(), int(after)
    while True:
        try:
            rows = _page(cur, int(before), get)
        except Exception as e:  # noqa: BLE001 — returned with the pages already served
            return out, e
        new = [x for x in rows if x.get("id") not in seen]
        for x in new:
            seen.add(x.get("id"))
            out.append(x)
        if len(rows) < 100:
            return out, None
        ts = [int(x.get("created_utc") or 0) for x in rows]
        cur = max(ts) if min(ts) < max(ts) and new else max(ts) + 1
        if deadline is not None and time.time() > deadline:
            logger.warning("[feeds] wsb: the run's budget cut the fetch; the rest is due next run")
            return out, None
        FD.pace()


def fetch(after: int, before: int, http_get: Optional[Callable] = None, deadline: Optional[float] = None) -> List[dict]:
    """Every post created in (after, before) seconds UTC, in order (a page of one second's posts is stepped past)."""
    out, err = _fetch(after, before, http_get or _default_get(), deadline)
    if err is not None:
        raise err
    return out


def _archive(st: dict, before: int, get: Callable, deadline: Optional[float]) -> Tuple[int, Optional[Exception]]:
    """One pass over the archive from the resume point; the posts it served stored and the resume point moved."""
    after = int(st["after_utc"]) - int(float(settings.feeds_wsb_overlap_minutes) * 60)
    posts, err = _fetch(after, before, get, deadline)
    _store(posts)
    if posts:
        st["after_utc"] = max(int(st["after_utc"]), max(int(p.get("created_utc") or 0) for p in posts))
        ids = [i for i in (_id_int(p.get("id")) for p in posts) if i is not None]
        if ids:
            st["after_id"] = max(int(st.get("after_id") or 0), max(ids))
    return len(posts), err


def _behind_minutes(st: dict, now: datetime) -> float:
    """How old the store's newest post is (the archive's or a fill's), in minutes."""
    newest = max(int(st.get("after_utc") or 0), int(st.get("fill_newest_utc") or 0))
    return (now.timestamp() - newest) / 60.0


def _cut(now: datetime) -> Optional[datetime]:
    """The dip book's WallStreetBets cut of ``now``'s ET day when it is a market day, else None."""
    from src.performance import market_calendar as MC
    from src.utils import ET
    loc = now.astimezone(ET)
    if not MC.is_market_day(loc.date()):
        return None
    hh, mm = str(settings.dip_long_wsb_cut_et).split(":")
    return loc.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)


def in_precut(now: datetime) -> bool:
    cut = _cut(now)
    if cut is None:
        return False
    return (cut - timedelta(minutes=float(settings.feeds_wsb_precut_lead_minutes)) <= now
            <= cut + timedelta(minutes=float(settings.feeds_wsb_precut_tail_minutes)))


# ── Reddit's own API (the user's app): the gap read by post id ─────────────────────────────────────────────────────

def reddit_configured() -> bool:
    return bool(settings.enable_feeds_wsb_reddit_fill) and bool(str(settings.reddit_client_id or "").strip()) \
        and bool(str(settings.reddit_client_secret or "").strip())


class RedditAPI:
    """Application-only OAuth (client credentials: no user password) with Reddit's rate-limit headers honoured."""

    def __init__(self, client=None):
        import httpx
        self.ua = {"User-Agent": str(settings.reddit_user_agent or FD.UA["User-Agent"])}
        self.client = client or httpx.Client(timeout=30.0)
        self.token: Optional[str] = None
        self.calls = 0
        self._last = 0.0

    def _auth(self) -> None:
        r = self.client.post(REDDIT_TOKEN_URL, data={"grant_type": "client_credentials"},
                             auth=(str(settings.reddit_client_id).strip(), str(settings.reddit_client_secret).strip()),
                             headers=self.ua)
        if r.status_code != 200:
            raise RuntimeError(f"reddit token http {r.status_code}"
                               + (" (the app's credentials were refused)" if r.status_code == 401 else ""))
        tok = (r.json() or {}).get("access_token")
        if not tok:
            raise RuntimeError("reddit token: no access_token in the answer")
        self.token = tok

    def get(self, path: str, params: dict) -> dict:
        if self.token is None:
            self._auth()
        gap = 60.0 / max(1.0, float(settings.feeds_wsb_reddit_calls_per_minute))
        for attempt in range(3):
            wait = self._last + gap - time.time()
            if wait > 0:
                _sleep(wait)
            self._last = time.time()
            r = self.client.get(REDDIT_API + path, params=params,
                                headers=dict(self.ua, Authorization=f"bearer {self.token}"))
            self.calls += 1
            if r.status_code == 200:
                try:
                    left = float(r.headers.get("x-ratelimit-remaining", "100"))
                    reset = float(r.headers.get("x-ratelimit-reset", "0"))
                except ValueError:
                    left, reset = 100.0, 0.0
                if left < 2:
                    _sleep(reset + 1.0)
                return r.json()
            if r.status_code == 401 and attempt == 0:
                self._auth()
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                try:
                    reset = float(r.headers.get("x-ratelimit-reset", "10"))
                except ValueError:
                    reset = 10.0
                _sleep(min(60.0, max(5.0, reset)))
                continue
            raise RuntimeError(f"reddit {path} http {r.status_code}")
        raise RuntimeError(f"reddit {path}: no answer")


def _store_newest_id(now: datetime) -> Optional[int]:
    """The highest post id in the archive's copies of the last two UTC days (a state written before ``after_id``)."""
    best = None
    for k in (1, 0):
        for p in read_day((now - timedelta(days=k)).date()):
            if p.get("src"):
                continue
            i = _id_int(p.get("id"))
            if i is not None and (best is None or i > best):
                best = i
    return best


def fill(st: dict, now: datetime, deadline: Optional[float] = None, api: Optional[RedditAPI] = None) -> str:
    """Read every post id from the newest stored one to Reddit's newest and keep r/wallstreetbets' (``st`` updated in
    place: ``fill_next_id``, ``fill_reach_utc``, ``fill_newest_utc``). Returns a summary; raises on a failure before
    any id was read."""
    api = api or RedditAPI()
    sub = str(settings.feeds_wsb_subreddit).lower()
    head_js = api.get("/r/all/new", {"limit": 25, "raw_json": 1})
    heads = [_id_int((ch.get("data") or {}).get("id")) for ch in (head_js.get("data") or {}).get("children", [])]
    heads = [h for h in heads if h is not None]
    if not heads:
        raise RuntimeError("reddit /r/all/new: no posts")
    head = max(heads)
    base = st.get("after_id") or _store_newest_id(now)
    if base is None:
        raise RuntimeError("no stored post id to start from")
    i = max(int(st.get("fill_next_id") or 0), int(base) + 1)       # oldest first: a long gap is read over runs
    kept, reach, calls, err = [], st.get("fill_reach_utc"), 0, None
    stop = None if deadline is None else deadline - float(settings.feeds_wsb_reddit_margin_seconds)
    while i <= head:
        if calls >= int(settings.feeds_wsb_reddit_max_calls) or (stop is not None and time.time() > stop):
            break
        ids = list(range(i, min(i + 100, head + 1)))
        try:
            js = api.get("/api/info", {"id": ",".join("t3_" + b36(k) for k in ids), "raw_json": 1})
        except Exception as e:  # noqa: BLE001 — what was read is kept; the cursor stays here
            err = e
            break
        calls += 1
        for ch in (js.get("data") or {}).get("children", []):
            d = ch.get("data") or {}
            c = d.get("created_utc")
            if c is None:
                continue
            reach = max(int(reach or 0), int(c))
            if str(d.get("subreddit") or "").lower() == sub:
                p = {k: d.get(k) for k in FIELDS}
                p["created_utc"] = int(c)
                p["src"] = "reddit"
                kept.append(p)
        i = ids[-1] + 1
    _store(kept)
    st["fill_next_id"] = i
    st["fill_reach_utc"] = reach
    if kept:
        st["fill_newest_utc"] = max(int(st.get("fill_newest_utc") or 0), max(p["created_utc"] for p in kept))
    st["fill_last_calls"] = calls
    if err is not None and calls == 0:
        raise err
    left = max(0, head - i + 1)
    reach_s = datetime.fromtimestamp(reach, timezone.utc).strftime("%H:%M UTC") if reach else "?"
    return (f"Reddit's API {len(kept)} post(s) from {calls} call(s), read through {reach_s}"
            + (f", {left:,} ids left" if left else "") + (f" ({type(err).__name__}: {str(err)[:80]})" if err else ""))


# ── a run ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def _rereads_due(st: dict, get: Callable, deadline: Optional[float], now: datetime) -> int:
    """Read the archive's outage spans again once due (late captures beyond the overlap); returns the posts read."""
    keep, n = [], 0
    for span in st.get("rereads") or []:
        a, b, due = int(span[0]), int(span[1]), str(span[2])
        if datetime.fromisoformat(due) > now:
            keep.append(span)
            continue
        posts, err = _fetch(a, b, get, deadline)
        _store(posts)
        n += len(posts)
        if err is not None:
            keep.append(span)
    st["rereads"] = keep
    return n


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None, api: Optional[RedditAPI] = None) -> str:
    """One fetch of what is new since ``st['after_utc']`` (``st`` updated in place), retried and gap-filled when the
    archive fails. Returns a summary; raises when the archive failed and nothing filled the gap."""
    now = now or FD.now_utc()
    t_now = int(now.timestamp())
    get = http_get or _default_get()
    if st.get("after_utc") is None:
        start = str(settings.feeds_wsb_start or "").strip()
        st["after_utc"] = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()) if start \
            else t_now - 2 * 86400
    n, err = _archive(st, t_now + 60, get, deadline)
    wait = float(settings.feeds_wsb_retry_seconds)
    for _ in range(int(settings.feeds_wsb_retries)):
        if err is None or (deadline is not None and time.time() + wait > deadline):
            break
        logger.info(f"[feeds] wsb: Arctic Shift failed ({err}); asked again in {wait:.0f} s")
        _sleep(wait)
        wait *= 2
        k, err = _archive(st, int(_clock().timestamp()) + 60, get, deadline)
        n += k
    notes = []

    def _fill(reason: str) -> bool:
        if not reddit_configured():
            return False
        try:
            msg = fill(st, _clock(), deadline=deadline, api=api)
            st.update(fill_last_ok=_clock().isoformat(), fill_error=None)
            notes.append(f"{reason}: {msg}")
            return True
        except Exception as e:  # noqa: BLE001 — recorded; the archive's error stands
            st["fill_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            notes.append(f"{reason}: Reddit's API failed ({st['fill_error']})")
            return False

    filled = False
    lag = float(settings.dip_long_wsb_max_feed_lag_minutes)
    if err is not None:
        filled = _fill("Arctic Shift failed")
    elif in_precut(_clock()) and _behind_minutes(st, _clock()) > lag:
        filled = _fill(f"the store {_behind_minutes(st, _clock()):.0f} min behind before the cut")
    # inside the pre-cut window the archive is asked every minute while the store is behind and nothing filled it
    step = max(1.0, float(settings.feeds_wsb_precut_retry_seconds))
    left = int((float(settings.feeds_wsb_precut_lead_minutes) + float(settings.feeds_wsb_precut_tail_minutes)) * 60
               / step) + 1
    while err is not None and not filled and in_precut(_clock()) and left > 0:
        left -= 1
        if deadline is not None and time.time() + step > deadline:
            break
        _sleep(step)
        k, err = _archive(st, int(_clock().timestamp()) + 60, get, deadline)
        n += k
    if err is None:
        if st.get("archive_fail_since"):
            a = int(st.get("archive_fail_after_utc") or st["after_utc"]) - int(
                float(settings.feeds_wsb_overlap_minutes) * 60)
            due = (now + timedelta(hours=float(settings.feeds_wsb_reread_hours))).isoformat()
            st.setdefault("rereads", []).append([a, t_now + 60, due])
            notes.append(f"Arctic Shift back after failing since {st['archive_fail_since']}")
        st.update(archive_fail_since=None, archive_fail_after_utc=None, archive_error=None,
                  archive_last_ok=now.isoformat())
        r = _rereads_due(st, get, deadline, now)
        if r:
            notes.append(f"{r} post(s) re-read from an outage's span")
    else:
        st["archive_error"] = f"{type(err).__name__}: {str(err)[:160]}"
        if not st.get("archive_fail_since"):
            st.update(archive_fail_since=now.isoformat(), archive_fail_after_utc=int(st["after_utc"]))
    st["last_posts"] = n
    msg = f"{n} posts, newest {st['after_utc']}" + ("; " + "; ".join(notes) if notes else "")
    if err is not None and not filled:
        raise RuntimeError(f"{err}" + ("" if reddit_configured() else " (no Reddit app configured to fill the gap)")
                           + ("; " + "; ".join(notes) if notes else ""))
    return msg


def import_dir(src: Path) -> int:
    """Merge a directory of <day>.jsonl.gz files (PREREG208's research download) into the store; returns the days."""
    n = 0
    for f in sorted(Path(src).glob("????-??-??.jsonl.gz")):
        d = date.fromisoformat(f.name[:10])
        posts = []
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    posts.append(json.loads(line))
                except ValueError:
                    continue
        write_day(d, posts)
        n += 1
    return n


def tickers_in(title: str) -> set:
    out = set()
    for cash, tk in _RX.findall(title or ""):
        if not cash and (tk in WORDS or len(tk) == 1):
            continue
        out.add(tk)
    return out


_COUNTS: Dict[str, tuple] = {}


def day_counts(d: date) -> Dict[str, int]:
    """Ticker -> the number of day ``d``'s titles mentioning it (cached per file version)."""
    p = path_for(d)
    if not p.exists():
        return {}
    key, mt = str(p), p.stat().st_mtime
    hit = _COUNTS.get(key)
    if hit is None or hit[0] != mt:
        c: Dict[str, int] = defaultdict(int)
        for x in read_day(d):
            for tk in tickers_in(x.get("title") or ""):
                c[tk] += 1
        hit = (mt, dict(c))
        _COUNTS[key] = hit
    return hit[1]


def mentions(ticker: str, start: date, end: date) -> Optional[int]:
    """Titles mentioning ``ticker`` over UTC days [start, end]; None when a day of the span is not in the store."""
    tk = str(ticker or "").upper().replace(".", "").replace("-", "")
    total, d = 0, start
    while d <= end:
        if not path_for(d).exists():
            return None
        total += day_counts(d).get(tk, 0)
        d += timedelta(days=1)
    return total
