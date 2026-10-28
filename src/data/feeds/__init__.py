"""THE DATA FEEDS (from 2026-10-25; the user: "Can you make sure we can fetch those new data sources on production as
needed, like daily or every 30min?"): the free sources PREREG208 tests, archived by production on their own cadence;
from the 2026-10-26 evening also the free-data census's (the user: "make sure we can pull it in live for our
production"): `efts` (EDGAR's full-text search: PDUFA dates, red-flag filings), `cboe` (Cboe's daily put/call and
volumes), `plans` (the Rule 10b5-1 plan disclosures of the SEC's notes data sets) and `gdelt` (GDELT's daily news
themes).

- `wsb` — every r/wallstreetbets submission (title, time, score, comment count, flair; never the author) from the
  Arctic Shift archive, which captures a post about half a minute after it appears: every run (`feeds_poll_minutes`, 30)
  fetches what is new, re-reading the last `feeds_wsb_overlap_minutes` for late captures.
- `occ` — the OCC's daily options volume by underlying (account type x put / call x exchange); the OCC serves only its
  last 2 years and keeps no older history, so every completed session is archived raw, once (never delete).
- `fda` — the FDA's Drugs@FDA export (every application's submissions and approval dates); its export date is checked
  every `feeds_fda_check_hours` (24) and a new export downloaded.

`python -m src.data.feeds --run` runs the sources that are due under one OS lock and a time budget; the scheduler
launches it as a subprocess from its own timer (`runner._start_feeds_timer`), off the trading slots, never waited
for. Every download is DATA from a public site: checked for its expected shape and parsed as CSV / JSON, never
evaluated. No live book reads these feeds yet; their health is a note in the email's scorer banner, never an alarm.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from config import settings

UA = {"User-Agent": "llm-trader/1.0 (personal research archive; sequential, paced requests)"}


def state_dir() -> Path:
    return Path(settings.feeds_state_dir)


def state_path() -> Path:
    return state_dir() / "state.json"


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def run_lock() -> Iterator[bool]:
    """An OS-level exclusive lock across processes (`feeds.lock`): yields False when another run holds it."""
    p = state_dir() / "feeds.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    got = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            got = True
        except OSError:
            got = False
        yield got
    finally:
        if got:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


def pace() -> None:
    time.sleep(max(0.0, float(settings.feeds_pace_seconds)))


def age_hours(iso: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return ((now or now_utc()) - t).total_seconds() / 3600.0


def health(now: Optional[datetime] = None) -> dict:
    """The feeds' notes for the email's scorer banner (never problems: no live book reads them yet)."""
    out = {"problems": [], "notes": []}
    if not bool(settings.enable_data_feeds):
        return out
    st = load_state()
    now = now or now_utc()
    later = {"efts": (settings.enable_feeds_efts, 30.0), "cboe": (settings.enable_feeds_cboe, 30.0),
             "plans": (settings.enable_feeds_plans, 72.0), "gdelt": (settings.enable_feeds_gdelt, 36.0)}
    for src, limit in (("wsb", 3.0), ("occ", 30.0), ("fda", 72.0)) + tuple(
            (k, v[1]) for k, v in later.items() if v[0]):
        s = st.get(src) or {}
        a = age_hours(s.get("last_ok"), now)
        if a is None:
            out["notes"].append(f"data feed {src}: never fetched yet")
        elif a > limit:
            out["notes"].append(f"data feed {src}: last good fetch {a:.0f} h ago ({s.get('last_error') or 'no error'})")
    w = st.get("wsb") or {}
    if w.get("archive_fail_since"):                  # the archive failing, the gap filled from Reddit's API or not
        from src.data.feeds import wsb as W
        if not W.reddit_configured():
            how = "no Reddit app configured to fill the gap (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET)"
        elif w.get("fill_error"):
            how = f"Reddit's API failed too ({w.get('fill_error')})"
        else:
            r = w.get("fill_reach_utc")
            how = "Reddit's API filled the gap" + (
                f" through {datetime.fromtimestamp(int(r), timezone.utc).strftime('%H:%M UTC')}" if r else "")
        out["notes"].append(f"data feed wsb: Arctic Shift failing since {str(w['archive_fail_since'])[:16]} "
                            f"({w.get('archive_error') or 'error'}); {how}")
    try:
        from src.data.feeds import occ as OCC
        gap = OCC.missing_recent(now)
        if gap:
            out["notes"].append(f"data feed occ: {len(gap)} completed session(s) not archived, latest {gap[-1]}")
    except Exception as e:  # noqa: BLE001 — a note, never a failure
        logger.debug(f"[feeds] occ health: {e}")
    return out
