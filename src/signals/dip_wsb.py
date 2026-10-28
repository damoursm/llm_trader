"""THE WALLSTREETBETS PREFERENCE in the dip book's daily top pick (PREREG209's KD+; user 2026-10-25: "Build KD+ for
production and deploy it."; from the 2026-10-26 session; `enable_dip_long_wsb_prefer`, settings `dip_long_wsb_*`,
runbook docs/DIP_LONG.md, the study `llm_trader_research/studies/prereg209/`, report
https://claude.ai/artifact/3UfKEYNQ81TJJSh11UpXqg).

THE FLAG: a signal of session D is FLAGGED when its stock was named in at least `dip_long_wsb_min_titles` (5)
r/wallstreetbets submission titles CREATED in the `dip_long_wsb_window_hours` (120) before `dip_long_wsb_cut_et`
(09:00) ET of D — the production feed's store (`feeds_wsb_dir`/<UTC day>.jsonl.gz, `src.data.feeds.wsb`, every 30
minutes; seeded with the whole history from 2018-01-01). The counting is PREREG208's (`wsb._RX`: an upper-case whole
word of 1-5 letters or the cashtag $TICKER; its `WORDS` and every single letter count only as cashtags) with PREREG209's
eight more cashtag-only tokens (`EXTRA_CASHTAG_ONLY`: RH = Robinhood, TD = TD Ameritrade, "EDIT:", IMO, PSA, EV, TV, DOW);
a post counts once per name; a name's token is its plain ticker (a class share's root before "." or "-").

THE PREFERENCE (`prefer`): in `dip_long._daily_top`, when any of the day's signals at or above the order floor is
flagged, the flagged ones come first (each group in the production ranking's order: the meta-model's score, an earnings
dip after the others), so the pick is the best-ranked flagged signal; a day without a flagged signal picks as before.
The simulator's construction (`mimic.wsb_preferred`, PREREG209's patched `rank_pieces`) keeps only the flagged orders of
such a day — the same pick for the daily top 1.

FAIL-OPEN: the window must lie in the store — every UTC day it touches present and its newest post before the cut no
more than `dip_long_wsb_max_feed_lag_minutes` (60) older than the cut — and the computation must come after the cut;
otherwise nothing is preferred (status ``missing_days`` / ``stale`` / ``early`` / ``error``, journaled in the signal
file's ``top`` block, a note in the scorer banner): the pick is production's without the preference.

POINT IN TIME: only posts created before 09:00 ET of D are read (`tests/test_no_lookahead.py`); the opening tick
decides at 09:30, after the feed's run that covers the cut.

CLI: ``python -m src.signals.dip_wsb --day D [--tickers A,B]`` (the day's window and counts) / ``--parity`` (the store's
file counts against the history index the simulator reads, on the recent signal files).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import settings
from src.utils import ET

RULE = "PREREG209 KD+: the day's WallStreetBets-flagged signal first"
EXTRA_CASHTAG_ONLY = frozenset({"RH", "TD", "EDIT", "IMO", "PSA", "EV", "TV", "DOW"})
STATUSES = ("ok", "missing_days", "stale", "early", "error")


def enabled() -> bool:
    return bool(getattr(settings, "enable_dip_long_wsb_prefer", False))


def min_titles() -> int:
    return max(1, int(getattr(settings, "dip_long_wsb_min_titles", 5) or 5))


def window_hours() -> float:
    return float(getattr(settings, "dip_long_wsb_window_hours", 120.0) or 120.0)


def cashtag_only() -> frozenset:
    """The tokens that count only as cashtags: PREREG208's words + PREREG209's eight."""
    from src.data.feeds import wsb
    return frozenset(wsb.WORDS) | EXTRA_CASHTAG_ONLY


def names_in(title: str, words: Optional[frozenset] = None) -> set:
    """The names a title counts for (each once)."""
    from src.data.feeds import wsb
    words = cashtag_only() if words is None else words
    out = set()
    for cash, tk in wsb._RX.findall(title or ""):
        if not cash and (tk in words or len(tk) == 1):
            continue
        out.add(tk)
    return out


def token_of(ticker: str) -> str:
    """A ticker's token: the plain symbol (a backtest name's ``@D`` / ``@X`` / ``@E`` suffix removed), a class share's
    root before "." or "-" (BRK-B -> BRK)."""
    s = str(ticker or "").split("@")[0].upper()
    return re.split(r"[.\-/ ]", s)[0]


def cut_utc(d: date) -> datetime:
    """`dip_long_wsb_cut_et` (09:00) ET of session ``d``, in UTC."""
    hh, mm = (int(x) for x in str(getattr(settings, "dip_long_wsb_cut_et", "09:00") or "09:00").split(":"))
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(timezone.utc)


def window(d: date) -> Tuple[int, int]:
    """[start, cut) of session ``d``'s window in UTC seconds."""
    c = cut_utc(d)
    return int((c - timedelta(hours=window_hours())).timestamp()), int(c.timestamp())


def window_counts(d: date, tickers: Iterable[str], now: Optional[datetime] = None) -> dict:
    """Session ``d``'s window read from the store: {"status", "cut", "start", "days", "missing", "newest_post",
    "counts": {ticker: titles}} — the counts only with status "ok"."""
    from src.data.feeds import wsb
    t0, t1 = window(d)
    cut = datetime.fromtimestamp(t1, timezone.utc)
    toks = {str(t): token_of(t) for t in tickers}
    want = set(toks.values())
    found = {k: 0 for k in want}
    days, missing, newest = [], [], None
    day = datetime.fromtimestamp(t0, timezone.utc).date()
    while day <= cut.date():
        if not wsb.path_for(day).exists():
            missing.append(day.isoformat())
        else:
            days.append(day.isoformat())
            words = cashtag_only()
            for x in wsb.read_day(day):
                ts = x.get("created_utc")
                if ts is None:
                    continue
                ts = int(ts)
                if ts >= t1:
                    continue
                if newest is None or ts > newest:
                    newest = ts
                if ts >= t0:
                    for tk in names_in(x.get("title") or "", words) & want:
                        found[tk] += 1
        day += timedelta(days=1)
    lag = float(getattr(settings, "dip_long_wsb_max_feed_lag_minutes", 60.0) or 60.0)
    now = now or datetime.now(timezone.utc)
    if now < cut:
        status = "early"
    elif missing:
        status = "missing_days"
    elif newest is None or newest < t1 - lag * 60:
        status = "stale"
    else:
        status = "ok"
    out = {"status": status, "cut": cut.isoformat(), "start": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
           "days": days, "missing": missing,
           "newest_post": datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest is not None else None}
    out["counts"] = {t: int(found[k]) for t, k in toks.items()} if status == "ok" else {}
    return out


def flag_signals(sig: Sequence[dict], d: date, now: Optional[datetime] = None) -> dict:
    """Stamp each signal ``wsb_titles`` / ``wsb_flagged`` (None when the window cannot be read) and return the
    journal block: {"rule", "min_titles", "window_hours", "status", "cut", ..., "flagged": [tickers]}."""
    block = {"rule": RULE, "min_titles": min_titles(), "window_hours": window_hours()}
    w = window_counts(d, [s["ticker"] for s in sig], now=now)
    block.update({k: v for k, v in w.items() if k != "counts"})
    n = min_titles()
    flagged = []
    for s in sig:
        c = w["counts"].get(s["ticker"]) if w["status"] == "ok" else None
        s["wsb_titles"] = c
        s["wsb_flagged"] = (c >= n) if c is not None else None
        if s["wsb_flagged"]:
            flagged.append(s["ticker"])
    block["flagged"] = flagged
    block["titles"] = {s["ticker"]: s["wsb_titles"] for s in sig}
    return block


def prefer(eligible: Sequence[dict], block: Mapping) -> Tuple[List[dict], bool]:
    """(the day's eligible signals with the flagged ones first, each group in its given order; True when a flagged one
    exists) — unchanged unless the window was read ("ok")."""
    if block.get("status") != "ok":
        return list(eligible), False
    f = [s for s in eligible if s.get("wsb_flagged") is True]
    if not f:
        return list(eligible), False
    return f + [s for s in eligible if s.get("wsb_flagged") is not True], True


# ── the history index (the simulator's flags: `mimic.wsb_preferred`) ─────────

def _store_key() -> tuple:
    from src.data.feeds import wsb
    files = sorted(wsb.root().glob("????-??-??.jsonl.gz"))
    return (str(wsb.root()), len(files), max((f.stat().st_mtime for f in files), default=0.0))


@lru_cache(maxsize=2)
def _index(key: tuple) -> dict:
    import numpy as np
    from src.data.feeds import wsb
    acc: Dict[str, list] = {}
    days = set()
    words = cashtag_only()
    for f in sorted(wsb.root().glob("????-??-??.jsonl.gz")):
        d = date.fromisoformat(f.name[:10])
        days.add(d)
        for x in wsb.read_day(d):
            ts = x.get("created_utc")
            if ts is None:
                continue
            for tk in names_in(x.get("title") or "", words):
                acc.setdefault(tk, []).append(int(ts))
    return {"tokens": {k: np.sort(np.asarray(v, np.int64)) for k, v in acc.items()}, "days": days}


def history_index() -> dict:
    """{"tokens": {token: sorted creation instants}, "days": the UTC days in the store} — built once per store
    version (its files and their newest write)."""
    return _index(_store_key())


def history_titles(ticker: str, d: date, idx: Optional[dict] = None) -> Optional[int]:
    """Session ``d``'s count from the history index (as `window_counts` reads the store, without the freshness
    test); None when a UTC day of the window is not in the store."""
    import numpy as np
    idx = idx or history_index()
    t0, t1 = window(d)
    day = datetime.fromtimestamp(t0, timezone.utc).date()
    last = datetime.fromtimestamp(t1, timezone.utc).date()
    while day <= last:
        if day not in idx["days"]:
            return None
        day += timedelta(days=1)
    a = idx["tokens"].get(token_of(ticker))
    if a is None:
        return 0
    return int(np.searchsorted(a, t1, "left") - np.searchsorted(a, t0, "left"))


def history_flag(ticker: str, d: date, idx: Optional[dict] = None) -> bool:
    n = history_titles(ticker, d, idx)
    return n is not None and n >= min_titles()


# ── checks ───────────────────────────────────────────────────────────────────

def parity(days: int = 60) -> dict:
    """The store's file counts (`window_counts`) against the history index (`history_titles`) for every signal of the
    dip book's last ``days`` signal files."""
    from src.signals import dip_long
    files = sorted(Path(dip_long.root(), "signals").glob("????-??-??.json"))[-int(days):]
    idx = history_index()
    pairs, same, diff = 0, 0, []
    for f in files:
        js = json.loads(f.read_text(encoding="utf-8"))
        d = date.fromisoformat(js["day"])
        tks = [s["ticker"] for s in (js.get("signals") or []) + (js.get("not_top") or [])]
        if not tks:
            continue
        w = window_counts(d, tks)
        if w["status"] != "ok":
            continue
        for tk in tks:
            pairs += 1
            a, b = w["counts"].get(tk), history_titles(tk, d, idx)
            if a == b:
                same += 1
            else:
                diff.append({"day": str(d), "ticker": tk, "store": a, "index": b})
    return {"pairs": pairs, "same": same, "differ": diff[:20]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.signals.dip_wsb")
    ap.add_argument("--day", default=None)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args(argv)
    if a.parity:
        print(json.dumps(parity(), indent=1))
        return 0
    d = date.fromisoformat(a.day) if a.day else datetime.now(ET).date()
    tks = [t for t in a.tickers.split(",") if t]
    if not tks:
        from src.signals import dip_long
        p = dip_long.signals_path(d)
        if p.exists():
            js = json.loads(p.read_text(encoding="utf-8"))
            tks = [s["ticker"] for s in (js.get("signals") or []) + (js.get("not_top") or [])]
    w = window_counts(d, tks)
    print(json.dumps(w, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
