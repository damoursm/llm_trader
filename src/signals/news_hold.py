"""THE NEWS HOLD — a vol short's exit postponed after positive headlines, SHADOW (PREREG136; user 2026-10-17:
"Deploy it as shadow").

The rule (PREREG136's LG, the vol arm only; `mimic.news_hold_piece` in the simulator): when a vol short leaves on its
OWN exit — the target or the deadline (the volatility exit too, were it on for the arm) — never the squeeze cover nor
an account-level close, the local LLM (production's model, PREREG131's prompt: one headline -> -1 / 0 / +1 for the
stock's share price, temperature 0, seed 7, reasoning off) rates Polygon's headlines about the name published after
the entry and at least `sel_short_news_hold_lag_minutes` (30) before the exit; when their mean tone is above 0 the
rule HOLDS the short k more sessions: its exit becomes the first mark at or after the same ET clock on the k-th
session after the exit's day — or earlier the first mark whose price reaches the arm's squeeze cover (4x the entry) —,
the borrow running on.

SHADOW: the exit happens as always. The decision is journaled and, for a hold, its outcome is measured afterwards for
every k in `sel_short_news_hold_sessions` (the primary `sel_short_news_hold_primary`) with the BACKTEST's own
construction, so the shadow tests the rule the simulator measured: `mimic.news_extend` on Polygon's 30-minute bars
04:00-20:00 as traded, brought into the exit's terms by the split data — its marks are the regular-hours closes, the
09:30 open and the extended-hours slots, NO overnight marks (the simulator's default; an exit at an overnight tick is
held to the first of those marks at or after its clock, a squeeze through the night is seen at the morning's first
mark; such holds are flagged ``exit_overnight``). The added borrow is the trade's own IBKR rate (its last mark's, else
its entry stamp, else `short_borrow_annual_pct`) on 102% of each prior close for every calendar day added — no shared
ledger state is read; the exit leg's cost is paid either way and left out. It never trades and never touches the
ledger or the broker.

Evidence (memory/shorts-news-hold-account-2026-10.md; report FSovXHxAiVkhrU9byjUiB7): in the production-mimicking
live account 2018-2026 (20-40 draws a set, paired against the live configuration) the hold passes the pooled add
rule for k = 2 .. 10 (+0.85 .. +1.96 pp a year; k 5 +1.89) with no added margin call — every gain from set A
(2021-24): Polygon's feed thinned from 2025 (the rule fires on ~5 vol trades a draw in set B against 28 in set A) and
it holds no headline before 2021 —, and the rule was chosen on the same history (PREREG135's lone per-test pass), so
the overfitting report shrinks every pass under the bar. Hence a shadow: the out-of-sample record.

The decision runs off the tick: `tracker.monitor_sel_short_positions` hands its exits to `after_exits`, which journals
them and kicks ONE daemon thread (single-flight; one pass at a time across processes, `run.lock`). The headlines are
fetched then — what a live rule would read — and each title's rating is cached (``ratings.jsonl``); a Polygon or LLM
failure leaves the exit pending, retried at the next kick, given up (``undecided``) after `PENDING_GIVE_UP_HOURS`.
Later kicks measure the holds' outcomes as their sessions pass; a k still unmeasurable `GIVE_UP_DAYS` after the exit
is closed ``unmeasured``.

Files (`settings.sel_short_dir`/news_hold/): pending/<trade>.json (exits awaiting their decision), decisions/<exit
day>.jsonl (every decided exit: its headlines, their tones, the decision), open/<trade>.json (holds awaiting their
outcomes), settled/<exit day>.jsonl (the outcomes by k), ratings.jsonl, run.lock. Problems never alarm (a shadow):
its notes ride the scorer banner's green line (`health`).

    python -m src.signals.news_hold --status | --run | --backfill-from YYYY-MM-DD | --probe <recommendation_id>
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from config import settings

ET = ZoneInfo("America/New_York")
ARM = "vol"
OWN_EXITS = ("sel_target", "sel_time", "sel_volnorm")   # PREREG136 extends every exit but the squeeze cover
COVER_EXIT = "sel_cover"
PROMPT_VERSION = "prereg131"
# PREREG131's prompt, word for word (llm_trader_research/studies/prereg131/llm131.py) — the backtest's ratings
PROMPT = ("You read one financial news headline about the stock {sym}.\n"
          "Headline: \"{title}\"\n"
          "What does this headline imply for {sym}'s share price: negative, neutral or unclear, or positive?\n"
          "Answer with exactly one number: -1 (negative), 0 (neutral or unclear) or 1 (positive).")
_NUM = re.compile(r"-?1|0")
RETRY_MINUTES = 20                  # an open hold's outcome is re-tried at most this often
GIVE_UP_DAYS = 40                   # a k still unmeasured this long after the exit is closed as `unmeasured`
STALE_PENDING_HOURS = 6             # a decision still pending this long is a banner note
PENDING_GIVE_UP_HOURS = 72          # ... and this long, it is journaled `undecided` (never held)
LOCK_STALE_SECONDS = 7200           # a pass's `run.lock` older than this is a dead pass's: broken
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


# ── files ─────────────────────────────────────────────────────────────────────

def root() -> Path:
    return Path(settings.sel_short_dir) / "news_hold"


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _append(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _key(trade: dict) -> str:
    k = trade.get("recommendation_id") or f"{trade.get('ticker')}_{trade.get('entry_datetime')}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(k))


def _ts(x) -> Optional[datetime]:
    if x is None or x == "":
        return None
    try:
        t = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.to_pydatetime()


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _ns(t: datetime) -> int:
    return int(pd.Timestamp(t).value)


def ks() -> List[int]:
    """The sessions held the shadow measures (`sel_short_news_hold_sessions`), the primary included."""
    out = {int(settings.sel_short_news_hold_primary)}
    for s in str(settings.sel_short_news_hold_sessions).split(","):
        s = s.strip()
        if s:
            out.add(int(s))
    bad = [k for k in out if not 1 <= k <= 20]
    if bad:
        raise ValueError(f"sel_short_news_hold_sessions: {bad} outside 1 .. 20")
    return sorted(out)


def enabled() -> bool:
    return bool(getattr(settings, "enable_sel_short_news_hold_shadow", False))


# ── one pass at a time, across processes ─────────────────────────────────────

def _lock_path() -> Path:
    return root() / "run.lock"


def _acquire() -> bool:
    """`run.lock` created exclusively (a CLI pass and the scheduler's thread never overlap); a lock older than
    `LOCK_STALE_SECONDS` is a dead pass's and is broken."""
    p = _lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - p.stat().st_mtime > LOCK_STALE_SECONDS:
                    p.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            return False
        except OSError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()} {_iso(datetime.now(timezone.utc))}")
        return True
    return False


def _touch_lock() -> None:
    try:
        os.utime(_lock_path(), None)
    except OSError:
        pass


def _release() -> None:
    try:
        _lock_path().unlink(missing_ok=True)
    except OSError:
        pass


# ── the exits ─────────────────────────────────────────────────────────────────

def eligible(trade: dict) -> Optional[str]:
    """What the shadow does with a closed selection-short trade: "decide" (a vol short's own exit), "cover" (the
    squeeze cover: journaled, never held), None (another arm's trade, an account-level close)."""
    if str(trade.get("entry_mechanism") or "") != "sel_short" or str(trade.get("sel_arm") or "") != ARM:
        return None
    if str(trade.get("status") or "").upper() != "CLOSED":
        return None
    r = str(trade.get("exit_reason") or "")
    if r == COVER_EXIT:
        return "cover"
    return "decide" if r in OWN_EXITS else None


_TRADE_FIELDS = ("recommendation_id", "ticker", "action", "type", "entry_mechanism", "sel_arm", "status",
                 "entry_datetime", "entry_date", "entry_price", "entry_session", "exit_datetime",
                 "exit_decision_datetime", "exit_date", "exit_price", "exit_reason", "exit_session", "return_pct",
                 "sel_account_shares", "split_factor", "borrow_rate_now", "borrow_fee_pct", "sel_bar_end",
                 "sel_target_price", "sel_deadline", "sel_fallback_of", "sel_pressure_extra_of")


def _exit_instant(trade: dict) -> Optional[datetime]:
    """The exit's decision instant (its mark's time; the executed instant when none) — the simulator's exit mark."""
    return _ts(trade.get("exit_decision_datetime")) or _ts(trade.get("exit_datetime"))


def after_exits(trades: Sequence[dict], now: Optional[datetime] = None, backfill: bool = False,
                kick_after: bool = True) -> int:
    """Journal the vol shorts this pass closed as pending (a squeeze cover too: the background pass journals it as
    ``cover``, never held — every journal write is that pass's) and kick the background pass (``kick_after``), which
    also measures the open holds. Returns the exits journaled. Never raises into the tick."""
    if not enabled():
        return 0
    now = now or datetime.now(timezone.utc)
    n = 0
    for t in trades or ():
        try:
            if eligible(t) is None:
                continue
            key = _key(t)
            rec = {"key": key, "journaled_at": _iso(now), "backfill": bool(backfill),
                   "trade": {k: t.get(k) for k in _TRADE_FIELDS}}
            if not (root() / "pending" / f"{key}.json").exists() and not _was_decided(rec):
                _write_json(root() / "pending" / f"{key}.json", rec)
                n += 1
        except Exception as e:                                       # noqa: BLE001 — a shadow never costs the exits
            logger.warning(f"[news_hold] could not journal {t.get('ticker')}'s exit: {e}")
    if kick_after:
        kick(now)
    return n


def _was_decided(rec: dict) -> bool:
    x = _exit_instant(rec["trade"])
    if x is None:
        return False
    return any(r.get("key") == rec["key"] for r in _read_jsonl(_decisions_path(x)))


def _decisions_path(exit_at: datetime) -> Path:
    return root() / "decisions" / f"{exit_at.astimezone(ET).date().isoformat()}.jsonl"


def _settled_path(exit_at: datetime) -> Path:
    return root() / "settled" / f"{exit_at.astimezone(ET).date().isoformat()}.jsonl"


# ── the headlines and their tones ─────────────────────────────────────────────

def fetch_headlines(ticker: str, gte: datetime, lte: datetime) -> Optional[List[dict]]:
    """Polygon's articles about ``ticker`` published in [gte, lte] (every page); None when the API fails (the
    decision then waits) — an empty list is a hold without news."""
    from src.data import polygon_client as pc
    if not pc.is_available():
        return None
    sym = pc.to_polygon_symbol(ticker)
    j = pc._get("/v2/reference/news", {"ticker": sym, "published_utc.gte": _iso(gte).replace("+00:00", "Z"),
                                       "published_utc.lte": _iso(lte).replace("+00:00", "Z"),
                                       "order": "asc", "limit": 1000})
    if j is None:
        return None
    out = list(j.get("results") or [])
    nxt, pages = j.get("next_url"), 1
    while nxt and pages < 10:
        try:
            j = pc._follow_next_url(nxt)
        except Exception:                                            # noqa: BLE001
            return None
        if not j:
            return None
        out.extend(j.get("results") or [])
        nxt, pages = j.get("next_url"), pages + 1
    arts = []
    for a in out:
        pub = _ts(a.get("published_utc"))
        if pub is None or not a.get("id"):
            continue
        ins = next((i for i in (a.get("insights") or []) if str(i.get("ticker") or "").upper() == sym.upper()), None)
        arts.append({"id": str(a["id"]), "published_utc": _iso(pub), "title": str(a.get("title") or ""),
                     "publisher": (a.get("publisher") or {}).get("name"),
                     "polygon_sentiment": (ins or {}).get("sentiment")})
    arts.sort(key=lambda a: a["published_utc"])
    return arts


_RATINGS: Dict[Tuple[str, str], dict] = {}
_RATINGS_MTIME: Optional[float] = None


def _ratings() -> Dict[Tuple[str, str], dict]:
    global _RATINGS_MTIME
    p = root() / "ratings.jsonl"
    try:
        m = p.stat().st_mtime
    except OSError:
        return _RATINGS
    if m != _RATINGS_MTIME:
        for r in _read_jsonl(p):
            if r.get("prompt") == PROMPT_VERSION and r.get("model") == settings.local_sentiment_model:
                _RATINGS[(str(r["sym"]), str(r["id"]))] = r
        _RATINGS_MTIME = m
    return _RATINGS


def rate(sym: str, title: str) -> dict:
    """The local LLM's tone of one headline for ``sym`` — PREREG131's call: {"score": -1 / 0 / 1 or None (an answer
    that is no number: final, left out like the backtest's), "raw"}; {"error": ...} when the server did not answer
    (the decision waits)."""
    import httpx
    url = settings.local_sentiment_base_url.rstrip("/") + "/chat/completions"
    try:
        extra = json.loads(settings.local_sentiment_extra_body or "{}")
    except ValueError:
        extra = {}
    body = {"model": settings.local_sentiment_model, "temperature": 0, "seed": 7, "max_tokens": 8, **extra,
            "messages": [{"role": "user", "content": PROMPT.format(sym=sym, title=str(title).replace('"', "'"))}]}
    err = ""
    for i in range(3):
        try:
            r = httpx.post(url, json=body, timeout=60.0,
                           headers={"Authorization": f"Bearer {settings.local_sentiment_api_key}"})
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"].get("content") or "").strip()
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
            m = _NUM.search(txt)
            return {"score": float(m.group(0)) if m else None, "raw": txt[:40]}
        except Exception as e:                                       # noqa: BLE001
            err = str(e)[:120]
            time.sleep(2 * (i + 1))
    return {"error": err}


def _rated(sym: str, art: dict) -> Optional[dict]:
    """The cached rating of ``art`` for ``sym``, rated now when missing; None when the LLM did not answer (or the
    local LLM is switched off: the decision waits, then gives up)."""
    hit = _ratings().get((sym, art["id"]))
    if hit is not None:
        return hit
    if not getattr(settings, "enable_local_llm", False):
        return None
    r = rate(sym, art["title"])
    if "error" in r:
        return None
    rec = {"sym": sym, "id": art["id"], "score": r["score"], "raw": r["raw"], "prompt": PROMPT_VERSION,
           "model": settings.local_sentiment_model, "rated_at": _iso(datetime.now(timezone.utc))}
    _append(root() / "ratings.jsonl", rec)
    _RATINGS[(sym, art["id"])] = rec
    return rec


def tone_decision(entry: datetime, exit_at: datetime, arts: Sequence[dict],
                  lag_minutes: Optional[int] = None) -> Tuple[Optional[float], List[dict]]:
    """(mean tone, the headlines it counts): the rated headlines published after the entry and at least the lag before
    the exit — `mimic.llm_positive`'s window. Each counted headline carries its ``score`` (None = no number)."""
    lag = timedelta(minutes=int(settings.sel_short_news_hold_lag_minutes if lag_minutes is None else lag_minutes))
    counted = [a for a in arts if entry < _ts(a["published_utc"]) <= exit_at - lag]
    scores = [float(a["score"]) for a in counted if a.get("score") is not None and np.isfinite(float(a["score"]))]
    return (float(np.mean(scores)) if scores else None), counted


def decide(rec: dict, now: Optional[datetime] = None) -> Optional[dict]:
    """The exit's decision record — "hold" (mean tone above 0), "other_news", "no_news" —, or None when Polygon or
    the LLM did not answer (it stays pending)."""
    from src.data import polygon_client as pc
    now = now or datetime.now(timezone.utc)
    t = rec["trade"]
    if eligible(t) == "cover":
        return _decided(rec, "cover", now, headlines=[], mean_tone=None)
    entry, exit_at = _ts(t.get("entry_datetime")), _exit_instant(t)
    if entry is None or exit_at is None:
        return _decided(rec, "no_times", now, headlines=[], mean_tone=None)
    arts = fetch_headlines(str(t["ticker"]), entry, exit_at)
    if arts is None:
        return None
    sym = pc.to_polygon_symbol(str(t["ticker"]))
    lag = timedelta(minutes=int(settings.sel_short_news_hold_lag_minutes))
    for a in arts:
        a["counted"] = entry < _ts(a["published_utc"]) <= exit_at - lag
        if a["counted"]:
            r = _rated(sym, a)
            if r is None:
                return None
            a["score"], a["raw"] = r.get("score"), r.get("raw")
    mean, counted = tone_decision(entry, exit_at, arts)
    what = "no_news" if mean is None else ("hold" if mean > 0 else "other_news")
    return _decided(rec, what, now, headlines=arts, mean_tone=mean, fetched_at=_iso(now))


def _is_overnight(t: Optional[datetime]) -> bool:
    if t is None:
        return False
    e = t.astimezone(ET)
    m = e.hour * 60 + e.minute
    return m >= 20 * 60 or m < 4 * 60


def _decided(rec: dict, what: str, now: datetime, headlines: List[dict], mean_tone: Optional[float],
             **kw) -> dict:
    """Journal the decision — for a hold, its OPEN record first (written before the decision line: a decision
    journaled is never decided again, `run_once` checks it)."""
    t = rec["trade"]
    x = _exit_instant(t)
    out = {"key": rec["key"], "decision": what, "decided_at": _iso(now), "ticker": t.get("ticker"),
           "exit_reason": t.get("exit_reason"), "entry_datetime": t.get("entry_datetime"),
           "exit_at": _iso(x) if x else None, "exit_overnight": _is_overnight(x), "entry_price": t.get("entry_price"),
           "exit_price": t.get("exit_price"), "return_pct": t.get("return_pct"),
           "shares": t.get("sel_account_shares"), "mean_tone": mean_tone,
           "n_counted": sum(1 for a in headlines if a.get("counted")), "n_headlines": len(headlines),
           "headlines": headlines, "prompt": PROMPT_VERSION, "model": settings.local_sentiment_model,
           "lag_minutes": int(settings.sel_short_news_hold_lag_minutes), "backfill": rec.get("backfill", False),
           "trade": t, **kw}
    if what == "hold":
        open_rec = {**out, "cover_multiple": _cover_multiple(), "ks": {str(k): {"status": "pending"} for k in ks()},
                    "primary_k": int(settings.sel_short_news_hold_primary), "last_try": None}
        _write_json(root() / "open" / f"{rec['key']}.json", open_rec)
    _append(_decisions_path(x or now), out)
    logger.info(f"[news_hold] {t.get('ticker')} ({t.get('exit_reason')}): {what}"
                + (f" — mean tone {mean_tone:+.2f} over {out['n_counted']} headline(s)" if mean_tone is not None else "")
                + (" — the shadow would HOLD it" if what == "hold" else ""))
    return out


def _cover_multiple() -> float:
    from src.signals import sel_short
    return float(sel_short.cover_multiple(ARM))


# ── the holds' outcomes ───────────────────────────────────────────────────────

class _Bars:
    def __init__(self, b: dict):
        self.b = b

    def get(self, name):
        return self.b


def bars_in_exit_terms(ticker: str, first: date, last: date, exit_day: date, now: datetime,
                       frame: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """The name's 30-minute bars 04:00-20:00 ET over [first, last] as traded (Polygon, unadjusted), each brought into
    the exit day's terms by the split data (`intraday_store.split_factor_between`), only the bars ended by ``now`` —
    in the simulator's bar dict (`mimic.ExtBars`'s)."""
    from src.backtest import mimic as M
    from src.data import intraday_store
    if frame is None:
        from src.data import polygon_client as pc
        frame = pc.get_intraday_bars_range(ticker, first.isoformat(), last.isoformat(), adjusted=False,
                                           rth_only=False)
    if frame is None or frame.empty:
        return None
    idx = pd.DatetimeIndex(frame.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    start = idx.asi8.astype(np.int64)
    end = start + M.BAR_NS
    keep = end <= _ns(now)
    if not keep.any():
        return None
    et = idx.tz_convert(ET)
    mins = np.asarray(et.hour * 60 + et.minute)
    days = np.asarray((et.normalize().tz_localize(None) - M.EPOCH).days, np.int64)
    rows = intraday_store.split_rows()
    if rows is None:
        return None
    ex = int((pd.Timestamp(exit_day) - M.EPOCH).days)
    g = {}
    for d in np.unique(days):
        dd = (M.EPOCH + pd.Timedelta(days=int(d))).date()
        if d < ex:
            f = intraday_store.split_factor_between(ticker, dd, exit_day, rows)
        elif d > ex:
            f = intraday_store.split_factor_between(ticker, exit_day, dd, rows)
            f = 1.0 / f if f else None
        else:
            f = 1.0
        if not f:
            return None
        g[int(d)] = float(f)
    k = np.array([g[int(d)] for d in days], float)
    col = {c: frame[c].to_numpy(float) * k for c in ("Open", "High", "Low", "Close")}
    return {"start": start[keep], "end": end[keep], "o": col["Open"][keep], "h": col["High"][keep],
            "l": col["Low"][keep], "c": col["Close"][keep], "v": frame["Volume"].to_numpy(float)[keep],
            "rth": ((mins >= 570) & (mins < 960))[keep], "day": days[keep]}


def _piece(rec: dict, b: dict) -> dict:
    """The live trade as the simulator's piece: its entry, its exit mark, its squeeze cover — and as its marks the
    stored regular-hours closes of its last week (the bars are in its terms already: `robust_scale` reads 1)."""
    from src.backtest import mimic as M
    t = rec["trade"]
    ens, xns = _ns(_ts(t["entry_datetime"])), _ns(_exit_instant(t))
    m = b["rth"] & (b["end"] <= xns) & (b["end"] > xns - 7 * M.DAY_NS)
    return {"tkn": str(t["ticker"]), "ens": ens, "xns": xns, "e": float(t["entry_price"]),
            "x": float(t["exit_price"]), "cover_mult": float(rec.get("cover_multiple") or _cover_multiple()),
            "pt": b["end"][m].astype(np.int64), "pc": b["c"][m].astype(float), "why": "target"}


def _session_of(ns: int) -> str:
    e = pd.Timestamp(int(ns)).tz_localize("UTC").tz_convert(ET)
    m = e.hour * 60 + e.minute
    if 570 <= m <= 960:
        return "rth"
    return "extended" if 240 <= m < 1200 else "overnight"


def _rate_of(t: dict) -> Tuple[float, str]:
    """The trade's IBKR borrow rate (%/yr): its last mark's (`borrow_rate_now`), else its entry stamp
    (`borrow_fee_pct`), else `short_borrow_annual_pct`."""
    for k in ("borrow_rate_now", "borrow_fee_pct"):
        v = t.get(k)
        try:
            if v is not None and np.isfinite(float(v)) and float(v) >= 0:
                return float(v), k
        except (TypeError, ValueError):
            continue
    return float(getattr(settings, "short_borrow_annual_pct", 0.0) or 0.0), "default"


def _borrow_added(rec: dict, new_exit_ns: int, b: dict) -> Tuple[Optional[float], str]:
    """The borrow fraction of the entry the ADDED charged days cost — the days IBKR charges through the new exit
    minus those through the actual one (`borrow_fees.charged_days`: settlement to settlement, a pure calendar
    function) — at the trade's rate (`_rate_of`) on 102% of each prior regular-hours close (the bars, in the exit's
    terms), rounded up to the dollar like IBKR's collateral, a 360-day year — and the rate's source; (None, why) when
    it cannot be read. Reads none of the ledger's cached borrow state."""
    import math
    from src.performance.borrow_fees import charged_days
    t = rec["trade"]
    try:
        x = _exit_instant(t)
        new_x = pd.Timestamp(int(new_exit_ns)).tz_localize("UTC").to_pydatetime()
        entry = t.get("entry_datetime") or t.get("entry_date")
        added = sorted(set(charged_days(entry, new_x)) - set(charged_days(entry, x)))
        rate, src = _rate_of(t)
        e = float(t["entry_price"])
        closes = {}
        for day, c, rth in zip(b["day"], b["c"], b["rth"]):
            if rth:
                closes[int(day)] = float(c)                          # the session's last regular-hours close
        sess = np.array(sorted(closes), np.int64)
        epoch = date(1970, 1, 1)
        fee = 0.0
        for d in added:
            j = int(np.searchsorted(sess, (d - epoch).days, side="left")) - 1
            px = closes[int(sess[j])] if j >= 0 else float(t["exit_price"])
            fee += math.ceil(1.02 * px - 1e-9) * rate / 100.0 / 360.0
        return fee / e, src
    except Exception as ex:                                          # noqa: BLE001
        return None, f"error: {str(ex)[:80]}"


def measure(rec: dict, now: Optional[datetime] = None, frame: Optional[pd.DataFrame] = None) -> dict:
    """Measure every pending k of an open hold whose sessions have passed (`mimic.news_extend` on the name's bars in
    the exit's terms): the new exit (instant, price, session, ``news_hold`` / ``news_cap``), the gross change of the
    trade's return (the actual exit price minus the new one, over the entry), the added borrow and their net, in % of
    the entry and in dollars on the trade's shares (None when the borrow cannot be read). A k still unmeasurable
    `GIVE_UP_DAYS` after the exit — bars missing, the session not there — is closed ``unmeasured``. Returns the record
    updated in place."""
    from src.backtest import mimic as M
    now = now or datetime.now(timezone.utc)
    t = rec["trade"]
    x = _exit_instant(t)
    rec["last_try"] = _iso(now)
    give_up = (now - x).days >= GIVE_UP_DAYS

    def _close_pending(why: str) -> None:
        for k, st in rec["ks"].items():
            if st.get("status") == "pending":
                rec["ks"][k] = {"status": "unmeasured", "why": why}
    try:
        exit_day = x.astimezone(ET).date()
        first = _ts(t["entry_datetime"]).astimezone(ET).date() - timedelta(days=10)
        b = bars_in_exit_terms(str(t["ticker"]), first, now.astimezone(ET).date(), exit_day, now, frame=frame)
        if b is None:
            rec["last_error"] = "no_bars"
            if give_up:
                _close_pending("no_bars")
            return rec
        p = _piece(rec, b)
        e, x0 = float(t["entry_price"]), float(t["exit_price"])
        shares = float(t.get("sel_account_shares") or 0.0)
        for k, st in rec["ks"].items():
            if st.get("status") != "pending":
                continue
            q, why = M.news_extend(p, int(k), _Bars(b), lambda d: d)
            if why in ("news_hold", "news_cap"):
                gross = (x0 - float(q["x"])) / e * 100.0
                bor, bsrc = _borrow_added(rec, int(q["xns"]), b)
                net = None if bor is None else gross - bor * 100.0
                rec["ks"][k] = {"status": "done", "why": why, "exit_ns": int(q["xns"]),
                                "exit_at": _iso(pd.Timestamp(int(q["xns"])).tz_localize("UTC").to_pydatetime()),
                                "exit_session": _session_of(int(q["xns"])), "exit_price": float(q["x"]),
                                "gross_pct": round(gross, 4), "borrow_pct": None if bor is None else round(bor * 100.0, 4),
                                "borrow_rate_source": bsrc, "net_pct": None if net is None else round(net, 4),
                                "net_usd": None if net is None else round(net / 100.0 * e * shares, 2)}
            elif give_up:
                rec["ks"][k] = {"status": "unmeasured", "why": why}
            else:
                st["waiting"] = why
    except Exception as ex:                                          # noqa: BLE001
        rec["last_error"] = str(ex)[:200]
        if give_up:
            _close_pending(f"error: {str(ex)[:80]}")
    return rec


def _finished(rec: dict) -> bool:
    return all(st.get("status") != "pending" for st in rec["ks"].values())


# ── the background pass ───────────────────────────────────────────────────────

def _pending_pass(now: datetime, out: dict) -> None:
    for p in sorted((root() / "pending").glob("*.json")):
        try:
            rec = _read_json(p)
            if rec is None:
                continue
            _touch_lock()
            if _was_decided(rec):                                    # decided by an earlier pass: never twice
                p.unlink(missing_ok=True)
                continue
            try:
                d = decide(rec, now)
            except Exception as e:                                   # noqa: BLE001
                logger.warning(f"[news_hold] decision failed for {rec.get('key')} ({e}) — retried at the next kick")
                d = None
            if d is None:
                age = now - (_ts(rec.get("journaled_at")) or now)
                if age > timedelta(hours=PENDING_GIVE_UP_HOURS):
                    _decided(rec, "undecided", now, headlines=[], mean_tone=None,
                             why="Polygon's news or the local LLM unavailable for "
                                 f"{PENDING_GIVE_UP_HOURS} h")
                    p.unlink(missing_ok=True)
                    out["decided"] += 1
                else:
                    out["waiting"] += 1
                continue
            p.unlink(missing_ok=True)
            out["decided"] += 1
        except Exception as e:                                       # noqa: BLE001 — one item never stops the pass
            logger.warning(f"[news_hold] pending {p.name}: {e}")


def _open_pass(now: datetime, out: dict) -> None:
    for p in sorted((root() / "open").glob("*.json")):
        try:
            rec = _read_json(p)
            if rec is None:
                continue
            last = _ts(rec.get("last_try"))
            if last is not None and now - last < timedelta(minutes=RETRY_MINUTES):
                continue
            _touch_lock()
            measure(rec, now)
            out["measured"] += 1
            if _finished(rec):
                rec["settled_at"] = _iso(now)
                _append(_settled_path(_ts(rec.get("exit_at")) or now), rec)
                p.unlink(missing_ok=True)
                out["settled"] += 1
                pk = rec["ks"].get(str(rec.get("primary_k")), {})
                logger.info(f"[news_hold] {rec.get('ticker')}: held {rec.get('primary_k')} sessions the short would "
                            f"have changed by {pk.get('net_pct')}% of its entry ({pk.get('why')})")
            else:
                _write_json(p, rec)
        except Exception as e:                                       # noqa: BLE001 — one item never stops the pass
            logger.warning(f"[news_hold] open {p.name}: {e}")


def run_once(now: Optional[datetime] = None) -> dict:
    """Decide every pending exit and measure every open hold due a try (finished holds move to ``settled/``) — one
    pass at a time across processes (`run.lock`; a pass that finds it held returns ``locked``). Returns a summary."""
    now = now or datetime.now(timezone.utc)
    out = {"decided": 0, "waiting": 0, "measured": 0, "settled": 0}
    if not _acquire():
        return {**out, "locked": 1}
    try:
        _pending_pass(now, out)
        _open_pass(now, out)
    finally:
        _release()
    return out


def _has_work() -> bool:
    return any((root() / d).is_dir() and any((root() / d).glob("*.json")) for d in ("pending", "open"))


def kick(now: Optional[datetime] = None) -> bool:
    """Run `run_once` in a daemon thread unless one is running or nothing waits (single-flight)."""
    global _THREAD
    if not enabled() or not _has_work():
        return False
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return False

        def _work():
            try:
                s = run_once()
                if any(v for k, v in s.items() if k != "locked"):
                    logger.info(f"[news_hold] background pass: {s}")
            except Exception as e:                                   # noqa: BLE001 — never the tick's problem
                logger.warning(f"[news_hold] background pass failed: {e}")

        _THREAD = threading.Thread(target=_work, name="news-hold", daemon=True)
        _THREAD.start()
    return True


# ── reading ───────────────────────────────────────────────────────────────────

def status() -> dict:
    dec = [r for p in sorted((root() / "decisions").glob("*.jsonl")) for r in _read_jsonl(p)]
    st = [r for p in sorted((root() / "settled").glob("*.jsonl")) for r in _read_jsonl(p)]
    pend = [_read_json(p) or {} for p in sorted((root() / "pending").glob("*.json"))]
    opn = [_read_json(p) or {} for p in sorted((root() / "open").glob("*.json"))]
    by = {}
    for r in dec:
        by[r.get("decision")] = by.get(r.get("decision"), 0) + 1
    k0 = str(int(settings.sel_short_news_hold_primary))
    prim = [r["ks"][k0]["net_pct"] for r in st if r.get("ks", {}).get(k0, {}).get("status") == "done"
            and r["ks"][k0].get("net_pct") is not None]
    return {"decisions": by, "pending": len(pend), "open": len(opn), "settled": len(st),
            "primary_k": int(k0), "primary_net_pct": prim,
            "oldest_pending": min((r.get("journaled_at") for r in pend if r.get("journaled_at")), default=None)}


def health(now: Optional[datetime] = None) -> dict:
    """Notes for the scorer banner's green line — a shadow never alarms: what the shadow decided and measured, and
    exits whose decision has waited `STALE_PENDING_HOURS` (the local LLM or Polygon's news unreachable)."""
    now = now or datetime.now(timezone.utc)
    notes: List[str] = []
    if not enabled():
        return {"problems": [], "notes": notes}
    try:
        s = status()
    except Exception as e:                                           # noqa: BLE001
        return {"problems": [], "notes": [f"news hold (shadow): journal unreadable ({e})"]}
    old = _ts(s.get("oldest_pending"))
    if old is not None and now - old > timedelta(hours=STALE_PENDING_HOURS):
        notes.append(f"news hold (shadow): {s['pending']} vol exit(s) undecided since {old.astimezone(ET):%m-%d %H:%M} "
                     f"ET — the local LLM or Polygon's news unreachable")
    n = sum(s["decisions"].values())
    if n:
        holds = s["decisions"].get("hold", 0)
        txt = f"news hold (shadow): {n} vol exit(s) read, {holds} would have been held"
        if s["primary_net_pct"]:
            v = s["primary_net_pct"]
            txt += (f"; {s['primary_k']} more sessions measured on {len(v)}: {float(np.mean(v)):+.2f}% of the entry "
                    f"on average")
        if s["open"]:
            txt += f"; {s['open']} hold(s) still running"
        notes.append(txt)
    return {"problems": [], "notes": notes}


def backfill(since: date, now: Optional[datetime] = None) -> int:
    """Journal the ledger's vol exits closed since ``since`` that the shadow never read (flagged ``backfill``: their
    headlines are fetched now, still cut at the exit minus the lag) — without kicking the background thread (the CLI
    runs the pass itself)."""
    from src.performance import tracker
    now = now or datetime.now(timezone.utc)
    trades = [t for t in tracker._load_trades() if eligible(t)
              and (_exit_instant(t) or now).astimezone(ET).date() >= since]
    return after_exits(trades, now, backfill=True, kick_after=False)


def probe(recommendation_id: str) -> dict:
    """A dry run on any closed selection short (any arm): its headlines, their tones, the decision and — whatever
    the decision — the outcome for every k. Writes nothing but the ratings cache."""
    from src.performance import tracker
    t = next((x for x in tracker._load_trades() if x.get("recommendation_id") == recommendation_id), None)
    if t is None:
        return {"error": f"no trade {recommendation_id}"}
    from src.data import polygon_client as pc
    rec = {"key": _key(t), "trade": {k: t.get(k) for k in _TRADE_FIELDS}}
    entry, x = _ts(t.get("entry_datetime")), _exit_instant(t)
    arts = fetch_headlines(str(t["ticker"]), entry, x) or []
    sym = pc.to_polygon_symbol(str(t["ticker"]))
    lag = timedelta(minutes=int(settings.sel_short_news_hold_lag_minutes))
    for a in arts:
        a["counted"] = entry < _ts(a["published_utc"]) <= x - lag
        if a["counted"]:
            r = _rated(sym, a) or {}
            a["score"], a["raw"] = r.get("score"), r.get("raw")
    mean, _ = tone_decision(entry, x, arts)
    from src.signals import sel_short
    rec.update({"cover_multiple": float(sel_short.cover_multiple(t.get("sel_arm"))),
                "ks": {str(k): {"status": "pending"} for k in ks()}})
    measure(rec)
    return {"ticker": t.get("ticker"), "arm": t.get("sel_arm"), "exit_reason": t.get("exit_reason"),
            "mean_tone": mean, "headlines": arts, "ks": rec["ks"], "last_error": rec.get("last_error")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.signals.news_hold")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--run", action="store_true", help="one background pass, in the foreground")
    g.add_argument("--backfill-from", metavar="YYYY-MM-DD")
    g.add_argument("--probe", metavar="RECOMMENDATION_ID")
    a = ap.parse_args(argv)
    from src.db import repo
    repo.set_read_only(True)                 # the scheduler is the database's only writer; the shadow writes no row
    if a.status:
        print(json.dumps(status(), indent=1, default=str))
    elif a.run:
        print(json.dumps(run_once(), indent=1))
    elif a.backfill_from:
        n = backfill(date.fromisoformat(a.backfill_from))
        print(f"journaled {n} exit(s)")
        print(json.dumps(run_once(), indent=1))
    else:
        print(json.dumps(probe(a.probe), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
