"""THE VERIFIED SPLIT TABLE — every split record checked against the prices that traded, and the corrections the price
stores and the ledger apply (user directive 2026-10-16: "Apply the fixes. Also, implement automated verification or
testing and guardrails to prevent error in the data on the future (i.e. wrong or missing splits)").

Polygon's split records (the deep store's `splits` family) decide every store's adjustment (production's 30-minute
store resets on them, Polygon's ADJUSTED bars apply them) and the ledger's rescaling of open trades
(`tracker.apply_splits`). They can be WRONG: XRX's "2-for-3" of 2026-02-09 never happened — yfinance trades it
continuously at ~$2.20, but Polygon's adjusted bars scaled every earlier bar by 2/3, a fake +50% that day in our
store. Or MISSING: a split that traded but has no record leaves the old scale in every adjusted series.

THE TABLE (`CHECKS_PATH`, JSON, atomic): ``{"TICKER|YYYY-MM-DD": record}``, ``record["status"]``:
  * ``verified``   — the raw (unadjusted) prices moved by the split's factor that day;
  * ``false``      — they did not (within `NOSPLIT_TOL`) on its date nor through the `MISDATE_SESSIONS` sessions after
                     it, and a second source (yfinance) holds no split then: the record is LEFT OUT of every effective
                     set and Polygon's adjustment is UNDONE;
  * ``suspect``    — the raw prices say "no split" but the second source could not be asked, or the sessions after
                     its date have not all traded yet (``provisional``: a record dated a day early shows its move then,
                     and yfinance lists a split only once it has traded): still applied, alerted, re-checked;
  * ``ambiguous``  — the move fits neither (a big move on a small split): still applied, reported;
  * ``missing``    — a split the records lack, found as a jump of a standard ratio in production's store and
                     CONFIRMED by yfinance: applied to the PRICE STORES (`corrections`), never to the ledger (a spin-off
                     looks the same and leaves IBKR's share count unchanged);
  * ``candidate``  — such a jump yfinance did not confirm: reported, never applied;
  * ``pending`` / ``unverifiable`` — not traded yet / no raw prices (a delisted name): applied as recorded.

Conventions: ``factor`` = split_from / split_to = the price AFTER over the price BEFORE (a 1-for-10 reverse split: 10;
a 2-for-1 split: 0.5) — `intraday_store.split_factor_between`'s. `corrections(ticker)` lists, per execution date, the
multiplier that turns a Polygon-ADJUSTED price before that date into the true adjusted one: a false record x 1/factor
(undo), a missing split x factor (apply); `correct_frame` applies them (prices x m, volumes / m).

Every price store applies `correct_frame` to what it fetches (`intraday_store._fetch_range`, `market_data.get_history`,
`deep.polygon_deep.bars30m_full`); `intraday_store.split_rows` / `latest_splits` return the EFFECTIVE set (the records
minus the false ones), so the store resets, the ledger, the broker and every other reader follow it; a confirmed missing
split corrects the price stores only (`corrections`: it can be a spin-off, which IBKR's share count ignores). The checks run
in `src.data.integrity` (the nightly deep refresh, the pre-open run) and, for a held name on its split day, at the
tick's marks (`verify_held`: from 04:00 ET of the split's date, on Polygon's raw prices of that day — never a mark;
it alerts, it never drops a record). Tests: tests/test_split_check.py.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

CHECKS_PATH = Path("cache/ml/deep/split_checks.json")
ET = "America/New_York"
SPLIT_TOL = 0.20          # |ln(observed) - ln(factor)| within this: the prices moved by the split
NOSPLIT_TOL = 0.10        # |ln(observed)| within this: the prices did not move like a split
MIN_FACTOR_LN = 0.25      # a split must move the price at least this much (ln) to be told from a day's move
YF_RATIO_TOL = 0.05       # yfinance's ratio matches the record's within this (ln)
YF_DAYS = 5               # ... dated within this many days
# the ratios a missing split is searched at (price after / before), with their tolerance
STANDARD_FACTORS = tuple(sorted({1 / k for k in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)}
                                | {float(k) for k in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)}
                                | {2 / 3, 3 / 2}))
MISSING_TOL = 0.03
# a contradicted record is called false only once its date and the sessions after it have traded: a record dated a day
# or two early shows its move then, and yfinance lists a split only once it has traded (until then: suspect, applied)
MISDATE_SESSIONS = 5
PENDING_DAYS = 10         # a past record with no session after it this many days on: the name stopped trading there
EFFECT_HOUR_ET = 4        # a split takes effect at 04:00 ET of its date (`tracker._split_day`); every price before quotes
#                           the old terms and says "no split" of every real split
SUSPECT_RECHECK_MINUTES = 60
_LOCK = threading.Lock()
_CACHE: Dict[str, tuple] = {}

STATUS_APPLIED = ("verified", "suspect", "ambiguous", "pending", "unverifiable", "unchecked")


def key(ticker: str, d: date) -> str:
    return f"{str(ticker).upper()}|{d.isoformat()}"


# ── the table ─────────────────────────────────────────────────────────────────

def _valid(data) -> Dict[str, dict]:
    """The table's own shape — ``{"TICKER|YYYY-MM-DD": {...}}`` — of a parsed file (ValueError otherwise; records of
    another shape are left out)."""
    if not isinstance(data, dict):
        raise ValueError(f"not a JSON object ({type(data).__name__})")
    return {str(k): v for k, v in data.items() if isinstance(v, dict) and "|" in str(k)}


def load_checks(path: Optional[Path] = None) -> Dict[str, dict]:
    """The table (cached per file version); {} when absent. An unreadable or malformed file keeps the last version
    this process read (else {}: every record applied as recorded) — a hand edit never breaks the ledger's readers."""
    p = Path(path or CHECKS_PATH)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    hit = _CACHE.get(p.as_posix())
    if hit is not None and hit[0] == mt:
        return hit[1]
    try:
        data = _valid(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:                                       # noqa: BLE001
        data = hit[1] if hit is not None else {}
        logger.warning(f"[split_check] {p} unreadable ({e}) — "
                       f"{'the last version read is kept' if hit is not None else 'every split record applied as recorded'}")
    _CACHE[p.as_posix()] = (mt, data)
    return data


def _lock_file(p: Path, timeout: float = 20.0, stale: float = 120.0) -> Optional[Path]:
    """An exclusive lock beside the table across PROCESSES (the nightly refresh, the pre-open run and the scheduler all
    write it): a file created with O_EXCL; one older than ``stale`` seconds is a dead writer's. None on a timeout (the
    write goes ahead unlocked: losing it would be worse)."""
    lock = p.with_name(p.name + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink()
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                logger.warning(f"[split_check] {lock} held for {timeout:.0f}s — writing without it")
                return None
            time.sleep(0.05)
        except OSError:
            return None


def _replace(tmp: Path, p: Path, tries: int = 6) -> None:
    """`os.replace` with retries: on Windows it fails (WinError 5) while another process holds the target open."""
    for i in range(tries):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.2 * (i + 1))


def save_checks(updates: Dict[str, dict], path: Optional[Path] = None) -> Dict[str, dict]:
    """Merge ``updates`` into the table (one read-modify-write at a time across threads and processes, atomic
    replace); returns the table."""
    p = Path(path or CHECKS_PATH)
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        lock = _lock_file(p)
        try:
            cur = {}
            if p.exists():
                try:
                    cur = _valid(json.loads(p.read_text(encoding="utf-8")))
                except Exception as e:                           # noqa: BLE001
                    hit = _CACHE.get(p.as_posix())
                    cur = dict(hit[1]) if hit is not None else {}
                    logger.warning(f"[split_check] {p} unreadable before a write ({e}) — rewritten from "
                                   f"{'the last version read' if hit is not None else 'the updates alone'}")
            cur.update(updates)
            tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cur, indent=1, sort_keys=True, default=str), encoding="utf-8")
            _replace(tmp, p)
            _CACHE.pop(p.as_posix(), None)
            return cur
        finally:
            if lock is not None:
                try:
                    lock.unlink()
                except OSError:
                    pass


def status_of(ticker: str, d: date, checks: Optional[Dict[str, dict]] = None) -> str:
    checks = load_checks() if checks is None else checks
    rec = checks.get(key(ticker, d)) if isinstance(checks, dict) else None
    return str(rec.get("status") or "unchecked") if isinstance(rec, dict) else "unchecked"


def effective_rows(rows: Dict[str, List[Tuple[date, float, float]]],
                   checks: Optional[Dict[str, dict]] = None,
                   with_missing: bool = False) -> Dict[str, List[Tuple[date, float, float]]]:
    """``rows`` (Polygon's records, `intraday_store.split_rows`' shape) minus the FALSE ones — the set the LEDGER and the
    broker follow. ``with_missing`` also adds the MISSING ones; they are left out by default because the jump a second
    source calls a split can be a SPIN-OFF (POST 2022-03-11: BellRing, which yfinance lists as a 1.5 split): it belongs
    in an adjusted price series (`corrections`), never in a held position's share count, which IBKR does not change."""
    checks = load_checks() if checks is None else checks
    if not checks:
        return rows
    out: Dict[str, List[Tuple[date, float, float]]] = {}
    for tk, lst in rows.items():
        keep = [(d, a, b) for d, a, b in lst if status_of(tk, d, checks) != "false"]
        if keep:
            out[tk] = keep
    if not with_missing:
        return out
    for k, rec in checks.items():
        if not isinstance(rec, dict) or rec.get("status") != "missing":
            continue
        try:
            tk, ds = k.split("|", 1)
            d = date.fromisoformat(ds)
            a, b = float(rec["from"]), float(rec["to"])
        except (KeyError, TypeError, ValueError):
            continue
        lst = out.setdefault(tk, [])
        if not any(x == d for x, _, _ in lst):
            lst.append((d, a, b))
    return out


def corrections(ticker: str, checks: Optional[Dict[str, dict]] = None) -> List[Tuple[date, float]]:
    """``[(execution date, multiplier for the Polygon-adjusted prices before it)]`` of ``ticker``."""
    checks = load_checks() if checks is None else checks
    out = []
    pre = f"{str(ticker).upper()}|"
    for k, rec in checks.items():
        if not k.startswith(pre):
            continue
        st = rec.get("status")
        if st not in ("false", "missing"):
            continue
        try:
            f = float(rec["from"]) / float(rec["to"])
            d = date.fromisoformat(k.split("|", 1)[1])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            logger.warning(f"[split_check] {k}: a {st} record without a readable ratio or date — not applied")
            continue
        if not (f > 0 and math.isfinite(f)):
            continue
        out.append((d, 1.0 / f if st == "false" else f))
    return sorted(out)


def correct_frame(ticker: str, df: pd.DataFrame, *, utc_naive: bool = True, ts_col: Optional[str] = None,
                  checks: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """``df`` with `corrections` applied: the bars dated (ET session) before each correction's date have their prices
    x the multiplier and their volume / it. ``utc_naive``: the index (or ``ts_col``) holds UTC-naive bar starts (the
    30-minute stores); else naive session dates (the daily cache). Unchanged without corrections."""
    if df is None or len(df) == 0:
        return df
    corr = corrections(ticker, checks)
    if not corr:
        return df
    out = df.copy()
    t = pd.DatetimeIndex(pd.to_datetime(out[ts_col]) if ts_col else out.index)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    day = (t.tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize() if utc_naive else t.normalize())
    price_cols = [c for c in out.columns if str(c).lower() in ("open", "high", "low", "close", "vwap", "o", "h", "l", "c")]
    vol_cols = [c for c in out.columns if str(c).lower() in ("volume", "v")]
    for c in price_cols + vol_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for d, m in corr:
        sel = np.asarray(day < pd.Timestamp(d))
        if not sel.any():
            continue
        for c in price_cols:
            out.loc[sel, c] = out.loc[sel, c] * m
        for c in vol_cols:
            out.loc[sel, c] = out.loc[sel, c] / m
    return out


# ── the verdicts (pure: tested on synthetic prices) ───────────────────────────

def judge(factor: float, before_close: float, after_open: Optional[float], after_close: Optional[float]) -> dict:
    """The raw-price verdict on one split record: ``verified`` (moved by the factor), ``nosplit`` (moved like no
    split), ``ambiguous`` (neither), ``tiny`` (a factor too close to 1 to tell from a day's move)."""
    obs = [x / before_close for x in (after_open, after_close) if x is not None and x > 0 and before_close > 0]
    if not obs:
        return {"verdict": "unverifiable"}
    lf = math.log(factor)
    dev = min(abs(math.log(r) - lf) for r in obs)
    dev0 = min(abs(math.log(r)) for r in obs)
    out = {"observed": [round(r, 5) for r in obs], "dev": round(dev, 4), "dev0": round(dev0, 4)}
    if abs(lf) < MIN_FACTOR_LN:
        out["verdict"] = "verified" if dev <= 0.03 else "tiny"
    elif dev <= SPLIT_TOL and dev < dev0:
        out["verdict"] = "verified"
    elif dev0 <= NOSPLIT_TOL:
        out["verdict"] = "nosplit"
    else:
        out["verdict"] = "ambiguous"
    return out


def yf_confirms(yf_splits: Optional[pd.Series], d: date, factor: float) -> Optional[bool]:
    """yfinance's split list against a record: True (a split of the same factor within `YF_DAYS`), False (none),
    None (the list could not be read). yfinance's ratio is new shares per old share: the price factor is 1 / it."""
    if yf_splits is None:
        return None
    for t, ratio in yf_splits.items():
        try:
            td = pd.Timestamp(t).tz_localize(None).date() if pd.Timestamp(t).tzinfo else pd.Timestamp(t).date()
            if abs((td - d).days) <= YF_DAYS and ratio and abs(math.log(1.0 / float(ratio)) - math.log(factor)) <= YF_RATIO_TOL:
                return True
        except Exception:                                        # noqa: BLE001
            continue
    return False


def decide(raw: dict, yf: Optional[bool]) -> str:
    """The table's status from the raw verdict and yfinance's answer."""
    v = raw.get("verdict")
    if v == "verified":
        return "verified"
    if v == "unverifiable":
        return "unverifiable"
    if yf is True:
        return "verified"                                        # the second source holds it: keep it
    if v == "nosplit":
        return "false" if yf is False else "suspect"
    return "ambiguous"                                           # "ambiguous" / "tiny": applied as recorded


# ── the data the checks read ──────────────────────────────────────────────────

def raw_daily(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Polygon's UNADJUSTED daily bars (o / c by naive session date), or None."""
    try:
        from src.data import polygon_client as pc
        data = pc._get(f"/v2/aggs/ticker/{pc.to_polygon_symbol(ticker)}/range/1/day/{start.isoformat()}/{end.isoformat()}",
                       {"adjusted": "false", "sort": "asc", "limit": 5000})
    except Exception as e:                                       # noqa: BLE001
        logger.debug(f"[split_check] {ticker}: raw bars failed ({e})")
        return None
    res = (data or {}).get("results") or []
    if not res:
        return None
    df = pd.DataFrame({"o": [float(r["o"]) for r in res], "c": [float(r["c"]) for r in res]},
                      index=pd.DatetimeIndex([pd.Timestamp(r["t"], unit="ms", tz="UTC").tz_convert(ET).tz_localize(None)
                                              .normalize() for r in res]))
    return df[~df.index.duplicated(keep="last")]


def yf_split_list(ticker: str) -> Optional[pd.Series]:
    """yfinance's split history of ``ticker`` (new shares per old), or None when it cannot be read — an EMPTY list
    from a symbol yfinance does not know (no price history either: CBCY, an OTC name) is no evidence of "no split"."""
    try:
        import yfinance as yf
        t = yf.Ticker(str(ticker).replace(".", "-"))
        s = t.splits
        if s is not None and len(s):
            return s
        h = t.history(period="1mo", timeout=30)
        return pd.Series(dtype=float) if h is not None and len(h) else None
    except Exception as e:                                       # noqa: BLE001
        logger.debug(f"[split_check] {ticker}: yfinance splits failed ({e})")
        return None


def verify(ticker: str, d: date, split_from: float, split_to: float, *, today: Optional[date] = None,
           raw: Optional[Callable[[str, date, date], Optional[pd.DataFrame]]] = None,
           yf: Optional[Callable[[str], Optional[pd.Series]]] = None) -> dict:
    """One record's check: the raw close of the session before ``d`` against the raw open / close of ``d``'s
    session (the first on or after it); yfinance asked only when the raw prices do not confirm the split. A record the
    prices contradict is ``false`` only once ``d`` and the `MISDATE_SESSIONS` sessions after it have traded (a record
    dated early shows its move then, and yfinance lists it once it has traded): before, ``suspect`` (provisional). A
    past record with no session after it `PENDING_DAYS` days on is ``unverifiable`` (the name stopped trading), not
    pending."""
    today = today or date.today()
    raw = raw or raw_daily
    yf = yf or yf_split_list
    factor = float(split_from) / float(split_to)
    rec = {"from": float(split_from), "to": float(split_to), "factor": round(factor, 8),
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if d > today:
        return dict(rec, status="pending")
    bars = raw(ticker, d - timedelta(days=10), d + timedelta(days=2 * MISDATE_SESSIONS + 4))
    if bars is None or bars.empty:
        return dict(rec, status="unverifiable", why="no raw prices")
    before = bars[bars.index < pd.Timestamp(d)]
    after = bars[bars.index >= pd.Timestamp(d)]
    if before.empty:
        return dict(rec, status="unverifiable", why="no session before it")
    if after.empty:
        gone = (today - d).days > PENDING_DAYS
        return dict(rec, status="unverifiable" if gone else "pending",
                    why="no session after it: the name stopped trading" if gone else "no session on or after it yet")
    j = judge(factor, float(before["c"].iloc[-1]), float(after["o"].iloc[0]), float(after["c"].iloc[0]))
    rec.update({k: v for k, v in j.items() if k != "verdict"}, raw_verdict=j["verdict"],
               before_day=str(before.index[-1].date()), after_day=str(after.index[0].date()))
    y = None if j["verdict"] in ("verified", "unverifiable") else yf_confirms(yf(ticker), d, factor)
    rec["yfinance"] = y
    st = decide(j, y)
    traded = int(len(after))
    if st == "false" and traded < 1 + MISDATE_SESSIONS and (today - d).days <= 30:
        st = "suspect"
        rec.update(provisional=True, why=f"no split in the raw prices of {traded} of the {1 + MISDATE_SESSIONS} "
                                         f"sessions from its date: false once they have all traded")
    rec["status"] = st
    return rec


def verify_records(rows: Dict[str, List[Tuple[date, float, float]]], *, names: Optional[Iterable[str]] = None,
                   since: Optional[date] = None, today: Optional[date] = None, recheck: Sequence[str] = ("pending",),
                   budget: int = 4000, **kw) -> Dict[str, dict]:
    """Check every record of ``names`` (default all) executed since ``since`` that the table holds no verdict for
    (or one of ``recheck``); returns the updates (saved)."""
    today = today or date.today()
    checks = load_checks()
    want = {str(n).upper() for n in names} if names is not None else None
    updates: Dict[str, dict] = {}
    for tk, lst in sorted(rows.items()):
        if want is not None and tk not in want:
            continue
        for d, a, b in lst:
            if since is not None and d < since:
                continue
            k = key(tk, d)
            if k in checks and checks[k].get("status") not in recheck:
                continue
            if len(updates) >= budget:
                break
            try:
                updates[k] = verify(tk, d, a, b, today=today, **kw)
            except Exception as e:                               # noqa: BLE001
                logger.warning(f"[split_check] {k}: check failed ({e})")
    if updates:
        save_checks(updates)
        bad = {k: v["status"] for k, v in updates.items() if v["status"] in ("false", "suspect")}
        if bad:
            logger.warning(f"[split_check] split records the traded prices contradict: {bad}")
    return updates


# ── missing splits: jumps of a standard ratio no record explains ─────────────

def jump_candidates(o: pd.Series, c: pd.Series, known: Sequence[date], *, lookback: Optional[int] = None) -> List[dict]:
    """Session jumps (the session's first open over the previous close) within `MISSING_TOL` of a standard split
    ratio that persist two sessions and no record within 3 days explains — on one name's session opens / closes."""
    out = []
    days = list(c.index)
    lo = 1 if lookback is None else max(1, len(days) - lookback)
    for i in range(lo, len(days)):
        prev, op = float(c.iloc[i - 1]), float(o.iloc[i])
        if not (prev > 0 and op > 0):
            continue
        g = op / prev
        f = min(STANDARD_FACTORS, key=lambda x: abs(math.log(g) - math.log(x)))
        if abs(math.log(g) - math.log(f)) > MISSING_TOL:
            continue
        later = float(c.iloc[min(i + 2, len(days) - 1)])
        if not (later > 0 and abs(math.log(later / prev) - math.log(f)) <= 0.15):
            continue
        d = pd.Timestamp(days[i]).date()
        if any(abs((d - k).days) <= 3 for k in known):
            continue
        out.append({"day": d, "factor": f, "gap": g})
    return out


def find_missing(names: Iterable[str], rows: Dict[str, List[Tuple[date, float, float]]], *, lookback: int = 30,
                 load: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
                 yf: Optional[Callable[[str], Optional[pd.Series]]] = None) -> Dict[str, dict]:
    """Scan production's 30-minute store (the last ``lookback`` sessions of each name) for missing splits; each jump
    yfinance confirms becomes ``missing`` (added), the others ``candidate``. Returns the updates (saved)."""
    if load is None:
        from src.data.intraday_store import load_deep_30m as load
    yf = yf or yf_split_list
    checks = load_checks()
    updates: Dict[str, dict] = {}
    for tk in sorted({str(n).upper() for n in names}):
        if _not_a_common_listing(tk):
            continue
        try:
            df = load(tk)
        except Exception:                                        # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        et = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
        o = pd.to_numeric(df["Open"], errors="coerce").groupby(et).first()
        c = pd.to_numeric(df["Close"], errors="coerce").groupby(et).last()
        if "Volume" in df:
            dv = (pd.to_numeric(df["Close"], errors="coerce") * pd.to_numeric(df["Volume"], errors="coerce")).groupby(et).sum()
            if float(dv.tail(60).median()) < MISSING_MIN_DV:
                continue                                         # an illiquid name's prints jump on their own
        known = [d for d, _, _ in rows.get(tk, [])]
        for cand in jump_candidates(o, c, known, lookback=lookback):
            k = key(tk, cand["day"])
            if k in checks:
                continue
            ys = yf(tk)
            conf = yf_confirms(ys, cand["day"], cand["factor"])
            frac = _as_fraction(cand["factor"])
            updates[k] = {"status": "missing" if conf else "candidate", "from": frac[0], "to": frac[1],
                          "factor": round(cand["factor"], 8), "gap": round(cand["gap"], 5), "yfinance": conf,
                          "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "why": "a jump of a standard split ratio no record explains"}
    if updates:
        save_checks(updates)
        logger.warning(f"[split_check] jumps no split record explains: "
                       f"{ {k: v['status'] for k, v in updates.items()} }")
    return updates


MISSING_MIN_DV = 1_000_000.0      # the missing-split search skips names trading under $1M a day (60-session median)


def _not_a_common_listing(tk: str) -> bool:
    """A warrant / right / unit (a 5+ letter Nasdaq symbol ending W, R or U; ``.WS``): their prints jump by 1.5x or 2x
    on their own, and no live book trades them — the missing-split search leaves them out."""
    import re
    return bool(re.match(r"^[A-Z]{4,}[WRU]$", tk)) or tk.endswith(".WS") or tk.endswith("-WS")


def _as_fraction(f: float) -> Tuple[float, float]:
    """(split_from, split_to) of a price factor: 10 -> (10, 1) (a 1-for-10 reverse), 0.5 -> (1, 2)."""
    return (float(round(f, 6)), 1.0) if f >= 1 else (1.0, float(round(1.0 / f, 6)))


# ── the live guard: a held name on its split day ─────────────────────────────

def verify_held(trades: List[dict], *, now: Optional[datetime] = None, window_days: int = 3,
                raw: Optional[Callable[[str, date, date], Optional[pd.DataFrame]]] = None,
                yf: Optional[Callable[[str], Optional[pd.Series]]] = None) -> Dict[str, dict]:
    """THE LIVE GUARD (`tracker.update_open_trades`, after the marks): for each OPEN trade whose ticker has an effective
    split executed within ``window_days`` that the table has not decided, the record's check (`verify`) on Polygon's
    RAW prices of its own date — only once 04:00 ET of that date has passed (`EFFECT_HOUR_ET`: before it every price
    quotes the old terms and says "no split" of a real split), never on a mark (a stale mark, or one
    `tracker.apply_splits` has already converted, proves nothing). A date with no trade yet stays pending (re-checked
    at the next tick). A contradiction is ALERTED (CRITICAL; ``suspect``, still applied, re-checked at most every
    `SUSPECT_RECHECK_MINUTES`): the guard never drops a record — `verify` calls one false only once its date and the
    sessions after it have traded (the nightly check), and the next ledger load restores the trade then
    (`tracker.apply_splits` undoes a rescale whose record left the effective set)."""
    from src.data.intraday_store import split_rows
    now = now or datetime.now(timezone.utc)
    ts = pd.Timestamp(now)
    now_et = (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(ET)
    today = now_et.date()
    rows = split_rows() or {}
    checks = load_checks()
    updates: Dict[str, dict] = {}
    for t in trades:
        if t.get("status") != "OPEN" or not t.get("ticker"):
            continue
        tk = str(t["ticker"]).upper()
        for d, a, b in rows.get(tk, []):
            if not (today - timedelta(days=window_days) <= d <= today):
                continue
            if now_et < pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=EFFECT_HOUR_ET):
                continue                                     # not in effect yet: every price quotes the old terms
            k = key(tk, d)
            prev = checks.get(k) or {}
            st = prev.get("status")
            if k in updates or st in ("verified", "false", "missing", "ambiguous", "unverifiable"):
                continue
            if st == "suspect" and _minutes_since(prev.get("checked_at"), now_et) < SUSPECT_RECHECK_MINUTES:
                continue
            r = verify(tk, d, a, b, today=today, raw=raw, yf=yf)
            if r["status"] == "false":                       # verify's own gate keeps a fresh record from it
                r = dict(r, status="suspect", provisional=True)
            if r["status"] == "pending":
                continue
            updates[k] = dict(r, how="held: raw prices of its date",
                              checked_at=now_et.tz_convert("UTC").isoformat(timespec="seconds"))
            if r["status"] == "suspect" and st != "suspect":
                logger.critical(f"[split_check] {k}: a HELD name's split record (factor {r.get('factor')}) that the "
                                f"raw prices of its date contradict ({r.get('observed')}, yfinance "
                                f"{r.get('yfinance')}) — still applied; it leaves the effective set only once its "
                                f"sessions have traded with no split (the nightly check): check it")
    if updates:
        save_checks(updates)
    return updates


def _minutes_since(iso, now_et: pd.Timestamp) -> float:
    try:
        t = pd.Timestamp(str(iso))
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        return (now_et - t).total_seconds() / 60.0
    except Exception:                                            # noqa: BLE001
        return float("inf")
