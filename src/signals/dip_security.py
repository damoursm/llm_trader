"""THE OWN-SECURITY SERIES — each dip candidate's history read from its OWN security (user 2026-10-18: "Yes please
apply the fix"; PREREG142's construction with its Amendments 1-3, `llm_trader_research/studies/prereg142/`
spans142.py / panel142.py, ported here function for function).

The deep 30-minute store keeps whatever traded under a ticker. A REUSED ticker stitches an older security in front of
the new one: SpaceX's SPCX (listed 2026-06-12) sat behind a SPAC ETF's bars, so the rule's 220 bars and 200-session
average were the fund's and SpaceX was a dip signal on 2026-10-15 / 16 (AUDIT127). A RENAMED company's history sits
under its earlier ticker: EchoStar traded as SATS until 2026-06-23, so the store's ECHO holds ~80 sessions of it.
Before the dip rule reads a name, its stored bars are rebuilt from its security's TICKER RUNS (Polygon's ticker events):

* RUNS: the events' ``ticker_change`` entries in date order, consecutive entries of one ticker merged; without events,
  the ticker details' list date as the start of one run; a name none of whose runs carries its own ticker (GOOGL,
  BRK-B) is read as stored.
* JOINS: every run under ANOTHER ticker overlapping the store's window (from 2021-01-04) is that ticker's
  regular-hours 30-minute bars over the run's days, UNADJUSTED (fetched once into ``runs/``), brought into today's
  terms by the security's own splits (each split counted inside its ticker's run). A run is GENUINE with bars on at
  least 20 sessions. On the days the stored series has bars too: closes within 2% (median |log ratio|) = the same
  security (the stored bars kept); more = another security used the name then (the stored bars inside the run cut,
  the run's bars kept) — believed only when the run's last close is within 25% of the first close after it.
* THE CUT: the stored bars before the security's first run (its own runs and the genuine ones) are another security's
  when the assembled series BREAKS there: a gap of more than 30 calendar days, or the close moving more than x1.5 AND
  the 20-session median dollar volume more than x5, either way.
* JUNCTIONS: every boundary between segments (stored / runs) in date order holds its closes within 25% across at most
  30 days; a failing one drops everything before it.

The simulator's dip panels are this construction's (PREREG142's `panel142.pkl` / `panel142_a0.pkl`, the defaults of
`strategies.dip.build_panel` / `a0.dip_panel`); production's output equals the study's on every name it changed
(`python -m src.signals.dip_security --parity`). Two parts of the study are not live: the DUPLICATE drop (a joined run's
days removed from the earlier ticker's own series — that series stopped trading, so it never signals) and the daily
cache's foreign segments (history before 2021).

LIVE (`dip_long.compute_signals`): the events and the joined runs' bars are cached under `dip_long_dir`/security
(``events.json``, ``runs/``). A candidate never asked is asked at the day's computation and a day's SIGNAL whose events
are a day old or older is asked again before it is kept (both within `dip_long_security_budget_seconds`; one request
a name, no retry); after the computation the candidates' entries older than `dip_long_security_max_age_days` are
refreshed in the background (single-flight). FAIL-OPEN: a name whose construction raises is read as stored — as
before the fix — and journaled (a scorer-banner problem when it is the day's pick); a name Polygon cannot answer keeps
its last answer, else the ticker details' list date.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings

ET = "America/New_York"
WINDOW_FROM = pd.Timestamp("2021-01-04")    # the deep store's first session: a joined run starts no earlier
GAP_DAYS = 30                               # Amendment 1: a break = a gap of more than 30 calendar days, or
PX_BREAK = math.log(1.5)                    # the close moving more than x1.5 either way AND
DV_BREAK = math.log(5.0)                    # the 20-session median dollar volume more than x5 either way
MIN_RUN_SESSIONS = 20                       # Amendment 2: a genuine run has bars on >= 20 sessions of its span
SAME_SECURITY = 0.02                        # overlapping closes within 2% (median |log ratio|) = the same security
JUNCTION = math.log(1.25)                   # a boundary's closes within 25% (and no gap over 30 days: Amendment 3)
REQUEST_TIMEOUT = 10.0
RECHECK_DAYS = 1                            # a day's signal is asked again when its answer is this old
PRICE_COLS = ("Open", "High", "Low", "Close")


def enabled() -> bool:
    return bool(getattr(settings, "enable_dip_long_own_security", True))


def root() -> Path:
    return Path(settings.dip_long_dir) / "security"


def events_path() -> Path:
    return root() / "events.json"


def runs_dir() -> Path:
    return root() / "runs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def et_dates(idx) -> pd.DatetimeIndex:
    """The ET session date of each naive-UTC bar start."""
    return pd.DatetimeIndex(idx).tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()


def project(sym: str) -> str:
    """Polygon's class-share spelling in the project's form (BRK.B -> BRK-B), as the stores name files."""
    return str(sym).replace(".", "-")


def own(S: str, t: str) -> bool:
    return project(t).upper() == str(S).upper()


# ── the caches ───────────────────────────────────────────────────────────────

_LOCK = threading.RLock()
_EV: Dict[str, object] = {"key": None, "data": None}
_LD: Dict[str, object] = {"key": None, "data": None}
_RUNS: Dict[str, Tuple[tuple, Optional[pd.DataFrame]]] = {}
_SPLITS: Dict[str, object] = {"key": None, "data": None}
LAST: Dict[str, dict] = {}                  # name -> the latest construction's journal record (this process)


def load_events() -> Dict[str, dict]:
    """{name: Polygon's ticker events answer} — ``status`` (HTTP; 200 / 404 are answers), ``events``, ``at``."""
    p = events_path()
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        if _EV["key"] == key and _EV["data"] is not None:
            return _EV["data"]                                          # type: ignore[return-value]
        try:
            data = json.loads(p.read_text(encoding="utf-8")).get("names") or {}
        except (OSError, ValueError) as e:
            logger.warning(f"[dip_security] {p} unreadable ({type(e).__name__}) — the last version read is kept")
            return _EV["data"] or {}                                    # type: ignore[return-value]
        _EV.update(key=key, data=data)
        return data


def save_events(upd: Mapping[str, dict]) -> None:
    """Merge ``upd`` into the events cache (atomic replace). A failed answer never replaces a good one: it is kept as
    the entry's ``last_failure``."""
    if not upd:
        return
    with _LOCK:
        cur = dict(load_events())
        for n, rec in upd.items():
            old = cur.get(n)
            if rec.get("status") in (200, 404) or not old or old.get("status") not in (200, 404):
                cur[n] = dict(rec)
            else:
                cur[n] = dict(old, last_failure={"status": rec.get("status"), "at": rec.get("at")})
        p = events_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"names": cur, "saved_at": _now_iso()}, default=str), encoding="utf-8")
        os.replace(tmp, p)
        _EV.update(key=None, data=None)


def list_dates() -> Dict[str, str]:
    """{ticker: its list date} from the deep store's ticker details (`cache/ml/deep/ticker_details.parquet`)."""
    p = Path("cache/ml/deep/ticker_details.parquet")
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        if _LD["key"] == key and _LD["data"] is not None:
            return _LD["data"]                                          # type: ignore[return-value]
        try:
            import duckdb
            con = duckdb.connect()
            try:
                df = con.execute(f"select ticker, list_date from read_parquet('{p.as_posix()}')").fetchdf()
            finally:
                con.close()
            data = {project(t).upper(): str(d)[:10] for t, d in zip(df["ticker"], df["list_date"])
                    if d is not None and str(d)[:4].isdigit()}
        except Exception as e:                                          # noqa: BLE001
            logger.warning(f"[dip_security] ticker details unreadable ({type(e).__name__}: {e})")
            return _LD["data"] or {}                                    # type: ignore[return-value]
        _LD.update(key=key, data=data)
        return data


def split_table(asof: Optional[date] = None) -> Dict[str, List[Tuple[date, float, float]]]:
    """The security's splits as the price stores apply them: the records minus the false ones plus the confirmed
    missing ones (`split_check.effective_rows(..., with_missing=True)`, PREREG142's C2); with ``asof``, only the
    splits executed on or before it (the store's terms: a split announced for a later day is not in its bars yet)."""
    from src.data.intraday_store import raw_split_rows
    from src.data.split_check import effective_rows
    raw = raw_split_rows() or {}
    key = (id(raw), len(raw))
    with _LOCK:
        if _SPLITS["key"] != key or _SPLITS["data"] is None:
            _SPLITS.update(key=key, data=effective_rows(raw, with_missing=True))
        rows: Dict[str, List[Tuple[date, float, float]]] = _SPLITS["data"]      # type: ignore[assignment]
    if asof is None:
        return rows
    return {k: [x for x in v if x[0] <= asof] for k, v in rows.items()}


def run_path(t: str, a: str, b: str) -> Path:
    return runs_dir() / f"{project(t)}_{a}_{b}.m.pkl"


def run_bars(t: str, a: str, b: str) -> Optional[pd.DataFrame]:
    """A joined run's unadjusted 30-minute bars as fetched (None: never fetched, or fetched empty)."""
    p = run_path(t, a, b)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _RUNS.get(str(p))
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        f = pickle.load(open(p, "rb"))
    except Exception as e:                                              # noqa: BLE001
        logger.warning(f"[dip_security] {p.name} unreadable ({type(e).__name__})")
        return None
    f = f if isinstance(f, pd.DataFrame) and len(f) else None
    _RUNS[str(p)] = (key, f)
    return f


# ── Polygon ──────────────────────────────────────────────────────────────────

def _polygon(url: str, params: Optional[dict] = None) -> Tuple[Optional[int], Optional[dict]]:
    """(HTTP status, JSON) of ONE Polygon GET through the project's client — no retry; (None, None) on a transport
    failure. The key is never logged (only the exception's type)."""
    if not settings.polygon_api_key:
        return None, None
    from src.data import polygon_client as PC
    q = dict(params or {})
    q["apiKey"] = settings.polygon_api_key
    try:
        r = PC._http_get(url if url.startswith("http") else f"{PC._BASE}{url}", params=q, timeout=REQUEST_TIMEOUT)
    except Exception as e:                                              # noqa: BLE001
        logger.debug(f"[dip_security] a Polygon request failed ({type(e).__name__})")
        return None, None
    try:
        js = r.json()
    except Exception:                                                   # noqa: BLE001
        js = None
    return int(r.status_code), js


def fetch_events(ticker: str) -> dict:
    """Polygon's ticker events of ``ticker`` (`/vX/reference/tickers/{t}/events`): {status, events, name, figi, at}."""
    from src.data.polygon_client import to_polygon_symbol
    st, js = _polygon(f"/vX/reference/tickers/{to_polygon_symbol(ticker)}/events")
    res = (js or {}).get("results") or {} if st == 200 else {}
    return {"status": st, "events": res.get("events") or [], "name": res.get("name"),
            "figi": res.get("composite_figi"), "at": _now_iso()}


def fetch_run(ticker: str, frm: str, to: str, deadline: Optional[float] = None) -> Tuple[Optional[int], Optional[pd.DataFrame]]:
    """Polygon's UNADJUSTED regular-hours 30-minute bars of ``ticker`` over [frm, to] (pages followed; naive-UTC bar
    starts 09:30 .. 15:30 ET; Open High Low Close Volume — the deep store's frame contract). Status -1: a page lost."""
    from src.data.polygon_client import to_polygon_symbol
    st, js = _polygon(f"/v2/aggs/ticker/{to_polygon_symbol(ticker)}/range/30/minute/{frm}/{to}",
                      {"adjusted": "false", "sort": "asc", "limit": 50000})
    if js is None:
        return st, None
    rows = list(js.get("results") or [])
    nxt, pages = js.get("next_url"), 0
    while nxt and pages < 60:
        if deadline is not None and time.monotonic() > deadline:
            return -1, None
        s2, j2 = _polygon(nxt)
        if s2 != 200 or j2 is None:
            return -1, None
        rows.extend(j2.get("results") or [])
        nxt = j2.get("next_url")
        pages += 1
    if not rows:
        return st, None
    ts = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True)
    et = ts.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    keep = np.asarray((mins >= 570) & (mins <= 930))
    df = pd.DataFrame({"Open": [float(r["o"]) for r in rows], "High": [float(r["h"]) for r in rows],
                       "Low": [float(r["l"]) for r in rows], "Close": [float(r["c"]) for r in rows],
                       "Volume": [float(r.get("v") or 0.0) for r in rows]}, index=ts.tz_localize(None))
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return st, df


# ── the security's ticker runs and the plan (spans142.py) ───────────────────

def runs_of(events: Optional[Sequence[dict]]) -> List[dict]:
    """The ticker runs of one security from Polygon's events: ``ticker_change`` entries in date order, consecutive
    entries of one ticker merged -> [{"ticker", "start", "end"}] (end = the next run's start - 1 day; the last open)."""
    ev = []
    for e in events or []:
        if e.get("type") != "ticker_change":
            continue
        t = ((e.get("ticker_change") or {}).get("ticker") or "").strip()
        d = str(e.get("date") or "")[:10]
        if not t or len(d) != 10:
            continue
        ev.append((d, t.upper()))
    ev.sort()
    runs: List[dict] = []
    for d, t in ev:
        if runs and runs[-1]["ticker"] == t:
            continue
        runs.append({"ticker": t, "start": d, "end": None})
    for a, b in zip(runs, runs[1:]):
        a["end"] = (date.fromisoformat(b["start"]) - timedelta(days=1)).isoformat()
    return runs


def break_between(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Amendment 1's break test between the last bar of ``before`` and the first of ``after`` (daily o h l c v)."""
    out = {"bars_before": int(len(before)), "bars_after": int(len(after))}
    if not len(before) or not len(after):
        out.update({"break": False, "untestable": True})
        return out
    gap = int((after.index[0] - before.index[-1]).days)
    pr = float(after["c"].iloc[0] / before["c"].iloc[-1])
    dvb = float(np.nanmedian((before["c"] * before["v"]).to_numpy(float)[-20:]))
    dva = float(np.nanmedian((after["c"] * after["v"]).to_numpy(float)[:20]))
    dvr = float(dva / dvb) if dvb > 0 and dva > 0 else float("nan")
    brk = gap > GAP_DAYS or (abs(math.log(pr)) > PX_BREAK and math.isfinite(dvr) and abs(math.log(dvr)) > DV_BREAK)
    out.update({"break": bool(brk), "gap_days": gap, "close_ratio": round(pr, 4),
                "dv_ratio": round(dvr, 4) if math.isfinite(dvr) else None,
                "last_before": str(before.index[-1].date()), "first_after": str(after.index[0].date())})
    return out


def plan(S: str, ev: Optional[dict], ld: Optional[str], stored_d: pd.DataFrame, today: pd.Timestamp) -> dict:
    """The cut / join plan of name ``S`` over its stored daily series (spans142's `plan_name`, the deep store's
    source): its runs, how they were found, the first own run's break test (``cut``) and the runs to join."""
    runs = runs_of((ev or {}).get("events")) if ev and ev.get("status") == 200 else []
    how = "events"
    if not runs and ld:
        runs, how = [{"ticker": S, "start": ld, "end": None}], "list_date"
    out: dict = {"how": how if runs else "none", "runs": runs}
    if not runs:
        return out
    mine = [r for r in runs if own(S, r["ticker"])]
    if not mine:
        out["how"] = "no_own_run"
        return out
    s0 = pd.Timestamp(mine[0]["start"])
    out["first_own_start"] = mine[0]["start"]
    d = stored_d
    b = break_between(d[d.index < s0], d[d.index >= s0])
    out["cut"] = dict(b, at=str(s0.date()), applied=bool(b.get("break")))
    kept = d[d.index >= s0] if out["cut"]["applied"] else d
    joins = []
    for k, r in enumerate(runs):
        if own(S, r["ticker"]):
            continue
        a = pd.Timestamp(r["start"])
        if r.get("end"):
            e = pd.Timestamp(r["end"])
        else:                                       # an open run under another ticker: until the name resumes
            nxt = kept[kept.index > a]
            e = (nxt.index[0] - pd.Timedelta(days=1)) if len(nxt) else today
        a = max(a, WINDOW_FROM)
        if e < a:
            continue
        joins.append({"ticker": r["ticker"], "start": str(a.date()), "end": str(e.date()), "run": k})
    out["joins"] = joins
    return out


def factor_runs(S: str, p: dict, today: pd.Timestamp) -> List[dict]:
    """The security's runs with the open runs under another ticker closed where the name resumes (the joins' ends)
    and the name's resumed run after them — the split factor's spans (panel142's `factor_runs`)."""
    runs = [dict(r) for r in p["runs"]]
    ends = {j["run"]: j["end"] for j in p.get("joins") or []}
    for k, r in enumerate(runs):
        if r.get("end") is None and not own(S, r["ticker"]) and k in ends:
            r["end"] = ends[k]
            if pd.Timestamp(ends[k]) < today:
                runs.append({"ticker": S, "start": str((pd.Timestamp(ends[k]) + pd.Timedelta(days=1)).date()),
                             "end": None})
    return runs


def factor_after(days: pd.DatetimeIndex, runs: Sequence[dict], rows: Mapping[str, list]) -> np.ndarray:
    """For each session day: the product of split_from / split_to over the security's splits AFTER that day, each
    run's ticker's splits counted only inside its run. Prices x it, volume / it."""
    sp = []
    for r in runs:
        lo = date.fromisoformat(r["start"])
        hi = date.fromisoformat(r["end"]) if r.get("end") else date(2100, 1, 1)
        for d, a, b in (rows.get(r["ticker"].upper()) or rows.get(project(r["ticker"]).upper()) or []):
            if lo <= d <= hi and a > 0 and b > 0:
                sp.append((pd.Timestamp(d), a / b))
    f = np.ones(len(days))
    dd = pd.DatetimeIndex(days).normalize()
    for d, m in sp:
        f[np.asarray(dd < d)] *= m
    return f


def factored(f: Optional[pd.DataFrame], runs, rows) -> Optional[pd.DataFrame]:
    if f is None or not len(f):
        return f
    f = f.copy()
    fac = factor_after(et_dates(f.index), runs, rows)
    for c in PRICE_COLS:
        f[c] = f[c].to_numpy(float) * fac
    f["Volume"] = f["Volume"].to_numpy(float) / fac
    return f


def assemble(S: str, p: dict, stored_d: pd.DataFrame, fetched: Mapping[str, Optional[pd.DataFrame]]) -> dict:
    """The day-level decision for name ``S`` (panel142's `assemble`, Amendments 2-3): which stored days and which
    joined runs' days are kept. ``fetched``: run label -> the run's DAILY bars in today's terms (None = fetched empty).
    Returns {"keep_stored", "keep_runs": {label: DatetimeIndex}, "foreign", "log"}."""
    log: dict = {"runs": {}}
    drop_stored = pd.DatetimeIndex([])
    runs_kept: Dict[str, pd.DataFrame] = {}
    starts = [pd.Timestamp(x["start"]) for x in p["runs"] if own(S, x["ticker"])]
    raw_runs = {f"{j['ticker']}:{j['start']}": fetched.get(f"{j['ticker']}:{j['start']}") for j in p.get("joins") or []}

    def successor_close(lab: str, end: pd.Timestamp) -> Optional[float]:
        cands = []
        nxt = stored_d[stored_d.index > end]
        if len(nxt):
            cands.append((nxt.index[0], float(nxt["c"].iloc[0])))
        for l2, f2 in raw_runs.items():
            if l2 == lab or f2 is None or not len(f2):
                continue
            n2 = f2[f2.index > end]
            if len(n2):
                cands.append((n2.index[0], float(n2["c"].iloc[0])))
        return min(cands)[1] if cands else None
    for j in p.get("joins") or []:
        lab = f"{j['ticker']}:{j['start']}"
        f = fetched.get(lab)
        a, b = pd.Timestamp(j["start"]), pd.Timestamp(j["end"])
        if f is None or len(f) < MIN_RUN_SESSIONS:
            log["runs"][lab] = {"genuine": False, "sessions": 0 if f is None else int(len(f))}
            continue
        f = f[(f.index >= a) & (f.index <= b)]
        if len(f) < MIN_RUN_SESSIONS:
            log["runs"][lab] = {"genuine": False, "sessions": int(len(f))}
            continue
        starts.append(pd.Timestamp(j["start"]))
        ov = f.index.intersection(stored_d.index)
        rec = {"genuine": True, "sessions": int(len(f)), "overlap": int(len(ov))}
        if len(ov):
            dl = np.abs(np.log(f.loc[ov, "c"].to_numpy(float) / stored_d.loc[ov, "c"].to_numpy(float)))
            med = float(np.nanmedian(dl))
            rec["overlap_median_abs_log"] = round(med, 5)
            succ = successor_close(lab, b)
            last = float(f["c"].iloc[-1])
            rec["successor_ratio"] = round(succ / last, 4) if succ and last > 0 else None
            if med <= SAME_SECURITY:
                rec["verdict"] = "same security (stored kept)"
                f = f.drop(ov)
            elif succ is not None and last > 0 and abs(math.log(succ / last)) > JUNCTION:
                rec["verdict"] = "conflicting and not continuous with its successor: ignored (stored kept)"
                rec["genuine"] = False
                log["runs"][lab] = rec
                continue
            else:
                rec["verdict"] = "another security under the name (stored cut inside the run)"
                inside = stored_d.index[(stored_d.index >= a) & (stored_d.index <= b)]
                drop_stored = drop_stored.union(inside)
        runs_kept[lab] = f
        log["runs"][lab] = rec
    st = stored_d.drop(drop_stored)
    s_first = min(starts) if starts else None
    if s_first is not None:
        allf = pd.concat([st[["o", "h", "l", "c", "v"]]] + [f[["o", "h", "l", "c", "v"]] for f in runs_kept.values()])
        allf = allf[~allf.index.duplicated(keep="first")].sort_index()
        b = break_between(allf[allf.index < s_first], allf[allf.index >= s_first])
        log["first_run"] = dict(b, at=str(s_first.date()))
        if b["break"]:
            pre = st.index[st.index < s_first]
            drop_stored = drop_stored.union(pre)
            st = st.drop(pre)
            for lab in list(runs_kept):
                runs_kept[lab] = runs_kept[lab][runs_kept[lab].index >= s_first]
    lab_of = pd.concat([pd.Series("stored", index=st.index)] + [pd.Series(lab, index=f.index)
                                                              for lab, f in runs_kept.items()]).sort_index()
    lab_of = lab_of[~lab_of.index.duplicated(keep="first")]
    frames = {"stored": st, **runs_kept}
    segs, cur, s0, prev = [], None, None, None
    for t, lab in lab_of.items():
        if lab != cur:
            if cur is not None:
                segs.append((cur, s0, prev))
            cur, s0 = lab, t
        prev = t
    if cur is not None:
        segs.append((cur, s0, prev))
    js, first = [], 0
    for k in range(1, len(segs)):
        (la, _, ea), (lb, sb, _) = segs[k - 1], segs[k]
        ca, cb = float(frames[la].loc[ea, "c"]), float(frames[lb].loc[sb, "c"])
        rt = cb / ca if ca > 0 else float("nan")
        gap = int((sb - ea).days)
        ok = bool(math.isfinite(rt) and abs(math.log(rt)) <= JUNCTION and gap <= GAP_DAYS)
        js.append({"between": [la, lb], "at": str(sb.date()), "ratio": round(rt, 4), "gap_days": gap, "ok": ok})
        if not ok:
            first = k
    log["junctions"] = js
    cut_from = segs[first][1] if segs else None
    if first and cut_from is not None:
        log["dropped_before_failed_junction"] = str(cut_from.date())
        pre = st.index[st.index < cut_from]
        drop_stored = drop_stored.union(pre)
        st = st.drop(pre)
        runs_kept = {lab: f[f.index >= cut_from] for lab, f in runs_kept.items()}
    log["stored_kept"], log["stored_cut"] = int(len(st)), int(len(drop_stored))
    log["joined"] = {lab: int(len(f)) for lab, f in runs_kept.items() if len(f)}
    return {"keep_stored": st.index, "keep_runs": {lab: f.index for lab, f in runs_kept.items() if len(f)},
            "foreign": drop_stored.sort_values(), "log": log}


# ── the series ───────────────────────────────────────────────────────────────

def _today() -> pd.Timestamp:
    """Today's ET date (a naive midnight): an open run's end, the splits the store's bars carry."""
    return pd.Timestamp.now(tz=ET).tz_localize(None).normalize()


def own_bars(S: str, b30: Optional[pd.DataFrame], through: Optional[date] = None,
             today: Optional[pd.Timestamp] = None, asof_splits: bool = True) -> Tuple[Optional[pd.DataFrame], dict]:
    """Name ``S``'s regular-hours 30-minute bars (the deep store's, ``b30``) rebuilt from its own security — (the bars,
    the journal record: ``changed``, ``how``, ``cut_from``, ``joined``, ``stored_cut``, ``error``). With ``through``
    only the bars of sessions on or before it are read (the dip rule's D-1: nothing later can move the answer). A name
    the construction does not change comes back as given. Never raises: on a failure the bars as stored, ``error``."""
    rec: dict = {"name": S, "changed": False}
    if b30 is None or not len(b30):
        rec["how"] = "no_bars"
        return b30, rec
    try:
        if through is not None:
            b30 = b30[np.asarray(et_dates(b30.index) <= pd.Timestamp(through))]
            if not len(b30):
                rec["how"] = "no_bars"
                return b30, rec
        today = pd.Timestamp(today) if today is not None else _today()
        from src.signals.dip_long import daily_bars
        stored_d = daily_bars(b30)
        if stored_d is None or not len(stored_d):
            rec["how"] = "no_bars"
            return b30, rec
        ev = load_events().get(S)
        p = plan(S, ev, list_dates().get(S.upper()), stored_d, today)
        rec["how"] = p["how"]
        if ev is not None:
            rec["events_at"] = ev.get("at")
        if "cut" not in p or not (p["cut"].get("bars_before") or p.get("joins")):
            return b30, rec
        runs = factor_runs(S, p, today)
        rows = split_table(today.date() if asof_splits else None)
        m30: Dict[str, Optional[pd.DataFrame]] = {}
        dly: Dict[str, Optional[pd.DataFrame]] = {}
        for j in p["joins"]:
            lab = f"{j['ticker']}:{j['start']}"
            f = factored(run_bars(j["ticker"], j["start"], j["end"]), runs, rows)
            m30[lab] = f
            dly[lab] = daily_bars(f) if f is not None and len(f) else None
        dec = assemble(S, p, stored_d, dly)
        lg = dec["log"]
        rec.update(stored_cut=lg["stored_cut"], joined=lg["joined"],
                   missing_runs=[lab for lab, f in m30.items() if f is None])
        if not lg["stored_cut"] and not lg["joined"]:
            return b30, rec
        parts = [b30[np.isin(et_dates(b30.index).values, dec["keep_stored"].values)]]
        for lab, days in dec["keep_runs"].items():
            f = m30[lab]
            parts.append(f[np.isin(et_dates(f.index).values, days.values)])
        out = pd.concat(parts).sort_index()
        out = out[~out.index.duplicated(keep="first")]
        first = et_dates(out.index[:1])
        rec.update(changed=True, first_session=str(first[0].date()) if len(first) else None,
                   cut_from=(lg.get("first_run") or {}).get("at") if (lg.get("first_run") or {}).get("break")
                   else lg.get("dropped_before_failed_junction"),
                   junctions=lg.get("junctions"))
        return out, rec
    except Exception as e:                                              # noqa: BLE001
        rec.update(changed=False, error=f"{type(e).__name__}: {e}")
        logger.error(f"[dip_security] {S}: the own-security construction failed ({rec['error']}) — read as stored")
        return b30, rec


def series(S: str, through: Optional[date] = None) -> Tuple[Optional[pd.DataFrame], dict]:
    """``own_bars`` of the deep store's bars of ``S``; the record is kept in `LAST` (this process's latest)."""
    from src.data.intraday_store import load_deep_30m
    b, rec = own_bars(S, load_deep_30m(S), through)
    with _LOCK:
        LAST[S] = rec
    return b, rec


# ── keeping the caches current ──────────────────────────────────────────────

def _age_days(rec: Optional[dict], now: Optional[datetime] = None) -> float:
    try:
        t = datetime.fromisoformat(str((rec or {}).get("at")))
    except (TypeError, ValueError):
        return float("inf")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return ((now or _now()) - t).total_seconds() / 86400.0


def needs_events(names: Iterable[str], max_age_days: Optional[float] = None) -> List[str]:
    """The names never answered (or answered only by failures), and with ``max_age_days`` those answered longer ago."""
    ev = load_events()
    out = []
    for n in names:
        r = ev.get(n)
        if r is None or r.get("status") not in (200, 404):
            out.append(n)
        elif max_age_days is not None and _age_days(r) > float(max_age_days):
            out.append(n)
    return out


def ensure(names: Sequence[str], budget_s: float, through: Optional[date] = None) -> dict:
    """Ask Polygon for ``names``' events (in the order given, one request each, until ``budget_s`` is spent), save
    the answers, then fetch every joined run those names' plans need and the cache lacks (within what is left)."""
    t0 = time.monotonic()
    deadline = t0 + max(0.0, float(budget_s))
    got, failed, skipped = {}, [], []
    for n in names:
        if time.monotonic() > deadline:
            skipped.append(n)
            continue
        r = fetch_events(n)
        got[n] = r
        if r.get("status") not in (200, 404):
            failed.append(n)
    save_events(got)
    runs = run_needs([n for n in names if n not in skipped], through)
    fetched_runs, failed_runs = [], []
    idx = _run_index()
    for t, a, b in runs:
        if time.monotonic() > deadline:
            break
        st, df = fetch_run(t, a, b, deadline)
        key = run_path(t, a, b).name
        if df is not None and len(df):
            p = run_path(t, a, b)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            pickle.dump(df, open(tmp, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)
            fetched_runs.append(key)
            idx[key] = {"status": st, "bars": int(len(df)), "at": _now_iso()}
        else:
            failed_runs.append(key)
            idx[key] = {"status": st, "bars": 0, "at": _now_iso()}
    if fetched_runs or failed_runs:
        _save_run_index(idx)
    return {"asked": len(got), "failed": failed, "skipped": skipped, "runs_fetched": fetched_runs,
            "runs_failed": failed_runs, "seconds": round(time.monotonic() - t0, 1)}


def _run_index() -> Dict[str, dict]:
    p = runs_dir() / "index.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_run_index(idx: Mapping[str, dict]) -> None:
    p = runs_dir() / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(idx, indent=0, default=str), encoding="utf-8")
    os.replace(tmp, p)


def run_needs(names: Iterable[str], through: Optional[date] = None) -> List[Tuple[str, str, str]]:
    """The joined runs ``names``' plans need that the cache lacks — never fetched, or fetched empty longer ago than
    `dip_long_security_max_age_days` (a run Polygon answered empty is not asked every day)."""
    from src.data.intraday_store import load_deep_30m
    from src.signals.dip_long import daily_bars
    ev, lds, idx = load_events(), list_dates(), _run_index()
    today = _today()
    out: List[Tuple[str, str, str]] = []
    for S in names:
        e = ev.get(S)
        if not e or e.get("status") != 200 or len(runs_of(e.get("events"))) < 2:
            continue
        b30 = load_deep_30m(S)
        if b30 is None or not len(b30):
            continue
        if through is not None:
            b30 = b30[np.asarray(et_dates(b30.index) <= pd.Timestamp(through))]
        d = daily_bars(b30)
        if d is None or not len(d):
            continue
        p = plan(S, e, lds.get(S.upper()), d, today)
        for j in p.get("joins") or []:
            key = (j["ticker"], j["start"], j["end"])
            if run_path(*key).exists():
                continue
            rec = idx.get(run_path(*key).name)
            if rec is not None and _age_days(rec) <= float(settings.dip_long_security_max_age_days):
                continue
            out.append(key)
    return out


_BG: Dict[str, object] = {"thread": None}


def refresh_async(names: Sequence[str]) -> bool:
    """Refresh, in a daemon thread, the events of ``names`` answered longer ago than
    `dip_long_security_max_age_days` (oldest first; single-flight; four times the day's budget). False when one
    still runs or nothing is due."""
    if float(settings.dip_long_security_budget_seconds) <= 0:
        return False                                                    # no asking at all (the tests' baseline)
    due = needs_events(names, float(settings.dip_long_security_max_age_days))
    if not due:
        return False
    ev = load_events()
    due.sort(key=lambda n: -_age_days(ev.get(n)))
    with _LOCK:
        th = _BG.get("thread")
        if th is not None and th.is_alive():                            # type: ignore[union-attr]
            return False

        def work():
            try:
                r = ensure(due, 4.0 * float(settings.dip_long_security_budget_seconds))
                logger.info(f"[dip_security] background refresh: {r['asked']} asked, {len(r['failed'])} failed, "
                            f"{len(r['skipped'])} left for later, {len(r['runs_fetched'])} run(s) fetched "
                            f"({r['seconds']}s)")
            except Exception as e:                                      # noqa: BLE001
                logger.warning(f"[dip_security] background refresh failed ({type(e).__name__}: {e})")
        th = threading.Thread(target=work, name="dip-security-refresh", daemon=True)
        _BG["thread"] = th
        th.start()
    return True


# ── the parity with PREREG142's panel, and the CLI ──────────────────────────

def parity(study_dir: Optional[str] = None, names: Optional[Sequence[str]] = None) -> dict:
    """Production's construction against PREREG142's corrected AB panel, name by name: every name the study changed
    or planned (its ``assemble142.json``), the deep store's bars cut at the panel's last session, the study's events
    (the cache's) and runs; ``identical`` = the same daily bars (o h l c v, within 1e-9 relative)."""
    from src.data.intraday_store import load_deep_30m
    from src.signals.dip_long import daily_bars
    sd = Path(study_dir or Path(settings.backtest_research_dir) / "studies" / "prereg142")
    rep = json.loads((sd / "assemble142.json").read_text(encoding="utf-8"))
    z = pickle.load(open(sd / "panel142.pkl", "rb"))
    panel, bars = z["panel"], z["bars"]
    dup = {d["series"] for d in rep.get("ab_duplicates") or []}
    todo = sorted(names or rep["ab"].keys())
    last = max(v.index[-1] for v in panel.values() if len(v))
    built = pd.Timestamp(str(rep.get("built_at", ""))[:10] or str(last.date()))
    out: dict = {"names": len(todo), "panel_last": str(last.date()), "identical": 0, "differ": {}, "dedupe_only": []}
    for S in todo:
        b30 = load_deep_30m(S)
        if b30 is None or not len(b30):
            out["differ"][S] = "no stored bars"
            continue
        mine, rec = own_bars(S, b30, through=last.date(), today=built, asof_splits=False)
        want = panel.get(S)
        got = daily_bars(mine)
        if want is not None:
            want = want[want.index <= last]
        same = (want is not None and got is not None and len(want) == len(got)
                and bool((want.index == got.index).all())
                and all(np.allclose(want[c].to_numpy(float), got[c].to_numpy(float), rtol=1e-9, atol=0.0,
                                    equal_nan=True) for c in ("o", "h", "l", "c", "v")))
        if same:
            out["identical"] += 1
        elif S in dup and not rec.get("changed"):
            out["dedupe_only"].append(S)
        else:
            out["differ"][S] = {"rec": rec, "want": None if want is None else [str(want.index[0].date()), len(want)],
                                "got": None if got is None else [str(got.index[0].date()), len(got)],
                                "study_file": S in bars}
    out["n_differ"] = len(out["differ"])
    return out


def seed(study_dir: Optional[str] = None) -> dict:
    """Seed the caches from PREREG142's fetches (2026-10-17: the same Polygon answers): its events (``events142.json``)
    and its joined runs' 30-minute bars (``bars142/*.m.pkl``). Entries the cache already holds are kept."""
    import shutil
    sd = Path(study_dir or Path(settings.backtest_research_dir) / "studies" / "prereg142")
    ev = json.loads((sd / "events142.json").read_text(encoding="utf-8"))
    have = load_events()
    add = {n: dict(r, at=r.get("at")) for n, r in (ev.get("names") or {}).items() if n not in have}
    save_events(add)
    runs_dir().mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in (sd / "bars142").glob("*.m.pkl"):
        q = runs_dir() / p.name
        if not q.exists():
            shutil.copy2(p, q)
            copied += 1
    return {"events_added": len(add), "runs_copied": copied}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.signals.dip_security", description=__doc__.split("\n")[0])
    ap.add_argument("--seed", action="store_true", help="seed the caches from PREREG142's fetches")
    ap.add_argument("--parity", action="store_true", help="production's construction against PREREG142's panel")
    ap.add_argument("--name", help="one name's construction now (the deep store's bars, the cached events)")
    ap.add_argument("--refresh", help="comma list of names to ask Polygon now (events + their joined runs)")
    a = ap.parse_args(argv)
    if a.seed:
        print(json.dumps(seed(), indent=1))
    if a.refresh:
        names = [x.strip().upper() for x in a.refresh.split(",") if x.strip()]
        print(json.dumps(ensure(names, 120.0), indent=1, default=str))
    if a.name:
        b, rec = series(a.name.upper())
        print(json.dumps(rec, indent=1, default=str))
    if a.parity:
        r = parity()
        print(json.dumps({k: v for k, v in r.items() if k != "differ"}, indent=1, default=str))
        for S, v in list(r["differ"].items())[:40]:
            print(S, json.dumps(v, default=str)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
