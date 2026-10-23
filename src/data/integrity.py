"""THE PRICE DATA'S INTEGRITY CHECK — automated verification of the split records and of the price stores against each
other (user directive 2026-10-16: "implement automated verification or testing and guardrails to prevent error in the
data on the future (i.e. wrong or missing splits)"; the audit of that day found Polygon's false split records adjusted
into our stores (XRX), additive caches left on the pre-split scale (`cache/ohlcv`, the deep `bars30m_full`), and IBKR's
overnight bars in another price regime (WDC)).

    python -m src.data.integrity [--day D] [--full] [--no-repair]     -> cache/ml/integrity/<day>.json

`run` (the nightly deep refresh's last step; `verify_today` in the pre-open run):
  1. SPLIT RECORDS — `split_check.verify` on every record executed in the last `RECENT_DAYS` days or still pending,
     plus up to ``budget`` older unchecked ones (a rolling backfill of the stores' span): verified / false / suspect /
     ambiguous ...;
  2. MISSING SPLITS — `split_check.find_missing` over production's 30-minute store (the last ``lookback`` sessions):
     a jump of a standard split ratio no record explains, confirmed by yfinance -> added;
  3. REPAIRS — every store that does not carry a correction yet is refetched with it: production's store
     (`intraday_store.reset_corrected`, for EVERY name whose stored corrections differ from the table's, whoever
     changed it), the daily cache (`market_data._refetch_daily_whole`), the 04:00-20:00 store
     (`refresh.reset_full_part`);
  4. CROSS-STORE CONSISTENCY — for every name with an effective split in the last `RECENT_DAYS` days, its last sessions'
     closes in production's store against the daily cache and the 04:00-20:00 store: a level off by more than
     `LEVEL_TOL` is refetched whole (repair) and reported;
  5. BAD BARS — zero / negative prices, a high under its low, duplicate bars in production's store (the names given,
     the last ``lookback`` sessions).
`health(trades)` turns the latest report and the split table into the email's scorer-banner problems (a held name's
contradicted or suspect split, a confirmed-missing split, an unrepaired store, bad bars in a held name, no integrity run
for two days) and notes. Tests: tests/test_integrity.py.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

OUT_DIR = Path("cache/ml/integrity")
RECENT_DAYS = 45
LEVEL_TOL = 0.03          # a split moves a level by 25% or more; under 3% is the closing print against a bar's last trade
ET = "America/New_York"
STORE_FROM = date(2021, 1, 4)


def _universe() -> List[str]:
    from src.data import deep
    return list(deep.deep_universe())


def _sessions(df: pd.DataFrame) -> pd.Series:
    et = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
    return pd.to_numeric(df["Close"], errors="coerce").groupby(et).last()


def cross_store(tk: str, n_sessions: int = 10) -> Dict[str, Optional[float]]:
    """The median close ratio of the daily cache and of the 04:00-20:00 store to production's store over the name's
    last ``n_sessions`` (None where a store holds nothing to compare)."""
    from src.data.intraday_store import load_deep_30m
    out: Dict[str, Optional[float]] = {"daily": None, "full": None}
    d = load_deep_30m(tk)
    if d is None or d.empty:
        return out
    prod = _sessions(d).tail(n_sessions)
    prod = prod[prod > 0]
    if prod.empty:
        return out
    try:
        from src.data.cache import load_ohlcv
        c = load_ohlcv(tk)
        if c is not None and not c.empty:
            dc = pd.Series(pd.to_numeric(c["Close"], errors="coerce").to_numpy(float),
                           index=pd.DatetimeIndex(c.index).tz_localize(None).normalize()
                           if pd.DatetimeIndex(c.index).tz is None else
                           pd.DatetimeIndex(c.index).tz_convert(None).normalize())
            j = pd.concat([prod.rename("p"), dc.rename("d")], axis=1, join="inner").dropna()
            j = j[(j.p > 0) & (j.d > 0)]
            if len(j):
                out["daily"] = float(np.median(j.d / j.p))
    except Exception as e:                                       # noqa: BLE001
        logger.debug(f"[integrity] {tk}: daily cache unreadable ({e})")
    try:
        from src.data import deep
        p = deep.family_dir("bars30m_full") / "parts" / f"{tk}.parquet"
        if p.exists():
            f = deep.read_parquet(p)
            f = f[f["session"] == "rth"] if "session" in f.columns else f
            fs = pd.Series(pd.to_numeric(f["close"], errors="coerce").to_numpy(float),
                           index=pd.DatetimeIndex(pd.to_datetime(f["ts"])))
            fs = fs.groupby(fs.index.tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()).last()
            j = pd.concat([prod.rename("p"), fs.rename("f")], axis=1, join="inner").dropna()
            j = j[(j.p > 0) & (j.f > 0)]
            if len(j):
                out["full"] = float(np.median(j.f / j.p))
    except Exception as e:                                       # noqa: BLE001
        logger.debug(f"[integrity] {tk}: 04:00-20:00 store unreadable ({e})")
    return out


def bad_bars(tk: str, lookback: int = 10) -> Dict[str, int]:
    """Counts of zero / negative prices, highs under their lows and duplicate bars in production's store (the last
    ``lookback`` sessions)."""
    from src.data.intraday_store import load_deep_30m
    d = load_deep_30m(tk)
    if d is None or d.empty:
        return {}
    et = pd.DatetimeIndex(d.index).tz_localize("UTC").tz_convert(ET).tz_localize(None).normalize()
    days = sorted(set(et))[-lookback:]
    sub = d[et.isin(days)]
    px = sub[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    out = {"nonpositive": int((~(px > 0).all(axis=1)).sum()), "high_under_low": int((px["High"] < px["Low"]).sum()),
           "duplicates": int(pd.DatetimeIndex(sub.index).duplicated().sum())}
    return {k: v for k, v in out.items() if v}


def repair(tickers: Iterable[str], why: str) -> Dict[str, Dict[str, str]]:
    """Refetch ``tickers`` whole in every store that holds them (each with the split table's corrections)."""
    from src.data import deep
    from src.data.cache import load_ohlcv
    from src.data.deep import refresh as R
    from src.data.intraday_store import reset_corrected, reset_deep_30m, _path
    from src.data.market_data import _refetch_daily_whole
    out: Dict[str, Dict[str, str]] = {}
    for tk in sorted({t.upper() for t in tickers}):
        r: Dict[str, str] = {}
        try:
            if _path(tk).exists():
                rc = reset_corrected([tk])
                r["prod"] = rc.get(tk) or reset_deep_30m(tk, why=why)
        except Exception as e:                                   # noqa: BLE001
            r["prod"] = f"failed ({e})"
        try:
            c = load_ohlcv(tk)
            if c is not None and not c.empty:
                r["daily"] = "reset" if _refetch_daily_whole(tk, c, why) is not None else "empty"
        except Exception as e:                                   # noqa: BLE001
            r["daily"] = f"failed ({e})"
        try:
            if (deep.family_dir("bars30m_full") / "parts" / f"{tk}.parquet").exists():
                r["full"] = R.reset_full_part(tk, why)
        except Exception as e:                                   # noqa: BLE001
            r["full"] = f"failed ({e})"
        out[tk] = r
    return out


def run(today: Optional[date] = None, *, names: Optional[Sequence[str]] = None, full: bool = False,
        budget: int = 400, lookback: int = 10, do_repair: bool = True) -> dict:
    """The integrity check (module docstring); writes ``cache/ml/integrity/<today>.json`` and returns it."""
    from src.data import split_check as SC
    from src.data.intraday_store import raw_split_rows, split_rows
    today = today or date.today()
    names = list(names) if names is not None else _universe()
    rep: dict = {"day": today.isoformat(), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "names": len(names)}
    raw = raw_split_rows() or {}
    before = {k: v.get("status") for k, v in SC.load_checks().items()}
    # 1. split records: the recent and pending ones, then a rolling backfill of the stores' span
    recent, older = {}, {}
    try:
        recent = SC.verify_records(raw, since=today - timedelta(days=RECENT_DAYS), today=today,
                                   recheck=("pending", "unverifiable", "suspect"))
        older = SC.verify_records(raw, since=STORE_FROM, today=today, budget=(100_000 if full else budget))
    except Exception as e:                                       # noqa: BLE001
        rep["verify_error"] = str(e)
        logger.warning(f"[integrity] split-record checks failed ({e}) — the rest of the run goes on")
    rep["verified"] = {"recent": _tally(recent), "backfill": _tally(older)}
    # 2. missing splits in production's store
    miss = SC.find_missing(names, split_rows() or {}, lookback=(400 if full else lookback))
    rep["missing"] = {k: v["status"] for k, v in miss.items()}
    # 3. repairs: every new correction into every store
    after = SC.load_checks()
    corr_st = ("false", "missing")
    new_corr = sorted(k for k, v in after.items()                                   # added, changed or withdrawn
                      if (v.get("status") in corr_st or before.get(k) in corr_st) and before.get(k) != v.get("status"))
    rep["new_corrections"] = new_corr
    if do_repair and new_corr:
        rep["repairs"] = repair({k.split("|", 1)[0] for k in new_corr}, "split correction")
    if do_repair:
        # every name whose stored history carries other corrections than the table's — whoever changed the table
        # (the pre-open run, the scheduler's guard, a hand edit): production's store against `_split_corrections_applied`
        try:
            from src.data.intraday_store import reset_corrected
            rep["store_resets"] = reset_corrected()
        except Exception as e:                                   # noqa: BLE001
            rep["store_resets"] = {"error": str(e)}
    # 4. cross-store consistency for the names with an effective split lately
    eff = split_rows() or {}
    recent_split = sorted(tk for tk, lst in eff.items()
                          if any(today - timedelta(days=RECENT_DAYS) <= d <= today for d, _, _ in lst))
    off: Dict[str, dict] = {}
    for tk in recent_split:
        try:
            cs = cross_store(tk)
        except Exception as e:                                   # noqa: BLE001
            logger.debug(f"[integrity] {tk}: cross-store failed ({e})")
            continue
        bad = {s: r for s, r in cs.items() if r is not None and abs(r - 1.0) > LEVEL_TOL}
        if bad:
            off[tk] = bad
    rep["cross_store_off"] = off
    if do_repair and off:
        rep["cross_store_repairs"] = repair(off, "store level off production's after a split")
        still = {}
        for tk in off:
            cs = cross_store(tk)
            bad = {s: r for s, r in cs.items() if r is not None and abs(r - 1.0) > LEVEL_TOL}
            if bad:
                still[tk] = bad
        rep["cross_store_unrepaired"] = still
    # 5. bad bars
    bb = {}
    for tk in names:
        try:
            b = bad_bars(tk, lookback)
        except Exception:                                        # noqa: BLE001
            continue
        if b:
            bb[tk] = b
    rep["bad_bars"] = bb
    rep["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{today.isoformat()}.json"
    tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    logger.info(f"[integrity] {today}: verified {rep['verified']}, missing {len(rep['missing'])}, new corrections "
                f"{len(new_corr)}, cross-store off {len(off)}, bad bars {len(bb)} -> {p}")
    return rep


def verify_today(today: Optional[date] = None, names: Optional[Iterable[str]] = None) -> dict:
    """The pre-open run's part: the records executing today or in the last days, checked on the pre-market's raw
    prices, and their corrections into production's store before the snapshot is built."""
    from src.data import split_check as SC
    from src.data.intraday_store import raw_split_rows, reset_corrected
    today = today or date.today()
    raw = raw_split_rows() or {}
    want = {n.upper() for n in names} if names is not None else None
    sub = {tk: lst for tk, lst in raw.items() if (want is None or tk in want)
           and any(today - timedelta(days=5) <= d <= today for d, _, _ in lst)}
    up = SC.verify_records(sub, since=today - timedelta(days=5), today=today,
                           recheck=("pending", "unverifiable", "suspect", "ambiguous"))
    # every stored name whose corrections differ from the table's — a record this check decided or one decided since
    # the last run elsewhere (the scheduler's guard, a hand edit) — before the snapshot is built
    resets = reset_corrected()
    return {"checked": {k: v.get("status") for k, v in up.items()}, "resets": resets}


def _tally(d: Dict[str, dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in d.values():
        out[v.get("status", "?")] = out.get(v.get("status", "?"), 0) + 1
    return out


def health(trades: Optional[Sequence[dict]] = None, today: Optional[date] = None) -> dict:
    """The email banner's view: ``problems`` (red) and ``notes`` (green line)."""
    from src.data.split_check import load_checks
    today = today or date.today()
    problems: List[str] = []
    notes: List[str] = []
    held: Dict[str, date] = {}                     # each held name's earliest open entry (its split terms start there)
    for t in (trades or []):
        if t.get("status") != "OPEN" or not t.get("ticker"):
            continue
        tk = str(t.get("ticker")).upper()
        try:
            ed = date.fromisoformat(str(t.get("entry_date") or t.get("entry_datetime") or "")[:10])
        except ValueError:
            ed = today - timedelta(days=RECENT_DAYS)
        held[tk] = min(held.get(tk, ed), ed)
    checks = load_checks()
    for k, v in checks.items():
        tk, ds = k.split("|", 1)
        try:
            d = date.fromisoformat(ds)
        except ValueError:
            continue
        st = v.get("status")
        # a record dated before the position's entry never touched it (its terms start after it)
        if tk in held and d > held[tk] and st in ("false", "suspect"):
            problems.append(f"DATA: {tk}'s split record of {ds} — the traded prices contradict it ({st}"
                            f"{'; left out, the trade restored' if st == 'false' else '; still applied: check it'})")
        elif st == "candidate" and (today - d).days <= RECENT_DAYS and (tk in held):
            problems.append(f"DATA: {tk} jumped by a split ratio on {ds} with no split record (unconfirmed)")
        elif st in ("false", "missing") and (today - d).days <= 7:
            notes.append(f"split table: {tk} {ds} {st} (corrected in every store)")
    reps = sorted(OUT_DIR.glob("*.json")) if OUT_DIR.exists() else []
    if not reps:
        notes.append("price-data integrity: no run yet")
    else:
        last = reps[-1]
        try:
            rep = json.loads(last.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            rep = {}
        age = (today - date.fromisoformat(last.stem)).days if last.stem[:4].isdigit() else 99
        if age > 2:
            problems.append(f"DATA: the price-data integrity check has not run since {last.stem}")
        for tk, b in (rep.get("cross_store_unrepaired") or {}).items():
            problems.append(f"DATA: {tk}'s stores disagree after a split and the repair did not fix it ({b})")
        for tk, b in (rep.get("bad_bars") or {}).items():
            if tk in held:
                problems.append(f"DATA: bad bars in held {tk}'s 30-minute store ({b})")
    return {"problems": problems, "notes": notes}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the price data's integrity check")
    ap.add_argument("--day", help="YYYY-MM-DD (default today)")
    ap.add_argument("--full", action="store_true", help="every unchecked record and 400 sessions of jumps")
    ap.add_argument("--no-repair", action="store_true")
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else date.today()
    rep = run(d, full=a.full, do_repair=not a.no_repair)
    print(json.dumps({k: rep[k] for k in ("verified", "missing", "new_corrections", "cross_store_off")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
