"""The borrow fee IBKR charges a short, day by day (user 2026-10-06: "We should use IBKR data as much as we can ...
we want the real exact borrow fees we would have in live trading a real account and we want that data to be as
complete as possible").

IBKR's rule (ibkrguides "Borrow Fee Details"; the audited replay engine `cap5k4` charges the same): every CALENDAR
day from the short sale's settlement to the cover's settlement, fee = shares x roundup(1.02 x the prior session's
close, to the whole dollar) x THAT DAY's annual rate / 360 — the rate is not fixed at entry, an open short pays the
day's rate. Each day's figure comes from IBKR, in this order:

  1. ``ibkr_charged`` — what IBKR actually charged our account for the name that day (the Flex statement's Borrow Fees
     Details, ``broker_borrow_fees``; per share = fee / quantity, the same for every share of the name that day);
  2. IBKR's own rate for the day with the formula: the last copy of IBKR's short-stock file stamped that day (our
     archive: ``own_archive``), else IBKR's API FEE_RATE bar (``ibkr_api``), else the community archive of IBKR's file
     (``community_archive``) — `borrow_history.fee_daily`; a day none of them holds (a weekend, a name IBKR stopped
     listing) carries the last rate known before it (``<source>_carried``) — at most a week, or from around the
     trade's own entry; a day not over yet takes the file in force now (``file_now``, provisional until the day ends);
  3. the rate IBKR quoted at entry (``entry_stamp``), then ``short_borrow_annual_pct`` (``default``).

Trade dates follow IBKR: an execution in the overnight session (20:00-04:00 ET) belongs to the next session.
Settlement is T+1 (T+2 before 2024-05-28) on days both the NYSE and the settlement system are open — Columbus Day and
Veterans Day are trading days with no settlement.

Each short's schedule is stored on its trade (``borrow_days``: [day, source, rate %/yr, collateral price per share,
fee per share]; ``borrow_fee_ps``, ``borrow_fee_usd``, ``borrow_rate_now``, ``borrow_final``) and its total is the
return-reducing fraction the ledger, the daily NAV and the simulated account charge. A CLOSED trade that has no stored
schedule (closed before the schedule existed) keeps its old carry until ``borrow_backfill_closed`` — the backfill
waits for the formula to be checked against IBKR's own charges (``--verify``).

    python -m src.performance.borrow_fees --verify            # the formula vs IBKR's charges (Flex)
    python -m src.performance.borrow_fees --trade <trade_id>  # one short's day-by-day schedule
"""
from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config.settings import settings

ET = ZoneInfo("America/New_York")
T1_FROM = date(2024, 5, 28)               # US equities moved from T+2 to T+1
_RATE_TTL = 900.0                         # a name's merged daily-rate history is re-read after 15 minutes
MAX_CARRY_DAYS = 7                        # a rate carries over a weekend, a holiday, a few unlisted days — no further,
                                          # unless it was seen around the trade's own entry
_CHARGED_TTL = 600.0                      # IBKR's charges are re-read after 10 minutes (the EOD fetch adds rows)

_rates_cache: Dict[str, Tuple[float, List[Tuple[date, float, str]]]] = {}
_closes_cache: Dict[str, Tuple[float, List[Tuple[date, float]]]] = {}
_charged_cache: Dict[str, object] = {"at": 0.0, "map": {}, "ver": ""}


def reset() -> None:
    """Drop the in-process caches (tests; after a fetch of IBKR's charges)."""
    _rates_cache.clear()
    _closes_cache.clear()
    _charged_cache.update(at=0.0, map={}, ver="")


# ── calendar ──────────────────────────────────────────────────────────────────

def _settlement_only_holidays(year: int) -> set:
    """Federal Reserve holidays on which the NYSE trades but nothing settles: Columbus Day (the second Monday of
    October) and Veterans Day (November 11; on a Sunday the Fed observes Monday, on a Saturday nothing)."""
    out = set()
    oct1 = date(year, 10, 1)
    out.add(oct1 + timedelta(days=(7 - oct1.weekday()) % 7 + 7))
    vet = date(year, 11, 11)
    if vet.weekday() == 6:
        out.add(vet + timedelta(days=1))
    elif vet.weekday() < 5:
        out.add(vet)
    return out


def is_session(d: date) -> bool:
    from src.performance.market_calendar import is_market_day
    return is_market_day(d)


def is_settlement_day(d: date) -> bool:
    return is_session(d) and d not in _settlement_only_holidays(d.year)


def _as_date(x) -> date:
    """A plain date from a date, a datetime / pandas Timestamp (a DuckDB DATE comes back as one) or an ISO string."""
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _as_utc(ts) -> Optional[datetime]:
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        t = ts
    else:
        try:
            t = datetime.fromisoformat(str(ts).replace(" ", "T").replace("Z", "+00:00"))
        except ValueError:
            return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def trade_date(ts) -> Optional[date]:
    """IBKR's trade date of an execution: the ET date in the day and extended sessions (04:00-20:00), the NEXT
    session for the overnight session (from 20:00), the next session for an instant on a closed day."""
    t = _as_utc(ts)
    if t is None:
        return None
    e = t.astimezone(ET)
    d = e.date()
    if e.hour >= 20:
        d += timedelta(days=1)
    while not is_session(d):
        d += timedelta(days=1)
    return d


def settlement_date(td: date) -> date:
    """T+1 settlement days after the trade date (T+2 before 2024-05-28)."""
    n = 1 if td >= T1_FROM else 2
    d = td
    while n:
        d += timedelta(days=1)
        if is_settlement_day(d):
            n -= 1
    return d


def charged_days(entry_ts, end_ts) -> List[date]:
    """The calendar days IBKR charges a short opened at ``entry_ts`` and covered at ``end_ts`` (or still open then):
    from the short's settlement to the cover's settlement, the cover's day excluded."""
    a, b = trade_date(entry_ts), trade_date(end_ts)
    if a is None or b is None:
        return []
    s0, s1 = settlement_date(a), settlement_date(b)
    return [s0 + timedelta(days=k) for k in range(max(0, (s1 - s0).days))]


def collateral_price(close: float) -> float:
    """102% of the prior session's close, rounded UP to the whole dollar — IBKR's value per share."""
    return float(math.ceil(1.02 * float(close) - 1e-9))


# ── IBKR's data ───────────────────────────────────────────────────────────────

def _rates(ticker: str) -> List[Tuple[date, float, str]]:
    """IBKR's daily rate history of ``ticker``, oldest first: (day, %/yr, source) — our archive > IBKR's API > the
    community archive (`borrow_history.fee_daily`)."""
    hit = _rates_cache.get(ticker)
    if hit is not None and time.time() - hit[0] < _RATE_TTL:
        return hit[1]
    rows: List[Tuple[date, float, str]] = []
    try:
        from src.data.deep import borrow_history
        D = borrow_history.fee_daily(ticker)
        for d, f, s in zip(D["date"], D["fee"], D["source"]):
            if f is not None and f == f and float(f) >= 0:
                rows.append((_as_date(d), float(f), str(s)))
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[borrow_fees] {ticker}: no rate history ({type(e).__name__}: {e})")
    rows.sort(key=lambda r: r[0])
    _rates_cache[ticker] = (time.time(), rows)
    return rows


def _file_rate(ticker: str, when: datetime) -> Optional[Tuple[float, date]]:
    """The fee in the copy of IBKR's file in force at ``when`` (our archive / this process's download) and the ET
    date the file was stamped; None when no file or the file does not list the name."""
    try:
        from src.data import ibkr_borrow
        got = ibkr_borrow.snapshot_at(when)
        if got is None:
            return None
        ts, table = got
        b = ibkr_borrow.lookup(table, ticker)
        if b is None or b.fee_pct is None:
            return None
        return float(b.fee_pct), (ts.astimezone(ET).date() if ts is not None else when.astimezone(ET).date())
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[borrow_fees] {ticker}: file lookup failed ({type(e).__name__}: {e})")
        return None


def _charged() -> Dict[Tuple[str, date], dict]:
    """IBKR's charges on our account: (ticker, value date) -> fee per share, IBKR's rate and price."""
    if time.time() - float(_charged_cache["at"]) < _CHARGED_TTL:
        return _charged_cache["map"]                                # type: ignore[return-value]
    out: Dict[Tuple[str, date], dict] = {}
    ver = ""
    try:
        from src.db import repo
        rows = repo.load_broker_borrow_fees()
        for r in rows:
            q = abs(float(r.get("quantity") or 0.0))
            fee = r.get("fee")
            if not q or fee is None:
                continue
            k = (str(r["ticker"]).upper(), _as_date(r["value_date"]))
            got = out.setdefault(k, {"fee": 0.0, "qty": 0.0, "rate": r.get("fee_rate"), "price": r.get("price")})
            got["fee"] += abs(float(fee))
            got["qty"] += q
        for v in out.values():
            v["fee_ps"] = v["fee"] / v["qty"]
        ver = f"{len(rows)}:{max((str(r.get('fetched_at') or '') for r in rows), default='')}"
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[borrow_fees] IBKR's charges unavailable ({type(e).__name__}: {e})")
    _charged_cache.update(at=time.time(), map=out, ver=ver)
    return out


def charged_version() -> str:
    _charged()
    return str(_charged_cache["ver"])


def _closes(ticker: str) -> List[Tuple[date, float]]:
    """The name's session closes, oldest first: the daily OHLCV cache (refreshed every tick for a held name), else
    the deep 30-minute store's last regular-hours bar of each session (its 16:00 close)."""
    from src.performance.daily_nav import _load_close_series
    cs = _load_close_series(ticker)
    if cs:
        return sorted(cs.items())
    hit = _closes_cache.get(ticker)
    if hit is not None and time.time() - hit[0] < _RATE_TTL:
        return hit[1]
    out: List[Tuple[date, float]] = []
    try:
        import pandas as pd
        from src.data.intraday_store import load_deep_30m
        df = load_deep_30m(ticker)
        if df is not None and len(df):
            et = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET)
            last = pd.Series(df["Close"].to_numpy(float), index=et.date).groupby(level=0).last()
            out = [(d, float(c)) for d, c in last.items() if c == c and c > 0]
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[borrow_fees] {ticker}: no 30-minute closes ({type(e).__name__}: {e})")
    _closes_cache[ticker] = (time.time(), out)
    return out


def prior_close(ticker: str, d: date, closes: Optional[List[Tuple[date, float]]] = None) -> Optional[float]:
    """The close of the last session strictly before ``d`` (the settlement price IBKR's value is built on)."""
    cs = closes if closes is not None else _closes(ticker)
    lo, hi = 0, len(cs)
    while lo < hi:
        mid = (lo + hi) // 2
        if cs[mid][0] < d:
            lo = mid + 1
        else:
            hi = mid
    return cs[lo - 1][1] if lo > 0 else None


def day_rate(ticker: str, d: date, now: datetime, trade: Optional[dict] = None) -> Tuple[float, str]:
    """IBKR's annual rate (%) for ``ticker`` on calendar day ``d`` and where it came from (module docstring, 2-3)."""
    today = now.astimezone(ET).date()
    rows = _rates(ticker)
    # a day not over yet (today; an overnight cover's next session) knows only what is known now: the file in force,
    # else the last day already over — never a row dated today or later (its value is the day's END, not yet known)
    cut = d if d < today else today
    exact = last = None
    for r in rows:                                                  # short lists: a linear walk is fine
        if r[0] == d and d < today:
            exact = r
        if r[0] < cut:
            last = r
        if r[0] >= cut:
            break
    if d < today:
        if exact is not None:
            return exact[1], exact[2]
        end_of_day = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=ET)
        f = _file_rate(ticker, end_of_day)
        if f is not None and (last is None or f[1] >= last[0]):
            return f[0], ("own_archive" if f[1] == d else "own_archive_carried")
    else:
        f = _file_rate(ticker, now)
        if f is not None:
            return f[0], "file_now"
    if last is not None:
        entry = trade_date((trade or {}).get("entry_datetime") or (trade or {}).get("entry_date"))
        if (cut - last[0]).days <= MAX_CARRY_DAYS or (entry is not None and (entry - last[0]).days <= MAX_CARRY_DAYS):
            return last[1], f"{last[2]}_carried"
    stamp = (trade or {}).get("borrow_fee_pct")
    try:
        if stamp is not None and float(stamp) >= 0:
            return float(stamp), "entry_stamp"
    except (TypeError, ValueError):
        pass
    return max(0.0, float(getattr(settings, "short_borrow_annual_pct", 0.0) or 0.0)), "default"


# ── a short's schedule ────────────────────────────────────────────────────────

def _shares(trade: dict) -> Optional[float]:
    for k in ("sel_account_shares", "broker_fill_qty", "broker_requested_qty"):
        v = trade.get(k)
        try:
            if v is not None and abs(float(v)) > 0:
                return abs(float(v))
        except (TypeError, ValueError):
            continue
    return None


def _applies(trade: dict) -> bool:
    """A stock short: futures, indices, FX and crypto borrow nothing (`ibkr_borrow.is_equity_symbol`)."""
    if not (str(trade.get("action") or "").upper() == "SELL" and str(trade.get("type") or "STOCK").upper() == "STOCK"
            and trade.get("ticker") and (trade.get("entry_datetime") or trade.get("entry_date"))
            and float(trade.get("entry_price") or 0.0) > 0):
        return False
    from src.data.ibkr_borrow import is_equity_symbol
    return bool(is_equity_symbol(str(trade["ticker"])))


def schedule(trade: dict, end=None, now: Optional[datetime] = None) -> Optional[dict]:
    """The days IBKR charges ``trade`` (a short) up to ``end`` (default: its exit, else now), each with its rate,
    collateral price and fee per share, and the totals."""
    if not _applies(trade):
        return None
    now = now or datetime.now(timezone.utc)
    entry_ts = trade.get("entry_datetime") or trade.get("entry_date")
    end_ts = end or trade.get("exit_datetime") or trade.get("exit_date") or now
    days = charged_days(entry_ts, end_ts)
    ticker = str(trade["ticker"]).upper()
    charged = _charged()
    entry_px = float(trade["entry_price"])
    now_et = now.astimezone(ET)
    today = now_et.date()
    # a session's close is IBKR's settlement price once the session is over: today's counts from 16:15 ET only (a
    # forming bar in the cache, or a later rewrite, never reaches an earlier tick)
    closed_today = (now_et.hour, now_et.minute) >= (16, 15)
    closes = [(cd, c) for cd, c in (_closes(ticker) if days else []) if cd < today or (cd == today and closed_today)]
    recs = []
    for d in days:
        c = charged.get((ticker, d)) if d < today else None        # IBKR charges a day once it is over
        if c is not None:
            recs.append([d.isoformat(), "ibkr_charged", float(c.get("rate") or 0.0), float(c.get("price") or 0.0),
                         float(c["fee_ps"])])
            continue
        rate, src = day_rate(ticker, d, now, trade)
        close = prior_close(ticker, d, closes)
        px = collateral_price(close if close and close > 0 else entry_px)
        recs.append([d.isoformat(), src if close else f"{src}+entry_px", rate, px, px * rate / 100.0 / 360.0])
    fee_ps = float(sum(r[4] for r in recs))
    return {"days": recs, "fee_ps": fee_ps, "frac": fee_ps / entry_px,
            "rate_now": (recs[-1][2] if recs else None),
            "final": bool(days) and days[-1] < today and all(r[1] not in ("file_now",) for r in recs),
            "through": (days[-1].isoformat() if days else None)}


def cost_fraction(trade: dict, end_iso=None, now: Optional[datetime] = None) -> Optional[float]:
    """The return-reducing borrow fraction IBKR's schedule charges ``trade`` through ``end_iso`` (its exit, else now),
    stored on the trade — or None to keep the old flat carry (the schedule off, not a stock short, or a closed trade
    without a stored schedule while the backfill waits for the formula's check)."""
    if not getattr(settings, "enable_ibkr_borrow_schedule", False) or not _applies(trade):
        return None
    closed = str(trade.get("status") or "").upper() == "CLOSED"
    stored = trade.get("borrow_days")
    if closed and stored is None and not getattr(settings, "borrow_backfill_closed", False):
        return None
    natural_end = end_iso is None or (closed and str(end_iso) == str(trade.get("exit_datetime")))
    if (closed and natural_end and trade.get("borrow_final") and stored is not None
            and trade.get("borrow_charged_ver") == charged_version()):
        return float(trade.get("borrow_fee_ps") or 0.0) / float(trade["entry_price"])
    s = schedule(trade, end_iso, now)
    if s is None:
        return None
    if natural_end or not closed:
        trade["borrow_days"] = s["days"]
        trade["borrow_fee_ps"] = round(s["fee_ps"], 6)
        n = _shares(trade)
        trade["borrow_fee_usd"] = round(s["fee_ps"] * n, 4) if n else None
        trade["borrow_rate_now"] = s["rate_now"]
        trade["borrow_final"] = bool(closed and s["final"])
        trade["borrow_through"] = s["through"]
        trade["borrow_charged_ver"] = charged_version()
    return float(s["frac"])


def nav_day_fees(trade: dict) -> Optional[List[Tuple[date, float]]]:
    """(day, fee per share) of a short's STORED schedule for the daily NAV walk — None when there is none."""
    if not getattr(settings, "enable_ibkr_borrow_schedule", False):
        return None
    days = trade.get("borrow_days")
    if not days:
        return None
    out = []
    for r in days:
        try:
            out.append((date.fromisoformat(str(r[0])[:10]), float(r[4])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


# ── the formula vs IBKR's own charges ─────────────────────────────────────────

def verify(out_path: Optional[str] = None) -> dict:
    """Every (name, day) IBKR charged our account, against the formula with OUR inputs: IBKR's rate from our archive /
    its API and the prior close from our daily bars. Also checks IBKR's own arithmetic (fee = value x rate / 360).
    The formula is used for a backfill only when this holds (user 2026-10-06)."""
    import json
    from src.db import repo
    rows = repo.load_broker_borrow_fees()
    now = datetime.now(timezone.utc)
    res = {"rows": len(rows), "checked": 0, "arith_ok": 0, "fee_within_1pct": 0, "fee_exact_cent": 0,
           "rate_match": 0, "price_match": 0, "worst": []}
    diffs = []
    for r in rows:
        q = abs(float(r.get("quantity") or 0.0))
        if not q or r.get("fee") is None:
            continue
        tk = str(r["ticker"]).upper()
        d = _as_date(r["value_date"])
        fee = abs(float(r["fee"]))
        rate_ibkr = float(r.get("fee_rate") or 0.0)
        val = abs(float(r.get("value") or 0.0))
        if val and rate_ibkr and abs(val * rate_ibkr / 100.0 / 360.0 - fee) <= max(0.011, 0.005 * fee):
            res["arith_ok"] += 1
        rate, src = day_rate(tk, d, now)
        close = prior_close(tk, d)
        if close is None:
            continue
        px = collateral_price(close)
        ours = q * px * rate / 100.0 / 360.0
        res["checked"] += 1
        res["rate_match"] += int(abs(rate - rate_ibkr) <= 0.005 + 0.005 * rate_ibkr)
        res["price_match"] += int(abs(px - float(r.get("price") or 0.0)) < 0.005)
        res["fee_within_1pct"] += int(abs(ours - fee) <= 0.01 * max(fee, 0.01))
        res["fee_exact_cent"] += int(abs(ours - fee) <= 0.0051)
        diffs.append((abs(ours - fee), {"ticker": tk, "day": d.isoformat(), "ibkr_fee": fee, "formula_fee": round(ours, 4),
                                        "ibkr_rate": rate_ibkr, "our_rate": rate, "rate_source": src,
                                        "ibkr_price": r.get("price"), "our_price": px, "qty": q}))
    diffs.sort(key=lambda x: -x[0])
    res["worst"] = [d for _, d in diffs[:25]]
    if res["checked"]:
        for k in ("arith_ok", "fee_within_1pct", "fee_exact_cent", "rate_match", "price_match"):
            res[k + "_share"] = round(res[k] / res["checked"], 4)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
    return res


def _main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="the formula vs IBKR's charges (Flex)")
    ap.add_argument("--trade", help="print one short's day-by-day schedule")
    a = ap.parse_args()
    if a.verify:
        r = verify("cache/borrow_fee_verify.json")
        print(json.dumps({k: v for k, v in r.items() if k != "worst"}, indent=1))
        for w in r["worst"][:10]:
            print(w)
    if a.trade:
        from src.db import repo
        t = next((x for x in repo.load_trades() if str(x.get("trade_id")) == a.trade), None)
        if t is None:
            print("no such trade")
            return
        s = schedule(t)
        print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    _main()
