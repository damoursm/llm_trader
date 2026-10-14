"""LIVE vs the simulator's assumptions — the live record measured where the production-mimicking simulator
(`src/backtest/mimic.py`) assumes (2026-10-12; user 2026-10-11: "Do number 1, 2, and 3 tonight", number 2 being
"calibrate the execution and borrow model with live data ... a weekly comparison of each live day against the
simulator would catch execution gaps early").

Each market day the EOD chain's ``live_calibration`` step writes `cache/ml/live_calibration/<day>.json`: the
session's facts (``session``), the last five sessions' (``window``) and the record since the live account began
(``since``, from `LIVE_SINCE`), each beside the simulator's assumption —

* the selection short's live arms at the entry step (`cache/ml/sel_short/entries/`, one decision per pick): the
  share of picks IBKR could lend at their FIRST borrow check (the simulator reads our own archive from 2026-09-25, a
  model before), IBKR's refusals of any opening short among the stock picks it answered (the simulator:
  `mimic.REFUSAL_RATE`) and the house rates it quoted (the simulator: one of `mimic.house_mix()`'s names, drawn);
* the live books' trades (the ledger): a short's entry delay after the bar it was decided on and its entry price
  against that bar's close — the pick bar, or for a fall / ETF pick entered on its TURN (PREREG76 / 77) the turning
  bar; the simulator enters AT that close, stamped at its end — the immediate and the turn entries apart; the
  broker's entry fills (the simulator: the ledger's trade is the account's position, at the ledger's price) and the
  exits by reason and session (the simulator judges the regular and extended sessions; the overnight one is not
  modelled);
* the borrow fees IBKR charged the account (its Flex statement, `broker_borrow_fees`) against the borrow formula with
  OUR inputs — IBKR's rate from our archive / its API and the prior close (`borrow_fees.verify`'s construction: the
  simulator's fee for a listed name) — and IBKR's charged quantity against the ledger's shares that day (the ledger's
  stored schedule takes IBKR's charges over the formula once they arrive, so it is never the comparison). Statements
  arrive with a lag (the EOD chain's Flex thread may still be waiting): the record reads what the table holds;
* the dip book's limit orders by status (the simulator fills them by rule, cut to the account's room);
* EACH ARM'S RELIABILITY (``arms``; item 6 of the user's program, 2026-10-12: "the daily live-vs-simulator
  calibration gives each arm a reliability score: live return per day against the simulated one"): the live arms —
  the selection short's vol, ETF and fall arms by the trade's ``sel_arm``, the dip and crash books by their mechanism
  (`books.py`) — over their CLOSED trades: the mean net trade, the mean 24-hour hold, RETURN PER DAY by the standing
  definition, the GROSS win rate with its Wilson range and the net median with its order-statistic range, each beside
  the simulator's REFERENCE for the arm (`reference.json` under the record's directory, written by another module)
  and a verdict: "no reference", "too few trades" (under `MIN_N` closed), "consistent" (the reference inside the live
  ranges) or "gap" (`arm_compare`, its limitation stated there).

`health()` reads the latest file for the email's scorer banner: one line of the record and a note for every live
rate outside the simulator's assumption beyond its sampling error — a share's Wilson 95% range, a median's
order-statistic 95% range — over the record since `LIVE_SINCE` and over the last five sessions, so a gap shows the
week it appears; an arm's "gap" rides the same way. Nothing here trades or changes a rule; a gap is for the operator
and the next study.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import settings

LIVE_SINCE = date(2026, 10, 9)          # the live account trades the combination from this session (CLAUDE.md)
NO_BORROW = ("no_borrow", "no_borrow_retry", "no_borrow_fallback")
WINDOW_SESSIONS = 5
MIN_N = 5                               # a rate is judged once it rests on this many observations
DELAY_NOTE_MIN = 5.0                    # an entry later than this after its decision bar (the median's range above)
DRIFT_NOTE_BP = 50.0                    # an entry this far from the decision bar's close (the median's range beyond)
FILL_NOTE = 0.8                         # under this share of the live books' legs filled at IBKR (the range below)
FEE_RATIO_NOTE = (0.8, 1.25)            # IBKR's charge over the formula's, or its quantity over the ledger's shares
HOUSE_RATIO_NOTE = (0.5, 2.0)           # IBKR's quoted initial house rate over the simulator's mix (median)
ARM_KEYS = ("vol", "etf", "fall", "dip", "crash")   # the live arms as the simulator names them (`mimic.BOOK`'s keys)
REFERENCE_FILE = "reference.json"       # the simulator's reference per arm, under `out_dir()` (another module writes it)


def out_dir() -> Path:
    return Path(getattr(settings, "live_calibration_dir", "cache/ml/live_calibration"))


def out_path(d: date) -> Path:
    return out_dir() / f"{d.isoformat()}.json"


# ── small statistics ─────────────────────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """The 95% Wilson range of a binomial share (None when n = 0)."""
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def median_range(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """The distribution-free 95% range of a median: the order statistics x(j) .. x(n-j+1) with the largest j whose
    binomial tail P(Bin(n, 1/2) <= j-1) is at most 2.5% (None when n is too small for one — n < 6)."""
    v = np.sort(np.asarray([x for x in xs if x is not None and np.isfinite(x)], float))
    n = len(v)
    j, tail = 0, 0.0
    while j < n // 2:
        tail += math.comb(n, j) / 2.0 ** n                     # P(Bin <= j)
        if tail > 0.025:
            break
        j += 1
    if j == 0:
        return None, None
    return float(v[j - 1]), float(v[n - j])


def _share(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "share": (k / n) if n else None, "lo95": lo, "hi95": hi}


def _summary(xs: Sequence[float]) -> dict:
    v = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    lo, hi = median_range(v)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "median_lo95": lo,
            "median_hi95": hi, "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9))}


# ── the simulator's assumptions ──────────────────────────────────────────────────────────────────────────────────

def assumptions() -> dict:
    """What the production-mimicking simulator assumes where the live record measures. Fail-soft per item."""
    out = {"entry_delay_min": 0.0, "entry_drift_bp": 0.0, "fill_share": 1.0, "fee_ratio": 1.0}
    try:
        from src.backtest import mimic as M
        out["refusal_rate"] = float(M.REFUSAL_RATE)
        mix = M.house_mix()
        init = np.asarray([i for _, _, i in mix], float)
        maint = np.asarray([m for _, m, _ in mix], float)
        out["house_init_median"] = float(np.median(init))
        out["house_maint_median"] = float(np.median(maint))
        out["house_init_above_2_86"] = float(np.mean(init > 2.86 + 1e-9))
        out["house_names"] = int(len(mix))
        out["etf_lend_rate"] = float(M.ETF_LEND_RATE)
    except Exception as e:                                     # noqa: BLE001
        out["error"] = f"mimic assumptions unreadable: {e}"
    return out


# ── the live record ──────────────────────────────────────────────────────────────────────────────────────────────

def _sessions(lo: date, hi: date) -> List[date]:
    from src.signals.sel_short import is_session
    out, d = [], lo
    while d <= hi:
        if is_session(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _entries(days: Iterable[date]) -> List[dict]:
    from src.signals import sel_short as ss
    out = []
    for d in days:
        p = ss.entries_path(d)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _quoted(e: dict) -> bool:
    m = e.get("ibkr_margin")
    return isinstance(m, dict) and m.get("init_rate") is not None


def entry_facts(entries: Sequence[dict], live_arms: Sequence[str]) -> dict:
    """The live arms' entry-step record, ONE decision per pick (the journal holds a line per pass: a pick IBKR cannot
    lend is re-journaled every tick of its window): lendability at its FIRST borrow check, its LAST outcome, IBKR's
    what-if answer — refused, or quoted with its rates (an answer that is neither, a timeout or an error, counts in
    neither)."""
    from src.signals.sel_short import etf_names
    live = sorted((e for e in entries if (e.get("arm") or "model") in live_arms), key=lambda r: str(r.get("at") or ""))
    by_key: Dict[str, List[dict]] = {}
    for e in live:
        by_key.setdefault(str(e.get("key")), []).append(e)
    first_check = {k: next((e for e in es if e.get("borrow_checked")), None) for k, es in by_key.items()}
    lend = [e for e in first_check.values() if e is not None]
    by_arm = {}
    for a in live_arms:
        xs = [e for e in lend if (e.get("arm") or "model") == a]
        by_arm[a] = _share(sum(1 for e in xs if e.get("outcome") not in NO_BORROW), len(xs))
    try:
        etfs = set(etf_names(sorted({str(es[0].get("ticker")) for es in by_key.values()})))
    except Exception:                                          # noqa: BLE001
        etfs = set()
    refused, quoted = 0, []
    for es in by_key.values():
        if str(es[0].get("ticker")) in etfs:
            continue                                           # a fund's answer is not a stock's
        if any(e.get("outcome") == "ibkr_refused" for e in es):
            refused += 1
            continue
        q = next((e["ibkr_margin"] for e in es if _quoted(e)), None)
        if q is not None:
            quoted.append(q)
    init = [float(q["init_rate"]) for q in quoted]
    maint = [float(q["maint_rate"]) for q in quoted if q.get("maint_rate") is not None]
    return {"picks_settled": len(by_key),
            "outcomes": _count(es[-1].get("outcome") for es in by_key.values()),
            "lendable_first_check": _share(sum(1 for e in lend if e.get("outcome") not in NO_BORROW), len(lend)),
            "lendable_by_arm": by_arm,
            "ibkr_refused_stock": _share(refused, refused + len(quoted)),
            "house_init": _summary(init), "house_maint": _summary(maint),
            "house_init_above_2_86": _share(sum(1 for x in init if x > 2.86 + 1e-9), len(init))}


def _count(xs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        k = str(x)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _ts(x) -> Optional[datetime]:
    if not x:
        return None
    try:
        t = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _decision_bar(t: dict) -> Tuple[str, Optional[datetime], Optional[float]]:
    """The bar a short was decided on: its turning bar for a turn entry, else its pick bar — (kind, end, close)."""
    if t.get("sel_turn_bar_end"):
        return "turn", _ts(t.get("sel_turn_bar_end")), t.get("sel_turn_close")
    return "immediate", _ts(t.get("sel_bar_end")), t.get("sel_bar_close")


def trade_facts(trades: Sequence[dict], lo: date, hi: date) -> dict:
    """The live books' ledger record over [lo, hi]: the shorts' entries against their decision bar (the immediate and
    the turn entries apart), the broker's entry fills, the exits."""
    from src.performance import books
    live = [t for t in trades if books.is_live(t)]
    opened = [t for t in live if lo.isoformat() <= str(t.get("entry_date") or "")[:10] <= hi.isoformat()]
    closed = [t for t in live if t.get("status") == "CLOSED"
              and lo.isoformat() <= str(t.get("exit_date") or "")[:10] <= hi.isoformat()]
    delay: Dict[str, List[float]] = {"immediate": [], "turn": []}
    drift: Dict[str, List[float]] = {"immediate": [], "turn": []}
    for t in opened:
        if t.get("entry_mechanism") != "sel_short":
            continue
        kind, be, pc = _decision_bar(t)
        et = _ts(t.get("entry_datetime"))
        if be is not None and et is not None:
            delay[kind].append((et - be).total_seconds() / 60.0)
        ep = t.get("entry_price")
        try:
            if pc and ep and float(pc) > 0:
                drift[kind].append((float(ep) / float(pc) - 1.0) * 1e4)   # + = a higher entry: better for the short
        except (TypeError, ValueError):
            pass
    legs, fills, slip = 0, 0, []
    for t in opened:
        if not t.get("broker_status") and not t.get("broker_order_id"):
            continue
        legs += 1
        if t.get("broker_fill_qty") and float(t.get("broker_fill_qty") or 0) > 0:
            fills += 1
            fp, ep = t.get("broker_fill_price"), t.get("entry_price")
            if fp and ep and float(ep) > 0:
                # + = worse than the ledger's price: a buy filled higher, a short sold lower
                move = (float(fp) / float(ep) - 1.0) * 1e4
                slip.append(-move if str(t.get("action", "")).upper() in ("SELL", "SHORT") else move)
    return {"opened": len(opened), "opened_by_book": _count(t.get("entry_mechanism") for t in opened),
            "closed": len(closed), "exits": _count(f"{t.get('entry_mechanism')}:{t.get('exit_reason')}" for t in closed),
            "exit_sessions": _count(t.get("exit_session") for t in closed),
            "short_entry_delay_min": _summary(delay["immediate"]), "short_entry_drift_bp": _summary(drift["immediate"]),
            "turn_entry_delay_min": _summary(delay["turn"]), "turn_entry_drift_bp": _summary(drift["turn"]),
            "broker_entry_legs": _share(fills, legs), "broker_entry_slippage_bp": _summary(slip)}


def _ledger_shares(trades: Sequence[dict]) -> Dict[Tuple[str, str], float]:
    """(ticker, day) -> the selection short's shares charged that day by the ledger's own schedule (its
    ``borrow_days`` are exactly the days IBKR charges a short); a trade split since its entry is left out (its shares
    are in today's terms, IBKR's quantity in that day's)."""
    out: Dict[Tuple[str, str], float] = {}
    for t in trades:
        if t.get("entry_mechanism") != "sel_short" or t.get("split_factor"):
            continue
        sh = float(t.get("sel_account_shares") or t.get("broker_fill_qty") or 0.0)
        if sh <= 0:
            continue
        for row in t.get("borrow_days") or []:
            try:
                k = (str(t.get("ticker")).upper(), str(row[0])[:10])
            except (TypeError, IndexError):
                continue
            out[k] = out.get(k, 0.0) + sh
    return out


def fee_facts(trades: Sequence[dict], lo: date, hi: date) -> dict:
    """IBKR's charged borrow (Flex) over [lo, hi] against the borrow formula with OUR inputs, per (name, day) —
    `borrow_fees.verify`'s construction — and IBKR's quantity against the ledger's shares that day."""
    from src.db import repo
    from src.performance import borrow_fees as BF
    try:
        rows = repo.load_broker_borrow_fees()
    except Exception as e:                                     # noqa: BLE001
        return {"error": f"broker_borrow_fees unreadable: {e}"}
    now = datetime.now(timezone.utc)
    ledger = _ledger_shares(trades)
    ratio, qty_ratio, charged, no_close, errors, n = [], [], 0.0, 0, 0, 0
    for r in rows:
        try:
            d = BF._as_date(r["value_date"])
            if not (lo <= d <= hi):
                continue
            q, fee = abs(float(r.get("quantity") or 0.0)), r.get("fee")
            if not q or fee is None:
                continue
            fee = abs(float(fee))
            n += 1
            charged += fee
            tk = str(r["ticker"]).upper()
            sh = ledger.get((tk, d.isoformat()))
            if sh:
                qty_ratio.append(q / sh)
            rate, _src = BF.day_rate(tk, d, now)
            close = BF.prior_close(tk, d)
            if close is None:
                no_close += 1
                continue
            ours = q * BF.collateral_price(close) * rate / 100.0 / 360.0
            if ours > 0:
                ratio.append(fee / ours)
        except Exception:                                      # noqa: BLE001 — one row's failure is counted
            errors += 1
    return {"charged_rows": n, "charged_usd": float(charged), "no_prior_close": no_close, "errors": errors,
            "ratio": _summary(ratio), "qty_ratio": _summary(qty_ratio)}


def dip_order_facts(days: Iterable[date]) -> dict:
    """The dip book's limit orders by their last status (`cache/ml/dip_long/orders/<day>.json`)."""
    root = Path(getattr(settings, "dip_long_dir", "cache/ml/dip_long")) / "orders"
    st: Dict[str, int] = {}
    n = 0
    for d in days:
        p = root / f"{d.isoformat()}.json"
        if not p.exists():
            continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        orders = js.get("orders") if isinstance(js, dict) else js
        for o in (orders.values() if isinstance(orders, dict) else (orders or [])):
            if isinstance(o, dict):
                n += 1
                k = str(o.get("status"))
                st[k] = st.get(k, 0) + 1
    return {"orders": n, "status": dict(sorted(st.items(), key=lambda kv: -kv[1]))}


# ── each arm's reliability: its live trades against the simulator's reference ────────────────────────────────────

def _f(x) -> Optional[float]:
    """A finite float, else None (a stored number may be a string, None or NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def arm_books() -> Dict[str, str]:
    """Each arm key -> the ledger's entry mechanism (`books.py`'s constants, the one definition of the live books):
    the selection short's arms share `books.SEL_SHORT` and are told apart by the trade's ``sel_arm``; the dip and
    crash books ARE their mechanism. The keys are the simulator's (`mimic.BOOK`; a test pins the two sites equal)."""
    from src.performance import books
    return {"vol": books.SEL_SHORT, "etf": books.SEL_SHORT, "fall": books.SEL_SHORT,
            "dip": books.DIP_LONG, "crash": books.CRASH_LONG}


def arm_of(t: dict) -> Optional[str]:
    """The live arm a ledger trade belongs to; None for every other trade — a shadow book's, a selection short of a
    shadow arm (model, vol2, thin), the ETF dual-momentum book's. A selection short carries ONE arm since 2026-10-04
    (one trade per arm, `tracker.record_sel_short_trades`); an earlier shared stamp (``"model+vol"``) counts under
    its first live arm."""
    from src.performance import books
    bk = arm_books()
    m = t.get("entry_mechanism")
    if m == books.SEL_SHORT:
        return next((a for a in str(t.get("sel_arm") or "").split("+") if bk.get(a) == books.SEL_SHORT), None)
    return next((a for a, b in bk.items() if b == m and b != books.SEL_SHORT), None)


def hold_days(t: dict) -> Optional[float]:
    """A trade's hold in 24-hour days: its exit instant minus its entry instant — the ledger's execution stamps
    ``exit_datetime`` / ``entry_datetime``; a date-only legacy row counts from midnight to midnight. None when either
    is missing or unreadable (the standing definition's hold is a time, never the ledger's trading-day count)."""
    a = _ts(t.get("entry_datetime") or t.get("entry_date"))
    b = _ts(t.get("exit_datetime") or t.get("exit_date"))
    if a is None or b is None:
        return None
    return (b - a).total_seconds() / 86400.0


def return_per_day(avg_trade: Optional[float], avg_days: Optional[float]) -> Optional[float]:
    """RETURN PER DAY by the standing definition, in %/day: (1 + the average net trade)^(1 / the average 24-hour
    hold) - 1 — `account.per_day`, the simulator's own, so the live number and the reference are one construction.
    None without a trade or a positive hold."""
    if avg_trade is None or avg_days is None or not avg_days > 0:
        return None
    from src.backtest.account import per_day
    v = per_day(float(avg_trade), float(avg_days))
    return float(v) if math.isfinite(v) else None


def arm_facts(trades: Sequence[dict], lo: date, hi: date) -> dict:
    """Each live arm's record over its CLOSED trades entered in [lo, hi] (``entry_date``, the ET session) — the
    simulator's per-arm metrics (`account.metrics`) on the live ledger: ``n``, the mean net trade (``avg_trade``:
    ``return_pct``, cost-aware, in %), the mean hold in 24-hour days (``avg_days``), ``return_per_day`` by the
    standing definition, the GROSS win rate — `tracker.is_gross_win`, the right direction on raw prices; a trade
    without a usable price pair in neither count — as a share with its Wilson range (``gross_win``) and as the
    convention's helper reports it (``gross_win_rate``, %), the net return's median with its order-statistic range
    (``net``) and the hold's (``hold``). The trades still open are counted apart (``open_n``), never in the
    statistics: their return is not known yet."""
    from src.performance import tracker as T
    lo_s, hi_s = lo.isoformat(), hi.isoformat()
    groups: Dict[str, List[dict]] = {a: [] for a in ARM_KEYS}
    open_n: Dict[str, int] = {a: 0 for a in ARM_KEYS}
    for t in trades:
        a = arm_of(t)
        if a is None or not (lo_s <= str(t.get("entry_date") or "")[:10] <= hi_s):
            continue
        st = str(t.get("status") or "").upper()
        if st == "OPEN":
            open_n[a] += 1
        elif st == "CLOSED":
            groups[a].append(t)
    out: Dict[str, dict] = {}
    for a, ts in groups.items():
        rets = [r for r in (_f(t.get("return_pct")) for t in ts) if r is not None]
        holds = [h for h in (hold_days(t) for t in ts) if h is not None]
        wins = [w for w in (T.is_gross_win(t) for t in ts) if w is not None]
        avg_trade = float(np.mean(rets)) if rets else None
        avg_days = float(np.mean(holds)) if holds else None
        out[a] = {"n": len(ts), "open_n": open_n[a], "avg_trade": avg_trade, "avg_days": avg_days,
                  "return_per_day": return_per_day(avg_trade, avg_days),
                  "gross_win": _share(sum(1 for w in wins if w), len(wins)),
                  "gross_win_rate": T.gross_win_rate(ts),
                  "net": _summary(rets), "hold": _summary(holds)}
    return out


def reference() -> dict:
    """The simulator's reference per arm — `REFERENCE_FILE` under `out_dir()`, written by another module:
    ``{"as_of": ISO date, "source": str, "arms": {arm: {"return_per_day": %/day, "gross_win_rate": a share,
    "avg_trade": %, "avg_days": days, "n": int, "set": "A" | "B"}}}``. {} when the file is missing or unreadable:
    every arm then has no reference and nothing is compared."""
    p = out_dir() / REFERENCE_FILE
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return js if isinstance(js, dict) and isinstance(js.get("arms"), dict) else {}


def arm_compare(f: dict, ref: Optional[dict]) -> dict:
    """One arm's live record (`arm_facts`'s entry) against its simulator reference — ``verdict``: "no reference"
    (none for the arm, or one without a testable figure), "too few trades" (under `MIN_N` closed), "consistent"
    when every reference figure the live record can test lies inside the live range, "gap" otherwise; ``checks``
    holds each test — ``gross_win_rate``: the reference share against the live GROSS win rate's Wilson 95% range;
    ``avg_trade``: the reference's AVERAGE trade against the live net trades' MEDIAN order-statistic 95% range (from
    n = 6; at n = 5 only the win rate is tested). The second is a crude stand-in for a test of return per day, and
    its limitation is known: a mean is held against a median's range — a book whose few large winners carry its mean
    (the shorts) has a median under its mean, so a reference mean above the live median range can be skew rather than
    a gap, and the hold is not tested at all. The live and reference return per day are shown side by side
    (``checks["return_per_day"]``), their difference untested: no distribution-free range exists for a compounded
    ratio of two means at 5–20 trades. A "gap" is a note for the operator, never a trading change."""
    if not isinstance(ref, dict) or not ref:
        return {"verdict": "no reference", "checks": {}}
    checks: Dict[str, dict] = {}
    out: Dict[str, object] = {"reference": ref, "checks": checks}
    live_rpd, ref_rpd = _f(f.get("return_per_day")), _f(ref.get("return_per_day"))
    checks["return_per_day"] = {"live": live_rpd, "reference": ref_rpd,
                                "diff": (live_rpd - ref_rpd) if live_rpd is not None and ref_rpd is not None else None}
    if int(f.get("n", 0) or 0) < MIN_N:
        out["verdict"] = "too few trades"
        return out
    gw, r = f.get("gross_win") or {}, _f(ref.get("gross_win_rate"))
    if r is not None and gw.get("lo95") is not None:
        checks["gross_win_rate"] = {"reference": r, "live": gw.get("share"), "lo95": gw["lo95"], "hi95": gw["hi95"],
                                    "inside": bool(gw["lo95"] <= r <= gw["hi95"])}
    net, r = f.get("net") or {}, _f(ref.get("avg_trade"))
    if r is not None and net.get("median_lo95") is not None:
        checks["avg_trade"] = {"reference": r, "live_median": net.get("median"), "lo95": net["median_lo95"],
                               "hi95": net["median_hi95"], "inside": bool(net["median_lo95"] <= r <= net["median_hi95"])}
    tested = [c["inside"] for c in checks.values() if "inside" in c]
    out["verdict"] = "no reference" if not tested else ("consistent" if all(tested) else "gap")
    return out


def arm_records(trades: Sequence[dict], lo: date, hi: date, ref_arms: Optional[dict] = None) -> dict:
    """The record's ``arms`` section: `arm_facts` over [lo, hi] with each arm's comparison to its reference
    (`arm_compare`; ``ref_arms`` = the reference file's ``arms``, read here when not given)."""
    if ref_arms is None:
        ref_arms = reference().get("arms") or {}
    out = arm_facts(trades, lo, hi)
    for a, f in out.items():
        f.update(arm_compare(f, ref_arms.get(a)))
    return out


def _arm_notes(rec: dict) -> List[Tuple[str, str]]:
    """(key, note) for every arm whose verdict is "gap" on `MIN_N` or more closed trades (see `_notes`)."""
    out: List[Tuple[str, str]] = []
    for a, f in sorted((rec.get("arms") or {}).items()):
        if not isinstance(f, dict) or f.get("verdict") != "gap" or int(f.get("n", 0) or 0) < MIN_N:
            continue
        ch = f.get("checks") or {}
        bits = []
        g = ch.get("gross_win_rate") or {}
        if g and not g.get("inside"):
            gw = f.get("gross_win") or {}
            bits.append(f"gross win {gw.get('k')}/{gw.get('n')} ({(gw.get('share') or 0):.0%}; the simulator "
                        f"{g['reference']:.0%})")
        m = ch.get("avg_trade") or {}
        if m and not m.get("inside"):
            bits.append(f"net median {m['live_median']:+.2f}% a trade (the simulator's average {m['reference']:+.2f}%)")
        r = ch.get("return_per_day") or {}
        tail = (f" — {r['live']:.2f} vs the simulator's {r['reference']:.2f} %/day"
                if r.get("live") is not None and r.get("reference") is not None else "")
        out.append((f"arm_{a}", f"the {a} arm's {f['n']} live trades: " + "; ".join(bits) + tail))
    return out


def _notes(rec: dict, assume: dict) -> List[Tuple[str, str]]:
    """(key, note) for every live rate outside the simulator's assumption beyond its sampling error, n >= `MIN_N`."""
    out: List[Tuple[str, str]] = []
    e, t, f = rec.get("entries") or {}, rec.get("trades") or {}, rec.get("fees") or {}
    r = e.get("ibkr_refused_stock") or {}
    if r.get("n", 0) >= MIN_N and assume.get("refusal_rate") is not None and r.get("lo95") is not None:
        if not (r["lo95"] <= assume["refusal_rate"] <= r["hi95"]):
            out.append(("refusal", f"IBKR refused {r['k']}/{r['n']} stock picks ({r['share']:.0%}; the simulator "
                                   f"assumes {assume['refusal_rate']:.0%})"))
    h = e.get("house_init") or {}
    if h.get("n", 0) >= MIN_N and assume.get("house_init_median") and h.get("median_lo95") is not None:
        a = assume["house_init_median"]
        if h["median_lo95"] > HOUSE_RATIO_NOTE[1] * a or h["median_hi95"] < HOUSE_RATIO_NOTE[0] * a:
            out.append(("house", f"IBKR's quoted initial house rate median {h['median']:.2f} vs the simulator's mix "
                                 f"{a:.2f}"))
    for kind, label in (("short", "shorts"), ("turn", "turn shorts")):
        d = t.get(f"{kind}_entry_delay_min") or {}
        if d.get("n", 0) >= MIN_N and d.get("median_lo95") is not None and d["median_lo95"] > DELAY_NOTE_MIN:
            out.append((f"{kind}_delay", f"{label} entered a median {d['median']:.1f} min after their decision bar "
                                         f"(the simulator: at its close)"))
        dr = t.get(f"{kind}_entry_drift_bp") or {}
        if dr.get("n", 0) >= MIN_N and dr.get("median_lo95") is not None and (
                dr["median_lo95"] > DRIFT_NOTE_BP or dr["median_hi95"] < -DRIFT_NOTE_BP):
            out.append((f"{kind}_drift", f"{label}' entries a median {dr['median']:+.0f} bp from their decision bar's "
                                         f"close (the simulator: at it)"))
    b = t.get("broker_entry_legs") or {}
    if b.get("n", 0) >= MIN_N and b.get("hi95") is not None and b["hi95"] < FILL_NOTE:
        out.append(("fills", f"IBKR filled {b['k']}/{b['n']} of the live books' entries ({b['share']:.0%}; the "
                             f"simulator: all)"))
    for key, label in (("ratio", "IBKR charged {m:.2f}x the borrow formula's fee"),
                       ("qty_ratio", "IBKR charged borrow on {m:.2f}x the ledger's shares")):
        fr = f.get(key) or {}
        if fr.get("n", 0) >= MIN_N and fr.get("median_lo95") is not None and (
                fr["median_lo95"] > FEE_RATIO_NOTE[1] or fr["median_hi95"] < FEE_RATIO_NOTE[0]):
            out.append((f"fee_{key}", label.format(m=fr["median"]) + f" (median of {fr['n']} name-days)"))
    out.extend(_arm_notes(rec))
    return out


def compare(rec: dict, assume: dict) -> List[str]:
    """The notes of one section (see `_notes`)."""
    return [n for _, n in _notes(rec, assume)]


def build(day: date, trades: Optional[Sequence[dict]] = None) -> dict:
    """The day's record (``day`` its ISO date): the session, the last `WINDOW_SESSIONS` sessions and since
    `LIVE_SINCE`, with the notes — the record's since `LIVE_SINCE`, then the last sessions' that it does not show."""
    from src.signals import sel_short as ss
    if trades is None:
        from src.db import repo
        trades = repo.load_trades()
    arms = sorted(ss.traded_arms() or ss.live_arms())
    days_all = _sessions(LIVE_SINCE, day)
    win = days_all[-WINDOW_SESSIONS:]
    assume = assumptions()
    ref = reference()
    ref_arms = ref.get("arms") or {}
    rec: Dict[str, object] = {"day": day.isoformat(), "live_since": LIVE_SINCE.isoformat(), "live_arms": arms,
                              "assumptions": assume, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              # the simulator's reference per arm: what the arms' verdicts were judged against
                              "reference": ({"path": str(out_dir() / REFERENCE_FILE), "as_of": ref.get("as_of"),
                                             "source": ref.get("source"), "arms": sorted(ref_arms)} if ref
                                            else {"path": str(out_dir() / REFERENCE_FILE), "missing": True})}
    for name, ds in (("session", [day] if day in days_all else []), ("window", win), ("since", days_all)):
        if not ds:
            rec[name] = {"sessions": 0}
            continue
        lo, hi = ds[0], ds[-1]
        part = {"sessions": len(ds), "from": lo.isoformat(), "to": hi.isoformat()}
        for key, fn in (("entries", lambda: entry_facts(_entries(ds), arms)),
                        ("trades", lambda: trade_facts(trades, lo, hi)),
                        ("fees", lambda: fee_facts(trades, lo, hi)),
                        ("dip_orders", lambda: dip_order_facts(ds)),
                        ("arms", lambda: arm_records(trades, lo, hi, ref_arms))):
            try:
                part[key] = fn()
            except Exception as e:                             # noqa: BLE001 — one section's failure is reported
                part[key] = {"error": str(e)[:300]}
        rec[name] = part
    since = _notes(rec["since"], assume) if rec["since"].get("sessions") else []
    window = _notes(rec["window"], assume) if rec["window"].get("sessions") else []
    seen = {k for k, _ in since}
    rec["notes"] = [n for _, n in since] + [f"last {len(win)} sessions: {n}" for k, n in window if k not in seen]
    return rec


def write(day: date, trades: Optional[Sequence[dict]] = None) -> Path:
    from src.scheduler import db_fence
    rec = build(day, trades)
    p = out_path(day)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    db_fence.retry_os(lambda: os.replace(tmp, p))              # the scheduler may be reading the file this moment
    return p


def latest() -> Optional[dict]:
    try:
        files = sorted(out_dir().glob("????-??-??.json"))
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    except (OSError, ValueError):
        return None


def health() -> dict:
    """The email's scorer banner: one line of the live record since the live account began, and the notes."""
    rec = latest()
    if not rec:
        return {"problems": [], "notes": []}
    s = rec.get("since") or {}
    e, t = s.get("entries") or {}, s.get("trades") or {}
    lf = e.get("lendable_first_check") or {}
    ns = int(s.get("sessions", 0) or 0)
    parts = [f"live vs simulator since {rec.get('live_since')} ({ns} session{'' if ns == 1 else 's'}, as of {rec.get('day')})"]
    if lf.get("n"):
        parts.append(f"lendable at entry {lf['k']}/{lf['n']}")
    rf = e.get("ibkr_refused_stock") or {}
    if rf.get("n"):
        parts.append(f"IBKR refused {rf['k']}/{rf['n']}")
    d = t.get("short_entry_delay_min") or {}
    if d.get("n"):
        parts.append(f"entry {d['median']:.1f} min after the pick bar")
    dt = t.get("turn_entry_delay_min") or {}
    if dt.get("n"):
        parts.append(f"turn entries {dt['median']:.1f} min after the turning bar")
    b = t.get("broker_entry_legs") or {}
    if b.get("n"):
        parts.append(f"IBKR filled {b['k']}/{b['n']} entries")
    notes = [", ".join(parts)] + [f"live vs simulator: {n}" for n in rec.get("notes") or []]
    return {"problems": [], "notes": notes}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.scheduler import db_fence
    ap = argparse.ArgumentParser(description="The live record against the simulator's assumptions.")
    ap.add_argument("--day", default=None)
    a = ap.parse_args(argv)
    if a.day:
        d = date.fromisoformat(a.day)
    else:                                                      # the ET market day, never the UTC date
        from zoneinfo import ZoneInfo
        d = datetime.now(ZoneInfo("America/New_York")).date()
    db_fence.install()                                         # the database only between ticks, as the EOD process
    p = write(d)
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({"path": str(p), "notes": rec.get("notes"), "since": rec.get("since")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
