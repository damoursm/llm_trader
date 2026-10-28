"""IBKR'S REFUSALS, ASKED EVERY EVENING (from 2026-10-26; PREREG191 — the user: "Implement changes 1 to 4").

After the close the EOD process launches ``python -m src.signals.refusal_probe --day D``, which asks IBKR's what-if —
production's order (a ~$500 DAY limit SELL at the last price, priced by IBKR and never transmitted) and production's
refusal parsing (`src.broker.ibkr.whatif_refusal`) — on its own API client (``ibkr_client_id`` +
``refusal_probe_client_offset``), one name every ``refusal_probe_spacing_seconds``, in this order:

1. THE RESEARCH QUEUE (``sel_short_dir``/refusal/queue.json — PREREG191's microcap sample until it is asked), not counted
   in the cap;
2. every name whose last answer was a refusal, again after ``refusal_probe_reask_days`` — how long a refusal lasts;
3. THE WATCH (from 2026-10-24, the user's "Yes, go ahead with both"): the names IBKR last ACCEPTED that the model finds
   riskiest among the vol arm's universe of the day, again after ``refusal_probe_reask_days``, at most
   ``refusal_probe_watch_names`` a night — a name turning refused (close-only, a chill) is seen only by asking the same
   names over time;
4. the day's live picks and their backups IBKR was not asked about (the picks journal: change 3's evening half);
5. the day's new names in the vol arm's top 50 by ATR% — the candidates the simulator decides on;
6. the names the model is least sure of (the highest p(1-p) among the names the vol arm ranked that day);

2-6 together at most ``refusal_probe_max_names``, none asked within ``refusal_probe_fresh_days`` (re-asks and the watch
apart), the funds left out (the model never refuses one). GENTLE BY DESIGN — two bulk probes at 1-2 s spacing once each coincided
with the gateway losing IBKR (2026-10-05): one name a minute, none from one minute before to three minutes after any
:00 / :30 slot (and 19:50), paused while the EOD's spread sweep runs (its own client on the same gateway), STOPPED at
once when IBC logs a server reconnect or a re-login after the start, ended at ``refusal_probe_end_et``; paper accounts
only. Answers: ``probes/<day>.jsonl`` (each with its source and the model's probability at the ask), the run's state in
``probes/<day>.state.json``. The live calibration reads them (`live_calibration.refusal_probe_facts`).

    python -m src.signals.refusal_probe --day D [--plan]
    python -m src.signals.refusal_probe --queue FILE              # the research queue: FILE's "order" (or "names")
    python -m src.signals.refusal_probe --import FILE --source S  # a study's answers (PREREG190 / 191), never asked again
                                                                  # within the fresh days; refusals re-asked weekly
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config import settings

ET = ZoneInfo("America/New_York")
VALUE_USD = 500.0
LOCK = "probe.lock"
LIVE_ARMS = ("vol", "etf", "fall")
EXTRA_SLOTS = ("19:50",)


def root() -> Path:
    return Path(settings.sel_short_dir) / "refusal"


def probes_dir() -> Path:
    return root() / "probes"


def journal_path(d: date) -> Path:
    return probes_dir() / f"{d.isoformat()}.jsonl"


def state_path(d: date) -> Path:
    return probes_dir() / f"{d.isoformat()}.state.json"


def queue_path() -> Path:
    return root() / "queue.json"


def _at(a: dict) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(a.get("at")))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=ET)


def _answered(a: dict) -> bool:
    return bool(a.get("refused")) or a.get("init_rate") is not None


def answers(today: Optional[date] = None, live_days: int = 60) -> Dict[str, List[dict]]:
    """{ticker: answers} — every probe journal's answers, the studies' imported answers (``import_<source>.jsonl``)
    and the live entry journals' what-if answers (the last ``live_days`` days): what IBKR said about each name, and
    when."""
    out: Dict[str, List[dict]] = {}
    for f in sorted(probes_dir().glob("????-??-??.jsonl")) + sorted(probes_dir().glob("import_*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                a = json.loads(ln)
            except ValueError:
                continue
            if _answered(a) and _at(a) is not None:
                out.setdefault(str(a["ticker"]).upper(), []).append(a)
    from src.signals import sel_short as ss
    today = today or datetime.now(ET).date()
    for k in range(live_days):
        p = ss.main_entries_path(today - timedelta(days=k))
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            m = r.get("ibkr_margin")
            if isinstance(m, dict) and _answered(m) and r.get("ticker"):
                out.setdefault(str(r["ticker"]).upper(), []).append(
                    {"ticker": r["ticker"], "at": r.get("at"), "refused": m.get("refused"),
                     "init_rate": m.get("init_rate"), "source": "live"})
    return out


def queue() -> List[str]:
    try:
        return [str(t).upper() for t in json.loads(queue_path().read_text(encoding="utf-8")).get("names", [])]
    except (OSError, ValueError):
        return []


def day_picks(d: date) -> List[str]:
    """The live arms' picks of session ``d`` and their fresh pools / backups (the picks journal)."""
    from src.signals import sel_short as ss
    p = ss.picks_path(d)
    out: List[str] = []
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("arm") not in LIVE_ARMS:
            continue
        for t in [r.get("ticker")] + [x.get("ticker") if isinstance(x, dict) else (x[0] if isinstance(x, (list, tuple))
                                                                                   else x)
                                      for x in (r.get("fresh_pool") or []) + (r.get("backups") or [])]:
            if t and str(t).upper() not in out:
                out.append(str(t).upper())
    return out


def _scores(d: date):
    import pandas as pd
    p = Path(settings.sel_short_dir) / "scores_vol" / f"{d.isoformat()}.pkl"
    return pd.read_pickle(p) if p.exists() else None


def top_names(d: date, n: int) -> List[str]:
    """The vol arm's top ``n`` names by 30-minute ATR% of each bar of session ``d`` (its score file)."""
    s = _scores(d)
    out: List[str] = []
    if s is None:
        return out
    for _, g in s.groupby("bar"):
        for t in g.nlargest(n, "score").ticker.astype(str):
            if t.upper() not in out:
                out.append(t.upper())
    return out


def scored_names(d: date) -> List[str]:
    """Every name the vol arm ranked in session ``d`` — its universe that day."""
    s = _scores(d)
    return [] if s is None else sorted({str(t).upper() for t in s.ticker.astype(str)})


def plan(d: date, now: Optional[datetime] = None) -> List[Tuple[str, str]]:
    """[(ticker, source)] in asking order (the module docstring)."""
    from src.signals import sel_refusal as SR
    from src.signals.sel_short import etf_names
    now = now or datetime.now(ET)
    rec = answers(d)
    last = {t: max((_at(a) for a in xs if _at(a) is not None), default=None) for t, xs in rec.items()}
    fresh = timedelta(days=int(settings.refusal_probe_fresh_days))
    cap = int(settings.refusal_probe_max_names)
    out: List[Tuple[str, str]] = []
    seen: set = set()

    def recent(t: str) -> bool:
        return last.get(t) is not None and now - last[t] < fresh

    def add(t: str, src: str) -> None:
        if t not in seen:
            seen.add(t)
            out.append((t, src))
    for t in queue():                                          # 1. the research queue, until each name is answered
        if t not in rec:
            add(t, "queue")
    n_queue = len(out)
    reask = timedelta(days=int(settings.refusal_probe_reask_days))
    latest = {t: max((a for a in xs if _at(a) is not None), key=_at, default=None) for t, xs in rec.items()}
    for t in sorted(rec):                                      # 2. a refusal, asked again after a week
        lx = latest[t]
        if lx is not None and lx.get("refused") and now - _at(lx) >= reask:
            add(t, "reask")
    universe = scored_names(d)
    ps_all = SR.probabilities(universe, d) if universe else {}
    watch = [t for t in universe if t not in seen and latest.get(t) is not None and not latest[t].get("refused")
             and latest[t].get("init_rate") is not None and now - _at(latest[t]) >= reask and ps_all.get(t, 0.0) > 0]
    for t in sorted(watch, key=lambda t: -ps_all[t])[:max(0, int(settings.refusal_probe_watch_names))]:
        add(t, "watch")                                        # 3. the riskiest accepted names, again after a week
    pool3 = [t for t in day_picks(d) if not recent(t)]
    pool4 = [t for t in top_names(d, 50) if not recent(t)]
    try:
        funds = set(etf_names(sorted(set(pool3 + pool4))))
    except Exception:                                          # noqa: BLE001
        funds = set()
    for t in pool3:                                            # 4. the day's picks and backups
        if t not in funds:
            add(t, "pick")
    for t in pool4:                                            # 5. the day's new top-50 names
        if t not in funds:
            add(t, "top50")
    if len(out) - n_queue < cap:                               # 6. the names the model is least sure of
        cands = [t for t in universe if t not in seen and not recent(t)]
        ps = {t: ps_all.get(t, 0.0) for t in cands}
        for t in sorted((t for t in cands if ps.get(t, 0.0) > 0), key=lambda t: -ps[t] * (1 - ps[t])):
            if len(out) - n_queue >= cap:
                break
            add(t, "uncertain")
    return out[:n_queue + cap]


# ── the guards ───────────────────────────────────────────────────────────────

def near_slot(now: datetime) -> bool:
    """Within a minute before to three after any :00 / :30 slot, or an extra slot (19:50)."""
    m = now.hour * 60 + now.minute + now.second / 60.0
    slots = [h * 60 + mm for h in range(24) for mm in (0, 30)]
    slots += [int(s[:2]) * 60 + int(s[3:]) for s in EXTRA_SLOTS]
    return any(t - 1.0 <= m <= t + 3.0 for t in slots)


def sweep_running(d: date) -> bool:
    """The EOD's spread sweep (its own client on the same gateway) is running."""
    try:
        st = json.loads((Path("cache/eod/state") / f"{d.isoformat()}.json").read_text(encoding="utf-8"))
        return (st.get("steps", {}).get("spread_sweep") or {}).get("status") == "running"
    except (OSError, ValueError):
        return False


def ibc_mark() -> Tuple[Optional[str], int]:
    try:
        f = max(glob.glob(str(Path(settings.ibc_log_dir) / "IBC-*.txt")), key=os.path.getmtime)
        return f, os.path.getsize(f)
    except (ValueError, OSError):
        return None, 0


def ibc_events_since(mark: Tuple[Optional[str], int]) -> int:
    """IBC's server reconnects / re-login demands since ``mark`` (a newer log: the gateway restarted — one)."""
    f0, off = mark
    try:
        f = max(glob.glob(str(Path(settings.ibc_log_dir) / "IBC-*.txt")), key=os.path.getmtime)
        if f0 is None or f != f0:
            return 1
        with open(f, "rb") as fh:
            fh.seek(off)
            t = fh.read().decode("utf-8", "replace")
        return t.count("Connecting to server...; event=Opened") + t.count("Re-login is required; event=Opened")
    except (ValueError, OSError):
        return 0


def window_end(d: date) -> datetime:
    """The evening window's end, ``refusal_probe_end_et`` of session ``d``."""
    h, m = (int(x) for x in str(settings.refusal_probe_end_et).split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET)


# ── the asking ───────────────────────────────────────────────────────────────

def ask(ib, ticker: str, price: float, usd: float, ccy: Optional[str]) -> dict:
    """One what-if: production's order and refusal parsing. The answer's ``refused`` or rates, or ``error``."""
    from ib_async import LimitOrder, Stock
    from src.broker.ibkr import to_ib_symbol, whatif_refusal
    sym = to_ib_symbol(ticker)
    errs: List[Tuple[int, str]] = []

    def on_err(req_id, code, text, contract=None, *a):
        if getattr(contract, "symbol", None) == sym and int(code) not in (399, 10349):
            errs.append((int(code), str(text)))
    qty = max(1, int(math.floor(VALUE_USD / price)))
    out: dict = {"price": price, "qty": qty}
    ib.errorEvent += on_err
    try:
        c = Stock(sym, "SMART", "USD")
        if not ib.qualifyContracts(c):
            return {**out, "error": "not qualified"}
        o = LimitOrder("SELL", qty, round(price, 2 if price >= 1 else 4))
        o.tif = "DAY"
        prev = ib.RequestTimeout
        ib.RequestTimeout = float(settings.sel_short_whatif_timeout_seconds)
        try:
            st = ib.whatIfOrder(c, o)
        finally:
            ib.RequestTimeout = prev
        if not st and not errs:
            ib.sleep(0.5)
        refusal = whatif_refusal(errs)
        if refusal:
            return {**out, "refused": refusal}
        if not st:
            return {**out, "error": "empty what-if answer"}

        def num(x):
            try:
                v = float(x)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) and abs(v) < 1e300 else None
        ic, mc = num(getattr(st, "initMarginChange", None)), num(getattr(st, "maintMarginChange", None))
        if not (ic and mc and ic > 0 and mc > 0):
            return {**out, "error": "no margin in IBKR's answer"}
        val = qty * price
        return {**out, "init_rate": round(ic * usd / val, 4), "maint_rate": round(mc * usd / val, 4), "currency": ccy,
                "warning": (getattr(st, "warningText", "") or "")[:240] or None}
    except Exception as e:                                     # noqa: BLE001
        return {**out, "error": f"{type(e).__name__}: {e}"[:240]}
    finally:
        ib.errorEvent -= on_err


def _price(ticker: str) -> Optional[float]:
    try:
        from src.data.polygon_client import get_snapshots_batch
        px = float(((get_snapshots_batch([ticker]) or {}).get(ticker) or {}).get("price") or 0.0)
        if px > 0 and math.isfinite(px):
            return px
    except Exception:                                          # noqa: BLE001
        pass
    import pandas as pd
    for f in sorted(glob.glob(str(Path(settings.sel_short_dir) / "grouped" / "*.pkl")), reverse=True)[:15]:
        g = pd.read_pickle(f)
        hit = g.loc[g.ticker.astype(str) == ticker, "close"]
        if len(hit) and float(hit.iloc[0]) > 0:
            return float(hit.iloc[0])
    return None


def _write_state(d: date, st: dict) -> None:
    p = state_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def run(d: date) -> dict:
    """One evening's probe (the module docstring). Returns its state."""
    from src.scheduler import db_fence
    from src.signals import sel_refusal as SR
    if str(settings.broker_mode or "").lower() != "ibkr_paper":
        return {"skipped": f"broker_mode {settings.broker_mode}"}
    now = datetime.now(ET)
    if now.date() != d or now >= window_end(d) or now.hour < 16:
        return {"skipped": f"outside {d}'s evening window"}
    lock = root() / LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        holder = int(lock.read_text(encoding="utf-8").split()[0])
        if holder != os.getpid() and db_fence.pid_alive(holder):
            return {"skipped": f"another probe runs (pid {holder})"}
    except (OSError, ValueError, IndexError):
        pass
    lock.write_text(f"{os.getpid()} {datetime.now(ET).isoformat(timespec='seconds')}", encoding="utf-8")
    todo = plan(d)
    done_today = {str(a.get("ticker")).upper() for a in _read(journal_path(d)) if _answered(a)}
    todo = [(t, s) for t, s in todo if t not in done_today]
    st = {"day": d.isoformat(), "pid": os.getpid(), "started": datetime.now(ET).isoformat(timespec="seconds"),
          "planned": len(todo), "by_source": {}, "asked": 0, "answered": 0, "refused": 0, "stopped": None}
    for _, s in todo:
        st["by_source"][s] = st["by_source"].get(s, 0) + 1
    _write_state(d, st)
    if not todo:
        st["stopped"] = "nothing to ask"
        _write_state(d, st)
        lock.unlink(missing_ok=True)
        return st
    from ib_async import IB
    from src.broker.fx import usd_per_unit
    mark = ibc_mark()
    ib = IB()
    try:
        ib.connect(settings.ibkr_host, settings.ibkr_port,
                   clientId=int(settings.ibkr_client_id) + int(settings.refusal_probe_client_offset), timeout=20)
        ccy = None
        try:
            rows = ib.accountValues() or ib.accountSummary()
            ccy = next((r.currency for r in rows if r.tag == "NetLiquidation" and r.currency), None)
        except Exception:                                      # noqa: BLE001
            pass
        usd = usd_per_unit(ccy) if ccy else 1.0
        end = window_end(d)
        for tk, src in todo:
            while True:
                now = datetime.now(ET)
                if now >= end:
                    st["stopped"] = "end of the window"
                    break
                if near_slot(now) or sweep_running(d):
                    ib.sleep(15)
                    continue
                break
            if st["stopped"]:
                break
            if ibc_events_since(mark):
                st["stopped"] = f"IBC logged a server reconnect / re-login before {tk}"
                break
            if not ib.isConnected():
                st["stopped"] = f"the API session dropped before {tk}"
                break
            t0 = time.monotonic()
            row = {"ticker": tk, "source": src, "at": datetime.now(ET).isoformat(timespec="seconds")}
            px = _price(tk)
            if not px:
                row["error"] = "no price"
            else:
                row.update(ask(ib, tk, px, usd, ccy))
                try:
                    row["p_model"] = SR.probability(tk, d, px)
                except Exception:                              # noqa: BLE001
                    row["p_model"] = None
            row["seconds"] = round(time.monotonic() - t0, 2)
            p = journal_path(d)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            st["asked"] += 1
            st["answered"] += int(_answered(row))
            st["refused"] += int(bool(row.get("refused")))
            _write_state(d, st)
            ib.sleep(max(1.0, float(settings.refusal_probe_spacing_seconds) - (time.monotonic() - t0)))
    except Exception as e:                                     # noqa: BLE001
        st["stopped"] = f"failed: {type(e).__name__}: {e}"[:300]
        logger.warning(f"[refusal_probe] {st['stopped']}")
    finally:
        try:
            ib.disconnect()
        except Exception:                                      # noqa: BLE001
            pass
        st["ended"] = datetime.now(ET).isoformat(timespec="seconds")
        st["stopped"] = st["stopped"] or "the list ended"
        _write_state(d, st)
        lock.unlink(missing_ok=True)
    logger.info(f"[refusal_probe] {d}: {st['asked']} asked, {st['refused']} refused — {st['stopped']}")
    return st


def _read(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def set_queue(names: Sequence[str]) -> Path:
    """The research queue (asked first every evening until each name is answered)."""
    p = queue_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"written": datetime.now(ET).isoformat(timespec="seconds"),
                             "names": [str(t).upper() for t in names]}, indent=1), encoding="utf-8")
    return p


def import_answers(rows: Sequence[dict], source: str) -> Tuple[Path, int]:
    """A study's answers into ``import_<source>.jsonl`` (rewritten whole): IBKR's answers the probe counts as asked."""
    keep = [{"ticker": str(r["ticker"]).upper(), "at": r.get("at"), "refused": r.get("refused"),
             "init_rate": r.get("init_rate"), "maint_rate": r.get("maint_rate"), "source": source}
            for r in rows if r.get("ticker") and _answered(r) and _at(r) is not None]
    p = probes_dir() / f"import_{source}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    os.replace(tmp, p)
    return p, len(keep)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="IBKR's refusals, asked every evening (PREREG191).")
    ap.add_argument("--day", default=None)
    ap.add_argument("--plan", action="store_true", help="print the evening's plan, ask nothing")
    ap.add_argument("--queue", default=None, help="set the research queue from a JSON file's order / names")
    ap.add_argument("--import", dest="imp", default=None, help="a study's answers (JSON lines)")
    ap.add_argument("--source", default=None, help="the imported answers' source name")
    a = ap.parse_args(argv)
    if a.queue:
        js = json.loads(Path(a.queue).read_text(encoding="utf-8"))
        names = js.get("order") or js.get("names") or []
        print(f"queue: {len(names)} names -> {set_queue(names)}")
        return 0
    if a.imp:
        if not a.source or not a.source.replace("_", "").isalnum():
            ap.error("--import needs --source NAME (letters, digits, _)")
        p, n = import_answers(_read(Path(a.imp)), a.source)
        print(f"imported {n} answers -> {p}")
        return 0
    d = date.fromisoformat(a.day) if a.day else datetime.now(ET).date()
    if a.plan:
        p = plan(d)
        by = {}
        for _, s in p:
            by[s] = by.get(s, 0) + 1
        print(json.dumps({"day": d.isoformat(), "names": len(p), "by_source": by, "first": p[:20]}, indent=1))
        return 0
    from src.log_redaction import redaction_filter
    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=redaction_filter)
    st = run(d)
    print(json.dumps(st, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
