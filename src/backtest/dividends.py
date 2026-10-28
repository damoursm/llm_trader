"""THE DIVIDENDS a real account pays and receives, in the simulator — AUDIT151's M1 (``Mimic(dividends=True)``, the
default from 2026-10-19: the user, "Yes please do 1 and 2"; the ledger does the same, `src/performance/dividends.py`).

For each piece (one trade) and each ex-date with the position held through the close before it — entered before
20:00 ET of the session before the ex-date (IBKR dates an execution at/after 20:00 to the next session) and exited
at/after that instant — a SHORT pays the dividend in lieu and a LONG receives it: the piece's exit price ``x`` and
every mark ``pc`` / ``ph`` / ``pl`` from that instant on are raised by the cash per share (for a short the liability
grows by what it owes; for a long the value by what it is owed). The cash is Polygon's record (the deep store's
``dividends`` family, as declared: in the terms of its own date, USD, summed per ex-date), brought into the piece's
price terms by the effective splits executed after the ex-date (`split_checks.json`'s false records dropped) and
checked against the piece's own close before the ex-date vs Polygon's UNADJUSTED close that day (the research
grouped daily bars): a ratio off the split table by more than 8% takes the ratio itself.

THE TABLE is AUDIT151's, frozen (`TABLE`: built 2026-10-18 from the deep store, every USD record 2017-06 .. 2026-12;
`build_table` rebuilds one); with it the default reproduces AUDIT151's ALLW runs. Measured: +0.18 +- 0.15 pp a year on
the deployed configuration (the longs' dividends outweigh the shorts' payments; the vol and fall arms short non-payers).
"""
from __future__ import annotations

import json
import os
import pickle
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings

ET = "America/New_York"
SHORT = {"vol", "etf", "fall"}
LONG = {"dip", "crash"}
# PREREG227's S4 (`Mimic.div_all`, the default from 2026-10-27): every live long book's long receives its dividends —
# production's ledger pays every live book (`src/performance/dividends.py`); `LONG` is AUDIT151's construction
ALL_LONG = {"dip", "crash", "cap", "ins", "fear"}
TABLE = Path("studies") / "audit151" / "dividends" / "div_table.pkl"       # under settings.backtest_research_dir
GROUPED = Path("studies") / "prereg68" / "grouped_raw"
_T: Dict[str, object] = {}
_RAW: Dict[str, dict] = {}


def _research() -> Path:
    return Path(settings.backtest_research_dir)


def table(path: Optional[Path] = None) -> dict:
    """{plain symbol: {"ex", "cash", "start_ns", "prev", "s"}} — loaded once per file."""
    p = Path(path) if path is not None else _research() / TABLE
    key = str(p)
    if _T.get("key") != key:
        _T.update(key=key, table=pickle.load(open(p, "rb")))
    return _T["table"]                                                  # type: ignore[return-value]


def raw_close(day: str, sym: str) -> Optional[float]:
    """Polygon's unadjusted close of ``sym`` on ``day`` (the research grouped daily bars), None when absent."""
    if day not in _RAW:
        f = _research() / GROUPED / f"{day}.pkl"
        if f.exists():
            g = pickle.load(open(f, "rb"))
            _RAW[day] = dict(zip(g["sym"], g["c"].astype(float)))
        else:
            _RAW[day] = {}
    return _RAW[day].get(sym)


def psym(tkn) -> str:
    return str(tkn).split("@")[0].replace("-", ".")


def _close_before(p: dict, prev: str) -> Optional[float]:
    """The piece's last mark at or before 16:00 ET of ``prev`` (its regular close), else None."""
    pt, pc = p.get("pt"), p.get("pc")
    if pt is None or pc is None or len(pt) == 0:
        return None
    lim = pd.Timestamp(prev + " 16:00", tz=ET).tz_convert("UTC").value
    lo = pd.Timestamp(prev + " 09:30", tz=ET).tz_convert("UTC").value
    pt = np.asarray(pt).astype("int64")
    i = int(np.searchsorted(pt, lim, side="right")) - 1
    if i < 0 or pt[i] < lo:
        return None
    return float(np.asarray(pc)[i])


def adjust(p: dict, log: Optional[list] = None, longs=None) -> dict:
    """The piece with the dividends it owes (short) or receives (long) inside its hold — or ``p`` itself. ``longs``:
    the long books paid (None = `LONG`, the dip and crash books — AUDIT151's; PREREG227's S4 pays `ALL_LONG`, every
    live long book, as production's ledger does)."""
    side = "short" if p.get("strategy") in SHORT else ("long" if p.get("strategy") in (LONG if longs is None else longs)
                                                       else None)
    if side is None:
        return p
    t = table().get(psym(p["tkn"]))
    if not t:
        return p
    ens, xns = int(p["ens"]), int(p["xns"])
    st = t["start_ns"]
    lo = bisect_right(st, ens)                         # start_ns > ens (entered before 20:00 of the session before)
    hi = bisect_right(st, xns)                         # start_ns <= xns (still held at that instant)
    if lo >= hi:
        return p
    events = []
    for i in range(lo, hi):
        cash, sfac, prev = t["cash"][i], t["s"][i], t["prev"][i]
        mine = _close_before(p, prev)
        raw = raw_close(prev, psym(p["tkn"]))
        fac, flag = sfac, "split_table"
        if mine and raw:
            r = mine / raw
            if abs(r / sfac - 1.0) <= 0.08:
                fac, flag = sfac, "ok"
            elif abs(r - 1.0) <= 0.08:
                fac, flag = 1.0, "store_unadjusted_for_later_split"
            else:
                fac, flag = r, "price_ratio"
        elif not raw:
            flag = "no_raw_close"
        else:
            flag = "no_piece_close"
        events.append((int(st[i]), cash * fac))
        if log is not None:
            log.append({"tkn": p["tkn"], "strategy": p["strategy"], "ex": t["ex"][i], "cash": cash, "fac": fac,
                        "flag": flag, "d_piece": cash * fac, "e": float(p["e"]), "frac": cash * fac / float(p["e"]),
                        "ens": ens, "xns": xns, "oid": p.get("oid")})
    q = dict(p)
    tot = sum(d for _, d in events)
    q["x"] = float(p["x"]) + tot
    pt = p.get("pt")
    if pt is not None and len(pt):
        pt = np.asarray(pt).astype("int64")
        add = np.zeros(len(pt), dtype=float)
        for s_ns, d in events:
            add[pt >= s_ns] += d
        for key in ("pc", "ph", "pl"):
            if p.get(key) is not None:
                arr = np.asarray(p[key], dtype=float)
                q[key] = (arr + add).astype(arr.dtype if arr.dtype.kind == "f" else float)
                if isinstance(p[key], list):
                    q[key] = q[key].tolist()
    q["_div"] = tot
    return q


def build_table(out: Path, divs: Path = Path("cache/ml/deep/dividends.parquet"),
                splits: Path = Path("cache/ml/deep/splits.parquet"),
                checks: Path = Path("cache/ml/deep/split_checks.json")) -> dict:
    """A new table in `TABLE`'s format from the deep store as it stands (AUDIT151's construction): every USD record
    2017-06 .. 2026-12 on a session of the research grouped bars, summed per ex-date, its 20:00 ET entitlement
    instant and the split factor into today's terms. Written to ``out`` (never over AUDIT151's frozen table)."""
    import duckdb
    sess = sorted(f[:-4] for f in os.listdir(_research() / GROUPED) if f.endswith(".pkl"))
    extra = pd.bdate_range(sess[-1], "2026-12-31").strftime("%Y-%m-%d").tolist()
    sess = sess + [d for d in extra if d > sess[-1]]
    con = duckdb.connect()
    dv = con.execute(f"""select ticker, ex_dividend_date ex, cash_amount cash, dividend_type typ
                         from read_parquet('{Path(divs).as_posix()}')
                         where currency = 'USD' and cash_amount > 0
                           and ex_dividend_date >= '2017-06-01' and ex_dividend_date <= '2026-12-31'""").df()
    dv = dv.drop_duplicates(["ticker", "ex", "typ", "cash"]).groupby(["ticker", "ex"], as_index=False)["cash"].sum()
    sp = con.execute(f"select ticker, execution_date ex, split_from, split_to from "
                     f"read_parquet('{Path(splits).as_posix()}')").df()
    chk = json.loads(Path(checks).read_text(encoding="utf-8"))
    sp = sp[[(chk.get(f"{t}|{e}") or {}).get("status") != "false" for t, e in zip(sp["ticker"], sp["ex"])]]
    sp = sp[(sp["split_from"] > 0) & (sp["split_to"] > 0)]
    spl = {t: list(zip(g["ex"], g["split_from"] / g["split_to"])) for t, g in sp.groupby("ticker")}
    tab: dict = {}
    for t, g in dv.groupby("ticker"):
        exs, cash, start, prev, sfac = [], [], [], [], []
        for ex, c in zip(g.sort_values("ex")["ex"], g.sort_values("ex")["cash"]):
            j = bisect_left(sess, ex)
            if j == 0 or j >= len(sess) or sess[j] != ex:
                continue
            pv = sess[j - 1]
            f = 1.0
            for e2, r in spl.get(t, ()):
                if e2 > ex:
                    f *= float(r)
            exs.append(ex); cash.append(float(c)); prev.append(pv); sfac.append(f)
            start.append(int(pd.Timestamp(pv + " 20:00", tz=ET).tz_convert("UTC").value))
        if exs:
            tab[t] = {"ex": exs, "cash": cash, "start_ns": start, "prev": prev, "s": sfac}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(tab, open(out, "wb"))
    return tab
