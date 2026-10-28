"""THE LEGS' REAL QUOTED SPREADS in the simulator — AUDIT151's C_QUOTE (``Mimic(quote_costs=True)``, the default from
2026-10-19: the user, "Yes please do 1 and 2").

Every market leg pays Polygon's quoted half-spread at its instant — the last NBBO at or before it within 10 minutes, a
09:30:00 leg read at 09:30:30, when production's order reaches IBKR — capped at 400 bp (the broker's spread-aware
regular-hours cap), and IBKR's exact commission / SEC fee / TAF on its shares (`Strategy.fees="engine"`): the short
arms' entries and exits, the crash book's legs, the dip book's exits and its open-fill entries (an order filled by a
later touch of its resting limit pays none). The ledger's liquidity-bucket averages it replaces (~23 bp all-in for a
volatile name's leg) were calibrated on calm names' fills: the real quotes read 51 / 83 bp at the vol arm's entries /
exits (means; AUDIT151, -4.32 +- 0.19 pp a year on the deployed configuration). The marks stay net of the ledger's own
exit estimate (``hs_mark``), as production marks its open trades.

THE STORE: AUDIT151's 2,418 legs (`QUOTES151`, frozen: every leg the deployed configuration traded in its 60 draws) +
the legs fetched since (`QUOTES_EXT`, `fetch`). A leg the store lacks pays its arm's AUDIT151 median (`QuoteTable.med`,
the measured patch's own fallback, frozen) and is flagged on its piece (``q_fb``: 1 the entry, 2 the exit) — a study
counts the TRADED legs that fell back (`traded_misses`) and fetches them (`python -m src.backtest.quotes --fetch
legs.csv`) before it decides.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import settings

ET = "America/New_York"
QUOTES151 = Path("studies") / "audit151" / "quote_costs" / "quotes.csv"       # under settings.backtest_research_dir
QUOTES_EXT = Path("studies") / "audit151" / "quote_costs" / "quotes_ext.csv"
CACHE = Path("studies") / "audit151" / "quote_costs" / "q"
CAP_BP = 400.0
MIN_NS = 60 * 10 ** 9
_TABLES: Dict[str, "QuoteTable"] = {}


class QuoteTable:
    """{(name, leg instant ns): half-spread bp} and each (arm, "in" / "out")'s AUDIT151 median."""

    def __init__(self, research: Optional[str] = None):
        root = Path(research or settings.backtest_research_dir)
        Q = pd.read_csv(root / QUOTES151)
        ok = Q[Q.hs_bp.notna()]
        self.tab = {(str(t), int(n)): float(h) for t, n, h in zip(ok.tkn, ok.ns, ok.hs_bp)}
        self.med = {(a, l): float(g.hs_bp.median()) for (a, l), g in ok.groupby(["arm", "leg"])}
        self.ext = 0
        ext = root / QUOTES_EXT
        if ext.exists():
            E = pd.read_csv(ext)
            E = E[E.hs_bp.notna()]
            for t, n, h in zip(E.tkn, E.ns, E.hs_bp):
                if (str(t), int(n)) not in self.tab:
                    self.tab[(str(t), int(n))] = float(h)
                    self.ext += 1

    def _get(self, arm, tk: str, ns: int, leg: str, default: float) -> Tuple[float, bool]:
        v = self.tab.get((tk, ns))
        if v is not None:
            return v, False
        return self.med.get((arm, leg), default), True

    def apply(self, q: dict, p: dict, side: int, legs: bool, limit_arms: Sequence[str] = ("dip",),
              market_arms: Sequence[str] = ("crash",)) -> dict:
        """``q`` (the piece with the ledger's costs) with its market legs at their real half-spreads: a short's
        legs (``legs``) and the ``market_arms``' (the crash book's; PREREG227's S5 adds the insider book's buys and
        sells at the open), the ``limit_arms``' exit and open-fill entry (``ens`` 09:30 + 2 ns; a later touch rests: 0)
        — the dip book's, and with S5 the capitulation and fear books' (a leg the store lacks: the dip book's AUDIT151
        median)."""
        arm, tk = p.get("strategy"), str(p["tkn"])
        ens, xns = int(p["ens"]), int(p["xns"])
        fb = 0
        if side < 0 and legs or arm in market_arms:
            h_in, m_in = self._get(arm, tk, ens, "in", float(q["hs_in"]))
            h_out, m_out = self._get(arm, tk, xns, "out", float(q["hs_out"]))
            q["hs_in"], q["hs_out"] = min(CAP_BP, h_in), min(CAP_BP, h_out)
            fb = int(m_in) | 2 * int(m_out)
        elif arm in limit_arms and int(p.get("ord_k", 0)) >= 0:
            m_in = False
            if ens % MIN_NS == 2:
                h_in, m_in = self._get("dip", tk, ens, "in", 1.0)
                q["hs_in"] = min(CAP_BP, h_in)
            else:
                q["hs_in"] = 0.0
            h_out, m_out = self._get("dip", tk, xns, "out", 1.0)
            q["hs_out"] = min(CAP_BP, h_out)
            fb = int(m_in) | 2 * int(m_out)
        if fb:
            q["q_fb"], q["q_ens"], q["q_xns"] = fb, ens, xns      # the store's keys, before any later repricing
        return q


def table(research: Optional[str] = None) -> QuoteTable:
    """The store, loaded once per process (per research root and extension file version)."""
    root = Path(research or settings.backtest_research_dir)
    ext = root / QUOTES_EXT
    key = f"{root}|{ext.stat().st_mtime_ns if ext.exists() else 0}"
    if key not in _TABLES:
        _TABLES.clear()
        _TABLES[key] = QuoteTable(str(root))
    return _TABLES[key]


def traded_misses(P: Sequence[dict], positions: Sequence[dict]) -> List[dict]:
    """The legs of the TRADED pieces that fell back to their arm's median: positions matched to the pieces ``P`` the
    account ran by (strategy, name, entry instant), as `run_set` matches them. Each: arm, tkn, leg, ns (the store's
    key: the piece's own ``ens`` / ``xns`` before any later repricing — kept in ``q_ens`` / ``q_xns``)."""
    by = {(p.get("strategy"), p.get("tkn"), int(p["ens"])): p for p in P if p.get("q_fb")}
    out, seen = [], set()
    for x in positions:
        p = by.get((x.get("strategy"), x.get("tkn"), int(x.get("entry_ns") or -1)))
        if p is None:
            continue
        for bit, leg, key in ((1, "in", "q_ens"), (2, "out", "q_xns")):
            if int(p["q_fb"]) & bit:
                ns = int(p.get(key, p["ens"] if leg == "in" else p["xns"]))
                k = (str(p["tkn"]), ns)
                if k not in seen:
                    seen.add(k)
                    out.append({"arm": p.get("strategy"), "tkn": str(p["tkn"]), "leg": leg, "ns": ns})
    return out


def _read_at(ns: int) -> pd.Timestamp:
    t = pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET)
    if t.hour == 9 and t.minute == 30 and t.second == 0:
        t = t + pd.Timedelta(seconds=30)          # production's order reaches IBKR ~27-30 s after the open
    return t


def fetch(legs: Sequence[dict], research: Optional[str] = None, pause: float = 0.15) -> int:
    """Polygon's NBBO at each leg (`quote_costs/fetch_quotes.py`'s request, cached per leg) appended to `QUOTES_EXT`;
    legs the store already holds are skipped. Returns the number added."""
    from src.data import polygon_client as PC
    root = Path(research or settings.backtest_research_dir)
    have = set(QuoteTable(str(root)).tab)
    ext = root / QUOTES_EXT
    old = pd.read_csv(ext) if ext.exists() else pd.DataFrame(columns=["arm", "tkn", "leg", "ns", "at", "hs_bp"])
    have |= {(str(t), int(n)) for t, n in zip(old.tkn, old.ns)}
    cdir = root / CACHE
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in legs:
        key = (str(g["tkn"]), int(g["ns"]))
        if key in have:
            continue
        have.add(key)
        at = _read_at(int(g["ns"]))
        sym = PC.to_polygon_symbol(str(g["tkn"]).split("@")[0])
        f = cdir / f"{sym}_{int(at.value)}.json"
        if f.exists():
            js = json.loads(f.read_text())
        else:
            time.sleep(pause)
            js = PC._get(f"/v3/quotes/{sym}", {"timestamp.lte": int(at.value),
                                                "timestamp.gte": int(at.value) - 600 * 10 ** 9,
                                                "order": "desc", "sort": "timestamp", "limit": 1}) or {}
            f.write_text(json.dumps(js))
        res = js.get("results") or []
        hs = None
        if res:
            b, a = float(res[0].get("bid_price") or 0), float(res[0].get("ask_price") or 0)
            if a > 0 and b > 0 and a >= b:
                hs = 1e4 * (a - b) / (a + b)
        rows.append({"arm": g.get("arm"), "tkn": key[0], "leg": g.get("leg"), "ns": key[1], "at": str(at), "hs_bp": hs})
    if rows:
        add = pd.DataFrame(rows)
        new = pd.concat([old, add], ignore_index=True) if len(old) else add
        tmp = ext.with_suffix(".tmp")
        new.to_csv(tmp, index=False)
        tmp.replace(ext)
        _TABLES.clear()
    return len(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the simulator's quote store (AUDIT151's C_QUOTE)")
    ap.add_argument("--fetch", help="a CSV of legs (arm, tkn, leg, ns) to add from Polygon's NBBO")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    if a.fetch:
        L = pd.read_csv(a.fetch)
        print(json.dumps({"asked": len(L), "added": fetch(L.to_dict("records"))}))
        return 0
    t = table()
    print(json.dumps({"legs": len(t.tab), "extension": t.ext, "medians": {f"{k[0]}|{k[1]}": v for k, v in t.med.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
