"""THE FEAR BOOK as the account engine's pieces — production's `src/signals/fear_long.py` (from the 2026-10-28 session;
PREREG219-226, the research construction `llm_trader_research/studies/prereg221/h221.py` / `prereg224/h224.py`, which
these pieces reproduce: PREREG225's E2s4).

Every name of the dip panel that closed ABOVE its own 200-session average with a 2-session RSI under 10 trading
`fear_long_min_dollar_volume` ($1B) or more a day — the dip book's signal — whose implied volatility at the signal
close S sits at or above the `fear_long_ivp_min` percentile of its own `fear_long_ivp_window` sessions
(`fear_long.ivp_at` on IBKR's history in `fear_long_iv_dir`; a name without a series is not FEAR; set A0's symbols read
today's tickers through `mimic._a0map`) is the capitulation book's limit order (`strategies.dip.order_pieces` with
this book's signals: the limit at `fear_long_limit_atr` x the 14-session ATR under the close, alive
`fear_long_limit_sessions` sessions, the exit after the first close above the `fear_long_exit_sma_sessions` average or
`fear_long_max_hold_sessions` sessions), every signal an order, IVP highest first (the pieces' ``rank`` = -IVP + RSI2 x
1e-5). `mimic.live_books` drops the orders the dip book itself places that day (its daily top pick) and treats the
pieces as the capitulation book's (the extended-hours marks, the limit fills' costs; the open fills at the print).

The account (`mimic.live_strategies`): 1/`fear_long_account_slices` of the equity, at most `fear_long_max_open` open,
its working orders counted, capital quality 0.04 (`mimic.QUALITY`: it gives way to every other book), preemptible
whole, the latest first, de-levered first (``delever_order`` 0), ONE BOOK WITH THE DIP BOOK for one book per name
(`mimic.BOOK["fear"]` = the dip book's).

THE STORE IT READS (P7, the parity audit of 2026-10-27): production's store by default — whose pre-market fetch rewrites
a month of bars each session, journaling every bar it changes (`fear_long.merge_rows`) — or, inside `freeze`, ONE
FROZEN COPY of it through a date (`snapshot`: the store's bars to that date, named by their digest, written once under
`snapshot_root()` in the store's own format — ``IV_ROOT`` points the simulator at it). A card build reads one
(`reliability.build`), the cards, the live calibration's reference and the bank record its state (``iv_store``), and a
study that saves draws for the bank records the one it read beside them (`write_state`), so a fetch merging IBKR's bars
during a build moves nothing and a bank built on another store says so (`bank.iv_stores`, `diff`).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import settings
from src.backtest.strategies import dip as DIP

IV_ROOT: Optional[str] = None       # the store the simulator reads: a frozen snapshot (`freeze`); None = production's
STATE_FILE = "iv_store.json"        # a snapshot's state, inside it and beside what read it (a card build, a study's runs)


def iv_root() -> Path:
    """The implied-volatility store the simulator reads: `IV_ROOT` (a frozen snapshot) or production's store."""
    from src.signals import fear_long as FL
    return Path(IV_ROOT) if IV_ROOT else FL.iv_root()


def snapshot_root() -> Path:
    """Where the frozen snapshots are kept: the bank's ``iv_store/<digest>/`` (never delete: a bank's input)."""
    from src.backtest import bank as BK
    return Path(BK.BANK_DIR) / "iv_store"


def default_through() -> str:
    """The pooled history's last decision session (the bank's span): no card reads a later bar."""
    from src.backtest import periods
    return periods.pool_span(periods.POOLED)[1]


def store_rows(through=None, root: Optional[Path] = None) -> Dict[str, list]:
    """Every name's stored bars (`fear_long.bar_row`) dated through ``through`` (None: all) in the store ``root``
    (None: the one the simulator reads, `iv_root`): {name: rows}, a name without a bar left out."""
    from src.signals import fear_long as FL
    root = iv_root() if root is None else Path(root)
    hi = None if through is None else str(pd.Timestamp(through).date())
    out: Dict[str, list] = {}
    for p in sorted(root.glob("*.json.gz")):
        sym = p.name[: -len(".json.gz")]
        rows = [FL.bar_row(r) for r in FL.load_rows(sym, root) if hi is None or str(r[0])[:10] <= hi]
        if rows:
            out[sym] = rows
    return out


def state(rows: Dict[str, list], through) -> dict:
    """A store's state: the digest of every name's bars (its rows as stored, names sorted), the names and the bars."""
    h = hashlib.sha1()
    for sym in sorted(rows):
        h.update(f"{sym}\0".encode("utf-8"))
        h.update(json.dumps(rows[sym], separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return {"digest": h.hexdigest()[:12], "through": None if through is None else str(pd.Timestamp(through).date()),
            "names": len(rows), "bars": int(sum(len(v) for v in rows.values()))}


def snapshot(through=None, dest: Optional[Path] = None, root: Optional[Path] = None) -> dict:
    """The store the simulator reads (``root``: another one) frozen through ``through`` (default `default_through`):
    its `state`, the copy written ONCE under ``dest``/<digest>/ (default `snapshot_root()`) in the store's own format
    with the state beside (`STATE_FILE`). Returns the state with its ``path`` and ``source``."""
    through = through or default_through()
    src = iv_root() if root is None else Path(root)
    rows = store_rows(through, src)
    st = state(rows, through)
    base = Path(dest) if dest is not None else snapshot_root()
    path = base / st["digest"]
    if not (path / STATE_FILE).exists():
        tmp = base / f"{st['digest']}.{os.getpid()}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for sym, rs in rows.items():
            (tmp / f"{sym}.json.gz").write_bytes(gzip.compress(json.dumps(rs).encode("utf-8")))
        rec = dict(st, source=str(src), taken=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (tmp / STATE_FILE).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if path.exists():                              # a copy left incomplete: replaced
            shutil.rmtree(path, ignore_errors=True)
        try:
            os.replace(tmp, path)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not (path / STATE_FILE).exists():       # another process wrote the same copy meanwhile: kept
                raise
    return dict(st, path=str(path), source=str(src))


@contextmanager
def freeze(through=None) -> Iterator[dict]:
    """Inside the block the simulator reads ONE frozen copy of the store (`snapshot` through ``through``): a fetch
    merging IBKR's bars meanwhile moves nothing. Yields the snapshot's state."""
    global IV_ROOT
    st = snapshot(through)
    saved, IV_ROOT = IV_ROOT, st["path"]
    try:
        yield st
    finally:
        IV_ROOT = saved


def write_state(folder, st: dict) -> Path:
    """Record the store that what ``folder`` holds read (a study's runs, a card build): `STATE_FILE` there."""
    p = Path(folder) / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return p


def read_state(folder) -> Optional[dict]:
    try:
        return json.loads((Path(folder) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def diff(a, b, through=None) -> dict:
    """What changed from the store ``a`` to the store ``b`` (each a snapshot or production's store) in the bars
    through ``through`` (None: all): the names added and removed, and per name held by both the bars added, removed
    and revised (`fear_long.differs`)."""
    from src.signals import fear_long as FL
    ra, rb = store_rows(through, Path(a)), store_rows(through, Path(b))
    out = {"names_added": sorted(set(rb) - set(ra)), "names_removed": sorted(set(ra) - set(rb)),
           "bars_added": 0, "bars_removed": 0, "bars_revised": 0, "names_changed": {}}
    for sym in sorted(set(ra) & set(rb)):
        da, db = {r[0]: r for r in ra[sym]}, {r[0]: r for r in rb[sym]}
        ch = {"added": len(set(db) - set(da)), "removed": len(set(da) - set(db)),
              "revised": sum(1 for k in set(da) & set(db) if FL.differs(da[k], db[k]))}
        if any(ch.values()):
            out["names_changed"][sym] = ch
            out["bars_added"] += ch["added"]
            out["bars_removed"] += ch["removed"]
            out["bars_revised"] += ch["revised"]
    return out


def params() -> DIP.DipParams:
    """The book's rule as `DipParams` (production's settings: the dip book's signal — RSI2 under `dip_long_rsi_max`, a
    `dip_long_trend_sessions` trend — at `fear_long_min_dollar_volume`, the fear book's exit)."""
    return DIP.DipParams(rsi_max=float(settings.dip_long_rsi_max), min_dv=float(settings.fear_long_min_dollar_volume),
                         trend=int(settings.dip_long_trend_sessions), exit_sma=int(settings.fear_long_exit_sma_sessions),
                         max_hold=int(settings.fear_long_max_hold_sessions))


class IVP:
    """IVP at a signal session S for a panel name (None = unknown): production's percentile on the store the simulator
    reads (`iv_root`: production's, or the frozen snapshot of `freeze`; ``root`` another); a delisted ``@D`` name has
    none; set A0's symbols through `mimic._a0map`."""

    def __init__(self, root: Optional[Path] = None):
        from src.signals import fear_long as FL
        self.FL = FL
        self.root = iv_root() if root is None else Path(root)
        self.cache: Dict[str, Optional[pd.Series]] = {}
        self._a0 = None

    def name(self, tk: str, a0: bool) -> Optional[str]:
        if str(tk).endswith("@D"):
            return None
        if a0:
            if self._a0 is None:
                from src.backtest import mimic as M
                self._a0 = M._a0map()
            tk = self._a0.get(tk, tk)
        return tk if self.FL.iv_path(tk, self.root).exists() else None

    def at(self, tk: str, S, a0: bool) -> Optional[float]:
        t = self.name(tk, a0)
        if t is None:
            return None
        if t not in self.cache:
            self.cache[t] = self.FL.iv_series(t, self.FL.load_rows(t, self.root))
        v = self.FL.ivp_at(self.cache[t], pd.Timestamp(S))
        return None if v is None else float(v[0])


def rank_of(ivp: Optional[float], rsi2: float) -> float:
    return (-float(ivp) + float(rsi2) * 1e-5) if ivp is not None else (10.0 + float(rsi2))


def signals_fn(iv: IVP, a0: bool, store: dict):
    """The book's signals in `strategies.dip.signals`' form: the dip rule (above its trend), FEAR at the signal
    session only; ``store`` keeps the IVP of each order id (``ivp``) and the counts."""
    from src.signals.dip_long import rsi2
    cut = float(settings.fear_long_ivp_min)

    def signals(panel: Dict[str, pd.DataFrame], prm, lo, hi, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        data_last = max(v.index[-1] for v in panel.values() if len(v))
        rows = []
        for tk in names if names is not None else DIP.eligible_names(panel):
            d = panel.get(tk)
            if d is None or len(d) < DIP.MIN_BARS:
                continue
            c = d["c"].to_numpy(float)
            n = len(c)
            s = pd.Series(c)
            sma_t = s.rolling(int(prm.trend)).mean().to_numpy()
            sma_x = s.rolling(int(prm.exit_sma)).mean().to_numpy()
            dv20 = (d["c"] * d["v"]).rolling(20).mean().to_numpy()
            r2 = rsi2(c)
            dates = d.index
            with np.errstate(invalid="ignore"):
                sig = (c > sma_t) & (r2 < float(prm.rsi_max)) & (dv20 >= float(prm.min_dv)) & (dates >= lo) & (dates <= hi)
            sig[:DIP.MIN_BARS - 1] = False
            sig[n - 1:] = False
            ended = dates[-1] < data_last - pd.Timedelta(days=7)
            for i in np.flatnonzero(sig):
                j = i + 1
                x = next((k for k in range(j, min(n, i + 1 + int(prm.max_hold))) if DIP.exit_hit(prm, c, sma_x, r2, k)),
                         min(n - 1, i + int(prm.max_hold)))
                last_close = False
                if x + 1 >= n:
                    if not ended and not DIP.OPEN_END:          # PREREG227's S8: set D keeps it, as the dip book's
                        continue
                    xd, last_close = n - 1, True
                else:
                    xd = x + 1
                if (dates[xd] - dates[j]).days > DIP.gap_days(prm):
                    continue
                store["cand"] = store.get("cand", 0) + 1
                v = iv.at(tk, dates[i], a0)
                if v is None:
                    store["unknown"] = store.get("unknown", 0) + 1
                    continue
                if v < cut:
                    continue
                store["fear"] = store.get("fear", 0) + 1
                store.setdefault("ivp", {})[f"{tk}|{dates[j].date()}"] = v
                rows.append((tk, dates[i], dates[j], dates[xd], float(r2[i]), float(dv20[i]), float(c[i]), last_close))
        return pd.DataFrame(rows, columns=["tk", "sig", "d_in", "d_out", "rsi2", "dv20", "close", "last_close"])
    return signals


def pieces(lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, period: Optional[str] = None,
           strategy: str = "fear", log: Optional[dict] = None) -> List[dict]:
    """The book's orders decided on sessions [lo, hi] as the dip book's limit-order pieces (every FEAR signal's
    order) on ``panel`` (None: set A0's own-security panel with ``period="A0"``, else `strategies.dip.build_panel()`),
    each with its ``ivp`` and ``rank``."""
    a0 = period == "A0"
    if panel is None:
        if a0:
            from src.backtest import a0 as A0
            panel = A0.dip_panel()
        else:
            panel = DIP.build_panel()
    store: dict = {}
    iv = IVP()
    P = DIP.order_pieces(params(), lo, hi, panel=panel, strategy=strategy,
                         atr_mult=float(settings.fear_long_limit_atr), sessions=int(settings.fear_long_limit_sessions),
                         keep=None, signals_fn=signals_fn(iv, a0, store))
    ivp = store.get("ivp") or {}
    for p in P:
        p["ivp"] = ivp[p["oid"]]
        p["rank"] = rank_of(p["ivp"], p["rsi2"])
    if log is not None:
        log.update({f"fear_{k}": v for k, v in store.items() if k != "ivp"})
        log["fear_iv_root"] = str(iv.root)
    return P
