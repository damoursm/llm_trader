"""SET A0 (PREREG101, registered 2026-10-14): the live combination on the years BEFORE the deep 30-minute store, with
IBKR's archived short-stock file (the community archive from 2017-10-09) — decisions 2018-01-02 .. 2021-01-29, built
exactly as set A on its own data (`periods.get("A0")`; reported beside A and B, never pooled).

The data (new files only; the existing stores are never written):

* the bars — every common stock / ADR and every exchange-traded product that cleared the live floors on its day
  (Polygon's whole-market daily bars, unadjusted; `<research>/studies/prereg101/fetch101.py`), their 30-minute bars
  of every session 2017-06-01 .. 2021-06-30 (the dip book's names from 2016-11-01, QQQ / TQQQ / AAPL / SPY from
  2016-01-04: the 200-session averages), split-adjusted as of the fetch, prices as Polygon serves them
  (`cache/ml/deep_a0/bars30m/parts/<SYM>.parquet`, the `bars30m_full` layout, float64);
* the names keep the research engine's classes (`CLASSES`): plain = a name of the research engine's deep store
  (`<research>/vol/bars`), ``@D`` a delisted name (since 2021 in `vol/bars_d`, or before 2021: new here), ``@X`` a
  common stock outside the deep store (`vol/bars_x`), ``@E`` an exchange-traded product (`vol/bars_e`, or new here);
  one engine name per symbol (a reused symbol's bars are the symbol's, stitched, as every existing store's);
* the research engine's arrays and stores rebuilt by the SAME code over the new period
  (`<research>/studies/prereg101/build101.py`): `vol/a0/bars*/<SYM>.npz` (`volengine.ticker_arrays`, each class in
  its store's precision), `vol/a0/panel_a0.npz`, `optvol/store_a0/` (`build.main`: candidates, freshness, run-up
  bases, deadlines, paths, Rule 201, the corporate-action gap flags; `mm/`), `optvol/a0_closes.pkl`,
  `optvol/a0_borrow{,_pick}.npz` (`prereg26_optuna_borrow.borrow_arrays` from 2017-10-09),
  `etf/vol_events_a0_arm.pkl` (the ETF arm's picks, `etf_vol.book`'s construction), the dip book's daily panel
  (`cache/backtest/dip_daily_a0.pkl`), IBKR's archived fees per name (`cache/ml/ib_shorting_archive/a0/`), the borrow
  pressure of every A0 session the way production builds it (`sel_pressure.build`, `.../deep_a0/sel_short/pressure/`)
  and V6's HIGH flag per store row (`optvol/store_a0/pressure_hi.npy`).

How the simulator uses it: `strategies.vol.VolEngine(period="A0")`, `strategies.etf.EtfArm(period="A0")`,
`dip_panel()` (the dip adapter reads the panel's own bars and names), `strategies.crash.pieces(..., period="A0")`,
`mimic.ExtBars(period="A0")`, `settle()`; `mimic.live_books` takes a window that starts before the deep store
(`periods.DATA_START`) from these components and refuses another period's (`check_components`). `harness(W)` swaps a
study harness's components (``R63._W``) for A0's during a block — then every context a study applies (V6's
`R75.Applied`, `grid97.applied`, ...) patches the A0 engine — and installs A0's V6 flags and halt files.

Conventions kept from set A (stated with every A0 result): lendability and fees from IBKR's archived file (fees:
IBKR's API bar where the name's API history covers the day, else the archive's file — set A's two sources); where
neither holds a fee the store's own fill (IBKR's 2026-09-26 file fee, else 9.75 %/yr: the fee model PREREG94 needs
inputs A0 lacks); FINRA days to cover by symbol (settlements from 2017-12-29: unknown — passes — before
2018-01-12); the 2021+ ledger costs, house rates and refusals unchanged.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.backtest import periods

REPO = Path(__file__).resolve().parents[2]
EPOCH = pd.Timestamp("1970-01-01")
ET = "America/New_York"
PERIOD = "A0"

# ── the spans ────────────────────────────────────────────────────────────────
LO, HI = periods.A0_LO, periods.A0_HI                  # the set's decisions
STORE_LO, STORE_HI = LO, "2021-04-30"                  # the candidate store: A0 + PREREG101's overlap (parity) window
OVERLAP = ("2021-02-01", "2021-04-30")                 # the build's validation window against the existing stores
FETCH_FROM = "2017-06-01"                              # every screened name: the 400-bar ATR floor, the 260-bar
LONG_FROM = "2016-11-01"                               # rvol, 20 sessions of freshness; the dip names: 220 daily bars
INDEX_FROM = "2016-01-04"                              # QQQ / TQQQ / AAPL / SPY: the 200-session averages, the calendar
DATA_END = "2021-06-30"                                # the last session fetched (the overlap's paths and deadlines)
STORE_START = "2017-06-02"                             # MINBARS: a name whose bars begin by then carries its history
SCREEN_FROM = "2017-12-01"                             # a name eligible on a day from here on is in the universe
INDEX_NAMES = ("AAPL", "QQQ", "TQQQ", "SPY")
CLASSES = ("", "@D", "@X", "@E")
ETP_TYPES = ("ETF", "ETN", "ETV", "ETS")
STOCK_TYPES = ("CS", "ADRC")
PATHS = ("PC", "PA", "PNS", "PSD", "PBD")
SECTIONS = ("deep_a0", "vol_a0", "store_a0")


# ── where the data lives ─────────────────────────────────────────────────────

def research() -> Path:
    return Path(settings.backtest_research_dir)


def deep_dir() -> Path:
    """Raw A0 data (Polygon bars, reference snapshots, short interest, Reg SHO, the universe)."""
    return REPO / "cache" / "ml" / "deep_a0"


def parts_dir() -> Path:
    """Every session's 30-minute bars per symbol (the `bars30m_full` layout, float64 prices)."""
    return deep_dir() / "bars30m" / "parts"


def vol_dir() -> Path:
    return research() / "vol" / "a0"


def bars_dir(cls: str) -> Path:
    return vol_dir() / {"": "bars", "@D": "bars_d", "@X": "bars_x", "@E": "bars_e"}[cls]


def panel_path() -> Path:
    return vol_dir() / "panel_a0.npz"


def panel_meta_path() -> Path:
    return vol_dir() / "panel_a0_meta.json"


def store_dir() -> Path:
    return research() / "optvol" / "store_a0"


def closes_path() -> Path:
    return research() / "optvol" / "a0_closes.pkl"


def borrow_path(at: str = "next") -> Path:
    return research() / "optvol" / ("a0_borrow_pick.npz" if at == "pick" else "a0_borrow.npz")


def etf_events_path() -> Path:
    return research() / "etf" / "vol_events_a0_arm.pkl"


def archive_fee_dir() -> Path:
    return REPO / "cache" / "ml" / "ib_shorting_archive" / "a0"


def dip_panel_path() -> Path:
    return REPO / "cache" / "backtest" / "dip_daily_a0.pkl"


def universe_path() -> Path:
    return deep_dir() / "universe.json"


def si_path() -> Path:
    return deep_dir() / "short_interest_by_symbol.parquet"


def regsho_path() -> Path:
    return deep_dir() / "regsho.parquet"


def halts_dir() -> Path:
    return REPO / "cache" / "ml" / "halts_a0"


def pressure_root() -> Path:
    """The A0 copy of production's pressure directory (`sel_pressure.root`): the model, the grouped bars, the scores."""
    return deep_dir() / "sel_short" / "pressure"


def pressure_hi_path() -> Path:
    return store_dir() / "pressure_hi.npy"


def pressure_score_path() -> Path:
    return store_dir() / "pressure_score.npy"


# ── names ────────────────────────────────────────────────────────────────────

def plain(name) -> str:
    """The traded symbol of an engine name (``XYZ@D`` -> ``XYZ``)."""
    return str(name).split("@")[0]


def cls_of(name) -> str:
    s = str(name)
    return s[-2:] if s.endswith(("@D", "@X", "@E")) else ""


def load_bars(name) -> Optional[dict]:
    """The research engine's arrays of an A0 engine name (`volengine.load_bars` over the A0 directories)."""
    p = bars_dir(cls_of(name)) / f"{plain(name)}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def is_a0(obj) -> bool:
    """The component belongs to set A0 (an engine, an ETF arm, a dip panel, extended bars, a harness)."""
    return getattr(obj, "period", None) == PERIOD


# ── calendars ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def a0_sessions() -> np.ndarray:
    """The sessions of A0's own data: the days AAPL traded in its A0 bars (2016-01-04 .. 2021-06-30)."""
    a = load_bars("AAPL")
    if a is None:
        raise FileNotFoundError(f"set A0's AAPL bars are not built ({bars_dir('')}): run build101.py")
    return np.unique(np.asarray(a["sday"], np.int64))


def _research_path() -> None:
    """The research engine on the path, as `VolEngine` puts it (its store tag set first: `build` reads it)."""
    root = str(research())
    os.environ.setdefault("OPTVOL_TAG", "_dx")
    for p in (os.path.join(root, "optvol"), os.path.join(root, "pylib")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import build  # noqa: F401  (puts the vol engine's own directory on the path)


def extend(head, tail) -> np.ndarray:
    """A calendar ``head`` (A0's days) continued by ``tail`` (a default calendar) after its last day."""
    s = np.asarray(head, np.int64)
    t = np.asarray(tail, np.int64)
    return np.r_[s, t[t > s[-1]]].astype(np.int64) if len(s) else t


@lru_cache(maxsize=1)
def calendar() -> np.ndarray:
    """`volengine.calendar`'s sessions extended back over A0: A0's AAPL sessions, then the research calendar after
    them (its AAPL sessions from 2021-01-04, the production calendar past its data). The ETF arm's calendar."""
    _research_path()
    import volengine as ve
    return extend(a0_sessions(), ve.calendar())


@lru_cache(maxsize=1)
def c4_sessions() -> np.ndarray:
    """`cap5k4.sessions` extended back over A0 (the vol engine's ``C4.CAL``: the run-up base, the deadlines, the turn
    and limit windows, the lendability draws' calendar)."""
    _research_path()
    import cap5k4 as C4
    return extend(a0_sessions(), C4.sessions())


@lru_cache(maxsize=1)
def settlement() -> np.ndarray:
    """The settlement days over A0 and after (`account.settlement_days` of A0's sessions + the default calendar's)."""
    from src.backtest import account as A
    A.default_settle()                                   # fills the default calendar (AAPL's deep store, weekdays after)
    return extend(A.settlement_days(a0_sessions()), A._CAL["s"])


def settle() -> Callable[[int], int]:
    """`account.default_settle` over A0's sessions: a trade's settlement day (T+2; T+1 from 2024-05-28)."""
    from src.backtest import account as A
    return A.make_settle(settlement())


def settles_a0(fn: Callable[[int], int]) -> bool:
    """``fn`` is a settlement function that knows A0's sessions (a trade of 2018-01-02 settles 2018-01-04)."""
    try:
        return int(fn(dnum("2018-01-02"))) == dnum("2018-01-04")
    except Exception:                                    # noqa: BLE001
        return False


def dnum(day) -> int:
    return int((pd.Timestamp(day) - EPOCH).days)


@contextmanager
def research_calendar():
    """`cap5k4.CAL` set to A0's sessions for a block (`cap5k4.settle` reads it), the previous value restored."""
    _research_path()
    import cap5k4 as C4
    prev = C4.CAL
    C4.CAL = c4_sessions()
    try:
        yield
    finally:
        C4.CAL = prev


# ── the vol engine's store ───────────────────────────────────────────────────

def mm_store():
    """`wf10.mm_store` over A0's store (the evaluator's `Store`: candidates, freshness, paths and Rule 201 memory
    mapped; the same spread model)."""
    _research_path()
    import build as BLD
    import evaluator as EV
    d = store_dir()
    st = object.__new__(EV.Store)
    st.C = pd.read_pickle(d / "cand.pkl")
    z = np.load(d / "fresh.npz")
    st.FM, st.PRE, st.LAST = z["FM"], z["PRE"], z["LAST"]
    st.fcol = {tuple(int(x) for x in c): i for i, c in enumerate(z["cols"])}
    for k in PATHS + ("SSR", "PH"):
        setattr(st, k, np.load(d / "mm" / f"{k}.npy", mmap_mode="r"))
    st.nb = {}
    m = json.load(open(os.path.join(BLD.OLD, "spread_model.json")))
    st.beta, st.smear = np.array(m["beta"]), m["smear"]
    return st


def load_borrow(at: str = "next") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IBKR's archived lendability per A0 store row (`wf10.load_borrow`'s layout), at the bar after the pick or at
    the pick bar's end."""
    z = np.load(borrow_path(at), allow_pickle=False)
    return z["lend"], z["lend_usd"], z["fee"], z["drop"]


def real_fees():
    """IBKR's own daily fee for A0's names: its API bar where the name's API history covers the day (set A's source
    for a listed name), else IBKR's archived file of that day for the A0 name (fenced by its A0 bar window — set A's
    source for a delisted name), else the fallback; each carried at most a week."""
    _research_path()
    import realfee as RF

    class RealFeesA0(RF.RealFees):
        def __init__(self):
            super().__init__(parts=RF.PARTS, field="fee", archive=str(archive_fee_dir()))

        def _api(self, tkn):
            p = plain(tkn)
            if p not in self.cache:
                f = os.path.join(self.parts, f"{p}.csv")
                if not os.path.exists(f):
                    self.cache[p] = None
                else:
                    D = pd.read_csv(f)
                    dn = np.array([dnum(str(x)[:10]) for x in D["date"]], np.int64)
                    o = np.argsort(dn, kind="mergesort")
                    self.cache[p] = (dn[o], D[self.field].to_numpy(float)[o])
            return self.cache[p]

        def _arch(self, tkn):
            key = "@A0:" + plain(tkn)
            if key not in self.cache:
                f = os.path.join(self.archive, f"{plain(tkn)}.csv")
                if not os.path.exists(f):
                    self.cache[key] = None
                else:
                    D = pd.read_csv(f)
                    dn = np.array([dnum(str(x)[:10]) for x in D["date"]], np.int64)
                    o = np.argsort(dn, kind="mergesort")
                    self.cache[key] = (dn[o], D["fee"].to_numpy(float)[o], int(dn[o][-1]) + RF.ARCHIVE_STALE)
            return self.cache[key]

        def series(self, tkn):
            return self._api(tkn) or self._arch(tkn)

        def daily(self, tkn, days, fallback):
            days = np.asarray(days, np.int64)
            out = np.full(len(days), np.nan)
            for s in (self._api(tkn), self._arch(tkn)):
                if s is None or not len(s[0]):
                    continue
                d, f = s[0], s[1]
                j = np.searchsorted(d, days, side="right") - 1
                ok = (j >= 0) & ~np.isfinite(out)
                if len(s) > 2:
                    ok &= days <= s[2]
                ok &= (days - d[np.maximum(j, 0)]) <= 7
                v = f[np.maximum(j, 0)]
                ok &= np.isfinite(v)
                out[ok] = v[ok]
            return np.where(np.isfinite(out), out, float(fallback))

        def covered(self, tkn, day):
            day = int(day)
            for s in (self._api(tkn), self._arch(tkn)):
                if s is not None and len(s[0]) and s[0][0] <= day and (len(s) < 3 or day <= s[2]):
                    return True
            return False

    return RealFeesA0()


def assembler(st, closes, rf, cap: int = 400_000):
    """`wf10.Assembler` with `cap5k4.settle` on A0's sessions (the borrow schedule's settlement days)."""
    _research_path()
    import wf10

    class AssemblerA0(wf10.Assembler):
        def pieces(self, ev, cfg):
            with research_calendar():
                return super().pieces(ev, cfg)
    return AssemblerA0(st, closes, rf, cap=cap)


def c4_proxy():
    """What the vol engine reads of `cap5k4` (``CAL``, ``settle``), over A0's sessions, without touching the module."""
    return SimpleNamespace(CAL=c4_sessions(), settle=settle())


def pressure_hi(n_rows: int) -> Optional[np.ndarray]:
    """V6's HIGH flag per A0 store row (production's pressure scores at each pick's session), None when not built."""
    p = pressure_hi_path()
    if not p.exists():
        return None
    v = np.load(p)
    if len(v) != int(n_rows):
        raise ValueError(f"{p}: {len(v)} flags for a store of {n_rows} rows — rebuild it (build101.py pressure)")
    return v.astype(bool)


# ── the dip book, the crash book, the extended bars ──────────────────────────

class DipPanel(dict):
    """The dip book's daily panel over A0 (`dip_long.daily_bars` of each name's A0 bars), with the bars the
    adapter's fills read (``bars_fn``) and its common stocks / ADRs (``eligible``: Polygon's type at the time)."""
    period = PERIOD

    def __init__(self, data: Dict[str, pd.DataFrame], eligible: Sequence[str]):
        super().__init__(data)
        self.eligible = sorted(eligible)

    @staticmethod
    def bars_fn(name: str) -> Optional[pd.DataFrame]:
        return bars_rth_frame(name)


def raw_volume_path() -> Path:
    return vol_dir() / "raw_volume.json"


@lru_cache(maxsize=1)
def raw_volume_names() -> frozenset:
    """The deep store's names whose stored volumes are Polygon's as served (fractional after a reverse split: the
    research engine's existing arrays carry fractions); every other deep-store name's are whole shares
    (`build101.py vbars` writes the list)."""
    p = raw_volume_path()
    return frozenset(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else frozenset()


def in_precision(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """A frame of A0 bars (float64 prices and raw volumes, as Polygon serves them) in the precision of the store
    its class comes from: whole-share volumes for ``@D`` and ``@X`` (the delisted and outside stores truncate them)
    and for a deep-store name unless its stored volumes are Polygon's (`raw_volume_names`), Polygon's volumes for
    ``@E``; float32 prices for ``@D`` (the delisted store's parquet)."""
    c = cls_of(name)
    cols = {k: k for k in df.columns}
    vcol = next((k for k in ("Volume", "volume") if k in cols), None)
    whole = c in ("@D", "@X") or (c == "" and plain(name) not in raw_volume_names())
    if vcol is not None and whole:
        df[vcol] = np.floor(pd.to_numeric(df[vcol], errors="coerce").fillna(0.0).to_numpy(float))
    if c == "@D":
        for k in ("Open", "High", "Low", "Close", "open", "high", "low", "close"):
            if k in cols:
                df[k] = pd.to_numeric(df[k], errors="coerce").to_numpy(float).astype(np.float32).astype(float)
    return df


def bars_rth_frame(name: str) -> Optional[pd.DataFrame]:
    """A name's regular-hours 30-minute bars (naive-UTC bar starts; Open High Low Close Volume) from its A0 part,
    in its store's precision (`in_precision`) — the dip adapter's `bars_30m`."""
    p = parts_dir() / f"{plain(name)}.parquet"
    if not p.exists():
        return None
    import duckdb
    df = duckdb.query(f"select ts, open, high, low, close, volume from '{p.as_posix()}' where session = 'rth' "
                      f"order by ts").df()
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.pop("ts"))
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return in_precision(df, name)


def daily(name: str) -> Optional[pd.DataFrame]:
    """`dip_long.daily_bars` of a name's A0 bars (the crash book's decision symbol)."""
    from src.signals.dip_long import daily_bars
    return daily_bars(bars_rth_frame(name))


_CACHE: Dict[str, object] = {}


def dip_panel() -> DipPanel:
    """The dip book's A0 panel, loaded once per process."""
    if "panel" not in _CACHE:
        p = dip_panel_path()
        if not p.exists():
            raise FileNotFoundError(f"set A0's dip panel is not built ({p}): run build101.py dip")
        z = pickle.load(open(p, "rb"))
        _CACHE["panel"] = DipPanel(z["panel"], z["eligible"])
    return _CACHE["panel"]


def vol_engine():
    if "eng" not in _CACHE:
        from src.backtest.strategies.vol import VolEngine
        _CACHE["eng"] = VolEngine(period=PERIOD)
    return _CACHE["eng"]


def etf_arm():
    if "etf" not in _CACHE:
        from src.backtest.strategies.etf import EtfArm
        _CACHE["etf"] = EtfArm(period=PERIOD)
    return _CACHE["etf"]


def ext_bars(extra_dir: Optional[str] = None):
    from src.backtest.mimic import ExtBars
    return ExtBars(extra_dir, period=PERIOD)


def dip_pieces(prm=None, lo: str = LO, hi: str = STORE_HI, **kw) -> List[dict]:
    """The dip book's limit orders on A0 (`dip.order_pieces` over `dip_panel`), the live rule by default; built once
    per (rule, window, options) in a process."""
    from src.backtest.strategies import dip as DIP
    prm = prm or DIP.LIVE
    key = ("dip", repr(prm), str(lo), str(hi), repr(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = DIP.order_pieces(prm, lo, hi, panel=dip_panel(), **kw)
    return _CACHE[key]


def in_a0(lo) -> bool:
    """A window starting before the deep store (`periods.DATA_START`) is set A0's."""
    return pd.Timestamp(lo) < pd.Timestamp(periods.DATA_START)


def check_window(lo, hi) -> None:
    """An A0 window must lie inside the A0 store's span (no window straddles two periods' data)."""
    if pd.Timestamp(lo) < pd.Timestamp(STORE_LO) or pd.Timestamp(hi) > pd.Timestamp(STORE_HI):
        raise ValueError(f"window {lo} .. {hi} is outside set A0's store ({STORE_LO} .. {STORE_HI}): an A0 window "
                         f"starts on/after {STORE_LO} and ends on/before {STORE_HI}")


def check_components(eng, etf, panel, bars, settle_fn, dip) -> tuple:
    """`mimic.live_books` on an A0 window: each component given must be A0's (a component of another period would
    build its own period's pieces silently); one not given is A0's (loaded once per process)."""
    bad = [n for n, x in (("eng", eng), ("etf", etf), ("panel", panel), ("bars", bars)) if x is not None and not is_a0(x)]
    if settle_fn is not None and not settles_a0(settle_fn):
        bad.append("settle")
    if dip is not None and not dip_is_a0(dip):
        bad.append("dip")
    if bad:
        raise ValueError(f"set A0's window needs A0's components; given another period's {bad} — use "
                         f"src.backtest.a0.harness(W) (or pass None to load A0's)")
    return (eng if eng is not None else vol_engine(), etf if etf is not None else etf_arm(),
            panel if panel is not None else dip_panel(), bars if bars is not None else ext_bars(),
            settle_fn if settle_fn is not None else settle())


def check_v6(eng) -> None:
    """V6 (`run71.make_real6`'s ``Real6``) installed on an A0 engine reads its module's ``NFHI``: it must be A0's
    (one flag per A0 store row: `harness` installs it), never another store's."""
    cls = getattr(eng, "Real", None)
    if getattr(cls, "__name__", "") != "Real6":
        return
    mod = sys.modules.get(getattr(cls, "__module__", ""))
    v = getattr(mod, "NFHI", None)
    n = len(eng.st.C)
    if v is None or len(v) != n:
        raise ValueError(f"V6 is installed on set A0's engine but its pressure flags hold {None if v is None else len(v)} "
                         f"rows, not the A0 store's {n}: run the A0 build inside src.backtest.a0.harness(W)")


def dip_is_a0(dip: Sequence[dict]) -> bool:
    """Dip pieces built on A0's panel: every decision on/before the A0 store's last day, every name with A0 bars."""
    if any(int(p.get("pick_day", -1)) > dnum(STORE_HI) for p in dip):
        return False
    return all((parts_dir() / f"{plain(t)}.parquet").exists() for t in {str(p.get("tkn")) for p in dip})


# ── the harness ──────────────────────────────────────────────────────────────

KEYS = ("eng", "etf", "panel", "dip", "bars", "settle")


@contextmanager
def harness(W: dict, extra_dir: Optional[str] = None, dip_prm=None):
    """A study harness (``R63._W``: eng, etf, panel, dip, bars, settle, costs) with set A0's components for a block,
    restored after — so the contexts a study applies to ``W["eng"]`` / ``W["etf"]`` (V6's `R75.Applied`, ...) patch
    A0's, `R76.run_set` settles on A0's calendar, V6's flags are A0's (the loaded `run71`'s ``NFHI`` while the block
    runs) and the halt model reads A0's halt files beside the existing ones. The ledger costs are kept."""
    saved = {k: W.get(k) for k in KEYS}
    eng = vol_engine()
    W.update(eng=eng, etf=etf_arm(), panel=dip_panel(), bars=ext_bars(extra_dir), settle=settle(),
             dip=dip_pieces(dip_prm))
    v6 = sys.modules.get("run71")
    v6_saved = (getattr(v6, "NFHI", None), getattr(v6, "NF", None)) if v6 is not None else None
    if v6 is not None:
        hi = pressure_hi(len(eng.st.C))
        if hi is None:
            raise FileNotFoundError(f"V6 is loaded but set A0's pressure flags are not built ({pressure_hi_path()})")
        v6.NFHI = hi
        sc = pressure_score_path()
        v6.NF = np.load(sc) if sc.exists() else np.where(hi, 1.0, 0.0)
    from src.backtest import halts as H
    h_saved = H.halt_paths()
    H.set_halt_paths(halt_files())
    try:
        yield W
    finally:
        W.update(saved)
        if v6 is not None:
            v6.NFHI, v6.NF = v6_saved
        H.set_halt_paths(h_saved)


def halt_files() -> List[str]:
    """NYSE's halt files: A0's years (`halts_dir`) and the existing ones (`halts.HALT_DIR`)."""
    from src.backtest import halts as H
    own = sorted(str(p) for p in halts_dir().glob("nyse_halts_*.csv"))
    return own + sorted(str(p) for p in Path(H.HALT_DIR).glob("nyse_halts_*.csv"))
