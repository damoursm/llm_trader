"""SETS A AND B REBUILT WITH SET A0'S CONSTRUCTION (PREREG105, registered 2026-10-14 13:07 ET) — the sets ``A2`` and
``B2`` of `src/backtest/periods.py`: the decisions of sets A and B (2021-02-01 .. 2024-06-24, 2024-06-25 .. 2026-09-24)
on stores built by set A0's pipeline (`studies/prereg101`) extended through set B, as ONE continuous construction from
A0's start (2017-06) to the latest session: the universe screened day by day from Polygon's whole-market daily bars by
the symbol traded THAT day (tickers renamed or merged after 2021 included), split factors measured, histories warmed up
from 2017-06, IBKR's archived lendability and fees to 2024-06-23 and set B's borrow construction after it (PREREG94's
v4 models applied to the rebuilt store's rows), V6's pressure from production's model.

The data (new files only; the existing stores and set A0's files are never written):

* the bars — `cache/ml/deep_ab2/bars30m/parts/<SYM>.parquet` (`studies/prereg105/fetch105.py`): every session's
  30-minute bars from the name's start (2017-06; the dip names 2016-11, QQQ / TQQQ / AAPL / SPY 2016-01) to the data's
  end (`data_end`: the latest completed session when the construction started, frozen in `spans.json`),
  assembled from set A0's parts (to 2021-06-30), the production stores where they are the same series (the deep store's
  regular-hours pickles + its extended-hours parts, the delisted store — each checked equal to the early segment on
  2021-01-04 .. 2021-06-30), and Polygon (everything else), each row already in its class's store precision
  (`store_precision`);
* the research engine's arrays and stores rebuilt by build101's code (`studies/prereg105/build105.py`): `vol/ab2/`
  (arrays, the traded-price factors `factors.json`, the panel by name batch and year), `optvol/store_ab2/` (built in
  yearly chunks, merged: candidates, freshness, run-up bases, deadlines, paths, Rule 201, the corporate-action gap
  flags, `mm/`; the lend and fee models' arrays; V6's flags), `optvol/ab2_closes.pkl`, `optvol/ab2_borrow{,_pick}.npz`,
  the ETF arm's picks (`etf/vol_events_ab2_arm.pkl` and the picks re-ranked without the VIX-linked products), the dip
  panel (`cache/backtest/dip_daily_ab2.pkl`), IBKR's archived fees per name (`cache/ml/ib_shorting_archive/ab2/`).

How the simulator uses it: `harness(W)` swaps a study harness's components (``R63._W``) for the rebuilt ones during a
block — `VolEngineR`, `EtfArmR`, `DipPanelR`, `ExtBarsR`, `settle()`, the dip orders — installs the rebuilt store's V6
flags in the loaded `run71`, points `mimic`'s lend / fee model arrays at the rebuilt store and the crash book at the
rebuilt QQQ / TQQQ bars, and restores everything after. `mimic.live_books` on a set A2 / B2 window then runs its default
path with these components (no A0 code runs: a rebuilt window starts on/after the deep store's first session).

Conventions (stated with every result): set A0's (`src/backtest/a0.py`) — IBKR's fee: its API bar where the name's API
history covers the day, else IBKR's archived file of that day (fenced by the name's bar window), else the fee model's
fill (PREREG94's v4 for these rows); lendability: IBKR's archived file to 2024-06-24, PREREG94's linked draws of the v2
lend model (applied to these rows) after; the 2026 ledger costs, house rates and refusals unchanged; halts from the
existing NYSE files (2021 on).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.backtest import a0 as A0
from src.backtest import periods

REPO = Path(__file__).resolve().parents[2]
EPOCH = pd.Timestamp("1970-01-01")
ET = "America/New_York"
PERIOD = "AB2"                     # the rebuilt components' marker (`live_books` reads "A0" only: never this)
TAG = "ab2"
PERIODS = ("A2", "B2")             # `periods.get("A2")` / `get("B2")`: sets A and B's decisions on the rebuilt stores
OF_SET = {"A2": "A", "B2": "B"}    # the existing set each rebuilt one re-runs

# ── the spans ────────────────────────────────────────────────────────────────
FETCH_FROM, LONG_FROM, INDEX_FROM = A0.FETCH_FROM, A0.LONG_FROM, A0.INDEX_FROM   # set A0's (the warm-up from 2017-06)
SCREEN_FROM = A0.SCREEN_FROM                           # a name eligible on a day from here on is in the universe
STORE_START = A0.STORE_START                           # the 400-bar floor's exemption: bars beginning by then
EARLY_END = A0.DATA_END                                # set A0's data end: its parts are reused to here
CHECK_LO = "2021-01-04"                                # a production store is reused when it equals the early segment
                                                       # on CHECK_LO .. EARLY_END (the deep store's first months)
LATE_FROM = "2021-07-01"                               # the late segment (reused or fetched) starts here
STORE_LO = "2021-01-04"                                # the candidates: set A0's overlap month, sets A, B and D (to
                                                       # the data's end: `store_hi`)
OVERLAP = ("2021-01-04", "2021-04-30")                 # PREREG105 parity 1: equal to set A0's build row for row
DIP_LO = periods.DATA_START                            # the dip book's orders: from set A's first decision (`R63`'s)
FRACTIONAL_FROM = "2026-02-23"                         # Polygon serves fractional share volumes from this session
LEAD_SESSIONS = 80                                     # panel sessions read before a chunk (freshness W <= 60)
INDEX_NAMES = A0.INDEX_NAMES
CLASSES, ETP_TYPES, STOCK_TYPES, PATHS = A0.CLASSES, A0.ETP_TYPES, A0.STOCK_TYPES, A0.PATHS


def dnum(day) -> int:
    return int((pd.Timestamp(day) - EPOCH).days)


def dstr(d) -> str:
    return (EPOCH + pd.Timedelta(days=int(d))).date().isoformat()


plain, cls_of, extend = A0.plain, A0.cls_of, A0.extend


# ── where the data lives (PREREG105_SANDBOX=<dir> moves every rebuilt file under <dir>: light validation runs) ──

def _sandbox() -> Optional[Path]:
    v = os.environ.get("PREREG105_SANDBOX")
    return Path(v) if v else None


def research() -> Path:
    return Path(settings.backtest_research_dir)


def repo_out() -> Path:
    s = _sandbox()
    return s / "repo" if s else REPO


def research_out() -> Path:
    s = _sandbox()
    return s / "research" if s else research()


def deep_dir() -> Path:
    """Raw rebuilt data (bars, reference snapshots, the screen, the universe, short interest, grouped bars' tail)."""
    return repo_out() / "cache" / "ml" / "deep_ab2"


# ── the data's end: the latest completed session, frozen once per construction ──

def spans_path() -> Path:
    return deep_dir() / "spans.json"


def default_data_end() -> str:
    """The data's end for a NEW construction: ``PREREG105_DATA_END`` when set, else the latest completed NYSE session
    now (`periods.latest_completed_session`)."""
    v = os.environ.get("PREREG105_DATA_END")
    return pd.Timestamp(v).date().isoformat() if v else periods.latest_completed_session()


def spans() -> dict:
    """{"data_end": the last session of the rebuilt bars, "store_hi": the candidates' last decision (= the data's end:
    set D's decisions from 2026-09-25 are in the store), ...}: `spans.json` once the fetch has fixed it (every stage
    reads the same), else what a new construction would take now."""
    p = spans_path()
    try:
        m = p.stat().st_mtime_ns
    except OSError:
        d = default_data_end()
        return {"data_end": d, "store_hi": d, "fixed": False}
    hit = _SPANS.get(str(p))
    if hit is None or hit[0] != m:
        hit = _SPANS[str(p)] = (m, json.loads(p.read_text(encoding="utf-8")))
    return dict(hit[1])


_SPANS: Dict[str, tuple] = {}                          # spans.json read once per change (every unit asks)


def data_end() -> str:
    return str(spans()["data_end"])


def store_hi() -> str:
    """The candidates' (and the dip orders') last decision session: the data's end."""
    return str(spans()["store_hi"])


def fix_spans(day: Optional[str] = None) -> dict:
    """Freeze the construction's data end (`spans.json`): the first fetch stage calls it; ``day`` moves it (an
    extension: `fetch105.py extend`, after which every stale unit is redone). The new end must be a weekday on/after
    set B's last decision."""
    cur = spans() if spans_path().exists() else None
    d = pd.Timestamp(day).date().isoformat() if day else (cur["data_end"] if cur else default_data_end())
    if pd.Timestamp(d) < pd.Timestamp(periods.get("B").hi):
        raise ValueError(f"the rebuilt data must reach set B's last decision ({periods.get('B').hi}), not {d}")
    out = {"data_end": d, "store_hi": d, "fixed": True, "set_at": pd.Timestamp.now(tz=ET).isoformat(timespec="seconds"),
           "history": (cur or {}).get("history", []) + ([cur["data_end"]] if cur and cur["data_end"] != d else [])}
    p = spans_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, p)
    return out


REUSE_LAG = 30                                         # sessions: a production store's bars are reused only this far
REUSE_OVERLAP = 5                                      # back from the data's end (Polygon still revises the latest
                                                       # weeks: GME's volumes from 2026-09-14, a TQQQ bar of 10-06, the
                                                       # deep store's truncated volumes of the products from 10-02 —
                                                       # `fetch105.py prove`, 2026-10-14); the rest is fetched, checked
                                                       # on the REUSE_OVERLAP sessions before the cut


def reuse_to() -> str:
    """The last session a production store's bars are reused for: `REUSE_LAG` NYSE sessions before the data's end."""
    from src.performance import market_calendar as MC
    d = pd.Timestamp(data_end()).date()
    for _ in range(REUSE_LAG):
        d = MC.previous_market_day(d)
    return d.isoformat()


def chunks() -> Tuple[int, ...]:
    """The store's yearly chunks: 2021 .. the data's end's year."""
    return tuple(range(2021, pd.Timestamp(store_hi()).year + 1))


def dip_hi() -> str:
    return store_hi()


def window(key: str, now=None) -> Tuple[str, str]:
    """(first, last) decision session of a set on the rebuilt stores: "A2" / "B2" (sets A and B's spans), "D" / "D2"
    (PREREG106's set D — from 2026-09-25 to the latest completed session — cut at the rebuilt data's end)."""
    if key in ("D", "D2"):
        s = periods.get("D", now=now)
        hi = min(pd.Timestamp(s.hi), pd.Timestamp(store_hi()))
        return s.lo, hi.date().isoformat()
    s = periods.get(key)
    return s.lo, s.hi


def parts_dir() -> Path:
    return deep_dir() / "bars30m" / "parts"


def prov_dir() -> Path:
    """Each name's bars provenance (`fetch105.py bars`): the sources, the checks, the requests."""
    return deep_dir() / "bars30m" / "prov"


def universe_path() -> Path:
    return deep_dir() / "universe.json"


def si_path() -> Path:
    return deep_dir() / "short_interest_by_symbol.parquet"


def vol_dir() -> Path:
    return research_out() / "vol" / TAG


def bars_dir(cls: str) -> Path:
    return vol_dir() / {"": "bars", "@D": "bars_d", "@X": "bars_x", "@E": "bars_e"}[cls]


def factors_path() -> Path:
    return vol_dir() / "factors.json"


def panel_dir() -> Path:
    return vol_dir() / "panel"


def panel_meta_path() -> Path:
    return vol_dir() / "panel_meta.json"


def store_dir() -> Path:
    return research_out() / "optvol" / f"store_{TAG}"


def chunk_dir(year: int) -> Path:
    return store_dir() / "chunks" / str(int(year))


def closes_path() -> Path:
    return research_out() / "optvol" / f"{TAG}_closes.pkl"


def borrow_path(at: str = "next") -> Path:
    return research_out() / "optvol" / (f"{TAG}_borrow_pick.npz" if at == "pick" else f"{TAG}_borrow.npz")


def etf_events_path(tag: Optional[str] = None) -> Path:
    """The ETF arm's picks (`vol_events_ab2_arm.pkl`); ``tag``: the picks re-ranked without an exclusion's products
    (`vol_events_ab2_<tag>_arm.pkl` + its `.json` sidecar)."""
    stem = f"vol_events_{TAG}_{tag}_arm" if tag else f"vol_events_{TAG}_arm"
    return research_out() / "etf" / f"{stem}.pkl"


def archive_fee_dir() -> Path:
    return repo_out() / "cache" / "ml" / "ib_shorting_archive" / TAG


def dip_panel_path() -> Path:
    return repo_out() / "cache" / "backtest" / f"dip_daily_{TAG}.pkl"


def pressure_root() -> Path:
    return deep_dir() / "sel_short" / "pressure"


def pressure_hi_path() -> Path:
    return store_dir() / "pressure_hi.npy"


def pressure_score_path() -> Path:
    return store_dir() / "pressure_score.npy"


def lend_p_path(model: str = "v2") -> Path:
    """The lend model's probability per rebuilt store row (`mimic.LEND_P_FILES`' names)."""
    from src.backtest.mimic import LEND_P_FILES
    return store_dir() / LEND_P_FILES[model]


def fee_pred_path(version: str = "v4") -> Path:
    """The fee model's mean fee per rebuilt store row (`mimic.FEE_PRED_FILES`' names)."""
    from src.backtest.mimic import FEE_PRED_FILES
    return store_dir() / FEE_PRED_FILES[version]


# ── the store precision of each class ────────────────────────────────────────

def raw_volume_names() -> frozenset:
    """Set A0's list (`vol/a0/raw_volume.json`): the deep-store names whose stored volumes are Polygon's as served
    before 2026 (fractional after a reverse split); every other deep-store name's were whole shares."""
    return A0.raw_volume_names()


def store_precision(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Bars (``open high low close volume`` and ``ts`` naive UTC, as Polygon serves them) in the precision of the store
    the name's class comes from — every row of a rebuilt part is written in it, whatever its source: ``@D`` float32
    prices and whole-share volumes (the delisted store's parquet); ``@X`` whole shares (the outside store); ``@E``
    Polygon's volumes (the ETF store); a deep-store name Polygon's volumes when it is in `raw_volume_names` or the bar
    is on/after `FRACTIONAL_FROM` (Polygon's fractional shares, which the deep store keeps), else whole shares — set
    A0's `in_precision` extended past 2026's switch."""
    c = cls_of(name)
    df = df.copy()
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(float)
        if c in ("@D", "@X"):
            df["volume"] = np.floor(v)
        elif c == "":
            if plain(name) in raw_volume_names():
                df["volume"] = v
            else:
                day = et_days(df["ts"])
                df["volume"] = np.where(day >= dnum(FRACTIONAL_FROM), v, np.floor(v))
        else:
            df["volume"] = v
    if c == "@D":
        for k in ("open", "high", "low", "close", "vwap"):
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce").to_numpy(float).astype(np.float32).astype(float)
    return df


def ext_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Bars in the EXTENDED stores' precision (float32 prices, whole-share volumes): what every reader of the
    extended-hours bars sees (`mimic.ExtBars`, the deep store's `bars30m_full` and delisted parts)."""
    df = df.copy()
    for k in ("open", "high", "low", "close"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").to_numpy(float).astype(np.float32).astype(float)
    if "volume" in df.columns:
        df["volume"] = np.floor(pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(float))
    return df


def same_series(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[bool, dict]:
    """Two frames of every session's bars (each in the class's store precision) are the same series: the
    regular-hours bars equal bar for bar and value for value (the arrays, the dip and crash books read them as they
    are), the extended-hours bars equal in the extended stores' precision (the only one they are read in — the deep
    store keeps them in float32)."""
    ra, rb = a[a["session"] == "rth"], b[b["session"] == "rth"]
    ok, why = same_rows(ra, rb)
    if not ok:
        return False, dict(why, part="rth")
    ea, eb = ext_precision(a[a["session"] != "rth"]), ext_precision(b[b["session"] != "rth"])
    ok, why2 = same_rows(ea, eb)
    if not ok:
        return False, dict(why2, part="extended")
    return True, {"bars": int(len(a)), "rth": int(len(ra))}


def et_days(ts) -> np.ndarray:
    """The ET session day numbers of naive-UTC bar starts."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    if not len(idx):
        return np.zeros(0, np.int64)
    et = idx.tz_localize("UTC").tz_convert(ET)
    return ((et.normalize().tz_localize(None) - EPOCH).days).to_numpy().astype(np.int64)


def same_rows(a: pd.DataFrame, b: pd.DataFrame, cols=("open", "high", "low", "close", "volume")) -> Tuple[bool, dict]:
    """Two bar frames (``ts`` + ``cols``, one precision) are the same series: the same bar starts in the same order and
    equal values (NaN equal to NaN). Returns (same, a short record of the first difference)."""
    ta, tb = pd.to_datetime(a["ts"]).to_numpy(), pd.to_datetime(b["ts"]).to_numpy()
    if len(ta) != len(tb) or not np.array_equal(ta, tb):
        sa, sb = set(ta.tolist()), set(tb.tolist())
        return False, {"bars": [len(ta), len(tb)], "only_a": len(sa - sb), "only_b": len(sb - sa)}
    for k in cols:
        x, y = a[k].to_numpy(float), b[k].to_numpy(float)
        ok = (x == y) | (np.isnan(x) & np.isnan(y))
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            return False, {"column": k, "rows": int((~ok).sum()), "first": [str(ta[i]), float(x[i]), float(y[i])]}
    return True, {"bars": int(len(ta))}


# ── the traded-price factor (A0's construction, extended) ────────────────────

def _measure(sd: np.ndarray, c: np.ndarray, rows: np.ndarray, cands: List[float], gclose: Callable[[str], Optional[float]],
             ) -> Tuple[float, str]:
    """`build101.post_factor`'s measurement over ``rows`` (indices of a segment, ascending): the unadjusted daily close
    over the stored close of the segment's last <= 5 sessions (each session's last bar), its median snapped to the
    nearest of ``cands`` within 15%, else the measured ratio; no close: the last candidate."""
    if not len(rows):
        return float(cands[-1]), "empty"
    s = sd[rows]
    last = rows[np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]]
    rs = []
    for j in last[::-1][:5]:
        cu = gclose(dstr(int(sd[j])))
        if cu is not None and np.isfinite(cu) and cu > 0 and c[j] > 0:
            rs.append(float(cu) / float(c[j]))
    if not rs:
        return float(cands[-1]), "no_close"
    r = float(np.median(rs))
    best = min(cands, key=lambda x: abs(np.log(r / x)))
    if abs(np.log(r / best)) <= np.log(1.15):
        return float(best), ("table" if best != 1.0 else "none")
    return float(r), "measured"


def factor_record(sday, c, splits: Sequence[Tuple[int, float, float]],
                  gclose: Callable[[str], Optional[float]]) -> dict:
    """How each row's traded price (stored close x factor) is read: SET A0'S CONSTRUCTION over the rebuilt data —
    a split executed on/before K (the name's last session on/before set A0's data end, `EARLY_END`) by the stored
    series' jump (`volengine.split_factor`, the splits in the table's order, as build101's panel); every split after
    K by MEASUREMENT (build101's `post_factor`): the rows <= K carry the ratio measured on the last sessions <= K
    against the products of the splits after K (set A0's own measurement, so set A0's rows are reproduced exactly),
    and the rows after K are cut into segments at each later split's execution day, each measured on its own last
    sessions against the products of the splits after it. Returns {"K", "jump": [[ex, frm, to]...], "segments":
    [[first day, last day, factor, how]...]} (`row_factors` applies it)."""
    sd = np.asarray(sday, np.int64)
    c = np.asarray(c, float)
    n = len(sd)
    if not n:
        return {"K": None, "jump": [], "segments": []}
    k_rows = np.flatnonzero(sd <= dnum(EARLY_END))
    K = int(sd[k_rows[-1]]) if len(k_rows) else None
    jump = [[int(ex), float(frm), float(to)] for ex, frm, to in splits if K is not None and int(ex) <= K]
    post = sorted((int(ex), float(frm), float(to)) for ex, frm, to in splits if K is None or int(ex) > K)

    def cands_after(day: int) -> List[float]:
        out, f = [1.0], 1.0
        for ex, frm, to in post:
            if ex > day:
                f *= to / frm
                out.append(f)
        return out
    segs = []
    if K is not None:
        f, how = _measure(sd, c, k_rows, cands_after(K), gclose)
        segs.append([int(sd[0]), K, f, how])
    rest = np.flatnonzero(sd > (K if K is not None else -10**9))
    if len(rest):
        cuts = sorted({ex for ex, _, _ in post if int(sd[rest[0]]) < ex <= int(sd[-1])})
        bounds = [int(sd[rest[0]])] + cuts + [int(sd[-1]) + 1]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            rows = rest[(sd[rest] >= lo) & (sd[rest] < hi)]
            if not len(rows):
                continue
            seg_last = int(sd[rows[-1]])
            f, how = _measure(sd, c, rows, cands_after(seg_last), gclose)
            segs.append([int(sd[rows[0]]), seg_last, f, how])
    return {"K": K, "jump": jump, "segments": segs}


def row_factors(sday, c, rec: dict, split_factor: Callable) -> np.ndarray:
    """Each row's factor from `factor_record`'s record: the jump part (`volengine.split_factor` over the record's
    jump splits) times the row's segment factor."""
    sd = np.asarray(sday, np.int64)
    J = split_factor(sd, np.asarray(c, float), [tuple(x) for x in rec.get("jump") or []]) if rec.get("jump") \
        else np.ones(len(sd))
    M = np.ones(len(sd))
    for lo, hi, f, _ in rec.get("segments") or []:
        M[(sd >= int(lo)) & (sd <= int(hi))] = float(f)
    return J * M


# ── calendars ────────────────────────────────────────────────────────────────

def _research_path() -> None:
    A0._research_path()


@lru_cache(maxsize=1)
def sessions() -> np.ndarray:
    """The rebuilt data's sessions: the days AAPL traded in its rebuilt bars (2016-01-04 .. `data_end`)."""
    a = load_bars("AAPL")
    if a is None:
        raise FileNotFoundError(f"the rebuilt AAPL bars are not built ({bars_dir('')}): run build105.py")
    return np.unique(np.asarray(a["sday"], np.int64))


@lru_cache(maxsize=1)
def calendar() -> np.ndarray:
    """`volengine.calendar`'s sessions over the rebuilt data: the rebuilt AAPL sessions, then the research calendar
    after them (the production calendar past its data). The ETF arm's calendar."""
    _research_path()
    import volengine as ve
    return extend(sessions(), ve.calendar())


@lru_cache(maxsize=1)
def c4_sessions() -> np.ndarray:
    """`cap5k4.sessions` over the rebuilt data (the vol engine's ``C4.CAL``: run-up bases, deadlines, turns, the lend
    draws' calendar)."""
    _research_path()
    import cap5k4 as C4
    return extend(sessions(), C4.sessions())


def settlement() -> np.ndarray:
    from src.backtest import account as A
    A.default_settle()
    return extend(A.settlement_days(sessions()), A._CAL["s"])


def settle() -> Callable[[int], int]:
    """`account.default_settle` over the rebuilt sessions (T+2; T+1 from 2024-05-28; the bank holidays skipped)."""
    from src.backtest import account as A
    return A.make_settle(settlement())


@contextmanager
def research_calendar():
    """`cap5k4.CAL` set to the rebuilt sessions for a block (`cap5k4.settle` reads it), restored after."""
    _research_path()
    import cap5k4 as C4
    prev = C4.CAL
    C4.CAL = c4_sessions()
    try:
        yield
    finally:
        C4.CAL = prev


def c4_proxy():
    from types import SimpleNamespace
    return SimpleNamespace(CAL=c4_sessions(), settle=settle())


# ── the rebuilt bars and arrays ──────────────────────────────────────────────

def load_bars(name) -> Optional[dict]:
    """The research engine's arrays of a rebuilt engine name."""
    p = bars_dir(cls_of(name)) / f"{plain(name)}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def bars_rth_frame(name: str) -> Optional[pd.DataFrame]:
    """A name's regular-hours 30-minute bars (naive-UTC starts; Open High Low Close Volume) from its rebuilt part,
    in its store precision (the parts hold it) — the arrays' and the dip adapter's input."""
    p = parts_dir() / f"{plain(name)}.parquet"
    if not p.exists():
        return None
    import duckdb
    df = duckdb.query(f"select ts, open, high, low, close, volume from '{p.as_posix()}' where session = 'rth' "
                      f"order by ts").df()
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.pop("ts"))
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})


def daily(name: str) -> Optional[pd.DataFrame]:
    """`dip_long.daily_bars` of a name's rebuilt bars (the crash book's decision symbol)."""
    from src.signals.dip_long import daily_bars
    return daily_bars(bars_rth_frame(name))


# ── the vol engine's store ───────────────────────────────────────────────────

def mm_store():
    """`wf10.mm_store` over the rebuilt store (candidates, freshness, paths and Rule 201 memory mapped)."""
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


def load_borrow(at: str = "next"):
    z = np.load(borrow_path(at), allow_pickle=False)
    return z["lend"], z["lend_usd"], z["fee"], z["drop"]


def real_fees():
    """IBKR's daily fee for the rebuilt names, set A0's construction (`a0.real_fees`): its API bar where the name's
    API history covers the day, else IBKR's archived file of that day for the name (`archive_fee_dir`, fenced by its
    bar window), else the fallback; each carried at most a week. PREREG188 (set D): from 2026-09-25 OUR OWN archive's
    daily close (`borrow_daily.parquet`, `borrow_history.fee_daily`'s ``own_archive``) comes first on its days."""
    _research_path()
    import realfee as RF
    arch = str(archive_fee_dir())

    class RealFeesR(RF.RealFees):
        def __init__(self):
            super().__init__(parts=RF.PARTS, field="fee", archive=arch)

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
            key = "@AB2:" + plain(tkn)
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

        def _own(self, tkn):
            """Our own archive's daily close fee of the name (set D's days; None before 2026-09-25 / unlisted)."""
            if "@OWN" not in self.cache:
                self.cache["@OWN"] = _own_archive_fees()
            return self.cache["@OWN"].get(plain(tkn))

        def series(self, tkn):
            return self._api(tkn) or self._arch(tkn) or self._own(tkn)

        def daily(self, tkn, days, fallback):
            days = np.asarray(days, np.int64)
            out = np.full(len(days), np.nan)
            for s in (self._own(tkn), self._api(tkn), self._arch(tkn)):
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
            for s in (self._own(tkn), self._api(tkn), self._arch(tkn)):
                if s is not None and len(s[0]) and s[0][0] <= day and (len(s) < 3 or day <= s[2]):
                    return True
            return False

    return RealFeesR()


def _own_archive_fees() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{plain symbol: (day numbers, our archive's daily close fee)} from `borrow_daily.parquet` (PREREG188: set D's
    fees), read once; empty when the file is absent."""
    import duckdb
    from src.data.deep import borrow_history as BH
    p = BH._deep.DEEP_DIR / "borrow_daily.parquet"
    if not p.exists():
        return {}
    con = duckdb.connect()
    try:
        D = con.execute(f"SELECT ticker, CAST(date AS DATE) AS date, fee FROM read_parquet('{p.as_posix()}') "
                        "WHERE fee IS NOT NULL").fetchdf()
    finally:
        con.close()
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tk, g in D.groupby("ticker"):
        dn = np.array([dnum(str(x)[:10]) for x in g["date"]], np.int64)
        o = np.argsort(dn, kind="mergesort")
        out[str(tk).replace(".", "-")] = (dn[o], g["fee"].to_numpy(float)[o])
    return out


def assembler(st, closes, rf, cap: int = 400_000):
    """`wf10.Assembler` with `cap5k4.settle` on the rebuilt sessions."""
    _research_path()
    import wf10

    class AssemblerR(wf10.Assembler):
        def pieces(self, ev, cfg):
            with research_calendar():
                return super().pieces(ev, cfg)
    return AssemblerR(st, closes, rf, cap=cap)


def pressure_hi(n_rows: int) -> Optional[np.ndarray]:
    p = pressure_hi_path()
    if not p.exists():
        return None
    v = np.load(p)
    if len(v) != int(n_rows):
        raise ValueError(f"{p}: {len(v)} flags for a store of {n_rows} rows — rebuild it (build105.py pressure)")
    return v.astype(bool)


def check_v6(eng) -> None:
    """V6 (`run71.make_real6`'s ``Real6``) on the rebuilt engine reads its module's ``NFHI``: it must hold one flag per
    rebuilt store row (`harness` installs them), never another store's (an index into another store's flags would be
    silently wrong)."""
    cls = getattr(eng, "Real", None)
    if getattr(cls, "__name__", "") != "Real6":
        return
    mod = sys.modules.get(getattr(cls, "__module__", ""))
    v = getattr(mod, "NFHI", None)
    n = len(eng.st.C)
    if v is None or len(v) != n:
        raise ValueError(f"V6 is installed on the rebuilt engine but its pressure flags hold "
                         f"{None if v is None else len(v)} rows, not the rebuilt store's {n}: build inside "
                         f"src.backtest.rebuild.harness(W)")


# ── the components ───────────────────────────────────────────────────────────

def _vol_engine_class():
    from src.backtest.strategies.vol import VolEngine

    class VolEngineR(VolEngine):
        """The research vol engine over the rebuilt store (set A0's `VolEngine._init_period`, the rebuilt files): the
        same evaluator, assembler and rules; the rebuilt calendar for deadlines, turns and settlement; IBKR's
        archived lendability and fees. Its fee fill must be one per rebuilt row (a guard), and V6 must carry the
        rebuilt flags (`check_v6`)."""
        period = PERIOD

        def __init__(self, research_dir: Optional[str] = None, tag: str = "_dx"):
            root = research_dir or settings.backtest_research_dir
            self.dir = os.path.join(root, "optvol")
            if not os.path.isdir(self.dir):
                raise FileNotFoundError(f"research engine not found at {self.dir} (settings.backtest_research_dir)")
            os.environ.setdefault("OPTVOL_TAG", tag)
            for p in (self.dir, os.path.join(root, "pylib")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import build  # noqa: F401  (puts the vol engine's own directory on the path)
            import volengine as ve
            import wf10
            import wf5 as W5
            from evaluator import WHY
            from prereg23_optuna import FIXED, Real
            from prereg26_optuna_borrow import mask
            self.logger = ve.setup()
            self.ve = ve
            self.C4 = c4_proxy()
            self.W5, self.wf10, self.Real, self.mask, self.FIXED = W5, wf10, Real, mask, FIXED
            self.WHY = WHY
            self.B = load_borrow("next")
            self._B_pick = load_borrow("pick")
            self.rf = real_fees()
            self.nbbo = False
            self.st = mm_store()
            self.asm = assembler(self.st, pickle.load(open(closes_path(), "rb")), self.rf, cap=400_000)
            self.settle = self.C4.settle
            self.fee_fallback = None
            n = len(self.st.C)
            if len(self.B[0]) != n or len(self._B_pick[0]) != n:
                raise ValueError(f"the rebuilt borrow arrays do not match the rebuilt store ({n} rows): rebuild them "
                                 f"(build105.py mm)")
            self._fee_pred = None                    # `live_books` loads it through `fee_pred_path` (the harness's)
            self._fee_pred_loaded = None
            self.pressure_hi = pressure_hi(n)

        @property
        def fee_fallback(self):
            return self.__dict__.get("_fee_fallback")

        @fee_fallback.setter
        def fee_fallback(self, v):
            if v is not None and "st" in self.__dict__ and len(v) != len(self.st.C):
                raise ValueError(f"a fee fill of {len(v)} rows for the rebuilt store's {len(self.st.C)}: another "
                                 f"store's — run inside src.backtest.rebuild.harness(W)")
            self.__dict__["_fee_fallback"] = v

        def _real(self, lo, hi):
            check_v6(self)
            return super()._real(lo, hi)
    return VolEngineR


def _etf_arm_class():
    from src.backtest.strategies import etf as ETF

    class EtfArmR(ETF.EtfArm):
        """The ETF arm over the rebuilt picks (`etf/vol_events_ab2_arm.pkl`, build101's construction over the rebuilt
        panel), bars, calendar, settlement and fees; the picks re-ranked without an exclusion's products are the
        rebuilt ones (`vol_events_ab2_<tag>_arm.pkl` + sidecar, `build105.py etf`). Deadlines are judged to the
        rebuilt data's end (`data_end`) unless a call sets ``data_end``."""

        def __init__(self, research_dir: Optional[str] = None):
            super().__init__(research_dir)
            self.period = PERIOD
            self.rf = real_fees()
            self.E = pd.read_pickle(etf_events_path())
            self.cal = np.asarray(calendar(), np.int64)
            self.settle = settle()
            self._load = load_bars
            self._bars, self._ca, self._ex = {}, {}, {}

        def events(self, exclude: Optional[Iterable[str]] = None) -> pd.DataFrame:
            key = ETF.exclusion_key(exclude)
            if key is None:
                return self.E
            if key not in self._ex:
                self._ex[key] = excluded_events(key, recorded=self.E)[0]
            E = self._ex[key]
            return E[~E.tkn.map(ETF.plain).isin(key)]

        def events_source(self, exclude: Optional[Iterable[str]] = None) -> Optional[str]:
            key = ETF.exclusion_key(exclude)
            if key is None:
                return "recorded"
            hit = excluded_sidecar(key)
            return None if hit is None else str(hit[0].with_suffix(".pkl"))

        def pieces(self, lo, hi, **kw):
            kw.setdefault("data_end", data_end())
            return super().pieces(lo, hi, **kw)
    return EtfArmR


def excluded_sidecar(key: frozenset):
    """(sidecar, record) of the rebuilt picks re-ranked without exactly ``key``'s products, or None."""
    d = research_out() / "etf"
    for js in sorted(d.glob(f"vol_events_{TAG}_*_arm.json")):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("period") == PERIOD and frozenset(meta.get("excluded") or ()) == frozenset(key) \
                and js.with_suffix(".pkl").exists():
            return js, meta
    return None


def excluded_events(key: frozenset, recorded: Optional[pd.DataFrame] = None):
    """(picks, sidecar) re-ranked without ``key``'s products on the rebuilt panel; its source signature is checked
    against ``recorded`` (the rebuilt recorded picks). None built -> FileNotFoundError."""
    from src.backtest.strategies import etf as ETF
    hit = excluded_sidecar(key)
    if hit is None:
        raise FileNotFoundError(f"no rebuilt ETF picks re-ranked without {sorted(key)} under {research_out() / 'etf'}"
                                f" — build them (studies/prereg105/build105.py work etf)")
    js, meta = hit
    if recorded is not None and list(meta.get("source_sig") or []) != ETF.events_signature(recorded):
        raise ValueError(f"{js.with_suffix('.pkl').name} was re-ranked from another version of the rebuilt picks: "
                         f"rebuild it (build105.py etf)")
    return pd.read_pickle(js.with_suffix(".pkl")), js


class DipPanelR(A0.DipPanel):
    """The dip book's daily panel over the rebuilt bars (its own names and their 30-minute bars)."""
    period = PERIOD

    @staticmethod
    def bars_fn(name: str) -> Optional[pd.DataFrame]:
        return bars_rth_frame(name)


def ext_bars_class():
    from src.backtest import mimic as M

    class ExtBarsR(M.ExtBars):
        """The rebuilt parts' bars of every session, read in the extended stores' precision (float32 prices, whole
        shares) — `mimic.ExtBars`' own mechanism for a period's parts."""

        def __init__(self, extra_dir: Optional[str] = None):
            super().__init__(extra_dir)
            self.period, self._a0 = PERIOD, Path(parts_dir())
    return ExtBarsR


_CACHE: Dict[str, object] = {}


def vol_engine():
    if "eng" not in _CACHE:
        _CACHE["eng"] = _vol_engine_class()()
    return _CACHE["eng"]


def etf_arm():
    if "etf" not in _CACHE:
        _CACHE["etf"] = _etf_arm_class()()
    return _CACHE["etf"]


def ext_bars(extra_dir: Optional[str] = None):
    return ext_bars_class()(extra_dir)


def dip_panel() -> DipPanelR:
    if "panel" not in _CACHE:
        p = dip_panel_path()
        if not p.exists():
            raise FileNotFoundError(f"the rebuilt dip panel is not built ({p}): run build105.py dip")
        z = pickle.load(open(p, "rb"))
        _CACHE["panel"] = DipPanelR(z["panel"], z["eligible"])
    return _CACHE["panel"]


def dip_pieces(prm=None, lo: str = DIP_LO, hi: Optional[str] = None, **kw) -> List[dict]:
    """The dip book's limit orders on the rebuilt panel (`dip.order_pieces`), the live rule by default, built once per
    (rule, window, options) in a process."""
    from src.backtest.strategies import dip as DIP
    prm = prm or DIP.LIVE
    hi = hi or dip_hi()
    key = ("dip", repr(prm), str(lo), str(hi), repr(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = DIP.order_pieces(prm, lo, hi, panel=dip_panel(), **kw)
    return _CACHE[key]


def check_window(lo, hi) -> None:
    """A rebuilt window lies inside the rebuilt store (no window straddles two constructions)."""
    if pd.Timestamp(lo) < pd.Timestamp(STORE_LO) or pd.Timestamp(hi) > pd.Timestamp(store_hi()):
        raise ValueError(f"window {lo} .. {hi} is outside the rebuilt store ({STORE_LO} .. {store_hi()})")


def built() -> List[str]:
    """What a rebuilt run needs (`build105.py`), missing."""
    need = [store_dir() / "mm" / "PH.npy", closes_path(), borrow_path("next"), borrow_path("pick"), etf_events_path(),
            dip_panel_path(), pressure_hi_path(), archive_fee_dir() / "extract.json", lend_p_path("v2"),
            fee_pred_path("v4"), parts_dir() / "QQQ.parquet", parts_dir() / "TQQQ.parquet"]
    return [str(p) for p in need if not Path(p).exists()]


# ── the harness ──────────────────────────────────────────────────────────────

KEYS = ("eng", "etf", "panel", "dip", "bars", "settle")


@contextmanager
def harness(W: dict, extra_dir: Optional[str] = None, dip_prm=None):
    """A study harness (``R63._W``: eng, etf, panel, dip, bars, settle, costs) with the REBUILT components for a block,
    restored after — the contexts a study applies to ``W["eng"]`` / ``W["etf"]`` (V6's `R75.Applied`, ...) then patch
    the rebuilt engine — and, for the block: the loaded `run71`'s V6 flags (``NFHI`` / ``NF``) are the rebuilt
    store's; `mimic.lend_p_path` / `fee_pred_path` give the rebuilt store's lend and fee model arrays; the crash book
    reads the rebuilt QQQ / TQQQ bars (`crash._daily`, `dm.bars_30m`); the halt model reads the existing NYSE files
    (2021 on). The ledger costs are kept."""
    from src.backtest import halts as H
    from src.backtest import mimic as M
    from src.backtest.strategies import crash as CRASH
    from src.backtest.strategies import dm as DM
    eng = vol_engine()
    v6 = sys.modules.get("run71")
    hi = sc = None
    if v6 is not None:                                    # checked before anything is swapped
        hi = pressure_hi(len(eng.st.C))
        if hi is None:
            raise FileNotFoundError(f"V6 is loaded but the rebuilt pressure flags are not built ({pressure_hi_path()})")
        p = pressure_score_path()
        sc = np.load(p) if p.exists() else np.where(hi, 1.0, 0.0)
    comps = dict(eng=eng, etf=etf_arm(), panel=dip_panel(), bars=ext_bars(extra_dir), settle=settle(),
                 dip=dip_pieces(dip_prm))
    saved = {k: W[k] for k in KEYS if k in W}
    v6_saved = (getattr(v6, "NFHI", None), getattr(v6, "NF", None)) if v6 is not None else None
    m_saved = (M.lend_p_path, M.fee_pred_path)
    c_saved = (CRASH._daily, DM.bars_30m)
    h_saved = H.halt_paths()
    try:
        W.update(comps)
        if v6 is not None:
            v6.NFHI, v6.NF = hi, sc
        M.lend_p_path, M.fee_pred_path = lend_p_path, fee_pred_path
        CRASH._daily, DM.bars_30m = daily, bars_rth_frame
        H.set_halt_paths(None)
        yield W
    finally:
        for k in KEYS:
            if k in saved:
                W[k] = saved[k]
            else:
                W.pop(k, None)
        if v6 is not None:
            v6.NFHI, v6.NF = v6_saved
        M.lend_p_path, M.fee_pred_path = m_saved
        CRASH._daily, DM.bars_30m = c_saved
        H.set_halt_paths(h_saved)
