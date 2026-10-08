"""The mega-cap dip long book as backtest pieces — the live rule (`src/signals/dip_long.py`, the study PREREG43)
with tunable parameters, survivorship-free (today's names from the deep 30-minute store + the names delisted since
2021 from `cache/ml/deep/bars30m_delisted`, suffixed ``@D``).

A signal at the close of session D-1 (common stock / ADR, 20-session mean of close x volume >= ``min_dv``, close above
its ``trend``-session average, 2-session RSI under ``rsi_max``) is bought at the OPEN of session D (09:30 ET); it is
sold at the open after the first close above its ``exit_sma``-session average since the entry (``exit="next_open"``,
the live rule) or at that close (``exit="close"``, the study's tested variant), at the latest after ``max_hold``
sessions. Each piece carries the 30-minute path it is held through (bar END times, the vol engine's convention:
closes and lows) — or one mark a day with ``granularity="daily"`` — the half-spread of its dollar-volume class (the
study's table: 1 bp at $1B+ a day) and its rank (the RSI: the deepest dip first).

The daily panel (the study's own construction, `dip_long.daily_bars`) is built once and cached in
``cache/backtest/dip_daily.pkl`` (`build_panel`).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.signals.dip_long import daily_bars, rsi2

PANEL = Path("cache/backtest/dip_daily.pkl")
DELISTED = Path("cache/ml/deep/bars30m_delisted/parts")
TYPES_OK = ("CS", "ADRC")
MIN_BARS = 220                                       # the study's minimum daily bars per name
ET = "America/New_York"
BAR_NS = 30 * 60 * 10**9


@dataclass(frozen=True)
class DipParams:
    rsi_max: float = 10.0
    min_dv: float = 1e9
    trend: int = 200
    exit_sma: int = 5
    max_hold: int = 10
    exit: str = "next_open"                          # "next_open" (live) or "close" (the study's account)

    def key(self) -> tuple:
        return tuple(sorted(asdict(self).items()))


LIVE = DipParams()


def half_spread_bps(dv) -> np.ndarray:
    """The study's cost classes by 20-session dollar volume (lg_patterns.half_spread_bps)."""
    dv = np.asarray(dv, float)
    return np.select([dv >= 1e9, dv >= 2e8, dv >= 5e7, dv >= 2e7, dv >= 1e7], [1, 2, 4, 8, 12], 20).astype(float)


# ── data ─────────────────────────────────────────────────────────────────────

def _delisted_30m(sym: str) -> Optional[pd.DataFrame]:
    p = DELISTED / f"{sym}.parquet"
    if not p.exists():
        return None
    import duckdb
    df = duckdb.query(f"select ts, open, high, low, close, volume from '{p.as_posix()}' where session = 'rth' "
                      f"order by ts").df()
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.pop("ts"))
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})


def bars_30m(name: str) -> Optional[pd.DataFrame]:
    """A name's 30-minute regular-hours bars (naive-UTC bar starts): the deep store, or the delisted store for
    ``SYM@D``."""
    if name.endswith("@D"):
        return _delisted_30m(name[:-2])
    from src.data.intraday_store import load_deep_30m
    return load_deep_30m(name)


def build_panel(force: bool = False, workers: int = 8) -> Dict[str, pd.DataFrame]:
    """Daily regular-hours bars (o h l c v) for every name of the deep store and of the delisted store (``@D``),
    cached in `PANEL`."""
    if PANEL.exists() and not force:
        return pickle.load(open(PANEL, "rb"))
    from concurrent.futures import ThreadPoolExecutor
    from src.data.intraday_store import DEEP_DIR
    names = [p.stem for p in DEEP_DIR.glob("*.pkl")] + [p.stem + "@D" for p in DELISTED.glob("*.parquet")]

    def one(n):
        try:
            return n, daily_bars(bars_30m(n))
        except Exception:                                      # noqa: BLE001
            return n, None
    with ThreadPoolExecutor(workers) as ex:
        out = {n: d for n, d in ex.map(one, names) if d is not None and len(d)}
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = PANEL.with_suffix(".tmp")
    pickle.dump(out, open(tmp, "wb"))
    os.replace(tmp, PANEL)
    return out


def eligible_names(panel: Dict[str, pd.DataFrame]) -> List[str]:
    """Common stocks / ADRs (Polygon's type; a delisted name with no recorded type passes, as in the study)."""
    from src.data.company_names import security_type
    out = []
    for n in panel:
        t = security_type(n[:-2] if n.endswith("@D") else n)
        if t in TYPES_OK or (n.endswith("@D") and t in TYPES_OK + (None,)):
            out.append(n)
    return sorted(out)


# ── the rule ─────────────────────────────────────────────────────────────────

def signals(panel: Dict[str, pd.DataFrame], prm: DipParams, lo, hi, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Every signal with its entry and exit sessions: one row per (name, signal session in [lo, hi])."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    rows = []
    for tk in names if names is not None else eligible_names(panel):
        d = panel.get(tk)
        if d is None or len(d) < MIN_BARS:
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
            sig = (c > sma_t) & (r2 < prm.rsi_max) & (dv20 >= prm.min_dv) & (dates >= lo) & (dates <= hi)
        sig[:200] = False
        sig[n - 1:] = False
        for i in np.flatnonzero(sig):
            j = i + 1
            x = next((k for k in range(j, min(n, i + 1 + int(prm.max_hold))) if c[k] > sma_x[k]),
                     min(n - 1, i + int(prm.max_hold)))
            if prm.exit == "next_open":
                if x + 1 >= n:
                    continue
                xd = x + 1
            else:
                xd = x
            if (dates[xd] - dates[j]).days > 20:                 # a gap in the stored bars, not a real hold
                continue
            rows.append((tk, dates[i], dates[j], dates[xd], float(r2[i]), float(dv20[i]), float(c[i])))
    return pd.DataFrame(rows, columns=["tk", "sig", "d_in", "d_out", "rsi2", "dv20", "close"])


def _session_open_ns(day: pd.Timestamp) -> int:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} 09:30", tz=ET).value


def pieces(prm: DipParams, lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, strategy: str = "dip",
           granularity: str = "30m", names: Optional[Sequence[str]] = None) -> List[dict]:
    """The account engine's pieces for the dip rule over signals in [lo, hi]."""
    panel = panel if panel is not None else build_panel()
    S = signals(panel, prm, lo, hi, names)
    out: List[dict] = []
    for tk, g in S.groupby("tk"):
        d = panel[tk]
        if granularity == "30m":
            b = bars_30m(tk)
            if b is None or b.empty:
                continue
            starts = pd.DatetimeIndex(b.index)
            ends_ns = (starts + pd.Timedelta(minutes=30)).tz_localize("UTC").asi8
            et_day = starts.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None)
            bc, bl, bo = (b["Close"].to_numpy(float), b["Low"].to_numpy(float), b["Open"].to_numpy(float))
        for r in g.itertuples(index=False):
            e = float(d.loc[r.d_in, "o"])
            if prm.exit == "next_open":
                x, xns = float(d.loc[r.d_out, "o"]), _session_open_ns(r.d_out)
                last_day = d.index[d.index.get_loc(r.d_out) - 1]
            else:
                x, xns = float(d.loc[r.d_out, "c"]), pd.Timestamp(f"{r.d_out.date()} 16:00", tz=ET).value
                last_day = r.d_out
            ens = _session_open_ns(r.d_in)
            if not (e > 0 and x > 0):
                continue
            if granularity == "30m":
                m = (et_day >= r.d_in) & (et_day <= last_day)
                pt, pc, pl = ends_ns[m], bc[m], bl[m]
            else:
                dd = d.loc[r.d_in:last_day]
                pt = np.array([pd.Timestamp(f"{t.date()} 16:00", tz=ET).value for t in dd.index], np.int64)
                pc, pl = dd["c"].to_numpy(float), dd["l"].to_numpy(float)
            hs = float(half_spread_bps([r.dv20])[0])
            gross = 100.0 * (x / e - 1)
            out.append({"strategy": strategy, "tkn": tk, "ens": int(ens), "xns": int(xns), "e": e, "x": x,
                        "hs_in": hs, "hs_out": hs, "dv20": float(r.dv20), "pt": np.asarray(pt, np.int64),
                        "pc": np.asarray(pc, float), "pl": np.asarray(pl, float), "rank": float(r.rsi2),
                        "net": gross - 2 * hs / 100.0, "days": (xns - ens) / 86400e9,
                        "signal_day": str(r.sig.date()), "rsi2": float(r.rsi2),
                        # the decision's session (the signal close) as a day number: the study's week assignment
                        "pick_day": int((pd.Timestamp(r.sig).normalize() - pd.Timestamp("1970-01-01")).days)})
    return out
