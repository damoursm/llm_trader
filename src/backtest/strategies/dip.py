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

The daily panel (the study's own construction, `dip_long.daily_bars`): by default THE OWN-SECURITY PANEL (2026-10-18,
the reused-ticker fix — PREREG142's corrected construction, production's `src/signals/dip_security.py`: every series
cut to its own security, every renamed company's earlier tickers joined; `OWN_PANEL`, with the walk-forward scores of
its signals `OWN_SCORES`) — from 2026-10-19 WITH set A's warm-up (AUDIT151's D_WARM: the 223 names shared with the A0
panel carry their 2016-2020 rows; `OWN_PANEL_NOWARM` / `OWN_SCORES_NOWARM` the studies 10-18 .. 10-19); ``OWN_DEFAULT =
False`` (or ``build_panel(own=False)``) restores the stored construction cached in ``cache/backtest/dip_daily.pkl``
and PREREG96's scores — the studies before 2026-10-18. A piece filled at the open PRINT is repriced at IBKR's
attainable 09:30:30 price by `reprice_open` (AUDIT151's D_OPEN, `Mimic(dip_open=True)`).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from config import settings
from src.signals.dip_long import daily_bars, rsi2

PANEL = Path("cache/backtest/dip_daily.pkl")
DELISTED = Path("cache/ml/deep/bars30m_delisted/parts")
# THE OWN-SECURITY PANELS (2026-10-18; user: "Yes please apply the fix"): PREREG142's corrected AB / A0 panels and the
# walk-forward scores of their signals (PREREG142's OLD: model96's yearly models on the corrected inputs), under
# settings.backtest_research_dir — the simulator's defaults, production's construction (`dip_security --parity`: its
# output equals the study's on 255 of the 256 names the study changed; RBC differs only by the study's duplicate drop,
# never a $1B name). The deployed configuration on them: R142_OLD's draws, +56.8% a year (+0.6 points over the stored
# panels). False: the stored panels (`PANEL`, `a0.dip_panel_path()`) and PREREG96's scores (`FILTER_SCORES`).
# AUDIT151 (2026-10-19, the user: "Yes please do 1 and 2"): the A / B panel WITH ITS WARM-UP — it started 2021-01-04 for
# every name, so with 220 sessions needed set A's first dip signal fell in November 2021 (live has the history): each of
# the 223 names the A0 panel shares with it (identical on the overlap) carries its 2016-2020 rows
# (`studies/audit151/dip_warm/save_warm_panel.py`), and the walk-forward scores add the 2021 signals scored by the 2021
# fold model; +5.4 pp a year on set A's account. `OWN_PANEL_NOWARM` / `OWN_SCORES_NOWARM`: the studies 10-18 .. 10-19.
OWN_DEFAULT = True
# PREREG188 (set D, the live period): a signal or fill whose exit lies past the data is KEPT, its exit marked at the
# data's last close ("open_at_end") — a position still open when the data ends. False (every study before it: their
# data end lies past their decisions) drops it. A process sets it for a set-D run only.
OPEN_END = False
OWN_PANEL = Path("studies") / "audit151" / "dip_warm" / "panel151_warm.pkl"
OWN_PANEL_A0 = Path("studies") / "prereg142" / "panel142_a0.pkl"
OWN_SCORES = Path("studies") / "audit151" / "dip_warm" / "wf_scores_warm.pkl"
OWN_PANEL_NOWARM = Path("studies") / "prereg142" / "panel142.pkl"
OWN_SCORES_NOWARM = Path("studies") / "prereg142" / "wf_scores_old.pkl"
# AUDIT151's D_OPEN: the price an order RESTING AT IBKR could get at a session's open (production's resting order
# arrives ~09:30:30): every order piece that fills at an open in A0 / A / B, from Polygon's minute bars and trades
# (`studies/audit151/dip_open/extend_fills.py`) — `reprice_open`
OPEN_FILLS = Path("studies") / "audit151" / "dip_open" / "fills_all.csv"
_OPEN_FILLS_AUDIT151 = OPEN_FILLS
# PREREG227's S1: the same table EXTENDED to the capitulation and fear books' open fills (AUDIT151's rule and fetch,
# `studies/prereg227/open_fills227.py`) — every AUDIT151 row unchanged, so the dip book's fills are the same; read
# instead of AUDIT151's once it exists, unless a study set `OPEN_FILLS` to its own table (PREREG188's set D)
OPEN_FILLS_EXT = Path("studies") / "prereg227" / "open_fills" / "fills_all227.csv"
TYPES_OK = ("CS", "ADRC")
MIN_BARS = 220                                       # the study's minimum daily bars per name


def exit_hit(prm: "DipParams", c: np.ndarray, sma_x: np.ndarray, r2: Optional[np.ndarray], k: int) -> bool:
    """The exit's condition at session k: the close above its exit_sma average (live), or (PREREG98 S2, ``exit_rule``
    "rsi2") the session's RSI2 above ``exit_level``."""
    if prm.exit_rule == "rsi2":
        return bool(r2[k] > float(prm.exit_level))
    return bool(c[k] > sma_x[k])


def gap_days(prm: "DipParams") -> int:
    """The calendar span past which a signal's hold is read as a GAP in the stored bars (a halt, a data hole), not a
    real hold: 20 days at the live 10-session limit, twice the limit beyond it — a 20-day guard dropped the longest
    REAL holds of a 15- or 20-session limit, the worst trades (PREREG97; live trading has no such guard)."""
    return max(20, 2 * int(prm.max_hold))
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
    # PREREG98 S2: "sma" (live) = the first close above the exit_sma average; "rsi2" = the first close whose RSI2 is
    # above exit_level (the oversold state that triggered the entry has resolved)
    exit_rule: str = "sma"
    exit_level: float = 50.0

    def key(self) -> tuple:
        return tuple(sorted(asdict(self).items()))


# production's exits: from the 2026-10-20 session PREREG159's RPD_IS (the first close above the 3-session average, or
# 5 sessions; `dip_long_exit_sma_sessions` / `dip_long_max_hold_sessions`); `DipParams()` keeps the 5 / 10 before it
LIVE = DipParams(exit_sma=int(settings.dip_long_exit_sma_sessions), max_hold=int(settings.dip_long_max_hold_sessions))


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


class OwnPanel(dict):
    """The own-security AB panel: the daily frames, and the 30-minute bars its order fills read — a changed name's from
    PREREG142's corrected series (``bars``: name -> file), every other name's from the deep / delisted store."""
    own = True

    def __init__(self, data: Dict[str, pd.DataFrame], bars: Mapping[str, str]):
        super().__init__(data)
        self.bars = dict(bars)

    def bars_fn(self, name: str) -> Optional[pd.DataFrame]:
        p = self.bars.get(name)
        if p is None:
            return bars_30m(name)
        return pickle.load(open(p, "rb"))


_OWN: Dict[str, object] = {}


def own_panel() -> OwnPanel:
    """The own-security AB panel (`OWN_PANEL`), loaded once per process (per file version)."""
    from config import settings
    p = Path(settings.backtest_research_dir) / OWN_PANEL
    key = (str(p), p.stat().st_mtime_ns)
    if _OWN.get("key") != key:
        z = pickle.load(open(p, "rb"))
        _OWN.clear()
        _OWN.update(key=key, panel=OwnPanel(z["panel"], z["bars"]))
    return _OWN["panel"]                                                # type: ignore[return-value]


def build_panel(force: bool = False, workers: int = 8, own: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
    """Daily regular-hours bars (o h l c v) for every name of the deep store and of the delisted store (``@D``): the
    own-security panel by default (`OWN_DEFAULT`, `own_panel`), else the stored construction cached in `PANEL`
    (``force`` rebuilds it; the own-security panel is PREREG142's build: studies/prereg142 spans142.py raw / events /
    spans -> bars142.py -> panel142.py)."""
    if OWN_DEFAULT if own is None else own:
        if force:
            raise ValueError("the own-security panel is PREREG142's build (studies/prereg142: spans142.py raw / "
                             "events / spans -> bars142.py -> panel142.py); build_panel(force=True, own=False) "
                             "rebuilds the stored one")
        return own_panel()
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
    """Common stocks / ADRs (Polygon's type; a delisted name with no recorded type passes, as in the study), each
    company once: a class share stored under both spellings (BRK-B and BRK.B) keeps the project's form, as
    production's `dip_long.candidates` does (PREREG64). A panel that carries its own list (PREREG101's set A0:
    ``panel.eligible``, Polygon's type at the time) gives it."""
    own = getattr(panel, "eligible", None)
    if own is not None:
        return sorted(n for n in own if n in panel)
    from src.data.company_names import security_type
    from src.signals.sel_short import _internal_symbol
    out = []
    for n in panel:
        base = n[:-2] if n.endswith("@D") else n
        canon = _internal_symbol(base) + ("@D" if n.endswith("@D") else "")
        if canon != n and canon in panel:
            continue
        t = security_type(n[:-2] if n.endswith("@D") else n)
        if t in TYPES_OK or (n.endswith("@D") and t in TYPES_OK + (None,)):
            out.append(n)
    return sorted(out)


# ── the rule ─────────────────────────────────────────────────────────────────

def signals(panel: Dict[str, pd.DataFrame], prm: DipParams, lo, hi, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Every signal with its entry and exit sessions: one row per (name, signal session in [lo, hi])."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    data_last = max(v.index[-1] for v in panel.values() if len(v))
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
        sig[:MIN_BARS - 1] = False                       # production: MIN_BARS daily bars through the signal close
        sig[n - 1:] = False
        ended = dates[-1] < data_last - pd.Timedelta(days=7)   # the name stopped trading before the data's end
        for i in np.flatnonzero(sig):
            j = i + 1
            x = next((k for k in range(j, min(n, i + 1 + int(prm.max_hold))) if exit_hit(prm, c, sma_x, r2, k)),
                     min(n - 1, i + int(prm.max_hold)))
            last_close = False
            if prm.exit == "next_open":
                if x + 1 >= n:
                    if not ended and not OPEN_END:
                        continue                                 # the exit lies past the data
                    xd, last_close = n - 1, True                 # delisted / acquired (or OPEN_END): its last close
                else:
                    xd = x + 1
            else:
                xd = x
            if (dates[xd] - dates[j]).days > gap_days(prm):      # a gap in the stored bars, not a real hold
                continue
            rows.append((tk, dates[i], dates[j], dates[xd], float(r2[i]), float(dv20[i]), float(c[i]), last_close))
    return pd.DataFrame(rows, columns=["tk", "sig", "d_in", "d_out", "rsi2", "dv20", "close", "last_close"])


def _session_open_ns(day: pd.Timestamp) -> int:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} 09:30", tz=ET).value


def order_pieces(prm: DipParams, lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, strategy: str = "dip",
                 names: Optional[Sequence[str]] = None, atr_mult: Optional[float] = None,
                 sessions: Optional[int] = None, keep: Optional[Mapping[str, bool]] = None,
                 signals_fn: Optional[Callable] = None, open_end: bool = False) -> List[dict]:
    """PRODUCTION'S LIMIT ENTRY (`dip_long_entry_mode` "limit", 2026-10-09) as the account engine's limit ORDERS
    (`Strategy(orders="fill")`, `Rules(one_book_per_name=True)` to mimic the live account): every signal of [lo, hi]
    is one order at `dip_long.limit_price(close, atr14, atr_mult)` (the price IBKR rests), PLACED at 09:30 (+1 ns) of
    each of its ``sessions`` window sessions (D .. D+4), lapsing after the last one's 16:00 (+1 ns); one piece per
    session it touches — the session's OPEN (+2 ns) when it opened at/under the limit (filled at the open), else the
    END of the first 30-minute bar whose low reaches the limit (filled at the limit), the live tick's fill — each with
    the live exit counted from that session and its 30-minute marks; a never-touched order is a placeholder piece
    (``ord_k`` -1). Defaults: `dip_long_limit_atr`, `dip_long_limit_sessions`. ``keep`` (PREREG96's RP33, production's
    `enable_dip_long_filter`; `filter_keep`): {order id: kept} — a signal mapped to False never becomes an order (no
    piece, not even a placeholder: it takes no slot); one the map does not hold is ordered as usual. ``signals_fn``:
    another book's signals in `signals`' form (the capitulation book's, `strategies.cap`) — the same orders, fills and
    exits. ``open_end`` (PREREG188, set D): a fill whose exit lies past the data is kept, marked at the data's last
    close (``why`` "open_at_end": a position still open when the data ends) instead of dropped — off for every set
    whose data end lies past its decisions (A0 / A / B unchanged)."""
    from config import settings
    from src.signals.dip_long import atr14, limit_price
    panel = panel if panel is not None else build_panel()
    W = int(sessions if sessions is not None else settings.dip_long_limit_sessions)
    S = (signals_fn or signals)(panel, prm, lo, hi, names)
    data_last = max(v.index[-1] for v in panel.values() if len(v))
    bars_of = getattr(panel, "bars_fn", None) or bars_30m      # PREREG101: an A0 panel reads its own bars
    out: List[dict] = []
    for tk, g in S.groupby("tk"):
        d = panel[tk]
        b = bars_of(tk)
        if b is None or b.empty:
            continue
        ended = d.index[-1] < data_last - pd.Timedelta(days=7)
        starts = pd.DatetimeIndex(b.index)
        st_utc = starts.tz_localize("UTC")
        ends_ns = (st_utc + pd.Timedelta(minutes=30)).asi8
        et = st_utc.tz_convert(ET)
        et_day = et.normalize().tz_localize(None)
        first = (et.hour == 9) & (et.minute == 30)
        bc, bl, bo = (b["Close"].to_numpy(float), b["Low"].to_numpy(float), b["Open"].to_numpy(float))
        dates = d.index
        n = len(d)
        c = d["c"].to_numpy(float)
        sma_x = pd.Series(c).rolling(int(prm.exit_sma)).mean().to_numpy()
        r2x = rsi2(c) if prm.exit_rule == "rsi2" else None
        for r in g.itertuples(index=False):
            i = int(dates.get_loc(r.sig))
            if keep is not None and not keep.get(f"{tk}|{dates[i + 1].date()}", True):
                continue                                     # the meta-model dropped the signal: no order
            lim = limit_price(r.close, atr14(d.iloc[:i + 1]), atr_mult)
            if lim is None:
                continue
            win = list(range(i + 1, min(n, i + 1 + W)))
            if not win:
                continue
            place = [_session_open_ns(dates[t]) + 1 for t in win]
            end = pd.Timestamp(f"{dates[win[-1]].date()} 16:00", tz=ET).value + 1
            hs = float(half_spread_bps([r.dv20])[0])
            base = {"strategy": strategy, "tkn": tk, "hs_in": hs, "hs_out": hs, "dv20": float(r.dv20),
                    "rank": float(r.rsi2), "oid": f"{tk}|{dates[i + 1].date()}", "ord_place": place,
                    "ord_lim": float(lim), "ord_end": int(end), "signal_day": str(r.sig.date()), "rsi2": float(r.rsi2),
                    "pick_day": int((pd.Timestamp(r.sig).normalize() - pd.Timestamp("1970-01-01")).days)}
            made = 0
            for m, j in enumerate(win):
                k0 = int(np.searchsorted(et_day, dates[j], side="left"))
                k1 = int(np.searchsorted(et_day, dates[j], side="right"))
                if k1 <= k0:
                    continue
                if first[k0] and bo[k0] <= lim:
                    ens, e = _session_open_ns(dates[j]) + 2, float(bo[k0])
                else:
                    ks = np.flatnonzero(bl[k0:k1] <= lim)
                    if not len(ks):
                        continue
                    ens, e = int(ends_ns[k0 + int(ks[0])]), float(lim)
                x = next((k for k in range(j, min(n, j + int(prm.max_hold))) if exit_hit(prm, c, sma_x, r2x, k)),
                         min(n - 1, j + int(prm.max_hold) - 1))
                why_end = None
                if x + 1 >= n:
                    if not ended and not (open_end or OPEN_END):
                        continue                                 # the exit lies past the data
                    x = n - 1                                    # delisted / acquired: sold at its last close
                    xv, xns = float(c[x]), pd.Timestamp(f"{dates[x].date()} 16:00", tz=ET).value
                    why_end = None if ended else "open_at_end"   # PREREG188: still open when the data ends
                elif (dates[x + 1] - dates[j]).days > gap_days(prm):
                    continue
                else:
                    xd = x + 1
                    xv, xns = float(d.iloc[xd]["o"]), _session_open_ns(dates[xd])
                held = (et_day >= dates[j]) & (et_day <= dates[x]) & (ends_ns > ens)
                p = dict(base)
                p.update(ens=int(ens), xns=int(xns), e=e, x=xv, pt=np.asarray(ends_ns[held], np.int64),
                         pc=np.asarray(bc[held], float), pl=np.asarray(bl[held], float), ord_k=m,
                         net=100.0 * (xv / e - 1) - 2 * hs / 100.0, days=(xns - ens) / 86400e9)
                if why_end:
                    p["why"] = why_end
                out.append(p)
                made += 1
            if not made:
                p = dict(base)
                p.update(ens=int(end), xns=int(end) + 86400 * 10**9, e=float(lim), x=float(lim),
                         pt=np.zeros(0, np.int64), pc=np.zeros(0), pl=np.zeros(0), ord_k=-1)
                out.append(p)
    return out


def _ext_scale(b: Optional[pd.DataFrame], xb: Optional[dict], tol: float = 0.02) -> Optional[float]:
    """The factor from the extended store's prices to the regular-hours store's (the dip panel's terms): the median
    ratio of their regular-hours bar closes ending at the same instants; None when they disagree by more than ``tol``."""
    if b is None or b.empty or xb is None or not len(xb["end"]):
        return None
    ends = (pd.DatetimeIndex(b.index).tz_localize("UTC") + pd.Timedelta(minutes=30)).asi8
    j = np.searchsorted(xb["end"], ends)
    ok = (j < len(xb["end"])) & (np.take(xb["end"], np.minimum(j, len(xb["end"]) - 1)) == ends)
    if ok.sum() < 5:
        return None
    r = b["Close"].to_numpy(float)[ok] / np.where(xb["c"][j[ok]] > 0, xb["c"][j[ok]], np.nan)
    r = r[np.isfinite(r) & (r > 0)]
    lo_, hi_ = np.quantile(r, [0.05, 0.95]) if len(r) else (np.nan, np.nan)
    if not len(r) or hi_ / lo_ - 1.0 > tol:
        return None
    return float(np.median(r))


_OPEN: Dict[str, object] = {}


def open_fills(path=None) -> Dict[tuple, tuple]:
    """{(engine name, ET day, limit rounded to 1e-6): (how, price, touch minute, terms ok)} — `OPEN_FILLS`, loaded
    once per file version."""
    from config import settings
    if path is not None:
        p = Path(path)
    else:
        root = Path(settings.backtest_research_dir)
        ext = OPEN_FILLS == _OPEN_FILLS_AUDIT151 and (root / OPEN_FILLS_EXT).exists()
        p = root / OPEN_FILLS_EXT if ext else root / OPEN_FILLS
    key = (str(p), p.stat().st_mtime_ns)
    if _OPEN.get("key") != key:
        F = pd.read_csv(p)
        tm = F["a30.touch_min"] if "a30.touch_min" in F else pd.Series([None] * len(F))
        ok = F["terms_ok"] if "terms_ok" in F else pd.Series([True] * len(F))
        _OPEN.update(key=key, map={
            (str(t), str(d), round(float(lim), 6)): (str(h), float(px) if pd.notna(px) else None,
                                                     str(m) if pd.notna(m) else None, bool(k) if pd.notna(k) else True)
            for t, d, lim, h, px, m, k in zip(F.tkn, F.day, F.limit, F["a30.how"], F["a30.price"], tm, ok)})
    return _OPEN["map"]                                                  # type: ignore[return-value]


def reprice_open(p: dict, log: Optional[dict] = None, books: Sequence[str] = ("dip",), none_drops: bool = False,
                 stamp_unknown: bool = False, touch_free: bool = False) -> Optional[dict]:
    """AUDIT151's D_OPEN (`Mimic(dip_open=True)`, the default from 2026-10-19): a dip piece filled at the open PRINT
    (``ens`` 09:30 + 2 ns) -> at the price IBKR could give its resting order arriving 09:30:30 — the arrival trades'
    median at or under the limit (entered then), else the limit at a later touch (booked at that 30-minute bar's end,
    the marks before it dropped), from `open_fills` by (name, day, limit). The print was never attainable; on the
    deployed configuration the fills average 9 bp better (+0.59 pp a year). A piece the table lacks — or whose terms
    disagree with the store's — keeps the print (counted in ``log``).

    PREREG227's S1 (`Mimic.open_order`, the default from 2026-10-27): ``books`` adds the capitulation and fear books
    (production books IBKR's own fill for all three, and places the dip book's orders first, then the capitulation
    book's, then the fear book's — at one instant the engine fills them in that order, `mimic.PRIORITY`);
    ``none_drops`` — a fill the rule finds no price for that session ("none": the resting order never reached its
    limit after arriving) is NO fill: None, the caller drops the piece and the order keeps working; ``stamp_unknown`` —
    a fill the table cannot price keeps the print, stamped 09:30:30 (production's fill arrives then: the order holds).
    S5 (`Mimic.quote_all`): ``touch_free`` — a fill moved to a later touch rested at its limit: no spread (``hs_in``
    0; the open's quoted half-spread was its entry cost)."""
    if p.get("strategy") not in books or int(p["ens"]) % (60 * 10 ** 9) != 2:
        return p
    t = pd.Timestamp(int(p["ens"]), unit="ns", tz="UTC").tz_convert(ET)
    lim = p.get("ord_lim")
    hit = open_fills().get((str(p["tkn"]), str(t.date()), round(float(lim), 6))) if lim is not None else None

    def count(k):
        if log is not None:
            log[k] = log.get(k, 0) + 1
    if hit is None or not hit[3]:
        count("dip_open_unknown" if hit is None else "dip_open_terms")
        if stamp_unknown:
            q = dict(p)
            q["ens"] = int(pd.Timestamp(f"{t.date()} 09:30:30", tz=ET).value)
            count("dip_open_stamped")
            return q
        return p
    how, price, tmin, _ = hit
    q = dict(p)
    if how == "arrival" and price is not None:
        q["e"], q["ens"] = float(price), int(pd.Timestamp(f"{t.date()} 09:30:30", tz=ET).value)
        count("dip_open_arrival")
        return q
    if how == "touch" and tmin:
        o = pd.Timestamp(f"{t.date()} 09:30", tz=ET)
        m = pd.Timestamp(f"{t.date()} {tmin}", tz=ET)
        end = o + pd.Timedelta(minutes=30 * (1 + int((m - o).total_seconds() // 1800)))
        q["e"], q["ens"] = float(lim), int(end.value)
        pt = np.asarray(p.get("pt", np.zeros(0)), np.int64)
        keep = pt > int(end.value)
        for k, v in p.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == pt.shape[:1] and k != "pt":
                q[k] = v[keep]
        q["pt"] = pt[keep]
        if touch_free:
            q["hs_in"] = 0.0
        count("dip_open_touch")
        return q
    count("dip_open_none")
    if none_drops:
        count("dip_open_none_dropped")
        return None
    return p


def order_pieces_ext(prm: DipParams, lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, ext=None, nights=None,
                     strategy: str = "dip", names: Optional[Sequence[str]] = None, atr_mult: Optional[float] = None,
                     sessions: Optional[int] = None, min_usd: float = 15_000.0) -> List[dict]:
    """PREREG121's E1 / E2: production's limit orders also working OUTSIDE regular hours. A signal of D-1's close is
    placed at D-1's 16:30 after-hours tick and re-placed at each window session's 04:00 (an IBKR DAY order outside
    regular hours lives to 20:00), lapsing after D+4's 20:00; with ``nights`` (E2, `mimic.NightBars`) also each night
    between (20:00-03:50, IBKR's OVERNIGHT venue, the same limit). In each such segment the order fills at the first
    30-minute bar (pre-market, regular or after-hours; overnight) that opens at/under the limit (at that open) or
    whose low reaches it (at the limit, at the bar's end) — outside regular hours only a bar with at least ``min_usd``
    traded; the LATE-PRINT bar (`mimic.late_print_bar`: the TRF's report of the night's trades sets its open and low)
    fills only by its close (amendment 2). The exit is the rule's, counted from the first regular close after the fill.
    A name without extended bars or a price scale between the stores keeps `order_pieces`' pieces. ``ext``: a
    `mimic.ExtBars`."""
    from config import settings
    from src.backtest.mimic import late_print_bar
    from src.signals.dip_long import atr14, limit_price
    panel = panel if panel is not None else build_panel()
    W = int(sessions if sessions is not None else settings.dip_long_limit_sessions)
    S = signals(panel, prm, lo, hi, names)
    data_last = max(v.index[-1] for v in panel.values() if len(v))
    bars_of = getattr(panel, "bars_fn", None) or bars_30m
    out: List[dict] = []

    def at(day, hhmm):
        return pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}", tz=ET).value

    for tk, g in S.groupby("tk"):
        d = panel[tk]
        b = bars_of(tk)
        if b is None or b.empty:
            continue
        xb = ext.get(tk) if ext is not None else None
        s = _ext_scale(b, xb)
        if s is None:                                               # no extended prices to trade on: the RTH book
            out += order_pieces(prm, lo, hi, panel=panel, strategy=strategy, names=[tk], atr_mult=atr_mult,
                                sessions=sessions)
            continue
        nb = nights.get(tk) if nights is not None else None
        ended = d.index[-1] < data_last - pd.Timedelta(days=7)
        starts = pd.DatetimeIndex(b.index)
        st_utc = starts.tz_localize("UTC")
        ends_ns = (st_utc + pd.Timedelta(minutes=30)).asi8
        et_day = st_utc.tz_convert(ET).normalize().tz_localize(None)
        bc, bl = b["Close"].to_numpy(float), b["Low"].to_numpy(float)
        dates = d.index
        n = len(d)
        c = d["c"].to_numpy(float)
        sma_x = pd.Series(c).rolling(int(prm.exit_sma)).mean().to_numpy()
        r2x = rsi2(c) if prm.exit_rule == "rsi2" else None
        for r in g.itertuples(index=False):
            i = int(dates.get_loc(r.sig))
            lim = limit_price(r.close, atr14(d.iloc[:i + 1]), atr_mult)
            if lim is None:
                continue
            win = list(range(i + 1, min(n, i + 1 + W)))
            if not win:
                continue
            # the segments, in time order: (placement instant, [t_from, t_to), store, "count from" session index)
            segs = [(at(dates[i], "16:30") + 1, at(dates[i], "16:30"), at(dates[i], "20:00"), "pm0", i + 1)]
            if nb is not None:
                segs.append((at(dates[i], "20:00") + 1, at(dates[i], "20:00"), at(dates[win[0]], "04:00"), "night", i + 1))
            for m, j in enumerate(win):
                segs.append((at(dates[j], "04:00") + 1, at(dates[j], "04:00"), at(dates[j], "20:00"), "ext", j))
                if nb is not None and m + 1 < len(win):
                    segs.append((at(dates[j], "20:00") + 1, at(dates[j], "20:00"), at(dates[win[m + 1]], "04:00"),
                                 "night", j + 1))
            end = at(dates[win[-1]], "20:00") + 1
            hs = float(half_spread_bps([r.dv20])[0])
            base = {"strategy": strategy, "tkn": tk, "hs_in": hs, "hs_out": hs, "dv20": float(r.dv20),
                    "rank": float(r.rsi2), "oid": f"{tk}|{dates[i + 1].date()}", "ord_place": [sg[0] for sg in segs],
                    "ord_lim": float(lim), "ord_end": int(end), "signal_day": str(r.sig.date()), "rsi2": float(r.rsi2),
                    "pick_day": int((pd.Timestamp(r.sig).normalize() - pd.Timestamp("1970-01-01")).days)}
            made = 0
            for k, (place, t0, t1, src, cnt) in enumerate(segs):
                src_b = nb if src == "night" else xb
                if src_b is None or not len(src_b["start"]):
                    continue
                a0_, a1_ = int(np.searchsorted(src_b["start"], t0)), int(np.searchsorted(src_b["start"], t1))
                late = late_print_bar(src_b["start"][a0_:a1_]) if src != "night" else np.zeros(a1_ - a0_, bool)
                hit = None
                for q in range(a0_, a1_):
                    rth = bool(src_b["rth"][q]) if "rth" in src_b else False
                    usd = float(src_b["c"][q] * src_b["v"][q])
                    if not rth and usd < min_usd:
                        continue
                    if late[q - a0_]:              # the TRF's late overnight prints set its open / low: its close only
                        c_ = float(src_b["c"][q]) * s
                        if c_ <= lim:
                            hit = (int(src_b["end"][q]), c_, rth, int(src_b["start"][q]))
                            break
                        continue
                    o_, l_ = float(src_b["o"][q]) * s, float(src_b["l"][q]) * s
                    if o_ <= lim:
                        hit = (int(src_b["start"][q]) + 2, o_, rth, int(src_b["start"][q]))
                        break
                    if l_ <= lim:
                        hit = (int(src_b["end"][q]), float(lim), rth, int(src_b["start"][q]))
                        break
                if hit is None:
                    continue
                ens, e, _, bstart = hit
                # the exit counts from the first regular close after the fill (an after-hours fill: the next session)
                j0 = cnt
                if src == "ext" and pd.Timestamp(bstart, unit="ns", tz="UTC").tz_convert(ET).hour >= 16:   # its own close
                    j0 = cnt + 1
                if j0 >= n:
                    continue
                x = next((kk for kk in range(j0, min(n, j0 + int(prm.max_hold))) if exit_hit(prm, c, sma_x, r2x, kk)),
                         min(n - 1, j0 + int(prm.max_hold) - 1))
                if x + 1 >= n:
                    if not ended:
                        continue
                    x = n - 1
                    xv, xns = float(c[x]), pd.Timestamp(f"{dates[x].date()} 16:00", tz=ET).value
                elif (dates[x + 1] - dates[j0]).days > gap_days(prm):
                    continue
                else:
                    xv, xns = float(d.iloc[x + 1]["o"]), _session_open_ns(dates[x + 1])
                held = (et_day >= dates[j0]) & (et_day <= dates[x]) & (ends_ns > ens)
                p = dict(base)
                p.update(ens=int(ens), xns=int(xns), e=e, x=xv, pt=np.asarray(ends_ns[held], np.int64),
                         pc=np.asarray(bc[held], float), pl=np.asarray(bl[held], float), ord_k=k, fill_src=src,
                         net=100.0 * (xv / e - 1) - 2 * hs / 100.0, days=(xns - ens) / 86400e9)
                out.append(p)
                made += 1
            if not made:
                p = dict(base)
                p.update(ens=int(end), xns=int(end) + 86400 * 10**9, e=float(lim), x=float(lim),
                         pt=np.zeros(0, np.int64), pc=np.zeros(0), pl=np.zeros(0), ord_k=-1)
                out.append(p)
    return out


# PREREG96's RP33 on production's construction (`src/signals/dip_filter.py`, switched off in production until the user
# deploys it): each signal's WALK-FORWARD score — the model of its year (model96's fit on the filled signals exiting
# before Y-01-01, P33's cut from their out-of-fold scores) on production's inputs (the deep store's signals: sets A, B)
# or wf96_a0's (set A0's panel) — written by llm_trader_research/studies/prereg96/prod96/parity96.py.
FILTER_SCORES = Path("studies") / "prereg96" / "prod96" / "wf_scores_prod.pkl"


def scores_path(path=None, research: bool = False) -> Path:
    """The walk-forward score file: ``path``, else the own-security panels' (`OWN_SCORES`, `OWN_DEFAULT`) — PREREG96's
    `FILTER_SCORES` with the stored panels, and always for ``research`` (only it carries the study's keep flags)."""
    from config import settings
    if path is not None:
        return Path(path)
    return Path(settings.backtest_research_dir) / (OWN_SCORES if OWN_DEFAULT and not research else FILTER_SCORES)


def filter_keep(path=None, research: bool = False) -> Dict[str, bool]:
    """{order id (engine name | entry session): kept} for `order_pieces(keep=...)` / `filter_pieces` — PREREG96's RP33
    as production applies it (`enable_dip_long_filter`): EVERY signal scored at its signal close, filled or not (a
    dropped order takes no slot), by the walk-forward model of its year (nothing of a year trains on it). ``research``:
    the construction of the study's add-rule draws (run95's ``_dip_keep``) — only the FILLED signals the study scored
    carry a flag, an unfilled order is kept as is (keyed by engine name here: run95 keyed the plain ticker, which left 6
    delisted ``@D`` orders of 2,370 unfiltered on sets A / B). A signal the map does not hold (outside the scored
    years) is kept."""
    p = scores_path(path, research)
    W = pd.read_pickle(p)
    if research:
        W = W[W["keep_research"].notna()]
        return {str(o): bool(k) for o, k in zip(W["oid"], W["keep_research"])}
    return {str(o): bool(k) for o, k in zip(W["oid"], W["keep"])}


def filter_scores(path=None) -> Dict[str, float]:
    """{order id: its walk-forward score} — PREREG96's map (`filter_keep`'s file), every signal scored at its signal
    close by the model of its year (PREREG107's ranking key)."""
    p = scores_path(path)
    W = pd.read_pickle(p)
    W = W[W["score"].notna()]
    return {str(o): float(s) for o, s in zip(W["oid"], W["score"])}


def rank_pieces(pieces: Sequence[dict], scores: Mapping[str, float], by: str = "score",
                topk: Optional[int] = None, demote: Optional[Set[str]] = None) -> List[dict]:
    """PREREG107 (user 2026-10-14: "does it favor its best picks first? Would it be better the buy the top 1 pick even
    if we have more slots available?"): the dip book's order pieces (`order_pieces`, shared: copied, never modified)
    ranked by the meta-model's walk-forward score, highest first (``by="score"``: rank = -score; an unscored order
    after every scored one, by its RSI2 — production's key) or by the deepest dip (``by="rsi2"``: the order's RSI2, its
    ``rank``; ``scores`` unread), and with ``topk`` only each signal day's ``topk`` best orders by that key (the day =
    the order's first session, its ``oid``) — production's `dip_long.rank_key` either way. ``demote`` (PREREG137's
    DEMOTE_E, production's earnings-dip demotion: `mimic.earnings_demoted`): the order ids that rank after every other
    order of their day, among themselves by the same key (their rank + 1e7, so they also fill after the others).
    2026-10-17: ``by="rsi2"`` with ``topk`` chose each day's best by the SCORE (only the fill order read the RSI2) —
    production's "rsi2" picks the deepest dip; only PREREG137 Amendment 3's RS1 / RS1D ran it."""
    if by not in ("score", "rsi2"):
        raise ValueError(f"by must be 'score' or 'rsi2', not {by!r}")
    demote = demote or frozenset()

    def key(p) -> float:
        if by == "score":
            s = scores.get(str(p.get("oid")))
            k = -float(s) if s is not None else 1e6 + float(p.get("rank", 0.0))
        else:                                    # the deepest dip first: the order's RSI2
            k = float(p.get("rank", 0.0))
        return k + 1e7 if str(p.get("oid")) in demote else k
    keys = [key(p) for p in pieces]                  # each order's key, from its own piece (never a re-keyed copy)
    out = [dict(p, rank=k) for p, k in zip(pieces, keys)]
    if topk is None:
        return out
    best: Dict[str, List[tuple]] = {}
    for p, k in zip(out, keys):
        oid = str(p.get("oid"))
        day = oid.split("|")[-1]
        best.setdefault(day, [])
        if (k, oid) not in best[day]:
            best[day].append((k, oid))
    keep = {oid for day, rows in best.items() for _, oid in sorted(rows)[:int(topk)]}
    return [p for p in out if str(p.get("oid")) in keep]


def filter_pieces(pieces: Sequence[dict], keep: Mapping[str, bool]) -> List[dict]:
    """``pieces`` (`order_pieces`, built once and shared) without every piece of an order ``keep`` maps to False — the
    same pieces as building them with ``keep``."""
    return [p for p in pieces if keep.get(p.get("oid"), True)]


def pieces(prm: DipParams, lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, strategy: str = "dip",
           granularity: str = "30m", names: Optional[Sequence[str]] = None) -> List[dict]:
    """The account engine's pieces for the dip rule over signals in [lo, hi]."""
    panel = panel if panel is not None else build_panel()
    S = signals(panel, prm, lo, hi, names)
    bars_of = getattr(panel, "bars_fn", None) or bars_30m      # PREREG101: an A0 panel reads its own bars
    out: List[dict] = []
    for tk, g in S.groupby("tk"):
        d = panel[tk]
        if granularity == "30m":
            b = bars_of(tk)
            if b is None or b.empty:
                continue
            starts = pd.DatetimeIndex(b.index)
            ends_ns = (starts + pd.Timedelta(minutes=30)).tz_localize("UTC").asi8
            et_day = starts.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None)
            bc, bl, bo = (b["Close"].to_numpy(float), b["Low"].to_numpy(float), b["Open"].to_numpy(float))
        for r in g.itertuples(index=False):
            e = float(d.loc[r.d_in, "o"])
            if prm.exit == "next_open" and bool(getattr(r, "last_close", False)):
                x, xns = float(d.loc[r.d_out, "c"]), pd.Timestamp(f"{r.d_out.date()} 16:00", tz=ET).value
                last_day = r.d_out                               # delisted / acquired: sold at its last close
            elif prm.exit == "next_open":
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
