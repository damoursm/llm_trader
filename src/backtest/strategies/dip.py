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
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.signals.dip_long import daily_bars, rsi2

PANEL = Path("cache/backtest/dip_daily.pkl")
DELISTED = Path("cache/ml/deep/bars30m_delisted/parts")
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
                    if not ended:
                        continue                                 # the exit lies past the data
                    xd, last_close = n - 1, True                 # delisted / acquired: sold at its last close
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
                 sessions: Optional[int] = None, keep: Optional[Mapping[str, bool]] = None) -> List[dict]:
    """PRODUCTION'S LIMIT ENTRY (`dip_long_entry_mode` "limit", 2026-10-09) as the account engine's limit ORDERS
    (`Strategy(orders="fill")`, `Rules(one_book_per_name=True)` to mimic the live account): every signal of [lo, hi]
    is one order at `dip_long.limit_price(close, atr14, atr_mult)` (the price IBKR rests), PLACED at 09:30 (+1 ns) of
    each of its ``sessions`` window sessions (D .. D+4), lapsing after the last one's 16:00 (+1 ns); one piece per
    session it touches — the session's OPEN (+2 ns) when it opened at/under the limit (filled at the open), else the
    END of the first 30-minute bar whose low reaches the limit (filled at the limit), the live tick's fill — each with
    the live exit counted from that session and its 30-minute marks; a never-touched order is a placeholder piece
    (``ord_k`` -1). Defaults: `dip_long_limit_atr`, `dip_long_limit_sessions`. ``keep`` (PREREG96's RP33, production's
    `enable_dip_long_filter`; `filter_keep`): {order id: kept} — a signal mapped to False never becomes an order (no
    piece, not even a placeholder: it takes no slot); one the map does not hold is ordered as usual."""
    from config import settings
    from src.signals.dip_long import atr14, limit_price
    panel = panel if panel is not None else build_panel()
    W = int(sessions if sessions is not None else settings.dip_long_limit_sessions)
    S = signals(panel, prm, lo, hi, names)
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
                if x + 1 >= n:
                    if not ended:
                        continue                                 # the exit lies past the data
                    x = n - 1                                    # delisted / acquired: sold at its last close
                    xv, xns = float(c[x]), pd.Timestamp(f"{dates[x].date()} 16:00", tz=ET).value
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


def filter_keep(path=None, research: bool = False) -> Dict[str, bool]:
    """{order id (engine name | entry session): kept} for `order_pieces(keep=...)` / `filter_pieces` — PREREG96's RP33
    as production applies it (`enable_dip_long_filter`): EVERY signal scored at its signal close, filled or not (a
    dropped order takes no slot), by the walk-forward model of its year (nothing of a year trains on it). ``research``:
    the construction of the study's add-rule draws (run95's ``_dip_keep``) — only the FILLED signals the study scored
    carry a flag, an unfilled order is kept as is (keyed by engine name here: run95 keyed the plain ticker, which left 6
    delisted ``@D`` orders of 2,370 unfiltered on sets A / B). A signal the map does not hold (outside the scored
    years) is kept."""
    from config import settings
    p = Path(path) if path is not None else Path(settings.backtest_research_dir) / FILTER_SCORES
    W = pd.read_pickle(p)
    if research:
        W = W[W["keep_research"].notna()]
        return {str(o): bool(k) for o, k in zip(W["oid"], W["keep_research"])}
    return {str(o): bool(k) for o, k in zip(W["oid"], W["keep"])}


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
