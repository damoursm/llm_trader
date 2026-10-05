"""V2 of the selection short's MODEL arm — its extra inputs (user directive 2026-10-02: "Deploy to live
production the V2 model arm").

V2 is the live model's recipe (LightGBM `tailreg`, long, 400 rounds, fit on sessions <= 2026-04-30) trained
on today's names AND the companies delisted since 2021, with 161 inputs: the v1 model's 146 (in the same
order) + the 15 below, and the 8 yfinance inputs the delisted names lack BLANKED (NaN) in every row
(`MASKED`). Built and judged out of sample in research (scratchpad fx_build.py / fx_post.py / fx_train.py
v2 / fx_eval.py; memory overnight-models-2026-10): 2025-26, survivorship-free, 275 trades +1.20 %/day vs the
v1 rule's 270 at +1.00 (+0.20, 95% -0.28..+0.67); its own out-of-sample phase May-Sep 2026 -0.51 vs v1.

THE INPUTS — each computed here by ONE function used both for the history (`sel_short.backfill_days`) and
for the live bar (`sel_short._score_chunk`), the research code verbatim:
  per bar, from the name's own regular-hours 30-minute series (`bar_extras`):
    sx_runup5   close vs the last close at or before the same ET clock time 5 sessions earlier, % (the
                riser rule's run-up, `sel_short._close_at`)
    sx_atr_own  ATR% / the max per-session ATR% over the previous 20 sessions (>= 10 with bars) - 1
    sx_rvol260  bar volume / mean of the previous 260 bars (>= 20) (`sel_short.rvol_at`)
  per session, known at its 08:30 ET cutoff (`session_extras`, the deep store's rules):
    dp_rsplit_n_730d / dp_rsplit_d_last   reverse splits in 730 days / days since the last (cap 730)
    dp_sec_d_def / _nt / _restate / _shell / _bk   days since an 8-K item 3.01 / an NT 10-K|10-Q|20-F /
                item 4.02 / item 5.06 / item 1.03 (cap 730; NaN beyond or never)
    dp_sec_n_off_365d   offering registrations / prospectuses in 365 days
  per bar, across the bar's TRADEABLE names (price >= $5, 20-session dollar volume >= $5M; `cross_section`):
    xrank_ret5 / ret21 / ret63 / vol20 / rsi14 / dvol   2*pct-1 (base inputs v1 was trained with NaN)
    sx_atr_xrank, sx_runup5_xrank   2*pct-1 of ATR% / sx_runup5
    sx_runup5_secrel   run-up minus the median run-up of its SIC major group (2 digits); groups of >= 5
    sx_atr_secrel      ATR% / the group's median ATR% - 1
A non-tradeable row keeps NaN cross-sectional inputs (never trained on, never picked). The cross-sectional
inputs are the reason a V2 bar is scored in one place (`sel_short.score_bar`), not per worker.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

E_BAR = ["sx_runup5", "sx_atr_own", "sx_rvol260"]
E_SESS = ["dp_rsplit_n_730d", "dp_rsplit_d_last", "dp_sec_d_def", "dp_sec_d_nt", "dp_sec_d_restate",
          "dp_sec_d_shell", "dp_sec_d_bk", "dp_sec_n_off_365d"]
E_XS = ["sx_atr_xrank", "sx_runup5_xrank", "sx_runup5_secrel", "sx_atr_secrel"]
EXTRA = E_BAR + E_SESS + E_XS
XRANK = (("ret_5", "xrank_ret5"), ("ret_21", "xrank_ret21"), ("ret_63", "xrank_ret63"),
         ("realized_vol_20", "xrank_vol20"), ("rsi_14", "xrank_rsi14"), ("dollar_vol_log", "xrank_dvol"))
XS_SOURCES = [src for src, _ in XRANK] + ["atr_pct_14", "sx_runup5"]
CROSS = [dst for _, dst in XRANK] + E_XS              # the inputs filled across the bar's names
MASKED = ("dp_earn_d_next", "dp_earn_surprise", "dp_an_net_30d", "dp_an_n_30d", "dp_an_pt_upside",
          "dp_an_d_last", "dp_an_pt_chg", "dp_an_ret_since")
NT_FORMS = ("NT 10-K", "NT 10-Q", "NT 20-F")
SECTOR_MIN_NAMES = 5
EPOCH = pd.Timestamp("1970-01-01")
NY = "America/New_York"
DAY_NS = 86_400 * 10 ** 9
RUNUP_SESSIONS = 5
OWN_ATR_SESSIONS = 20
OWN_ATR_MIN_SESSIONS = 10
RVOL_LOOKBACK_BARS = 260
RVOL_MIN_PRIOR_BARS = 20


def is_v2(meta: Optional[dict]) -> bool:
    """True when the installed model reads the V2 inputs (its model.json lists them)."""
    return bool((meta or {}).get("extra_features"))


_CAL: Dict[str, np.ndarray] = {}


def calendar_days() -> np.ndarray:
    """Market sessions (day numbers) from 2019-06-01, as the training rows' calendar."""
    if "c" not in _CAL:
        from src.signals.sel_short import is_session
        d, end, out = date(2019, 6, 1), max(date(2026, 12, 31), date.today() + timedelta(days=90)), []
        while d <= end:
            if is_session(d):
                out.append((pd.Timestamp(d) - EPOCH).days)
            d += timedelta(days=1)
        _CAL["c"] = np.asarray(out, np.int64)
    return _CAL["c"]


# ── per bar, from the name's own series ──────────────────────────────────────

def bar_extras(hlc, atr: np.ndarray, cal: Optional[np.ndarray] = None) -> np.ndarray:
    """``(n_bars, 3)``: sx_runup5, sx_atr_own, sx_rvol260 at every bar of ``hlc`` (naive-UTC bar starts),
    ``atr`` = the series' `atr_pct_14` column. Each value reads only bars at or before its own."""
    from src.analysis import deep_features as dfe
    cal = calendar_days() if cal is None else cal
    idx, _high, _low, close, vol = hlc
    n = len(idx)
    E = np.full((n, len(E_BAR)), np.nan, np.float64)
    if n == 0:
        return E
    c = np.asarray(close, dtype=float)
    v = np.asarray(vol, dtype=float)
    ts = pd.DatetimeIndex(idx)
    t_utc = ts.values.astype("datetime64[ns]")
    sday = dfe.session_days(ts)
    so = np.searchsorted(cal, sday)
    okc = (so < len(cal)) & (cal[np.minimum(so, len(cal) - 1)] == sday)
    # 1. run-up: the last close at or before the same ET clock time 5 market sessions earlier
    et = ts.tz_localize("UTC").tz_convert(NY)
    clock = (et - et.normalize()).to_numpy()
    pre = so - RUNUP_SESSIONS
    okp = okc & (pre >= 0)
    base = pd.to_datetime(np.where(okp, cal[np.maximum(pre, 0)], 0), unit="D") + pd.to_timedelta(clock)
    base_utc = pd.DatetimeIndex(base).tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward") \
        .tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")
    j = np.searchsorted(t_utc, base_utc, side="right") - 1
    ok = okp & (j >= 0) & ~pd.isna(base_utc)
    E[ok, 0] = (c[ok] / c[j[ok]] - 1.0) * 100.0
    # 2. ATR% against its own previous 20 sessions' max (>= 10 sessions with bars)
    atr = np.asarray(atr, dtype=float)
    if len(atr) == n and okc.any():
        o0 = int(so[okc].min())
        span = int(so[okc].max()) - o0 + 1
        per = pd.Series(atr[okc]).groupby(so[okc] - o0).max()
        dense = np.full(span, np.nan)
        dense[per.index.to_numpy()] = per.to_numpy()
        win = pd.Series(dense).rolling(OWN_ATR_SESSIONS, min_periods=OWN_ATR_MIN_SESSIONS).max().shift(1).to_numpy()
        w = np.full(n, np.nan)
        w[okc] = win[so[okc] - o0]
        with np.errstate(invalid="ignore", divide="ignore"):
            E[:, 1] = atr / w - 1.0
    # 3. relative volume: bar volume / mean of the previous 260 bars (>= 20)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(v))]
    i = np.arange(n)
    lo = np.maximum(0, i - RVOL_LOOKBACK_BARS)
    cnt = i - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cs[i] - cs[lo]) / np.maximum(cnt, 1)
        E[:, 2] = np.where((cnt >= RVOL_MIN_PRIOR_BARS) & (mean > 0), v / mean, np.nan)
    return E


# ── per session, known at its 08:30 ET cutoff ────────────────────────────────

_SPLITS: Dict[str, Dict[str, pd.DataFrame]] = {}


def splits_by_ticker(refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """The deep store's whole-market splits table, by ticker (memoised per process)."""
    if refresh or "t" not in _SPLITS:
        import duckdb
        from src.data import deep
        p = (Path(deep.DEEP_DIR) / "splits.parquet").as_posix()
        try:
            df = duckdb.connect().execute(f"select ticker, execution_date, split_from, split_to "
                                          f"from read_parquet('{p}')").fetchdf()
        except Exception:                                      # noqa: BLE001 — no table: no splits
            df = pd.DataFrame(columns=["ticker", "execution_date", "split_from", "split_to"])
        _SPLITS["t"] = {t: g for t, g in df.groupby("ticker")}
    return _SPLITS["t"]


SEC_COLUMNS = ["accession", "form", "acceptance", "filing_date", "items"]


def session_extras(tk: str, sdays: Sequence[int], splits: Optional[pd.DataFrame] = None,
                   sec: Optional[pd.DataFrame] = None) -> np.ndarray:
    """``(len(sdays), 8)`` E_SESS at each session's 08:30 ET cutoff — a row dated after the cutoff never
    counts. ``splits`` = the name's rows of the splits table (None: read it), ``sec`` = its SEC filings
    (None: read its deep-store part)."""
    from src.analysis import deep_features as dfe
    uniq = np.asarray(sdays, np.int64)
    S = np.full((len(uniq), len(E_SESS)), np.nan)
    if len(uniq) == 0:
        return S
    cut = dfe._et_ns(uniq, dfe.CUTOFF_ET_MIN)
    if splits is None:
        splits = splits_by_ticker().get(tk)
    if splits is not None and len(splits):
        sp = splits[(pd.to_numeric(splits["split_from"], errors="coerce")
                     > pd.to_numeric(splits["split_to"], errors="coerce"))]
        if len(sp):
            t = np.sort(dfe.midnight_ns(dfe.to_days(sp["execution_date"])))
            S[:, 0] = dfe._count(t, cut, 730 * DAY_NS)
            S[:, 1] = dfe._days_since(t, cut, 730)
        else:
            S[:, 0] = 0.0
    else:
        S[:, 0] = 0.0
    if sec is None:
        sec = dfe._part("sec_filings", tk, SEC_COLUMNS)
    if sec is not None and not sec.empty:
        t = dfe.to_ns(sec["acceptance"])
        fb = dfe._known_from_days(dfe.to_days(sec["filing_date"]), 1)
        t = np.where(t == np.iinfo(np.int64).min, fb, t)
        form = sec["form"].fillna("").astype(str).to_numpy()
        items = [set(s.split(",")) for s in sec["items"].fillna("").astype(str)]
        is8k = np.isin(form, ["8-K", "8-K/A"])

        def item(code):
            return is8k & np.array([code in s for s in items], dtype=bool)

        def ev(m):
            return np.sort(t[m])
        S[:, 2] = dfe._days_since(ev(item("3.01")), cut, 730)
        S[:, 3] = dfe._days_since(ev(np.array([f.startswith(NT_FORMS) for f in form], dtype=bool)), cut, 730)
        S[:, 4] = dfe._days_since(ev(item("4.02")), cut, 730)
        S[:, 5] = dfe._days_since(ev(item("5.06")), cut, 730)
        S[:, 6] = dfe._days_since(ev(item("1.03")), cut, 730)
        S[:, 7] = dfe._count(ev(np.array([f.startswith(dfe.SEC_OFFERING_PREFIX) for f in form], dtype=bool)),
                             cut, 365 * DAY_NS)
    return S


def day_extras(names: Iterable[str], day_number: int, workers: int = 8) -> Dict[str, List[float]]:
    """``{ticker: the 8 E_SESS values}`` for one session (threads: one deep-store part read per name)."""
    from concurrent.futures import ThreadPoolExecutor
    names = list(names)
    spl = splits_by_ticker(refresh=True)

    def one(tk):
        try:
            return tk, [float(x) for x in session_extras(tk, [int(day_number)], splits=spl.get(tk))[0]]
        except Exception:                                      # noqa: BLE001 — unknown, as a missing part
            return tk, [float("nan")] * len(E_SESS)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return dict(ex.map(one, names))


# ── per bar, across the bar's tradeable names ────────────────────────────────

_SIC: Dict[str, Dict[str, int]] = {}


def sic2_map(refresh: bool = False) -> Dict[str, int]:
    """ticker -> SIC major group (the code's first two characters, as the training rows read it)."""
    if refresh or "m" not in _SIC:
        import duckdb
        from src.data import deep
        p = (Path(deep.DEEP_DIR) / "ticker_details.parquet").as_posix()
        out: Dict[str, int] = {}
        try:
            for t, v in duckdb.connect().execute(f"select ticker, cast(sic_code as varchar) from read_parquet('{p}') "
                                                 f"where sic_code is not null").fetchall():
                try:
                    out[str(t)] = int(str(v).split(".")[0][:2])
                except ValueError:
                    pass
        except Exception:                                      # noqa: BLE001
            pass
        _SIC["m"] = out
    return _SIC["m"]


def cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """The 10 cross-sectional inputs (`CROSS`) for every row of ``df`` — columns ``run`` (the bar: rows
    of one run are ranked together), ``tradeable`` (bool), the `XS_SOURCES` and ``sic2`` (-1 unknown).
    Non-tradeable rows get NaN. Sources are read as float32, as the training matrix held them."""
    out = pd.DataFrame(np.nan, index=df.index, columns=CROSS, dtype=float)
    t = df[df["tradeable"].astype(bool)]
    if t.empty:
        return out
    g0 = pd.DataFrame({"run": t["run"].to_numpy()}, index=t.index)

    def xrank(vals):
        g0["v"] = vals
        return (2.0 * g0.groupby("run")["v"].rank(pct=True) - 1.0).to_numpy(np.float32)

    def f32(col):
        return t[col].to_numpy(np.float64).astype(np.float32).astype(float)
    for src, dst in XRANK:
        out.loc[t.index, dst] = xrank(f32(src))
    atr, ru = f32("atr_pct_14"), f32("sx_runup5")
    out.loc[t.index, "sx_atr_xrank"] = xrank(atr)
    out.loc[t.index, "sx_runup5_xrank"] = xrank(ru)
    g = pd.DataFrame({"run": t["run"].to_numpy(), "sic": t["sic2"].to_numpy(np.int64), "atr": atr, "ru": ru},
                     index=t.index)
    g = g[g["sic"] >= 0]
    if len(g):
        grp = g.groupby(["run", "sic"])
        ok = (grp["atr"].transform("count") >= SECTOR_MIN_NAMES).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[g.index, "sx_runup5_secrel"] = np.where(
                ok, g["ru"].to_numpy() - grp["ru"].transform("median").to_numpy(), np.nan).astype(np.float32)
            out.loc[g.index, "sx_atr_secrel"] = np.where(
                ok, g["atr"].to_numpy() / grp["atr"].transform("median").to_numpy() - 1.0, np.nan).astype(np.float32)
    return out


def apply_mask(M: np.ndarray, feats: Sequence[str], masked: Optional[Sequence[str]] = None) -> np.ndarray:
    """Blank the inputs the model was trained without (NaN in every training row)."""
    cols = [i for i, f in enumerate(feats) if f in set(MASKED if masked is None else masked)]
    if cols:
        M[:, cols] = np.nan
    return M


def read_day_extras(path: Path) -> Optional[Dict[str, List[float]]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None
