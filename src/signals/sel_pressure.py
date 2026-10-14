"""BORROW PRESSURE for the vol arm's extra shorts (PREREG71's V6; user directive 2026-10-10: "Deploy the narrow part
that is working in live").

The no-file pressure model NF (``cache/ml/sel_short/pressure/nf_model.txt`` + ``nf_model.json``; research
`studies/prereg71/nf71.py`): LightGBM on PREREG69's label — a stock lendable at the open becomes UNLENDABLE (under
$10,000 available, or gone from IBKR's file) within 5 sessions — over PREREG69's population (the 200 most volatile
lendable stocks each session, 2017-10 .. 2024-06, the delisted included), from 22 inputs that need NO IBKR file: the
daily bars to the previous close (price, 20-session dollar volume, 14-session ATR%, 5- / 10-session run-up, relative
volume, the name's daily-ATR rank among the session's tradeable stocks), FINRA short interest (days to cover and its
change), SEC fails-to-deliver, the Reg SHO lists, reverse splits, the listing age, and IBKR's daily fee (the previous
session's, its 5-session change and 20-session max, its status, the session's median over its 20 most volatile
tradeable names — the closest stand-in, known at the open, for the vol arm's candidates the model was tested on: the
HIGH flags agree with the tested scores on 98% of the candidates of 14 random set-B days, against 96% with the 200 most
volatile). Judged on PREREG69's unseen years 2019 .. 2024-06: AUC 0.79-0.91 (pooled 0.86). HIGH = at or above the cut
in the model's meta (its 90th percentile on its own training predictions).

Every input is known at the session's open, so the day's scores are computed ONCE (`build`, run by the sel_short
prepare; the entry step builds them if they are missing) for every name that traded the previous session at >= $5 and
>= $1M a day, and stored in ``pressure/scores/<day>.json``. The daily bars are Polygon's grouped daily bars UNADJUSTED
(the model's construction: ratios masked after a split), cached per session in ``pressure/grouped/``. IBKR's daily fee
is its API bar (`borrow_history/fee_api_daily.parquet`, the model's training source) where one exists for the date, else
the close of our own archive of IBKR's file (`borrow_daily.parquet`; the two matched within 0.004 %/yr).

The rule it serves (`tracker.record_sel_short_trades`, `enable_sel_short_vol_pressure_extras`): on a bar where the vol
arm has a live pick, each of the bar's other candidates that pass the whole rule (the journal's ``backups``) and are
HIGH is shorted too — set B (2024-06-25 .. 2026-09-24, out of sample): the extras +2.42 %/day (95% +0.41 .. +4.42,
Holm p 0.033), the account +58.4 vs +54.4 %/yr.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

EPOCH = pd.Timestamp("1970-01-01")
LOOKBACK = 60                       # sessions of daily bars and fees read before the day (the inputs need <= 41)
TOP_POP = 20                        # the session's median fee: over its 20 most volatile tradeable names
MIN_PRICE, MIN_DV = 5.0, 5e6        # the model's tradeable stocks (its daily-ATR rank's universe)
SCORE_MIN_DV = 1e6                  # the names scored: >= $5 and >= $1M a day at the previous close
LEND_USD = 1e4
# the security types the model's universe leaves out (PREREG68's construction)
FUND_TYPES = {"ETF", "ETN", "ETV", "ETS", "FUND", "SP", "UNIT", "WARRANT", "RIGHT", "PFD", "BOND", "BASKET", "LT", "GDR",
              "OS", "NYRS", "AGEN", "EQLK", "ETD", "IGC"}
INPUTS = ["rk", "log_price", "log_dv20", "log_atr", "runup5", "runup10", "log_rvol", "log1p_dtc", "si_change",
          "ftd_ratio", "regsho_on", "regsho_streak", "regsho_60", "days_since_reverse_split", "days_listed",
          "api_fee_prev", "api_fee_chg5", "api_fee_max20", "api_listed", "api_not_listed", "api_unknown",
          "day_median_api_fee"]
_CACHE: Dict[str, dict] = {}        # one day's scores per process


def root() -> Path:
    return Path(settings.sel_short_dir) / "pressure"


def deep_dir() -> Path:
    from src.data.deep import DEEP_DIR
    return Path(DEEP_DIR)


def dnum(d) -> int:
    return int((pd.Timestamp(d) - EPOCH).days)


def _iso(d: date) -> str:
    return d.isoformat()


def enabled() -> bool:
    return bool(getattr(settings, "enable_sel_short_vol_pressure_extras", False))


def load_model():
    """(the LightGBM booster, its meta: ``inputs``, ``hi_cut``, ``trained_through``)."""
    import lightgbm as lgb
    meta = json.loads((root() / "nf_model.json").read_text(encoding="utf-8"))
    if list(meta.get("inputs") or []) != INPUTS:
        raise ValueError(f"the pressure model's inputs {meta.get('inputs')} are not this module's {INPUTS}")
    return lgb.Booster(model_file=str(root() / "nf_model.txt")), meta


# ── the daily bars (Polygon's grouped daily, UNADJUSTED) ──────────────────────────────────────────────────

def _symbol(raw: str) -> Optional[str]:
    """Polygon's symbol in the project's form (BRK.B -> BRK-B); None for a lowercase class symbol (preferreds,
    rights: TpC, PLMr), which upper-cased would merge into a common stock (PREREG69 A1)."""
    s = str(raw or "").strip()
    if not s or any(ch.islower() for ch in s):
        return None
    s = s.upper()
    return s[:-2] + "-" + s[-1] if len(s) > 2 and s[-2] == "." and s[-1].isalpha() else s


def grouped_ohlc(day: date, fetch: bool = True) -> Optional[pd.DataFrame]:
    """The session's daily bars of every US stock (sym in the project's form, o h l c v), cached per session; None
    when Polygon has none (a holiday: an empty marker is cached) or cannot be reached."""
    p = root() / "grouped" / f"{_iso(day)}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    if p.with_suffix(".empty").exists():
        return None
    if not fetch:
        return None
    from src.data import polygon_client as pc
    rows = pc.get_grouped_daily(_iso(day), adjusted=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if day < date.today() - timedelta(days=3):         # a past session with no bars: a holiday, cached as such
            p.with_suffix(".empty").write_text("", encoding="utf-8")
        return None
    raw = pd.DataFrame(rows)
    df = pd.DataFrame({"sym": raw["T"].map(_symbol), "o": pd.to_numeric(raw.get("o"), errors="coerce"),
                       "h": pd.to_numeric(raw.get("h"), errors="coerce"), "l": pd.to_numeric(raw.get("l"), errors="coerce"),
                       "c": pd.to_numeric(raw.get("c"), errors="coerce"), "v": pd.to_numeric(raw.get("v"), errors="coerce")})
    df = df[df.sym.notna()].drop_duplicates("sym").reset_index(drop=True)
    tmp = p.with_suffix(".tmp")
    df.to_pickle(tmp)
    os.replace(tmp, p)
    return df


def panel(d: date, sessions: Sequence[date], fetch: bool = True) -> pd.DataFrame:
    """The daily bars of ``sessions`` (strictly before ``d``), one row per (sym, session)."""
    parts = []
    for s in sessions:
        if s >= d:
            raise ValueError(f"the pressure panel of {d} must not read the session {s}")
        g = grouped_ohlc(s, fetch=fetch)
        if g is not None and len(g):
            parts.append(g.assign(d=np.int64(dnum(s))))
    if not parts:
        return pd.DataFrame(columns=["sym", "o", "h", "l", "c", "v", "d"])
    B = pd.concat(parts, ignore_index=True)
    return B.drop_duplicates(["sym", "d"]).sort_values(["sym", "d"]).reset_index(drop=True)


def splits_table() -> pd.DataFrame:
    import duckdb
    f = deep_dir() / "splits.parquet"
    if not f.exists():
        return pd.DataFrame(columns=["ticker", "d", "split_from", "split_to"])
    S = duckdb.connect().execute(f"select ticker, execution_date, split_from, split_to from '{f.as_posix()}'").df()
    S["d"] = (pd.to_datetime(S.execution_date) - EPOCH).dt.days.astype(np.int64)
    return S


def bar_features(B: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """PREREG68's daily features on each row's own close (shifted by the caller): 20-session mean dollar volume,
    the 14-session ATR% (a true range over the previous close), the 5- / 10-session log run-up, the relative volume —
    the run-ups masked within 16 days of a split, the ATR% within 30 (unadjusted bars)."""
    B = B.copy()
    g = B.groupby("sym", sort=False)
    prev_c = g["c"].shift(1)
    tr = np.maximum(B.h - B.l, np.maximum((B.h - prev_c).abs(), (B.l - prev_c).abs()))
    B["dv"] = B.c * B.v
    B["dv20"] = g["dv"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    B["atr14"] = tr.groupby(B.sym).transform(lambda s: s.rolling(14, min_periods=10).mean()) / B.c * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        B["ru5"] = np.log(B.c / g["c"].shift(5))
        B["ru10"] = np.log(B.c / g["c"].shift(10))
    B["rvol"] = B.v / g["v"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    sp = {t: np.sort(x.d.to_numpy()) for t, x in splits.groupby("ticker")}
    dd_all = B.d.to_numpy()
    m10, m20 = np.zeros(len(B), bool), np.zeros(len(B), bool)
    for sym, idx in B.groupby("sym", sort=False).indices.items():
        d = sp.get(sym)
        if d is None:
            continue
        dd = dd_all[idx]
        j = np.searchsorted(d, dd, side="right") - 1
        last = np.where(j >= 0, d[np.maximum(j, 0)], -10**9)
        m10[idx] = (dd - last) <= 16
        m20[idx] = (dd - last) <= 30
    B.loc[m10, ["ru5", "ru10"]] = np.nan
    B.loc[m20, ["atr14"]] = np.nan
    return B


def _types() -> Dict[str, str]:
    """The security type of every symbol (Polygon's sweep, then the delisted names' own)."""
    import duckdb
    out: Dict[str, str] = {}
    try:
        from src.data import company_names as cn
        raw = json.loads(cn._sec_type_path().read_text(encoding="utf-8"))
        out = {(_symbol(k) or k): str(v).upper() for k, v in (raw.get("types") or {}).items()}
    except Exception as e:                                       # noqa: BLE001 — no table: every name kept
        logger.warning(f"[sel_pressure] no security-type table ({e})")
    f = deep_dir() / "delisted.parquet"
    if f.exists():
        for t, ty in duckdb.connect().execute(f"select ticker, type from '{f.as_posix()}'").fetchall():
            k = _symbol(t) or str(t).upper()
            out.setdefault(k, str(ty or "").upper())
    return out


# ── IBKR's daily fee: its API bars, our own archive's closes where the API has none ──────────────────────────

def fee_frames(names: Sequence[str], lo: date, hi: date) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]],
                                                                  Dict[str, Tuple[int, int]]]:
    """({name: (day numbers, fee %/yr)} over [lo, hi], {name: (first, last) day numbers the name is COVERED}):
    IBKR's API bar of a date, else our archive's close of that ET date. A name is covered by the API's own range
    (status ok) and, from our archive, on the days its rows span."""
    import duckdb
    con = duckdb.connect()
    con.register("tk", pd.DataFrame({"t": [str(n).upper() for n in names]}))
    bh = deep_dir() / "borrow_history"
    frames = []
    api_f = bh / "fee_api_daily.parquet"
    if api_f.exists():
        A = con.execute(f"""select upper(ticker) t, cast(date as date) dt, fee from '{api_f.as_posix()}'
                            where upper(ticker) in (select t from tk) and cast(date as date) between ? and ?""",
                        [lo, hi]).df()
        A["src"] = 0
        frames.append(A)
    own_f = deep_dir() / "borrow_daily.parquet"
    if own_f.exists():
        O = con.execute(f"""select upper(ticker) t, cast(date as date) dt, fee from '{own_f.as_posix()}'
                            where upper(ticker) in (select t from tk) and cast(date as date) between ? and ?""",
                        [lo, hi]).df()
        O["src"] = 1
        frames.append(O)
    fees: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cover: Dict[str, Tuple[int, int]] = {}
    if frames:
        F = pd.concat(frames, ignore_index=True)
        F = F[np.isfinite(pd.to_numeric(F.fee, errors="coerce"))]
        F["d"] = (pd.to_datetime(F.dt) - EPOCH).dt.days.astype(np.int64)
        F = F.sort_values(["t", "d", "src"]).drop_duplicates(["t", "d"], keep="first")   # the API's bar first
        for t, x in F.groupby("t"):
            fees[t] = (x.d.to_numpy(np.int64), x.fee.to_numpy(float))
        for t, x in F[F.src == 1].groupby("t"):
            cover[t] = (int(x.d.min()), int(x.d.max()))
    st_f = bh / "fee_api_status.parquet"
    if st_f.exists():
        S = con.execute(f"""select upper(ticker) t, first, last from '{st_f.as_posix()}'
                            where status = 'ok' and upper(ticker) in (select t from tk)""").df()
        for r in S.itertuples(index=False):
            try:
                a, b = dnum(pd.Timestamp(r.first)), dnum(pd.Timestamp(r.last))
            except (TypeError, ValueError):
                continue
            c0 = cover.get(r.t)
            cover[r.t] = (min(a, c0[0]), max(b, c0[1])) if c0 else (a, b)
    return fees, cover


# ── the model's inputs ────────────────────────────────────────────────────────────────────────────────────

def inputs(d: date, sessions: Optional[Sequence[date]] = None, fetch: bool = True,
           frames: Optional[dict] = None) -> pd.DataFrame:
    """The 22 inputs of every name scored on session ``d`` — each read as of the open of ``d``: the bars of the
    sessions before it, FINRA's report settled 10+ days before, fails-to-deliver settled 15-45 days before, the
    Reg SHO lists of the sessions before, IBKR's fee of the sessions before. ``frames`` (tests): the deep tables
    and the fees already loaded ({"si", "ftd", "regsho", "splits", "details", "ipos", "fees", "cover", "types"})."""
    import duckdb
    sessions = list(sessions) if sessions is not None else _sessions_before(d, LOOKBACK)
    B = panel(d, sessions, fetch=fetch)
    if not len(B):
        return pd.DataFrame(columns=["ticker"] + INPUTS)
    frames = frames or {}
    splits = frames["splits"] if "splits" in frames else splits_table()
    last_day = int(B.d.max())
    keep = set(B.sym[(B.d == last_day) & (B.c >= MIN_PRICE)])     # only a name at >= $5 the session before can count
    B = bar_features(B[B.sym.isin(keep)], splits)
    last = B.groupby("sym", sort=False).tail(1)                    # each name's last bar before d: the "_p" values
    last = last[last.d == last_day]                                # traded on the session before
    P = pd.DataFrame({"ticker": last.sym.to_numpy(), "c_p": last.c.to_numpy(float), "dv20_p": last.dv20.to_numpy(float),
                      "atr14_p": last.atr14.to_numpy(float), "ru5_p": last.ru5.to_numpy(float),
                      "ru10_p": last.ru10.to_numpy(float), "rvol_p": last.rvol.to_numpy(float)})
    types = frames["types"] if "types" in frames else _types()
    fund = P.ticker.map(lambda s: types.get(s, "") in FUND_TYPES).to_numpy(bool)
    P["elig"] = ((P.c_p >= MIN_PRICE) & (P.dv20_p >= MIN_DV) & P.atr14_p.notna()).to_numpy(bool) & ~fund
    atr_e = np.sort(P.atr14_p[P.elig].to_numpy(float))[::-1]
    P["rk"] = [1 + int(np.sum(atr_e > a)) if np.isfinite(a) else np.nan for a in P.atr14_p.to_numpy(float)]
    P = P[(P.c_p >= MIN_PRICE) & (P.dv20_p >= SCORE_MIN_DV)].reset_index(drop=True)
    n = len(P)
    D = dnum(d)
    syms = P.ticker.to_numpy()
    con = duckdb.connect()
    con.register("tk", pd.DataFrame({"ticker": list(syms)}))
    deep = deep_dir()

    def table(key, sql):
        if key in frames:
            return frames[key]
        return con.execute(sql).df()
    # FINRA short interest: the last report settled at least 10 days before
    SI = table("si", f"select ticker, settlement_date, short_interest, days_to_cover from "
                     f"'{(deep / 'short_interest.parquet').as_posix()}' where ticker in (select ticker from tk)")
    SI["dn"] = (pd.to_datetime(SI.settlement_date) - EPOCH).dt.days.astype(np.int64)
    si = {t: (x.dn.to_numpy(), x.days_to_cover.to_numpy(float), x.short_interest.to_numpy(float))
          for t, x in SI.sort_values(["ticker", "dn"]).groupby("ticker")}
    dtc, sich = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        z = si.get(syms[i])
        if z is None:
            continue
        dd, c, s_ = z
        j = np.searchsorted(dd, D - 10, side="right") - 1
        if j >= 0:
            dtc[i] = np.log1p(max(c[j], 0.0)) if np.isfinite(c[j]) else np.nan
            if j >= 1 and s_[j - 1] > 0 and s_[j] > 0:
                sich[i] = np.log(s_[j] / s_[j - 1])
    P["log1p_dtc"], P["si_change"] = dtc, sich
    # SEC fails-to-deliver settled D-45 .. D-15, against the 20-session share volume
    FT = table("ftd", f"""select symbol, cast(settlement_date as date) sd, sum(quantity) q from
                          '{(deep / 'ftd.parquet').as_posix()}' where symbol in (select ticker from tk)
                          and settlement_date >= '{(pd.Timestamp(d) - pd.Timedelta(days=60)).date()}' group by 1, 2""")
    FT["dn"] = (pd.to_datetime(FT.sd) - EPOCH).dt.days.astype(np.int64)
    ft = {t: (x.dn.to_numpy(), np.cumsum(x.q.to_numpy(float))) for t, x in FT.sort_values(["symbol", "dn"]).groupby("symbol")}
    vol20 = (P.dv20_p / P.c_p).to_numpy(float)
    ftd = np.zeros(n)
    for i in range(n):
        z = ft.get(syms[i])
        if z is None:
            continue
        dd, cq = z
        a, b = np.searchsorted(dd, D - 45), np.searchsorted(dd, D - 15, side="right")
        tot = (cq[b - 1] if b > 0 else 0.0) - (cq[a - 1] if a > 0 else 0.0)
        ftd[i] = np.log1p(max(tot, 0.0) / max(np.nan_to_num(vol20[i], nan=1.0), 1.0))
    P["ftd_ratio"] = ftd
    # IBKR's daily fee of the session before, its 5-session change and 20-session max, its status
    cal = np.array(sorted(dnum(s) for s in sessions), np.int64)    # the sessions before d (the fee calendar)
    if "fees" in frames:
        fees, cover = frames["fees"], frames.get("cover", {})
    else:
        fees, cover = fee_frames(list(syms), sessions[0] - timedelta(days=10), sessions[-1])
    prev, chg5, mx20 = (np.full(n, np.nan) for _ in range(3))
    st = np.full(n, 2, np.int8)
    ip = len(cal) - 1
    if ip >= 20:
        tgt = cal[ip]
        for i in range(n):
            t = str(syms[i]).upper()
            z = fees.get(t)
            if z is None:
                continue
            dd, f = z
            cv = cover.get(t)
            j = np.searchsorted(dd, tgt)
            hit = j < len(dd) and dd[j] == tgt
            if hit:
                st[i], prev[i] = 0, np.log1p(max(f[j], 0.0))
            elif cv and cv[0] <= tgt <= cv[1]:
                st[i] = 1
            a, b = np.searchsorted(dd, cal[ip - 19]), np.searchsorted(dd, tgt, side="right")
            if b > a:
                mx20[i] = np.log1p(max(f[a:b].max(), 0.0))
            j5 = np.searchsorted(dd, cal[ip - 5], side="right") - 1
            if j5 >= 0 and dd[j5] >= cal[ip - 10] and np.isfinite(prev[i]):
                chg5[i] = prev[i] - np.log1p(max(f[j5], 0.0))
    P["api_fee_prev"], P["api_fee_chg5"], P["api_fee_max20"] = prev, chg5, mx20
    P["api_listed"], P["api_not_listed"], P["api_unknown"] = (st == 0) * 1.0, (st == 1) * 1.0, (st == 2) * 1.0
    # the session's median fee over its most volatile tradeable names (TOP_POP)
    pop = P[P.elig].nsmallest(TOP_POP, "rk")
    med = float(np.nanmedian(pop.api_fee_prev)) if np.isfinite(pop.api_fee_prev).any() else np.nan
    P["day_median_api_fee"] = med
    # Reg SHO: on a threshold list the session before, its streak, list days in the 60 sessions before
    RS = table("regsho", f"select distinct symbol, date from '{(deep / 'regsho.parquet').as_posix()}' "
                         f"where date >= '{(pd.Timestamp(d) - pd.Timedelta(days=120)).date()}'")
    RS["dn"] = (pd.to_datetime(RS.date) - EPOCH).dt.days.astype(np.int64)
    rs = {(_symbol(s) or str(s).upper()): set(x.dn.to_numpy().tolist()) for s, x in RS.groupby("symbol")}
    win = cal[-60:]
    on_, streak, cnt = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if D > dnum(date(2021, 1, 4)) + 90:
        for i in range(n):
            listed = rs.get(syms[i], set())
            hit = np.array([w in listed for w in win], bool)
            cnt[i] = hit.sum()
            on_[i] = float(hit[-1]) if len(hit) else 0.0
            k = 0
            for h_ in hit[::-1]:
                if not h_:
                    break
                k += 1
            streak[i] = k
    P["regsho_on"], P["regsho_streak"], P["regsho_60"] = on_, streak, cnt
    # the days since the last reverse split, the listing age
    rv = splits[splits.split_from > splits.split_to]
    rvd = {t: np.sort(x.d.to_numpy()) for t, x in rv.groupby("ticker")}
    srv = np.full(n, np.nan)
    for i in range(n):
        z = rvd.get(syms[i])
        if z is not None:
            j = np.searchsorted(z, D, side="right") - 1
            if j >= 0:
                srv[i] = min(D - z[j], 2000)
    P["days_since_reverse_split"] = srv
    if "details" in frames:
        L = frames["details"]
    else:
        L = con.execute(f"""select ticker, min(d) d from (
                select ticker, cast(list_date as date) d from '{(deep / 'ticker_details.parquet').as_posix()}' where list_date is not null
                union all select ticker, cast(listing_date as date) d from '{(deep / 'ipos.parquet').as_posix()}'
                where listing_date is not null) group by ticker""").df()
    lst = {t: dnum(x) for t, x in zip(L.ticker, L.d)}
    P["days_listed"] = [min(D - lst[t], 5000) if t in lst and lst[t] <= D else np.nan for t in syms]
    P["log_price"] = np.log(np.maximum(P.c_p.to_numpy(float), 0.01))
    P["log_dv20"] = np.log(np.maximum(P.dv20_p.to_numpy(float), 1.0))
    P["log_atr"] = np.log(np.maximum(P.atr14_p.to_numpy(float), 1e-3))
    P["runup5"], P["runup10"] = np.clip(P.ru5_p, -2, 3), np.clip(P.ru10_p, -2, 3)
    P["log_rvol"] = np.log(np.maximum(P.rvol_p.to_numpy(float), 1e-3))
    return P[["ticker"] + INPUTS + ["c_p", "dv20_p"]]


def _sessions_before(d: date, n: int):
    from src.signals.sel_short import sessions_before
    return sessions_before(d, n)


def scores_path(d: date) -> Path:
    return root() / "scores" / f"{_iso(d)}.json"


def build(d: date, fetch: bool = True) -> dict:
    """The day's pressure of every name scored, written once (``pressure/scores/<day>.json``)."""
    t0 = time.time()
    bst, meta = load_model()
    X = inputs(d, fetch=fetch)
    p = bst.predict(X[INPUTS].to_numpy(float)) if len(X) else np.array([])
    cut = float(meta["hi_cut"])
    out = {"day": _iso(d), "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": meta.get("name", "nf"), "cut": cut, "n": int(len(X)),
           "high": int(np.sum(p >= cut)), "seconds": 0.0,
           "scores": {str(t): round(float(v), 5) for t, v in zip(X.ticker, p)}}
    out["seconds"] = round(time.time() - t0, 1)
    path = scores_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, path)
    _CACHE[_iso(d)] = out
    logger.info(f"[sel_pressure] {d}: {out['n']} names scored, {out['high']} HIGH (cut {cut:.3f}) ({out['seconds']}s)")
    return out


def scores(d: date, build_missing: bool = True) -> Optional[dict]:
    """The day's scores (memoised per process); built when missing and ``build_missing``; None when unavailable."""
    k = _iso(d)
    if k in _CACHE:
        return _CACHE[k]
    p = scores_path(d)
    if p.exists():
        try:
            _CACHE[k] = json.loads(p.read_text(encoding="utf-8"))
            return _CACHE[k]
        except Exception as e:                                      # noqa: BLE001 — rebuilt below
            logger.warning(f"[sel_pressure] unreadable scores for {d} ({e})")
    if not build_missing:
        return None
    try:
        return build(d)
    except Exception as e:                                          # noqa: BLE001 — no pressure: no extras
        logger.error(f"[sel_pressure] {d}: the pressure scores could not be built ({type(e).__name__}: {e})")
        return None


def pressure_of(ticker: str, d: date, build_missing: bool = True) -> Tuple[Optional[float], bool]:
    """(the name's probability, HIGH) on session ``d``; (None, False) when it was not scored."""
    s = scores(d, build_missing=build_missing)
    if not s:
        return None, False
    v = s.get("scores", {}).get(str(ticker))
    if v is None:
        return None, False
    return float(v), bool(float(v) >= float(s.get("cut", 1.0)))
