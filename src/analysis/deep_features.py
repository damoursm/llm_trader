"""DEEP FEATURES — the deep history store (`cache/ml/deep`) as point-in-time
features for models served on 30-MINUTE bars (`ml_ohlcv` and successors; the
first LIVE consumer is the selection short, `signals/sel_short.py`).

The problem this solves (2026-09-23, user directive: "use these data sources as
features … the models are served intraday, so the features must be usable
alongside our OHLCV bars … make them relevant even if they are on a schedule
that is daily, weekly, etc."): the model rows are 30-minute regular-hours bars,
while the sources arrive at their own cadence — an SEC acceptance instant, a
news timestamp, a date-only analyst action, a twice-monthly short-interest
settlement, a quarterly 13F data set. Three rules make them one grid:

**1. One knowledge cutoff per SESSION: 08:30 ET.** Every row of session D sees
the sources exactly as they stood at D 08:30 ET — never later, whatever the
bar's own time. That is precisely what the live store holds when the model is
served: the scheduler refreshes the fast families at 08:30 ET (`preopen`
profile of `src/data/deep/refresh.py`) and everything else nightly, so a
training row and a live row of the same session are built from the same
information. A per-BAR cutoff would be finer, but the store is not refreshed
intraday, so a model trained on it would see same-day filings in training that
it never sees live — a train/serve skew that looks like skill. What happens
between 08:30 and the bar reaches the model through the bars themselves.

**2. Every source row carries a KNOWN-AT instant, from its publication lag,
not its event date** (`KNOWN_AT` below): an acceptance or publication instant
where the source has one; for date-only keys, local midnight after the date
plus the publisher's lag (short interest: 9 business days after settlement;
fails-to-deliver: 20 days after the half-month period; 13F data sets: 30 days
after the file's range; lobbying 45 / contracts 30 days, conservatively). A
row is visible to session D iff known_at <= D 08:30 ET. Insider transactions
are keyed on the filing DAY (+1): the EDGAR daily index that carries them is
final only after the day, which is what the pre-open refresh can fetch.

**3. Slow data is expressed as STATE + AGE + CHANGE, and the bar makes it
move.** A daily or weekly source enters as its latest level normalised to the
ticker's own history (log-ratio to its trailing baseline, so a provider's
volume regime change — Polygon news 2024-25 — cancels), its AGE (days since the
update or event, so the model learns decay), and its CHANGE (the update vs the
previous one). Event sources also leave an ANCHOR price (the last close known
at the event), and the bar-level features are the return from that anchor to
the CURRENT bar — "how much of the reaction has happened as of this bar" moves
every 30 minutes even though the event arrived once (`BAR_FEATURES`).

Market-level context (VIX, rates, credit, DIX/GEX, COT, FRED first prints) is
constant across tickers within a session: it cannot rank names by itself, only
condition how the per-name features act. It is its own group so it can be
ablated.

Fama-French factors are deliberately absent: the library publishes monthly with
a 1-2 month lag, so the live value would always be missing while training saw
it — the skew rule 1 exists to prevent.

Entry points: `ticker_snapshots` (one ticker, many sessions — training and the
serving snapshot share it), `market_context`, `bar_features`, `MarketTables`
(the whole-market families grouped per ticker), `build_session_snapshot` /
`load_session_snapshot` (the per-session file the live scorer reads).

A PRE-OPEN build has no bar of its own session in the 30-minute grid, so
`ticker_snapshots` appends that session bar-less (`RTH.with_session`) when it is
the market session right after the grid's last one — the previous close and
trailing volumes every price-dependent feature reads are then exactly the
training rows' (2026-09-26: without it a real 08:30 snapshot left them missing
on ~90% of names). A grid that is BEHIND is not stretched: the pre-open run
extends the 30-minute store first (`refresh.extend_bars_30m`).
"""
from __future__ import annotations

import copy
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data import deep

NY = "America/New_York"
CUTOFF_ET_MIN = 8 * 60 + 30               # 08:30 ET — the session snapshot's knowledge cutoff
DAY_NS = 86_400 * 10 ** 9
BAR_NS = 30 * 60 * 10 ** 9
EPOCH = np.datetime64("1970-01-01", "D")

# ── the feature set ──────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[str]] = {
    "xh": ["dp_xh_pre_ret", "dp_xh_pre_dv_rel", "dp_xh_pre_range", "dp_xh_post_ret", "dp_xh_post_dv_rel"],
    "sec": ["dp_sec_d_8k", "dp_sec_n_8k_30d", "dp_sec_d_earn", "dp_sec_n_mat_30d", "dp_sec_d_per",
            "dp_sec_n_off_30d", "dp_sec_n_13d_90d", "dp_sec_n_f4_7d", "dp_sec_n_144_30d"],
    "earn": ["dp_earn_d_next", "dp_earn_surprise"],
    "news": ["dp_news_n_1d", "dp_news_n_3d_rel", "dp_news_sent_3d", "dp_news_d_last", "dp_news_focus_3d"],
    "ins": ["dp_ins_buy_n_30d", "dp_ins_buy_rel_90d", "dp_ins_sell_rel_90d", "dp_ins_d_buy", "dp_ins_nbuyers_90d"],
    "an": ["dp_an_net_30d", "dp_an_n_30d", "dp_an_pt_chg", "dp_an_pt_upside", "dp_an_d_last"],
    "si": ["dp_si_dtc", "dp_si_pct_sh", "dp_si_chg", "dp_si_age"],
    "sv": ["dp_sv_ratio_1d", "dp_sv_ratio_5d_rel"],
    "dpi": ["dp_dpi_1d", "dp_dpi_5d_rel"],
    "wiki": ["dp_wiki_1d_rel", "dp_wiki_7d_rel"],
    "alt": ["dp_cong_net_90d", "dp_cong_d_last", "dp_lobby_log_365d", "dp_contract_rel_365d"],
    "div": ["dp_div_d_to_ex", "dp_div_yield"],
    "size": ["dp_mcap_log", "dp_shares_chg_1y", "dp_age_y"],
    "inst": ["dp_inst_n_log", "dp_inst_chg", "dp_inst_sh_pct", "dp_ftd_rel"],
    "fund": ["dp_fund_bm", "dp_fund_rev_g"],
    "rs": ["dp_rs_on", "dp_rs_streak", "dp_rs_n_60", "dp_rs_d_last"],
    "bw": ["dp_bw_fee", "dp_bw_avail_usd", "dp_bw_htb", "dp_bw_fee_chg5", "dp_bw_avail_chg5", "dp_bw_fee_max20",
           "dp_bw_age"],
    "mkt": ["dp_mkt_vix", "dp_mkt_vix_ts", "dp_mkt_vix_chg5", "dp_mkt_spy_ret5", "dp_mkt_iwm_spy5",
            "dp_mkt_hyg_lqd5", "dp_mkt_tnx_chg5", "dp_mkt_dix", "dp_mkt_gex_z", "dp_mkt_t10y2y",
            "dp_mkt_nfci", "dp_mkt_cot_lev"],
    "bar": ["dp_bar_index", "dp_xh_rth_vs_pre", "dp_earn_ret_since", "dp_an_ret_since", "dp_ins_ret_since"],
}
MARKET_FEATURES: List[str] = list(FEATURE_GROUPS["mkt"])
BAR_FEATURES: List[str] = list(FEATURE_GROUPS["bar"])
SNAPSHOT_FEATURES: List[str] = [f for g, fs in FEATURE_GROUPS.items() if g not in ("mkt", "bar") for f in fs]
# anchors the bar features need; carried in the snapshot, never fed to a model
ANCHOR_COLUMNS: List[str] = ["_prev_close", "_xh_pre_last", "_earn_anchor", "_an_anchor", "_ins_anchor"]
DEEP_FEATURES: List[str] = SNAPSHOT_FEATURES + MARKET_FEATURES + BAR_FEATURES

# Per-source KNOWN-AT rules. A lag is not the publisher's lag alone: it is the
# point by which OUR STORE reliably holds the row, given when the scheduler
# refreshes that family (`src/data/deep/refresh.py`): the fast families
# re-fetch at 08:30 ET before every session (`preopen`), the rest nightly from
# 23:45 ET — and a nightly family may land before or after midnight, or be cut
# by the budget and finish the next night. Training on the publisher's lag
# while the store lags further would let the model learn from rows it never
# sees live.
LAG_DAYS = {
    "form345": 1,        # EDGAR daily index of day X is final after X; preopen fetches it
    "yf_analyst": 1,     # grade DATE only; yf refreshed nightly (DAILY cadence)
    "yf_earnings_surprise": 1,  # the day after the release: yf is nightly, a 07:00 release is not in the 08:30 store
    "short_interest_bdays": 10,  # FINRA disseminates ~8 business days after settlement; +2 for Polygon + preopen
    "short_volume": 1,   # FINRA daily file for X is out the same evening; preopen
    "quiver_dpi": 2,     # nightly (DAILY cadence); X may land after the nightly of X
    "wiki": 4,           # day X final on X+1, but the 2.4 h sweep runs last and can take two nights
    "quiver_congress": 3,  # ReportDate X; Quiver's own ingestion can trail the report date
    "quiver_lobbying": 45, "quiver_contracts": 30,
    "dividends": 1, "yf_shares": 1,
    "form13f_after_range": 30, "ftd_after_period": 25,
    "market": 1,         # daily market series / DIX / FRED first prints of D-1; preopen
    "cot_after_report": 4,  # Tuesday positions, released Friday 15:30 ET
    "regsho": 1,         # the list of D: NYSE ~22:00 / Nasdaq ~23:00 ET on D, Cboe ~03:05 ET on D+1; preopen
    "borrow": 1,         # the day's last archived IBKR file; a day is summarised only after it ends
}
KNOWN_AT: Dict[str, str] = {
    "bars30m_full": "bar END (start + 30 min); preopen fetches the pre-market bars",
    "sec_filings": "acceptance instant (filing_date + 1 day when missing); preopen",
    "yf_earnings": "scheduled date known in advance; the SURPRISE from the day after the release",
    "polygon_news": "published_utc instant; preopen",
    "form345": "filing_date + 1 day (the EDGAR daily index is final after the day)",
    "yf_analyst": "grade_date + 1 day",
    "short_interest": "settlement_date + 10 business days",
    "short_volume": "date + 1 day",
    "quiver_dpi": "Date + 2 days",
    "wiki": "date + 4 days",
    "quiver_congress": "ReportDate + 3 days",
    "quiver_lobbying": "Date + 45 days (quarterly LDA reports)",
    "quiver_contracts": "Date + 30 days (award reporting lag)",
    "dividends": "declaration_date + 1 day (ex_dividend_date when undeclared)",
    "yf_shares": "date + 1 day",
    "form13f": "range end of the SEC data-set file + 30 days",
    "ftd": "half-month period end + 25 days",
    "companyfacts": "acceptance of the fact's accession (filed + 1 day when unmatched)",
    "market_daily": "date + 1 day", "dix": "date + 1 day",
    "fred_vintages": "realtime_start + 1 day", "cot_tff": "report_date + 4 days (Friday release)",
    "regsho": "list date + 1 day (the last market posts D's list by ~03:05 ET on D+1)",
    "borrow_daily": "date + 1 day (the day's last file of our IBKR archive)",
}

SEC_MATERIAL_ITEMS = {"1.01", "1.02", "1.03", "2.01", "2.03", "2.05", "2.06", "3.01", "4.01", "4.02", "5.01", "5.02"}
SEC_PERIODIC = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "10-KT", "10-QT"}
SEC_OFFERING_PREFIX = ("424B", "S-1", "S-3", "F-1", "F-3")
SEC_13D = {"SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"}
_SENT = {"positive": 1.0, "bullish": 1.0, "very positive": 1.0, "slightly positive": 0.5,
         "cautiously positive": 0.5, "neutral/positive": 0.5, "neutral to positive": 0.5,
         "neutral/slightly positive": 0.5, "neutral": 0.0, "mixed": 0.0, "hold": 0.0,
         "negative": -1.0, "bearish": -1.0, "very negative": -1.0, "slightly negative": -0.5,
         "neutral/negative": -0.5}
_REV_CONCEPTS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
_EQ_CONCEPTS = ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")


# ── time helpers ─────────────────────────────────────────────────────────────

def _et_ns(days: np.ndarray, minutes: int) -> np.ndarray:
    """naive-UTC nanoseconds of local ET wall time ``minutes`` after midnight of
    each session day (days since epoch). DST-correct (no ambiguity at 00:00 or
    08:30)."""
    days = np.asarray(days, dtype=np.int64)
    if not len(days):
        return np.zeros(0, dtype=np.int64)
    u, inv = np.unique(days, return_inverse=True)
    wall = pd.to_datetime(u, unit="D") + pd.Timedelta(minutes=int(minutes))
    ns = wall.tz_localize(NY).tz_convert("UTC").tz_localize(None).asi8
    return ns[inv]


def cutoff_ns(days: np.ndarray) -> np.ndarray:
    """D 08:30 ET for each session day, as naive-UTC nanoseconds."""
    return _et_ns(days, CUTOFF_ET_MIN)


def midnight_ns(days: np.ndarray) -> np.ndarray:
    """D 00:00 ET — the known-at of a date-granular row published 'by day D'."""
    return _et_ns(days, 0)


def to_days(values) -> np.ndarray:
    """Dates / date strings / timestamps -> int days since epoch (the calendar
    DATE as written; NaT -> INT64_MIN)."""
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    out = s.values.astype("datetime64[D]").astype(np.int64)
    out[s.isna().to_numpy()] = np.iinfo(np.int64).min
    return out


def to_ns(values) -> np.ndarray:
    """Timestamps (naive UTC) -> int ns; NaT -> INT64_MIN."""
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    out = s.values.astype("datetime64[ns]").astype(np.int64)
    out[s.isna().to_numpy()] = np.iinfo(np.int64).min
    return out


def _known_from_days(days: np.ndarray, lag_days: int) -> np.ndarray:
    ok = days > np.iinfo(np.int64).min
    out = np.full(len(days), np.iinfo(np.int64).max, dtype=np.int64)       # unknown date: never visible
    if ok.any():
        out[ok] = midnight_ns(days[ok] + int(lag_days))
    return out


def session_days(idx_utc: pd.DatetimeIndex) -> np.ndarray:
    """ET calendar session day (days since epoch) of each naive-UTC bar start."""
    et = pd.DatetimeIndex(idx_utc).tz_localize("UTC").tz_convert(NY).tz_localize(None)
    return et.normalize().values.astype("datetime64[D]").astype(np.int64)


# ── vectorised as-of primitives (events sorted by known time) ────────────────

def _sorted(t: np.ndarray, *cols):
    o = np.argsort(t, kind="stable")
    return (t[o],) + tuple(np.asarray(c)[o] for c in cols)


def _last_idx(t: np.ndarray, cut: np.ndarray) -> np.ndarray:
    return np.searchsorted(t, cut, side="right") - 1


def _count(t: np.ndarray, cut: np.ndarray, w_ns: int) -> np.ndarray:
    return (np.searchsorted(t, cut, side="right") - np.searchsorted(t, cut - w_ns, side="right")).astype(float)


def _wsum(t: np.ndarray, v: np.ndarray, cut: np.ndarray, w_ns: int) -> np.ndarray:
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(np.asarray(v, float)))]
    return cs[np.searchsorted(t, cut, side="right")] - cs[np.searchsorted(t, cut - w_ns, side="right")]


def _wmean(t: np.ndarray, v: np.ndarray, cut: np.ndarray, w_ns: int) -> np.ndarray:
    v = np.asarray(v, float); fin = np.isfinite(v)
    s = _wsum(t[fin], v[fin], cut, w_ns); n = _count(t[fin], cut, w_ns)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _days_since(t: np.ndarray, cut: np.ndarray, cap: float) -> np.ndarray:
    i = _last_idx(t, cut); out = np.full(len(cut), np.nan)
    ok = i >= 0
    out[ok] = (cut[ok] - t[i[ok]]) / DAY_NS
    out[out > cap] = np.nan
    return out


def _asof(t: np.ndarray, v: np.ndarray, cut: np.ndarray, max_age_days: Optional[float] = None) -> np.ndarray:
    i = _last_idx(t, cut); out = np.full(len(cut), np.nan)
    ok = i >= 0
    if max_age_days is not None:
        ok &= (cut - np.where(ok, t[np.maximum(i, 0)], 0)) <= max_age_days * DAY_NS
    out[ok] = np.asarray(v, float)[i[ok]]
    return out


def _log_rel(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log1p(np.maximum(x, 0)) - np.log1p(np.maximum(base, 0))


# ── the ticker's regular-hours grid ──────────────────────────────────────────

class RTH:
    """The ticker's 30-minute regular-hours bars (`ml_dataset.hlc_30m`, the grid
    the model rows live on) and the per-session quantities the builders need:
    previous close, trailing dollar volume, and the price known at an instant."""

    def __init__(self, idx: pd.DatetimeIndex, close: np.ndarray, volume: np.ndarray):
        self.idx = pd.DatetimeIndex(idx)
        self.close = np.asarray(close, float)
        self.volume = np.nan_to_num(np.asarray(volume, float))
        self.end_ns = self.idx.asi8 + BAR_NS
        self.sday = session_days(self.idx)
        self.sessions, first = np.unique(self.sday, return_index=True)
        last = np.r_[first[1:], len(self.sday)] - 1
        self.first, self.last = first, last
        self.s_close = self.close[last]
        dv = np.add.reduceat(self.close * self.volume, first) if len(first) else np.zeros(0)
        self.s_dv = dv
        self.prev_close = np.r_[np.nan, self.s_close[:-1]]
        dvs = pd.Series(dv)
        self.dv20 = dvs.rolling(20, min_periods=5).mean().shift(1).to_numpy()
        self.s_vol = np.add.reduceat(self.volume, first) if len(first) else np.zeros(0)
        sh = pd.Series(self.s_vol)
        self.vol20 = sh.rolling(20, min_periods=5).mean().shift(1).to_numpy()   # shares/day

    def with_session(self, day: int) -> "RTH":
        """This grid with session ``day`` appended WITHOUT bars: a pre-open
        snapshot's own session, whose first regular-hours bar has not traded at
        the 08:30 cutoff. The per-session quantities the builders read at a
        session's position — the previous close, the trailing dollar and share
        volume (all from the sessions BEFORE it), and the session list the
        pre-market bars are matched on — are exactly the ones a grid holding the
        session's bars carries there, which is what every training row had.
        Without it a pre-open build finds no position for its own session and
        every price-dependent feature is missing (2026-09-26: the live 09-25
        snapshot carried the price-dependent features on ~9% of names — those
        whose tick cache already held a 09-25 bar — against ~50-95% built after
        the fact). No bar of the new session is ever read through it."""
        r = copy.copy(self)
        nb = len(self.close)
        r.sessions = np.r_[self.sessions, np.int64(day)]
        r.first = np.r_[self.first, nb].astype(self.first.dtype if len(self.first) else np.int64)
        r.last = np.r_[self.last, nb - 1].astype(self.last.dtype if len(self.last) else np.int64)
        r.s_close = np.r_[self.s_close, np.nan]
        r.s_dv = np.r_[self.s_dv, np.nan]
        r.s_vol = np.r_[self.s_vol, np.nan]
        r.prev_close = np.r_[self.prev_close, self.s_close[-1] if len(self.s_close) else np.nan]
        tail = lambda a: pd.Series(a).rolling(20, min_periods=5).mean().to_numpy()[-1] if len(a) else np.nan  # noqa: E731
        r.dv20 = np.r_[self.dv20, tail(self.s_dv)]
        r.vol20 = np.r_[self.vol20, tail(self.s_vol)]
        return r

    @classmethod
    def for_ticker(cls, ticker: str) -> Optional["RTH"]:
        from src.analysis.ml_dataset import hlc_30m
        h = hlc_30m(ticker)
        if h is None:
            return None
        idx, _hi, _lo, cl, vol = h
        return cls(idx, cl.to_numpy(float), vol.to_numpy(float))

    def price_at(self, t_ns: np.ndarray) -> np.ndarray:
        """The last regular-hours close KNOWN at each instant (bar END <= t)."""
        t_ns = np.asarray(t_ns, dtype=np.int64)
        j = np.searchsorted(self.end_ns, t_ns, side="right") - 1
        out = np.full(len(t_ns), np.nan)
        ok = (j >= 0) & (t_ns < np.iinfo(np.int64).max)
        out[ok] = self.close[j[ok]]
        return out

    def session_pos(self, days: np.ndarray) -> np.ndarray:
        """Index into ``self.sessions`` for each day (-1 when not a session)."""
        k = np.searchsorted(self.sessions, days)
        k = np.clip(k, 0, max(0, len(self.sessions) - 1))
        ok = len(self.sessions) > 0
        return np.where(ok & (self.sessions[k] == days), k, -1) if ok else np.full(len(days), -1)


# ── per-ticker parts ─────────────────────────────────────────────────────────

def _part(family: str, ticker: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    p = deep.DEEP_DIR / family / "parts" / f"{ticker.upper()}.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return deep.read_parquet(p, columns=columns)
    except Exception as e:                               # noqa: BLE001
        # a column the part does not carry (schema drift across providers) —
        # read everything and let the builder pick
        try:
            return deep.read_parquet(p)
        except Exception:                                # noqa: BLE001
            logger.debug(f"[deep_features] {family}/{ticker}: unreadable part ({e})")
            return pd.DataFrame()


# ── family builders: each fills its columns for every requested session ─────

def _xh(tk: str, days: np.ndarray, cut: np.ndarray, rth: RTH, pos: np.ndarray, out: Dict[str, np.ndarray]):
    df = _part("bars30m_full", tk, ["ts", "high", "low", "close", "volume", "session"])
    if df.empty or rth is None:
        return
    ts = to_ns(df["ts"]); end = ts + BAR_NS
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0).to_numpy(float)
    sess = df["session"].astype(str).to_numpy()
    d = session_days(pd.DatetimeIndex(ts.astype("datetime64[ns]")))
    # per-session aggregates over the ticker's own session list
    S = rth.sessions; nS = len(S)
    pre_last = np.full(nS, np.nan); pre_dv = np.zeros(nS); pre_hi = np.full(nS, np.nan); pre_lo = np.full(nS, np.nan)
    post_last = np.full(nS, np.nan); post_dv = np.zeros(nS)
    s_cut = cutoff_ns(S)
    k = np.searchsorted(S, d); k = np.clip(k, 0, max(0, nS - 1))
    on = (nS > 0) & (S[k] == d) if nS else np.zeros(len(d), bool)
    is_pre = on & (sess == "pre") & (end <= s_cut[k])          # pre-market bars known by the cutoff
    is_post = on & (sess == "post")
    if is_pre.any():
        kk = k[is_pre]; o = np.lexsort((ts[is_pre], kk))
        kk, cc, hh, ll, vv = kk[o], c[is_pre][o], hi[is_pre][o], lo[is_pre][o], v[is_pre][o]
        np.add.at(pre_dv, kk, cc * vv)
        lastm = np.r_[kk[1:] != kk[:-1], True]               # sorted by (session, time): each group's last bar
        pre_last[kk[lastm]] = cc[lastm]
        g = pd.DataFrame({"k": kk, "h": hh, "l": ll}).groupby("k")
        pre_hi[g.h.max().index.to_numpy()] = g.h.max().to_numpy()
        pre_lo[g.l.min().index.to_numpy()] = g.l.min().to_numpy()
    if is_post.any():
        kk = k[is_post]; o = np.lexsort((ts[is_post], kk))
        kk, cc, vv = kk[o], c[is_post][o], v[is_post][o]
        np.add.at(post_dv, kk, cc * vv)
        lastm = np.r_[kk[1:] != kk[:-1], True]
        post_last[kk[lastm]] = cc[lastm]
    base_pre = pd.Series(pre_dv).rolling(20, min_periods=5).median().shift(1).to_numpy()
    post_prev = np.r_[np.nan, post_last[:-1]]                # the PREVIOUS session's after-hours
    post_dv_prev = np.r_[np.nan, post_dv[:-1]]
    base_post = pd.Series(post_dv).rolling(20, min_periods=5).median().shift(2).to_numpy()
    ok = pos >= 0; p = pos[ok]
    pc = rth.prev_close[p]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dp_xh_pre_ret"][ok] = pre_last[p] / pc - 1.0
        out["dp_xh_pre_dv_rel"][ok] = _log_rel(pre_dv[p], base_pre[p])
        out["dp_xh_pre_range"][ok] = (pre_hi[p] - pre_lo[p]) / pc
        out["dp_xh_post_ret"][ok] = post_prev[p] / pc - 1.0
        out["dp_xh_post_dv_rel"][ok] = _log_rel(post_dv_prev[p], base_post[p])
    out["_xh_pre_last"][ok] = pre_last[p]


def _sec(tk: str, cut: np.ndarray, rth: Optional[RTH], out: Dict[str, np.ndarray]) -> Optional[pd.DataFrame]:
    df = _part("sec_filings", tk, ["accession", "form", "acceptance", "filing_date", "items"])
    if df.empty:
        return None
    t = to_ns(df["acceptance"])
    fb = _known_from_days(to_days(df["filing_date"]), 1)
    t = np.where(t == np.iinfo(np.int64).min, fb, t)
    form = df["form"].fillna("").astype(str).to_numpy()
    items = df["items"].fillna("").astype(str).to_numpy()
    is8k = np.isin(form, ["8-K", "8-K/A"])
    earn = is8k & np.array(["2.02" in s for s in items])
    mat = is8k & np.array([bool(SEC_MATERIAL_ITEMS & set(s.split(","))) for s in items])
    per = np.isin(form, list(SEC_PERIODIC))
    off = np.array([f.startswith(SEC_OFFERING_PREFIX) for f in form])
    d13 = np.isin(form, list(SEC_13D))
    f4 = np.isin(form, ["4", "4/A"])
    f144 = form == "144"
    W7, W30, W90 = 7 * DAY_NS, 30 * DAY_NS, 90 * DAY_NS

    def ev(m):
        return np.sort(t[m])
    t8, te = ev(is8k), ev(earn)
    out["dp_sec_d_8k"][:] = _days_since(t8, cut, 400)
    out["dp_sec_n_8k_30d"][:] = _count(t8, cut, W30)
    out["dp_sec_d_earn"][:] = _days_since(te, cut, 400)
    out["dp_sec_n_mat_30d"][:] = _count(ev(mat), cut, W30)
    out["dp_sec_d_per"][:] = _days_since(ev(per), cut, 400)
    out["dp_sec_n_off_30d"][:] = _count(ev(off), cut, W30)
    out["dp_sec_n_13d_90d"][:] = _count(ev(d13), cut, W90)
    out["dp_sec_n_f4_7d"][:] = _count(ev(f4), cut, W7)
    out["dp_sec_n_144_30d"][:] = _count(ev(f144), cut, W30)
    if rth is not None and len(te):
        i = _last_idx(te, cut); ok = (i >= 0)
        recent = ok & ((cut - te[np.maximum(i, 0)]) <= 30 * DAY_NS)
        out["_earn_anchor"][recent] = rth.price_at(te[i[recent]])
    return df.assign(_known=t)[["accession", "_known"]]


def _earn(tk: str, cut: np.ndarray, out: Dict[str, np.ndarray]):
    df = _part("yf_earnings", tk, ["event_ts", "surprise_pct", "eps_reported"])
    if df.empty:
        return
    t = to_ns(df["event_ts"]); ok = t > np.iinfo(np.int64).min
    t, sp, rep = _sorted(t[ok], pd.to_numeric(df["surprise_pct"], errors="coerce").to_numpy(float)[ok],
                         pd.to_numeric(df["eps_reported"], errors="coerce").to_numpy(float)[ok])
    j = np.searchsorted(t, cut, side="right")                 # the next event strictly after the cutoff
    nxt = np.full(len(cut), np.nan); has = j < len(t)
    nxt[has] = (t[j[has]] - cut[has]) / DAY_NS
    nxt[nxt > 120] = np.nan
    out["dp_earn_d_next"][:] = nxt
    # the SURPRISE is only in the store once yf is re-fetched (nightly): known from
    # the ET day after the release, whatever the release's own time
    done = np.isfinite(rep)                                   # a reported event carries a surprise
    rel_day = session_days(pd.DatetimeIndex(t[done].astype("datetime64[ns]")))
    tk_ = _known_from_days(rel_day, LAG_DAYS["yf_earnings_surprise"])
    tk_, spd = _sorted(tk_, np.clip(sp[done], -200, 200))
    out["dp_earn_surprise"][:] = _asof(tk_, spd, cut, max_age_days=120)


def _news(tk: str, cut: np.ndarray, out: Dict[str, np.ndarray]):
    df = _part("polygon_news", tk, ["published_utc", "sentiment", "n_tickers"])
    if df.empty:
        return
    t = to_ns(df["published_utc"]); ok = t > np.iinfo(np.int64).min
    sent = df["sentiment"].map(lambda s: _SENT.get(str(s).strip().lower(), np.nan) if s is not None else np.nan)
    nt = pd.to_numeric(df["n_tickers"], errors="coerce").to_numpy(float)
    t, s, nt = _sorted(t[ok], sent.to_numpy(float)[ok], nt[ok])
    n1 = _count(t, cut, DAY_NS); n3 = _count(t, cut, 3 * DAY_NS)
    # baseline: the ticker's own daily rate over the 60 days BEFORE the 3-day window
    nb = _count(t, cut - 3 * DAY_NS, 60 * DAY_NS) / 60.0
    out["dp_news_n_1d"][:] = n1
    out["dp_news_n_3d_rel"][:] = _log_rel(n3, 3.0 * nb)
    out["dp_news_sent_3d"][:] = _wmean(t, s, cut, 3 * DAY_NS)
    out["dp_news_d_last"][:] = _days_since(t, cut, 90)
    with np.errstate(divide="ignore", invalid="ignore"):
        foc = np.where(nt > 0, 1.0 / nt, np.nan)
    out["dp_news_focus_3d"][:] = _wmean(t, foc, cut, 3 * DAY_NS)


def _insider(ins: Optional[pd.DataFrame], cut: np.ndarray, rth: Optional[RTH], pos: np.ndarray,
             out: Dict[str, np.ndarray]):
    if ins is None:
        return
    for c in ("dp_ins_buy_n_30d", "dp_ins_buy_rel_90d", "dp_ins_sell_rel_90d", "dp_ins_nbuyers_90d"):
        out[c][:] = 0.0                                       # covered issuer: no filing = a real zero
    if not len(ins):
        return
    t = _known_from_days(to_days(ins["filing_date"]), LAG_DAYS["form345"])
    code = ins["trans_code"].astype(str).to_numpy()
    insider = (ins["is_officer"].fillna(False).astype(bool) | ins["is_director"].fillna(False).astype(bool)).to_numpy()
    notional = pd.to_numeric(ins["notional"], errors="coerce").fillna(0.0).to_numpy(float)
    owner = ins["owner_cik"].astype(str).to_numpy()
    buy = (code == "P") & insider
    sell = (code == "S") & insider
    tb, nb_, ob = _sorted(t[buy], notional[buy], owner[buy])
    ts_, ns_ = _sorted(t[sell], notional[sell])
    dv = np.full(len(cut), np.nan)
    if rth is not None:
        ok = pos >= 0; dv[ok] = rth.dv20[pos[ok]]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dp_ins_buy_n_30d"][:] = _count(tb, cut, 30 * DAY_NS)
        out["dp_ins_buy_rel_90d"][:] = np.log1p(_wsum(tb, nb_, cut, 90 * DAY_NS) / dv)
        out["dp_ins_sell_rel_90d"][:] = np.log1p(_wsum(ts_, ns_, cut, 90 * DAY_NS) / dv)
    out["dp_ins_d_buy"][:] = _days_since(tb, cut, 400)
    lo = np.searchsorted(tb, cut - 90 * DAY_NS, side="right"); hi = np.searchsorted(tb, cut, side="right")
    out["dp_ins_nbuyers_90d"][:] = [len(set(ob[a:b])) if b > a else 0 for a, b in zip(lo, hi)]
    if rth is not None and len(tb):
        i = _last_idx(tb, cut); ok = i >= 0
        recent = ok & ((cut - tb[np.maximum(i, 0)]) <= 90 * DAY_NS)
        out["_ins_anchor"][recent] = rth.price_at(tb[i[recent]])


def _analyst(tk: str, cut: np.ndarray, rth: Optional[RTH], pos: np.ndarray, out: Dict[str, np.ndarray]):
    df = _part("yf_analyst", tk, ["grade_date", "action", "pt_current", "pt_prior"])
    if df.empty:
        return
    t = _known_from_days(to_days(df["grade_date"]), LAG_DAYS["yf_analyst"])
    ok = t < np.iinfo(np.int64).max
    act = df["action"].fillna("").astype(str).str.lower().to_numpy()
    ptc = pd.to_numeric(df["pt_current"], errors="coerce").to_numpy(float)
    ptp = pd.to_numeric(df["pt_prior"], errors="coerce").to_numpy(float)
    t, act, ptc, ptp = _sorted(t[ok], act[ok], ptc[ok], ptp[ok])
    up, dn = (act == "up").astype(float), (act == "down").astype(float)
    out["dp_an_net_30d"][:] = _wsum(t, up, cut, 30 * DAY_NS) - _wsum(t, dn, cut, 30 * DAY_NS)
    out["dp_an_n_30d"][:] = _count(t, cut, 30 * DAY_NS)
    with np.errstate(invalid="ignore", divide="ignore"):
        chg = np.where((ptc > 0) & (ptp > 0), ptc / ptp - 1.0, np.nan)
    has = np.isfinite(chg)
    out["dp_an_pt_chg"][:] = _asof(t[has], np.clip(chg[has], -0.9, 3.0), cut, max_age_days=30)
    lo = np.searchsorted(t, cut - 90 * DAY_NS, side="right"); hi = np.searchsorted(t, cut, side="right")
    med = np.array([np.nanmedian(ptc[a:b]) if b > a and np.isfinite(ptc[a:b]).any() else np.nan
                    for a, b in zip(lo, hi)])
    if rth is not None:
        okp = pos >= 0; pc = np.full(len(cut), np.nan); pc[okp] = rth.prev_close[pos[okp]]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_an_pt_upside"][:] = np.clip(med / pc - 1.0, -0.9, 5.0)
    out["dp_an_d_last"][:] = _days_since(t, cut, 400)
    if rth is not None and len(t):
        i = _last_idx(t, cut); okk = i >= 0
        recent = okk & ((cut - t[np.maximum(i, 0)]) <= 30 * DAY_NS)
        out["_an_anchor"][recent] = rth.price_at(t[i[recent]])


def _shares(tk: str) -> Tuple[np.ndarray, np.ndarray]:
    df = _part("yf_shares", tk, ["date", "shares"])
    if df.empty:
        return np.zeros(0, np.int64), np.zeros(0)
    t = _known_from_days(to_days(df["date"]), LAG_DAYS["yf_shares"])
    v = pd.to_numeric(df["shares"], errors="coerce").to_numpy(float)
    ok = (t < np.iinfo(np.int64).max) & np.isfinite(v) & (v > 0)
    return _sorted(t[ok], v[ok])


def _short_interest(tk: str, cut: np.ndarray, sh_t, sh_v, out: Dict[str, np.ndarray]):
    df = _part("short_interest", tk, ["settlement_date", "short_interest", "days_to_cover"])
    if df.empty:
        return
    sd = pd.to_datetime(df["settlement_date"], errors="coerce")
    okd = sd.notna().to_numpy()
    kd = np.busday_offset(sd[okd].values.astype("datetime64[D]"), LAG_DAYS["short_interest_bdays"],
                          roll="forward").astype(np.int64)
    t = np.full(len(df), np.iinfo(np.int64).max, dtype=np.int64); t[okd] = midnight_ns(kd)
    si = pd.to_numeric(df["short_interest"], errors="coerce").to_numpy(float)
    dtc = pd.to_numeric(df["days_to_cover"], errors="coerce").to_numpy(float)
    setl = np.full(len(df), np.iinfo(np.int64).min, dtype=np.int64)
    setl[okd] = midnight_ns(sd[okd].values.astype("datetime64[D]").astype(np.int64))
    t, si, dtc, setl = _sorted(t, si, dtc, setl)
    i = _last_idx(t, cut); ok = i >= 0
    out["dp_si_dtc"][ok] = dtc[i[ok]]
    prev = np.where(i >= 1, si[np.maximum(i - 1, 0)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        chg = np.log(si[np.maximum(i, 0)] / prev)
    out["dp_si_chg"][ok] = np.clip(chg[ok], -3, 3)
    out["dp_si_age"][ok] = (cut[ok] - setl[i[ok]]) / DAY_NS
    shares = _asof(sh_t, sh_v, cut)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dp_si_pct_sh"][ok] = np.clip(si[i[ok]] / shares[ok], 0, 2)


def _daily_series(tk: str, family: str, date_col: str, val_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = _part(family, tk, [date_col, val_col])
    if df.empty:
        return np.zeros(0, np.int64), np.zeros(0)
    d = to_days(df[date_col]); v = pd.to_numeric(df[val_col], errors="coerce").to_numpy(float)
    ok = (d > np.iinfo(np.int64).min) & np.isfinite(v)
    d, v = d[ok], v[ok]
    o = np.argsort(d, kind="stable"); d, v = d[o], v[o]
    keep = np.r_[d[1:] != d[:-1], True]                        # one value per date (the last)
    return d[keep], v[keep]


def _rolling_rel(d: np.ndarray, v: np.ndarray, cut: np.ndarray, short: int, long: int, lag: int = 1,
                 log: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(latest value, mean(last `short`) - mean(last `long`)) as of each cutoff,
    rows visible from date + `lag` days. `log` compares log1p levels."""
    if not len(d):
        n = len(cut)
        return np.full(n, np.nan), np.full(n, np.nan)
    t = midnight_ns(d + lag)
    x = np.log1p(np.maximum(v, 0)) if log else v
    s = pd.Series(x)
    ms = s.rolling(short, min_periods=max(1, short // 2)).mean().to_numpy()
    ml = s.rolling(long, min_periods=max(5, long // 3)).mean().to_numpy()
    i = _last_idx(t, cut); ok = (i >= 0) & ((cut - t[np.maximum(i, 0)]) <= 10 * DAY_NS)
    last = np.full(len(cut), np.nan); rel = np.full(len(cut), np.nan)
    last[ok] = x[i[ok]]; rel[ok] = ms[i[ok]] - ml[i[ok]]
    return last, rel


def _wiki(tk: str, cut: np.ndarray, out: Dict[str, np.ndarray]):
    d, v = _daily_series(tk, "wiki", "date", "views")
    if not len(d):
        return
    t = midnight_ns(d + LAG_DAYS["wiki"])
    lv = np.log1p(np.maximum(v, 0)); s = pd.Series(lv)
    base = s.rolling(60, min_periods=20).mean().shift(1).to_numpy()   # the 60 days BEFORE the latest
    m7 = s.rolling(7, min_periods=4).mean().to_numpy()
    i = _last_idx(t, cut); ok = (i >= 0) & ((cut - t[np.maximum(i, 0)]) <= 10 * DAY_NS)
    out["dp_wiki_1d_rel"][ok] = lv[i[ok]] - base[i[ok]]
    out["dp_wiki_7d_rel"][ok] = m7[i[ok]] - base[i[ok]]


def _alt(tk: str, cut: np.ndarray, rth: Optional[RTH], pos: np.ndarray, out: Dict[str, np.ndarray]):
    cg = _part("quiver_congress", tk, ["ReportDate", "Transaction"])
    if not cg.empty:
        t = _known_from_days(to_days(cg["ReportDate"]), LAG_DAYS["quiver_congress"])
        tx = cg["Transaction"].fillna("").astype(str).str.lower().to_numpy()
        buy = np.array([s.startswith("purchase") for s in tx], float)
        sell = np.array([s.startswith("sale") for s in tx], float)
        t, buy, sell = _sorted(t, buy, sell)
        out["dp_cong_net_90d"][:] = _wsum(t, buy, cut, 90 * DAY_NS) - _wsum(t, sell, cut, 90 * DAY_NS)
        out["dp_cong_d_last"][:] = _days_since(t, cut, 730)
    lb = _part("quiver_lobbying", tk, ["Date", "Amount"])
    if not lb.empty:
        t = _known_from_days(to_days(lb["Date"]), LAG_DAYS["quiver_lobbying"])
        a = pd.to_numeric(lb["Amount"], errors="coerce").fillna(0.0).to_numpy(float)
        t, a = _sorted(t, a)
        out["dp_lobby_log_365d"][:] = np.log1p(np.maximum(_wsum(t, a, cut, 365 * DAY_NS), 0))
    ct = _part("quiver_contracts", tk, ["Date", "Amount"])
    if not ct.empty and rth is not None:
        t = _known_from_days(to_days(ct["Date"]), LAG_DAYS["quiver_contracts"])
        a = pd.to_numeric(ct["Amount"], errors="coerce").fillna(0.0).to_numpy(float)
        t, a = _sorted(t, a)
        dv = np.full(len(cut), np.nan); ok = pos >= 0; dv[ok] = rth.dv20[pos[ok]]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_contract_rel_365d"][:] = np.log1p(np.maximum(_wsum(t, a, cut, 365 * DAY_NS), 0) / (dv * 252.0))


def _dividends(div: Optional[pd.DataFrame], days: np.ndarray, cut: np.ndarray, rth: Optional[RTH],
               pos: np.ndarray, out: Dict[str, np.ndarray]):
    if div is None or not len(div):
        return
    exd = to_days(div["ex_dividend_date"]); dec = to_days(div["declaration_date"])
    known_day = np.where(dec > np.iinfo(np.int64).min, dec + LAG_DAYS["dividends"], exd)
    t = _known_from_days(known_day, 0)
    amt = pd.to_numeric(div["cash_amount"], errors="coerce").fillna(0.0).to_numpy(float)
    rec = (div["dividend_type"].astype(str) == "CD").to_numpy()
    ok = exd > np.iinfo(np.int64).min
    t, exd, amt, rec = t[ok], exd[ok], amt[ok], rec[ok]
    d_to_ex = np.full(len(days), np.nan); yld = np.full(len(days), np.nan)
    pc = np.full(len(days), np.nan)
    if rth is not None:
        okp = pos >= 0; pc[okp] = rth.prev_close[pos[okp]]
    for k in range(len(days)):                              # small per ticker: dozens of rows
        kn = t <= cut[k]
        fut = kn & (exd >= days[k])
        if fut.any():
            dd = float(exd[fut].min() - days[k])
            d_to_ex[k] = dd if dd <= 60 else np.nan
        past = kn & rec & (exd <= days[k]) & (exd > days[k] - 365)
        if pc[k] > 0:
            yld[k] = amt[past].sum() / pc[k]
    out["dp_div_d_to_ex"][:] = d_to_ex
    out["dp_div_yield"][:] = np.clip(yld, 0, 0.5)


def _size(cut: np.ndarray, rth: Optional[RTH], pos: np.ndarray, sh_t, sh_v, list_day: Optional[int],
          days: np.ndarray, out: Dict[str, np.ndarray]):
    sh = _asof(sh_t, sh_v, cut)
    if rth is not None:
        ok = pos >= 0; pc = np.full(len(cut), np.nan); pc[ok] = rth.prev_close[pos[ok]]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_mcap_log"][:] = np.log(sh * pc)
    sh_1y = _asof(sh_t, sh_v, cut - 365 * DAY_NS)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dp_shares_chg_1y"][:] = np.clip(np.log(sh / sh_1y), -2, 2)
    if list_day is not None:
        age = (days - int(list_day)) / 365.25
        out["dp_age_y"][:] = np.where(age >= 0, age, np.nan)


def _inst(inst: Optional[pd.DataFrame], ftd: Optional[pd.DataFrame], cut: np.ndarray, rth: Optional[RTH],
          pos: np.ndarray, sh_t, sh_v, out: Dict[str, np.ndarray]):
    if inst is not None and len(inst):
        t, n, s = _sorted(inst["known"].to_numpy(np.int64), inst["n_filers"].to_numpy(float),
                          inst["total_shares"].to_numpy(float))
        i = _last_idx(t, cut); ok = i >= 0
        out["dp_inst_n_log"][ok] = np.log1p(n[i[ok]])
        prev = np.where(i >= 1, n[np.maximum(i - 1, 0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_inst_chg"][ok] = np.log1p(n[i[ok]]) - np.log1p(prev[ok])
            shares = _asof(sh_t, sh_v, cut)
            out["dp_inst_sh_pct"][ok] = np.clip(s[i[ok]] / shares[ok], 0, 3)
    if ftd is not None and len(ftd) and rth is not None:
        t, q = _sorted(ftd["known"].to_numpy(np.int64), ftd["qty"].to_numpy(float))
        i = _last_idx(t, cut); ok = (i >= 0) & ((cut - t[np.maximum(i, 0)]) <= 45 * DAY_NS)
        vol = np.full(len(cut), np.nan); okp = pos >= 0; vol[okp] = rth.vol20[pos[okp]]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_ftd_rel"][ok] = np.log1p(q[i[ok]] / (vol[ok] * 10.0))
        out["dp_ftd_rel"][(pos >= 0) & ~ok] = 0.0            # covered symbol, no fails in the latest period


def _fundamentals(tk: str, sec_known: Optional[pd.DataFrame], cut: np.ndarray, rth: Optional[RTH],
                  pos: np.ndarray, sh_t, sh_v, out: Dict[str, np.ndarray]):
    df = _part("companyfacts", tk, ["concept", "unit", "start", "end", "val", "accn", "filed"])
    if df.empty:
        return
    df = df[df["unit"].astype(str) == "USD"]
    if df.empty:
        return
    fb = _known_from_days(to_days(df["filed"]), 1)
    t = fb.copy()
    if sec_known is not None and len(sec_known):
        m = dict(zip(sec_known["accession"].astype(str), sec_known["_known"].astype(np.int64)))
        acc = df["accn"].astype(str).map(m)
        t = np.where(acc.notna().to_numpy(), acc.fillna(0).to_numpy(np.int64), fb)
    df = df.assign(_t=t, _end=to_days(df["end"]), _start=to_days(df["start"]),
                   _v=pd.to_numeric(df["val"], errors="coerce"))
    # book equity: the latest END among known facts, its latest-known value
    eq = df[df["concept"].isin(_EQ_CONCEPTS) & df["_v"].notna()].sort_values(["_t", "_end"])
    if len(eq) and rth is not None:
        best_end, best_v, ts_, vs_ = np.iinfo(np.int64).min, np.nan, [], []
        for tt, ee, vv in zip(eq["_t"].to_numpy(), eq["_end"].to_numpy(), eq["_v"].to_numpy(float)):
            if ee >= best_end:
                best_end, best_v = ee, vv
            ts_.append(tt); vs_.append(best_v)
        bv = _asof(np.asarray(ts_, np.int64), np.asarray(vs_), cut)
        ok = pos >= 0; pc = np.full(len(cut), np.nan); pc[ok] = rth.prev_close[pos[ok]]
        mcap = _asof(sh_t, sh_v, cut) * pc
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_fund_bm"][:] = np.clip(np.where(bv > 0, bv / mcap, np.nan), 0, 20)
    # quarterly revenue growth: latest known quarter vs the same quarter a year earlier
    rv = df[df["concept"].isin(_REV_CONCEPTS) & df["_v"].notna() & (df["_start"] > np.iinfo(np.int64).min)]
    if len(rv):
        dur = rv["_end"] - rv["_start"]
        rv = rv[(dur >= 80) & (dur <= 100)].sort_values(["_t", "_end"])
        by_end: Dict[int, float] = {}
        ts_, gs_ = [], []
        for tt, ee, vv in zip(rv["_t"].to_numpy(), rv["_end"].to_numpy(), rv["_v"].to_numpy(float)):
            by_end[int(ee)] = vv
            last = max(by_end)
            prior = [e for e in by_end if abs((last - e) - 365) <= 20]
            g = np.nan
            if prior:
                pv = by_end[min(prior, key=lambda e: abs((last - e) - 365))]
                if pv and pv > 0:
                    g = by_end[last] / pv - 1.0
            ts_.append(tt); gs_.append(g)
        out["dp_fund_rev_g"][:] = np.clip(_asof(np.asarray(ts_, np.int64), np.asarray(gs_), cut), -1, 5)


def _regsho(on_days: Optional[np.ndarray], cal: Optional[np.ndarray], days: np.ndarray, cut: np.ndarray,
            out: Dict[str, np.ndarray]):
    """Reg SHO threshold lists (`src/data/deep/regsho.py`) as of each cutoff. ``cal``
    holds the list dates every market published (sorted days); ``on_days`` the
    dates this name was on any list (empty = never: 0, not missing). The list of D
    is known from D+1 (LAG_DAYS); a calendar more than a week stale leaves NaN."""
    if cal is None or not len(cal):
        return
    cal = np.asarray(cal, np.int64)
    known = midnight_ns(cal + LAG_DAYS["regsho"])
    i = _last_idx(known, cut)
    ok = (i >= 0) & ((cut - known[np.maximum(i, 0)]) <= 7 * DAY_NS)
    if not ok.any():
        return
    on = np.isin(cal, np.asarray(on_days, np.int64)) if on_days is not None and len(on_days) else \
        np.zeros(len(cal), bool)
    k = np.arange(len(cal))
    streak = k - np.maximum.accumulate(np.where(~on, k, -1))         # consecutive list days ending here
    cs = np.r_[0, np.cumsum(on)]
    last_on = np.maximum.accumulate(np.where(on, k, -1))
    ii = i[ok]
    out["dp_rs_on"][ok] = on[ii].astype(float)
    out["dp_rs_streak"][ok] = np.minimum(streak[ii], 250).astype(float)
    out["dp_rs_n_60"][ok] = (cs[ii + 1] - cs[np.maximum(ii - 59, 0)]).astype(float)
    j = last_on[ii]
    dl = np.where(j >= 0, days[ok] - cal[np.maximum(j, 0)], np.nan).astype(float)
    dl[dl > 365] = np.nan
    out["dp_rs_d_last"][ok] = dl


def _borrow(bw: Optional[pd.DataFrame], cut: np.ndarray, rth: Optional[RTH], pos: np.ndarray,
            out: Dict[str, np.ndarray]):
    """IBKR's lendable shares and fee (`src/data/deep/borrow.py`: the day's last
    file of our own archive) as of each cutoff, known the day after its date;
    NaN when the latest row is more than 10 days old."""
    if bw is None or not len(bw):
        return
    d = to_days(bw["date"])
    avail = pd.to_numeric(bw["available"], errors="coerce").to_numpy(float)
    fee = np.clip(pd.to_numeric(bw["fee"], errors="coerce").to_numpy(float), 0, 2000)
    keep = (d > np.iinfo(np.int64).min) & np.isfinite(avail) & np.isfinite(fee)
    d, avail, fee = d[keep], avail[keep], fee[keep]
    if not len(d):
        return
    o = np.argsort(d, kind="stable"); d, avail, fee = d[o], avail[o], fee[o]
    t = midnight_ns(d + LAG_DAYS["borrow"])
    i = _last_idx(t, cut)
    ok = (i >= 0) & ((cut - t[np.maximum(i, 0)]) <= 10 * DAY_NS)
    if not ok.any():
        return
    ii = i[ok]
    out["dp_bw_fee"][ok] = fee[ii]
    out["dp_bw_age"][ok] = (cut[ok] - t[ii]) / DAY_NS
    if rth is not None:
        pc = np.full(len(cut), np.nan); okp = pos >= 0; pc[okp] = rth.prev_close[pos[okp]]
        usd = avail[ii] * pc[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_bw_avail_usd"][ok] = np.log10(1.0 + np.maximum(usd, 0))
        out["dp_bw_htb"][ok] = np.where(np.isfinite(usd), (usd < 10_000).astype(float), np.nan)
    i5 = ii - 5
    has5 = i5 >= 0
    f5 = np.where(has5, fee[np.maximum(i5, 0)], np.nan)
    a5 = np.where(has5, avail[np.maximum(i5, 0)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dp_bw_fee_chg5"][ok] = np.log((fee[ii] + 0.25) / (f5 + 0.25))
        out["dp_bw_avail_chg5"][ok] = np.log((avail[ii] + 1000.0) / (a5 + 1000.0))
    mx = pd.Series(fee).rolling(20, min_periods=1).max().to_numpy()
    out["dp_bw_fee_max20"][ok] = mx[ii]


# ── the per-ticker snapshot ──────────────────────────────────────────────────

def _blank(n: int) -> Dict[str, np.ndarray]:
    return {c: np.full(n, np.nan) for c in SNAPSHOT_FEATURES + ANCHOR_COLUMNS}


def ticker_snapshots(ticker: str, days: Optional[np.ndarray] = None, rth: Optional[RTH] = None,
                     slices: Optional[dict] = None) -> pd.DataFrame:
    """Snapshot features of ``ticker`` for each session day in ``days`` (default:
    every session of its regular-hours grid), as known at that day's 08:30 ET.

    ``slices`` carries the whole-market families already grouped for this
    ticker (``MarketTables.slices``): ``insider`` (None = issuer never filed),
    ``dividends``, ``inst``, ``ftd``, ``list_day``. Per-ticker families are read
    from their parts here. Returns a frame indexed by day with
    SNAPSHOT_FEATURES + ANCHOR_COLUMNS; a family with no data leaves NaN."""
    tk = ticker.upper()
    if rth is None:
        rth = RTH.for_ticker(tk)
    if days is None:
        days = rth.sessions if rth is not None else np.zeros(0, np.int64)
    days = np.asarray(days, dtype=np.int64)
    n = len(days)
    out = _blank(n)
    if n == 0:
        return pd.DataFrame(out, index=days)
    cut = cutoff_ns(days)
    if rth is not None and len(rth.sessions):
        # a pre-open build asks for the session right after the grid's last one:
        # give it its position (`RTH.with_session`). A grid that is BEHIND (the
        # 30-minute store not extended through the previous session) is left
        # alone — the features stay missing rather than reading a stale close.
        beyond = days[days > rth.sessions[-1]]
        if len(beyond) == 1 and previous_session_day(int(beyond[0])) == int(rth.sessions[-1]):
            rth = rth.with_session(int(beyond[0]))
    pos = rth.session_pos(days) if rth is not None else np.full(n, -1)
    if rth is not None:
        ok = pos >= 0; out["_prev_close"][ok] = rth.prev_close[pos[ok]]
    sl = slices or {}
    sh_t, sh_v = _shares(tk)
    try:
        _xh(tk, days, cut, rth, pos, out)
    except Exception as e:                               # noqa: BLE001
        logger.debug(f"[deep_features] {tk} xh: {e}")
    sec_known = None
    for name, fn in (("sec", lambda: _sec(tk, cut, rth, out)),
                     ("earn", lambda: _earn(tk, cut, out)),
                     ("news", lambda: _news(tk, cut, out)),
                     ("ins", lambda: _insider(sl.get("insider"), cut, rth, pos, out)),
                     ("an", lambda: _analyst(tk, cut, rth, pos, out)),
                     ("si", lambda: _short_interest(tk, cut, sh_t, sh_v, out)),
                     ("wiki", lambda: _wiki(tk, cut, out)),
                     ("alt", lambda: _alt(tk, cut, rth, pos, out)),
                     ("div", lambda: _dividends(sl.get("dividends"), days, cut, rth, pos, out)),
                     ("size", lambda: _size(cut, rth, pos, sh_t, sh_v, sl.get("list_day"), days, out)),
                     ("inst", lambda: _inst(sl.get("inst"), sl.get("ftd"), cut, rth, pos, sh_t, sh_v, out)),
                     ("rs", lambda: _regsho(sl.get("regsho"), sl.get("rs_cal"), days, cut, out)),
                     ("bw", lambda: _borrow(sl.get("borrow"), cut, rth, pos, out))):
        try:
            r = fn()
            if name == "sec":
                sec_known = r
        except Exception as e:                           # noqa: BLE001
            logger.debug(f"[deep_features] {tk} {name}: {e}")
    try:
        d, v = _daily_series(tk, "short_volume", "date", "short_volume_ratio")
        out["dp_sv_ratio_1d"], out["dp_sv_ratio_5d_rel"] = _rolling_rel(d, v, cut, 5, 60, lag=LAG_DAYS["short_volume"])
        d, v = _daily_series(tk, "quiver_dpi", "Date", "DPI")
        out["dp_dpi_1d"], out["dp_dpi_5d_rel"] = _rolling_rel(d, v, cut, 5, 60, lag=LAG_DAYS["quiver_dpi"])
    except Exception as e:                               # noqa: BLE001
        logger.debug(f"[deep_features] {tk} sv/dpi: {e}")
    try:
        _fundamentals(tk, sec_known, cut, rth, pos, sh_t, sh_v, out)
    except Exception as e:                               # noqa: BLE001
        logger.debug(f"[deep_features] {tk} fund: {e}")
    return pd.DataFrame(out, index=days)


# ── bar-level features (the snapshot meets the bar) ─────────────────────────

def bar_features(snap: pd.DataFrame, close: np.ndarray, bar_index: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-row features from the row's snapshot (aligned rows of ``snap``) and
    its bar: the return from each event's anchor to THIS bar's close, and the
    bar's position in the session. Recomputed every 30 minutes at serving."""
    close = np.asarray(close, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "dp_bar_index": np.asarray(bar_index, float),
            "dp_xh_rth_vs_pre": close / snap["_xh_pre_last"].to_numpy(float) - 1.0,
            "dp_earn_ret_since": close / snap["_earn_anchor"].to_numpy(float) - 1.0,
            "dp_an_ret_since": close / snap["_an_anchor"].to_numpy(float) - 1.0,
            "dp_ins_ret_since": close / snap["_ins_anchor"].to_numpy(float) - 1.0,
        }


# ── market context (constant across tickers within a session) ───────────────

def _market_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    return deep.read_parquet(path, columns=list(columns)) if path.exists() else pd.DataFrame()


def market_context(days: np.ndarray) -> pd.DataFrame:
    """MARKET_FEATURES per session day, each as known at D 08:30 ET (daily
    series from D-1, FRED first prints with realtime_start <= D-1, COT from
    the Friday release)."""
    days = np.unique(np.asarray(days, dtype=np.int64))
    out = {c: np.full(len(days), np.nan) for c in MARKET_FEATURES}
    cut = cutoff_ns(days)
    md = _market_table(deep.DEEP_DIR / "market_daily.parquet", ["symbol", "date", "close"])

    def series(sym: str) -> Tuple[np.ndarray, np.ndarray]:
        s = md[md["symbol"] == sym]
        d = to_days(s["date"]); v = pd.to_numeric(s["close"], errors="coerce").to_numpy(float)
        ok = (d > np.iinfo(np.int64).min) & np.isfinite(v)
        o = np.argsort(d[ok], kind="stable")
        return d[ok][o], v[ok][o]

    def asof_n(sym: str, back: int = 0) -> np.ndarray:
        d, v = series(sym)
        if not len(d):
            return np.full(len(days), np.nan)
        t = midnight_ns(d + LAG_DAYS["market"])
        i = _last_idx(t, cut) - back
        r = np.full(len(days), np.nan); ok = i >= 0
        r[ok] = v[i[ok]]
        return r

    if not md.empty:
        vix, vix5, vix3m = asof_n("^VIX"), asof_n("^VIX", 5), asof_n("^VIX3M")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dp_mkt_vix"] = vix
            out["dp_mkt_vix_ts"] = vix / vix3m
            out["dp_mkt_vix_chg5"] = np.log(vix / vix5)
            spy, spy5 = asof_n("SPY"), asof_n("SPY", 5)
            iwm, iwm5 = asof_n("IWM"), asof_n("IWM", 5)
            hyg, hyg5 = asof_n("HYG"), asof_n("HYG", 5)
            lqd, lqd5 = asof_n("LQD"), asof_n("LQD", 5)
            tnx, tnx5 = asof_n("^TNX"), asof_n("^TNX", 5)
            out["dp_mkt_spy_ret5"] = spy / spy5 - 1.0
            out["dp_mkt_iwm_spy5"] = (iwm / iwm5 - 1.0) - (spy / spy5 - 1.0)
            out["dp_mkt_hyg_lqd5"] = (hyg / hyg5 - 1.0) - (lqd / lqd5 - 1.0)
            out["dp_mkt_tnx_chg5"] = tnx - tnx5
    dx = _market_table(deep.DEEP_DIR / "dix.parquet", ["date", "dix", "gex"])
    if not dx.empty:
        d = to_days(dx["date"]); o = np.argsort(d, kind="stable"); d = d[o]
        dix = pd.to_numeric(dx["dix"], errors="coerce").to_numpy(float)[o]
        gex = pd.to_numeric(dx["gex"], errors="coerce").to_numpy(float)[o]
        gz = (pd.Series(gex) / pd.Series(gex).rolling(252, min_periods=60).std()).to_numpy()
        t = midnight_ns(d + LAG_DAYS["market"])
        out["dp_mkt_dix"] = _asof(t, dix, cut, max_age_days=10)
        out["dp_mkt_gex_z"] = _asof(t, gz, cut, max_age_days=10)
    fr = _market_table(deep.DEEP_DIR / "fred_vintages.parquet", ["series_id", "date", "realtime_start", "value"])
    if not fr.empty:
        for sid, col in (("T10Y2Y", "dp_mkt_t10y2y"), ("NFCI", "dp_mkt_nfci")):
            s = fr[fr["series_id"] == sid]
            if s.empty:
                continue
            rs = to_days(s["realtime_start"]); ob = to_days(s["date"])
            v = pd.to_numeric(s["value"], errors="coerce").to_numpy(float)
            ok = (rs > np.iinfo(np.int64).min) & np.isfinite(v)
            rs, ob, v = rs[ok], ob[ok], v[ok]
            # as of D: among vintages published by D-2, the latest OBSERVATION, its latest
            # print. D-2, not D-1: series ALFRED does not version are stamped at their
            # OBSERVATION date, a day before FRED actually posts them
            o = np.lexsort((rs, ob)); rs, ob, v = rs[o], ob[o], v[o]
            res = np.full(len(days), np.nan)
            for k, D in enumerate(days):
                m = rs <= D - 2
                if m.any():
                    j = np.flatnonzero(m)[-1]                       # max ob, then max rs, among visible
                    res[k] = v[j] if (D - ob[j]) <= 45 else np.nan
            out[col] = res
    ct = _market_table(deep.DEEP_DIR / "cot_tff.parquet",
                       ["CFTC_Contract_Market_Code", "report_date", "Open_Interest_All",
                        "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"])
    if not ct.empty:
        s = ct[ct["CFTC_Contract_Market_Code"].astype(str).str.strip() == "13874A"]
        if len(s):
            d = to_days(s["report_date"])
            with np.errstate(invalid="ignore", divide="ignore"):
                v = ((pd.to_numeric(s["Lev_Money_Positions_Long_All"], errors="coerce")
                      - pd.to_numeric(s["Lev_Money_Positions_Short_All"], errors="coerce"))
                     / pd.to_numeric(s["Open_Interest_All"], errors="coerce")).to_numpy(float)
            ok = (d > np.iinfo(np.int64).min) & np.isfinite(v)
            t, v = _sorted(midnight_ns(d[ok] + LAG_DAYS["cot_after_report"]), v[ok])
            out["dp_mkt_cot_lev"] = _asof(t, v, cut, max_age_days=21)
    return pd.DataFrame(out, index=days)


# ── whole-market families, grouped per ticker ────────────────────────────────

_RANGE_RE = re.compile(r"(\d{2})([a-z]{3})(\d{4})-(\d{2})([a-z]{3})(\d{4})_form13f", re.I)
_QTR_RE = re.compile(r"(\d{4})q([1-4])_form13f", re.I)
_FTD_RE = re.compile(r"cnsfails(\d{4})(\d{2})([ab]?)", re.I)


def _13f_range_end(stem: str) -> Optional[int]:
    m = _RANGE_RE.search(stem or "")
    if m:
        return int(to_days([f"{m.group(4)}{m.group(5)}{m.group(6)}"])[0])
    m = _QTR_RE.search(stem or "")
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        end = pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
        return int(to_days([end])[0])
    return None


def _ftd_period_end(stem: str) -> Optional[int]:
    m = _FTD_RE.search(stem or "")
    if not m:
        return None
    y, mo, half = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if half == "a":
        end = pd.Timestamp(year=y, month=mo, day=15)
    else:
        end = pd.Timestamp(year=y, month=mo, day=1) + pd.offsets.MonthEnd(0)
    return int(to_days([end])[0])


class MarketTables:
    """The families stored whole-market (insider, dividends, 13F, FTD, listing
    dates) loaded once and grouped per ticker. ``slices(tk)`` is what
    ``ticker_snapshots`` takes. Universe-restricted at load, so memory stays
    bounded (a few hundred MB for the 3,430-name universe)."""

    def __init__(self, tickers: Sequence[str]):
        import duckdb
        t0 = time.time()
        self.tickers = sorted({t.upper() for t in tickers})
        D = deep.DEEP_DIR
        con = duckdb.connect()
        con.register("u", pd.DataFrame({"ticker": self.tickers}))
        # insider transactions by issuer CIK
        self.insider: Dict[str, pd.DataFrame] = {}
        self._insider_ciks: set = set()
        try:
            from src.data.deep import sec as _sec_mod
            cmap = {t: c for t, c in _sec_mod.cik_map().items() if t in set(self.tickers)}
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep_features] CIK map unavailable: {e}")
            cmap = {}
        self.cik_of = cmap
        srcs = [p for p in (D / "form345.parquet", D / "form345_live.parquet") if p.exists()]
        if srcs and cmap:
            con.register("c", pd.DataFrame({"cik": list(set(cmap.values()))}))
            q = " UNION ALL ".join(
                f"SELECT issuer_cik, filing_date, trans_code, notional, is_officer, is_director, owner_cik "
                f"FROM '{p.as_posix()}' WHERE trans_code IN ('P','S') AND issuer_cik IN (SELECT cik FROM c)"
                for p in srcs)
            ins = con.execute(q).df()
            ever = con.execute(" UNION ".join(
                f"SELECT DISTINCT issuer_cik FROM '{p.as_posix()}' WHERE issuer_cik IN (SELECT cik FROM c)"
                for p in srcs)).fetchall()
            self._insider_ciks = {r[0] for r in ever}
            self.insider = {cik: g.drop(columns=["issuer_cik"]) for cik, g in ins.groupby("issuer_cik")}
        # dividends
        self.dividends: Dict[str, pd.DataFrame] = {}
        p = D / "dividends.parquet"
        if p.exists():
            dv = con.execute(f"SELECT ticker, declaration_date, ex_dividend_date, cash_amount, dividend_type "
                             f"FROM '{p.as_posix()}' WHERE ticker IN (SELECT ticker FROM u)").df()
            self.dividends = {tk: g for tk, g in dv.groupby("ticker")}
        # FTD (and the CUSIP bridge 13F needs)
        self.ftd: Dict[str, pd.DataFrame] = {}
        self.inst: Dict[str, pd.DataFrame] = {}
        p = D / "ftd.parquet"
        cusips: Dict[str, List[str]] = {}
        if p.exists():
            f = con.execute(f"SELECT symbol, cusip, file, sum(quantity) AS qty, count(*) AS n FROM '{p.as_posix()}' "
                            f"WHERE symbol IN (SELECT ticker FROM u) AND settlement_date >= '2019-01-01' "
                            f"GROUP BY 1, 2, 3").df()
            ends_by_file = {s: _ftd_period_end(s) for s in f["file"].dropna().unique()}   # ~430 files, not rows
            f["end"] = f["file"].map(ends_by_file)
            f = f[f["end"].notna()]
            f["known"] = midnight_ns(f["end"].astype(np.int64).to_numpy() + LAG_DAYS["ftd_after_period"])
            per = f.groupby(["symbol", "known"], as_index=False)["qty"].sum()
            self.ftd = {tk: g[["known", "qty"]] for tk, g in per.groupby("symbol")}
            cnt = f.groupby(["symbol", "cusip"])["n"].sum()
            for (sym, cu), n in cnt.items():
                if n >= 3 and isinstance(cu, str) and len(cu) >= 8:
                    cusips.setdefault(sym, []).append(cu.upper()[:9])
        p = D / "form13f.parquet"
        if p.exists() and cusips:
            cu_rows = [(cu, tk) for tk, cs in cusips.items() for cu in cs]
            con.register("cm", pd.DataFrame(cu_rows, columns=["cusip", "ticker"]))
            h = con.execute(f"""SELECT cm.ticker, f.period_of_report, f.filing_date, f.file,
                                       f.n_filers, f.total_shares
                                FROM '{p.as_posix()}' f JOIN cm ON substr(f.cusip, 1, 9) = cm.cusip
                                WHERE NOT f.is_amendment""").df()
            if len(h):
                ends_by_file = {s: _13f_range_end(s) for s in h["file"].dropna().unique()}  # ~60 files, not rows
                h["rng_end"] = h["file"].map(ends_by_file)
                h["per"] = to_days(h["period_of_report"])
                h["fd"] = to_days(h["filing_date"])
                h = h[h["rng_end"].notna() & (h["per"] > np.iinfo(np.int64).min)]
                # on-time filers only (<= period + 45 days): a period is COMPLETE once
                # the file covering period+45 is published (+30 days, conservative)
                h = h[h["fd"] <= h["per"] + 45]
                files = h[["file", "rng_end"]].drop_duplicates().sort_values("rng_end")
                ends = files["rng_end"].to_numpy(np.int64)
                agg = h.groupby(["ticker", "per"], as_index=False).agg(n_filers=("n_filers", "sum"),
                                                                       total_shares=("total_shares", "sum"))
                due = agg["per"].to_numpy(np.int64) + 45
                j = np.searchsorted(ends, due, side="left")
                okj = j < len(ends)
                agg = agg[okj].copy()
                agg["known"] = midnight_ns(ends[j[okj]] + LAG_DAYS["form13f_after_range"])
                self.inst = {tk: g[["known", "n_filers", "total_shares"]] for tk, g in agg.groupby("ticker")}
        # Reg SHO threshold lists: the dates each name was on one, and the list dates
        # EVERY market published (a day one market is still missing would read a
        # false "not on the list" for its names — the day waits instead)
        self.rs_on: Dict[str, np.ndarray] = {}
        self.rs_cal: Optional[np.ndarray] = None
        p, pc_ = D / "regsho.parquet", D / "regsho_calendar.parquet"
        if p.exists() and pc_.exists():
            from src.data.deep.regsho import MARKETS as _RS_MARKETS
            cal = con.execute(f"SELECT CAST(date AS VARCHAR) AS date, count(DISTINCT market) AS n "
                              f"FROM '{pc_.as_posix()}' GROUP BY 1").df()
            cal = cal[cal["n"] >= len(_RS_MARKETS)]
            self.rs_cal = np.unique(to_days(cal["date"]))
            rs = con.execute(f"SELECT DISTINCT CAST(symbol AS VARCHAR) AS symbol, CAST(date AS VARCHAR) AS date "
                             f"FROM '{p.as_posix()}' WHERE CAST(symbol AS VARCHAR) IN (SELECT ticker FROM u)").df()
            if len(rs):
                rs["d"] = to_days(rs["date"])
                self.rs_on = {tk: np.unique(g["d"].to_numpy(np.int64)) for tk, g in rs.groupby("symbol")}
        # IBKR borrow, one row per name per day (our own archive of IBKR's file)
        self.borrow: Dict[str, pd.DataFrame] = {}
        p = D / "borrow_daily.parquet"
        if p.exists():
            bw = con.execute(f"SELECT CAST(ticker AS VARCHAR) AS ticker, CAST(date AS VARCHAR) AS date, available, fee "
                             f"FROM '{p.as_posix()}' WHERE CAST(ticker AS VARCHAR) IN (SELECT ticker FROM u) "
                             f"ORDER BY 1, 2").df()
            self.borrow = {tk: g.drop(columns=["ticker"]) for tk, g in bw.groupby("ticker")}
        # listing dates
        self.list_day: Dict[str, int] = {}
        floor_day = int(to_days(["1950-01-01"])[0])          # Polygon writes 1900-01-01 for "unknown"
        for path, col in ((D / "ticker_details.parquet", "list_date"), (D / "ipos.parquet", "listing_date")):
            if not path.exists():
                continue
            t = con.execute(f"SELECT ticker, {col} FROM '{path.as_posix()}' WHERE ticker IN (SELECT ticker FROM u)").df()
            dd = to_days(t[col])
            for tk, d_ in zip(t["ticker"].astype(str), dd):
                if d_ > floor_day and (tk not in self.list_day or d_ < self.list_day[tk]):
                    self.list_day[tk] = int(d_)
        con.close()
        logger.info(f"[deep_features] market tables: insider {len(self.insider)} issuers, dividends "
                    f"{len(self.dividends)}, 13F {len(self.inst)}, FTD {len(self.ftd)}, listing {len(self.list_day)}, "
                    f"Reg SHO {len(self.rs_on)} names / {0 if self.rs_cal is None else len(self.rs_cal)} list days, "
                    f"borrow {len(self.borrow)} | {time.time() - t0:.0f}s")

    def slices(self, ticker: str) -> dict:
        tk = ticker.upper()
        cik = self.cik_of.get(tk)
        ins = None
        if cik is not None and cik in self._insider_ciks:
            ins = self.insider.get(cik, pd.DataFrame(columns=["filing_date", "trans_code", "notional", "is_officer",
                                                             "is_director", "owner_cik"]))
        return {"insider": ins, "dividends": self.dividends.get(tk), "inst": self.inst.get(tk),
                "ftd": self.ftd.get(tk), "list_day": self.list_day.get(tk),
                "regsho": self.rs_on.get(tk, np.zeros(0, np.int64)), "rs_cal": self.rs_cal,
                "borrow": self.borrow.get(tk)}


# ── the per-session snapshot the live scorer reads ──────────────────────────

SNAPSHOT_DIR = "snapshot"


def snapshot_path(day: int) -> Path:
    d = pd.Timestamp(np.datetime64(int(day), "D")).strftime("%Y-%m-%d")
    return deep.DEEP_DIR / SNAPSHOT_DIR / f"{d}.parquet"


def _snapshot_one(args) -> Optional[dict]:
    tk, day, slices = args
    try:
        snap = ticker_snapshots(tk, days=np.array([day], dtype=np.int64), slices=slices)
        r = snap.iloc[0].to_dict(); r["ticker"] = tk
        return r
    except Exception as e:                               # noqa: BLE001
        logger.debug(f"[deep_features] snapshot {tk}: {e}")
        return None


def snapshot_universe() -> List[str]:
    """The names a live snapshot covers: the deep universe plus every name the
    tick's own 30-minute cache holds (the scorer can serve a name with enough
    tick-cache history even when the deep store never swept it)."""
    names = set(deep.deep_universe())
    tick_dir = Path("cache/ohlcv_30m")
    if tick_dir.exists():
        names |= {p.stem.upper() for p in tick_dir.glob("*.json")}
    return sorted(n for n in names if n and n.replace("-", "").replace(".", "").isalnum())


def build_session_snapshot(day: int, tickers: Optional[Sequence[str]] = None, workers: int = 6) -> int:
    """Every ticker's snapshot for session ``day`` (+ market context) -> one
    parquet the scorer reads. Built by the pre-open refresh right after the fast
    families land; any process can rebuild it (deterministic given the store).
    ``workers`` > 1 fans the tickers out over processes (the whole-market
    tables are loaded ONCE here and handed to each task as its slice)."""
    t0 = time.time()
    tickers = snapshot_universe() if tickers is None else list(tickers)
    tables = MarketTables(tickers)
    jobs = [(tk, int(day), tables.slices(tk)) for tk in tables.tickers]
    rows: List[dict] = []
    if workers > 1 and len(jobs) > 50:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for r in ex.map(_snapshot_one, jobs, chunksize=16):
                if r is not None:
                    rows.append(r)
    else:
        rows = [r for r in map(_snapshot_one, jobs) if r is not None]
    df = pd.DataFrame(rows)
    mk = market_context(np.array([day], dtype=np.int64))
    for c in MARKET_FEATURES:
        df[c] = mk[c].iloc[0] if len(mk) else np.nan
    n = deep.write_parquet(df, snapshot_path(day))
    logger.info(f"[deep_features] session snapshot {snapshot_path(day).name}: {n} tickers in {time.time() - t0:.0f}s")
    return n


_SNAP_CACHE: Dict[int, Tuple[float, Dict[str, dict]]] = {}


def load_session_snapshot(day: int) -> Optional[Dict[str, dict]]:
    """``{ticker: snapshot row}`` for session ``day`` from its file, memoised on
    the file's mtime; None when the file does not exist yet."""
    p = snapshot_path(day)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _SNAP_CACHE.get(int(day))
    if hit is not None and hit[0] == mt:
        return hit[1]
    df = deep.read_parquet(p)
    d = {str(r["ticker"]): r for r in df.to_dict("records")}
    _SNAP_CACHE[int(day)] = (mt, d)
    if len(_SNAP_CACHE) > 4:
        for k in sorted(_SNAP_CACHE)[:-4]:
            _SNAP_CACHE.pop(k, None)
    return d


def serving_vector(ticker: str, session_day: int, close: float, bar_index: int,
                   snapshot: Optional[Dict[str, dict]] = None, compute_missing: bool = False) -> Dict[str, float]:
    """DEEP_FEATURES for one live row: the session snapshot's values (built at
    the pre-open cutoff, exactly as in training) plus the bar features from
    THIS bar. A ticker the snapshot does not carry is outside the deep store's
    universe, so its snapshot features are NaN — the same missing values a
    training row of an uncovered name had — and the market context comes from
    any row of the file (it is identical for every ticker). ``compute_missing``
    instead builds that ticker's row on the spot (seconds: offline use only)."""
    snapshot = snapshot if snapshot is not None else load_session_snapshot(session_day)
    row = (snapshot or {}).get(ticker.upper())
    if row is None:
        if compute_missing:
            snap = ticker_snapshots(ticker, days=np.array([session_day], dtype=np.int64),
                                    slices=MarketTables([ticker]).slices(ticker))
            row = snap.iloc[0].to_dict()
            mk = market_context(np.array([session_day], dtype=np.int64))
            row.update({c: mk[c].iloc[0] for c in MARKET_FEATURES})
        else:
            any_row = next(iter((snapshot or {}).values()), {})
            row = {c: any_row.get(c, np.nan) for c in MARKET_FEATURES}
    s = pd.DataFrame([row])
    for c in ANCHOR_COLUMNS:
        if c not in s.columns:
            s[c] = np.nan
    bf = bar_features(s, np.array([close], float), np.array([bar_index], float))

    def _f(v):
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    out = {c: _f(row.get(c)) for c in SNAPSHOT_FEATURES + MARKET_FEATURES}
    out.update({c: float(v[0]) for c, v in bf.items()})
    return out


# ── the scorer's fallback when a session snapshot is missing ────────────────

def previous_session_day(day: int) -> int:
    """The NYSE session strictly before epoch-day ``day``."""
    from src.performance.market_calendar import previous_market_day
    d = pd.Timestamp(np.datetime64(int(day), "D")).date()
    return int(np.datetime64(previous_market_day(d), "D").astype(np.int64))


def recent_session_days(today: Optional[int] = None) -> Tuple[int, int]:
    """The two sessions a LIVE score can need a snapshot for: today's ET date
    (RTH and after-hours ticks score today's bars) and the previous session
    (pre-market and overnight ticks score its last bar). A request for any other
    day comes from a STALE tick cache, not from the clock."""
    if today is None:
        today = int(pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
                    .to_datetime64().astype("datetime64[D]").astype(np.int64))
    return int(today), previous_session_day(int(today))


_BUILD_LOCK = None
_BUILD_THREAD = None


def trigger_snapshot_build(day: int) -> bool:
    """Start building session ``day``'s snapshot in a SUBPROCESS — the scorer's
    fallback when the pre-open run did not produce it. Returns True when a build
    was started now.

    GLOBALLY single-flight, and only for a day in ``recent_session_days()``: one
    whole-market build is ~150 s and ~2.7 GB over 7 processes. Measured
    2026-09-23 with a single-flight keyed per DAY: one scoring pass over names
    whose tick caches had stopped on older sessions asked for eight different
    days and launched eight concurrent whole-market builds (48 pool workers)."""
    import subprocess
    import sys
    import threading
    global _BUILD_LOCK, _BUILD_THREAD
    if int(day) not in recent_session_days():
        return False
    if _BUILD_LOCK is None:
        _BUILD_LOCK = threading.Lock()
    with _BUILD_LOCK:
        if _BUILD_THREAD is not None and _BUILD_THREAD.is_alive():
            return False
        d = pd.Timestamp(np.datetime64(int(day), "D")).strftime("%Y-%m-%d")

        def _work():
            try:
                subprocess.run([sys.executable, "-m", "src.analysis.deep_features", "--snapshot", d],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
            except Exception as e:                       # noqa: BLE001
                logger.warning(f"[deep_features] snapshot build for {d} failed: {e}")
        _BUILD_THREAD = threading.Thread(target=_work, name=f"deep-snapshot-{d}", daemon=True)
        _BUILD_THREAD.start()
    logger.warning(f"[deep_features] session snapshot {d} missing — building it in the background "
                   "(the pre-open run should have); deep-feature models abstain until it lands")
    return True


if __name__ == "__main__":                              # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="deep features: build a session snapshot")
    ap.add_argument("--snapshot", required=True, help="session date YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    logger.add("logs/deep_refresh.log", rotation="1 day", retention="30 days", level="INFO", enqueue=True)
    # The scorer's fallback launches this from the scheduler at NORMAL priority;
    # a whole-market build must yield the CPU to the tick like the refresh does
    # (the pool's workers inherit a below-normal class on Windows).
    from src.data.deep.refresh import _below_normal_priority
    _below_normal_priority()
    build_session_snapshot(int(to_days([a.snapshot])[0]), workers=a.workers)
