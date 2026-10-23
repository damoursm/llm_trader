"""Each live arm's RELIABILITY CARD, the plateau rule and the bundle's coverage (user 2026-10-12, "Implement all 6":
items 1, 2 and 5).

* SPECIFY, DON'T SEARCH (item 1): `SPEC` states each live arm as a MECHANISM with the condition that defines its
  market context and the exit that defines its horizon, and names its few parameters; a parameter is set at the
  middle of a PLATEAU (`plateau`: the widest run of neighbouring values within a tolerance of the best — never a
  peak whose neighbours fall away).
* THE CARD (item 2): the same statistics for every arm — return per day (the standing definition,
  `account.per_day`) and the GROSS win rate when the arm is active, by calendar year and by market regime (SPY above /
  below its 200-session average; SPY's 20-session realized volatility in terciles), over the longest history the arm's
  data allows, by borrow-data set (`periods.set_of`), where it LOSES (a group of >= `MIN_GROUP` trades with a
  negative average), and its sensitivity to doubled costs, a one-bar entry delay and the lendability model's level
  (`sensitivity`). The trades come from the production adapters through `arm_alone`: the arm's pieces run through the
  account engine on their own with a start large enough that capital never binds, so each trade is the RULE'S trade
  at the live costs; `trades_from_positions` reads an account run's positions instead.
* THE BUNDLE (item 5, `bundle`): each arm's activity (the share of sessions with an open position), its share of the
  P&L, the activity OVERLAP between arms (the share of sessions both are active), and the co-activation HEADROOM: the
  initial-margin requirement of everything open against the equity — its peak tells how much larger the slices could
  be before a margin call at the same history.

CLI: `python -m src.backtest.reliability --build` writes `cache/ml/reliability/<day>/` (one card per arm, the
sensitivities, the bundle) and the live calibration's `reference.json` (`live_calibration.REFERENCE_FILE`: set B's
numbers, the market the live account trades in) with the FINGERPRINTS it was built on — each arm's production
settings (`arm_fingerprint`: its SPEC's parameters, switches and sizing) and the simulator's defaults
(`simulator_fingerprint`) — so the live calibration can tell a reference built for other rules (a deploy since) from
a current one: a reference must be rebuilt after every deploy (2026-10-21, the user's "Fix all").
"""
from __future__ import annotations

from contextlib import contextmanager

import dataclasses
import hashlib
import json
import math
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.backtest import periods
from src.backtest.account import per_day

ET = "America/New_York"
ARMS = ("vol", "etf", "fall", "dip", "crash", "cap")
SIDE = {"vol": "short", "etf": "short", "fall": "short", "dip": "long", "crash": "long", "cap": "long"}
MIN_GROUP = 10                       # a group of fewer trades says nothing about where the arm loses
PLATEAU_TOL = 0.1                    # a neighbour within 10% of the best metric belongs to the plateau
INIT_RATE = {"vol": 2.86, "etf": 2.86, "fall": 2.86, "dip": 0.5, "crash": 0.75, "cap": 0.5}   # the live account's

# item 1 — each live arm as a mechanism; the numbers live in config/settings.py, named here, never copied (a copied
# number goes stale at the next deploy). ``parameters`` are the rule's numbers, ``switches`` the parts of the rule
# turned on or off, ``sizing`` its slice and its open-position limit: together they are the settings that shape the
# arm's own trades, and their values are its fingerprint (`arm_settings`, `arm_fingerprint`).
_SHORT_COMMON = ["sel_short_min_price", "sel_short_min_dollar_volume", "sel_short_own_min_history",
                 "sel_short_runup_sessions", "sel_short_max_days_to_cover", "sel_short_vol_min_rvol",
                 "sel_short_arm_max_hold_sessions", "sel_short_max_open_per_ticker"]
_SHORT_SWITCHES = ["enable_sel_short_dtc_filter", "enable_sel_short_vol_rvol_filter", "enable_sel_short_squeeze_cover",
                   "enable_sel_short_volnorm_exit", "enable_sel_short_halt_guard", "enable_sel_short_borrow_retry",
                   "sel_short_traded_arms"]
SPEC = {
    "vol": {"mechanism": "a stock whose 30-minute volatility is the market's highest and new (a fresh top-1 ATR%) "
                         "after a rise over sel_short_runup_sessions gives part of the rise back",
            "context": "the pick bar: fresh top-1 ATR% over sel_short_vol_own_window_days, a riser, relative volume "
                       ">= sel_short_vol_min_rvol, days to cover <= sel_short_max_days_to_cover, lendable at IBKR "
                       "(else the bar's best passing backup within sel_short_vol_fallback_ranks), the pressure extras",
            "horizon": "sel_short_vol_give_back of the run-up given back, the sel_short_vol_cover_multiple cover, "
                       "sel_short_vol_max_hold_sessions sessions",
            "parameters": [p for p in _SHORT_COMMON if p != "sel_short_arm_max_hold_sessions"]
                          + ["sel_short_vol_max_hold_sessions", "sel_short_vol_own_window_days", "sel_short_vol_give_back",
                             "sel_short_vol_cover_multiple", "sel_short_vol_fallback_ranks"],
            "switches": _SHORT_SWITCHES + ["enable_sel_short_vol", "enable_sel_short_vol_added_stocks",
                                           "enable_sel_short_vol_fallback", "enable_sel_short_vol_pressure_extras",
                                           "enable_sel_short_vol_volnorm_exit"],
            "sizing": ["sel_short_vol_account_slices"]},
    "etf": {"mechanism": "an exchange-traded product whose 30-minute volatility is the products' highest and new, "
                         "after a rise, gives part of the rise back; entered on the turn",
            "context": "the vol rule on the products (the VIX-linked ones out, sel_short_etf_exclude_vix); the turn "
                       "of sel_short_turn_atr x its ATR within sel_short_turn_sessions",
            "horizon": "sel_short_etf_give_back of the run-up given back, the sel_short_cover_multiple cover, the "
                       "volatility exit in profit (sel_short_volnorm_ratio), sel_short_arm_max_hold_sessions sessions",
            "parameters": _SHORT_COMMON + ["sel_short_etf_own_window_days", "sel_short_turn_atr",
                                           "sel_short_turn_sessions", "sel_short_etf_give_back",
                                           "sel_short_cover_multiple", "sel_short_volnorm_ratio"],
            "switches": _SHORT_SWITCHES + ["enable_sel_short_etf", "enable_sel_short_etf_turn",
                                           "sel_short_etf_exclude_vix"],
            "sizing": ["sel_short_etf_account_slices"]},
    "fall": {"mechanism": "the market's most volatile name that did NOT rise keeps falling once it turns down from "
                          "its high since the pick",
             "context": "the vol arm's pick on a non-riser with a valid base; the turn of sel_short_turn_atr x its "
                        "ATR within sel_short_turn_sessions",
             "horizon": "no target: the volatility exit in profit (sel_short_fall_volnorm_ratio), the "
                        "sel_short_cover_multiple cover, sel_short_arm_max_hold_sessions sessions",
             "parameters": _SHORT_COMMON + ["sel_short_vol_own_window_days", "sel_short_turn_atr",
                                            "sel_short_turn_sessions", "sel_short_fall_volnorm_ratio",
                                            "sel_short_cover_multiple"],
             "switches": _SHORT_SWITCHES + ["enable_sel_short_fall", "enable_sel_short_fall_turn"],
             "sizing": ["sel_short_fall_account_slices"]},
    "dip": {"mechanism": "a large stock in an uptrend, oversold over two sessions, bought on a further dip, "
                         "rebounds to its short average",
            "context": "dollar volume >= dip_long_min_dollar_volume (the order among those >= "
                       "dip_long_order_min_dollar_volume), close > the dip_long_trend_sessions average, RSI2 < "
                       "dip_long_rsi_max; the meta-model's cut (enable_dip_long_filter), then each day's top pick by "
                       "its score (dip_long_daily_top, dip_long_rank), an earnings dip ranked last "
                       "(dip_long_earnings_demote_sessions); the dip_long_limit_atr limit alive dip_long_limit_sessions",
            "horizon": "the first close above the dip_long_exit_sma_sessions average, or dip_long_max_hold_sessions",
            "parameters": ["dip_long_min_dollar_volume", "dip_long_order_min_dollar_volume", "dip_long_rsi_max",
                           "dip_long_trend_sessions", "dip_long_daily_top", "dip_long_rank",
                           "dip_long_earnings_demote_sessions", "dip_long_entry_mode", "dip_long_limit_atr",
                           "dip_long_limit_sessions", "dip_long_exit_sma_sessions", "dip_long_max_hold_sessions",
                           "dip_long_filter_model"],
            "switches": ["enable_dip_long", "enable_dip_long_filter", "enable_dip_long_earnings_demote",
                         "enable_dip_long_own_security"],
            "sizing": ["dip_long_account_slices"]},
    "crash": {"mechanism": "the index in a downtrend, capitulating over two sessions, rebounds; traded in its "
                           "leveraged fund (crash_long_trade_symbol)",
              "context": "crash_long_symbol below its crash_long_trend_sessions average with RSI2 < "
                         "crash_long_rsi_max; crash_long_trade_symbol bought at the open",
              "horizon": "the first close above the crash_long_exit_sma_sessions average, or "
                         "crash_long_max_hold_sessions — sold after hours that evening (crash_long_pm_exit)",
              "parameters": ["crash_long_symbol", "crash_long_trade_symbol", "crash_long_rsi_max",
                             "crash_long_trend_sessions", "crash_long_exit_sma_sessions",
                             "crash_long_max_hold_sessions", "crash_long_margin_rate"],
              "switches": ["enable_crash_long", "crash_long_pm_exit"],
              "sizing": ["crash_long_account_slices"]},
    "cap": {"mechanism": "on a market-wide capitulation day, every large stock oversold under its own downtrend, "
                         "bought on a further dip, rebounds to its short average",
            "context": "the share of the cap_long_breadth_top most-traded stocks under their trend with RSI2 < 10 at "
                       ">= cap_long_breadth_min; each stock with dollar volume >= cap_long_min_dollar_volume, close < "
                       "the cap_long_trend_sessions average, RSI2 < cap_long_rsi_max; the cap_long_limit_atr limit "
                       "alive cap_long_limit_sessions",
            "horizon": "the first close above the cap_long_exit_sma_sessions average, or cap_long_max_hold_sessions",
            "parameters": ["cap_long_breadth_top", "cap_long_breadth_min", "cap_long_min_dollar_volume",
                           "cap_long_trend_sessions", "cap_long_rsi_max", "cap_long_limit_atr",
                           "cap_long_limit_sessions", "cap_long_exit_sma_sessions", "cap_long_max_hold_sessions"],
            "switches": ["enable_cap_long"],
            "sizing": ["cap_long_account_slices", "cap_long_max_open"]},
}


def arm_settings(arm: str) -> Dict[str, object]:
    """The production settings that shape ``arm``'s own trades — its SPEC's parameters, switches and sizing — as
    {name: value} (a value JSON can hold: numbers, strings, booleans)."""
    sp = SPEC[arm]
    names = sorted(set(sp.get("parameters", [])) | set(sp.get("switches", [])) | set(sp.get("sizing", [])))
    return {n: getattr(settings, n) for n in names}


def _digest(values: Mapping) -> str:
    return hashlib.sha1(json.dumps(values, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def arm_fingerprint(arm: str) -> str:
    """A short digest of `arm_settings`: equal exactly when every setting that shapes the arm's trades is equal."""
    return _digest(arm_settings(arm))


def simulator_fingerprint() -> str:
    """A short digest of the production-mimicking simulator's defaults (every field of `mimic.Mimic()`): a market-model
    correction or a new default changes the simulator's trades without any production setting changing."""
    from src.backtest import mimic as M
    m = M.Mimic()
    return _digest({f.name: repr(getattr(m, f.name)) for f in dataclasses.fields(m)})


# ── the market's regimes ─────────────────────────────────────────────────────

def spy_daily(path: Optional[Path] = None) -> pd.DataFrame:
    """SPY's daily closes from the OHLCV cache (pandas' split layout), indexed by session date."""
    p = Path(path) if path else Path("cache") / "ohlcv" / "SPY.json"
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    df = pd.DataFrame(raw["data"], columns=raw["columns"], index=pd.to_datetime(raw["index"]))
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[["Close"]].astype(float).sort_index()


def regime_table(spy: Optional[pd.DataFrame] = None, trend_sessions: int = 200, vol_sessions: int = 20) -> pd.DataFrame:
    """Per session: ``trend`` = "above" / "below" (the close vs its ``trend_sessions`` average) and ``vol`` = "low" /
    "mid" / "high" (the ``vol_sessions`` realized volatility of daily log returns, in terciles over the whole history
    — a label for READING results, never an input of a decision)."""
    spy = spy_daily() if spy is None else spy
    c = spy["Close"].astype(float)
    sma = c.rolling(trend_sessions).mean()
    rv = np.log(c).diff().rolling(vol_sessions).std() * math.sqrt(252)
    out = pd.DataFrame(index=spy.index)
    out["trend"] = np.where(sma.isna(), None, np.where(c > sma, "above", "below"))
    q = rv.dropna().quantile([1 / 3, 2 / 3]).to_numpy()
    out["vol"] = np.where(rv.isna(), None, np.where(rv <= q[0], "low", np.where(rv <= q[1], "mid", "high")))
    out["rvol"] = rv
    return out


def regime_of(days: Iterable, table: pd.DataFrame) -> pd.DataFrame:
    """The regime labels in force at each day (the last session at or before it; None before the table starts)."""
    d = pd.DatetimeIndex(pd.to_datetime(list(days))).tz_localize(None).normalize()
    idx = table.index.searchsorted(d, side="right") - 1
    ok = idx >= 0
    out = pd.DataFrame({"trend": [None] * len(d), "vol": [None] * len(d)}, index=range(len(d)))
    if ok.any():
        sub = table.iloc[idx[ok]]
        out.loc[np.flatnonzero(ok), "trend"] = sub["trend"].to_numpy()
        out.loc[np.flatnonzero(ok), "vol"] = sub["vol"].to_numpy()
    return out


# ── the trades ───────────────────────────────────────────────────────────────

COLUMNS = ("book", "ticker", "entry", "exit", "days", "net", "gross", "pnl", "decision")


def _ts(ns: Iterable[int]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(list(ns), np.int64), unit="ns", utc=True)).tz_convert(ET)


def trades_from_positions(pos: Sequence[dict]) -> pd.DataFrame:
    """The trade table of an account run's positions — the engine's own records (``shares`` / ``entry_px`` /
    ``exit_px`` / ``pnl``: the gross return from the prices, the net one from the P&L on the notional) or a study's
    compact copy (``ret`` net, ``pnl``; no gross)."""
    rows = []
    for p in pos:
        if not p.get("exit_ns"):
            continue
        notional = float(p["shares"]) * float(p["entry_px"]) if "shares" in p and "entry_px" in p else \
            (abs(float(p["pnl"]) / float(p["ret"])) if p.get("ret") else float("nan"))
        if "shares" in p and "entry_px" in p:
            sign = -1.0 if (p.get("side", 1) < 0 or SIDE.get(str(p["strategy"])) == "short") else 1.0
            gross = sign * (float(p["exit_px"]) / float(p["entry_px"]) - 1.0) * 100.0 if p.get("exit_px") else float("nan")
            net = float(p["pnl"]) / notional * 100.0 if notional else float("nan")
        else:
            gross, net = float("nan"), float(p["ret"]) * 100.0
        pk = p.get("pick_day")
        rows.append((str(p["strategy"]), str(p["tkn"]), int(p["entry_ns"]), int(p["exit_ns"]), net, gross,
                     float(p.get("pnl", float("nan"))),
                     (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(pk))).date() if pk is not None and int(pk) >= 0 else None))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=["book", "ticker", "ens", "xns", "net", "gross", "pnl", "decision"])
    df["entry"], df["exit"] = _ts(df.ens), _ts(df.xns)
    df["days"] = (df.xns - df.ens) / 86400e9
    return df[list(COLUMNS)].reset_index(drop=True)


def trades_from_table(df: pd.DataFrame, book: str, entry: str, exit: str, net: str, gross_win: Optional[str] = None,
                      days: Optional[str] = None, ticker: Optional[str] = None) -> pd.DataFrame:
    """A study's own trade table (a long history: the dip book's signals 2006->, the crash book's 1986->) as the
    card's table. ``gross_win`` names a boolean column (the gross return itself is not kept: ``gross`` carries +1 / -1
    so the win rate reads from it); ``days`` the 24-hour hold, else the calendar days between the dates."""
    out = pd.DataFrame({"book": book, "ticker": df[ticker].astype(str) if ticker else "",
                        "entry": pd.to_datetime(df[entry]), "exit": pd.to_datetime(df[exit]),
                        "net": df[net].astype(float)})
    out["days"] = df[days].astype(float) if days else (out.exit - out.entry).dt.total_seconds() / 86400.0
    out["gross"] = np.where(df[gross_win].astype(bool), 1.0, -1.0) if gross_win else np.nan
    out["pnl"] = np.nan
    out["decision"] = out.entry.dt.date
    return out[list(COLUMNS)].reset_index(drop=True)


def arm_alone(pieces: Sequence[dict], strategies: dict, rules, settle=None, start: float = 100_000.0) -> pd.DataFrame:
    """The arm's own trades: its pieces through the account engine with a start that never binds (every piece at its
    slice, no give-way, no margin call at `start` = $100k: a 1/20 slice is $5k, under the 1% dollar-volume cap of any
    $5M name), so the table holds the RULE'S trades at the live costs — gross from the prices, net from the P&L."""
    from src.backtest import account as A
    books = {str(p["strategy"]) for p in pieces}
    strat = {k: v for k, v in _by_name(strategies).items() if k in books}
    res = A.simulate(list(pieces), strat, card_rules(rules, start), settle or A.default_settle())
    pick = {}
    for p in pieces:                                   # the decision day of each entry (an order's pieces: the order's)
        for ns in ([int(p["ens"])] if p.get("ens") is not None else []) + list(map(int, p.get("ord_place", []) or [])):
            pick.setdefault((str(p["strategy"]), str(p["tkn"]), ns), p.get("pick_day"))
    pos = [dict(q, pick_day=pick.get((str(q["strategy"]), str(q["tkn"]), int(q["entry_ns"])), q.get("pick_day")))
           for q in res["positions"]]
    return trades_from_positions(pos)


def card_rules(rules, start: float = 100_000.0):
    """An arm card's account rules: the live ones at a start that never binds, the margin judged as before PREREG174
    (at the tick marks, every position closed — which never fires at $100k): a card holds the RULE'S trades, never a
    margin model's liquidation (IBKR's intrabar check could take a lot out of a $100k book in a squeeze)."""
    return replace(rules, start=float(start), trigger="close", liquidation="all", sem=0.0, regt_close=False)


def _by_name(strategies) -> dict:
    """`mimic.live_strategies` hands a list of Strategy objects; a dict by name is accepted too."""
    return dict(strategies) if isinstance(strategies, dict) else {s.name: s for s in strategies}


# ── the statistics ───────────────────────────────────────────────────────────

def stats(t: pd.DataFrame) -> dict:
    """The arm's quality on these trades: n, the average and median net trade (%), the average hold (24-hour days),
    RETURN PER DAY (%/day), the GROSS win rate (from ``gross`` where known, else the net one, named), the worst and
    best trade, and how much of the net P&L the five best trades carry (concentration)."""
    n = int(len(t))
    if n == 0:
        return {"n": 0}
    net = t.net.to_numpy(float)
    days = t.days.to_numpy(float)
    out = {"n": n, "avg_trade": float(net.mean()), "med_trade": float(np.median(net)), "avg_days": float(days.mean()),
           "return_per_day": per_day(float(net.mean()), float(days.mean())),
           "worst": float(net.min()), "best": float(net.max())}
    g = t.gross.to_numpy(float)
    if np.isfinite(g).any():
        out["gross_win_rate"] = float((g[np.isfinite(g)] > 0).mean())
        out["win_basis"] = "gross"
    else:
        out["gross_win_rate"] = float((net > 0).mean())
        out["win_basis"] = "net"
    top = np.sort(net)[-5:]
    out["top5_share"] = float(top.sum() / net.sum()) if net.sum() > 0 else None
    return out


def by(t: pd.DataFrame, key: pd.Series) -> Dict[str, dict]:
    out = {}
    for k, g in t.groupby(key.to_numpy(object)):
        if k is None or (isinstance(k, float) and math.isnan(k)):
            continue
        out[str(k)] = stats(g)
    return out


def card(t: pd.DataFrame, arm: str, regimes: Optional[pd.DataFrame] = None) -> dict:
    """The arm's card: its spec, the statistics overall, by year, by borrow-data set, by trend and by volatility
    regime (at the entry), the groups where it loses (>= MIN_GROUP trades with a negative average trade)."""
    t = t[t.book == arm].reset_index(drop=True) if "book" in t else t
    years = pd.DatetimeIndex(t.entry).tz_localize(None).year if len(t) else pd.Index([])
    rec = {"arm": arm, "side": SIDE.get(arm), "spec": SPEC.get(arm), "overall": stats(t),
           "by_year": by(t, pd.Series(years.astype(str), index=t.index)) if len(t) else {},
           "by_set": by(t, pd.Series([periods.set_of(d, extra=True) or "pre" for d in
                                      [dd if dd is not None else ed for dd, ed in
                                       zip(t.decision if "decision" in t else [None] * len(t),
                                           pd.DatetimeIndex(t.entry).tz_localize(None).date)]], index=t.index)) if len(t) else {}}
    if regimes is not None and len(t):
        r = regime_of(pd.DatetimeIndex(t.entry).tz_localize(None), regimes)
        rec["by_trend"] = by(t, pd.Series(r.trend.to_numpy(object), index=t.index))
        rec["by_vol"] = by(t, pd.Series(r.vol.to_numpy(object), index=t.index))
    rec["loses_in"] = [f"{grp}={k}" for grp in ("by_year", "by_set", "by_trend", "by_vol")
                       for k, v in rec.get(grp, {}).items() if v.get("n", 0) >= MIN_GROUP and v.get("avg_trade", 0) < 0]
    return rec


def sensitivity(base: pd.DataFrame, variants: Dict[str, pd.DataFrame], arm: str) -> dict:
    """The arm's statistics under each variant (doubled costs, a one-bar entry delay, the lendability model's level
    shifted) beside the base, overall and by set, with the change in return per day."""
    out = {}
    b = {k: v for k, v in card(base, arm).items() if k in ("overall", "by_set")}
    for name, t in variants.items():
        c = {k: v for k, v in card(t, arm).items() if k in ("overall", "by_set")}
        d = {"overall": c["overall"], "by_set": c["by_set"],
             "rpd_change": (c["overall"].get("return_per_day", float("nan")) - b["overall"].get("return_per_day", float("nan")))
             if c["overall"].get("n") and b["overall"].get("n") else None,
             "rpd_change_by_set": {s: c["by_set"][s]["return_per_day"] - b["by_set"][s]["return_per_day"]
                                   for s in c["by_set"] if s in b["by_set"] and c["by_set"][s].get("n") and b["by_set"][s].get("n")}}
        out[name] = d
    return {"base": b, "variants": out}


# ── item 1: the plateau rule ─────────────────────────────────────────────────

def plateau(points: Sequence[Tuple[float, float]], tol: float = PLATEAU_TOL, higher_is_better: bool = True) -> dict:
    """Of (parameter, metric) points on a grid: the best value, the PLATEAU (the run of neighbouring parameters whose
    metric is within ``tol`` x |best| of the best, the run holding the best), its CENTER (the middle parameter of the
    run — the value to take) and whether the best is a PEAK (a run of one: its neighbours fall away). NaN metrics
    break a run."""
    pts = sorted((float(p), float(m)) for p, m in points)
    if not pts:
        return {"best": None, "center": None, "plateau": [], "on_peak": None}
    ms = np.array([m for _, m in pts], float)
    ok = np.isfinite(ms)
    if not ok.any():
        return {"best": None, "center": None, "plateau": [], "on_peak": None}
    sign = 1.0 if higher_is_better else -1.0
    i_best = int(np.nanargmax(sign * np.where(ok, ms, np.nan)))
    best = ms[i_best]
    band = abs(best) * float(tol)
    within = ok & (sign * (ms - best) >= -band)
    lo = i_best
    while lo - 1 >= 0 and within[lo - 1]:
        lo -= 1
    hi = i_best
    while hi + 1 < len(pts) and within[hi + 1]:
        hi += 1
    run = [pts[k][0] for k in range(lo, hi + 1)]
    return {"best": pts[i_best][0], "best_metric": float(best), "plateau": run, "center": run[len(run) // 2],
            "width": len(run), "on_peak": len(run) == 1}


# ── item 5: the bundle ───────────────────────────────────────────────────────

def bundle(pos: Sequence[dict], start: float = 10_000.0, init_rate: Optional[Dict[str, float]] = None,
           sessions: Optional[pd.DatetimeIndex] = None) -> dict:
    """The arms' coverage from an account run's positions: per arm its activity (the share of sessions with a
    position open), trades, P&L share and mean notional share of equity; the overlap matrix (the share of sessions
    both arms are active); and the co-activation HEADROOM — the initial-margin requirement of everything open
    (notional x the arm's initial rate: a short at the house 2.86 unless the position carries ``hi``, a dip long at
    Reg T's 0.5, the crash long at 0.75) against the equity (the start plus the P&L realized so far): its peak and
    95th percentile as a share of equity, and ``slice_multiple``: 1 / peak, how much larger every slice could have
    been before the requirement reached the equity on this history (a margin call lies at the maintenance level,
    lower still — read it as an upper bound)."""
    rate = dict(INIT_RATE, **(init_rate or {}))
    P = [p for p in pos if p.get("exit_ns")]
    if not P:
        return {"arms": {}, "overlap": {}, "headroom": None}
    ens = np.array([int(p["entry_ns"]) for p in P], np.int64)
    xns = np.array([int(p["exit_ns"]) for p in P], np.int64)
    books = np.array([str(p["strategy"]) for p in P], object)
    pnl = np.array([float(p.get("pnl", 0.0)) for p in P], float)
    notional = np.array([float(p["shares"]) * float(p["entry_px"]) if "shares" in p and "entry_px" in p
                         else (abs(float(p["pnl"]) / float(p["ret"])) if p.get("ret") else 0.0) for p in P], float)
    req_rate = np.array([float(p["hi"]) if p.get("hi") is not None else rate.get(str(p["strategy"]), 1.0) for p in P], float)
    if sessions is None:
        lo = pd.Timestamp(int(ens.min()), unit="ns", tz="UTC").tz_convert(ET).normalize()
        hi = pd.Timestamp(int(xns.max()), unit="ns", tz="UTC").tz_convert(ET).normalize()
        sessions = pd.bdate_range(lo, hi, tz=ET)
    t = np.array([pd.Timestamp(s).replace(hour=16, minute=0).value for s in sessions], np.int64)
    opn = (ens[None, :] <= t[:, None]) & (t[:, None] < xns[None, :])                    # sessions x positions
    order = np.argsort(xns)
    realized = np.zeros(len(t))
    for k in order:                                                                    # equity = start + realized P&L
        realized[t >= xns[k]] += pnl[k]
    equity = start + realized
    arms = sorted(set(books))
    active = {a: opn[:, books == a].any(axis=1) for a in arms}
    total_pnl = pnl.sum()
    out_arms = {}
    for a in arms:
        m = books == a
        held = (opn[:, m] * notional[m][None, :]).sum(axis=1) / np.maximum(equity, 1e-9)
        out_arms[a] = {"trades": int(m.sum()), "active_share": float(active[a].mean()),
                       "pnl": float(pnl[m].sum()), "pnl_share": float(pnl[m].sum() / total_pnl) if total_pnl else None,
                       "notional_share_mean": float(held.mean()), "notional_share_max": float(held.max())}
    overlap = {a: {b: float((active[a] & active[b]).mean()) for b in arms} for a in arms}
    req = (opn * (notional * req_rate)[None, :]).sum(axis=1) / np.maximum(equity, 1e-9)
    any_active = opn.any(axis=1)
    out = {"arms": out_arms, "overlap": overlap,
           "sessions": int(len(t)), "any_active_share": float(any_active.mean()),
           "headroom": {"req_peak": float(req.max()), "req_p95": float(np.percentile(req, 95)),
                        "req_mean": float(req.mean()), "slice_multiple": float(1.0 / req.max()) if req.max() > 0 else None,
                        "peak_session": str(pd.Timestamp(int(t[int(req.argmax())]), unit="ns", tz="UTC").tz_convert(ET).date()),
                        "active_at_peak": sorted({str(b) for b in books[opn[int(req.argmax())]]})}}
    return out


# ── the build (the simulator's world) ────────────────────────────────────────

def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) or isinstance(o, np.floating)) and not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def out_dir(day: Optional[date] = None) -> Path:
    d = Path(getattr(settings, "reliability_dir", "cache/ml/reliability"))
    return d / (day or datetime.now().date()).isoformat()


@contextmanager
def _halt_files(s: str):
    """Set A0's halt history beside the existing files while its books are built (`a0.harness`'s halt paths: NYSE's
    history and Nasdaq Trader's feed before August 2019) — production's halt guard is `Mimic()`'s default from
    2026-10-19 (AUDIT151); the other sets read the existing files. Restored after."""
    if s != "A0":
        yield
        return
    from src.backtest import a0 as A0
    from src.backtest import halts as H
    saved = H.halt_paths()
    H.set_halt_paths(A0.halt_files())
    try:
        yield
    finally:
        H.set_halt_paths(saved)


def build(sets: Sequence[str] = ("A", "B"), sensitivities: bool = True, seed: int = 0, cost_snapshot: Optional[str] = None,
          extra_bars: Optional[str] = None, day: Optional[date] = None, log=print) -> Path:
    """Build every live arm's card from the production-mimicking simulator (`mimic.live_books` under `Mimic()`),
    each arm alone at the live costs, for the borrow-data sets named; with ``sensitivities`` the same under doubled
    leg costs, a one-bar entry delay (`Mimic(halts=None).without("entry")`) and the lendability model's level shifted
    (`Mimic(lend_shift=+-0.37)`, set B). Writes the cards, the bundle of the base run and the live calibration's
    reference (set B's base statistics). Research paths default to PREREG62's cost snapshot and extra bars."""
    from src.backtest import account as A
    from src.backtest import live_calibration as LC
    from src.backtest import mimic as M
    from src.backtest.strategies import dip as DIP
    from src.backtest.strategies.etf import EtfArm
    from src.backtest.strategies.vol import VolEngine
    research = Path(settings.backtest_research_dir)
    snap = cost_snapshot or str(research / "studies" / "prereg62" / "cost_snapshot.json")
    extra = extra_bars or str(research / "studies" / "prereg62" / "ext_bars")
    out = out_dir(day)
    if tuple(sets) != ("A", "B"):                     # another selection of sets never overwrites the A / B cards
        out = out.with_name(f"{out.name}_{'-'.join(sets)}")
    out.mkdir(parents=True, exist_ok=True)
    comps: Dict[str, tuple] = {}

    def components(s: str) -> tuple:
        """(engine, ETF arm, dip panel, extended bars, settlement) of a set: PREREG101's set A0 its own data
        (`src.backtest.a0`), the others the default stores (loaded once, only when asked for)."""
        key = "A0" if s == "A0" else "default"
        if key not in comps:
            if key == "A0":
                from src.backtest import a0 as A0
                comps[key] = (A0.vol_engine(), A0.etf_arm(), A0.dip_panel(), A0.ext_bars(), A0.settle())
            else:
                comps[key] = (VolEngine(nbbo=False), EtfArm(), DIP.build_panel(), M.ExtBars(extra), A.default_settle())
        return comps[key]
    costs = M.LedgerCosts(snap)
    reg = regime_table()
    variants = {"base": M.Mimic(seed=seed)}
    if sensitivities:
        # the halt model needs the entry at the pick bar's close (`halts.Applier`): the delay is read without it
        variants["entry_delay"] = M.Mimic(seed=seed, halts=None).without("entry")
        variants["lend_lo"] = M.Mimic(seed=seed, lend_shift=-0.37)
        variants["lend_hi"] = M.Mimic(seed=seed, lend_shift=+0.37)
    tables: Dict[str, pd.DataFrame] = {}
    pos_base: List[dict] = []
    doubled: List[pd.DataFrame] = []
    for name, mim in variants.items():
        parts = []
        for s in sets:
            S = periods.get(s)
            eng, etf, panel, bars, settle = components(s)
            log(f"[reliability] {name} set {s}: building the books")
            with _halt_files(s):
                B = M.live_books(S.lo, S.hi, mim, eng=eng, etf=etf, panel=panel, bars=bars, costs=costs,
                                 settle=settle)
            strategies, rules = _by_name(M.live_strategies(mim)), M.live_rules(mim)
            lo_dn, hi_dn = M.dnum(S.lo), M.dnum(S.hi)
            for arm, pcs in B.items():
                pcs = [p for p in pcs if lo_dn <= int(p.get("pick_day", -1)) <= hi_dn]
                if not pcs:
                    continue
                parts.append(arm_alone(pcs, strategies, rules, settle))
                if name == "base":
                    res = A.simulate(pcs, {arm: strategies[arm]}, card_rules(rules), settle)
                    pos_base.extend(res["positions"])
            if name == "base" and sensitivities:
                for arm, pcs in B.items():
                    pcs2 = [dict(p, hs_in=2 * float(p["hs_in"]), hs_out=2 * float(p["hs_out"])) for p in pcs
                            if lo_dn <= int(p.get("pick_day", -1)) <= hi_dn]
                    if pcs2:
                        doubled.append(arm_alone(pcs2, strategies, rules, settle))
        tables[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
        if name == "base" and sensitivities:
            tables["costs_x2"] = pd.concat(doubled, ignore_index=True) if doubled else pd.DataFrame(columns=COLUMNS)
    base = tables["base"]
    cards = {arm: card(base, arm, reg) for arm in ARMS if (base.book == arm).any()}
    for arm, c in cards.items():
        c["sensitivity"] = sensitivity(base, {k: v for k, v in tables.items() if k != "base"}, arm) if sensitivities else None
        (out / f"{arm}.json").write_text(json.dumps(_json_ready(c), indent=1), encoding="utf-8")
    bnd = bundle(pos_base, start=100_000.0)
    (out / "bundle_arms_alone.json").write_text(json.dumps(_json_ready(bnd), indent=1), encoding="utf-8")
    base.to_pickle(out / "trades_base.pkl")
    if "B" not in sets:                               # the live calibration's reference is set B's: never another's
        log(f"[reliability] cards for {sorted(cards)} -> {out} (no set B: the live calibration's reference unchanged)")
        return out
    ref = {"as_of": (day or datetime.now().date()).isoformat(), "source": "reliability.build: each arm alone, set B, Mimic() seed %d" % seed,
           "built_at": datetime.now().astimezone().isoformat(timespec="seconds"), "arms": {},
           # what the reference was built on: the live calibration compares them with the deployed ones
           "simulator": simulator_fingerprint(), "fingerprints": {}, "settings": {}}
    for arm, c in cards.items():
        s = c["by_set"].get("B") or c["overall"]
        if s.get("n"):
            ref["arms"][arm] = {"return_per_day": s["return_per_day"], "gross_win_rate": s["gross_win_rate"],
                                "avg_trade": s["avg_trade"], "avg_days": s["avg_days"], "n": s["n"], "set": "B" if c["by_set"].get("B") else "all"}
            ref["fingerprints"][arm] = arm_fingerprint(arm)
            ref["settings"][arm] = arm_settings(arm)
    LC.out_dir().mkdir(parents=True, exist_ok=True)
    (LC.out_dir() / LC.REFERENCE_FILE).write_text(json.dumps(_json_ready(ref), indent=1), encoding="utf-8")
    log(f"[reliability] cards for {sorted(cards)} -> {out}; reference -> {LC.out_dir() / LC.REFERENCE_FILE}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="each live arm's reliability card from the production-mimicking simulator")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--no-sensitivities", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost-snapshot")
    ap.add_argument("--extra-bars")
    a = ap.parse_args(argv)
    if a.build:
        build(tuple(x.strip() for x in a.sets.split(",") if x.strip()), not a.no_sensitivities, a.seed, a.cost_snapshot,
              a.extra_bars)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
