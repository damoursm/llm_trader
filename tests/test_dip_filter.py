"""PREREG96's RP33 — the dip book's meta-model filter (`src/signals/dip_filter.py`; OFF in production until the user
deploys it: `enable_dip_long_filter`). What must hold: the inputs are data96's construction (verbatim copies below);
the fit is model96's (training rows = the filled signals exiting before the cut, the P33 cut = the 33rd percentile of
the out-of-fold scores with the signal years folded ``years[i::5]``, kept = score >= cut); the saved model is the fit
and its checksum is checked; a dropped signal never becomes an order — journaled ``filtered`` with its score in the
signal and order files, never placed, rested or filled; fail-open (no model, no SPY close: every signal kept, a
scorer-banner problem); with the switch off the book is unchanged. The look-ahead guards are in
`tests/test_no_lookahead.py`; the parity with the study's own numbers in
`llm_trader_research/studies/prereg96/prod96/parity96.py`."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import tracker
from src.signals import dip_filter as DF
from src.signals import dip_long as dl
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                                  # a Thursday session (signal close 2026-10-07)
PREV = date(2026, 10, 7)
F = list(DF.FEATURES)


# ── data96's construction, verbatim (TREND 200; `limit_price` at the live 1/2 ATR) ──────────────────────────────

def _d96_feature_arrays(d):
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("o", "h", "l", "c", "v"))
    s = pd.Series(c)
    pc = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid="ignore"):
        tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
        lr = np.log(c / pc)

    def back(k):
        return np.r_[np.full(k, np.nan), c[:-k]]
    return {"close": c, "sma200": s.rolling(200).mean().to_numpy(), "sma5": s.rolling(5).mean().to_numpy(),
            "rsi2": dl.rsi2(c), "atr14": pd.Series(tr).rolling(14).mean().to_numpy(),
            "dv20": pd.Series(c * v).rolling(20).mean().to_numpy(),
            "hi20": pd.Series(h).rolling(20).max().to_numpy(), "hi20c": s.rolling(20).max().to_numpy(),
            "ret1": c / pc - 1.0, "ret5": c / back(5) - 1.0, "ret20": c / back(20) - 1.0,
            "rvol20": pd.Series(lr).rolling(20).std(ddof=1).to_numpy()}


def _d96_features_at(F_, i):
    c = float(F_["close"][i])
    lim = dl.limit_price(c, F_["atr14"][i])
    dv = float(F_["dv20"][i])
    return {"rsi2": float(F_["rsi2"][i]), "dist_sma200_pct": 100.0 * (c / F_["sma200"][i] - 1.0),
            "dist_hi20_pct": 100.0 * (c / F_["hi20"][i] - 1.0), "dist_hi20c_pct": 100.0 * (c / F_["hi20c"][i] - 1.0),
            "atr14_pct": 100.0 * F_["atr14"][i] / c, "ret1_pct": 100.0 * F_["ret1"][i], "ret5_pct": 100.0 * F_["ret5"][i],
            "ret20_pct": 100.0 * F_["ret20"][i], "rvol20_pct": 100.0 * F_["rvol20"][i],
            "log_dv20": math.log10(dv) if dv > 0 else np.nan,
            "limit_dist_pct": np.nan if lim is None else 100.0 * (lim / c - 1.0)}


def _d96_spy(d):
    c = d["c"]
    lr = np.log(c / c.shift(1))
    return pd.DataFrame({"spy_dist_sma200_pct": 100.0 * (c / c.rolling(200).mean() - 1.0),
                         "spy_ret5_pct": 100.0 * (c / c.shift(5) - 1.0), "spy_ret20_pct": 100.0 * (c / c.shift(20) - 1.0),
                         "spy_rvol20_pct": 100.0 * lr.rolling(20).std(ddof=1)}, index=d.index)


def _same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


# ── synthetic series ─────────────────────────────────────────────────────────

def _daily(days, seed=3, drop_last=2, vol=3e7, dips=()):
    """A $1B+-a-day daily uptrend (o h l c v on the sessions ``days``) whose last ``drop_last`` sessions fall 4% each —
    a 2-session-RSI dip above the 200-session average at the last close; ``dips``: positions p where sessions p and
    p + 1 fall 4% each, the walk going on from there."""
    rng = np.random.default_rng(seed)
    n = len(days)
    r = rng.normal(0.003, 0.012, n)
    for p in dips:
        r[p] = r[p + 1] = math.log(0.96)
    c = 100.0 * np.exp(np.cumsum(r))
    for k in range(drop_last):
        c[n - drop_last + k] = c[n - drop_last + k - 1] * 0.96
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": rng.uniform(0.5, 1.5, n) * vol},
                        index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))


def test_the_name_inputs_are_data96s():
    d = _daily(ss.sessions_before(DAY, 400))
    want_arrays = _d96_feature_arrays(d)
    for i in (230, 321, 399):
        got = DF.name_features(d, d.index[i].date())
        assert got.pop("session") == d.index[i].date()
        want = _d96_features_at(want_arrays, i)
        assert set(got) == set(DF.NAME_FEATURES)
        for k in DF.NAME_FEATURES:
            assert _same(got[k], want[k]), (i, k, got[k], want[k])


def test_the_spy_inputs_are_data96s_and_need_the_decision_close():
    d = _daily(ss.sessions_before(DAY, 300), seed=9, drop_last=0)
    pd.testing.assert_frame_equal(DF.spy_frame_features(d), _d96_spy(d))
    got = DF.spy_features_at(d, PREV)
    assert got == {k: float(_d96_spy(d).loc[pd.Timestamp(PREV), k]) for k in DF.SPY_FEATURES}
    assert DF.spy_features_at(d.iloc[:-1], PREV) is None              # the cache lacks D-1: not this source


def test_spy_comes_from_the_daily_cache_else_the_deep_store(monkeypatch):
    d = _daily(ss.sessions_before(DAY, 300), seed=9, drop_last=0)
    monkeypatch.setattr(DF, "cache_daily", lambda tk: d)
    assert DF.spy_inputs(PREV)[1] == "cache"
    monkeypatch.setattr(DF, "cache_daily", lambda tk: d.iloc[:-1])    # the EOD warm did not run
    monkeypatch.setattr(dl, "_daily", lambda tk: d)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    got, src = DF.spy_inputs(PREV)
    assert src == "deep_store" and got == DF.spy_features_at(d, PREV)
    monkeypatch.setattr(dl, "_daily", lambda tk: d.iloc[:-1])
    assert DF.spy_inputs(PREV) == (None, "missing")


def test_days_since_prev_is_the_names_last_firing_else_the_table_before_the_store():
    days = ss.sessions_before(DAY, 400)
    d = _daily(days, seed=5, drop_last=0)
    fired = d.index[DF.rule_mask(d)]
    assert len(fired) >= 2                                            # the walk fires (a $1B+ uptrend's dips)
    s = pd.Timestamp(fired[-1]).date()
    p = pd.Timestamp(fired[-2]).date()
    assert DF.prev_signal(d, s) == p                                  # strictly before the signal's own close
    dsp, prev = DF.days_since_prev(d, s, "2019-05-01")
    assert dsp == float((s - p).days) and prev == p.isoformat()      # the store's firing beats the older prior
    first = pd.Timestamp(fired[0]).date()
    assert DF.days_since_prev(d, first, "2019-05-01") == (float((first - date(2019, 5, 1)).days), "2019-05-01")
    dsp, prev = DF.days_since_prev(d, first, None)
    assert math.isnan(dsp) and prev is None                          # the name's first signal ever
    # the firings are the backtest's `strategies.dip.signals` construction (220 bars, the live rule)
    from src.backtest.strategies import dip as DIP
    S = DIP.signals({"SYN": d}, DIP.LIVE, d.index[0], d.index[-12], names=["SYN"])
    assert list(S.sig) == [t for t in fired if t <= d.index[-12]]


# ── the fit (model96's) ──────────────────────────────────────────────────────

def _table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2014, 2026):
        for k in range(75):
            sig = pd.Timestamp(f"{y}-01-03") + pd.Timedelta(days=int(rng.integers(0, 358)))
            x = rng.normal(size=len(F))
            filled = bool(rng.random() < 0.75)
            rows.append({"tk": f"N{k % 30}", "ticker": f"N{k % 30}", "sig": sig, "d0": sig + pd.Timedelta(days=1),
                         "source": "daily_cache" if y < 2021 else "deep_store", "filled": filled,
                         "exit_day": sig + pd.Timedelta(days=int(rng.integers(2, 16))) if filled else pd.NaT,
                         "net_ret_pct": (1.2 * x[0] - 0.8 * x[5] + rng.normal(0, 2.0)) if filled else np.nan,
                         **dict(zip(F, x))})
    return pd.DataFrame(rows)


def test_the_fit_is_model96s_construction(monkeypatch):
    tab = _table()
    cut_day = date(2025, 1, 1)
    trained = []
    real = DF._train

    def spy_train(X, y):
        trained.append(sorted(set(tab.loc[X.index, "sig"].dt.year)))
        return real(X, y)
    monkeypatch.setattr(DF, "_train", spy_train)
    b, cut, tr, oof = DF.fit(tab, cut_day)
    want = tab[tab.filled & (pd.to_datetime(tab.exit_day) < pd.Timestamp(cut_day))]
    assert sorted(tr.index) == sorted(want.index)                     # filled, exit before the cut
    years = sorted(set(tr.sig.dt.year))
    folds = [set(years[i::5]) for i in range(5)]
    assert trained[0] == years                                        # the model: every training year
    assert [set(years) - set(t) for t in trained[1:]] == folds        # the five out-of-fold models
    assert cut == float(np.nanquantile(oof, 1 / 3)) == DF.policy_cut(oof)
    assert DF.keep(cut, cut) and not DF.keep(np.nextafter(cut, -np.inf), cut)   # kept: score >= cut
    assert DF.PARAMS == {"num_leaves": 31, "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                         "bagging_freq": 1, "min_data_in_leaf": 50, "verbose": -1, "seed": 0, "num_threads": 2,
                         "objective": "regression"} and DF.ROUNDS == 300


def test_the_saved_model_is_the_fit_and_its_checksum_is_checked(tmp_path):
    tab = _table(seed=1)
    tp = tmp_path / "signals.pkl"
    tab.to_pickle(tp)
    out = tmp_path / "filter" / "model.txt"
    meta = DF.write_model(tp, date(2025, 1, 1), out)
    assert meta["reload_check"]["max_abs_diff"] == 0.0 and meta["features"] == F
    assert meta["prior_signals"] == {t: str(s.date()) for t, s in
                                     tab[tab.source == "daily_cache"].groupby("ticker").sig.max().items()}
    b, m = DF.load_model(out)
    b0, cut, tr, _ = DF.fit(tab, date(2025, 1, 1))
    np.testing.assert_array_equal(b.predict(DF.matrix(tr[F].to_dict("records"))), b0.predict(tr[F]))
    assert m["policy"]["cut"] == cut
    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")       # the file no longer the fit's
    with pytest.raises(ValueError, match="sha256"):
        DF.load_model(out)


# ── the day's signals ────────────────────────────────────────────────────────

class _Booster:
    """A stand-in model scoring each name by its own RSI2 input (the store's, through D-1): ``scores[rsi2]``."""

    def __init__(self, scores=None):
        self.scores = scores or {}

    def predict(self, X):
        return np.array([self.scores[r[F.index("rsi2")]] for r in np.asarray(X, float)])


SPY_IN = {"spy_dist_sma200_pct": 4.0, "spy_ret5_pct": -1.0, "spy_ret20_pct": 2.0, "spy_rvol20_pct": 0.9}


@pytest.fixture
def scored(monkeypatch):
    """The filter wired to the stand-in model (cut 7; AAA scores 9, BBB 5, CCC 8), SPY's inputs, the store's series."""
    series = {tk: _daily(ss.sessions_before(DAY, 300), seed=s) for tk, s in (("AAA", 1), ("BBB", 2), ("CCC", 3))}
    by_rsi = {DF.name_features(series[tk], PREV)["rsi2"]: v for tk, v in (("AAA", 9.0), ("BBB", 5.0), ("CCC", 8.0))}
    meta = {"policy": {"cut": 7.0}, "fit": {"cut_date": "2026-10-08"}, "checksums": {"model_sha256": "f" * 64},
            "prior_signals": {"OLD": "2019-05-01"}}
    monkeypatch.setattr(DF, "load_model", lambda path=None: (_Booster(by_rsi), meta))
    monkeypatch.setattr(DF, "spy_inputs", lambda through: (dict(SPY_IN), "cache"))
    monkeypatch.setattr(dl, "_daily", lambda tk: series.get(tk))
    return series


def _sig(tk, series):
    st = dl.stats_at(series[tk], PREV)
    return dict({"ticker": tk, **st}, limit=dl.limit_price(st["close"], st["atr14"]))


def test_apply_scores_every_signal_and_drops_those_under_the_cut(scored):
    sig = [_sig("AAA", scored), _sig("BBB", scored), _sig("CCC", scored)]
    kept, block = DF.apply(sig, DAY, PREV)
    assert [s["ticker"] for s in kept] == ["AAA", "CCC"]              # the order (deepest dip first) kept
    assert [s["ticker"] for s in block["filtered"]] == ["BBB"]
    info = block["info"]
    assert (info["breadth"], info["scored"], info["kept"], info["filtered"], info["cut"], info["spy_source"],
            info["error"]) == (3, 3, 2, 1, 7.0, "cache", None)
    bbb = block["filtered"][0]
    assert bbb["filter_score"] == pytest.approx(5.0) and bbb["filter_keep"] is False
    assert set(bbb["filter_inputs"]) == set(F) and bbb["filter_inputs"]["breadth"] == 3
    assert bbb["filter_inputs"]["spy_ret5_pct"] == -1.0
    want = DF.name_features(scored["BBB"], PREV)                      # the name's own inputs: its series through D-1
    assert all(bbb["filter_inputs"][k] == want[k] for k in DF.NAME_FEATURES)
    assert all(s["filter_keep"] is True and s["filter_score"] >= 7.0 for s in kept)


def test_apply_is_fail_open(monkeypatch, scored):
    sig = [_sig("AAA", scored), _sig("BBB", scored)]
    monkeypatch.setattr(DF, "spy_inputs", lambda through: (None, "missing"))
    kept, block = DF.apply([dict(s) for s in sig], DAY, PREV)
    assert [s["ticker"] for s in kept] == ["AAA", "BBB"] and "SPY" in block["info"]["error"]

    def boom(path=None):
        raise FileNotFoundError("model.txt")
    monkeypatch.setattr(DF, "load_model", boom)
    kept, block = DF.apply([dict(s) for s in sig], DAY, PREV)
    assert len(kept) == 2 and block["filtered"] == [] and "model unavailable" in block["info"]["error"]
    monkeypatch.setattr(dl, "_daily", lambda tk: None)                # one signal's inputs raise: kept, unscored
    monkeypatch.setattr(DF, "load_model", lambda path=None: (_Booster(), {"policy": {"cut": 7.0}}))
    monkeypatch.setattr(DF, "spy_inputs", lambda through: (dict(SPY_IN), "cache"))
    kept, block = DF.apply([dict(s) for s in sig], DAY, PREV)
    assert len(kept) == 2 and all("filter_error" in s for s in kept) and block["info"]["scored"] == 0


# ── compute_signals: off = unchanged, on = scored ────────────────────────────

def _bars_of(daily):
    """Flat 30-minute regular-hours bars reproducing ``daily`` exactly (13 bars a session: the first opens at the
    session's open, the extremes in the middle, the last closes at its close, the volume split evenly)."""
    idx, rows = [], []
    for t, r in daily.iterrows():
        d = t.date()
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            o = r.o if k == 0 else r.c
            rows.append((o, r.h if k == 6 else max(o, r.c), r.l if k == 6 else min(o, r.c), r.c, r.v / 13))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))


@pytest.fixture
def store(monkeypatch, tmp_path):
    from src.data import intraday_store
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    days = ss.sessions_before(DAY, 300)
    frames = {tk: _bars_of(_daily(days, seed=s)) for tk, s in (("AAA", 1), ("BBB", 2), ("CCC", 3))}
    monkeypatch.setattr(dl, "candidates", lambda d: (sorted(frames), "test"))
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: frames.get(tk))
    return frames


OLD_KEYS = {"day", "signal_session", "source", "considered", "evaluated", "eligible", "stale", "inactive", "rule",
            "signals", "next", "computed_at", "seconds"}
OLD_SIGNAL_KEYS = {"ticker", "session", "close", "sma_trend", "sma_exit", "rsi2", "dv20", "bars", "atr14", "limit"}


def test_with_the_switch_off_the_signal_file_is_unchanged(monkeypatch, store):
    def never(*a, **k):
        raise AssertionError("the filter ran with its switch off")
    monkeypatch.setattr(DF, "apply", never)
    monkeypatch.setattr(DF, "load_model", never)
    assert settings.enable_dip_long_filter is False                   # the default (conftest pins the baseline)
    out = dl.compute_signals(DAY)
    assert set(out) == OLD_KEYS and len(out["signals"]) == 3
    assert all(set(s) == OLD_SIGNAL_KEYS for s in out["signals"])
    assert json.loads(dl.signals_path(DAY).read_text(encoding="utf-8")).keys() == OLD_KEYS


def test_with_the_switch_on_the_dropped_signals_leave_the_list(monkeypatch, store):
    monkeypatch.setattr(settings, "enable_dip_long_filter", True)
    monkeypatch.setattr(DF, "spy_inputs", lambda through: (dict(SPY_IN), "cache"))
    rsis = {}

    class ByName:
        def predict(self, X):                                        # the stand-in: AAA under the cut, the rest over
            return np.array([0.0 if r[F.index("rsi2")] == rsis["AAA"] else 9.0 for r in np.asarray(X, float)])
    monkeypatch.setattr(DF, "load_model", lambda path=None: (ByName(), {"policy": {"cut": 7.0}, "prior_signals": {}}))
    full = {tk: dl.stats_at(dl.daily_bars(f), PREV) for tk, f in store.items()}
    rsis.update({tk: st["rsi2"] for tk, st in full.items()})
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == sorted(["BBB", "CCC"], key=lambda t: (full[t]["rsi2"], t))
    assert [s["ticker"] for s in out["filtered"]] == ["AAA"]
    assert out["filter"]["breadth"] == 3 and out["filter"]["filtered"] == 1 and out["filter"]["error"] is None
    on_disk = json.loads(dl.signals_path(DAY).read_text(encoding="utf-8"))
    assert on_disk["filtered"][0]["filter_score"] == 0.0 and on_disk["filtered"][0]["filter_keep"] is False
    assert all(s["filter_keep"] is True for s in on_disk["signals"])


def test_a_filter_crash_keeps_every_signal(monkeypatch, store):
    monkeypatch.setattr(settings, "enable_dip_long_filter", True)

    def crash(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(DF, "apply", crash)
    out = dl.compute_signals(DAY)
    assert len(out["signals"]) == 3 and out["filtered"] == [] and "boom" in out["filter"]["error"]


# ── the orders: a dropped signal is journaled, never placed ──────────────────

def at(d, h, m, s=20):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).astimezone(timezone.utc)


def _bars(d, rows):
    idx = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k)).astimezone(timezone.utc)
           .replace(tzinfo=None) for k in range(len(rows))]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx)).assign(Volume=1e6)


def _lsig(tk, rsi, close, atr, score):
    return {"ticker": tk, "session": PREV.isoformat(), "close": close, "sma_trend": close * 0.9,
            "sma_exit": close * 1.04, "rsi2": rsi, "dv20": 2e9, "bars": 1400, "atr14": atr,
            "limit": dl.limit_price(close, atr), "filter_score": score, "filter_keep": score >= 0.25}


@pytest.fixture
def limit_day(monkeypatch, tmp_path):
    for k, v in {"enable_dip_long": True, "dip_long_dir": str(tmp_path / "dip_long"), "dip_long_entry_mode": "limit",
                 "dip_long_limit_atr": 0.5, "dip_long_limit_sessions": 5, "dip_long_account_slices": 10,
                 "enable_sel_short": True, "enable_sel_short_vol": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_traded_arms": "vol,etf,fall", "sel_short_account_initial": 10_000.0,
                 "sel_short_account_slices": 24, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86, "enable_dip_long_filter": True}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    day = {"day": DAY.isoformat(), "signal_session": PREV.isoformat(),
           "signals": [_lsig("AAA", 4.0, 100.0, 4.0, 1.7)],                         # kept: limit 98.00
           "filtered": [_lsig("BBB", 3.0, 50.0, 2.0, -0.4)],                        # dropped: limit 49.00
           "filter": {"model": "PREREG96 RP33", "cut": 0.25, "breadth": 2, "kept": 1, "filtered": 1}}
    monkeypatch.setattr(dl, "signals_for", lambda d: day if d == DAY else {"signals": []})
    feed = {}
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: feed.get((tk, d), _bars(d, [])))
    return feed


def test_a_dropped_signal_is_journaled_filtered_and_never_placed_or_filled(limit_day):
    feed = limit_day
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    feed[("BBB", DAY)] = _bars(DAY, [(48.0, 48.5, 47.0, 48.0)])     # BBB opens UNDER its limit: it would fill
    assert tracker.record_dip_long_trades(now=at(DAY, 9, 30)) == 1    # AAA placed; BBB neither placed nor bought
    o = {x["ticker"]: x for x in dl.load_orders(DAY)}
    assert o["AAA"]["status"] == "working" and o["AAA"]["filter_score"] == 1.7
    assert o["BBB"]["status"] == "filtered" and o["BBB"]["filter_score"] == -0.4 and o["BBB"]["filter_cut"] == 0.25
    assert o["BBB"]["qty"] is None and [e["what"] for e in o["BBB"]["events"]] == ["filtered"]
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 97.5, 98.2)])   # a bar reaches AAA's limit (and BBB's)
    feed[("BBB", DAY)] = _bars(DAY, [(48.0, 48.5, 47.0, 48.0)])
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 1
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "dip_long"}
    assert set(t) == {"AAA"} and t["AAA"]["dip_filter_score"] == 1.7
    assert {x["ticker"]: x for x in dl.load_orders(DAY)}["BBB"]["status"] == "filtered"
    assert tracker.record_dip_long_trades(now=at(DAY, 10, 0)) == 0    # the next tick: nothing journaled twice
    assert [x["ticker"] for x in dl.load_orders(DAY)] == ["AAA", "BBB"]
    tracker.monitor_dip_long_orders(now=at(date(2026, 10, 14), 16, 1))    # the window's end: nothing to lapse
    assert {x["ticker"]: x for x in dl.load_orders(DAY)}["BBB"]["status"] == "filtered"


def test_the_broker_never_rests_a_filtered_order(limit_day, monkeypatch):
    feed = limit_day
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    feed[("BBB", DAY)] = _bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=at(DAY, 9, 30))
    import src.broker.reconcile as rec
    from tests.test_broker_reconcile import FakeBroker
    store = {"trades": tracker._load_trades()}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    monkeypatch.setattr(dl, "today_et", lambda now=None: DAY)

    class Resting(FakeBroker):
        def submit_order(self, req):
            res = super().submit_order(req)
            if getattr(req, "rest", False):
                res.filled_qty, res.status, res.avg_fill_price = 0, "Submitted", None
            return res
    b = Resting()
    rec.sync(broker=b)
    assert [(o.ticker, o.side) for o in b.orders] == [("AAA", "BUY")]


def test_with_the_switch_off_the_orders_carry_no_filter_field(limit_day, monkeypatch):
    """A day computed with the switch off has neither scores nor a ``filtered`` list: the orders are as before."""
    day = {"day": DAY.isoformat(), "signal_session": PREV.isoformat(),
           "signals": [{k: v for k, v in _lsig("AAA", 4.0, 100.0, 4.0, 1.7).items() if not k.startswith("filter")}]}
    monkeypatch.setattr(settings, "enable_dip_long_filter", False)
    monkeypatch.setattr(dl, "signals_for", lambda d: day if d == DAY else {"signals": []})
    limit_day[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    tracker.record_dip_long_trades(now=at(DAY, 9, 30))
    o = dl.load_orders(DAY)
    assert [x["ticker"] for x in o] == ["AAA"] and not any(k.startswith("filter") for k in o[0])


# ── health ───────────────────────────────────────────────────────────────────

def test_the_banner_reports_the_filter_only_when_it_is_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    noon = at(DAY, 12, 0)
    base = {"signals": [{"ticker": "AAA"}], "stale": [], "signal_session": PREV.isoformat()}
    dl._write_json(dl.signals_path(DAY), dict(base, filter={"error": "model unavailable (x) — every signal kept"}))
    assert dl.health(noon)["problems"] == []                          # off: the filter's line never shows
    monkeypatch.setattr(settings, "enable_dip_long_filter", True)
    assert any("did not score" in p for p in dl.health(noon)["problems"])
    dl._write_json(dl.signals_path(DAY), dict(base, filter={"error": None, "kept": 1, "breadth": 2, "filtered": 1,
                                                          "cut": 0.25, "spy_source": "deep_store"}))
    h = dl.health(noon)
    assert h["problems"] == [] and any("kept 1 of 2" in n and "deep store" in n for n in h["notes"])
    dl._write_json(dl.signals_path(DAY), base)                        # computed before the switch went on
    assert any("without it" in p for p in dl.health(noon)["problems"])


# ── the simulator's option (strategies/dip.py) ───────────────────────────────

class _Panel(dict):
    bars_fn = None


def _key(P):
    return [(p["oid"], p["ord_k"], p["ens"], p["xns"], p["e"], p["x"]) for p in P]


def test_the_simulator_drops_a_filtered_order_whole(tmp_path):
    from src.backtest.strategies import dip as DIP
    days = ss.sessions_before(DAY, 300)
    after = {tk: _daily(days, seed=s, drop_last=0, dips=(250, 270)) for tk, s in (("AAA", 1), ("BBB", 2))}
    panel = _Panel({tk: dl.daily_bars(_bars_of(after[tk])) for tk in after})
    panel.bars_fn = lambda tk: _bars_of(after[tk])
    lo, hi = days[230], days[-2]
    P = DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA", "BBB"])
    oids = sorted({p["oid"] for p in P})
    assert len(oids) >= 2
    keep = {oids[0]: False, oids[1]: True}
    P1 = DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA", "BBB"], keep=keep)
    assert oids[0] not in {p["oid"] for p in P1}                      # no piece, not even a placeholder
    assert _key(P1) == _key([p for p in P if p["oid"] != oids[0]])    # every other order as before
    assert _key(DIP.filter_pieces(P, keep)) == _key(P1)               # pieces built once, filtered after: the same
    assert _key(DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel, names=["AAA", "BBB"], keep=None)) == _key(P)
    W = pd.DataFrame({"oid": oids[:2], "keep": [False, True], "keep_research": [None, True]})
    p = tmp_path / "wf.pkl"
    W.to_pickle(p)
    assert DIP.filter_keep(p) == {oids[0]: False, oids[1]: True}
    assert DIP.filter_keep(p, research=True) == {oids[1]: True}       # the study's draws: filled rows only
