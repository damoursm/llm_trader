"""The borrow-pressure model's live inputs (`src/signals/sel_pressure.py`, PREREG71's V6).

What must hold: Polygon's lowercase class symbols never reach the panel; the daily features are PREREG68's (20-session
dollar volume, a 14-session ATR% over the previous close, log run-ups, relative volume, the split masks); each input
is read as of the session's open — FINRA's report settled 10+ days before, fails-to-deliver settled 15-45 days
before, IBKR's fee of the session before (its API bar, else our archive), the Reg SHO lists of the 60 sessions
before, the reverse split and the listing date on or before the day; the daily-ATR rank is over the tradeable
stocks; a name scores HIGH at or above the model's cut, and the day's scores are built once and read back.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import sel_pressure as SP

EPOCH = pd.Timestamp("1970-01-01")


def _sessions(n, end=date(2026, 9, 30)):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _bars(sessions, names=("AAA", "BBB", "CCC"), seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for k, nm in enumerate(names):
        px = 20.0 + 5 * k
        for i, s in enumerate(sessions):
            px *= float(np.exp(rng.normal(0.0, 0.03)))
            rows.append({"sym": nm, "o": px * 0.99, "h": px * 1.04, "l": px * 0.95, "c": px,
                         "v": 1e6 * (1 + k) * float(rng.uniform(0.5, 1.5)), "s": s})
    return pd.DataFrame(rows)


def _seed(tmp_path, monkeypatch, bars):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel_short"))
    g = SP.root() / "grouped"
    g.mkdir(parents=True, exist_ok=True)
    for s, x in bars.groupby("s"):
        x.drop(columns="s").reset_index(drop=True).to_pickle(g / f"{s.isoformat()}.pkl")


def _frames(d, **over):
    D = SP.dnum(d)
    f = {"si": pd.DataFrame({"ticker": ["AAA", "AAA", "AAA"],
                             "settlement_date": [EPOCH + pd.Timedelta(days=D - 40), EPOCH + pd.Timedelta(days=D - 20),
                                                 EPOCH + pd.Timedelta(days=D - 5)],
                             "short_interest": [1e5, 2e5, 9e9], "days_to_cover": [1.0, 2.0, 99.0]}),
         "ftd": pd.DataFrame({"symbol": ["AAA", "AAA", "AAA"],
                              "sd": [EPOCH + pd.Timedelta(days=D - 50), EPOCH + pd.Timedelta(days=D - 30),
                                     EPOCH + pd.Timedelta(days=D - 10)], "q": [7e9, 3e5, 7e9]}),
         "regsho": pd.DataFrame({"symbol": [], "date": []}),
         "splits": pd.DataFrame({"ticker": ["BBB"], "d": [D - 100], "split_from": [10.0], "split_to": [1.0]}),
         "details": pd.DataFrame({"ticker": ["AAA"], "d": [EPOCH + pd.Timedelta(days=D - 365)]}),
         "fees": {}, "cover": {}, "types": {"CCC": "ETF"}}
    f.update(over)
    return f


def test_lowercase_class_symbols_never_reach_the_panel():
    assert SP._symbol("TpC") is None and SP._symbol("PLMr") is None
    assert SP._symbol("BRK.B") == "BRK-B" and SP._symbol("AAPL") == "AAPL"


def test_the_daily_features_are_prereg68s():
    sess = _sessions(30)
    B = _bars(sess, names=("AAA",)).assign(d=lambda x: x.s.map(SP.dnum)).drop(columns="s")
    F = SP.bar_features(B, pd.DataFrame(columns=["ticker", "d", "split_from", "split_to"]))
    last = F.iloc[-1]
    c, h, l, v = B.c.to_numpy(), B.h.to_numpy(), B.l.to_numpy(), B.v.to_numpy()
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    assert last.dv20 == pytest.approx(np.mean((c * v)[-20:]))
    assert last.atr14 == pytest.approx(np.mean(tr[-14:]) / c[-1] * 100)
    assert last.ru5 == pytest.approx(np.log(c[-1] / c[-6]))
    assert last.rvol == pytest.approx(v[-1] / np.mean(v[-21:-1]))
    # a split 10 days before masks the run-ups (16 days) and the ATR% (30 days)
    S = pd.DataFrame({"ticker": ["AAA"], "d": [int(B.d.iloc[-1]) - 10], "split_from": [1.0], "split_to": [2.0]})
    G = SP.bar_features(B, S).iloc[-1]
    assert np.isnan(G.ru5) and np.isnan(G.ru10) and np.isnan(G.atr14) and np.isfinite(G.dv20)


def test_each_input_is_read_as_of_the_open(tmp_path, monkeypatch):
    sess = _sessions(60)
    d = sess[-1] + timedelta(days=1)
    _seed(tmp_path, monkeypatch, _bars(sess))
    cal = [SP.dnum(s) for s in sess]
    fees = {"AAA": (np.array(cal[-25:], np.int64), np.r_[np.full(19, 10.0), 40.0, 30.0, 25.0, 20.0, 50.0, 60.0])}
    X = SP.inputs(d, sessions=sess, fetch=False, frames=_frames(d, fees=fees, cover={"AAA": (cal[0], cal[-1])}))
    a = X.set_index("ticker").loc["AAA"]
    assert a.log1p_dtc == pytest.approx(np.log1p(2.0))           # the report settled 20 days before, not 5
    assert a.si_change == pytest.approx(np.log(2.0))
    vol20 = a.dv20_p / a.c_p
    assert a.ftd_ratio == pytest.approx(np.log1p(3e5 / vol20))     # settled 30 days before only (15-45)
    assert a.api_fee_prev == pytest.approx(np.log1p(60.0))         # the fee of the session before
    assert a.api_fee_max20 == pytest.approx(np.log1p(60.0))
    assert a.api_fee_chg5 == pytest.approx(np.log1p(60.0) - np.log1p(40.0))
    assert (a.api_listed, a.api_unknown) == (1.0, 0.0)
    assert a.days_listed == pytest.approx(365.0)
    b = X.set_index("ticker").loc["BBB"]
    assert b.days_since_reverse_split == pytest.approx(100.0) and b.api_unknown == 1.0
    assert "CCC" in set(X.ticker)                                  # scored, though a fund
    # the ATR rank counts the tradeable stocks only (the fund CCC is not one of them)
    elig = X[X.ticker.isin(["AAA", "BBB"])]
    assert sorted(elig.rk.tolist()) == [1.0, 2.0]


def test_a_missing_fee_on_the_session_before_is_not_listed_when_covered(tmp_path, monkeypatch):
    sess = _sessions(60)
    d = sess[-1] + timedelta(days=1)
    _seed(tmp_path, monkeypatch, _bars(sess))
    cal = [SP.dnum(s) for s in sess]
    fees = {"AAA": (np.array(cal[-25:-2], np.int64), np.full(23, 10.0))}
    X = SP.inputs(d, sessions=sess, fetch=False, frames=_frames(d, fees=fees, cover={"AAA": (cal[0], cal[-1])}))
    a = X.set_index("ticker").loc["AAA"]
    assert (a.api_listed, a.api_not_listed) == (0.0, 1.0) and np.isnan(a.api_fee_prev)


def _model(tmp_path, monkeypatch, cut=0.5):
    import lightgbm as lgb
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, len(SP.INPUTS)))
    y = (X[:, 15] > 0).astype(float)                               # driven by api_fee_prev
    bst = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 20)
    SP.root().mkdir(parents=True, exist_ok=True)
    bst.save_model(str(SP.root() / "nf_model.txt"))
    (SP.root() / "nf_model.json").write_text(json.dumps({"inputs": SP.INPUTS, "hi_cut": cut}), encoding="utf-8")


def test_the_days_scores_are_built_once_and_read_back(tmp_path, monkeypatch):
    sess = _sessions(60)
    d = sess[-1] + timedelta(days=1)
    _seed(tmp_path, monkeypatch, _bars(sess))
    _model(tmp_path, monkeypatch)
    cal = [SP.dnum(s) for s in sess]
    fr = _frames(d, fees={"AAA": (np.array(cal[-25:], np.int64), np.full(25, 80.0))}, cover={"AAA": (cal[0], cal[-1])})
    orig = SP.inputs
    monkeypatch.setattr(SP, "inputs", lambda d_, fetch=True: orig(d_, sessions=sess, fetch=False, frames=fr))
    SP._CACHE.clear()
    out = SP.build(d, fetch=False)
    assert out["n"] == 3 and SP.scores_path(d).exists()
    SP._CACHE.clear()
    p, high = SP.pressure_of("AAA", d, build_missing=False)
    assert p == pytest.approx(out["scores"]["AAA"]) and high == (p >= 0.5)
    assert SP.pressure_of("ZZZ", d, build_missing=False) == (None, False)
    assert SP.scores(d + timedelta(days=7), build_missing=False) is None


def test_the_model_must_carry_the_modules_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel_short"))
    _model(tmp_path, monkeypatch)
    (SP.root() / "nf_model.json").write_text(json.dumps({"inputs": SP.INPUTS[::-1], "hi_cut": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError):
        SP.load_model()

