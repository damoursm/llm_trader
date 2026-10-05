"""The V2 model arm's extra inputs (`src/signals/sel_v2.py`) and its scoring path in `sel_short`
(user directive 2026-10-02: "Deploy to live production the V2 model arm"). The no-look-ahead guards of
these inputs live in `tests/test_no_lookahead.py`."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.signals import sel_short as ss
from src.signals import sel_v2

ET = ZoneInfo("America/New_York")
DAY = date(2026, 9, 28)


def _frame(days, seed=3):
    rng = np.random.default_rng(seed)
    idx = [pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
           .tz_convert("UTC").tz_localize(None) for d in days for k in range(13)]
    c = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx))))
    v = rng.integers(20_000, 200_000, len(idx)).astype(float)
    return pd.DataFrame({"Open": c, "High": c * 1.004, "Low": c * 0.996, "Close": c, "Volume": v},
                        index=pd.DatetimeIndex(idx))


def _hlc(df):
    from src.analysis.ml_dataset import hlc_from_frames
    return hlc_from_frames([df])


# ── per bar ──────────────────────────────────────────────────────────────────

def test_runup_and_relative_volume_are_the_scorers_own_quantities():
    """sx_runup5 IS the riser rule's run-up (`_close_at` 5 sessions back) and sx_rvol260 IS
    `rvol_at` — the same quantities the live rule decides on."""
    days = ss.sessions_before(DAY, 30)
    hlc = _hlc(_frame(days))
    E = sel_v2.bar_extras(hlc, np.full(len(hlc[0]), np.nan))
    for t in (200, 250, 300, 389):
        bar_ts = pd.Timestamp(hlc[0][t])
        d = bar_ts.tz_localize("UTC").tz_convert(ET).date()
        bar_of_day = int((bar_ts.tz_localize("UTC").tz_convert(ET) - datetime(d.year, d.month, d.day, 9, 30,
                                                                               tzinfo=ET)).total_seconds() // 1800)
        pre = ss._close_at(hlc[0], hlc[3], ss.sessions_before(d, 5)[0], bar_of_day)
        assert E[t, 0] == pytest.approx((float(hlc[3].iloc[t]) / pre - 1.0) * 100.0, rel=1e-12)
        assert E[t, 2] == pytest.approx(ss.rvol_at(hlc[0], hlc[4], bar_ts), rel=1e-9)


def test_own_atr_reads_the_previous_twenty_sessions_and_needs_ten():
    days = ss.sessions_before(DAY, 30)
    hlc = _hlc(_frame(days))
    n = len(hlc[0])
    atr = np.full(n, 2.0)
    sd = np.repeat(np.arange(30), 13)
    atr[sd == 25] = 4.0                                  # one hot session inside the window of session 29
    atr[(sd == 29)] = 3.0                                # the current session: never its own reference
    E = sel_v2.bar_extras(hlc, atr)
    assert np.isnan(E[sd == 9, 1]).all()                 # 9 prior sessions: under the 10 needed
    assert np.isfinite(E[sd == 10, 1]).all()
    assert E[sd == 29, 1] == pytest.approx(np.full((sd == 29).sum(), 3.0 / 4.0 - 1.0))
    assert E[sd == 26, 1] == pytest.approx(np.full((sd == 26).sum(), 2.0 / 4.0 - 1.0))


# ── per session ──────────────────────────────────────────────────────────────

def _cut_utc(d):
    return pd.Timestamp(datetime(d.year, d.month, d.day, 8, 30, tzinfo=ET)).tz_convert("UTC").tz_localize(None)


def test_session_inputs_count_rows_known_by_the_cutoff():
    cut = _cut_utc(DAY)
    sec = pd.DataFrame({
        "accession": ["a", "b", "c", "d", "e"],
        "form": ["8-K", "NT 10-Q", "8-K", "424B4", "S-1"],
        "acceptance": [cut - pd.Timedelta(days=3), cut - pd.Timedelta(hours=1), cut - pd.Timedelta(days=800),
                       cut - pd.Timedelta(days=10), cut - pd.Timedelta(days=400)],
        "filing_date": ["2026-09-25", "2026-09-28", "2024-07-20", "2026-09-18", "2025-08-24"],
        "items": ["3.01,9.01", "", "1.03", "", ""]})
    splits = pd.DataFrame({"ticker": ["SYN", "SYN"], "execution_date": ["2026-08-19", "2026-05-01"],
                           "split_from": [10.0, 1.0], "split_to": [1.0, 2.0]})       # one reverse, one forward
    s = dict(zip(sel_v2.E_SESS, sel_v2.session_extras("SYN", [ss.dnum(DAY)], splits=splits, sec=sec)[0]))
    assert s["dp_sec_d_def"] == pytest.approx(3.0)
    assert s["dp_sec_d_nt"] == pytest.approx(1.0 / 24.0)
    assert np.isnan(s["dp_sec_d_bk"])                    # 800 days: past the 730-day cap
    assert np.isnan(s["dp_sec_d_restate"]) and np.isnan(s["dp_sec_d_shell"])
    assert s["dp_sec_n_off_365d"] == 1.0                 # the 424B4; the S-1 is 400 days old
    assert s["dp_rsplit_n_730d"] == 1.0                  # the forward split does not count
    assert s["dp_rsplit_d_last"] == pytest.approx(40.0 + 8.5 / 24.0)


def test_session_inputs_without_rows():
    s = sel_v2.session_extras("SYN", [ss.dnum(DAY)], splits=None, sec=pd.DataFrame())[0]
    assert s[0] == 0.0 and np.isnan(s[1:]).all()         # no split: zero of them; no filing part: unknown


# ── across the bar ───────────────────────────────────────────────────────────

def _xs_frame(n=12, run=0, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"run": run, "tradeable": True}, index=range(n))
    for s in sel_v2.XS_SOURCES:
        df[s] = rng.normal(size=n)
    df["sic2"] = [28] * 6 + [73] * 4 + [-1] * 2
    return df


def test_cross_section_ranks_within_the_bar_and_sectors_of_five():
    df = _xs_frame()
    df.loc[11, "tradeable"] = False
    out = sel_v2.cross_section(df)
    t = df[df.tradeable]
    exp = 2.0 * t["ret_5"].astype(np.float32).astype(float).rank(pct=True) - 1.0
    assert out.loc[t.index, "xrank_ret5"].to_numpy() == pytest.approx(exp.to_numpy().astype(np.float32))
    assert out.loc[11].isna().all()                      # a non-tradeable row: never ranked
    g = t[t.sic2 == 28]                                  # 6 names: a sector
    med = np.median(g["sx_runup5"].astype(np.float32).astype(float))
    assert out.loc[g.index, "sx_runup5_secrel"].to_numpy() == pytest.approx(
        (g["sx_runup5"].astype(np.float32).astype(float) - med).to_numpy(), abs=1e-5)
    assert out.loc[t.index[t.sic2 == 73], "sx_atr_secrel"].isna().all()    # 4 names: under 5
    assert out.loc[[10], "sx_atr_secrel"].isna().all()                     # no SIC code


def test_cross_section_runs_are_ranked_apart():
    a, b = _xs_frame(run=0, seed=1), _xs_frame(run=1, seed=2)
    both = sel_v2.cross_section(pd.concat([a, b], ignore_index=True))
    alone = sel_v2.cross_section(a)
    assert both.iloc[:len(a)].to_numpy() == pytest.approx(alone.to_numpy(), nan_ok=True)


# ── the scoring path ─────────────────────────────────────────────────────────

class _Booster:
    def __init__(self):
        self.M = None

    def predict(self, M, num_iteration=None):
        self.M = np.array(M)
        return np.arange(len(M), dtype=float) + 0.5


def test_score_v2_fills_the_cross_section_blanks_the_mask_and_scores(monkeypatch):
    feats = ["ret_5", "xrank_ret5", "dp_an_n_30d", "sx_runup5", "sx_atr_xrank"]
    b = _Booster()
    monkeypatch.setattr(ss, "load_model", lambda: (b, {"features": feats, "extra_features": sel_v2.EXTRA,
                                                       "masked": ["dp_an_n_30d"], "num_iteration": 400}))
    monkeypatch.setattr(sel_v2, "sic2_map", lambda refresh=False: {})
    rows = []
    for i, (tk, px) in enumerate((("AAA", 10.0), ("BBB", 20.0), ("CCC", 3.0))):
        vec = np.array([0.1 * (i + 1), np.nan, 7.0, 5.0 * (i + 1), np.nan])
        rec = {"ticker": tk, "px": px, "dv20": 9e6, "status": "OK", "score": np.nan, "_vec": vec}
        for s in sel_v2.XS_SOURCES:
            rec[f"_x_{s}"] = {"ret_5": 0.1 * (i + 1), "sx_runup5": 5.0 * (i + 1)}.get(s, float(i))
        rows.append(rec)
    rows.append({"ticker": "DDD", "px": 9.0, "dv20": 9e6, "status": "NO_BAR", "score": np.nan})
    out = ss.score_v2(pd.DataFrame(rows))
    assert not [c for c in out.columns if str(c).startswith("_")]
    assert out.set_index("ticker").loc[["AAA", "BBB", "CCC"], "score"].tolist() == [0.5, 1.5, 2.5]
    assert np.isnan(out.set_index("ticker").loc["DDD", "score"])
    M = b.M
    assert np.isnan(M[:, 2]).all()                                   # the masked input
    assert M[:2, 1].tolist() == pytest.approx([-1.0 + 2 * 0.5, 1.0])   # AAA, BBB ranked; CCC is under $5
    assert np.isnan(M[2, 1]) and np.isnan(M[2, 4])
    assert M[:, 3].tolist() == pytest.approx([5.0, 10.0, 15.0])         # the worker's own values pass through


def test_a_v1_model_passes_score_v2_untouched():
    df = pd.DataFrame({"ticker": ["AAA"], "score": [0.3], "status": ["OK"]})
    assert ss.score_v2(df) is df


def test_install_v2_keeps_v1_and_its_history(tmp_path, monkeypatch):
    import lightgbm as lgb
    monkeypatch.setattr(ss.settings, "sel_short_dir", str(tmp_path))
    v1 = ["f1", "f2"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2 + len(sel_v2.EXTRA)))
    y = rng.normal(size=200)
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 4}, lgb.Dataset(
        X, y, feature_name=v1 + sel_v2.EXTRA), num_boost_round=3)
    bp = tmp_path / "v2.txt"
    b.save_model(str(bp))
    (tmp_path / "model.txt").write_text("v1 booster", encoding="utf-8")
    (tmp_path / "model.json").write_text(json.dumps({"features": v1, "tickers": ["AAA"], "cut": "2026-04-30"}),
                                         encoding="utf-8")
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "2026-09-30.pkl").write_bytes(b"x")
    meta = ss.install_v2(bp)
    assert meta["features"] == v1 + sel_v2.EXTRA and meta["masked"] == list(sel_v2.MASKED)
    assert sel_v2.is_v2(json.loads((tmp_path / "model.json").read_text()))
    assert (tmp_path / "model_v1.txt").read_text() == "v1 booster"
    assert json.loads((tmp_path / "model_v1.json").read_text())["features"] == v1
    assert (tmp_path / "scores_v1" / "2026-09-30.pkl").exists() and not (tmp_path / "scores").exists()
    with pytest.raises(SystemExit):
        ss.install_v2(bp)                                            # already V2


def test_install_v2_refuses_a_booster_that_is_not_v1_plus_the_extras(tmp_path, monkeypatch):
    import lightgbm as lgb
    monkeypatch.setattr(ss.settings, "sel_short_dir", str(tmp_path))
    rng = np.random.default_rng(0)
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 4},
                  lgb.Dataset(rng.normal(size=(100, 3)), rng.normal(size=100), feature_name=["f2", "f1", "x"]),
                  num_boost_round=2)
    b.save_model(str(tmp_path / "bad.txt"))
    (tmp_path / "model.txt").write_text("v1", encoding="utf-8")
    (tmp_path / "model.json").write_text(json.dumps({"features": ["f1", "f2"], "tickers": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        ss.install_v2(tmp_path / "bad.txt")
    assert (tmp_path / "model.txt").read_text() == "v1"


def test_day_extras_are_computed_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.settings, "sel_short_dir", str(tmp_path))
    calls = []

    def fake(names, dn, workers=8):
        calls.append(list(names))
        return {t: [1.0] * len(sel_v2.E_SESS) for t in names}
    monkeypatch.setattr(sel_v2, "day_extras", fake)
    assert ss.ensure_day_extras(DAY, ["AAA", "BBB"]) == 2
    assert ss.ensure_day_extras(DAY, ["AAA", "BBB"]) == 0
    assert ss.ensure_day_extras(DAY, ["AAA", "CCC"]) == 1 and calls[-1] == ["CCC"]
    assert set(sel_v2.read_day_extras(ss.day_extras_path(DAY))) == {"AAA", "BBB", "CCC"}
