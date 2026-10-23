"""PREREG183's wiring of the Sharpe-ratio inference into the research tools — REPORTS, never decisions:
`bank.monthly_unit_pnl` / `bank.dsr_months` (a card's deflated Sharpe on its months), `bank.track_record` (how long
the live record needs) and `add_rule.market_noise` (a change against the market's noise, not the draws').

The property that motivated the months: a book that buys in same-day batches must not read as more evidence than
the episodes it has — duplicating every trade inside its month leaves the monthly reading unchanged while the
per-trade one (`audit.dsr_trades`, trades as independent observations) rises.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import stats as S
from src.backtest import add_rule, audit, bank, periods


def _trades(n=120, seed=0, months=None):
    rng = np.random.default_rng(seed)
    months = months or [str(p) for p in pd.period_range("2018-01", "2026-09", freq="M")]
    m = rng.choice(months, size=n)
    return pd.DataFrame({"book": "x", "month": m, "net": rng.normal(0.6, 3.0, size=n), "days": 3.0})


def test_monthly_unit_pnl_fills_every_month():
    t = pd.DataFrame({"month": ["2018-01", "2018-01", "2018-03"], "net": [1.0, 2.0, -0.5]})
    s = bank.monthly_unit_pnl(t, "2018-01-02", "2018-04-30")
    assert list(s.index) == ["2018-01", "2018-02", "2018-03", "2018-04"]
    assert s.to_list() == pytest.approx([0.03, 0.0, -0.005, 0.0], abs=1e-15)


def test_a_duplicated_batch_counts_once_on_months():
    t = _trades()
    twice = pd.concat([t, t], ignore_index=True)                     # every trade bought twice, the same month
    a, b = bank.dsr_months(t, (10, 50)), bank.dsr_months(twice, (10, 50))
    assert a["T"] == b["T"] == 105
    assert b["psr0"] == pytest.approx(a["psr0"], abs=1e-12) and b["low"]["dsr"] == pytest.approx(a["low"]["dsr"], abs=1e-12)
    per_a = audit.dsr_trades(t.net.to_numpy(float), 10)["psr0"]
    per_b = audit.dsr_trades(twice.net.to_numpy(float), 10)["psr0"]
    assert per_b > per_a                                             # the per-trade reading counts the copies
    assert 0.0 <= a["high"]["dsr"] <= a["low"]["dsr"] <= a["psr0"] <= 1.0


def test_card_carries_both_deflated_sharpes():
    t = _trades(n=150, seed=3)
    t["entry_day"] = pd.to_datetime(t.month + "-15")
    t["year"] = t.entry_day.dt.year
    t["set"] = "A"
    t["gross"] = np.sign(t.net)
    c = bank.card(t, "x", n_trials=(10, 50), bench=pd.Series(dtype=float))
    assert "dsr" in c and c["dsr_months"]["T"] == 105 and c["dsr_months"]["low"]["N"] == 10


def test_track_record_matches_the_formulas():
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.normal(0.04, 0.08, size=(105, 6)))
    tr = bank.track_record(M)
    per = [S.sharpe_moments(M[c].to_numpy(float)) for c in M.columns]
    sr = float(np.mean([p["sr"] for p in per]))
    assert tr["sr_month"] == pytest.approx(sr, abs=1e-12)
    kw = {"skew": tr["skew"], "kurt": tr["kurt"], "rho": tr["rho"]}
    assert tr["profit"]["backtest"]["min_trl"] == pytest.approx(S.min_track_record(sr, 0.0, **kw), rel=1e-12)
    assert tr["shortfall"]["months_for_power"] == pytest.approx(
        S.track_record_for_power(sr, 0.5 * sr, 0.8, **kw), rel=1e-12)
    pw = [tr["profit"]["backtest"]["power_at"][h] for h in ("3", "6", "12", "24", "36")]
    assert pw == sorted(pw) and tr["profit"]["decayed"]["months_for_power"] > tr["profit"]["backtest"]["months_for_power"]


def _runs(extra_per_day=0.0, seed=0, nav=True):
    out = {}
    for s in ("A0", "A", "B"):
        p = periods.get(s)
        days = pd.bdate_range(p.lo, p.hi)
        out[s] = {}
        for k in range(4):
            rng = np.random.default_rng(1000 * k + {"A0": 0, "A": 1, "B": 2}[s] + seed)
            r = rng.normal(0.001, 0.01, size=len(days))
            base = np.cumprod(1.0 + r)
            noise = np.random.default_rng(7 + 10 * k + {"A0": 0, "A": 1, "B": 2}[s]).normal(0.0, 0.002, size=len(days))
            noise -= noise.mean()                                     # the change's own drift is exactly extra_per_day
            v = base * np.exp(np.arange(len(days)) * extra_per_day + np.cumsum(noise) * (extra_per_day != 0.0))
            rec = {"growth": float(v[-1] / v[0] - 1.0), "maxdd": 10.0, "calls": []}
            if nav:
                rec["nav"] = {"days": [d.strftime("%Y-%m-%d") for d in days], "values": (10_000.0 * v).tolist()}
            out[s][k] = rec
    return out


def test_market_noise_reads_a_known_gain():
    live, variant = _runs(), _runs(extra_per_day=0.0002)
    mn = add_rule.market_noise(live, variant)
    assert mn["seeds"] == 4 and mn["months"] >= 100
    assert mn["gain"] == pytest.approx(0.0002 * 100 * 252, rel=0.08)          # ~5 log points a year
    assert mn["t"] > 3 and mn["p"] < 0.01 and mn["addrule_false_pass"] < 0.5
    assert set(mn["ofdr"]) == {"0.1", "0.25", "0.5"} and all(0.0 <= v <= 1.0 for v in mn["ofdr"].values())
    assert "error" in add_rule.market_noise(_runs(nav=False), _runs(extra_per_day=0.0002, nav=False))


def test_pooled_verdict_market_flag_is_report_only():
    live, variant = _runs(), _runs(extra_per_day=0.0002)
    for st in live:                                                 # 20 draws a set: the rule's minimum
        for k in range(4, 20):
            live[st][k], variant[st][k] = live[st][k % 4], variant[st][k % 4]
    plain = add_rule.pooled_verdict(live, variant)
    with_m = add_rule.pooled_verdict(live, variant, market=True)
    assert "market_noise" in with_m and "market_noise" not in plain
    assert with_m["decision"] == plain["decision"] and with_m["pooled"] == plain["pooled"]
    assert "market noise (PREREG183)" in add_rule.pooled_summary(with_m)
