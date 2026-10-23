"""The Sharpe-ratio inference in `src/analysis/stats.py` (PREREG183): López de Prado, Lipton & Zoonekynd, "Sharpe Ratio
Inference: A New Standard for Reporting and Decision-Making" (SSRN 5520741), ported from the formulas of its reference
code (github.com/zoonek/2025-sharpe-ratio, `functions.py`), and the robust regressions a book's market exposure needs.

Three duties: the REFERENCE CODE'S OWN TEST VALUES reproduced (its asserts, to its three decimals — the paper's
numeric example: a hedge fund's monthly returns, SR = 0.036 / 0.079, skewness -2.448, kurtosis 10.164, 24 months);
the closed forms where they exist (the maximum of 2 and 3 normals, the sample-size formula plugged back into the
power, the regressions written out by hand); the behaviour the formulas promise (autocorrelation widens the
variance, more trials narrow the maximum's, the old Bailey & López de Prado functions untouched).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from src.analysis import stats as S

ND = NormalDist()
SR = 0.036 / 0.079                       # the paper's numeric example (monthly)
G3, G4, T24 = -2.448, 10.164, 24


# ── the reference code's asserts ─────────────────────────────────────────────

def test_reference_values_reproduced():
    assert round(math.sqrt(S.sharpe_variance(SR, T24, G3, G4)), 3) == 0.329
    assert round(math.sqrt(S.sharpe_variance(SR, T24)), 3) == 0.214                 # normal, iid
    assert round(S.min_track_record(SR, 0.0, G3, G4, alpha=0.05), 3) == 13.029
    assert round(S.sharpe_psr(SR, 0.0, T24, G3, G4), 3) == 0.987
    assert round(S.sharpe_psr(SR, 0.1, T24, G3, G4), 3) == 0.939
    assert round(1.0 - S.sharpe_power(0.0, 0.5, T24, G3, G4), 3) == 0.315            # beta
    assert round(S.pfdr(0.05, 0.05, 0.315), 3) == 0.581
    assert round(S.ofdr(SR, 0.0, 0.5, T24, 0.05, G3, G4), 3) == 0.306


def test_autocorrelation_terms_written_out():
    """rho = 0.2: a = 1.5, b = 1.24 / 0.96, c = 1.04 / 0.96 — the registration's hand value 0.3796 was taken at SR
    rounded to 0.456 (0.3795 at the exact 0.036 / 0.079)."""
    a, b, c = S.ar1_terms(0.2)
    assert (a, b, c) == pytest.approx((1.5, 1.24 / 0.96, 1.04 / 0.96), abs=1e-15)
    for sr in (0.456, SR):
        by_hand = (a - b * G3 * sr + c * (G4 - 1) / 4 * sr * sr) / T24
        assert S.sharpe_variance(sr, T24, G3, G4, rho=0.2) == pytest.approx(by_hand, rel=1e-14)
    assert round(math.sqrt(S.sharpe_variance(0.456, T24, G3, G4, rho=0.2)), 4) == 0.3796
    assert round(math.sqrt(S.sharpe_variance(SR, T24, G3, G4, rho=0.2)), 4) == 0.3795
    assert S.ar1_terms(0.0) == (1.0, 1.0, 1.0)
    assert S.sharpe_variance(SR, T24, rho=0.3) > S.sharpe_variance(SR, T24) > S.sharpe_variance(SR, T24, rho=-0.3)
    with pytest.raises(ValueError):
        S.ar1_terms(1.0)


def test_maximum_of_normals_against_closed_forms():
    e, e2, v = S.max_normal_moments(2)
    assert (e, e2, v) == pytest.approx((1 / math.sqrt(math.pi), 1.0, 1 - 1 / math.pi), abs=1e-7)
    e, e2, v = S.max_normal_moments(3)
    assert e == pytest.approx(3 / (2 * math.sqrt(math.pi)), abs=1e-7)
    assert e2 == pytest.approx(1 + math.sqrt(3) / (2 * math.pi), abs=1e-7)
    assert S.max_normal_moments(10)[0] == pytest.approx(1.5387527, abs=1e-6)        # Harter's table
    assert S.max_normal_moments(1) == (0.0, 1.0, 1.0)
    var = [S.max_normal_moments(k)[2] for k in (1, 2, 10, 100, 10_000)]
    assert var == sorted(var, reverse=True)                                         # the best of many: sharper
    for k in (10, 100, 1000, 100_000):                                              # Bailey & LdP's approximation
        assert abs(S.max_normal_moments(k)[0] - S.expected_max_sharpe(k, 1.0)) < 0.04
    assert S.sharpe_variance(SR, T24, n_trials=10) == pytest.approx(
        S.sharpe_variance(SR, T24) * S.max_normal_moments(10)[2], rel=1e-14)


# ── the tests built on the variance ──────────────────────────────────────────

def test_psr_takes_the_variance_under_the_null():
    """sharpe_psr at SR0 = 0: Φ(SR sqrt(T)) whatever the skew and kurtosis (they vanish at SR = 0); the old
    probabilistic_sharpe (Bailey & LdP 2012) keeps its own construction, the variance at the estimate with T - 1."""
    assert S.sharpe_psr(0.3, 0.0, 50, skew=-1.0, kurt=9.0) == pytest.approx(ND.cdf(0.3 * math.sqrt(50)), abs=1e-15)
    old = ND.cdf(0.3 * math.sqrt(49) / math.sqrt(1 + 1.0 * 0.3 + 8.0 / 4 * 0.09))
    assert S.probabilistic_sharpe(0.3, 0.0, 50, skew=-1.0, kurt=9.0) == pytest.approx(old, abs=1e-15)
    rho = 0.25
    a = (1 + rho) / (1 - rho)
    assert S.sharpe_psr(0.3, 0.0, 50, rho=rho) == pytest.approx(ND.cdf(0.3 * math.sqrt(50 / a)), abs=1e-15)
    assert math.isnan(S.sharpe_psr(2.0, 1.5, 100, skew=5.0))                       # a variance term <= 0


def test_power_and_track_record_agree():
    for sr0, sr1, rho in ((0.0, SR, 0.0), (0.1, SR, 0.2), (SR, SR / 2, 0.1), (0.3, 0.05, -0.1)):
        T = S.track_record_for_power(sr0, sr1, 0.8, G3, G4, rho)
        assert S.sharpe_power(sr0, sr1, T, G3, G4, rho) == pytest.approx(0.8, abs=1e-12)
    assert S.min_track_record(SR, 0.1, G3, G4) == pytest.approx(S.track_record_for_power(0.1, SR, 0.5, G3, G4),
                                                                rel=1e-12)
    assert S.min_track_record(0.1, 0.2) == math.inf                                 # a lower Sharpe never clears
    # the power at the MinTRL is one half, and the critical Sharpe equals the observed one there
    T = S.min_track_record(SR, 0.0, G3, G4)
    assert S.critical_sharpe(0.0, T, G3, G4) == pytest.approx(SR, rel=1e-12)
    assert S.critical_sharpe(SR, 30, lower=True) < SR < S.critical_sharpe(SR, 30)


def test_false_discovery_rates():
    p, a, b = 0.1, 0.05, 0.3
    assert S.pfdr(p, a, b) == pytest.approx(a * (1 - p) / (a * (1 - p) + (1 - b) * p), abs=1e-15)
    # oFDR written out from its two p-values
    p0 = 1 - S.sharpe_psr(SR, 0.0, T24, G3, G4)
    p1 = 1 - S.sharpe_psr(SR, 0.5, T24, G3, G4)
    assert S.ofdr(SR, 0.0, 0.5, T24, 0.2, G3, G4) == pytest.approx(p0 * 0.8 / (p0 * 0.8 + p1 * 0.2), abs=1e-15)
    with pytest.raises(ValueError):
        S.pfdr(0.0, 0.05, 0.3)


def test_fdr_control_hits_its_target():
    """At the returned critical value P[H0 | SR > c] is q exactly, computed here from the two normal tails."""
    r = S.fdr_critical_sharpe(0.25, 0.0, 0.5, 0.10, T24, G3, G4)
    s0, s1 = r["sigma0"], r["sigma1"]
    a = 1 - ND.cdf((r["sr_c"] - 0.0) / s0)
    pw = 1 - ND.cdf((r["sr_c"] - 0.5) / s1)
    assert a * 0.9 / (a * 0.9 + pw * 0.1) == pytest.approx(0.25, abs=1e-9)
    assert r["q_hat"] == pytest.approx(0.25, abs=1e-9)
    assert r["alpha"] == pytest.approx(a, rel=1e-9) and r["power"] == pytest.approx(pw, rel=1e-9)
    assert S.fdr_critical_sharpe(0.95, 0.0, 0.5, 0.10, T24)["sr_c"] == -math.inf    # q above 1 - P[H1]: no bar


def test_deflated_sharpe_k():
    v0 = S.sharpe_variance(0.0, 60, rho=0.1)
    one = S.deflated_sharpe_k(0.4, 1, 60, -0.5, 6.0, 0.1, sr_var=v0)
    assert one["dsr"] == pytest.approx(S.sharpe_psr(0.4, 0.0, 60, -0.5, 6.0, 0.1), abs=1e-15)   # one trial
    d = S.deflated_sharpe_k(0.4, 20, 60, -0.5, 6.0, 0.1, sr_var=v0)
    thr = S.expected_max_sharpe(20, v0)
    by_hand = ND.cdf((0.4 - thr) / math.sqrt(S.sharpe_variance(thr, 60, -0.5, 6.0, 0.1)
                                             * S.max_normal_moments(20)[2]))
    assert d["threshold"] == pytest.approx(thr, abs=1e-15) and d["dsr"] == pytest.approx(by_hand, abs=1e-14)
    out = [S.deflated_sharpe_k(0.4, n, 60, sr_var=v0)["dsr"] for n in (1, 5, 50, 500)]
    assert out == sorted(out, reverse=True)                                         # more trials deflate more
    with pytest.raises(ValueError):
        S.deflated_sharpe_k(0.4, 20, 60)


def test_sharpe_moments():
    m = S.sharpe_moments([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m["sr"] == pytest.approx(3.0 / math.sqrt(2.5), abs=1e-15)
    assert m["skew"] == pytest.approx(0.0, abs=1e-15) and m["kurt"] == pytest.approx(1.7, abs=1e-14)
    assert m["rho"] == pytest.approx(1.0, abs=1e-15)
    x = np.random.default_rng(0).normal(size=20_000)
    m = S.sharpe_moments(x)
    assert abs(m["skew"]) < 0.05 and abs(m["kurt"] - 3) < 0.1 and abs(m["rho"]) < 0.03
    assert math.isnan(S.sharpe_moments([1.0, 1.0, 1.0])["sr"])


# ── the regressions ──────────────────────────────────────────────────────────

def _data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.3 + 1.2 * x + rng.normal(size=n) * (1 + 0.5 * np.abs(x))
    return y, x


def test_ols_hac_written_out():
    y, x = _data()
    n, k = len(y), 2
    X = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ coef
    B = np.linalg.inv(X.T @ X)
    for L in (0, 1, 3):
        Sm = sum(u[t] ** 2 * np.outer(X[t], X[t]) for t in range(n))
        for lag in range(1, L + 1):
            w = 1 - lag / (L + 1)
            for t in range(lag, n):
                g = u[t] * u[t - lag] * np.outer(X[t], X[t - lag])
                Sm = Sm + w * (g + g.T)
        se = np.sqrt(np.diag(B @ Sm @ B * n / (n - k)))
        r = S.ols_hac(y, x, lags=L)
        assert r["coef"] == pytest.approx(list(coef), abs=1e-12)
        assert r["se"] == pytest.approx(list(se), abs=1e-12)
        assert r["p"][1] == pytest.approx(S.t_sf_two_sided(coef[1] / se[1], n - k), abs=1e-14)
    assert S.ols_hac(y, x)["lags"] == math.floor(4 * (n / 100) ** (2 / 9))


def test_ols_cluster_written_out():
    y, x = _data(n=30, seed=2)
    n, k = len(y), 2
    groups = np.repeat(np.arange(6), 5)
    X = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ coef
    B = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[groups == g].T @ u[groups == g], X[groups == g].T @ u[groups == g]) for g in range(6))
    se = np.sqrt(np.diag(B @ meat @ B * (6 / 5) * ((n - 1) / (n - k))))
    r = S.ols_cluster(y, x, groups)
    assert r["clusters"] == 6 and r["df"] == 5
    assert r["se"] == pytest.approx(list(se), abs=1e-12)
    # every row its own cluster: CR1 = the HAC with no lag (both HC0 x n / (n - k))
    assert S.ols_cluster(y, x, np.arange(n))["se"] == pytest.approx(S.ols_hac(y, x, lags=0)["se"], abs=1e-12)
    with pytest.raises(ValueError):
        S.ols_cluster(y, x, np.zeros(n))
    with pytest.raises(ValueError):
        S.ols_hac(np.r_[y[:-1], np.nan], x)
