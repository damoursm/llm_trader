"""Dependency-free significance tests.

These exist because the scipy versions they replace were unreachable: both call
sites wrapped `from scipy.stats import ...` in `except Exception`, scipy was
never a declared dependency, and so the fallback was the only branch that ever
ran — every method UNPROVEN, every regime "test unavailable", silently, for the
whole life of those modules.

So the first duty of this suite is to pin the ARITHMETIC against independently
known values (exact integer sums for the binomial, a textbook Welch example),
and the second is to pin the guards, so a degenerate input can never be
mistaken for a confident verdict.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.stats import (binom_p_greater, binom_p_less, t_sf_two_sided,
                                welch_t_test)


# ── binomial: checked against exact integer arithmetic ────────────────────────

def _exact_cdf(k: int, n: int) -> float:
    """P(X <= k) for Binomial(n, 1/2) computed in exact integer arithmetic."""
    from fractions import Fraction
    return float(Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n))


@pytest.mark.parametrize("k,n", [(3, 10), (0, 5), (10, 10), (5, 10),
                                 (450, 1000), (2399, 5000)])
def test_binom_matches_exact_integer_arithmetic(k, n):
    assert binom_p_less(k, n, 0.5) == pytest.approx(_exact_cdf(k, n), abs=1e-12)


def test_binom_tails_are_complementary():
    """P(X<=k) + P(X>=k+1) == 1 — a cheap invariant that catches an off-by-one
    in either tail. Holds at the boundary too: P(X>=n+1) is an IMPOSSIBLE event
    and must be 0, not a tail silently clamped back to n."""
    for k in (0, 3, 7, 9, 10):
        assert (binom_p_less(k, 10, 0.5)
                + binom_p_greater(k + 1, 10, 0.5)) == pytest.approx(1.0, abs=1e-12)


def test_binom_out_of_domain_counts():
    assert binom_p_greater(11, 10, 0.5) == 0.0      # impossible
    assert binom_p_less(11, 10, 0.5) == 1.0         # certain


def test_binom_large_n_does_not_overflow():
    """Computed in log space: the raw coefficient C(20000, 10000) overflows a
    float, so a naive implementation returns inf/nan here."""
    p = binom_p_less(9800, 20000, 0.5)
    assert 0.0 < p < 0.01 and math.isfinite(p)


def test_binom_degenerate_n_is_uninformative_not_significant():
    assert binom_p_less(0, 0, 0.5) == 1.0
    assert binom_p_greater(0, 0, 0.5) == 1.0


# ── Student-t / Welch ─────────────────────────────────────────────────────────

def test_t_critical_values():
    """The classic two-sided 5% critical points."""
    assert t_sf_two_sided(2.228139, 10) == pytest.approx(0.05, abs=1e-6)
    assert t_sf_two_sided(1.959964, 1e7) == pytest.approx(0.05, abs=1e-5)
    assert t_sf_two_sided(0.0, 5) == pytest.approx(1.0, abs=1e-12)


def test_t_is_symmetric_in_sign():
    for t, df in ((1.7, 12), (3.1, 40), (0.4, 6)):
        assert t_sf_two_sided(t, df) == pytest.approx(t_sf_two_sided(-t, df))


def test_welch_matches_textbook_example():
    """Wikipedia's Welch's t-test worked example (A1 vs A2)."""
    a = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1,
         19.6, 19.0, 21.7, 21.4]
    b = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9,
         22.1, 22.9, 20.5, 24.4]
    t, p = welch_t_test(a, b)
    assert t == pytest.approx(-2.455356, abs=1e-6)
    assert p == pytest.approx(0.021378, abs=1e-6)


def test_welch_detects_a_real_difference_and_ignores_a_fake_one():
    far = welch_t_test([10.0] * 5 + [10.5, 9.5], [1.0] * 5 + [1.5, 0.5])[1]
    same = welch_t_test([1.0, 2.0, 3.0, 2.0], [1.1, 2.1, 2.9, 2.0])[1]
    assert far < 0.001
    assert same > 0.10


def test_welch_is_undefined_not_significant_on_degenerate_input():
    """A single observation, or zero variance in both arms, has no answer. It
    must read NaN — a caller that saw 0.0 would report a certain difference."""
    assert all(math.isnan(v) for v in welch_t_test([1.0], [1.0, 2.0]))
    assert all(math.isnan(v) for v in welch_t_test([5.0, 5.0], [5.0, 5.0]))
    assert all(math.isnan(v) for v in welch_t_test([], []))


def test_welch_ignores_nan_observations():
    clean = welch_t_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    dirty = welch_t_test([1.0, 2.0, 3.0, float("nan")],
                         [4.0, float("nan"), 5.0, 6.0])
    assert clean[0] == pytest.approx(dirty[0])
    assert clean[1] == pytest.approx(dirty[1])


# ── the regression that motivated the module ──────────────────────────────────

def test_callers_do_not_depend_on_scipy():
    """The two consumers must not reach for an undeclared dependency again: an
    optional import guarding a verdict the system ACTS on is an invisible off
    switch, not a graceful fallback."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "analysis"
    # Match a real import, not the word (the modules explain the history in a
    # comment, and a substring check would flag their own post-mortem).
    pat = re.compile(r"^\s*(?:from\s+scipy|import\s+scipy)", re.MULTILINE)
    for name in ("method_horizons.py", "regime_performance.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert not pat.search(src), f"{name} reintroduced a scipy import"
