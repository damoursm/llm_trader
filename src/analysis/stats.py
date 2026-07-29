"""Significance tests, without scipy.

This project already hand-rolls its Spearman correlation on pandas ranks rather
than taking a scipy dependency, and these follow that convention — but the
reason to write them is stronger than consistency.

`method_horizons` and `regime_performance` each reached for `scipy.stats` inside
a `try/except Exception` that fell back to a neutral verdict. scipy was never in
requirements and was not installed, so the except branch was the ONLY branch
that ever ran: every method classified UNPROVEN (never PROVEN, never DISPROVEN)
and every regime comparison reported "test unavailable". Both failures were
invisible — a degraded verdict looks exactly like a real one, and the UNPROVEN
verdict silently fed live method weighting. An optional import guarding a
decision the system acts on is not a fallback, it is an off switch nobody can
see.

So the tests live here, exact and dependency-free, and the callers do NOT wrap
them in a bare except: a genuine failure must surface.

Verified against scipy's `binomtest` / `ttest_ind(equal_var=False)` to 1e-12 —
see `tests/test_stats.py` for the reference values.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

__all__ = ["binom_p_less", "binom_p_greater", "welch_t_test", "t_sf_two_sided"]


# ── binomial ──────────────────────────────────────────────────────────────────

def binom_p_less(successes: int, n: int, p: float = 0.5) -> float:
    """One-sided P(X <= successes) for X ~ Binomial(n, p). Exact.

    The probability of a win rate this low OR LOWER arising by chance — so a
    SMALL value is evidence the method is genuinely worse than the bar.
    """
    if n <= 0:
        return 1.0
    if successes >= n:          # P(X <= n) is certain; do not clamp silently
        return 1.0
    return _binom_cdf(max(0, int(successes)), int(n), float(p))


def binom_p_greater(successes: int, n: int, p: float = 0.5) -> float:
    """One-sided P(X >= successes). SMALL = genuinely better than the bar."""
    if n <= 0:
        return 1.0
    if successes > n:           # impossible event — 0, not a clamped tail
        return 0.0
    k = max(0, int(successes))
    if k == 0:
        return 1.0
    return 1.0 - _binom_cdf(k - 1, int(n), float(p))


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    # Sum in log space so large n cannot overflow the binomial coefficient.
    total = 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    for i in range(k + 1):
        log_term = (_log_comb(n, i) + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(1.0, max(0.0, total))


def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


# ── Welch's t-test ────────────────────────────────────────────────────────────

def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Welch (unequal-variance) t-test. Returns ``(t, p)``.

    ``(nan, nan)`` when the test is undefined — fewer than 2 observations per
    arm, or zero variance in both. Callers must treat that as "no verdict",
    never as "not significant".
    """
    a = [float(x) for x in a if x == x]
    b = [float(x) for x in b if x == x]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")

    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)

    se2 = v1 / n1 + v2 / n2
    if se2 <= 0.0:
        return float("nan"), float("nan")      # identical constants: undefined
    t = (m1 - m2) / math.sqrt(se2)

    num = se2 * se2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if den <= 0.0:
        return float("nan"), float("nan")
    df = num / den
    return t, t_sf_two_sided(t, df)


def t_sf_two_sided(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student-t with ``df`` degrees of freedom.

    Uses the identity ``P = I_x(df/2, 1/2)`` with ``x = df / (df + t^2)``, where
    ``I`` is the regularized incomplete beta function.
    """
    if df <= 0 or t != t or df != df:
        return float("nan")
    x = df / (df + float(t) * float(t))
    return max(0.0, min(1.0, _betainc(df / 2.0, 0.5, x)))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) — Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    # The CF converges fast only on one side of the symmetry point; reflect.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-16) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
