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

The overfitting statistics (`pbo_cscv`, `deflated_sharpe`, `probabilistic_sharpe`)
are the construction PREREG81 ran in its study script (`part2_81.py`), ported
here so the next study reuses one audited copy — `tests/test_stats_overfitting.py`
pins them.

The SHARPE-RATIO INFERENCE section (PREREG183, the user 2026-10-22: "Copy the
formulas to improve our own skills") ports López de Prado, Lipton & Zoonekynd,
"Sharpe Ratio Inference: A New Standard for Reporting and Decision-Making" (SSRN
5520741; the reference code github.com/zoonek/2025-sharpe-ratio, `functions.py`):
the Sharpe estimate's variance under skewness, fat tails and AR(1)
autocorrelation, the PSR with that variance UNDER THE NULL, the minimum track
record, the critical Sharpe, the power, the planned and observed false discovery
rates (pFDR, oFDR), the FDR-controlling critical value and the deflated (FWER)
test with the variance of the maximum of K trials — plus the two regressions a
book's market exposure needs (Newey-West and cluster-robust OLS). The older
functions above keep their numbers (Bailey & López de Prado 2012/2014: the
variance at the estimate, T - 1, no autocorrelation); every study before PREREG183
reads them. `tests/test_stats_sharpe_inference.py` pins the reference code's own
test values.
"""

from __future__ import annotations

import functools
import itertools
import math
from statistics import NormalDist
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["binom_p_less", "binom_p_greater", "welch_t_test", "t_sf_two_sided",
           "pbo_cscv", "deflated_sharpe", "probabilistic_sharpe", "expected_max_sharpe",
           "EULER_GAMMA",
           "ar1_terms", "max_normal_moments", "sharpe_variance", "sharpe_psr", "min_track_record",
           "track_record_for_power", "critical_sharpe", "sharpe_power", "pfdr", "ofdr",
           "fdr_critical_sharpe", "deflated_sharpe_k", "sharpe_moments", "ols_hac", "ols_cluster"]


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


# ── the probability of backtest overfitting (CSCV) ────────────────────────────

Scorer = Union[str, Callable[[np.ndarray], np.ndarray]]


def pbo_cscv(M, n_splits: int = 16, scorer: Scorer = "mean",
             max_combinations: Optional[int] = None, seed: int = 0) -> Dict[str, object]:
    """The Probability of Backtest Overfitting by combinatorially symmetric
    cross-validation (Bailey, Borwein, López de Prado & Zhu 2014).

    ``M`` is a T x N matrix: the per-period returns (or log growth — anything
    whose mean over a span measures the span's performance) of N candidate
    configurations over the same T periods. The periods are cut into
    ``n_splits`` (even) contiguous groups; for every choice of ``n_splits / 2``
    groups as the in-sample half (the rest out of sample) the configuration
    with the best in-sample score is picked and its out-of-sample score ranked
    among the N — rank 1 the worst, N the best, ties sharing their mean rank.
    The relative rank ``w = rank / (N + 1)`` has logit ``log(w / (1 - w))``;
    PBO is the share of halvings whose logit is negative, i.e. where the
    in-sample winner lands BELOW THE MEDIAN out of sample. Pure noise gives
    ~0.5, a configuration with a real edge ~0, a family whose winner was
    selected by luck → 1.

    ``scorer``: ``"mean"`` (the default — the mean per-period return of the
    half; identical to ranking by the half's total, which is what PREREG81's
    script did), ``"sharpe"`` (the half's mean over its sample standard
    deviation, ddof 1; a constant half scores ±inf by the sign of its mean), or
    a callable taking the half's rows (an n x N array) and returning N scores
    (slower: it is called twice per halving — cap the halvings).

    ``max_combinations``: when the number of halvings, C(n_splits, n_splits/2)
    (12,870 for 16 groups), exceeds it, a random subset of that size is used,
    drawn with ``seed``. The built-in scorers work on per-group sums, so every
    halving of 16 groups costs a few matrix products — the cap matters for a
    callable scorer or many more groups.

    ``M`` must be finite: align the configurations on their common periods
    before calling (a NaN period would poison every halving it falls in).

    Returns a dict: ``pbo``, ``n_combinations`` (the halvings judged),
    ``n_total_combinations`` (the halvings that exist), ``n_splits``,
    ``n_configs``, ``n_periods``, ``scorer``, ``logits`` (their mean, median,
    standard deviation, min and max), ``mean_rel_rank`` (the mean of ``w``:
    0.5 is the median), ``p_oos_loss`` (the share of halvings where the
    in-sample winner's out-of-sample MEAN return is negative, whatever the
    scorer) and ``selected_top`` (the configurations chosen in sample most
    often, column index → halvings). The groups follow PREREG81's cut —
    ``linspace(0, T, n_splits + 1)`` truncated to integers — so a matrix the
    study scored gets the same halvings here.
    """
    X = np.asarray(M, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"M must be a T x N matrix, got shape {X.shape}")
    T, N = X.shape
    S = int(n_splits)
    if S < 2 or S % 2:
        raise ValueError(f"n_splits must be an even number >= 2, got {n_splits}")
    if N < 2:
        raise ValueError(f"at least two configurations are needed, got {N}")
    if T < S:
        raise ValueError(f"{T} periods cannot be split into {S} groups")
    if not np.isfinite(X).all():
        raise ValueError("M must be finite — align the configurations on their common periods first")
    if isinstance(scorer, str):
        if scorer not in ("mean", "sharpe"):
            raise ValueError(f"scorer must be 'mean', 'sharpe' or a callable, got {scorer!r}")
    elif not callable(scorer):
        raise ValueError(f"scorer must be 'mean', 'sharpe' or a callable, got {scorer!r}")

    edges = np.linspace(0, T, S + 1).astype(int)                      # PREREG81's cut
    lengths = np.diff(edges).astype(float)
    sums = np.array([X[edges[i]:edges[i + 1]].sum(0) for i in range(S)])          # S x N
    sums2 = np.array([(X[edges[i]:edges[i + 1]] ** 2).sum(0) for i in range(S)])  # S x N

    combos = _half_combinations(S, max_combinations, seed)            # C x S/2 group indices
    C = len(combos)
    mask = np.zeros((C, S), dtype=bool)
    mask[np.arange(C)[:, None], combos] = True
    w_in, w_out = mask.astype(float), (~mask).astype(float)
    n_in, n_out = w_in @ lengths, w_out @ lengths
    mean_in, mean_out = (w_in @ sums) / n_in[:, None], (w_out @ sums) / n_out[:, None]

    if scorer == "mean":
        score_in, score_out = mean_in, mean_out
    elif scorer == "sharpe":
        score_in = _sharpe_from_sums(w_in @ sums, w_in @ sums2, n_in)
        score_out = _sharpe_from_sums(w_out @ sums, w_out @ sums2, n_out)
    else:
        group_of_row = np.repeat(np.arange(S), np.diff(edges))
        score_in = np.empty((C, N))
        score_out = np.empty((C, N))
        for c in range(C):
            rows_in = mask[c][group_of_row]
            score_in[c] = np.asarray(scorer(X[rows_in]), dtype=float)
            score_out[c] = np.asarray(scorer(X[~rows_in]), dtype=float)
        if score_in.shape != (C, N) or not np.isfinite(score_in).all() or not np.isfinite(score_out).all():
            raise ValueError("the scorer must return one finite score per configuration")

    best = score_in.argmax(1)
    rows = np.arange(C)
    oos_best = score_out[rows, best][:, None]
    # 1 = the worst out of sample, N = the best; ties share their mean rank.
    rank = ((score_out < oos_best).sum(1) + 0.5 * ((score_out == oos_best).sum(1) - 1) + 1)
    w = rank / (N + 1)
    logits = np.log(w / (1.0 - w))
    counts = np.bincount(best, minlength=N)
    top = np.argsort(-counts, kind="stable")[:3]
    return {
        "pbo": float((logits < 0).mean()),
        "n_combinations": int(C),
        "n_total_combinations": int(math.comb(S, S // 2)),
        "n_splits": S,
        "n_configs": int(N),
        "n_periods": int(T),
        "scorer": scorer if isinstance(scorer, str) else getattr(scorer, "__name__", "callable"),
        "logits": {"mean": float(logits.mean()), "median": float(np.median(logits)),
                   "std": float(logits.std(ddof=1)) if C > 1 else 0.0,
                   "min": float(logits.min()), "max": float(logits.max())},
        "mean_rel_rank": float(w.mean()),
        "p_oos_loss": float((mean_out[rows, best] < 0).mean()),
        "selected_top": {int(k): int(counts[k]) for k in top if counts[k] > 0},
    }


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """mean / sample sd (ddof 1) of each half from its sums and sums of squares;
    a half with no variance scores +inf / -inf / 0 by the sign of its mean."""
    n = n[:, None]
    mean = s1 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(n > 1, (s2 - s1 * s1 / n) / np.maximum(n - 1, 1), 0.0)
        var = np.where(var > 1e-300, var, 0.0)                     # rounding below zero = flat
        sharpe = np.where(var > 0, mean / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
    flat = var <= 0
    sharpe = np.where(flat & (mean > 0), np.inf, sharpe)
    sharpe = np.where(flat & (mean < 0), -np.inf, sharpe)
    return sharpe


def _half_combinations(S: int, max_combinations: Optional[int], seed: int) -> np.ndarray:
    """Every choice of S/2 of S groups (C x S/2, lexicographic), or a seeded
    random subset of ``max_combinations`` distinct ones when there are more."""
    total = math.comb(S, S // 2)
    if max_combinations is not None and int(max_combinations) < 1:
        raise ValueError(f"max_combinations must be >= 1, got {max_combinations}")
    if max_combinations is None or int(max_combinations) >= total:
        return np.array(list(itertools.combinations(range(S), S // 2)), dtype=int)
    k = int(max_combinations)
    rng = np.random.default_rng(seed)
    if total <= 500_000:                                             # enumerable: an exact uniform subset
        every = np.array(list(itertools.combinations(range(S), S // 2)), dtype=int)
        pick = np.sort(rng.choice(total, size=k, replace=False))
        return every[pick]
    chosen = set()                                                   # too many to enumerate: draw distinct ones
    while len(chosen) < k:
        chosen.add(tuple(sorted(rng.choice(S, size=S // 2, replace=False).tolist())))
    return np.array(sorted(chosen), dtype=int)


# ── the deflated Sharpe ratio ─────────────────────────────────────────────────

EULER_GAMMA = 0.5772156649015329          # the Euler–Mascheroni constant
_ND = NormalDist()


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """SR0 of Bailey & López de Prado (2014): the expected maximum Sharpe ratio
    among ``n_trials`` independent trials whose true Sharpe is zero and whose
    estimated Sharpes have variance ``sr_var`` (per-period units, the same as
    the Sharpe it will deflate):

        SR0 = sqrt(sr_var) * ((1 - γ) Z⁻¹(1 - 1/n) + γ Z⁻¹(1 - 1/(n e)))

    with γ the Euler–Mascheroni constant — the asymptotic expected maximum of n
    standard normals, scaled. A single trial has nothing to select from, so
    ``n_trials = 1`` gives 0 (the formula itself breaks there: Z⁻¹(0) = -inf);
    ``sr_var = 0`` gives 0 for any n.
    """
    n = int(n_trials)
    v = float(sr_var)
    if n < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not (v >= 0.0):
        raise ValueError(f"sr_var must be a non-negative number, got {sr_var}")
    if n == 1 or v == 0.0:
        return 0.0
    z_n = _ND.inv_cdf(1.0 - 1.0 / n)
    z_ne = _ND.inv_cdf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(v) * ((1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne)


def probabilistic_sharpe(sr: float, sr_benchmark: float, T: int, skew: float = 0.0,
                         kurt: float = 3.0) -> float:
    """The Probabilistic Sharpe Ratio (Bailey & López de Prado 2012): the
    probability that the true Sharpe exceeds ``sr_benchmark`` given the
    estimate ``sr`` over ``T`` periods, with the returns' skewness and plain
    (Pearson) kurtosis — 3 for a normal; pandas' ``.kurt()`` is the EXCESS
    kurtosis, add 3:

        PSR = Φ( (sr - sr_benchmark) sqrt(T - 1) / sqrt(1 - skew·sr + (kurt - 1)/4 · sr²) )

    ``sr`` and ``sr_benchmark`` in per-period units (daily for daily returns).
    ``nan`` when the test is undefined — fewer than 2 periods, or a variance
    term at or below zero (the asymptotic variance of the Sharpe estimator
    failing, which a large positive skew with a high Sharpe can do). Callers
    treat that as "no verdict", never as a probability.
    """
    T = int(T)
    sr, sr_benchmark = float(sr), float(sr_benchmark)
    if T < 2 or any(x != x for x in (sr, sr_benchmark, skew, kurt)):
        return float("nan")
    var_term = 1.0 - float(skew) * sr + (float(kurt) - 1.0) / 4.0 * sr * sr
    if not (var_term > 0.0) or not math.isfinite(var_term):
        return float("nan")
    return _ND.cdf((sr - sr_benchmark) * math.sqrt(T - 1.0) / math.sqrt(var_term))


def deflated_sharpe(sr: float, n_trials: int, T: int, skew: float = 0.0, kurt: float = 3.0,
                    sr_var: Optional[float] = None) -> Dict[str, float]:
    """The Deflated Sharpe Ratio (Bailey & López de Prado 2014): the PSR of the
    selected strategy's Sharpe ``sr`` against SR0, the Sharpe that the best of
    ``n_trials`` skill-less trials would be expected to show
    (`expected_max_sharpe`) — the probability that the true Sharpe is above
    zero once the selection among the trials is accounted for.

    ``sr_var`` is the variance of the Sharpe ratios across the trials that
    were tried (per-period units like ``sr``; PREREG81 used the sample
    variance over its 67 configurations). It is REQUIRED: nothing is estimated
    here, and a silent default of 0 would deflate nothing while reading like a
    verdict — pass 0.0 explicitly to mean "the trials all showed the same
    Sharpe" (DSR = PSR against zero). ``T``, ``skew``, ``kurt`` as in
    `probabilistic_sharpe`.

    Returns a dict: ``dsr``, ``sr0``, ``psr0`` (the PSR against zero, the
    undeflated verdict) and the inputs (``sr``, ``n_trials``, ``T``, ``skew``,
    ``kurt``, ``sr_var``). ``dsr`` is ``nan`` when the PSR is undefined.
    """
    if sr_var is None:
        raise ValueError("deflated_sharpe needs sr_var, the variance of the trials' Sharpe ratios "
                         "(pass 0.0 to deflate by nothing)")
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return {
        "dsr": probabilistic_sharpe(sr, sr0, T, skew, kurt),
        "sr0": float(sr0),
        "psr0": probabilistic_sharpe(sr, 0.0, T, skew, kurt),
        "sr": float(sr),
        "n_trials": int(n_trials),
        "T": int(T),
        "skew": float(skew),
        "kurt": float(kurt),
        "sr_var": float(sr_var),
    }


# ── Sharpe-ratio inference (López de Prado, Lipton & Zoonekynd 2025; PREREG183) ─
#
# The reference code's names: sharpe_ratio_variance -> `sharpe_variance`, moments_Mk -> `max_normal_moments`,
# probabilistic_sharpe_ratio -> `sharpe_psr`, minimum_track_record_length -> `min_track_record`,
# critical_sharpe_ratio -> `critical_sharpe`, sharpe_ratio_power -> `sharpe_power`, pFDR -> `pfdr`, oFDR -> `ofdr`,
# control_for_FDR -> `fdr_critical_sharpe`; the FWER construction of its numeric example -> `deflated_sharpe_k`.
# Units: a Sharpe ratio per period of the returns it was measured on (monthly returns give a monthly Sharpe; never
# annualize before testing). Skewness γ3 and the PLAIN kurtosis γ4 (3 for a normal) of those returns; ρ their
# first-order autocorrelation, the returns read as AR(1).

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _z(p: float) -> float:
    return _ND.inv_cdf(float(p))


def _norm_logsf(x: float) -> float:
    """log P[Z > x] for a standard normal, accurate in both tails (the asymptotic series past erfc's underflow)."""
    x = float(x)
    if x < 25.0:
        return math.log(0.5 * math.erfc(x / _SQRT2))
    x2 = x * x
    return -0.5 * x2 - math.log(x) - _LOG_SQRT_2PI + math.log1p(-1.0 / x2 + 3.0 / (x2 * x2) - 15.0 / (x2 * x2 * x2))


def ar1_terms(rho: float = 0.0) -> Tuple[float, float, float]:
    """The paper's AR(1) factors (a, b, c) of the Sharpe variance, ρ the returns' first-order autocorrelation:
    a = (1 + ρ) / (1 - ρ), b = (1 + ρ + ρ²) / (1 - ρ²), c = (1 + ρ²) / (1 - ρ²) — all 1 for independent returns."""
    r = float(rho)
    if not -1.0 < r < 1.0:
        raise ValueError(f"rho must lie strictly between -1 and 1, got {rho}")
    return (1.0 + r) / (1.0 - r), (1.0 + r + r * r) / (1.0 - r * r), (1.0 + r * r) / (1.0 - r * r)


@functools.lru_cache(maxsize=1)
def _max_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The integration grid of `max_normal_moments`: x on [-12, 12] by 0.001, log φ(x) and log Φ(x) (both tails
    exact: erfc below 0, log1p above)."""
    x = np.linspace(-12.0, 12.0, 24_001)
    log_phi = -0.5 * x * x - _LOG_SQRT_2PI
    log_cdf = np.array([math.log(0.5 * math.erfc(-v / _SQRT2)) if v < 0.0 else math.log1p(-0.5 * math.erfc(v / _SQRT2))
                        for v in x.tolist()])
    return x, log_phi, log_cdf


@functools.lru_cache(maxsize=4096)
def max_normal_moments(k: int) -> Tuple[float, float, float]:
    """(E[M], E[M²], Var[M]) of M = the maximum of ``k`` independent standard normals — the density k φ(x) Φ(x)^(k-1)
    integrated on a fine grid (the reference code takes the same integral by Gauss-Hermite quadrature). k = 1 is the
    normal itself (0, 1, 1); the variance shrinks as k grows: the best of many trials is a sharper statistic."""
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return 0.0, 1.0, 1.0
    x, log_phi, log_cdf = _max_grid()
    f = np.exp(math.log(k) + log_phi + (k - 1) * log_cdf)
    w = np.full(len(x), x[1] - x[0])
    w[0] = w[-1] = 0.5 * (x[1] - x[0])
    mass = float((w * f).sum())
    e1 = float((w * f * x).sum()) / mass
    e2 = float((w * f * x * x).sum()) / mass
    return e1, e2, e2 - e1 * e1


def sharpe_variance(sr: float, T: float, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
                    n_trials: int = 1) -> float:
    """The asymptotic variance of a Sharpe ratio estimated over ``T`` periods (the paper's equation):

        V[SR] = (a - b·γ3·SR + c·(γ4 - 1)/4·SR²) / T  ×  Var[max of K standard normals]

    with (a, b, c) = `ar1_terms(rho)` and K = ``n_trials`` (1: no selection, the factor is 1). ``nan`` when the
    bracket is not positive (a large positive skew with a high Sharpe can do that) or an input is not finite."""
    T = float(T)
    if not T > 0.0:
        raise ValueError(f"T must be positive, got {T}")
    sr, skew, kurt = float(sr), float(skew), float(kurt)
    if not all(math.isfinite(v) for v in (sr, skew, kurt, float(rho))):
        return float("nan")
    a, b, c = ar1_terms(rho)
    bracket = a - b * skew * sr + c * (kurt - 1.0) / 4.0 * sr * sr
    if not bracket > 0.0:
        return float("nan")
    v = bracket / T
    if int(n_trials) > 1:
        v *= max_normal_moments(int(n_trials))[2]
    return v


def sharpe_psr(sr: float, sr0: float, T: float, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
               n_trials: int = 1) -> float:
    """The paper's Probabilistic Sharpe Ratio: 1 - the p-value of H0: SR = sr0 against SR > sr0,

        PSR = Φ( (SR - SR0) / sqrt(V[SR0]) )

    the variance taken UNDER THE NULL (at sr0, over T periods) — where `probabilistic_sharpe` (Bailey & López de
    Prado 2012) takes it at the estimate with T - 1. With ``n_trials`` > 1, sr0 must already be the deflated
    threshold (`deflated_sharpe_k` builds it); only the variance is the maximum's here. ``nan`` when undefined."""
    v = sharpe_variance(sr0, T, skew, kurt, rho, n_trials)
    if not v > 0.0 or not math.isfinite(float(sr)):
        return float("nan")
    return _ND.cdf((float(sr) - float(sr0)) / math.sqrt(v))


def min_track_record(sr: float, sr0: float = 0.0, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
                     alpha: float = 0.05) -> float:
    """The paper's MINIMUM TRACK RECORD LENGTH, in periods: the T at which an observed Sharpe ``sr`` (per period)
    would reject H0: SR = sr0 at the one-sided level ``alpha``,

        MinTRL = V[SR0](T = 1) · ( z_{1-α} / (SR - SR0) )²

    (the T for a 50% power when the true Sharpe is ``sr``: `track_record_for_power` gives any other power). ``inf``
    when sr <= sr0 — no record makes a lower Sharpe significantly higher (the reference code squares the gap and
    returns a number there)."""
    sr, sr0 = float(sr), float(sr0)
    if not sr > sr0:
        return float("inf")
    v1 = sharpe_variance(sr0, 1.0, skew, kurt, rho)
    if not v1 > 0.0:
        return float("nan")
    return v1 * (_z(1.0 - float(alpha)) / (sr - sr0)) ** 2


def critical_sharpe(sr0: float, T: float, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
                    alpha: float = 0.05, n_trials: int = 1, lower: bool = False) -> float:
    """The critical Sharpe of the one-sided test of H0: SR = sr0 at level ``alpha`` over ``T`` periods:
    SR_c = SR0 + z_{1-α} sqrt(V[SR0]) (``lower``: SR0 - z_{1-α} sqrt(V[SR0]), the test against a SHORTFALL)."""
    v = sharpe_variance(sr0, T, skew, kurt, rho, n_trials)
    if not v > 0.0:
        return float("nan")
    step = _z(1.0 - float(alpha)) * math.sqrt(v)
    return float(sr0) - step if lower else float(sr0) + step


def sharpe_power(sr0: float, sr1: float, T: float, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
                 alpha: float = 0.05, n_trials: int = 1) -> float:
    """The POWER (1 - β) over ``T`` periods of the one-sided test of H0: SR = sr0 at level ``alpha``, against the
    alternative H1: SR = sr1 — upward when sr1 > sr0 (the paper's: 1 - Φ((SR_c - SR1) / sqrt(V[SR1]))), downward when
    sr1 < sr0 (the test that a record falls SHORT of sr0: Φ((SR_c,lower - SR1) / sqrt(V[SR1]))). ``n_trials`` applies
    to both variances (the reference code leaves it out of the critical value; the same for one trial)."""
    sr0, sr1 = float(sr0), float(sr1)
    lower = sr1 < sr0
    c = critical_sharpe(sr0, T, skew, kurt, rho, alpha, n_trials, lower=lower)
    v1 = sharpe_variance(sr1, T, skew, kurt, rho, n_trials)
    if not v1 > 0.0 or c != c:
        return float("nan")
    z = (c - sr1) / math.sqrt(v1)
    return _ND.cdf(z) if lower else 1.0 - _ND.cdf(z)


def track_record_for_power(sr0: float, sr1: float, power: float = 0.8, skew: float = 0.0, kurt: float = 3.0,
                           rho: float = 0.0, alpha: float = 0.05) -> float:
    """The periods T at which the one-sided level-``alpha`` test of H0: SR = sr0 reaches ``power`` against
    H1: SR = sr1 (either direction; `sharpe_power` at this T returns ``power``):

        T = ( (z_{1-α} sqrt(V[SR0](1)) + z_power sqrt(V[SR1](1))) / |SR1 - SR0| )²

    ``power`` 0.5 with sr1 the observed Sharpe is the paper's MinTRL. ``inf`` when sr1 == sr0."""
    sr0, sr1 = float(sr0), float(sr1)
    if sr1 == sr0:
        return float("inf")
    v0, v1 = sharpe_variance(sr0, 1.0, skew, kurt, rho), sharpe_variance(sr1, 1.0, skew, kurt, rho)
    if not (v0 > 0.0 and v1 > 0.0):
        return float("nan")
    num = _z(1.0 - float(alpha)) * math.sqrt(v0) + _z(float(power)) * math.sqrt(v1)
    return (num / abs(sr1 - sr0)) ** 2 if num > 0.0 else 0.0


def pfdr(p_h1: float, alpha: float, beta: float) -> float:
    """The PLANNED false discovery rate: P[H0 | the test rejects], for a test of level ``alpha`` and type II error
    ``beta`` (1 - its power against the alternative) when H1 is true with prior probability ``p_h1``:

        pFDR = 1 / (1 + (1 - β) · P[H1] / (α · P[H0]))

    It reads the DECISION RULE (its level and power), not the observed Sharpe — `ofdr` reads the observation."""
    p, a, b = float(p_h1), float(alpha), float(beta)
    if not (0.0 < p < 1.0 and 0.0 < a < 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(f"need 0 < p_h1 < 1, 0 < alpha < 1, 0 <= beta <= 1; got {p_h1}, {alpha}, {beta}")
    return 1.0 / (1.0 + (1.0 - b) * p / (a * (1.0 - p)))


def ofdr(sr: float, sr0: float, sr1: float, T: float, p_h1: float, skew: float = 0.0, kurt: float = 3.0,
         rho: float = 0.0, n_trials: int = 1) -> float:
    """The OBSERVED false discovery rate: P[H0 | a Sharpe at least ``sr``], H0: SR = sr0 against the point alternative
    H1: SR = sr1 with prior P[H1] = ``p_h1``:

        oFDR = p0 · P[H0] / (p0 · P[H0] + p1 · P[H1]),   p0 = 1 - PSR(SR; SR0),  p1 = 1 - PSR(SR; SR1)

    — the observed p-value weighed against the chance a true sr1 would show a Sharpe this high. The alternative is
    the caller's assumption and moves the answer most: say which one with every reading."""
    p = float(p_h1)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p_h1 must lie strictly between 0 and 1, got {p_h1}")
    p0 = 1.0 - sharpe_psr(sr, sr0, T, skew, kurt, rho, n_trials)
    p1 = 1.0 - sharpe_psr(sr, sr1, T, skew, kurt, rho, n_trials)
    if p0 != p0 or p1 != p1:
        return float("nan")
    den = p0 * (1.0 - p) + p1 * p
    return p0 * (1.0 - p) / den if den > 0.0 else float("nan")


def fdr_critical_sharpe(q: float, sr0: float = 0.0, sr1: float = 0.5, p_h1: float = 0.05, T: float = 24,
                        skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0, n_trials: int = 1) -> Dict[str, float]:
    """The critical Sharpe that CONTROLS THE FALSE DISCOVERY RATE at ``q`` (the paper's control_for_FDR): the c with
    P[H0 | SR > c] = q when SR ~ N(sr0, σ0²) under H0 and N(sr1, σ1²) under H1 (σ = sqrt(V[·]) over ``T`` periods,
    P[H1] = ``p_h1``), found by bisection on [-10, 10]. Returns {"sr_c", "alpha" (P[SR > c | H0]), "beta" (P[SR <= c |
    H1]), "power", "q_hat" (`pfdr` at that alpha and beta: q at the root), "sigma0", "sigma1"}; ``sr_c`` is -inf when
    even accepting every Sharpe keeps the FDR under q, nan when no c reaches q (as the reference code)."""
    q, p = float(q), float(p_h1)
    if not (sr0 < sr1 and 0.0 < q < 1.0 and 0.0 < p < 1.0):
        raise ValueError(f"need sr0 < sr1, 0 < q < 1, 0 < p_h1 < 1; got {sr0}, {sr1}, {q}, {p_h1}")
    s0 = math.sqrt(sharpe_variance(sr0, T, skew, kurt, rho, n_trials))
    s1 = math.sqrt(sharpe_variance(sr1, T, skew, kurt, rho, n_trials))
    odds = p / (1.0 - p)

    def f(c: float) -> float:                       # P[H0 | SR > c]
        lr = math.exp(min(700.0, _norm_logsf((c - sr1) / s1) - _norm_logsf((c - sr0) / s0)))
        return 1.0 / (1.0 + lr * odds)
    out = {"sigma0": s0, "sigma1": s1}
    lo, hi = -10.0, 10.0
    if f(lo) < q:
        return dict(out, sr_c=float("-inf"), alpha=1.0, beta=0.0, power=1.0, q_hat=f(lo))
    if (f(lo) - q) * (f(hi) - q) > 0.0:
        return dict(out, sr_c=float("nan"), alpha=float("nan"), beta=float("nan"), power=float("nan"),
                    q_hat=float("nan"))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) - q) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    c = 0.5 * (lo + hi)
    alpha = math.exp(_norm_logsf((c - sr0) / s0))
    beta = _ND.cdf((c - sr1) / s1)
    q_hat = 1.0 / (1.0 + (1.0 - beta) * odds / alpha) if alpha > 0.0 else float("nan")
    return dict(out, sr_c=c, alpha=alpha, beta=beta, power=1.0 - beta, q_hat=q_hat)


def deflated_sharpe_k(sr: float, n_trials: int, T: float, skew: float = 0.0, kurt: float = 3.0, rho: float = 0.0,
                      sr_var: Optional[float] = None, sr0: float = 0.0) -> Dict[str, float]:
    """The paper's DEFLATED (FWER) test of the best of ``n_trials`` trials: the threshold SR0 + E[max], E[max] =
    `expected_max_sharpe`(K, ``sr_var``), and the MAXIMUM's variance V[SR0 + E[max]] × Var[max of K normals]:

        DSR = Φ( (SR - SR0 - E[max]) / sqrt(V[SR0 + E[max]] · Var[M_K]) )

    ``sr_var`` — the variance of the trials' Sharpe ratios — is REQUIRED (as in `deflated_sharpe`: a silent default
    would deflate by an unstated amount). For trials whose Sharpe varies only by sampling noise around zero, pass
    ``sharpe_variance(0, T, rho=rho)`` = a / T (PREREG103 passed 1 / (T - 1)). Returns {"dsr", "threshold" (SR0 +
    E[max]), "e_max", "psr0" (the undeflated `sharpe_psr` against sr0), the inputs}."""
    if sr_var is None:
        raise ValueError("deflated_sharpe_k needs sr_var, the variance of the trials' Sharpe ratios (for skill-less "
                         "trials: sharpe_variance(0, T, rho=rho))")
    e_max = expected_max_sharpe(n_trials, sr_var)
    thr = float(sr0) + e_max
    return {"dsr": sharpe_psr(sr, thr, T, skew, kurt, rho, int(n_trials)), "threshold": thr, "e_max": e_max,
            "psr0": sharpe_psr(sr, sr0, T, skew, kurt, rho), "sr": float(sr), "sr0": float(sr0),
            "n_trials": int(n_trials), "T": float(T), "skew": float(skew), "kurt": float(kurt), "rho": float(rho),
            "sr_var": float(sr_var)}


def sharpe_moments(x: Sequence[float]) -> Dict[str, float]:
    """A return series' inputs to the formulas above: {"n", "mean", "sd" (ddof 1), "sr" (mean / sd), "skew" (the
    population third standardized moment), "kurt" (the plain fourth: 3 for a normal), "rho" (the lag-1 Pearson
    autocorrelation, the reference code's `autocorrelation`)}. Finite values only; ``nan`` where undefined (fewer than
    3 values, or a constant series)."""
    v = np.asarray([float(a) for a in x if a == a and math.isfinite(float(a))], float)
    n = int(len(v))
    out = {"n": n, "mean": float(v.mean()) if n else float("nan"), "sd": float("nan"), "sr": float("nan"),
           "skew": float("nan"), "kurt": float("nan"), "rho": float("nan")}
    if n < 3:
        return out
    sd = float(v.std(ddof=1))
    sd0 = float(v.std(ddof=0))
    if not sd > 0.0:
        return out
    z = (v - v.mean()) / sd0
    out.update(sd=sd, sr=float(v.mean()) / sd, skew=float((z ** 3).mean()), kurt=float((z ** 4).mean()))
    a, b = v[1:], v[:-1]
    if a.std() > 0.0 and b.std() > 0.0:
        out["rho"] = float(np.corrcoef(a, b)[0, 1])
    return out


# ── regressions with robust standard errors (a book's market exposure) ─────────

def _design(y, X, const: bool, names: Optional[Sequence[str]]):
    yv = np.asarray(y, float).ravel()
    Xv = np.asarray(X, float)
    if Xv.ndim == 1:
        Xv = Xv[:, None]
    if Xv.shape[0] != len(yv):
        raise ValueError(f"y has {len(yv)} rows, X {Xv.shape[0]}")
    if not (np.isfinite(yv).all() and np.isfinite(Xv).all()):
        raise ValueError("y and X must be finite — align and drop the missing rows first")
    nm = list(names) if names is not None else [f"x{i + 1}" for i in range(Xv.shape[1])]
    if len(nm) != Xv.shape[1]:
        raise ValueError(f"{len(nm)} names for {Xv.shape[1]} regressors")
    if const:
        Xv = np.column_stack([np.ones(len(yv)), Xv])
        nm = ["const"] + nm
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"{n} rows cannot fit {k} coefficients")
    XtX = Xv.T @ Xv
    if np.linalg.matrix_rank(XtX) < k:
        raise ValueError("the regressors are collinear")
    B = np.linalg.inv(XtX)
    coef = B @ (Xv.T @ yv)
    u = yv - Xv @ coef
    return yv, Xv, nm, n, k, B, coef, u


def _fit_out(yv, nm, n, k, coef, V, u, df, const) -> Dict[str, object]:
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    t = np.where(se > 0, coef / np.where(se > 0, se, 1.0), np.nan)
    tss = float(((yv - yv.mean()) ** 2).sum()) if const else float((yv ** 2).sum())
    return {"names": nm, "coef": [float(c) for c in coef], "se": [float(s) for s in se], "t": [float(v) for v in t],
            "p": [t_sf_two_sided(float(v), df) if v == v else float("nan") for v in t],
            "n": int(n), "k": int(k), "df": float(df),
            "r2": (1.0 - float(u @ u) / tss) if tss > 0 else float("nan"),
            "resid_sd": math.sqrt(float(u @ u) / (n - k))}


def ols_hac(y, X, lags: Optional[int] = None, const: bool = True,
            names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """OLS of ``y`` on ``X`` (a constant first unless ``const`` is False) with NEWEY-WEST standard errors: the
    Bartlett kernel, weights 1 - l / (L + 1), ``lags`` L by default floor(4 (n / 100)^(2/9)), the covariance scaled by
    n / (n - k); t on n - k degrees of freedom, two-sided p. Rows in time order. Returns {"names", "coef", "se", "t",
    "p", "n", "k", "df", "r2", "resid_sd", "lags"} — coefficient 0 the intercept (the ALPHA per period)."""
    yv, Xv, nm, n, k, B, coef, u = _design(y, X, const, names)
    L = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else int(lags)
    if L < 0:
        raise ValueError(f"lags must be >= 0, got {lags}")
    S_ = Xv * u[:, None]
    S = S_.T @ S_
    for lag in range(1, min(L, n - 1) + 1):
        G = S_[lag:].T @ S_[:-lag]
        S = S + (1.0 - lag / (L + 1.0)) * (G + G.T)
    V = B @ S @ B * (n / (n - k))
    out = _fit_out(yv, nm, n, k, coef, V, u, n - k, const)
    out["lags"] = L
    return out


def ols_cluster(y, X, clusters, const: bool = True, names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """OLS of ``y`` on ``X`` with CLUSTER-ROBUST standard errors (CR1: the sandwich over the clusters' score sums,
    scaled by G / (G - 1) · (n - 1) / (n - k)); t on G - 1 degrees of freedom. ``clusters``: one label per row (the
    bank's unit is the entry month: a month's trades share its market). Returns `ols_hac`'s keys with ``"clusters"``
    (G) in place of ``"lags"``."""
    yv, Xv, nm, n, k, B, coef, u = _design(y, X, const, names)
    lab = np.asarray(list(clusters), dtype=object)
    if len(lab) != n:
        raise ValueError(f"{len(lab)} cluster labels for {n} rows")
    _, inv = np.unique(lab.astype(str), return_inverse=True)
    G = int(inv.max()) + 1
    if G < 2:
        raise ValueError("at least two clusters are needed")
    Sg = np.zeros((G, k))
    np.add.at(Sg, inv, Xv * u[:, None])
    V = B @ (Sg.T @ Sg) @ B * (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    out = _fit_out(yv, nm, n, k, coef, V, u, G - 1, const)
    out["clusters"] = G
    return out
