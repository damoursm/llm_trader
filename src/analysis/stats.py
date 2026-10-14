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
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["binom_p_less", "binom_p_greater", "welch_t_test", "t_sf_two_sided",
           "pbo_cscv", "deflated_sharpe", "probabilistic_sharpe", "expected_max_sharpe",
           "EULER_GAMMA"]


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
