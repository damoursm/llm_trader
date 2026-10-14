"""The overfitting statistics in `src/analysis/stats.py`: the Probability of Backtest Overfitting (CSCV) and the
Deflated Sharpe Ratio, ported from PREREG81's study script (`studies/prereg81/part2_81.py`).

Three duties: the ARITHMETIC against values computed independently in this file (the expected-maximum formula
written out, PREREG81's own `cscv` on the same seeded matrix — the golden pins below were produced by the study's
function and reproduced here to 1e-12), the BEHAVIOUR a statistician expects (noise ~0.5, a true edge ~0, more
trials deflate more, one trial deflates nothing), and the module's independence from scipy — the reason this file's
siblings exist at all (an optional scipy import guarding a verdict is an invisible off switch).
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from src.analysis import stats
from src.analysis.stats import (EULER_GAMMA, deflated_sharpe, expected_max_sharpe, pbo_cscv,
                                probabilistic_sharpe)

STATS_PATH = Path(stats.__file__)


def _noise(seed: int, T: int = 64, N: int = 20) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(T, N))


# ── PBO ───────────────────────────────────────────────────────────────────────

def test_pbo_on_pure_noise_is_near_one_half():
    """20 skill-less configurations: the in-sample winner's out-of-sample rank is uniform, so PBO ~ 0.5. One
    64 x 20 matrix is a noisy estimate (its 12,870 halvings reuse the same 16 groups), hence the seed and the
    band, plus the mean over 20 seeded matrices."""
    r = pbo_cscv(_noise(12))
    assert 0.35 < r["pbo"] < 0.65
    assert r["n_combinations"] == r["n_total_combinations"] == math.comb(16, 8) == 12870
    assert r["n_configs"] == 20 and r["n_periods"] == 64 and r["n_splits"] == 16
    mean20 = float(np.mean([pbo_cscv(_noise(s))["pbo"] for s in range(20)]))
    assert 0.40 < mean20 < 0.65


def test_pbo_near_zero_with_one_true_edge():
    """One configuration earning +1 per period over the same noise wins every in-sample half and ranks at the top
    out of sample: nothing to overfit."""
    for seed in (0, 1, 2):
        M = _noise(seed)
        M[:, 7] += 1.0
        r = pbo_cscv(M)
        assert r["pbo"] < 0.05
        assert r["selected_top"] == {7: 12870}
        assert r["mean_rel_rank"] > 0.9 and r["p_oos_loss"] == 0.0


def test_pbo_matches_prereg81_cscv_on_a_seeded_matrix():
    """Golden pins from PREREG81's `cscv` (block sums, rank 1 = worst, ties at their mean rank, w = rank/(N+1))
    on the seed-0 matrix: PBO, the mean relative rank and the share of halvings the winner loses out of sample."""
    r = pbo_cscv(_noise(0))
    assert r["pbo"] == pytest.approx(0.3519036519036519, abs=1e-12)
    assert r["mean_rel_rank"] == pytest.approx(0.5756465756465757, abs=1e-12)
    assert r["p_oos_loss"] == pytest.approx(0.4307692307692308, abs=1e-12)
    assert r["selected_top"] == {19: 5615, 7: 3383, 4: 1212}
    # the logit of the relative rank is bounded by the extreme ranks 1 and N: log(1/N) .. log(N)
    assert r["logits"]["min"] == pytest.approx(math.log(1 / 20), abs=1e-12)
    assert r["logits"]["max"] == pytest.approx(math.log(20), abs=1e-12)
    assert r["pbo"] == pytest.approx(0.3519036519036519, abs=1e-12)


def test_pbo_logits_are_the_relative_rank_logit_and_a_tie_is_not_below_the_median():
    """Identical configurations tie everywhere: every rank is the mean (N+1)/2, w = 0.5, every logit 0 — not
    below the median, so PBO is 0 (PREREG81's convention)."""
    r = pbo_cscv(np.ones((32, 5)))
    assert r["pbo"] == 0.0 and r["mean_rel_rank"] == 0.5
    assert r["logits"] == {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_pbo_mean_scorer_equals_sum_ranking_and_sharpe_matches_a_row_scorer():
    """The block-sum paths equal the same statistics computed on the rows of each half."""
    M = _noise(5, T=64, N=12)

    def mean_rows(rows):
        return rows.mean(0)

    def sharpe_rows(rows):
        return rows.mean(0) / rows.std(0, ddof=1)

    a, b = pbo_cscv(M, scorer="mean"), pbo_cscv(M, scorer=mean_rows)
    assert a["pbo"] == b["pbo"] and a["mean_rel_rank"] == pytest.approx(b["mean_rel_rank"], abs=1e-12)
    a, b = pbo_cscv(M, scorer="sharpe"), pbo_cscv(M, scorer=sharpe_rows)
    assert a["pbo"] == b["pbo"] and a["mean_rel_rank"] == pytest.approx(b["mean_rel_rank"], abs=1e-12)
    assert a["scorer"] == "sharpe" and b["scorer"] == "sharpe_rows"


def test_pbo_combination_cap_is_a_seeded_subset():
    M = _noise(1)
    r = pbo_cscv(M, max_combinations=500, seed=3)
    assert r["n_combinations"] == 500 and r["n_total_combinations"] == 12870
    assert r["pbo"] == pbo_cscv(M, max_combinations=500, seed=3)["pbo"]          # deterministic
    assert 0.0 <= r["pbo"] <= 1.0
    full = pbo_cscv(M, max_combinations=20000)                                    # a cap above the total: all
    assert full["n_combinations"] == 12870
    small = pbo_cscv(M, n_splits=4)                                               # C(4, 2) = 6 halvings
    assert small["n_combinations"] == 6


def test_pbo_rejects_what_it_cannot_judge():
    with pytest.raises(ValueError):
        pbo_cscv(_noise(0), n_splits=15)                                          # odd
    with pytest.raises(ValueError):
        pbo_cscv(_noise(0, T=10), n_splits=16)                                    # fewer periods than groups
    M = _noise(0)
    M[3, 4] = np.nan
    with pytest.raises(ValueError):
        pbo_cscv(M)                                                               # align first
    with pytest.raises(ValueError):
        pbo_cscv(_noise(0)[:, :1])                                                # one configuration
    with pytest.raises(ValueError):
        pbo_cscv(_noise(0), scorer="median")


# ── DSR / PSR ─────────────────────────────────────────────────────────────────

def test_dsr_with_one_trial_and_no_trial_variance_is_the_psr_against_zero():
    r = deflated_sharpe(0.1, n_trials=1, T=500, sr_var=0.0)
    assert r["sr0"] == 0.0
    assert r["dsr"] == r["psr0"] == probabilistic_sharpe(0.1, 0.0, 500)
    assert r["dsr"] == pytest.approx(NormalDist().cdf(0.1 * math.sqrt(499) / math.sqrt(1 + 0.5 * 0.01)), abs=1e-12)
    # one trial has nothing to select from whatever the trials' spread; no spread deflates nothing whatever n
    assert deflated_sharpe(0.1, n_trials=1, T=500, sr_var=0.5)["sr0"] == 0.0
    assert deflated_sharpe(0.1, n_trials=1000, T=500, sr_var=0.0)["dsr"] == r["dsr"]


def test_dsr_decreases_as_the_number_of_trials_grows():
    out = [deflated_sharpe(0.1, n, T=500, sr_var=0.01)["dsr"] for n in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(out, out[1:]))
    sr0 = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(sr0, sr0[1:]))
    assert out[0] > 0.98 and out[-1] < 0.001


def test_expected_max_sharpe_hand_checked_for_ten_trials():
    """SR0 = sqrt(V) ((1 - γ) Z⁻¹(1 - 1/n) + γ Z⁻¹(1 - 1/(n e))), n = 10, V = 1, written out here independently;
    Z⁻¹(0.9) = 1.2815515655 is the textbook 90th percentile."""
    nd = NormalDist()
    gamma = 0.5772156649015329
    z_n = nd.inv_cdf(1 - 1 / 10)
    z_ne = nd.inv_cdf(1 - 1 / (10 * math.e))
    assert z_n == pytest.approx(1.2815515655, abs=1e-9)
    by_hand = (1 - gamma) * z_n + gamma * z_ne
    assert expected_max_sharpe(10, 1.0) == pytest.approx(by_hand, abs=1e-12)
    assert expected_max_sharpe(10, 1.0) == pytest.approx(1.5745983013, abs=1e-9)
    assert expected_max_sharpe(10, 4.0) == pytest.approx(2 * by_hand, abs=1e-12)        # scales with sqrt(V)
    assert EULER_GAMMA == gamma
    r = deflated_sharpe(0.2, 10, T=250, skew=-0.5, kurt=6.0, sr_var=1.0)
    assert r["sr0"] == pytest.approx(by_hand, abs=1e-12)
    assert r == {"dsr": r["dsr"], "sr0": r["sr0"], "psr0": r["psr0"], "sr": 0.2, "n_trials": 10, "T": 250,
                 "skew": -0.5, "kurt": 6.0, "sr_var": 1.0}


def test_psr_formula_and_its_undefined_cases():
    """PSR = Φ((sr - b) sqrt(T - 1) / sqrt(1 - skew sr + (kurt - 1)/4 sr²)); fat tails and negative skew widen the
    estimator's variance and lower the probability."""
    sr, T = 0.08, 300
    normal = probabilistic_sharpe(sr, 0.0, T)
    expected = NormalDist().cdf(sr * math.sqrt(T - 1) / math.sqrt(1 - 0 + (3 - 1) / 4 * sr * sr))
    assert normal == pytest.approx(expected, abs=1e-12)
    assert probabilistic_sharpe(sr, 0.0, T, skew=-1.0, kurt=10.0) < normal
    assert probabilistic_sharpe(sr, sr, T) == pytest.approx(0.5, abs=1e-12)
    assert math.isnan(probabilistic_sharpe(sr, 0.0, 1))                           # one period: undefined
    assert math.isnan(probabilistic_sharpe(2.0, 0.0, 100, skew=5.0))              # variance term <= 0: no verdict
    assert math.isnan(deflated_sharpe(2.0, 10, 100, skew=5.0, sr_var=0.1)["dsr"])


def test_dsr_requires_the_trials_variance():
    """A silent default would deflate nothing while reading like a verdict."""
    with pytest.raises(ValueError):
        deflated_sharpe(0.1, 10, 500)
    with pytest.raises(ValueError):
        deflated_sharpe(0.1, 0, 500, sr_var=0.1)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1.0)


# ── no scipy ──────────────────────────────────────────────────────────────────

def test_stats_module_never_imports_scipy():
    """The module's docstring names scipy as the dependency it replaces, so the check is on the IMPORTS: no
    import statement (static, by AST — wherever it sits, including inside a function or a try) names scipy,
    and importing the module in a fresh interpreter does not load it."""
    source = STATS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not [m for m in imported if m.split(".")[0] == "scipy"], imported
    assert not re.search(r"^\s*(import|from)\s+scipy\b", source, flags=re.M)
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, src.analysis.stats; print('scipy' in sys.modules)"],
        cwd=str(STATS_PATH.resolve().parents[2]), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
