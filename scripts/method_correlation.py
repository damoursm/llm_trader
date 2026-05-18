"""Pairwise interaction diagnostic for the 10 aggregator methods.

Two questions this script answers from ``cache/trades.json``:

  1. Are any two methods saying the same thing?
     If two methods' entry-time scores are highly correlated across trades,
     the linear weighted sum in ``aggregator.py`` is double-counting them.
     Concretely: a redundant pair carries roughly twice the influence its
     individual weights suggest.

  2. Is any method's score uncorrelated with the actual outcome?
     A method whose score has no statistical link to ``return_pct`` is
     contributing pure noise to the combined signal — whatever weight it
     has is wasted.

What this script is NOT
───────────────────────
- Not a regression / ML model. It only inspects bivariate correlations.
- Not a substitute for proper walk-forward testing on more data.
- Not significance-tested rigorously — with n≈16, |r|≳0.5 is suggestive
  but ≲0.5 is statistically indistinguishable from zero.

Run
───
    python scripts/method_correlation.py            # full report
    python scripts/method_correlation.py --csv      # also write a CSV of the matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


TRADES_FILE = PROJECT_ROOT / "cache" / "trades.json"
ALL_METHODS = ("news", "tech", "insider", "put_call", "max_pain",
               "oi_skew", "vwap", "pattern", "momentum", "money_flow")

# Thresholds for flagging
REDUNDANT_PAIR = 0.70   # |r| >= this between methods → flag as double-counting risk
USELESS_METHOD = 0.10   # |r| <= this between method score and outcome → flag as noise
SUGGESTIVE = 0.50       # n=16 critical value for p<0.05 is ~0.5; below this is noise-floor


# ── Data loading ──────────────────────────────────────────────────────────────

def load_feature_frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y_return, y_win)`` for every closed trade with method_scores.

    X  — DataFrame, one column per method, one row per closed trade.
    y_return — Series of ``return_pct`` (continuous outcome).
    y_win    — Series of int {0, 1} = ``return_pct > 0``  (binary outcome).
    """
    trades = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
    rows: List[Dict] = []
    rets: List[float] = []
    for t in trades:
        if t.get("status") != "CLOSED":
            continue
        ms = t.get("method_scores")
        if not isinstance(ms, dict) or not ms:
            continue
        ret = t.get("return_pct")
        if ret is None:
            continue
        rows.append({m: float(ms.get(m, 0.0)) for m in ALL_METHODS})
        rets.append(float(ret))
    if not rows:
        print("No closed trades with method_scores found in cache/trades.json", file=sys.stderr)
        sys.exit(1)
    X = pd.DataFrame(rows, columns=list(ALL_METHODS))
    y_return = pd.Series(rets, name="return_pct")
    y_win = (y_return > 0).astype(int).rename("win")
    return X, y_return, y_win


# ── Correlation computations ──────────────────────────────────────────────────

def matrix_pearson(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr(method="pearson")


def matrix_spearman(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr(method="spearman")


def outcome_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-method Pearson + Spearman + p-values against ``y``.

    For methods whose score is zero on every trade (no view ever), correlation
    is undefined — we skip them with NaN rather than failing.
    """
    rows = []
    for m in X.columns:
        col = X[m]
        non_zero = (col != 0).sum()
        if col.std() == 0 or len(col) < 3:
            rows.append({
                "method": m, "non_zero_n": int(non_zero),
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
            })
            continue
        pr, pp = stats.pearsonr(col, y)
        sr, sp = stats.spearmanr(col, y)
        rows.append({
            "method": m, "non_zero_n": int(non_zero),
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        })
    return pd.DataFrame(rows).set_index("method")


def redundant_pairs(corr: pd.DataFrame, threshold: float) -> List[tuple]:
    """List of ``(method_a, method_b, |r|)`` with abs(r) >= threshold."""
    out = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if abs(r) >= threshold:
                out.append((a, b, float(r)))
    return sorted(out, key=lambda t: -abs(t[2]))


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt_matrix(corr: pd.DataFrame) -> str:
    """Compact aligned correlation matrix with NaN as '   .  '."""
    methods = list(corr.columns)
    short = {m: m[:6] for m in methods}
    out = []
    header = "        " + " ".join(f"{short[m]:>7}" for m in methods)
    out.append(header)
    for r in methods:
        row = [f"{short[r]:>7}"]
        for c in methods:
            v = corr.loc[r, c]
            if pd.isna(v):
                row.append(f"{'.':>7}")
            elif r == c:
                row.append(f"{'-':>7}")
            else:
                marker = "*" if abs(v) >= REDUNDANT_PAIR else " "
                row.append(f"{v:>+6.2f}{marker}")
        out.append(" ".join(row))
    return "\n".join(out)


def print_report(X: pd.DataFrame, y_ret: pd.Series, y_win: pd.Series, args) -> None:
    n = len(X)
    print("=" * 80)
    print(f"METHOD CORRELATION DIAGNOSTIC  —  n = {n} closed trades")
    print("=" * 80)
    print()
    if n < 30:
        print(f"!  Small-sample warning: with n={n}, the p<0.05 critical |r| is around 0.50.")
        print(f"   |r| below ~0.50 is statistically indistinguishable from zero.")
        print(f"   Treat results below as suggestive, not conclusive.")
        print()

    # ── (1) Per-method base stats ─────────────────────────────────────────────
    print(f"PER-METHOD STATS (across {n} closed trades)")
    print("-" * 80)
    print(f"  {'method':<14}{'non-zero n':>12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    for m in X.columns:
        c = X[m]
        nz = int((c != 0).sum())
        print(f"  {m:<14}{nz:>12}{c.mean():>+9.3f}{c.std():>9.3f}{c.min():>+9.3f}{c.max():>+9.3f}")
    print()

    # ── (2) Outcome correlations ─────────────────────────────────────────────
    out_p = outcome_correlations(X, y_ret)
    out_b = outcome_correlations(X, y_win.astype(float))
    print("METHOD vs OUTCOME  —  is each method's score predictive of return?")
    print("-" * 80)
    print(f"  {'method':<14}{'non-zero':>10}    "
          f"{'r_pearson(ret)':>16}{'p':>8}    "
          f"{'r_spearman(ret)':>17}{'p':>8}    "
          f"{'r(win/loss)':>13}")
    for m in X.columns:
        rp = out_p.loc[m]
        rb = out_b.loc[m]
        if pd.isna(rp["pearson_r"]):
            print(f"  {m:<14}{rp['non_zero_n']:>10}    {'  no variance':<33}")
            continue
        flag_useless = "  <-useless?" if abs(rp["pearson_r"]) <= USELESS_METHOD else ""
        flag_strong  = "  <-strong"   if abs(rp["pearson_r"]) >= SUGGESTIVE else ""
        print(
            f"  {m:<14}{rp['non_zero_n']:>10}    "
            f"{rp['pearson_r']:>+15.3f}{rp['pearson_p']:>8.3f}    "
            f"{rp['spearman_r']:>+16.3f}{rp['spearman_p']:>8.3f}    "
            f"{rb['pearson_r']:>+12.3f}"
            f"{flag_strong or flag_useless}"
        )
    print()

    # ── (3) Inter-method correlation matrix ──────────────────────────────────
    pearson = matrix_pearson(X)
    print("INTER-METHOD PEARSON CORRELATION (entry-time scores)")
    print(f"  '*' marks |r| >= {REDUNDANT_PAIR} — these pairs may be double-counted.")
    print("-" * 80)
    print(_fmt_matrix(pearson))
    print()

    spearman = matrix_spearman(X)
    print("INTER-METHOD SPEARMAN CORRELATION (rank-based; less outlier-sensitive)")
    print(f"  '*' marks |r| >= {REDUNDANT_PAIR}.")
    print("-" * 80)
    print(_fmt_matrix(spearman))
    print()

    # ── (4) Redundant pairs report ───────────────────────────────────────────
    pairs_p = redundant_pairs(pearson, REDUNDANT_PAIR)
    pairs_s = redundant_pairs(spearman, REDUNDANT_PAIR)
    pairs_suggestive = redundant_pairs(pearson, SUGGESTIVE)

    print(f"REDUNDANT-PAIR FLAGS")
    print("-" * 80)
    if pairs_p:
        print(f"  Pearson  |r| >= {REDUNDANT_PAIR}:")
        for a, b, r in pairs_p:
            print(f"    {a:<12} <-> {b:<12}  r = {r:+.3f}")
    else:
        print(f"  Pearson  |r| >= {REDUNDANT_PAIR}: (none)")
    if pairs_s:
        print(f"  Spearman |r| >= {REDUNDANT_PAIR}:")
        for a, b, r in pairs_s:
            print(f"    {a:<12} <-> {b:<12}  r = {r:+.3f}")
    else:
        print(f"  Spearman |r| >= {REDUNDANT_PAIR}: (none)")
    if pairs_suggestive and not pairs_p:
        print(f"  Pearson  |r| >= {SUGGESTIVE} (suggestive only, below conclusive threshold):")
        for a, b, r in pairs_suggestive:
            print(f"    {a:<12} <-> {b:<12}  r = {r:+.3f}")
    print()

    # ── (5) Verdict summary ──────────────────────────────────────────────────
    print("VERDICT")
    print("-" * 80)
    useless = [m for m in X.columns
               if not pd.isna(out_p.loc[m, 'pearson_r'])
               and abs(out_p.loc[m, 'pearson_r']) <= USELESS_METHOD]
    no_data = [m for m in X.columns if pd.isna(out_p.loc[m, 'pearson_r'])]
    strong = [m for m in X.columns
              if not pd.isna(out_p.loc[m, 'pearson_r'])
              and abs(out_p.loc[m, 'pearson_r']) >= SUGGESTIVE]

    if strong:
        print(f"  Methods with the strongest link to outcome (|r| >= {SUGGESTIVE}):")
        for m in strong:
            r = out_p.loc[m, "pearson_r"]
            print(f"    {m:<12} r = {r:+.3f}")
    else:
        print(f"  No method shows a strong link (|r| >= {SUGGESTIVE}) to return_pct at n={n}.")

    if useless:
        print(f"\n  Methods with near-zero outcome correlation (|r| <= {USELESS_METHOD}):")
        for m in useless:
            r = out_p.loc[m, "pearson_r"]
            print(f"    {m:<12} r = {r:+.3f}    <- weight may be wasted noise")

    if no_data:
        print(f"\n  Methods with no variance in scores (never fired): {', '.join(no_data)}")
        print(f"  These can't be evaluated yet — they need to actually produce signals first.")

    if not pairs_p and not pairs_s:
        print(f"\n  No redundant pairs at |r| >= {REDUNDANT_PAIR}. The linear sum is not")
        print(f"  obviously double-counting any two methods on the current sample.")
    print()

    # ── (6) Optional CSV output ──────────────────────────────────────────────
    if args.csv:
        out_dir = PROJECT_ROOT / "cache"
        pearson.to_csv(out_dir / "method_pearson.csv")
        spearman.to_csv(out_dir / "method_spearman.csv")
        out_p.to_csv(out_dir / "method_vs_return.csv")
        print(f"Wrote method_pearson.csv, method_spearman.csv, method_vs_return.csv to {out_dir}/")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Method correlation diagnostic.")
    parser.add_argument("--csv", action="store_true",
                        help="Also write the matrices to CSV in cache/")
    args = parser.parse_args()

    if not TRADES_FILE.exists():
        print(f"No trades file at {TRADES_FILE}", file=sys.stderr)
        return 1

    X, y_ret, y_win = load_feature_frame()
    print_report(X, y_ret, y_win, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
