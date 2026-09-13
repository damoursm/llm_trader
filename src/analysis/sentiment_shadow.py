"""Per-ticker comparison of the two sentiment engines.

Every tick, each ticker the primary engine scores with an LLM is ALSO scored by
the other engine on the SAME article digest (``sentiment._submit_shadow``, table
``sentiment_shadow``). This module reads those paired rows and answers the two
questions the pairing exists for:

  DIRECTION — how often do the engines agree on the SIGN, and when they disagree,
  which way (one abstains, or they take opposite sides)?
  VALUE     — how far apart are the magnitudes, and do they ORDER tickers the
  same way (per-run Spearman over the tickers both engines scored)?

What this is NOT: a skill verdict. Agreement says the two models read the same
news the same way, not that either read is right — that is per-day Spearman IC
against the signed pivot target over accrued panel rows
(``.claude/skills/evaluate/SKILL.md``). Use this to decide whether the engines
are INTERCHANGEABLE (a routing/cost/latency decision) and to catch an engine
drifting into abstention or into the tails.

The comparison is on the RAW verdict (``*_raw``), not the scaled score: both
sides carry the identical evidence-mass and source-diversity scaler, so the
scaled pair differs only by the raw verdict and dividing it out keeps the model
difference clean. The scaled columns ride along for the record.

CLI::

    python -m src.analysis.sentiment_shadow [--days 7] [--by-run] [--disagree 20]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Optional

import pandas as pd
from loguru import logger

from src.db import repo


ABSTAIN = 1e-9          # a raw verdict of exactly 0.0 is an ABSTENTION, not a read


def load_shadow(days: int = 7) -> pd.DataFrame:
    """Paired verdicts from the last *days* days (empty frame when none)."""
    sql = """
        SELECT run_id, generated_at, ticker, digest_hash, n_articles,
               primary_engine, primary_model, primary_raw, primary_score, primary_catalyst,
               shadow_engine, shadow_model, shadow_raw, shadow_score, shadow_catalyst,
               shadow_latency_s
        FROM sentiment_shadow
        WHERE generated_at >= ?
        ORDER BY generated_at, ticker
    """
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).isoformat()
    try:
        df = repo.fetch_df(sql, [cutoff])
    except Exception as e:                    # table absent on an old DB
        logger.warning(f"[sentiment-shadow] read failed: {e}")
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _orient(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel primary/shadow into fixed ENGINE columns.

    The per-run flip means the same engine is 'primary' on some runs and 'shadow'
    on others; comparing the raw column names would silently mix them. Rows are
    re-keyed onto the two engine names present, alphabetically, so 'a' and 'b'
    mean the same engine on every row.
    """
    if df.empty:
        return df
    engines = sorted(set(df["primary_engine"].dropna()) | set(df["shadow_engine"].dropna()))
    if len(engines) != 2:
        logger.warning(f"[sentiment-shadow] expected 2 engines, found {engines}")
        if not engines:
            return pd.DataFrame()
    a, b = (engines + [None, None])[:2]
    is_a_primary = df["primary_engine"] == a
    out = df.copy()
    out["engine_a"], out["engine_b"] = a, b
    for col in ("raw", "score", "catalyst"):
        out[f"a_{col}"] = df[f"primary_{col}"].where(is_a_primary, df[f"shadow_{col}"])
        out[f"b_{col}"] = df[f"shadow_{col}"].where(is_a_primary, df[f"primary_{col}"])
    return out


def _sign(x: float) -> int:
    if x is None or pd.isna(x) or abs(float(x)) <= ABSTAIN:
        return 0
    return 1 if float(x) > 0 else -1


def _spearman(xs, ys) -> Optional[float]:
    """Average-rank Spearman (dependency-free, per src/analysis/stats.py rule)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def compare(df: pd.DataFrame) -> dict:
    """Direction + value agreement over the paired rows."""
    d = _orient(df)
    if d.empty:
        return {}
    a, b = d["engine_a"].iloc[0], d["engine_b"].iloc[0]
    sa = d["a_raw"].map(_sign)
    sb = d["b_raw"].map(_sign)
    both = (sa != 0) & (sb != 0)
    n = len(d)
    stats = {
        "engine_a": a, "engine_b": b, "n_pairs": n,
        "n_runs": d["run_id"].nunique(), "n_tickers": d["ticker"].nunique(),
        f"abstain_{a}": float((sa == 0).mean()),
        f"abstain_{b}": float((sb == 0).mean()),
        "both_scored": float(both.mean()),
        "sign_agree_both_scored": float((sa[both] == sb[both]).mean()) if both.any() else None,
        "opposite_sides": float((sa[both] != sb[both]).mean()) if both.any() else None,
        "one_abstains": float(((sa == 0) ^ (sb == 0)).mean()),
        f"mean_abs_{a}": float(d["a_raw"].abs().mean()),
        f"mean_abs_{b}": float(d["b_raw"].abs().mean()),
        "mean_abs_diff_both_scored": float((d["a_raw"][both] - d["b_raw"][both]).abs().mean())
        if both.any() else None,
        "spearman_pooled": _spearman(d["a_raw"][both].tolist(), d["b_raw"][both].tolist())
        if both.sum() >= 3 else None,
    }
    # Per-RUN Spearman: the combine consumes each method's WITHIN-RUN rank, so
    # whether the engines order the cross-section alike is the question that
    # actually decides whether swapping one for the other changes any decision.
    per_run = []
    for rid, g in d[both].groupby("run_id"):
        r = _spearman(g["a_raw"].tolist(), g["b_raw"].tolist())
        if r is not None:
            per_run.append(r)
    if per_run:
        stats["spearman_per_run_mean"] = sum(per_run) / len(per_run)
        stats["spearman_per_run_n"] = len(per_run)
    # Catalyst-class agreement (both engines emit the same fixed taxonomy).
    cat = d[d["a_catalyst"].notna() & d["b_catalyst"].notna()]
    if len(cat):
        stats["catalyst_agree"] = float((cat["a_catalyst"] == cat["b_catalyst"]).mean())
    return stats


def biggest_disagreements(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The pairs where the two engines differ most (opposite signs first)."""
    d = _orient(df)
    if d.empty:
        return d
    d = d.assign(
        sign_a=d["a_raw"].map(_sign), sign_b=d["b_raw"].map(_sign),
        gap=(d["a_raw"] - d["b_raw"]).abs(),
    )
    d["opposed"] = ((d["sign_a"] * d["sign_b"]) < 0).astype(int)
    cols = ["generated_at", "ticker", "n_articles", "a_raw", "b_raw", "gap",
            "a_catalyst", "b_catalyst"]
    return d.sort_values(["opposed", "gap"], ascending=False).head(n)[cols]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--by-run", action="store_true", help="per-run agreement table")
    ap.add_argument("--disagree", type=int, default=15, help="rows of worst disagreement")
    args = ap.parse_args()

    df = load_shadow(args.days)
    if df.empty:
        print(f"No paired sentiment verdicts in the last {args.days} day(s). "
              f"(enable_sentiment_shadow accrues them one run at a time.)")
        return 1
    st = compare(df)
    a, b = st["engine_a"], st["engine_b"]
    print(f"PAIRED SENTIMENT — {a} vs {b}   "
          f"{st['n_pairs']} pairs / {st['n_runs']} runs / {st['n_tickers']} tickers "
          f"(last {args.days}d)\n")
    print(f"  abstention (raw == 0.0)      {a}: {st[f'abstain_{a}']:6.1%}   "
          f"{b}: {st[f'abstain_{b}']:6.1%}")
    print(f"  both engines took a view     {st['both_scored']:6.1%}"
          f"   (exactly one abstained: {st['one_abstains']:.1%})")
    if st.get("sign_agree_both_scored") is not None:
        print(f"  DIRECTION agreement          {st['sign_agree_both_scored']:6.1%}"
              f"   (opposite sides: {st['opposite_sides']:.1%})")
    print(f"  mean |raw|                   {a}: {st[f'mean_abs_{a}']:6.3f}   "
          f"{b}: {st[f'mean_abs_{b}']:6.3f}")
    if st.get("mean_abs_diff_both_scored") is not None:
        print(f"  mean |difference|            {st['mean_abs_diff_both_scored']:6.3f}")
    if st.get("spearman_pooled") is not None:
        print(f"  Spearman (pooled)            {st['spearman_pooled']:+6.3f}")
    if st.get("spearman_per_run_mean") is not None:
        print(f"  Spearman (per run, mean)     {st['spearman_per_run_mean']:+6.3f}"
              f"   over {st['spearman_per_run_n']} run(s)  <- the rank the combine consumes")
    if st.get("catalyst_agree") is not None:
        print(f"  catalyst class agreement     {st['catalyst_agree']:6.1%}")

    if args.by_run:
        d = _orient(df)
        rows = []
        for rid, g in d.groupby("run_id"):
            sa, sb = g["a_raw"].map(_sign), g["b_raw"].map(_sign)
            both = (sa != 0) & (sb != 0)
            rows.append({
                "run_id": rid, "n": len(g), "primary": g["primary_engine"].iloc[0],
                f"abst_{a}": (sa == 0).mean(), f"abst_{b}": (sb == 0).mean(),
                "sign_agree": (sa[both] == sb[both]).mean() if both.any() else float("nan"),
                "spearman": _spearman(g["a_raw"][both].tolist(), g["b_raw"][both].tolist())
                if both.sum() >= 3 else float("nan"),
            })
        print("\nPER RUN\n" + pd.DataFrame(rows).to_string(index=False,
                                                           float_format=lambda v: f"{v:.3f}"))

    if args.disagree:
        dis = biggest_disagreements(df, args.disagree)
        if not dis.empty:
            print(f"\nBIGGEST DISAGREEMENTS ({a} = a_raw, {b} = b_raw)\n"
                  + dis.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
