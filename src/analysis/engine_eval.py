"""Paired evaluation of the synthesis ENGINES (`engine_recommendations`).

`engine_shadow` records, per run, every engine's decision on the same signal
cross-section: the LIVE engine (the one the per-run A/B flip picked, whose
BUY/SELL calls ran the gate cascade and reached the broker) and the shadow
engine(s), whose decisions nobody acted on. This module reads that table
the way the house evaluation standard asks:

* UNPAIRED per-arm summary (calls, action mix, confidence shape, latency,
  rule-fill share) — descriptive, so a collapsed confidence grid or a
  runaway BUY rate is visible before any outcome is read.
* PAIRED head-to-head per arm pair on the ticker-runs BOTH answered with a
  model's own verdict (rule-filled rows excluded — a fill is not a decision).
  Only the DISAGREEMENT subset carries information: everything else is a
  shared call no engine can take credit for. The edge is the mean oriented
  pivot return of arm A's call minus arm B's on those rows, with a
  DAY-CLUSTERED t (one observation per signal date), split by which side arm
  A took — the same bar every decider surface here uses (t ≥ 2 AND same-sign
  halves, `.claude/skills/evaluate`).
* Per-run Spearman of the two arms' confidences on common tickers — the
  within-run rank Gate 1c consumes — and action agreement.

An arm is an ``engine:prompt_variant`` pair, because the two engines cannot
run the same prompt: the local model takes the COMPACT prompt (the full one
does not fit beside 8B weights in 8 GB of VRAM). ``deepseek:compact`` (via
`synthesis_shadow_extra`) is the arm that separates the engine effect from
the prompt effect; without it the local-vs-remote pair confounds the two.

CLI: ``python -m src.analysis.engine_eval [--days 14] [--horizon pv|1|5|10]
[--by-run] [--disagree N]``.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

from src.analysis.arm_eval import attach_forward_returns, _side
from src.analysis.sentiment_shadow import _spearman
from src.db import repo

_DIRECTIONAL = ("BUY", "SELL")


# ── load ──────────────────────────────────────────────────────────────────────

def load_engine_calls(days: int = 14) -> pd.DataFrame:
    """Every engine decision row from the last *days* days (empty when none)."""
    sql = """
        SELECT run_id, generated_at, signal_date, engine, model, prompt_variant,
               live, ticker, action, direction, confidence, time_horizon,
               snap_price, rule_filled, latency_s, n_signals, n_recs
        FROM engine_recommendations
        WHERE generated_at >= ?
        ORDER BY generated_at, engine, prompt_variant, ticker
    """
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).isoformat()
    try:
        df = repo.fetch_df(sql, [cutoff])
    except Exception as e:                        # table absent on an old DB
        logger.warning(f"[engine-eval] read failed: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["arm"] = df["engine"].astype(str) + ":" + df["prompt_variant"].astype(str)
    df["_side"] = df["action"].map(_side)
    return df


# ── statistics ────────────────────────────────────────────────────────────────

def day_clustered_t(values: Sequence[float], dates: Sequence) -> Dict[str, Optional[float]]:
    """Mean of per-DAY means and its t-stat — one observation per signal date,
    so a day on which the two arms disagreed about forty names counts once.
    NaN below 3 days (no verdict), never 0.0."""
    s = pd.DataFrame({"v": list(values), "d": list(dates)}).dropna()
    if s.empty:
        return {"mean": None, "t": None, "n_days": 0, "n_rows": 0}
    per_day = s.groupby("d")["v"].mean()
    n = int(len(per_day))
    mean = float(per_day.mean())
    if n < 3:
        return {"mean": mean, "t": float("nan"), "n_days": n, "n_rows": int(len(s))}
    sd = float(per_day.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return {"mean": mean, "t": t, "n_days": n, "n_rows": int(len(s))}


# ── unpaired: each arm on its own calls ───────────────────────────────────────

def engine_summary(df: pd.DataFrame, horizon: str = "pv") -> List[dict]:
    """One row per arm: volume, action mix, confidence shape, latency, and
    the mean oriented forward return of its DIRECTIONAL calls (settled rows)."""
    if df is None or df.empty:
        return []
    sfx = "pv" if str(horizon) == "pv" else f"{horizon}d"
    col = f"ret_{sfx}"
    rows: List[dict] = []
    for arm, g in df.groupby("arm"):
        own = g[~g["rule_filled"].astype(bool)]
        dirn = own[own["action"].isin(_DIRECTIONAL)]
        conf = pd.to_numeric(dirn["confidence"], errors="coerce").dropna()
        lat = pd.to_numeric(g.drop_duplicates("run_id")["latency_s"], errors="coerce").dropna()
        settled = dirn[dirn[col].notna()] if col in dirn.columns else dirn.iloc[0:0]
        rows.append({
            "arm": arm,
            "runs": int(g["run_id"].nunique()),
            "live_runs": int(g[g["live"].astype(bool)]["run_id"].nunique()),
            "calls": int(len(g)),
            "rule_filled_pct": round(100.0 * float(g["rule_filled"].astype(bool).mean()), 1),
            "buy_pct": round(100.0 * float((own["action"] == "BUY").mean()), 1) if len(own) else None,
            "sell_pct": round(100.0 * float((own["action"] == "SELL").mean()), 1) if len(own) else None,
            "conf_mean": round(float(conf.mean()), 3) if len(conf) else None,
            "conf_distinct": int(conf.round(4).nunique()) if len(conf) else 0,
            "conf_at_1": round(100.0 * float((conf >= 0.995).mean()), 1) if len(conf) else None,
            "latency_s_p50": round(float(lat.median()), 1) if len(lat) else None,
            "settled": int(len(settled)),
            f"ret_{sfx}": (round(float(settled[col].mean()), 3) if len(settled) else None),
        })
    return rows


# ── paired: head-to-head on the ticker-runs both arms answered ────────────────

def engine_pairs(df: pd.DataFrame, horizon: str = "pv") -> List[dict]:
    """Per arm pair: agreement, and the DISAGREEMENT-subset edge of A over B
    (mean oriented return of A's call minus B's, day-clustered t), split by
    the side A took. Rule-filled rows are excluded on either side."""
    if df is None or df.empty:
        return []
    sfx = "pv" if str(horizon) == "pv" else f"{horizon}d"
    col, fwd_col = f"ret_{sfx}", f"fwd_ret_{sfx}"
    if col not in df.columns:
        return []
    own = df[~df["rule_filled"].astype(bool)]
    arms = sorted(own["arm"].unique())
    rows: List[dict] = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            da = own[own["arm"] == a].drop_duplicates(["run_id", "ticker"]).set_index(["run_id", "ticker"])
            db = own[own["arm"] == b].drop_duplicates(["run_id", "ticker"]).set_index(["run_id", "ticker"])
            common = da.index.intersection(db.index)
            if len(common) == 0:
                continue
            da, db = da.loc[common], db.loc[common]
            same_action = (da["action"].values == db["action"].values)
            same_side = (da["_side"].values == db["_side"].values)
            ok = da[fwd_col].notna().values & db[fwd_col].notna().values
            dis = ok & ~same_side
            diff = pd.Series(da[col].values - db[col].values, index=range(len(common)))
            dates = pd.Series(da["signal_date"].values, index=range(len(common)))
            a_side = pd.Series(da["_side"].values, index=range(len(common)))
            b_side = pd.Series(db["_side"].values, index=range(len(common)))
            overall = day_clustered_t(diff[dis], dates[dis])
            # Split by the side A took (B took something else on these rows).
            a_buy = dis & (a_side.values > 0)
            a_sell = dis & (a_side.values < 0)
            a_none = dis & (a_side.values == 0)      # A passed, B acted
            t_buy = day_clustered_t(diff[a_buy], dates[a_buy])
            t_sell = day_clustered_t(diff[a_sell], dates[a_sell])
            t_none = day_clustered_t(diff[a_none], dates[a_none])
            # Per-run confidence rank agreement on common DIRECTIONAL calls.
            sp = []
            for rid, g in pd.DataFrame({
                "run": [k[0] for k in common],
                "ca": pd.to_numeric(da["confidence"].values, errors="coerce"),
                "cb": pd.to_numeric(db["confidence"].values, errors="coerce"),
                "both_dir": (a_side.values != 0) & (b_side.values != 0),
            }).groupby("run"):
                g = g[g["both_dir"]].dropna()
                if len(g) >= 3:
                    r = _spearman(g["ca"].tolist(), g["cb"].tolist())
                    if r is not None and not (isinstance(r, float) and math.isnan(r)):
                        sp.append(r)
            rows.append({
                "pair": f"{a} vs {b}", "a": a, "b": b,
                "common": int(len(common)),
                "action_agree_pct": round(100.0 * float(same_action.mean()), 1),
                "side_agree_pct": round(100.0 * float(same_side.mean()), 1),
                "disagree": int(dis.sum()),
                "a_ret": round(float(da[col].values[dis].mean()), 3) if dis.sum() else None,
                "b_ret": round(float(db[col].values[dis].mean()), 3) if dis.sum() else None,
                "edge": (round(overall["mean"], 3) if overall["mean"] is not None else None),
                "edge_t": (round(overall["t"], 2) if overall["t"] is not None
                           and not math.isnan(overall["t"]) else None),
                "edge_days": overall["n_days"],
                "edge_a_buy": (round(t_buy["mean"], 3) if t_buy["mean"] is not None else None),
                "edge_a_buy_t": (round(t_buy["t"], 2) if t_buy["t"] is not None
                                 and not math.isnan(t_buy["t"]) else None),
                "n_a_buy": t_buy["n_rows"],
                "edge_a_sell": (round(t_sell["mean"], 3) if t_sell["mean"] is not None else None),
                "edge_a_sell_t": (round(t_sell["t"], 2) if t_sell["t"] is not None
                                  and not math.isnan(t_sell["t"]) else None),
                "n_a_sell": t_sell["n_rows"],
                "edge_a_pass": (round(t_none["mean"], 3) if t_none["mean"] is not None else None),
                "n_a_pass": t_none["n_rows"],
                "conf_spearman_per_run": round(float(sum(sp) / len(sp)), 3) if sp else None,
                "conf_spearman_runs": len(sp),
            })
    return rows


def biggest_disagreements(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """The ticker-runs where the two most-populated arms took OPPOSITE sides,
    highest combined confidence first — the rows to read by hand."""
    if df is None or df.empty:
        return pd.DataFrame()
    own = df[~df["rule_filled"].astype(bool)]
    top = own["arm"].value_counts().index.tolist()[:2]
    if len(top) < 2:
        return pd.DataFrame()
    a, b = top
    da = own[own["arm"] == a].drop_duplicates(["run_id", "ticker"]).set_index(["run_id", "ticker"])
    db = own[own["arm"] == b].drop_duplicates(["run_id", "ticker"]).set_index(["run_id", "ticker"])
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return pd.DataFrame()
    da, db = da.loc[common], db.loc[common]
    opp = (da["_side"].values * db["_side"].values) < 0
    out = pd.DataFrame({
        "run_id": [k[0] for k in common], "ticker": [k[1] for k in common],
        f"{a}": da["action"].values, f"{a}_conf": da["confidence"].values,
        f"{b}": db["action"].values, f"{b}_conf": db["confidence"].values,
        "ret_pv": da["ret_pv"].values if "ret_pv" in da.columns else None,
    })[opp]
    if out.empty:
        return out
    out["_k"] = pd.to_numeric(out[f"{a}_conf"], errors="coerce").fillna(0) + \
        pd.to_numeric(out[f"{b}_conf"], errors="coerce").fillna(0)
    return out.sort_values("_k", ascending=False).drop(columns="_k").head(n)


def evaluate(days: int = 14, horizon: str = "pv") -> dict:
    df = load_engine_calls(days)
    if df.empty:
        return {"summary": [], "pairs": [], "n_rows": 0}
    df = attach_forward_returns(df)
    return {"summary": engine_summary(df, horizon), "pairs": engine_pairs(df, horizon),
            "n_rows": int(len(df)), "df": df}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--horizon", default="pv", help="pv (default) | 1 | 5 | 10")
    ap.add_argument("--by-run", action="store_true", help="per-run agreement table")
    ap.add_argument("--disagree", type=int, default=15, help="rows of opposite-side calls")
    args = ap.parse_args()

    res = evaluate(args.days, args.horizon)
    if not res["n_rows"]:
        print(f"No engine decisions in the last {args.days} day(s). "
              f"(enable_synthesis_shadow accrues them one run at a time.)")
        return 1
    df = res["df"]
    fmt = lambda v: f"{v:.3f}"
    print(f"SYNTHESIS ENGINES — {res['n_rows']} decision rows / "
          f"{df['run_id'].nunique()} runs / {df['ticker'].nunique()} tickers "
          f"(last {args.days}d, horizon {args.horizon})\n")
    print("PER ARM (unpaired — descriptive)")
    print(pd.DataFrame(res["summary"]).to_string(index=False, float_format=fmt))
    if res["pairs"]:
        print("\nPAIRED (ticker-runs both arms answered; edge = A minus B on the "
              "DISAGREEMENT rows, day-clustered t; house bar t >= 2 AND same-sign halves)")
        print(pd.DataFrame(res["pairs"]).to_string(index=False, float_format=fmt))
    else:
        print("\nNo ticker-run answered by two arms yet — the shadow engine has not "
              "delivered a decision on a run the live engine also answered.")

    if args.by_run:
        rows = []
        for rid, g in df.groupby("run_id"):
            live = g[g["live"].astype(bool)]["arm"].unique().tolist()
            rows.append({
                "run_id": rid, "live": ",".join(live) or "-",
                "arms": ",".join(sorted(g["arm"].unique())),
                "n": int(len(g)),
                "buy": int((g["action"] == "BUY").sum()),
                "sell": int((g["action"] == "SELL").sum()),
                "rule_filled": int(g["rule_filled"].astype(bool).sum()),
            })
        print("\nPER RUN\n" + pd.DataFrame(rows).to_string(index=False))

    if args.disagree:
        dis = biggest_disagreements(df, args.disagree)
        if not dis.empty:
            print("\nOPPOSITE-SIDE CALLS (highest joint confidence first)\n"
                  + dis.to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
