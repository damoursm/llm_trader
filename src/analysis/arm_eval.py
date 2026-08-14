"""Per-arm evaluation of the synthesis prompt bake-off.

Reads ``arm_recommendations`` — every prompt arm's call on every ticker each
tick, one of them live and the rest shadow (see ``arm_shadow``) — joins forward
returns from the OHLCV cache, and answers two different questions:

**Unpaired** (``arm_summary``): how did each arm's calls do overall? Easy to
read, and the WEAKEST evidence here — it is the shape of comparison that made
Qwen look best and pro-thinking look broken in the 2026-07-22 bake-off, when
both were pure calendar-overlap artifacts. It is reported because a large
divergence is worth seeing, not because it settles anything.

**Paired** (``arm_pairs``): on the ticker-days where two arms BOTH answered —
which, thanks to shadow arms, is now nearly all of them — how often did they
agree, and on the subset where they DISAGREED, which one was right? The
disagreement subset is the only place an arm can add or destroy value, so its
sample size is the real sample size of this experiment. Two arms agreeing on
90% of tickers means 90% of the raw comparison is noise about a shared decision.

Strategy convention: a HOLD/WATCH earns 0 (no position taken), so an arm that
declines to trade a loser genuinely beats one that takes it. This matters
because the dual-case arm is explicitly instructed that declining is a valid
output — scoring only its directional calls would hide exactly what it was
built to do.

CLI:  python -m src.analysis.arm_eval [--horizons pv,1,5,10] [--days 60]
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.analysis.signal_panel import _close_series, _spearman
from src.db import repo

from loguru import logger  # project configures loguru sinks only

ARMS = ("dual", "blind", "sighted")
ARM_LABEL = {"dual": "Dual-case", "blind": "Blind", "sighted": "Sighted"}

_DIRECTIONAL = {"BUY": 1.0, "SELL": -1.0}


def _side(action: Optional[str]) -> float:
    """+1 long, -1 short, 0 for HOLD/WATCH (no position ⇒ no P&L)."""
    return _DIRECTIONAL.get(str(action or "").upper(), 0.0)


def load_arm_calls(days: Optional[int] = None) -> pd.DataFrame:
    """Every persisted arm call, optionally limited to the last ``days``."""
    where = ""
    params: List = []
    if days:
        where = "WHERE signal_date >= (CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR"
        params = [int(days)]
    try:
        df = repo.fetch_df(
            f"""SELECT signal_date, generated_at, arm, live, ticker,
                       action, direction, confidence, snap_price
                FROM arm_recommendations {where}""",
            params,
        )
    except Exception as exc:
        logger.debug(f"[arm_eval] no arm_recommendations table yet: {exc}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # One call per (day, arm, ticker) — the last of the day wins, matching the
    # signals panel's dedupe so both surfaces describe the same decision.
    df = (df.sort_values("generated_at")
            .groupby(["signal_date", "arm", "ticker"], as_index=False).tail(1))
    return df


def attach_forward_returns(df: pd.DataFrame,
                           horizons: Sequence[int] = (1, 5, 10)) -> pd.DataFrame:
    """Add ``fwd_ret_<h>d`` (%) and the oriented ``ret_<h>d`` per row.

    Forward returns are read from the OHLCV cache at signal_date + h SESSIONS
    (not calendar days), the same anchor the signals panel uses, so an arm's
    numbers are directly comparable with every other panel on the dashboard.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["_sig_date"] = df["signal_date"].map(
        lambda d: date.fromisoformat(str(d)[:10]))

    closes_by_ticker: Dict[str, dict] = {}
    for tk in df["ticker"].unique():
        try:
            closes_by_ticker[tk] = _close_series(tk) or {}
        except Exception:
            closes_by_ticker[tk] = {}
    dates_by_ticker = {tk: sorted(c.keys()) for tk, c in closes_by_ticker.items()}

    def fwd(row, h: int) -> Optional[float]:
        dates = dates_by_ticker.get(row["ticker"]) or []
        closes = closes_by_ticker.get(row["ticker"]) or {}
        i = bisect_left(dates, row["_sig_date"])
        if i >= len(dates) or i + h >= len(dates):
            return None
        base = closes[dates[i]]
        if not base or base <= 0:
            return None
        return (closes[dates[i + h]] / base - 1.0) * 100.0

    df["_side"] = df["action"].map(_side)
    for h in horizons:
        df[f"fwd_ret_{h}d"] = df.apply(lambda r: fwd(r, h), axis=1)
        # Oriented as a STRATEGY return: a declined call earns 0, not NaN.
        df[f"ret_{h}d"] = df["_side"] * df[f"fwd_ret_{h}d"]
    # The pivot pseudo-horizon (2026-08-13 standardization): the signed move to
    # the next H/L pivot from the call's session close — the decision basis,
    # comparable across every eval surface. Settled rows only (None otherwise).
    try:
        from bisect import bisect_left as _bl
        from src.analysis.simulated_trades import _pivot_targets
        pv_cache: Dict[str, tuple] = {}

        def fwd_pv(row) -> Optional[float]:
            tk = row["ticker"]
            if tk not in pv_cache:
                pv_cache[tk] = _pivot_targets(dates_by_ticker.get(tk) or [],
                                              closes_by_ticker.get(tk) or {}, ticker=tk)
            pdts, sp, endx = pv_cache[tk]
            if not pdts:
                return None
            i = _bl(pdts, row["_sig_date"])
            if i < len(pdts) and endx[i] >= 0:
                return float(sp[i])
            return None

        df["fwd_ret_pv"] = df.apply(fwd_pv, axis=1)
        df["ret_pv"] = df["_side"] * df["fwd_ret_pv"]
    except Exception:
        df["fwd_ret_pv"] = None
        df["ret_pv"] = None
    return df.drop(columns=["_sig_date"])


# ── unpaired: each arm on its own calls ────────────────────────────────────

def arm_summary(df: pd.DataFrame, horizon="pv") -> List[dict]:
    """Per-arm action mix and realized outcome. Weak evidence — see module doc.
    ``horizon="pv"`` (default, 2026-08-13) judges on the signed pivot target;
    an int keeps the fixed-horizon view."""
    if df is None or df.empty:
        return []
    sfx = "pv" if horizon == "pv" else f"{horizon}d"
    col, fwd_col = f"ret_{sfx}", f"fwd_ret_{sfx}"
    if col not in df.columns:
        return []
    rows: List[dict] = []
    for arm in ARMS:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        scored = sub[sub[fwd_col].notna()]
        directional = scored[scored["_side"] != 0]
        n = len(sub)
        rows.append({
            "arm": arm,
            "label": ARM_LABEL[arm],
            "calls": n,
            "buy_pct": round(100.0 * (sub["_side"] > 0).sum() / n, 1),
            "sell_pct": round(100.0 * (sub["_side"] < 0).sum() / n, 1),
            "flat_pct": round(100.0 * (sub["_side"] == 0).sum() / n, 1),
            "scored": len(scored),
            # Strategy return: declines included at 0.
            "mean_ret": (round(float(scored[col].mean()), 3)
                         if not scored.empty else None),
            # Directional-only: how good were the calls it DID make.
            "dir_calls": len(directional),
            "dir_ret": (round(float(directional[col].mean()), 3)
                        if not directional.empty else None),
            "dir_win": (round(100.0 * float((directional[col] > 0).mean()), 1)
                        if not directional.empty else None),
            # Does the arm's own confidence rank its outcomes?
            "conf_ic": (_spearman(directional["confidence"], directional[col])
                        if len(directional) >= 8 else None),
        })
    return rows


# ── paired: arm vs arm on the same ticker-day ──────────────────────────────

def arm_pairs(df: pd.DataFrame, horizon="pv") -> List[dict]:
    """Head-to-head on ticker-days both arms answered.

    The ``disagree`` subset is where the prompt actually changed a decision —
    everything else is a shared call that no arm can take credit for. Read
    ``a_ret``/``b_ret`` on that subset, not the overall means.
    """
    if df is None or df.empty:
        return []
    sfx = "pv" if horizon == "pv" else f"{horizon}d"
    col, fwd_col = f"ret_{sfx}", f"fwd_ret_{sfx}"
    if col not in df.columns:
        return []
    rows: List[dict] = []
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            da = df[df["arm"] == a].set_index(["signal_date", "ticker"])
            db = df[df["arm"] == b].set_index(["signal_date", "ticker"])
            common = da.index.intersection(db.index)
            if len(common) == 0:
                continue
            da, db = da.loc[common], db.loc[common]
            same = (da["_side"].values == db["_side"].values)
            n = len(common)
            # Restrict outcome stats to rows whose forward return exists for
            # BOTH (identical ticker-day, so it is one return either way).
            ok = da[fwd_col].notna().values & db[fwd_col].notna().values
            dis = ok & ~same
            rows.append({
                "pair": f"{ARM_LABEL[a]} vs {ARM_LABEL[b]}",
                "a": a, "b": b,
                "common": n,
                "agree_pct": round(100.0 * float(same.mean()), 1),
                "disagree": int(dis.sum()),
                "a_ret": (round(float(da[col].values[dis].mean()), 3)
                          if dis.sum() else None),
                "b_ret": (round(float(db[col].values[dis].mean()), 3)
                          if dis.sum() else None),
                "edge": (round(float(da[col].values[dis].mean()
                                     - db[col].values[dis].mean()), 3)
                         if dis.sum() else None),
            })
    return rows


def evaluate(days: Optional[int] = None,
             horizons: Sequence = ("pv", 1, 5, 10)) -> dict:
    """Everything the dashboard needs, in one pass over the arm table.
    ``"pv"`` (the pivot pseudo-horizon, first = the decision basis) rides the
    same summary/pairs machinery as the fixed horizons."""
    df = load_arm_calls(days)
    if df.empty:
        return {"summary": {}, "pairs": {}, "horizons": list(horizons), "calls": 0}
    df = attach_forward_returns(df, [h for h in horizons if h != "pv"] or [5])
    return {
        "summary": {h: arm_summary(df, h) for h in horizons},
        "pairs": {h: arm_pairs(df, h) for h in horizons},
        "horizons": list(horizons),
        "calls": len(df),
        "shadow": int((~df["live"].astype(bool)).sum()) if "live" in df else 0,
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Synthesis prompt-arm bake-off")
    ap.add_argument("--horizons", default="pv,1,5,10")
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args()
    hs = [(x.strip() if x.strip() == "pv" else int(x))
          for x in a.horizons.split(",") if x.strip()]

    res = evaluate(days=a.days, horizons=hs)
    if not res["calls"]:
        print("No arm_recommendations rows yet — shadow arms populate this "
              "from the next tick onward.")
        raise SystemExit(0)

    print(f"\n{res['calls']} arm calls ({res['shadow']} shadow)\n")
    for h in hs:
        print(f"── {h}-day horizon ─────────────────────────────────────────")
        print(f"{'arm':<10} {'calls':>6} {'buy%':>6} {'sell%':>6} {'flat%':>6} "
              f"{'strat%':>8} {'dir%':>8} {'win%':>6} {'confIC':>7}")
        for r in res["summary"].get(h, []):
            def _f(v, w=8, p=3):
                return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
            print(f"{r['label']:<10} {r['calls']:>6} {r['buy_pct']:>6.1f} "
                  f"{r['sell_pct']:>6.1f} {r['flat_pct']:>6.1f} "
                  f"{_f(r['mean_ret'])} {_f(r['dir_ret'])} "
                  f"{_f(r['dir_win'], 6, 1)} {_f(r['conf_ic'], 7, 3)}")
        print()
        for p in res["pairs"].get(h, []):
            print(f"  {p['pair']:<26} common={p['common']:>5} "
                  f"agree={p['agree_pct']:>5.1f}%  disagree={p['disagree']:>4}"
                  + (f"  {p['a']}={p['a_ret']:+.3f}% {p['b']}={p['b_ret']:+.3f}% "
                     f"edge={p['edge']:+.3f}%" if p['disagree'] else ""))
        print()
