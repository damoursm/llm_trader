"""Trade return & signal score bucketed by stock PRICE and DOLLAR VOLUME (2026-07-07).

Do penny / thin-volume names perform and score differently from pricier / liquid
ones? (The question behind widening the discovery filter to < $1 / < $5M.) Two
datasets, both bucketed on interpretable FIXED bands aligned to the discovery-gate
thresholds so the penny slice is legible:

  • realized TRADE returns from the ledger (small, selection-biased) — the
    strategy's actual P&L, by the trade's entry price and the stock's
    as-of-entry 20d dollar volume;
  • combined_score over the unbiased SIGNALS panel (large sample) — how model
    conviction varies across the price / volume grid, with the panel's mean 5-day
    forward return alongside (the unbiased performance view the tiny ledger can't
    give).

Dollar volume is attached as-of the signal/entry date via predictability's causal
feature panel (no look-ahead). Fail-soft: empty buckets when there's no data yet.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Sequence

import pandas as pd

# Fixed, interpretable bands. Price bands straddle the $1 / $5 discovery floors;
# dollar-volume bands straddle the $5M / $20M floors (old vs new).
PRICE_BANDS = [(0, 1, "<$1"), (1, 5, "$1–5"), (5, 20, "$5–20"),
               (20, 50, "$20–50"), (50, 200, "$50–200"), (200, float("inf"), "$200+")]
DVOL_BANDS = [(0, 5, "<$5M"), (5, 20, "$5–20M"), (20, 100, "$20–100M"),
              (100, 1000, "$100M–1B"), (1000, float("inf"), "$1B+")]


def _band(value, bands) -> Optional[str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:                      # NaN
        return None
    for lo, hi, label in bands:
        if lo <= v < hi:
            return label
    return None


def _iso_date(s) -> Optional[str]:
    s = str(s or "")[:10]
    try:
        _dt.date.fromisoformat(s)
        return s
    except ValueError:
        return None


def _bucket_stats(df: pd.DataFrame, value_col: str, band_col: str, bands) -> List[dict]:
    """One row per band (in band order, empty bands included as n=0) with the
    mean of ``value_col`` over the rows in that band."""
    out = []
    for _lo, _hi, label in bands:
        vals = pd.Series(dtype=float)
        if len(df) and band_col in df.columns:
            sub = df[df[band_col] == label]
            if len(sub):
                vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        out.append({"band": label, "n": int(len(vals)),
                    "mean": (round(float(vals.mean()), 4) if len(vals) else None)})
    return out


def trade_return_by_price_volume(trades: Sequence[dict]) -> dict:
    """Avg realized ``return_pct`` by PRICE band and by DOLLAR-VOLUME band, from
    the trade ledger (closed + open at their live M2M)."""
    rows = []
    for t in trades or []:
        try:
            price = float(t.get("entry_price"))
        except (TypeError, ValueError):
            continue
        ret = t.get("return_pct")
        tk = t.get("ticker")
        ed = _iso_date(t.get("entry_date") or t.get("entry_datetime"))
        if not (tk and ed) or ret is None or not (price and price > 0):
            continue
        rows.append({"ticker": tk, "signal_date": ed,
                     "entry_price": price, "return_pct": float(ret)})
    if not rows:
        return {"by_price": _bucket_stats(pd.DataFrame(), "", "price_band", PRICE_BANDS),
                "by_dvol": _bucket_stats(pd.DataFrame(), "", "dvol_band", DVOL_BANDS),
                "n_trades": 0, "n_with_dvol": 0}
    df = pd.DataFrame(rows)
    try:                                    # attach as-of-entry 20d dollar volume
        from src.analysis.predictability import build_feature_panel
        df = build_feature_panel(df)
    except Exception:
        df["dollar_vol"] = None
    df["price_band"] = df["entry_price"].map(lambda p: _band(p, PRICE_BANDS))
    dvol = df["dollar_vol"] if "dollar_vol" in df.columns else pd.Series([None] * len(df))
    df["dvol_band"] = dvol.map(lambda v: _band(v, DVOL_BANDS))
    dfv = df.dropna(subset=["dvol_band"])
    return {
        "by_price": _bucket_stats(df, "return_pct", "price_band", PRICE_BANDS),
        "by_dvol": _bucket_stats(dfv, "return_pct", "dvol_band", DVOL_BANDS),
        "n_trades": int(len(df)),
        "n_with_dvol": int(len(dfv)),
    }


def score_by_price_volume(days: Optional[int] = None,
                          horizons: Sequence[int] = (1, 5, 10)) -> dict:
    """Avg ``combined_score`` by PRICE / DOLLAR-VOLUME band over the signals panel,
    plus the mean 5-day forward return per band (unbiased performance view)."""
    empty = {"by_price": _bucket_stats(pd.DataFrame(), "", "price_band", PRICE_BANDS),
             "by_dvol": _bucket_stats(pd.DataFrame(), "", "dvol_band", DVOL_BANDS),
             "fwd_by_price": [], "fwd_by_dvol": [], "n_rows": 0, "fwd_col": None}
    try:
        from src.analysis.signal_panel import build_panel
        from src.analysis.predictability import build_feature_panel
        panel = build_panel(horizons=horizons, days=days)
    except Exception:
        return empty
    if panel is None or panel.empty:
        return empty
    fp = build_feature_panel(panel)         # adds dollar_vol
    fp["price_band"] = pd.to_numeric(fp.get("price"), errors="coerce").map(
        lambda p: _band(p, PRICE_BANDS))
    dvol = fp["dollar_vol"] if "dollar_vol" in fp.columns else pd.Series([None] * len(fp))
    fp["dvol_band"] = dvol.map(lambda v: _band(v, DVOL_BANDS))
    fpp = fp.dropna(subset=["price_band"])
    fpv = fp.dropna(subset=["dvol_band"])
    fwd_col = "fwd_ret_5d" if "fwd_ret_5d" in fp.columns else next(
        (c for c in fp.columns if c.startswith("fwd_ret_")), None)
    res = {
        "by_price": _bucket_stats(fpp, "combined_score", "price_band", PRICE_BANDS),
        "by_dvol": _bucket_stats(fpv, "combined_score", "dvol_band", DVOL_BANDS),
        "n_rows": int(len(fp)), "fwd_col": fwd_col,
        "fwd_by_price": (_bucket_stats(fpp.dropna(subset=[fwd_col]), fwd_col, "price_band", PRICE_BANDS)
                         if fwd_col else []),
        "fwd_by_dvol": (_bucket_stats(fpv.dropna(subset=[fwd_col]), fwd_col, "dvol_band", DVOL_BANDS)
                        if fwd_col else []),
    }
    return res


if __name__ == "__main__":       # quick manual check: python -m src.analysis.price_volume_perf
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from src.db import repo
    repo.set_read_only(True)
    tr = trade_return_by_price_volume(repo.load_trades())
    print(f"TRADES: n={tr['n_trades']} (with dvol {tr['n_with_dvol']})")
    print("  return by price :", [(b['band'], b['n'], b['mean']) for b in tr['by_price']])
    print("  return by dvol  :", [(b['band'], b['n'], b['mean']) for b in tr['by_dvol']])
    sc = score_by_price_volume()
    print(f"SIGNALS: n_rows={sc['n_rows']}  fwd_col={sc['fwd_col']}")
    print("  score by price  :", [(b['band'], b['n'], b['mean']) for b in sc['by_price']])
    print("  score by dvol   :", [(b['band'], b['n'], b['mean']) for b in sc['by_dvol']])
    print("  fwd5d by price  :", [(b['band'], b['n'], b['mean']) for b in sc['fwd_by_price']])
    print("  fwd5d by dvol   :", [(b['band'], b['n'], b['mean']) for b in sc['fwd_by_dvol']])
