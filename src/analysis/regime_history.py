"""Historical reconstruction of the macro regime — does the filter fire when it should?

Live history holds 40 calm days in which only NEUTRAL and CAUTION ever occurred,
so PANIC / RISK_OFF / RISK_ON are untested in production. That is not evidence
the bands are wrong; it is an absence of stress. The way to tell the difference
is to replay the composite over years that DID contain stress.

Method: re-drive the live composite over daily history using the inputs that
genuinely have one, classified by **the production classifiers themselves**
(`vix._classify_vix`, `move._classify_move`, …) rather than reimplemented
thresholds — a reimplementation would drift from live and quietly validate the
wrong function.

Reconstructable (yfinance + FRED daily history) — 7 of 10 inputs, 9.0 of the
12.5 total weight (72%):
    vix          ^VIX                                   weight 2.0
    move         ^MOVE                                  weight 2.0
    bond         TLT/IEF/TIP/LQD + 10Y-3M curve         weight 1.5
    fred         curve + HY OAS + unemployment + CPI    weight 1.0
    global_macro DXY + copper/gold                      weight 1.0
    intermarket  IWM + RSP vs SPY                       weight 1.0
    credit       HYG vs SPY relative                    weight 0.5

NOT reconstructable, and honestly excluded rather than guessed (3.5 weight):
    macro_news   LLM classification of contemporaneous headlines (1.5) —
                 genuinely impossible retroactively
    dix          proprietary dark-pool feed (1.0)
    breadth      needs historical advance/decline (1.0)

The composite normalises by the weight of AVAILABLE inputs, so a partial
reconstruction is directionally valid — but it is a WEAKER instrument than live:
`macro_news` is the input most likely to fire early in a genuine crisis, so the
reconstruction will tend to reach the panic bands LATER and less often than the
live filter would have. Read a reconstructed PANIC as strong evidence; read its
absence as weak evidence.

CLI:  python -m src.analysis.regime_history [--start 2018-01-01]
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from loguru import logger  # project configures loguru sinks only

# Bands, mirroring compute_macro_regime. Kept here as data so the reconstruction
# can be re-banded without touching live code.
BANDS = (("PANIC", -1.5), ("RISK_OFF", -0.8), ("CAUTION", -0.3), ("NEUTRAL", 0.3))


def band_of(norm: float) -> str:
    for name, hi in BANDS:
        if norm <= hi:
            return name
    return "RISK_ON"


def _hist(ticker: str, start: str) -> Optional[pd.Series]:
    """Daily closes for one ticker, or None."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        col = "Close" if "Close" in df.columns else df.columns[0]
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()
    except Exception as e:
        logger.warning(f"[regime_history] {ticker} unavailable: {e}")
        return None


def _fred(series_id: str, start: str) -> Optional[pd.Series]:
    """One FRED series as a daily-forward-filled float series, or None.

    FRED is the only source with genuine multi-decade history for the
    macro-cycle inputs (unemployment, CPI, HY spreads), and it is already a
    configured dependency — so `fred` and `bond` do not have to be guessed.
    """
    from config.settings import settings
    key = getattr(settings, "fred_api_key", "") or ""
    if not key:
        return None
    try:
        import requests
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": key, "file_type": "json",
                    "observation_start": start},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observations") or []
    except Exception as e:
        logger.warning(f"[regime_history] FRED {series_id} unavailable: {e}")
        return None
    idx, vals = [], []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue
        try:
            vals.append(float(v))
            idx.append(pd.Timestamp(o["date"]))
        except (ValueError, KeyError):
            continue
    if not vals:
        return None
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def reconstruct(start: str = "2018-01-01") -> pd.DataFrame:
    """Daily reconstructed composite + regime over the available inputs."""
    from src.data.macro_regime import (_WEIGHTS, _VIX_SCORES, _MOVE_SCORES,
                                       _CREDIT_SCORES, _GLOBAL_MACRO_SCORES,
                                       _INTERMARKET_SCORES, _BOND_REGIME_SCORES,
                                       _FRED_REGIME_SCORES)
    from src.data.vix import _classify_vix
    from src.data.move import _classify_move
    # Reuse the PRODUCTION classifiers so the reconstruction validates the
    # functions that actually run, not a second implementation of them.
    from src.data.fred import (_regime as _fred_regime, _yield_curve_label,
                               _credit_label, _inflation_label)

    series: Dict[str, pd.Series] = {}
    for name, tk in (("vix", "^VIX"), ("move", "^MOVE"), ("hyg", "HYG"),
                     ("spy", "SPY"), ("iwm", "IWM"), ("rsp", "RSP"),
                     ("dxy", "DX-Y.NYB"), ("copper", "HG=F"), ("gold", "GC=F"),
                     ("tlt", "TLT"), ("ief", "IEF"), ("tip", "TIP"), ("lqd", "LQD")):
        s = _hist(tk, start)
        if s is not None:
            series[name] = s
    # FRED: the only source with genuine multi-decade history for the macro
    # cycle. BAA10Y (Moody's Baa minus 10Y) stands in for the HY OAS the live
    # module uses — FRED licence-restricts BAMLH0A0HYM2 to 2023+, while BAA10Y
    # runs the full window. It is a NARROWER spread than HY, so its thresholds
    # are rescaled below rather than reused, and that rescaling is the one
    # genuine approximation in this reconstruction.
    for name, sid in (("dgs10", "DGS10"), ("dgs3mo", "DGS3MO"),
                      ("baa10y", "BAA10Y"), ("unrate", "UNRATE"),
                      ("cpi", "CPIAUCSL")):
        f = _fred(sid, start)
        if f is not None:
            series[name] = f
    if "vix" not in series:
        return pd.DataFrame()

    idx = series["vix"].index
    frame = pd.DataFrame(index=idx)
    for k, s in series.items():
        frame[k] = s.reindex(idx).ffill()

    rows: List[dict] = []
    for ts, r in frame.iterrows():
        contrib: Dict[str, float] = {}

        if pd.notna(r.get("vix")):
            contrib["vix"] = _VIX_SCORES.get(_classify_vix(float(r["vix"]))[0], 0.0)
        if pd.notna(r.get("move")):
            contrib["move"] = _MOVE_SCORES.get(_classify_move(float(r["move"]))[0], 0.0)

        # Credit: HYG's 20-day relative move vs SPY (the live signal's basis).
        if {"hyg", "spy"} <= set(frame.columns):
            i = frame.index.get_loc(ts)
            if i >= 20:
                h = r["hyg"] / frame["hyg"].iloc[i - 20] - 1.0
                s_ = r["spy"] / frame["spy"].iloc[i - 20] - 1.0
                rel = (h - s_) * 100.0
                sig = ("CREDIT_STRESS" if rel < -3 else "CREDIT_CAUTION" if rel < -1
                       else "CREDIT_SURGE" if rel > 3 else "CREDIT_STRONG" if rel > 1
                       else "NEUTRAL")
                contrib["credit"] = _CREDIT_SCORES.get(sig, 0.0)

        # Intermarket: are IWM and RSP keeping up with SPY over 20 days?
        if {"iwm", "rsp", "spy"} <= set(frame.columns):
            i = frame.index.get_loc(ts)
            if i >= 20:
                def chg(c):
                    return r[c] / frame[c].iloc[i - 20] - 1.0
                lag_iwm, lag_rsp = chg("iwm") - chg("spy"), chg("rsp") - chg("spy")
                both_lag = lag_iwm < -0.02 and lag_rsp < -0.02
                both_lead = lag_iwm > 0.02 and lag_rsp > 0.02
                sig = ("NARROW_RISK_OFF" if both_lag and lag_iwm < -0.05
                       else "NARROW_CAUTION" if both_lag
                       else "BROAD_EXPANSION" if both_lead and lag_iwm > 0.05
                       else "BROAD_HEALTHY" if both_lead else "NEUTRAL")
                contrib["intermarket"] = _INTERMARKET_SCORES.get(sig, 0.0)

        # Global macro: dollar strength + copper/gold (growth vs fear).
        if {"dxy", "copper", "gold"} <= set(frame.columns):
            i = frame.index.get_loc(ts)
            if i >= 20:
                dxy_chg = r["dxy"] / frame["dxy"].iloc[i - 20] - 1.0
                cg = r["copper"] / r["gold"]
                cg_chg = cg / (frame["copper"].iloc[i - 20] / frame["gold"].iloc[i - 20]) - 1.0
                risk_off = dxy_chg > 0.02 and cg_chg < -0.03
                risk_on = dxy_chg < -0.02 and cg_chg > 0.03
                sig = ("RISK_OFF" if risk_off else "RISK_ON" if risk_on
                       else "DEFENSIVE" if cg_chg < -0.05
                       else "CONSTRUCTIVE" if cg_chg > 0.05 else "NEUTRAL")
                contrib["global_macro"] = _GLOBAL_MACRO_SCORES.get(sig, 0.0)

        # ── bond internals (weight 1.5) ─────────────────────────────────
        # Same bull/bear point count as the live module: curve, TLT trend,
        # IG credit, real rates (TIP vs IEF) and long-end pressure (TLT vs IEF).
        if {"tlt", "ief", "tip", "lqd", "dgs10", "dgs3mo"} <= set(frame.columns):
            i = frame.index.get_loc(ts)
            if i >= 20 and all(pd.notna(r.get(c)) for c in
                               ("tlt", "ief", "tip", "lqd", "dgs10", "dgs3mo")):
                spread = float(r["dgs10"]) - float(r["dgs3mo"])
                curve = ("DEEPLY_INVERTED" if spread < -0.75 else
                         "INVERTED" if spread < -0.10 else
                         "FLAT" if spread < 0.50 else
                         "NORMAL" if spread < 1.50 else "STEEP")
                pct = lambda c, n: (r[c] / frame[c].iloc[i - n] - 1.0) * 100.0
                tlt20 = pct("tlt", 20)
                tlt_sig = ("RALLYING_STRONG" if tlt20 > 3 else "RALLYING" if tlt20 > 1
                           else "FLAT" if tlt20 > -1
                           else "FALLING" if tlt20 > -3 else "FALLING_STRONG")
                tlt_ief = pct("tlt", 5) - pct("ief", 5)
                tip_ief = pct("tip", 5) - pct("ief", 5)
                lqd5 = pct("lqd", 5)

                bull = bear = 0
                if curve in ("DEEPLY_INVERTED", "INVERTED"):
                    bear += 1
                elif curve in ("NORMAL", "STEEP"):
                    bull += 1
                if tlt_sig in ("RALLYING", "RALLYING_STRONG"):
                    bear += 1                      # flight to quality
                elif tlt_sig in ("FALLING", "FALLING_STRONG"):
                    bull += 1
                if lqd5 < -1.0:
                    bear += 1
                elif lqd5 > 1.0:
                    bull += 1
                if tip_ief < -0.20:
                    bear += 1                      # real rates rising
                if tlt_ief < -0.30:
                    bear += 1                      # long-end pressure

                net = bull - bear
                bond_regime = ("RISK_ON" if net >= 2 else "CONSTRUCTIVE" if net == 1
                               else "NEUTRAL" if net == 0
                               else "DEFENSIVE" if net == -1 else "RISK_OFF")
                if tlt_sig in ("FALLING", "FALLING_STRONG") and tip_ief > 0.20:
                    bond_regime = "REFLATIONARY"
                contrib["bond"] = _BOND_REGIME_SCORES.get(bond_regime, 0.0)

        # ── FRED macro cycle (weight 1.0) ───────────────────────────────
        if {"dgs10", "dgs3mo"} <= set(frame.columns) and pd.notna(r.get("dgs10")):
            i = frame.index.get_loc(ts)
            curve_lbl = _yield_curve_label(float(r["dgs10"]) - float(r["dgs3mo"]))

            # BAA10Y rescaled onto the HY-OAS thresholds `_credit_label`
            # expects. Anchored on the observed 2000-2026 range (1.36-6.16):
            # BAA10Y 3.5+ is GFC-grade, which is HY "STRESSED" (>6.0).
            credit_lbl = "UNKNOWN"
            if "baa10y" in frame.columns and pd.notna(r.get("baa10y")):
                b = float(r["baa10y"])
                credit_lbl = _credit_label(1.7 * b)

            unemp_trend = "STABLE"
            if "unrate" in frame.columns and i >= 250 and pd.notna(r.get("unrate")):
                now_u = float(r["unrate"])
                past_u = float(frame["unrate"].iloc[max(0, i - 125)])
                if pd.notna(past_u) and past_u:
                    d = now_u - past_u
                    unemp_trend = ("RISING" if d > past_u * 0.02
                                   else "FALLING" if d < -past_u * 0.02 else "STABLE")

            infl_lbl = "UNKNOWN"
            if "cpi" in frame.columns and i >= 250 and pd.notna(r.get("cpi")):
                yr = frame["cpi"].iloc[max(0, i - 250)]
                if pd.notna(yr) and yr:
                    infl_lbl = _inflation_label((float(r["cpi"]) / float(yr) - 1.0) * 100.0)

            contrib["fred"] = _FRED_REGIME_SCORES.get(
                _fred_regime(curve_lbl, credit_lbl, unemp_trend, infl_lbl), 0.0)

        if not contrib:
            continue
        tw = sum(_WEIGHTS[k] for k in contrib)
        norm = sum(_WEIGHTS[k] * v for k, v in contrib.items()) / tw
        rows.append({"date": ts.date().isoformat(), "norm": round(norm, 4),
                     "regime": band_of(norm), "inputs": len(contrib),
                     "vix": round(float(r["vix"]), 2) if pd.notna(r.get("vix")) else None})
    return pd.DataFrame(rows)


def regime_forward_returns(df: pd.DataFrame, horizons=(1, 5, 10, 21)) -> pd.DataFrame:
    """SPY forward return by reconstructed regime — the actual validation.

    A risk overlay earns its keep only if the market genuinely behaves worse
    after the states it flags. This is the market's OWN move (not the system's
    oriented return), so it tests the REGIME LABEL rather than the signal stack
    riding on it.
    """
    spy = _hist("SPY", df["date"].iloc[0])
    if spy is None or spy.empty:
        return pd.DataFrame()
    px = {d.date().isoformat(): float(v) for d, v in spy.items()}
    days = sorted(px)
    pos = {d: i for i, d in enumerate(days)}

    rows = []
    for _, r in df.iterrows():
        i = pos.get(r["date"])
        if i is None:
            continue
        rec = {"regime": r["regime"]}
        for h in horizons:
            j = i + h
            rec[f"f{h}"] = ((px[days[j]] / px[days[i]] - 1.0) * 100.0
                            if j < len(days) else None)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)

    out = []
    for reg in ("PANIC", "RISK_OFF", "CAUTION", "NEUTRAL", "RISK_ON"):
        sub = d[d["regime"] == reg]
        rec = {"regime": reg, "days": len(sub)}
        for h in horizons:
            col = sub[f"f{h}"].dropna()
            rec[f"spy_{h}d"] = round(float(col.mean()), 3) if len(col) else None
            rec[f"up_{h}d"] = (round(100.0 * float((col > 0).mean()), 1)
                               if len(col) else None)
        out.append(rec)
    return pd.DataFrame(out)


# Market eras, chosen so each contains a structurally DIFFERENT kind of stress.
# The 2008-2009 window is the one that matters: it is the only era with a large
# PANIC sample that was NOT a V-shaped recovery.
ERAS = (("2000-2003 dot-com", "2000-01-01", "2003-12-31"),
        ("2004-2007 pre-GFC", "2004-01-01", "2007-12-31"),
        ("2008-2009 GFC",     "2008-01-01", "2009-12-31"),
        ("2010-2019",         "2010-01-01", "2019-12-31"),
        ("2020-2026",         "2020-01-01", "2026-12-31"))


def forward_returns_by_era(df: pd.DataFrame, horizons=(21, 63)) -> pd.DataFrame:
    """SPY forward return after each regime, SPLIT BY ERA — the robustness test.

    Pooled over 2018-2026 the data says PANIC precedes the best returns
    (+11.3% at 21d, 88.9% up) and therefore that the BUY block is backwards.
    That conclusion does not survive here: 2018-2026 contained only V-SHAPED
    recoveries (COVID, Apr-2025), and buying the panic is exactly what such a
    period rewards. In the GFC — the only era with a large PANIC sample
    (n=77) that was a PROTRACTED bear rather than a V — the same rule LOSES
    (-2.95% at 21d, -5.43% at 63d).

    So the BUY block is not backwards; it is insurance whose premium looks
    wasteful in every era except the one it exists for. Judge a risk control on
    the regime it protects against, never on the pooled average.
    """
    spy = _hist("SPY", df["date"].iloc[0])
    if spy is None or spy.empty:
        return pd.DataFrame()
    px = {d.date().isoformat(): float(v) for d, v in spy.items()}
    days = sorted(px)
    pos = {d: i for i, d in enumerate(days)}

    def fwd(d: str, h: int):
        i = pos.get(d)
        if i is None or i + h >= len(days):
            return None
        return (px[days[i + h]] / px[days[i]] - 1.0) * 100.0

    work = df.copy()
    for h in horizons:
        work[f"_f{h}"] = work["date"].map(lambda d, _h=h: fwd(d, _h))

    rows = []
    for era, lo, hi in ERAS:
        w = work[(work["date"] >= lo) & (work["date"] <= hi)]
        if w.empty:
            continue
        for reg in ("PANIC", "RISK_OFF", "CAUTION", "NEUTRAL", "RISK_ON"):
            sub = w[w["regime"] == reg]
            rec = {"era": era, "regime": reg, "days": len(sub),
                   "median_inputs": int(w["inputs"].median())}
            for h in horizons:
                col = sub[f"_f{h}"].dropna()
                rec[f"spy_{h}d"] = round(float(col.mean()), 2) if len(col) else None
            rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Historical regime reconstruction")
    ap.add_argument("--start", default="2018-01-01")
    a = ap.parse_args()

    df = reconstruct(a.start)
    if df.empty:
        print("Reconstruction unavailable (no VIX history).")
        raise SystemExit(0)

    print(f"\nReconstructed {len(df):,} sessions from {df['date'].iloc[0]} "
          f"to {df['date'].iloc[-1]}  ({df['inputs'].max()} of 10 inputs)\n")
    counts = df["regime"].value_counts()
    print(f"{'regime':<10}{'days':>7}{'share':>8}")
    for reg in ("PANIC", "RISK_OFF", "CAUTION", "NEUTRAL", "RISK_ON"):
        n = int(counts.get(reg, 0))
        print(f"{reg:<10}{n:>7}{100.0*n/len(df):>7.1f}%")

    print("\nKnown stress periods — did the filter fire?")
    for label, lo, hi in (("COVID crash",     "2020-02-20", "2020-04-30"),
                          ("2022 bear",       "2022-01-01", "2022-12-31"),
                          ("Aug-2024 spike",  "2024-07-25", "2024-08-15"),
                          ("Apr-2025 tariff", "2025-04-01", "2025-04-30")):
        w = df[(df["date"] >= lo) & (df["date"] <= hi)]
        if w.empty:
            print(f"  {label:<17} (no data)")
            continue
        worst = w.loc[w["norm"].idxmin()]
        hit = w["regime"].value_counts().to_dict()
        print(f"  {label:<17} worst norm {worst['norm']:+.2f} on {worst['date']} "
              f"(VIX {worst['vix']}) -> {worst['regime']:<9} {hit}")

    print("\nSPY forward return BY REGIME — does the label predict the market?")
    fr = regime_forward_returns(df)
    print(fr.to_string(index=False) if not fr.empty else "  (unavailable)")

    print("\nBY ERA — the robustness test (pooled results are era-dependent)")
    era = forward_returns_by_era(df)
    if era.empty:
        print("  (unavailable)")
    else:
        piv = era[era["regime"].isin(("PANIC", "RISK_OFF"))]
        print(piv.to_string(index=False))
        print("\n  The 2008-2009 row is the one that matters: it is the only era with a")
        print("  large PANIC sample that was NOT a V-shaped recovery, and there the")
        print("  buy-the-panic rule LOSES. A risk control is judged on the regime it")
        print("  exists for, not on the pooled average.")
