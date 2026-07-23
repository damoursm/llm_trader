"""Confidence-formula component isolation — which multiplier actually helps?

``src/signals/aggregator.py::_score_ticker`` computes the final per-ticker
confidence as a chain of multipliers on a raw base:

    raw_confidence = min(1, |combined_score| / 0.5)
    confidence     = round(min(1, raw_confidence * coherence_factor * movement_factor
                                  * volume_factor * family_factor * tape_conf_factor), 2)

Each factor was added independently over time (coherence/movement/volume first,
family + tape 2026-07-19) on the belief that it improves the signal — but the
LIVE number only ever shows the fully-blended product, so there is no way to
tell whether any ONE factor is pulling its weight, is dead weight (~neutral,
no effect either way), or is actively hurting. This module isolates each
factor: for every scored ticker (the unbiased ``signals`` panel, not just the
gate-selected trade ledger), it reconstructs what confidence WOULD have been
under "raw alone" and under "raw × exactly one factor" (never cumulatively
stacked — that's a different, harder question this module doesn't answer),
then measures whether that variant's value actually predicts the
DIRECTION-ORIENTED forward return (``sign(combined_score) × forward_return``,
the same "would following this ticker's call have worked" convention used
throughout the signals panel — see ``horizon_edge.py``'s "edge").

Two views per variant, both over the unbiased panel:

* ``compute_component_ic`` — one row per variant, Spearman IC (+ its ICIR
  reliability) at each horizon, UNGATED (every row). IC is the headline "does
  this variant discriminate good calls from bad ones" number — a variant that
  is truly inert (multiplies by ~1 everywhere, or its spread is unrelated to
  outcomes) shows IC ≈ 0. Deliberately excludes win%/return here: those never
  depend on the variant's VALUE when ungated (only IC does) — see the
  function docstring.
* ``compute_component_bands`` — the same panel split into the Low
  (0.10–0.35) / Medium (0.35–0.65) / High (0.65+) conviction bands already
  used by ``tracker._eval_stats`` for per-method calibration, applied to each
  variant's OWN value. A well-behaved component shows win%/return rising
  Low → Medium → High; a flat or inverted spread means that multiplier isn't
  actually separating good trades from bad ones despite moving the number.

The six raw ingredients (``raw_confidence`` + the five factors) are persisted
verbatim on the ``signals`` table (2026-07-21, ``SIGNAL_CONFIDENCE_COMPONENT_
COLUMNS``) rather than re-derived from other stored fields — re-deriving
``coherence_factor`` etc. from the per-method score JSON would silently apply
TODAY's formula to a HISTORICAL row (which methods were active/filtered
changes over time), corrupting the very thing being measured. This means the
dataset is FORWARD-COLLECTED from the day this shipped — old signals rows
have NULL factor columns and are excluded; judge nothing until it accrues
(same caveat as every other signals-panel analysis in this codebase).

Two applications, sharing the same variant math:

* ``compute_entry_component_report`` — every scored ticker is a hypothetical
  ENTRY, oriented by its OWN ``combined_score`` sign. "If I'd opened today
  using variant X's confidence, was it well-calibrated?"
* ``compute_exit_component_report`` — signals-panel rows that fall inside an
  ALREADY-OPEN position's holding window (``entry_date < signal_date <=
  exit_date_or_today``), oriented by the TRADE's own direction (not the
  ticker's possibly-since-drifted current call), forward return measured
  FROM the re-read tick. "If I recomputed variant X mid-hold, did a
  deteriorating reading predict trouble ahead?" No new capture path is
  needed for this — held tickers stay in the scored universe every tick
  (``pipeline.py`` folds open-trade tickers into the universe), so the
  general signals panel already has continuous rows across a hold; this
  function just joins against the ``trades`` ledger's open interval.

Usage:  python -m src.analysis.confidence_components [--days 90] [--min-n 10]
                                                       [--source entry|exit]
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.signal_panel import _spearman, periodic_ic_stats, session_filter_mask

_DEFAULT_HORIZONS = (1, 5, 10)
MIN_N = 10

# (key, label, factor_column). factor_column=None means "raw" (no multiplier) for
# every entry EXCEPT "live", which is special-cased to read the stored `confidence`
# column directly (the actual product of ALL factors — the reference row).
VARIANTS: Sequence[tuple] = (
    ("raw",           "Raw score only",         None),
    ("raw_coherence", "Raw × Coherence",         "coherence_factor"),
    ("raw_movement",  "Raw × Movement",          "movement_factor"),
    ("raw_volume",    "Raw × Volume",            "volume_factor"),
    ("raw_family",    "Raw × Family agreement",  "family_conf_factor"),
    ("raw_tape",      "Raw × Tape confirmation", "tape_conf_factor"),
    ("live",          "Live (all combined)",     None),
)
_LIVE_KEY = "live"

# Same cut points as tracker._eval_stats' per-method conviction bands (documented
# in CLAUDE.md: Low 0.10–0.35 / Medium 0.35–0.65 / High 0.65+ on |score|) — reused
# here so "conviction band" means the same thing everywhere in this dashboard.
BANDS: Sequence[tuple] = (
    ("low",  "Low (0.10–0.35)",  0.10, 0.35),
    ("med",  "Medium (0.35–0.65)", 0.35, 0.65),
    ("high", "High (0.65+)",     0.65, None),
)


def _variant_values(df: pd.DataFrame) -> dict:
    """{variant_key: Series} — each variant's confidence value for every row.

    Raw × single-factor products are capped at 1.0 (``.clip(upper=...)``), the
    same ceiling the live formula applies to the FULL product — without it a
    single high-span factor (coherence up to 1.35) could push a variant above
    the [0,1] confidence scale the bands are calibrated to, giving it an unfair
    reach into the "High" band that the live formula would never allow."""
    raw = pd.to_numeric(df.get("raw_confidence"), errors="coerce")
    out: dict = {}
    for key, _label, col in VARIANTS:
        if key == _LIVE_KEY:
            out[key] = pd.to_numeric(df.get("confidence"), errors="coerce")
        elif col is None:
            out[key] = raw
        else:
            factor = pd.to_numeric(df.get(col), errors="coerce")
            out[key] = (raw * factor).clip(upper=1.0)
    return out


def has_component_data(df: Optional[pd.DataFrame]) -> bool:
    """Whether the panel actually carries the persisted factor columns (forward-
    collected 2026-07-21 — any row written before that has them NULL)."""
    if df is None or df.empty or "raw_confidence" not in df.columns:
        return False
    return bool(pd.to_numeric(df["raw_confidence"], errors="coerce").notna().any())


def compute_component_ic(df: pd.DataFrame, dir_sign: pd.Series,
                         horizons: Sequence[int] = _DEFAULT_HORIZONS,
                         min_n: int = MIN_N, min_per_day: int = 5, min_days: int = 3,
                         ) -> pd.DataFrame:
    """One row per variant: ``n``/``ic``/``icir`` per horizon, UNGATED (every row
    with a usable value). ``dir_sign`` (+1/-1, aligned to ``df``'s index)
    resolves the orientation — the ticker's own combined_score sign for the
    entry view, the held position's direction for the exit view.

    Deliberately does NOT report win%/mean-return here: ``oriented = fwd_ret ×
    dir_sign`` never depends on the variant's VALUE (only IC/ICIR — which
    correlate the variant against ``oriented`` — do), so an ungated win/return
    would be identical across every variant modulo missingness noise, i.e.
    redundant when the factor columns are fully populated and actively
    misleading when they aren't (a variant with more NaN rows would silently
    average over a different, not-comparable subset). The banded view
    (``compute_component_bands``) is where win%/return legitimately vary by
    variant, because band MEMBERSHIP is what the variant's value determines.

    Below ``min_n`` a cell is None, not a misleadingly noisy number."""
    values = _variant_values(df)
    rows = []
    for key, label, _col in VARIANTS:
        v = values[key]
        row: dict = {"variant": key, "label": label}
        for h in horizons:
            fwd = pd.to_numeric(df.get(f"fwd_ret_{h}d"), errors="coerce")
            oriented = fwd * dir_sign
            valid = v.notna() & oriented.notna() & dir_sign.notna()
            n = int(valid.sum())
            row[f"n_{h}d"] = n
            if n < min_n:
                row[f"ic_{h}d"] = None
                row[f"icir_{h}d"] = None
                continue
            vv, oo = v[valid], oriented[valid]
            ic = _spearman(vv, oo)
            row[f"ic_{h}d"] = round(ic, 4) if ic is not None else None
            if "signal_date" in df.columns:
                _, _, icir, _ = periodic_ic_stats(df.loc[valid, "signal_date"], vv, oo,
                                                  min_per_day, min_days)
                row[f"icir_{h}d"] = round(icir, 3) if icir is not None else None
            else:
                row[f"icir_{h}d"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def compute_component_bands(df: pd.DataFrame, dir_sign: pd.Series,
                            horizons: Sequence[int] = _DEFAULT_HORIZONS,
                            min_n: int = MIN_N) -> pd.DataFrame:
    """One row per (variant, band): ``n``/``win``/``ret`` per horizon, banding
    each variant's OWN value into Low/Medium/High (``BANDS``). A component that
    is actually discriminating shows win%/ret% rising Low → Medium → High; flat
    or inverted means that multiplier isn't separating good calls from bad ones
    even though it moves the confidence number."""
    values = _variant_values(df)
    rows = []
    for key, label, _col in VARIANTS:
        v = values[key]
        for bkey, blabel, lo, hi in BANDS:
            mask_band = v.notna() & (v >= lo) & (v < hi if hi is not None else True)
            row: dict = {"variant": key, "label": label, "band": bkey, "band_label": blabel}
            for h in horizons:
                fwd = pd.to_numeric(df.get(f"fwd_ret_{h}d"), errors="coerce")
                oriented = fwd * dir_sign
                valid = mask_band & oriented.notna() & dir_sign.notna()
                n = int(valid.sum())
                row[f"n_{h}d"] = n
                if n < min_n:
                    row[f"win_{h}d"] = None
                    row[f"ret_{h}d"] = None
                    continue
                oo = oriented[valid]
                row[f"win_{h}d"] = round(float((oo > 0).mean() * 100.0), 2)
                row[f"ret_{h}d"] = round(float(oo.mean()), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ── entry-side: every scored ticker, its own direction ──────────────────────

def compute_entry_component_report(days: Optional[int] = None,
                                   horizons: Sequence[int] = _DEFAULT_HORIZONS,
                                   min_n: int = MIN_N,
                                   signals_df: Optional[pd.DataFrame] = None) -> dict:
    """``{panel_rows, has_factors, ic, bands}`` — every scored ticker treated
    as a hypothetical entry in its own ``combined_score`` direction."""
    from src.analysis.signal_panel import build_panel
    panel = build_panel(horizons=horizons, days=days, signals_df=signals_df)
    if panel is None or panel.empty:
        return {"panel_rows": 0, "has_factors": False,
                "ic": pd.DataFrame(), "bands": pd.DataFrame()}
    if not has_component_data(panel):
        return {"panel_rows": int(len(panel)), "has_factors": False,
                "ic": pd.DataFrame(), "bands": pd.DataFrame()}
    cs = pd.to_numeric(panel.get("combined_score"), errors="coerce")
    dir_sign = pd.Series(np.where(cs > 0, 1.0, np.where(cs < 0, -1.0, np.nan)),
                         index=panel.index)
    return {
        "panel_rows": int(len(panel)),
        "has_factors": True,
        "ic": compute_component_ic(panel, dir_sign, horizons, min_n),
        "bands": compute_component_bands(panel, dir_sign, horizons, min_n),
    }


# ── exit-side: signals-panel rows inside an already-open position's hold ────

def _load_trade_intervals() -> pd.DataFrame:
    from src.db import repo
    try:
        return repo.fetch_df(
            "SELECT ticker, direction, entry_date, exit_date, status FROM trades")
    except Exception:
        return pd.DataFrame()


def restrict_to_held_intervals(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Panel rows that fall STRICTLY inside an already-open position's holding
    window: ``entry_date < signal_date <= exit_date`` (still-open trades run
    through today). This is a mid-hold RE-READ, not the entry itself — the
    entry day belongs to the entry-side report, not this one. Adds a
    ``_dir_sign`` column from the TRADE's own direction (``exit_panel.
    _dir_sign_of`` convention), not the ticker's current (possibly since
    reversed) signal direction — a held position's exit question is "does this
    variant predict what happens to the position I'm actually in", not "does it
    predict today's fresh call". No new capture path: held tickers stay in the
    scored universe every tick (pipeline.py folds open-trade tickers into
    discovery), so the general panel already has rows spanning the hold."""
    if trades is None or trades.empty or panel is None or panel.empty:
        return pd.DataFrame()
    from src.analysis.exit_panel import _dir_sign_of
    t = trades.copy()
    t["_entry"] = pd.to_datetime(t["entry_date"], errors="coerce")
    today = pd.Timestamp(date.today())
    t["_exit"] = pd.to_datetime(t["exit_date"], errors="coerce").fillna(today)
    t["_dir_sign"] = t["direction"].map(_dir_sign_of).astype(float)
    t = t.dropna(subset=["_entry", "ticker"])
    if t.empty:
        return pd.DataFrame()

    p = panel.copy()
    p["_sigd"] = pd.to_datetime(p["signal_date"], errors="coerce")

    merged = p.merge(t[["ticker", "_entry", "_exit", "_dir_sign"]], on="ticker", how="inner")
    held = merged[(merged["_sigd"] > merged["_entry"]) & (merged["_sigd"] <= merged["_exit"])]
    dedupe_keys = [c for c in ("run_id", "ticker", "signal_date") if c in held.columns]
    if dedupe_keys:
        held = held.drop_duplicates(subset=dedupe_keys)
    return held.drop(columns=["_sigd", "_entry", "_exit"], errors="ignore")


def compute_exit_component_report(days: Optional[int] = None,
                                  horizons: Sequence[int] = _DEFAULT_HORIZONS,
                                  min_n: int = MIN_N,
                                  session: Optional[str] = None,
                                  direction: Optional[str] = None,
                                  signals_df: Optional[pd.DataFrame] = None,
                                  trades_df: Optional[pd.DataFrame] = None) -> dict:
    """``{panel_rows, has_factors, ic, bands}`` — signals-panel rows re-read on
    an already-open position mid-hold, oriented by the position's direction.
    ``session`` filters to the US-market session the RE-READ was generated in
    (``rth|premarket|afterhours|overnight|extended``); ``direction``
    (``long|short``) to the held position's side."""
    from src.analysis.signal_panel import build_panel
    panel = build_panel(horizons=horizons, days=days, signals_df=signals_df)
    if panel is None or panel.empty or not has_component_data(panel):
        return {"panel_rows": 0, "has_factors": has_component_data(panel),
                "ic": pd.DataFrame(), "bands": pd.DataFrame()}
    trades = trades_df if trades_df is not None else _load_trade_intervals()
    held = restrict_to_held_intervals(panel, trades)
    if held.empty:
        return {"panel_rows": 0, "has_factors": True,
                "ic": pd.DataFrame(), "bands": pd.DataFrame()}
    if session and "generated_at" in held.columns:
        held = held[session_filter_mask(held["generated_at"], session)]
    if direction:
        want = 1.0 if str(direction).lower() in ("long", "buy") else -1.0
        held = held[held["_dir_sign"] == want]
    if held.empty:
        return {"panel_rows": 0, "has_factors": True,
                "ic": pd.DataFrame(), "bands": pd.DataFrame()}
    dir_sign = held["_dir_sign"]
    return {
        "panel_rows": int(len(held)),
        "has_factors": True,
        "ic": compute_component_ic(held, dir_sign, horizons, min_n),
        "bands": compute_component_bands(held, dir_sign, horizons, min_n),
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame, cols: Sequence[str]) -> None:
    present = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[present].to_string(index=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Confidence-formula component isolation over the signals panel.")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--min-n", type=int, default=MIN_N)
    p.add_argument("--source", choices=("entry", "exit"), default="entry")
    p.add_argument("--session", default=None)
    p.add_argument("--direction", default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    from src.db import repo
    repo.set_read_only(True)

    if args.source == "entry":
        rep = compute_entry_component_report(days=args.days, min_n=args.min_n)
    else:
        rep = compute_exit_component_report(days=args.days, min_n=args.min_n,
                                            session=args.session, direction=args.direction)
    if not rep.get("has_factors"):
        print("No confidence-component data yet — forward-collected from 2026-07-21; "
              "accrues every run once the persisted factor columns are non-null.")
        return
    ic, bands = rep["ic"], rep["bands"]
    if ic is None or ic.empty:
        print(f"{rep['panel_rows']} row(s) with factor data — not enough forward-return "
              "history for a report yet.")
        return
    print(f"\n{args.source.upper()} — {rep['panel_rows']} row(s)\n")
    print("Overall (ungated) — does the variant's value discriminate?")
    _print_table(ic, ["label"] + [f"{m}_{h}d" for h in _DEFAULT_HORIZONS
                                  for m in ("n", "ic", "icir")])
    print("\nBy conviction band:")
    _print_table(bands, ["label", "band_label"] + [f"{m}_{h}d" for h in _DEFAULT_HORIZONS
                                                    for m in ("n", "win", "ret")])


if __name__ == "__main__":
    main()
