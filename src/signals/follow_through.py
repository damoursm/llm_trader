"""FOLLOW-THROUGH — the exit-as-entry mechanism (2026-08-25, user directive).

When a (real or hypothetical) held position's exit score is "too good to pass"
— an EXTREME within-tick rank of the ml_exit model's hold-conviction — the tape
has followed through against that position, and the measured edge is in JOINING
the move: entering the OPPOSITE direction in the SAME tick.

Validated offline 2026-08-24/25 (walk-forward fold models, live H/L pivot
labels, Gate-4, sides split — scratchpad reversal_candidates / follow_through_*
series; record in memory/ml-methods-plan-2026-07.md):
  - spec tail: within-day bottom-5% of the FULL exit score, score < −0.50,
    first-day-of-episode → +2.62% pivot potential per entry (t +4.05), both
    sides working (the momentum-control falsification passed: naive adverse-move
    ranking picks falling knives at −1.31%);
  - the edge is a POINT EVENT: next-day entry −0.88% ⇒ same-tick entry only;
  - capture is front-loaded (h=1 close realizes +1.42%, t +2.51; the remaining
    move after that exit is −1.05%, i.e. ADVERSE) ⇒ the one-session exit in
    `tracker._ft_exit_reason`, never the swing exit stack;
  - costs: RT 0.21% (real-fill calibrated) .. 0.53% (modeled) — early off-RTH
    exits measured NOT worth their session premium.

DESIGN: candidates are scored through the PRODUCTION serving path
(`ml_exit_dataset.compute_exit_model_score` on synthetic cohort trades), so the
live scores are the same quantity the validation measured. For each Gate-4
scored ticker, hypothetical positions entered 1..`ft_max_cohort_days` sessions
ago (in the panel direction persisted THAT day) are scored; the most negative
is the ticker's `ft_score` and its cohort's opposite side is `ft_dir`.
Everything is persisted panel-first (`signals.ft_*`), so the mechanism accrues
an IC record whether or not trading is enabled. Fail-soft everywhere: any
error → no candidates → the tick proceeds untouched.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings


def _gate4_ok(ticker: str, price: Optional[float], dfc=None) -> bool:
    """Cache-only Gate-4: price floor from the tick's own snapshot, dollar-volume
    floor from cached OHLCV (median close×volume, trailing 60 bars). Mirrors the
    validation studies' population; fail-CLOSED on missing data. Pass ``dfc``
    (the already-loaded OHLCV frame) to avoid a second cache copy per ticker."""
    try:
        if price is None or float(price) < float(settings.trade_min_price):
            return False
        if dfc is None:
            from src.data.cache import load_ohlcv
            dfc = load_ohlcv(ticker)
        if dfc is None or dfc.empty or "Volume" not in dfc.columns:
            return False
        import pandas as pd
        c = pd.to_numeric(dfc["Close"], errors="coerce")
        v = pd.to_numeric(dfc["Volume"], errors="coerce")
        m = (c * v).tail(60).median()
        return bool(m == m and m >= float(settings.trade_min_dollar_volume))
    except Exception:
        return False


def _panel_history(n_days: int) -> Dict[str, dict]:
    """Last `n_days` distinct panel dates' per-(date, ticker) direction, abs
    combine and method scores (last run of each day) — the cohort anchors.
    One query per tick; {} on any failure."""
    from src.db import repo
    from src.analysis.ml_stacker import STACKER_LIVE_FEATURES
    mcols = [m for m in STACKER_LIVE_FEATURES if m != "tape_score"]
    try:
        df = repo.fetch_df(f"""
            SELECT signal_date, ticker, direction,
                   COALESCE(combined_score_abs, combined_score) AS c_abs,
                   {', '.join(mcols)}
            FROM (SELECT *, row_number() OVER (PARTITION BY signal_date, ticker
                                               ORDER BY generated_at DESC) rn
                  FROM signals
                  WHERE signal_date >= (SELECT MIN(d) FROM (
                        SELECT DISTINCT signal_date AS d FROM signals
                        ORDER BY d DESC LIMIT {int(n_days)})))
            WHERE rn = 1
        """)
    except Exception as e:
        logger.warning(f"[follow_through] panel-history query failed: {e}")
        return {}
    out: Dict[str, dict] = {}
    for r in df.itertuples(index=False):
        d = str(r.signal_date)[:10]
        dirn = str(r.direction or "").upper()
        ds = 1.0 if "BULL" in dirn else (-1.0 if "BEAR" in dirn else 0.0)
        if ds == 0.0:
            continue
        rec = {"ds": ds, "c_abs": getattr(r, "c_abs", None),
               "scores": {m: getattr(r, m, None) for m in mcols}}
        out.setdefault(str(r.ticker), {})[d] = rec
    return out


def _prev_selected_tickers() -> set:
    """Tickers ft_selected on the most recent PRIOR panel date — the first-day-
    of-episode guard. Empty set on any failure (fail-open: an episode repeat is
    a wasted candidate slot, not a safety issue)."""
    from src.db import repo
    try:
        df = repo.fetch_df("""
            SELECT DISTINCT ticker FROM signals
            WHERE ft_selected = 1.0 AND signal_date = (
                SELECT MAX(signal_date) FROM signals
                WHERE signal_date < ? AND ft_score IS NOT NULL)
        """, [date.today().isoformat()])
        return set(df["ticker"].astype(str)) if df is not None and not df.empty else set()
    except Exception:
        return set()


def _cohort_rows(ticker: str, sig, cohorts: dict, today: str,
                 mh_fn, today_scores: dict, mark: float, dfc=None) -> List[dict]:
    """Feature vectors for one ticker's hypothetical cohorts — built DIRECTLY
    (one closes pass per ticker, mirroring the offline validation harness and
    `build_exit_dataset`'s construction; the per-cohort serving path measured
    ~100x too slow for a tick). `mark` = the tick's live price (resolved by
    the caller — TickerSignal itself carries no price); `dfc` = the ticker's
    already-loaded OHLCV frame (one cache copy per ticker, shared with the
    Gate-4 check). Returns [{entry_d, ds, feats}]."""
    import pandas as pd
    price = float(mark or 0.0)
    if price <= 0:
        return []
    if dfc is None or dfc.empty:
        return []
    c = pd.to_numeric(dfc["Close"], errors="coerce").dropna()
    dts = [i.date().isoformat() for i in c.index]
    px = c.to_numpy(dtype=float)
    pos = {d: k for k, d in enumerate(dts)}
    today_abs = getattr(sig, "combined_score_abs", None)
    if today_abs is None:
        today_abs = getattr(sig, "combined_score", None)
    tape = getattr(sig, "tape_confirmation_score", None)
    rows: List[dict] = []
    for entry_d, rec in cohorts.items():
        if entry_d >= today or entry_d not in pos:
            continue
        i0 = pos[entry_d]
        ep = px[i0]
        if not ep > 0:
            continue
        ds = rec["ds"]
        ex_ret = ds * (price / ep - 1.0) * 100.0
        path = [ds * (px[j] / ep - 1.0) * 100.0 for j in range(i0 + 1, len(px))]
        path.append(ex_ret)
        mfe = max([0.0] + path)
        mae = min([0.0] + path)
        k = float(len(px) - 1 - i0 + 1)          # bar-count days held incl. today
        mh = 0.0
        try:
            entry_scores = {m: v for m, v in (rec.get("scores") or {}).items()
                            if v is not None}
            mh = mh_fn({"method_scores": entry_scores}) or 0.0
        except Exception:
            pass
        entry_abs = rec.get("c_abs")
        feats = {
            "days_held": k, "ex_ret": ex_ret, "ex_mfe": mfe, "ex_mae": mae,
            "ex_giveback": mfe - ex_ret, "ex_from_mae": ex_ret - mae,
            "ex_combine": (ds * float(today_abs)
                           if today_abs is not None and today_abs == today_abs else None),
            "ex_combine_delta": (ds * (float(today_abs) - float(entry_abs))
                                 if today_abs is not None and entry_abs is not None
                                 and today_abs == today_abs and entry_abs == entry_abs
                                 else None),
            "ex_elapsed_ratio": (k / mh) if mh > 0 else None,
        }
        for m, v in (today_scores or {}).items():
            feats[f"ex_{m}"] = (ds * float(v) if v is not None and v == v else None)
        feats["ex_tape_score"] = (ds * float(tape)
                                  if tape is not None and tape == tape else None)
        rows.append({"entry_d": entry_d, "ds": ds, "feats": feats})
    return rows


def _score_rows(art: dict, feats_order: List[str], pend: List[dict]):
    """Batch-score cohort rows with the artifact: signed hold-conviction
    2·P(keep)−1 per row (one predict call for the whole cross-section)."""
    import numpy as np
    X = np.array([[(r["feats"].get(f) if r["feats"].get(f) is not None else np.nan)
                   for f in feats_order] for r in pend], dtype=float)
    keep, _ex = art["model"].bull_bear(X)
    return 2.0 * np.asarray(keep, dtype=float) - 1.0


def _price_for(ticker: str, sig, price_by_ticker: Optional[dict]) -> Optional[float]:
    """The tick's mark for a ticker: the snapshot price map (TickerSignal does
    NOT carry price — the persist layer joins it from snapshots, and this
    module must do the same), else a `price` attr (tests/DB-row stand-ins),
    else the last cached close."""
    v = (price_by_ticker or {}).get(ticker)
    if v is not None and v == v and float(v) > 0:
        return float(v)
    v = getattr(sig, "price", None)
    if v is not None and v == v and float(v) > 0:
        return float(v)
    try:
        import pandas as pd
        from src.data.cache import load_ohlcv
        c = pd.to_numeric(load_ohlcv(ticker)["Close"], errors="coerce").dropna()
        return float(c.iloc[-1]) if len(c) and c.iloc[-1] > 0 else None
    except Exception:
        return None


def compute_follow_through(signals_by_ticker: dict,
                           price_by_ticker: Optional[dict] = None) -> Dict[str, dict]:
    """Score + select this tick's follow-through candidates.

    Returns {ticker: {"score", "dir", "selected", "cohort_entry_date"}} for
    every SCORED ticker (selection True only for chosen candidates); {} when
    disabled or anything fails. Pure computation — persistence is the caller's
    (pipeline stamps TickerSignal fields; tracker opens trades).
    """
    if not getattr(settings, "enable_follow_through", False):
        return {}
    try:
        import numpy as np
        from src.analysis.ml_exit_dataset import _load_exit_artifact
        art = _load_exit_artifact()
        if art is None:
            logger.debug("[follow_through] no ml_exit artifact — skipping")
            return {}
        feats_order = list(art["features"])
        hist = _panel_history(int(settings.ft_max_cohort_days) + 1)
        if not hist:
            return {}
        # Warm the method-horizon calibration OUTSIDE the budget — its first
        # call builds the solo-sim tables (TTL-cached afterwards).
        from src.analysis.exit_methods import method_horizon_days as _mh
        try:
            _mh({"method_scores": {}})
        except Exception:
            pass
        from src.performance.tracker import _method_scores_from_signal
        t0 = time.monotonic()
        budget = float(settings.ft_score_budget_seconds)
        today = date.today().isoformat()
        pend: List[dict] = []
        n_seen = 0
        for ticker, sig in (signals_by_ticker or {}).items():
            if time.monotonic() - t0 > budget:
                logger.warning(f"[follow_through] scoring budget {budget:.0f}s hit "
                               f"after {n_seen} tickers — partial cross-section")
                break
            n_seen += 1
            cohorts = hist.get(ticker)
            if not cohorts:
                continue
            mark = _price_for(ticker, sig, price_by_ticker)
            if mark is None:
                continue
            from src.data.cache import load_ohlcv
            dfc = load_ohlcv(ticker)
            if not _gate4_ok(ticker, mark, dfc=dfc):
                continue
            tscores = _method_scores_from_signal(ticker, "BULLISH", signals_by_ticker)
            for r in _cohort_rows(ticker, sig, cohorts, today, _mh, tscores, mark, dfc=dfc):
                r["ticker"] = ticker
                pend.append(r)
        if not pend:
            return {}
        net = _score_rows(art, feats_order, pend)
        out: Dict[str, dict] = {}
        for r, sc in zip(pend, net):
            cur = out.get(r["ticker"])
            if cur is None or float(sc) < cur["score"]:
                out[r["ticker"]] = {"score": float(np.clip(sc, -1.0, 1.0)),
                                    "dir": -r["ds"],
                                    "cohort_entry_date": r["entry_d"],
                                    "selected": False}
        if not out:
            return {}
        # ── selection: within-tick tail + level + first-episode-day ─────────
        scored = sorted(out.items(), key=lambda kv: kv[1]["score"])
        n_tail = max(1, int(len(scored) * float(settings.ft_tail_pct)))
        prev = _prev_selected_tickers()
        n_sel = 0
        for ticker, rec in scored[:n_tail]:
            if rec["score"] >= float(settings.ft_score_max):
                continue
            if ticker in prev:                       # episode continuation — skip
                continue
            rec["selected"] = True
            n_sel += 1
        logger.info(f"[follow_through] scored {len(out)} tickers "
                    f"(tail n={n_tail}) → {n_sel} candidate(s) "
                    f"[{', '.join(t for t, r in scored[:n_tail] if r['selected'])}]"
                    if n_sel else
                    f"[follow_through] scored {len(out)} tickers — no candidates this tick")
        return out
    except Exception as e:
        logger.warning(f"[follow_through] failed (fail-soft, no candidates): {e}")
        return {}
