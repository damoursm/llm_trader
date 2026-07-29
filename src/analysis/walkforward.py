"""Walk-forward weight calibration — weights as production would have had them.

Tier 2 applied ONE weight set to all of history. That is the classic backtest
error: those weights were calibrated on data that, relative to any historical
row, is the future, so the backtest silently knew how every method turned out.
Production never has that — it recalibrates each tick from what had happened so
far, and the weights it uses on a Tuesday are strictly worse-informed than the
ones a full-history fit would give.

This walks the calendar instead. For each step D it installs an `analysis_asof`
cutoff, so the entire weighting stack — win-rate filters, IC weights, per-side
skill, market-relative filter, method-horizon states, inversions — recomputes
seeing only rows strictly before D. The resulting vector is stored in
`weight_history` and is the weight set a backtest must use for rows dated D.

**This is what makes walk-forward evidence rather than a curiosity.** Tier 2 is
firewalled off calibration because fitting weights on values derived from those
same weights is self-confirmation. Under a cutoff that objection dissolves:
weights at D cannot encode D's outcome, because D's data was not visible when
they were fitted. Walk-forward results may therefore be read as genuine
out-of-sample evidence — the one path from "backtest" to "measurement".

Cost is the honest catch: every step re-runs the whole calibration stack over a
growing window, so this is minutes-to-hours over a long history, not something
to put on the tick path. Run it from EOD maintenance or by hand.

CLI:  python -m src.analysis.walkforward --step 1 --write
      python -m src.analysis.walkforward --show
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.analysis.asof import analysis_asof


def weights_as_of(cutoff: str) -> Dict[str, object]:
    """The full weight state the system would have had on ``cutoff``.

    Every layer is read through the aggregator's own helpers rather than
    reimplemented, so this cannot drift into a private copy of the weighting
    stack — the whole value of a walk-forward is that it uses the REAL logic.
    """
    from src.signals.aggregator import (_BASE_WEIGHTS, _inverted_methods,
                                        side_filtered_methods,
                                        side_weight_multipliers,
                                        winrate_filtered_methods)

    state: Dict[str, object] = {"as_of": cutoff}
    with analysis_asof(cutoff):
        try:
            inverted = sorted(_inverted_methods())
        except Exception as e:
            logger.warning(f"[walkforward] {cutoff}: inversion unavailable ({e})")
            inverted = []
        try:
            filtered = sorted(winrate_filtered_methods())
        except Exception as e:
            logger.warning(f"[walkforward] {cutoff}: filter unavailable ({e})")
            filtered = []
        try:
            buy_filtered = sorted(side_filtered_methods("buy"))
            sell_filtered = sorted(side_filtered_methods("sell"))
            buy_mults = {k: round(float(v), 4)
                         for k, v in (side_weight_multipliers("buy") or {}).items()}
            sell_mults = {k: round(float(v), 4)
                          for k, v in (side_weight_multipliers("sell") or {}).items()}
        except Exception as e:
            logger.warning(f"[walkforward] {cutoff}: per-side layer unavailable ({e})")
            buy_filtered = sell_filtered = []
            buy_mults = sell_mults = {}

        inv = set(inverted)
        eff = {m: (-w if m in inv else w) for m, w in _BASE_WEIGHTS.items()
               if m not in set(filtered)}

    state.update({
        "weights": {k: round(float(v), 6) for k, v in sorted(eff.items())},
        "inverted": inverted,
        "filtered": filtered,
        "buy_filtered": buy_filtered,
        "sell_filtered": sell_filtered,
        "buy_mults": buy_mults,
        "sell_mults": sell_mults,
        "n_active": len(eff),
    })
    return state


def _panel_span() -> tuple:
    """(first, last) signal_date in the panel — the walk's natural bounds."""
    from src.db import repo
    df = repo.fetch_df("SELECT min(signal_date) AS lo, max(signal_date) AS hi "
                       "FROM signals")
    if df is None or df.empty or df.iloc[0]["lo"] is None:
        return None, None
    return str(df.iloc[0]["lo"])[:10], str(df.iloc[0]["hi"])[:10]


def walk(start: Optional[str] = None, end: Optional[str] = None,
         step_days: int = 1, min_history_days: int = 7) -> List[dict]:
    """Recalibrate at each step across the span. Returns one state per step.

    ``min_history_days`` skips the opening stretch where there is not yet enough
    data to calibrate anything — those steps would emit base weights and read as
    a finding ("the system used flat weights early on") when they are really an
    absence of evidence.
    """
    lo, hi = _panel_span()
    if lo is None:
        logger.warning("[walkforward] no signals rows")
        return []
    start = start or lo
    end = end or hi
    d0 = date.fromisoformat(max(start, lo))
    d1 = date.fromisoformat(min(end, hi))
    first_ok = date.fromisoformat(lo) + timedelta(days=int(min_history_days))

    out: List[dict] = []
    d = d0
    while d <= d1:
        if d < first_ok:
            d += timedelta(days=int(step_days))
            continue
        cutoff = d.isoformat()
        try:
            st = weights_as_of(cutoff)
            out.append(st)
            logger.info(f"[walkforward] {cutoff}: {st['n_active']} active, "
                        f"{len(st['filtered'])} filtered, "
                        f"{len(st['inverted'])} inverted")
        except Exception as e:
            logger.warning(f"[walkforward] {cutoff} failed: {e}")
        d += timedelta(days=int(step_days))
    return out


def materialize(start: Optional[str] = None, end: Optional[str] = None,
                step_days: int = 1) -> int:
    """Persist the walk into `weight_history`, replacing the same span."""
    from src.db.connection import connect

    states = walk(start=start, end=end, step_days=step_days)
    if not states:
        return 0
    stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{
        "as_of": s["as_of"],
        "computed_at": stamped,
        "n_active": int(s["n_active"]),
        "weights": json.dumps(s["weights"], sort_keys=True),
        "inverted": json.dumps(s["inverted"]),
        "filtered": json.dumps(s["filtered"]),
        "buy_filtered": json.dumps(s["buy_filtered"]),
        "sell_filtered": json.dumps(s["sell_filtered"]),
        "buy_mults": json.dumps(s["buy_mults"], sort_keys=True),
        "sell_mults": json.dumps(s["sell_mults"], sort_keys=True),
    } for s in states]
    df = pd.DataFrame(rows)
    lo, hi = df["as_of"].min(), df["as_of"].max()
    with connect() as con:
        con.execute("DELETE FROM weight_history WHERE as_of >= ? AND as_of <= ?",
                    [lo, hi])
        con.register("_wf_df", df)
        con.execute(f"INSERT INTO weight_history ({', '.join(df.columns)}) "
                    f"SELECT {', '.join(df.columns)} FROM _wf_df")
        con.unregister("_wf_df")
    logger.info(f"[walkforward] stored {len(df)} calibration steps {lo}..{hi}")
    return len(df)


def load_weight_history() -> pd.DataFrame:
    """The stored walk, ascending by date. Empty frame when absent."""
    from src.db import repo
    try:
        return repo.fetch_df("SELECT * FROM weight_history ORDER BY as_of")
    except Exception as e:
        logger.debug(f"[walkforward] weight_history unavailable: {e}")
        return pd.DataFrame()


def weights_for_date(signal_date: str,
                     history: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """The weight state in force for ``signal_date`` — the LATEST calibration
    strictly before it.

    "Strictly before" is the whole contract: a calibration stamped on the same
    day would have been computed from that day's own data.
    """
    h = history if history is not None else load_weight_history()
    if h is None or h.empty:
        return None
    d = str(signal_date)[:10]
    prior = h[h["as_of"].astype(str) < d]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {
        "as_of": str(row["as_of"]),
        "weights": json.loads(row["weights"]),
        "inverted": json.loads(row["inverted"]),
        "filtered": json.loads(row["filtered"]),
        "buy_filtered": json.loads(row["buy_filtered"]),
        "sell_filtered": json.loads(row["sell_filtered"]),
        "buy_mults": json.loads(row["buy_mults"]),
        "sell_mults": json.loads(row["sell_mults"]),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        description="Recalibrate the weighting stack tick by tick, point-in-time")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=1, help="days between steps")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the stored walk")
    a = ap.parse_args()

    if a.show:
        h = load_weight_history()
        if h.empty:
            print("weight_history is empty — run with --write first")
        else:
            print(h[["as_of", "n_active", "computed_at"]].to_string(index=False))
    elif a.write:
        print(f"stored {materialize(a.start, a.end, a.step)} calibration steps")
    else:
        for s in walk(a.start, a.end, a.step):
            print(f"{s['as_of']}  active={s['n_active']:>3}  "
                  f"filtered={len(s['filtered']):>2}  inverted={s['inverted']}")
