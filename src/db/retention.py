"""Table retention — bound the fastest-growing derived tables over the weeks.

``simulated_trades`` is a LONG-format reshape of ``signals`` (one row per
scored ticker-method, ~25× the wide signals row) and it grows ~130k rows/day
at 30–44 runs/day. The analysis over it (``compute_method_perf`` /
``compute_directional_perf``) dedupes to the last run per (signal_date, ticker,
method), so the many intraday re-runs of a day are pure bloat there — EXCEPT
the entry-EVENT detection (``extract_entry_events``, the default) needs the
intraday score SEQUENCE to spot sign flips. So retention keeps a recent RAW
window (full intraday resolution) and only COLLAPSES data older than that to
the last-per-(day, ticker, method) row — which is exactly what the deduped
analysis reads, so its output over the old window is unchanged. Beyond a
generous keep window everything is hard-deleted (the analysis looks back
≤ 90 days; ``signals`` remains the durable source and could rebuild
``simulated_trades`` if ever needed).

``exit_signals`` grows similarly as positions are held; it is age-pruned only
(primary data — the per-tick review scores). ``signals`` / ``trade_reviews``
are the source / primary tables and are left to the generous age prune only.

All operations are one short transaction each, deterministic, logged, and
fail-soft. Run daily by the scheduler's EOD hook, or manually:

    python -m src.db.retention
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from config.settings import settings
from src.db.connection import connect


def collapse_simulated_trades(raw_days: int) -> int:
    """Collapse ``simulated_trades`` OLDER than ``raw_days`` to the last row per
    (signal_date, ticker, method) — the deduped set the analysis actually reads.
    The recent ``raw_days`` window is left untouched (the entry-event detector
    needs its intraday sequence). Returns rows removed. No-op if nothing old."""
    cutoff = (date.today() - timedelta(days=max(0, int(raw_days)))).isoformat()
    with connect() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM simulated_trades WHERE signal_date < ?", [cutoff]).fetchone()[0]
        if not before:
            return 0
        conn.execute("BEGIN TRANSACTION")
        # Keep the latest-generated row per (day, ticker, method) among the old
        # partition; rebuild that partition from it.
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _sim_keep AS
            SELECT * FROM simulated_trades
            WHERE signal_date < ?
            QUALIFY row_number() OVER (
                PARTITION BY signal_date, ticker, method
                ORDER BY generated_at DESC) = 1
        """, [cutoff])
        conn.execute("DELETE FROM simulated_trades WHERE signal_date < ?", [cutoff])
        conn.execute("INSERT INTO simulated_trades SELECT * FROM _sim_keep")
        kept = conn.execute("SELECT COUNT(*) FROM _sim_keep").fetchone()[0]
        conn.execute("DROP TABLE _sim_keep")
        conn.execute("COMMIT")
    removed = int(before) - int(kept)
    if removed > 0:
        logger.info(f"[retention] simulated_trades: collapsed {before}→{kept} old rows "
                    f"(< {cutoff}), removed {removed} intraday duplicates")
    return removed


def prune_beyond(table: str, date_col: str, keep_days: int) -> int:
    """Hard-delete rows in ``table`` whose ``date_col`` is older than
    ``keep_days``. Returns rows removed."""
    cutoff = (date.today() - timedelta(days=max(1, int(keep_days)))).isoformat()
    with connect() as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {date_col} < ?",
                         [cutoff]).fetchone()[0]
        if n:
            conn.execute(f"DELETE FROM {table} WHERE {date_col} < ?", [cutoff])
    if n:
        logger.info(f"[retention] {table}: pruned {int(n)} row(s) older than {cutoff}")
    return int(n)


def run_retention() -> dict:
    """Daily retention pass: collapse old simulated_trades intraday bloat, then
    age-prune the fastest-growing derived tables. Fail-soft per step (one
    failure never blocks the rest). Returns a per-step counts dict."""
    out: dict = {}
    if not settings.enable_sim_retention:
        return out
    steps = (
        ("sim_collapsed", lambda: collapse_simulated_trades(settings.sim_retention_raw_days)),
        ("sim_pruned", lambda: prune_beyond("simulated_trades", "signal_date",
                                            settings.sim_retention_keep_days)),
        ("exit_signals_pruned", lambda: prune_beyond("exit_signals", "signal_date",
                                                     settings.exit_signals_keep_days)),
    )
    for name, fn in steps:
        try:
            out[name] = fn()
        except Exception as e:
            logger.warning(f"[retention] step {name} failed ({e}) — skipped")
            out[name] = None
    return out


def main() -> None:
    res = run_retention()
    print("retention:", res if res else "disabled (enable_sim_retention=false)")


if __name__ == "__main__":
    main()
