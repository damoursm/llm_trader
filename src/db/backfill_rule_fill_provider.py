"""One-off data repair (2026-07-22): re-stamp historical rule-based BACK-FILLS.

``recommendations.llm_provider`` is written as ONE per-run value to every row of
a run (``pipeline._persist_run``), but the synthesis model is only asked about
the ~40 highest-confidence tickers (``claude_analyst._MAX_SIGNALS``). Every other
ticker in the universe is back-filled by ``_fallback_recommendations`` so open
positions never fall silent — and those fills used to inherit the run's model id.
They are ~22% of all rows, they are IDENTICAL across engines, and they drag every
model toward a shared mean in any per-engine comparison (up to 36% of one arm's
BUY calls).

New rows are stamped correctly at write time (``Recommendation.rule_filled`` →
``tracker.RULE_FILL_MODEL``). This script fixes the rows written BEFORE that, so
the raw column is clean for ad-hoc SQL instead of relying on every reader to
apply ``tracker._is_rule_based_fill``.

Discriminator: a fill reuses the SIGNAL's rationale, which the aggregator
assembles from the fragments below; the synthesis LLM writes its own prose and
never emits them (the prompt renders the same figure as "Technical score=", with
an equals sign). Validated over 7.1k rows joined to ``signals``: every matching
row has a confidence EXACTLY equal to its signal's AND an action equal to the
rule-based mapping — 100% precision. Note a fill often READS authored, because
the news half of a signal rationale is written by the SENTIMENT model; that is
not the synthesis engine being attributed.

Nothing is lost: ``run_id`` still links each fill to its run, and
``runs.llm_synthesis_provider`` still records which engine ran that tick.

Idempotent — rows already carrying a rule-based label are skipped, so re-running
is a no-op. ``rule-based (no LLM)`` (a WHOLE-RUN outage) is a distinct population
and is deliberately left untouched.

Usage:
    python -m src.db.backfill_rule_fill_provider              # dry run (default)
    python -m src.db.backfill_rule_fill_provider --apply      # write
"""

import argparse

from loguru import logger

from src.db.connection import connect
from src.performance.tracker import RULE_FILL_MODEL, _RULE_FILL_MARKERS

# Rows already labelled as rule-based output — never re-stamped.
_RULE_LABELS = (RULE_FILL_MODEL, "rule-based (no LLM)", "rule-based")


def _where() -> str:
    """SQL predicate selecting legacy fills that still wear a model's name."""
    markers = " OR ".join(f"rationale LIKE '%{m}%'" for m in _RULE_FILL_MARKERS)
    labels = ", ".join(f"'{l}'" for l in _RULE_LABELS)
    return (f"({markers}) AND rationale IS NOT NULL "
            f"AND COALESCE(llm_provider, '') NOT IN ({labels})")


def run(apply: bool = False) -> int:
    """Report (and optionally apply) the re-stamp. Returns the affected count."""
    where = _where()
    with connect(read_only=False) as conn:
        total = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        rows = conn.execute(
            f"SELECT COALESCE(llm_provider,'(null)') AS provider, action, COUNT(*) n "
            f"FROM recommendations WHERE {where} GROUP BY 1,2 ORDER BY n DESC"
        ).fetchall()
        affected = sum(r[2] for r in rows)

        logger.info(f"[backfill] {affected} of {total} recommendation row(s) "
                    f"({affected / total * 100:.1f}%) are legacy rule-based fills")
        for provider, action, n in rows:
            logger.info(f"[backfill]   {provider:32s} {action:6s} {n:6d}")

        if not affected:
            logger.info("[backfill] nothing to do")
            return 0
        if not apply:
            logger.info("[backfill] DRY RUN — re-run with --apply to write")
            return affected

        conn.execute(
            f"UPDATE recommendations SET llm_provider = ? WHERE {where}",
            [RULE_FILL_MODEL],
        )
        left = conn.execute(
            f"SELECT COUNT(*) FROM recommendations WHERE {where}").fetchone()[0]
        now = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE llm_provider = ?",
            [RULE_FILL_MODEL]).fetchone()[0]
        after_total = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        logger.info(f"[backfill] re-stamped {affected} row(s) → '{RULE_FILL_MODEL}' "
                    f"({now} now carry the label); {left} unconverted; "
                    f"row count {total} → {after_total}")
        if left or after_total != total:
            logger.error("[backfill] POST-CHECK FAILED — inspect before trusting the table")
        return affected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default is a dry run)")
    run(apply=ap.parse_args().apply)
