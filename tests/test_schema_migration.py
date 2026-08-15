"""Schema migration — `_ADD_COLUMNS` must actually APPLY, not just be listed.

`CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
every column added after a table was first created reaches the production
database only through the `_ADD_COLUMNS` loop at the bottom of `ensure_schema`.
That loop is the one mechanism keeping a months-old DuckDB file in step with the
code.

Until now it was tested only as a LIST: several suites assert
`("signals", "news_shock") in _ADD_COLUMNS` and stop there. That checks the
registration and not the migration — and the two failure modes it misses are
both silent:

* a column registered but never applied reads as **NULL on every row forever**,
  which downstream is indistinguishable from "the method never fired" (exactly
  how the 2026-08-14 abs-shadow dead store hid);
* a migration that is not idempotent, or that rebuilds rather than alters, would
  DESTROY the history it was meant to extend — and the ledger is the one thing
  in this project that cannot be regenerated.

So these tests build a database at an OLD schema, run `ensure_schema` over it,
and check both halves: the new columns exist, and the existing rows survived.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.db.connection import connect
from src.db.schema import _ADD_COLUMNS, ensure_schema


def _columns(conn, table: str) -> set:
    return {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ?", [table]).fetchall()}


def _tables(conn) -> set:
    return {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}


# ── every registered column reaches a real database ────────────────────────

def test_every_registered_column_exists_after_ensure_schema():
    """The list and the database must agree. A registered column that never
    lands is a permanent NULL that reads as an absent signal."""
    with connect(read_only=False) as conn:
        missing = []
        for table, col, _type in _ADD_COLUMNS:
            if col not in _columns(conn, table):
                missing.append(f"{table}.{col}")
        assert not missing, f"registered but not applied: {missing}"


def test_every_registered_column_targets_a_table_that_exists():
    """An ALTER against a missing table raises inside `ensure_schema`, which
    runs on EVERY read-write connection — so a typo'd table name takes down the
    whole pipeline, not just its own column."""
    with connect(read_only=False) as conn:
        present = _tables(conn)
        bad = {t for t, _c, _ty in _ADD_COLUMNS if t not in present}
        assert not bad, f"_ADD_COLUMNS references non-existent tables: {sorted(bad)}"


def test_no_column_is_registered_twice_with_different_types():
    """Two entries for one column silently race: whichever ALTER runs second is
    a no-op, so the type that wins is decided by list order."""
    seen: dict = {}
    for table, col, coltype in _ADD_COLUMNS:
        key = (table, col)
        if key in seen:
            assert seen[key] == coltype, (
                f"{table}.{col} registered as both {seen[key]} and {coltype}")
        seen[key] = coltype


# ── the actual migration of an OLD database ────────────────────────────────

def test_a_column_dropped_from_an_existing_table_is_restored():
    """The real scenario: a database created before the column existed. Dropping
    it reproduces that state exactly, and `ensure_schema` must put it back —
    `CREATE TABLE IF NOT EXISTS` alone never would."""
    with connect(read_only=False) as conn:
        conn.execute("ALTER TABLE signals DROP COLUMN IF EXISTS news_shock")
        assert "news_shock" not in _columns(conn, "signals")
        ensure_schema(conn)
        assert "news_shock" in _columns(conn, "signals")


def test_migrating_an_old_database_preserves_its_rows():
    """The half that matters most. A migration that recreated the table instead
    of altering it would pass the column check above and quietly erase history."""
    at = datetime.now(timezone.utc).isoformat()
    with connect(read_only=False) as conn:
        conn.execute(
            "INSERT INTO signals (run_id, generated_at, signal_date, ticker, "
            "combined_score, confidence) VALUES ('r1', ?, '2026-08-14', 'AAPL', "
            "0.42, 0.87)", [at])
        before = conn.execute(
            "SELECT ticker, combined_score, confidence FROM signals").fetchall()

        conn.execute("ALTER TABLE signals DROP COLUMN IF EXISTS news_shock")
        ensure_schema(conn)

        after = conn.execute(
            "SELECT ticker, combined_score, confidence FROM signals").fetchall()
        assert after == before, "migration lost or altered existing rows"
        # ...and the restored column is NULL on the pre-existing row, which is
        # the correct "forward-collected" semantics, not 0.0.
        assert conn.execute(
            "SELECT news_shock FROM signals").fetchone()[0] is None


def test_ensure_schema_is_idempotent():
    """It runs on every read-write connection — many times per tick."""
    with connect(read_only=False) as conn:
        ensure_schema(conn)
        first = {t: _columns(conn, t) for t in _tables(conn)}
        for _ in range(3):
            ensure_schema(conn)
        assert {t: _columns(conn, t) for t in _tables(conn)} == first


def test_a_fresh_database_and_a_migrated_one_end_up_identical(tmp_path, monkeypatch):
    """The invariant that keeps a long-lived production DB honest: whatever path
    a database took to get here, its shape must match a brand-new one. Otherwise
    a query that works in testing fails on the box that has been running for
    months."""
    from config.settings import settings

    with connect(read_only=False) as conn:
        fresh = {t: _columns(conn, t) for t in _tables(conn)}

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aged.db"))
    with connect(read_only=False) as conn:
        # Age it: strip one column from each table that has a registered one.
        for table, col, _ty in _ADD_COLUMNS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {col}")
        ensure_schema(conn)
        aged = {t: _columns(conn, t) for t in _tables(conn)}

    assert aged == fresh, "a migrated database does not match a fresh one"


# ── the registration is kept in step with the column groups ────────────────

def test_the_later_signal_column_groups_are_registered_for_migration():
    """Each group added AFTER the signals table first existed has to be in
    `_ADD_COLUMNS` as well as in the CREATE statement. Registering it in only
    the latter works perfectly on a developer's fresh database and does nothing
    on the production one — the worst possible split, because it cannot be
    reproduced locally."""
    from src.db.schema import (SIGNAL_ABS_SHADOW_COLUMNS,
                               SIGNAL_CONFIDENCE_COMPONENT_COLUMNS,
                               SIGNAL_FUNDAMENTAL_COLUMNS,
                               SIGNAL_NEWS_ATTENTION_COLUMNS,
                               SIGNAL_TIMEFRAME_COLUMNS)
    registered = {(t, c) for t, c, _ty in _ADD_COLUMNS}
    groups = {
        "SIGNAL_CONFIDENCE_COMPONENT_COLUMNS": SIGNAL_CONFIDENCE_COMPONENT_COLUMNS,
        "SIGNAL_ABS_SHADOW_COLUMNS": SIGNAL_ABS_SHADOW_COLUMNS,
        "SIGNAL_NEWS_ATTENTION_COLUMNS": SIGNAL_NEWS_ATTENTION_COLUMNS,
        "SIGNAL_TIMEFRAME_COLUMNS": SIGNAL_TIMEFRAME_COLUMNS,
        "SIGNAL_FUNDAMENTAL_COLUMNS": SIGNAL_FUNDAMENTAL_COLUMNS,
    }
    missing = {name: [c for c in cols if ("signals", c) not in registered]
               for name, cols in groups.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, (
        f"column groups present in CREATE TABLE but not in _ADD_COLUMNS, so they "
        f"will never appear on an existing database: {missing}")


# The 19 method columns present when the `signals` table was first created.
# They need no ALTER — every database that has the table has them — which is
# why they are exempt from the guard below rather than a gap in it.
_ORIGINAL_METHOD_COLUMNS = frozenset({
    "news", "sent_velocity", "tech", "insider", "put_call", "max_pain", "oi_skew",
    "vwap", "pattern", "momentum", "sector_momentum", "money_flow", "trend_strength",
    "pead", "iv_rank", "iv_expr", "coint", "cross_sectional", "ext_gap",
})


def test_every_method_added_since_the_table_was_created_is_registered():
    """The forward-looking half. `add-method` says to add the column to
    `SIGNAL_BASE_METHOD_COLUMNS` and to run an ALTER on an existing DB; this is
    what makes forgetting the second step fail in CI instead of six weeks later,
    when the new method's panel column turns out to be NULL for its whole life.

    A method that legitimately predates the table belongs in the exempt set
    above — and adding one there is a deliberate act, not an accident."""
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    registered = {c for t, c, _ty in _ADD_COLUMNS if t == "signals"}
    unregistered = [c for c in SIGNAL_BASE_METHOD_COLUMNS
                    if c not in registered and c not in _ORIGINAL_METHOD_COLUMNS]
    assert not unregistered, (
        f"method column(s) {unregistered} will never be added to an existing "
        f"database — add ('signals', <col>, 'DOUBLE') to _ADD_COLUMNS")


def test_the_exempt_set_is_not_quietly_growing():
    """The exemption above is only safe while it names columns that really are
    original. Pinning its size stops it becoming the place new methods get
    parked to make the guard pass."""
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    assert len(_ORIGINAL_METHOD_COLUMNS) == 19
    assert _ORIGINAL_METHOD_COLUMNS <= set(SIGNAL_BASE_METHOD_COLUMNS)
