"""DuckDB in-process CONFIGURATION clash (2026-08-04).

DuckDB keeps ONE database instance per path per process and refuses a second
handle whose configuration differs, so a WRITER process that opened a read-only
handle raced its own writes:

    ConnectionException: Can't open a connection to same database file with a
    different configuration than existing connections

`repo.fetch_df` hardcoded ``read_only=True`` regardless of the process role, and
the pipeline is threaded (hold-review branch, shadow arms, EOD maintenance), so a
read overlapping a write was routine. The message contains no lock wording, so it
also bypassed the open retry and surfaced as a hard failure.
"""

from __future__ import annotations

import duckdb
import pytest

from src.db import repo
from src.db.connection import _is_lock_error, connect, reset_schema_cache


def test_fetch_df_follows_the_process_role_not_a_hardcoded_true():
    # Writer process (the scheduler): reads must use the read-write config, so
    # they cannot clash with the writes happening around them.
    repo.set_read_only(False)
    with connect(read_only=False):                 # a write handle is open...
        df = repo.fetch_df("SELECT 1 AS x")        # ...this must not raise
    assert df["x"].iloc[0] == 1


def test_fetch_df_stays_read_only_for_a_reader_process():
    # The dashboard sets the process read-only; it must NEVER open read-write
    # (that would contend with the scheduler's write lock across processes).
    with connect():                                # create the file first
        pass
    repo.set_read_only(True)
    try:
        with connect(read_only=True):
            df = repo.fetch_df("SELECT 1 AS x")
        assert df["x"].iloc[0] == 1
    finally:
        repo.set_read_only(False)


def test_explicit_read_only_argument_still_wins():
    repo.set_read_only(False)
    with connect():                                # create the file first
        pass
    with connect(read_only=True):
        assert repo.fetch_df("SELECT 1 AS x", read_only=True)["x"].iloc[0] == 1


def test_config_clash_is_treated_as_retryable():
    # Belt and braces: if any path still mismatches, the open retries instead of
    # dying — the conflicting handle is always a context manager about to close.
    exc = duckdb.Error("Can't open a connection to same database file with a "
                       "different configuration than existing connections")
    assert _is_lock_error(exc) is True


def test_schema_memo_still_creates_the_schema_after_a_reset():
    # The memo must not skip schema creation for a database this process has not
    # actually initialised (tests recreate files at fresh paths every run).
    reset_schema_cache()
    with connect() as conn:
        n = conn.execute("SELECT count(*) FROM information_schema.tables "
                         "WHERE table_name = 'trades'").fetchone()[0]
    assert n == 1
