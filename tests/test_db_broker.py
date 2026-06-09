"""Round-trip tests for repo.insert_broker_report against a temporary DuckDB file."""

import pytest

from config.settings import settings


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def _report():
    return {
        "run_id": "run-1", "mode": "ibkr_paper", "connected": True, "ok": True,
        "account_id": "DU111", "account_equity": 99000.0, "account_currency": "USD",
        "entries_submitted": 1, "exits_submitted": 1, "fills_repaired": 0, "rejects": 0,
        "drift": [{"ticker": "NVDA", "broker_qty": 5}],
        "errors": [],
        "orders": [
            {
                "event": "SUBMIT", "intent": "ENTRY", "ticker": "AAPL", "side": "BUY",
                "order_type": "MKT", "requested_qty": 7, "filled_qty": 7,
                "model_price": 100.0, "limit_price": None, "fill_price": 100.05,
                "slippage_bps": 5.0, "commission": 0.35, "status": "Filled",
                "ok": True, "error": None, "order_id": "1", "client_ref": "rec-AAPL",
                "submitted_at": "2026-06-09T15:00:00+00:00",
            },
            {
                "event": "SUBMIT", "intent": "EXIT", "ticker": "MSFT", "side": "SELL",
                "order_type": "MKT", "requested_qty": 3, "filled_qty": 3,
                "model_price": 200.0, "limit_price": None, "fill_price": 199.90,
                "slippage_bps": 5.0, "commission": 0.35, "status": "Filled",
                "ok": True, "error": None, "order_id": "2", "client_ref": "rec-MSFT-exit",
                "submitted_at": "2026-06-09T15:00:05+00:00",
            },
        ],
    }


def test_insert_broker_report_roundtrip(tmp_db):
    from src.db import repo

    repo.insert_broker_report("run-1", _report())

    df = repo.fetch_df("SELECT * FROM broker_reconciles", read_only=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_id"] == "run-1"
    assert row["mode"] == "ibkr_paper"
    assert bool(row["connected"]) is True
    assert int(row["entries_submitted"]) == 1
    assert int(row["n_drift"]) == 1
    assert "NVDA" in row["drift"]

    od = repo.fetch_df(
        "SELECT * FROM broker_orders ORDER BY submitted_at", read_only=False)
    assert len(od) == 2
    assert list(od["intent"]) == ["ENTRY", "EXIT"]
    assert od.iloc[0]["slippage_bps"] == pytest.approx(5.0)
    assert od.iloc[0]["commission"] == pytest.approx(0.35)


def test_insert_broker_report_idempotent_per_run(tmp_db):
    from src.db import repo

    repo.insert_broker_report("run-1", _report())
    repo.insert_broker_report("run-1", _report())   # re-persist same run → replaced
    repo.insert_broker_report("run-2", _report())   # different run → appended

    df = repo.fetch_df("SELECT run_id FROM broker_reconciles ORDER BY run_id",
                       read_only=False)
    assert list(df["run_id"]) == ["run-1", "run-2"]
    od = repo.fetch_df("SELECT count(*) AS n FROM broker_orders", read_only=False)
    assert int(od.iloc[0]["n"]) == 4


def test_insert_broker_report_skips_empty(tmp_db):
    from src.db import repo

    repo.insert_broker_report("run-x", {})           # no-op, must not raise
    # Table may not even exist yet (no write connection was opened) — opening
    # one now creates the schema; the run-x row must be absent.
    df = repo.fetch_df("SELECT count(*) AS n FROM broker_reconciles", read_only=False)
    assert int(df.iloc[0]["n"]) == 0
