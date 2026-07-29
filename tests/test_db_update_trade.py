"""repo.update_trade — targeted single-row ledger writes (2026-07-24).

save_trades full-replaces the trades table, which is right for a bulk save but
wasteful when only one broker leg moved. The reconciler persists after EVERY
order submission (so a watchdog kill can't orphan a live order), so a 10-order
tick was re-writing every unchanged row ~10 times. update_trade rewrites just
the matched row, and REFUSES the fast path when it can't identify exactly one.
"""

import pytest

from config.settings import settings
from src.db import repo


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def _trade(ticker="AAPL", when="2026-07-01T14:00:00+00:00", **kw):
    base = {"ticker": ticker, "action": "BUY", "direction": "BUY", "status": "OPEN",
            "entry_date": when[:10], "entry_datetime": when, "decision_datetime": when,
            "entry_price": 100.0, "confidence": 0.9, "position_size_multiplier": 1.0}
    base.update(kw)
    return base


def test_update_trade_rewrites_only_the_matched_row(tmp_db):
    a, b = _trade("AAPL"), _trade("MSFT", "2026-07-01T15:00:00+00:00")
    repo.save_trades([a, b], allow_shrink=True)

    a2 = dict(a, broker_order_id="42", broker_status="Filled", entry_price=101.5)
    assert repo.update_trade(a2) is True

    out = {t["ticker"]: t for t in repo.load_trades()}
    assert len(out) == 2, "the other row must survive untouched"
    assert out["AAPL"]["broker_order_id"] == "42"
    assert out["AAPL"]["entry_price"] == 101.5
    assert out["MSFT"]["entry_price"] == 100.0
    assert "broker_order_id" not in out["MSFT"]


def test_update_trade_declines_an_unknown_row(tmp_db):
    repo.save_trades([_trade("AAPL")], allow_shrink=True)
    # Different ticker+timestamp → different trade_id → not in the table.
    assert repo.update_trade(_trade("TSLA", "2026-07-02T14:00:00+00:00")) is False
    assert len(repo.load_trades()) == 1, "a declined update must not insert"


def test_update_trade_declines_ambiguous_duplicates(tmp_db):
    """trade_id is a ticker+timestamp hash, so duplicates are possible — a
    targeted DELETE would take out BOTH, so the fast path must decline."""
    dup = _trade("AAPL")
    repo.save_trades([dup, dict(dup)], allow_shrink=True)
    assert repo.update_trade(dict(dup, entry_price=999.0)) is False
    prices = [t["entry_price"] for t in repo.load_trades()]
    assert prices == [100.0, 100.0], "neither duplicate may be modified"


def test_persist_legs_falls_back_to_a_full_save(tmp_db, monkeypatch):
    """When the row can't be uniquely matched the reconciler must still persist
    — durability is the whole point of the call."""
    from src.broker import reconcile
    a = _trade("AAPL")
    repo.save_trades([a], allow_shrink=True)
    unknown = _trade("TSLA", "2026-07-02T14:00:00+00:00")
    reconcile._persist_legs([a, unknown], unknown)
    assert {t["ticker"] for t in repo.load_trades()} == {"AAPL", "TSLA"}
