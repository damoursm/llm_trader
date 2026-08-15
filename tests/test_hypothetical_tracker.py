"""Always-open hypothetical book (`src/performance/hypothetical_tracker.py`).

A passive long/short baseline the real book is compared against. Its whole value
is being an HONEST control, which makes two properties load-bearing:

* **entries are never silently re-anchored.** The book exists to answer "what if
  we had just held these since we started?"; moving an entry price resets that
  clock. Exactly one thing may re-anchor — the user flipping BUY↔SELL in config
  — and it must not happen on a missing price, a zero price, or an ordinary
  refresh.
* **the marks use the same cost model as the real ledger** (`_pct_return` with
  the asset type), or the comparison flatters whichever book is charged less.

Prices are stubbed everywhere: `_fetch_price` calls yfinance. The DuckDB
handle is the conftest's throwaway file, so the real hypothetical table is
never touched.
"""

from __future__ import annotations

from datetime import date

import pytest

from config.settings import settings
from src.performance import hypothetical_tracker as ht


@pytest.fixture(autouse=True)
def _stubbed(monkeypatch):
    monkeypatch.setattr(ht, "_reference_close", lambda t: None)
    monkeypatch.setattr(ht, "_fetch_price",
                        lambda t: pytest.fail(f"unstubbed price fetch for {t}"))


def _prices(monkeypatch, mapping):
    monkeypatch.setattr(ht, "_fetch_price", lambda t: mapping.get(t))


def _config(monkeypatch, spec: str):
    monkeypatch.setattr(settings, "enable_hypothetical_trades", True)
    monkeypatch.setattr(settings, "hypothetical_trades", spec)


# ── asset classification ────────────────────────────────────────────────────

def test_asset_type_drives_the_spread_tier():
    """ETF and commodity spread tiers are tighter than the stock tiers, so a
    misclassification changes every mark this book reports."""
    assert ht._classify_asset_type(settings.commodities_list[0]) == "COMMODITY"
    assert ht._classify_asset_type(settings.sectors_list[0]) == "ETF"
    assert ht._classify_asset_type(settings.factor_list[0]) == "ETF"
    assert ht._classify_asset_type("MSFT") == "STOCK"
    assert ht._classify_asset_type("msft") == "STOCK"


# ── legacy normalisation ────────────────────────────────────────────────────

def test_legacy_long_short_rows_are_normalised_on_read():
    rows = ht._normalize_loaded([
        {"ticker": "GLD", "direction": "LONG"},
        {"ticker": "SLV", "direction": "SHORT"},
        {"ticker": "NVDA", "direction": "BUY", "action": "BUY"},
    ])
    assert [r["direction"] for r in rows] == ["BUY", "SELL", "BUY"]
    # `action` is what daily_nav walks, so it must be filled in, not left None.
    assert rows[0]["action"] == "BUY" and rows[1]["action"] == "SELL"


def test_normalisation_leaves_an_existing_action_alone():
    rows = ht._normalize_loaded([{"ticker": "GLD", "direction": "LONG", "action": "SELL"}])
    assert rows[0]["action"] == "SELL", "an explicit action must not be overwritten"


# ── opening ─────────────────────────────────────────────────────────────────

def test_first_run_opens_every_configured_ticker(monkeypatch):
    _config(monkeypatch, "GLD:BUY,NVDA:SELL")
    _prices(monkeypatch, {"GLD": 300.0, "NVDA": 100.0})
    ht.update_hypothetical_trades()

    book = {t["ticker"]: t for t in ht._load()}
    assert set(book) == {"GLD", "NVDA"}
    assert book["GLD"]["action"] == "BUY" and book["NVDA"]["action"] == "SELL"
    assert book["GLD"]["entry_price"] == 300.0
    assert book["GLD"]["status"] == "OPEN"
    assert book["GLD"]["entry_date"] == date.today().isoformat()
    assert book["GLD"]["return_pct"] == 0.0
    assert book["GLD"]["position_size_multiplier"] == 1.0


@pytest.mark.parametrize("bad_price", [None, 0.0, -5.0])
def test_an_unusable_entry_price_opens_nothing(monkeypatch, bad_price):
    """Opening at 0 (or at a fabricated price) would produce an infinite or
    invented return for the life of the book."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": bad_price})
    ht.update_hypothetical_trades()
    assert ht._load() == []


def test_positions_stay_open_forever(monkeypatch):
    """No auto-close, no signal-decay exit — that is what makes it a baseline."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()
    for px in (250.0, 400.0, 100.0):
        _prices(monkeypatch, {"GLD": px})
        ht.update_hypothetical_trades()
    book = ht._load()
    assert len(book) == 1 and book[0]["status"] == "OPEN"


# ── marking ─────────────────────────────────────────────────────────────────

def test_marks_use_the_shared_cost_model(monkeypatch):
    """Same `_pct_return` the real ledger uses — including the asset type, so
    the two books are charged on the same basis."""
    from src.performance.spread import _pct_return
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()
    _prices(monkeypatch, {"GLD": 330.0})
    ht.update_hypothetical_trades()

    t = ht._load()[0]
    assert t["entry_price"] == 300.0, "entry must not move on a refresh"
    assert t["current_price"] == 330.0
    assert t["return_pct"] == pytest.approx(
        round(_pct_return("BUY", 300.0, 330.0, "COMMODITY"), 3))


def test_a_short_gains_when_price_falls(monkeypatch):
    _config(monkeypatch, "NVDA:SELL")
    _prices(monkeypatch, {"NVDA": 100.0})
    ht.update_hypothetical_trades()
    _prices(monkeypatch, {"NVDA": 90.0})
    ht.update_hypothetical_trades()
    assert ht._load()[0]["return_pct"] > 0


def test_a_missing_price_leaves_the_previous_mark_intact(monkeypatch):
    """A feed hiccup must not zero the position's return."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()
    _prices(monkeypatch, {"GLD": 330.0})
    ht.update_hypothetical_trades()
    marked = ht._load()[0]["return_pct"]

    _prices(monkeypatch, {"GLD": None})
    ht.update_hypothetical_trades()
    t = ht._load()[0]
    assert t["return_pct"] == marked and t["current_price"] == 330.0


# ── re-anchoring ────────────────────────────────────────────────────────────

def test_flipping_the_configured_direction_re_anchors(monkeypatch):
    """The one legitimate re-anchor: the position being held is now a different
    position, so keeping the old entry would report a return nobody earned."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()

    _config(monkeypatch, "GLD:SELL")
    _prices(monkeypatch, {"GLD": 350.0})
    ht.update_hypothetical_trades()

    t = ht._load()[0]
    assert t["action"] == "SELL" and t["direction"] == "SELL"
    assert t["entry_price"] == 350.0 and t["return_pct"] == 0.0


def test_a_flip_without_a_usable_price_changes_nothing(monkeypatch):
    """Re-anchoring at a guessed price would silently rewrite the baseline's
    cost basis — better to leave the old position until a price exists."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()

    _config(monkeypatch, "GLD:SELL")
    _prices(monkeypatch, {"GLD": None})
    ht.update_hypothetical_trades()

    t = ht._load()[0]
    assert t["action"] == "BUY" and t["entry_price"] == 300.0


def test_removing_a_ticker_keeps_its_history_but_hides_it(monkeypatch):
    """Documented behaviour: the row stays as history, is no longer refreshed,
    and drops out of the email."""
    _config(monkeypatch, "GLD:BUY,NVDA:BUY")
    _prices(monkeypatch, {"GLD": 300.0, "NVDA": 100.0})
    ht.update_hypothetical_trades()

    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 310.0, "NVDA": 999.0})
    ht.update_hypothetical_trades()

    stored = {t["ticker"]: t for t in ht._load()}
    assert set(stored) == {"GLD", "NVDA"}, "history was deleted"
    assert stored["NVDA"]["current_price"] == 100.0, "de-configured row was refreshed"
    assert [t["ticker"] for t in ht._active_trades()] == ["GLD"]


def test_no_config_is_a_no_op(monkeypatch):
    monkeypatch.setattr(settings, "enable_hypothetical_trades", False)
    ht.update_hypothetical_trades()
    assert ht._load() == []


# ── the email payload ───────────────────────────────────────────────────────

def test_disabled_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(settings, "enable_hypothetical_trades", False)
    assert ht.get_hypothetical_performance_for_email() == {}


def test_enabled_but_unopened_returns_the_skeleton(monkeypatch):
    _config(monkeypatch, "GLD:BUY")
    out = ht.get_hypothetical_performance_for_email()
    assert out["trades"] == [] and out["total"] is None
    assert out["buys"] is None and out["sells"] is None
    assert out["config"] == [("GLD", "BUY")]


def test_payload_splits_buys_and_sells(monkeypatch):
    _config(monkeypatch, "GLD:BUY,NVDA:SELL")
    _prices(monkeypatch, {"GLD": 300.0, "NVDA": 100.0})
    ht.update_hypothetical_trades()
    _prices(monkeypatch, {"GLD": 330.0, "NVDA": 90.0})       # both winners
    ht.update_hypothetical_trades()

    out = ht.get_hypothetical_performance_for_email()
    assert out["total"]["trades"] == 2
    assert out["buys"]["trades"] == 1 and out["sells"]["trades"] == 1
    assert out["total"]["win_rate"] == 100.0
    assert out["total"]["best"] >= out["total"]["worst"]
    assert [t["ticker"] for t in out["trades"]] == ["GLD", "NVDA"]   # sorted
    assert all("days_held" in t for t in out["trades"])


def test_win_rate_is_gross_while_returns_stay_cost_adjusted(monkeypatch):
    """The system-wide convention. A position whose direction is right but whose
    move is smaller than the round trip is a WIN carrying a NEGATIVE return."""
    _config(monkeypatch, "GLD:BUY")
    _prices(monkeypatch, {"GLD": 300.0})
    ht.update_hypothetical_trades()
    _prices(monkeypatch, {"GLD": 300.05})        # up, but by less than the spread
    ht.update_hypothetical_trades()

    out = ht.get_hypothetical_performance_for_email()
    assert out["total"]["win_rate"] == 100.0, "gross direction was right"
    assert out["total"]["avg_return"] < 0, "cost-adjusted return must stay negative"


def test_load_is_fail_soft(monkeypatch):
    """Called from the email path — a DB hiccup must not take the report down."""
    from src.db import repo
    monkeypatch.setattr(repo, "load_hypothetical",
                        lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert ht._load() == []
