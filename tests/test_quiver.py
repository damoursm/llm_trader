"""Quiver alt-data parsers: congress→InsiderTrade, contracts/lobbying/dark-pool→NewsArticle."""

import types
from datetime import date

import src.data.quiver as q


def _enable(monkeypatch):
    monkeypatch.setattr(q.settings, "quiver_api_key", "x")
    for flag in ("enable_quiver_congress", "enable_quiver_gov_contracts",
                 "enable_quiver_lobbying", "enable_quiver_offexchange"):
        monkeypatch.setattr(q.settings, flag, True)
    monkeypatch.setattr(q.settings, "quiver_lookback_days", 30)
    # tracked_politicians_list is a computed property → patch on the class to [].
    monkeypatch.setattr(type(q.settings), "tracked_politicians_list",
                        property(lambda self: []), raising=False)
    monkeypatch.setattr(q, "time", types.SimpleNamespace(sleep=lambda *a, **k: None))


def test_congress_maps_to_insider_trades(monkeypatch):
    _enable(monkeypatch)
    today = date.today().isoformat()
    rows = [
        {"Ticker": "AAPL", "Representative": "Nancy Pelosi", "Transaction": "Purchase",
         "Range": "$1,001 - $15,000", "TransactionDate": today, "ReportDate": today,
         "House": "Representatives"},
        {"Ticker": "NVDA", "Representative": "Tommy Tuberville", "Transaction": "Sale (Full)",
         "Range": "$50,001 - $100,000", "TransactionDate": today, "ReportDate": today,
         "House": "Senate"},
        {"Ticker": "OLD", "Representative": "X", "Transaction": "Purchase",
         "Range": "$1,001 - $15,000", "TransactionDate": "2020-01-01", "ReportDate": "2020-01-01",
         "House": "Senate"},
    ]
    monkeypatch.setattr(q, "_get", lambda path: rows)
    trades = q.fetch_congress_trades()
    assert len(trades) == 2                                # OLD excluded by lookback
    by = {t.ticker: t for t in trades}
    assert by["AAPL"].transaction_type == "purchase" and by["AAPL"].is_bullish is True
    assert by["AAPL"].trader_type == "politician" and by["AAPL"].role == "Representative"
    assert by["NVDA"].transaction_type == "sale" and by["NVDA"].is_bullish is False
    assert by["NVDA"].role == "Senator"


def test_gov_contracts_filtered_to_universe(monkeypatch):
    _enable(monkeypatch)
    today = date.today().isoformat()
    rows = [
        {"Ticker": "AAPL", "Date": today, "Amount": "5000000", "Agency": "Dept of Defense"},
        {"Ticker": "ZZZZ", "Date": today, "Amount": "1000", "Agency": "X"},
    ]
    monkeypatch.setattr(q, "_get", lambda path: rows)
    arts = q.fetch_gov_contracts(["AAPL"])
    assert len(arts) == 1                                  # ZZZZ not in universe
    assert "AAPL" in arts[0].title and "$5,000,000" in arts[0].title
    assert arts[0].source == "Quiver Gov Contracts" and arts[0].tickers == ["AAPL"]


def test_lobbying_maps_to_article(monkeypatch):
    _enable(monkeypatch)
    today = date.today().isoformat()
    rows = [{"Ticker": "AAPL", "Date": today, "Amount": "250000", "Issue": "AI regulation"}]
    monkeypatch.setattr(q, "_get", lambda path: rows)
    arts = q.fetch_lobbying(["AAPL"])
    assert len(arts) == 1 and arts[0].source == "Quiver Lobbying"
    assert "$250,000" in arts[0].title


def test_offexchange_emits_on_dpi_shift(monkeypatch):
    _enable(monkeypatch)
    from datetime import timedelta
    base = date.today()
    # NEWEST-FIRST like the real Quiver API: 3 recent high-DPI bars, then 7 older
    # low-DPI bars. The fetcher must sort by date so "recent" = the high ones.
    series = ([{"Date": (base - timedelta(days=i)).isoformat(), "DPI": 0.55} for i in range(3)] +
              [{"Date": (base - timedelta(days=3 + i)).isoformat(), "DPI": 0.40} for i in range(7)])
    monkeypatch.setattr(q, "_get", lambda path: series)
    arts = q.fetch_offexchange(["AAPL"])
    assert len(arts) == 1 and arts[0].source == "Quiver Dark Pool"
    assert "accumulation" in arts[0].title.lower()   # recent 0.55 > baseline 0.40


def test_disabled_or_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr(q.settings, "quiver_api_key", "")
    assert q.fetch_congress_trades() == []
    assert q.fetch_gov_contracts(["AAPL"]) == []
    assert q.fetch_lobbying(["AAPL"]) == []
    assert q.fetch_offexchange(["AAPL"]) == []
