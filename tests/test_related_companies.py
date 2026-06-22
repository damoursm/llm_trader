"""Related-company peer discovery (Massive related-companies graph)."""

import src.data.polygon_client as pc
import src.data.related_companies as rc


def test_get_related_companies(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    monkeypatch.setattr(pc, "_get",
                        lambda path, params=None: {"results": [{"ticker": "MSFT"}, {"ticker": "GOOGL"}, {}]})
    assert pc.get_related_companies("AAPL") == ["MSFT", "GOOGL"]


def test_get_related_companies_no_key(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "")
    assert pc.get_related_companies("AAPL") == []


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_cache_path", lambda: tmp_path / "rel.json")
    monkeypatch.setattr(rc.settings, "enable_related_discovery", True)
    monkeypatch.setattr(rc.settings, "enable_fetch_data", True)
    monkeypatch.setattr(rc.polygon_client, "is_available", lambda: True)


def test_discover_dedupes_excludes_seeds(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    graph = {"AAPL": ["MSFT", "GOOGL", "AAPL"], "MSFT": ["GOOGL", "NVDA"]}
    monkeypatch.setattr(rc.polygon_client, "get_related_companies", lambda t: graph.get(t, []))
    out = rc.discover_related_tickers(["AAPL", "MSFT"], max_results=10)
    assert out == ["GOOGL", "NVDA"]   # seeds excluded, GOOGL deduped


def test_discover_respects_cap(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(rc.polygon_client, "get_related_companies", lambda t: ["AA", "BB", "CC", "DD"])
    assert rc.discover_related_tickers(["AAPL"], max_results=2) == ["AA", "BB"]


def test_discover_disabled(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(rc.settings, "enable_related_discovery", False)
    assert rc.discover_related_tickers(["AAPL"]) == []
