"""StockTwits crowd sentiment → one synthetic per-ticker chatter-summary article
(LLM-scored, like Reddit). Token-gated (the public endpoint 403s without auth);
cache + sleep patched out for hermetic tests.
"""

import src.data.stocktwits as st


class _Resp:
    def __init__(self, status, msgs):
        self.status_code = status
        self._m = msgs

    def raise_for_status(self):
        pass

    def json(self):
        return {"messages": self._m}


def _msgs(bull, bear):
    return ([{"entities": {"sentiment": {"basic": "Bullish"}}}] * bull +
            [{"entities": {"sentiment": {"basic": "Bearish"}}}] * bear)


def _enable(monkeypatch):
    monkeypatch.setattr(st.settings, "enable_stocktwits", True)
    monkeypatch.setattr(st.settings, "stocktwits_access_token", "tok")
    monkeypatch.setattr(st.settings, "stocktwits_max_tickers", 30)
    monkeypatch.setattr(st, "_load_cache", lambda: None)
    monkeypatch.setattr(st, "_save_cache", lambda a: None)
    monkeypatch.setattr(st.time, "sleep", lambda *a, **k: None)


def test_disabled_or_no_token(monkeypatch):
    monkeypatch.setattr(st.settings, "enable_stocktwits", False)
    assert st.fetch_stocktwits_sentiment(["AAPL"]) == []
    monkeypatch.setattr(st.settings, "enable_stocktwits", True)
    monkeypatch.setattr(st.settings, "stocktwits_access_token", "")
    assert st.fetch_stocktwits_sentiment(["AAPL"]) == []


def test_summary_article_per_ticker(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(st.httpx, "get", lambda *a, **k: _Resp(200, _msgs(7, 3)))
    arts = st.fetch_stocktwits_sentiment(["AAPL"])
    assert len(arts) == 1
    a = arts[0]
    assert a.tickers == ["AAPL"] and a.source == "StockTwits"
    assert "bullish" in a.title.lower() and "70%" in a.title


def test_low_message_count_skipped(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(st.httpx, "get", lambda *a, **k: _Resp(200, _msgs(2, 1)))  # < _MIN_MESSAGES
    assert st.fetch_stocktwits_sentiment(["AAPL"]) == []


def test_auth_rejection_stops_early(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(st.httpx, "get", lambda *a, **k: _Resp(403, []))
    assert st.fetch_stocktwits_sentiment(["AAPL", "MSFT"]) == []


def test_skips_non_equity_symbols(monkeypatch):
    _enable(monkeypatch)
    seen = []
    monkeypatch.setattr(st.httpx, "get",
                        lambda url, **k: (seen.append(url) or _Resp(200, _msgs(6, 1))))
    st.fetch_stocktwits_sentiment(["^VIX", "GC=F", "AAPL"])
    assert all("VIX" not in u and "GC" not in u for u in seen)   # indices/futures skipped
