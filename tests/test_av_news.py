"""Alpha Vantage NEWS_SENTIMENT — one batched call, per-ticker sentiment mapped
onto provider_insights so the hybrid scores those tickers WITHOUT an LLM call.
Off by default (free tier ~25 req/day); cache patched out for hermetic tests.
"""

import src.data.provider_news as pn


class _Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def _enable(monkeypatch):
    monkeypatch.setattr(pn.settings, "enable_alpha_vantage_news", True)
    monkeypatch.setattr(pn.settings, "alpha_vantage_key", "x")
    monkeypatch.setattr(pn.settings, "alpha_vantage_news_max_tickers", 50)
    monkeypatch.setattr(pn, "_load_av_cache", lambda: None)     # force a fetch
    monkeypatch.setattr(pn, "_save_av_cache", lambda a: None)   # no filesystem writes


def test_label_mapping():
    assert pn._av_label("Bullish") == "bullish"
    assert pn._av_label("Somewhat-Bullish") == "bullish"
    assert pn._av_label("Somewhat-Bearish") == "bearish"
    assert pn._av_label("Neutral") == "neutral"
    assert pn._av_label("garbage") is None


def test_disabled_or_no_key(monkeypatch):
    monkeypatch.setattr(pn.settings, "enable_alpha_vantage_news", False)
    assert pn.fetch_alpha_vantage_news(["AAPL"]) == []
    monkeypatch.setattr(pn.settings, "enable_alpha_vantage_news", True)
    monkeypatch.setattr(pn.settings, "alpha_vantage_key", "")
    assert pn.fetch_alpha_vantage_news(["AAPL"]) == []


def test_parses_ticker_sentiment_into_insights(monkeypatch):
    _enable(monkeypatch)
    feed = {"feed": [{
        "title": "Apple up", "url": "http://a/1", "time_published": "20260619T120000",
        "summary": "s", "source": "Zacks",
        "ticker_sentiment": [
            {"ticker": "AAPL", "ticker_sentiment_label": "Bullish"},
            {"ticker": "ZZZZ", "ticker_sentiment_label": "Bearish"},   # outside universe
        ],
    }]}
    monkeypatch.setattr(pn.httpx, "get", lambda *a, **k: _Resp(feed))
    arts = pn.fetch_alpha_vantage_news(["AAPL"])
    assert len(arts) == 1
    a = arts[0]
    assert a.tickers == ["AAPL"]                              # ZZZZ filtered out
    assert a.provider_insights == {"AAPL": "bullish"}
    assert a.provider_sentiment_source == "alphavantage"


def test_article_not_touching_universe_is_skipped(monkeypatch):
    _enable(monkeypatch)
    feed = {"feed": [{"title": "x", "url": "http://a/2", "time_published": "20260619T120000",
                      "ticker_sentiment": [{"ticker": "ZZZZ", "ticker_sentiment_label": "Bullish"}]}]}
    monkeypatch.setattr(pn.httpx, "get", lambda *a, **k: _Resp(feed))
    assert pn.fetch_alpha_vantage_news(["AAPL"]) == []


def test_quota_note_returns_empty(monkeypatch):
    # AV signals the 25/day quota via an Information string (HTTP 200, no 'feed').
    _enable(monkeypatch)
    monkeypatch.setattr(pn.httpx, "get",
                        lambda *a, **k: _Resp({"Information": "rate limit is 25 requests/day"}))
    assert pn.fetch_alpha_vantage_news(["AAPL"]) == []
