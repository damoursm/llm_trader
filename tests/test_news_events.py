"""News-event dataset (2026-08-15): catalyst capture, persistence, backfill and
the event-study aggregation.

Pins: the taxonomy is interpolated into the prompt (one source of truth), the
LLM's catalyst survives parse → cache → TickerSignal → signals row, the
backfill upsert is idempotent per (ticker, signal_date), and the aggregation
orients returns by the news direction."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import settings
from src.models import NewsArticle


def _art(url="u1", title="headline"):
    return NewsArticle(title=title, summary="material catalyst " * 5, url=url,
                       source="Reuters", published_at=datetime.now(timezone.utc))


def _stub_deepseek(calls, payload):
    class _Completions:
        def create(self, **kw):
            calls.append(1)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=payload))])
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


# ── taxonomy ↔ prompt ───────────────────────────────────────────────────────

def test_taxonomy_is_interpolated_into_the_prompt():
    from src.analysis.sentiment import NEWS_CATALYST_TYPES, _SENTIMENT_PREFIX
    assert "__CATALYST_TYPES__" not in _SENTIMENT_PREFIX       # placeholder resolved
    for t in NEWS_CATALYST_TYPES:
        assert t in _SENTIMENT_PREFIX
    assert '"catalyst"' in _SENTIMENT_PREFIX
    # sane taxonomy: lowercase snake, unique, with the two sentinels
    assert len(set(NEWS_CATALYST_TYPES)) == len(NEWS_CATALYST_TYPES)
    assert all(t == t.lower() and " " not in t for t in NEWS_CATALYST_TYPES)
    assert "none" in NEWS_CATALYST_TYPES and "other" in NEWS_CATALYST_TYPES


def test_normalize_catalyst():
    from src.analysis.sentiment import normalize_catalyst
    assert normalize_catalyst("earnings") == "earnings"
    assert normalize_catalyst(" MA_Deal ") == "ma_deal"
    assert normalize_catalyst("fda-clinical") == "fda_clinical"
    assert normalize_catalyst("alien_invasion") == "other"     # invented class kept as other
    assert normalize_catalyst(None) is None                    # not captured ≠ "none"
    assert normalize_catalyst("") is None


def test_parse_response_carries_catalyst():
    from src.analysis.sentiment import _parse_response
    s, r, c = _parse_response('{"rationale": "x", "catalyst": "guidance", "score": 0.41}')
    assert (s, c) == (0.41, "guidance")
    # pre-v4 shape (no catalyst) → None, never a crash
    s, r, c = _parse_response('{"score": 0.5, "rationale": "x"}')
    assert (s, c) == (0.5, None)
    # salvage path recovers the catalyst too
    s, r, c = _parse_response('{"rationale": "trunc", "catalyst": "distress", "score": -0.62, ')
    assert (s, c) == (-0.62, "distress")


# ── analyse_sentiment meta + cache round-trip ───────────────────────────────

def test_analyse_sentiment_meta_and_cache_roundtrip(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []
    payload = '{"rationale": "beat", "catalyst": "earnings", "score": 0.63}'
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls, payload))
    sent.reset_sentiment_providers()
    arts = [_art()]
    s1, _, m1 = sent.analyse_sentiment("AAA", arts)
    s2, _, m2 = sent.analyse_sentiment("AAA", arts)            # cache hit
    assert len(calls) == 1
    assert m1["catalyst"] == "earnings" and m1["raw_score"] == pytest.approx(0.63)
    assert m2["catalyst"] == "earnings"                        # catalyst survives the cache
    assert s1 == s2


def test_analyse_sentiment_empty_paths_return_empty_meta():
    from src.analysis.sentiment import analyse_sentiment
    s, r, meta = analyse_sentiment("AAA", [])
    assert (s, meta) == (0.0, {})


# ── aggregator → TickerSignal → signals row ─────────────────────────────────

# The same every-optional-layer-off harness as tests/test_buy_sell_split.py.
_OFF = [
    "enable_sentiment_velocity", "enable_technical_analysis", "enable_options_flow",
    "enable_sec_filings", "enable_insider_trades", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_ic_weights", "enable_winrate_method_filter",
    "enable_market_mode_switching", "enable_catalyst_timing",
    "enable_multi_timeframe_signals", "enable_extended_gap",
    "enable_trend_predictability_methods", "enable_family_agreement",
    "enable_tape_confirmation",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_rsi2_rev", "enable_dloc_rev", "enable_ml_ohlcv",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


@pytest.fixture
def news_only(monkeypatch):
    import src.signals.aggregator as agg
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", False)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    return agg


def test_catalyst_lands_on_the_signal(news_only, monkeypatch):
    agg = news_only
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None:
                        (0.6, "beat", {"catalyst": "earnings", "raw_score": 0.71}))
    s = agg.build_signals(["TST"], articles=[], snapshots=[])[0]
    assert s.news_catalyst == "earnings"
    assert s.news_raw_score == pytest.approx(0.71)


def test_legacy_two_tuple_sentiment_still_works(news_only, monkeypatch):
    """Arity tolerance: a 2-tuple double (every pre-v4 test fixture) must keep
    building signals — meta is additive, never load-bearing."""
    agg = news_only
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.6, "n"))
    s = agg.build_signals(["TST"], articles=[], snapshots=[])[0]
    assert s.news_catalyst is None
    assert s.news_raw_score is None


# ── schema + persistence wiring ─────────────────────────────────────────────

def test_news_event_schema_wiring():
    from src.db.repo import _SIGNAL_COLS
    from src.db.schema import SIGNAL_NEWS_EVENT_COLUMNS, _ADD_COLUMNS
    from src.models import TickerSignal
    migrated = {(t, c) for t, c, _ in _ADD_COLUMNS}
    for col, _t in SIGNAL_NEWS_EVENT_COLUMNS:
        assert ("signals", col) in migrated
        assert col in TickerSignal.model_fields
        assert col in _SIGNAL_COLS
    assert dict(SIGNAL_NEWS_EVENT_COLUMNS)["news_catalyst"] == "VARCHAR"


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def test_insert_signals_roundtrips_catalyst(tmp_db):
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    row = {"ticker": "AAPL", "type": "STOCK", "direction": "BULLISH",
           "combined_score": 0.4, "confidence": 0.8, "n_methods_agreeing": 1,
           "dominant_method": "news", "price": 100.0,
           "news_catalyst": "guidance", "news_raw_score": 0.55,
           "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}
    repo.insert_signals("run-1", "2026-08-15T14:00:00+00:00", "2026-08-15", [row])
    df = repo.fetch_df("SELECT news_catalyst, news_raw_score FROM signals", read_only=False)
    assert df.iloc[0]["news_catalyst"] == "guidance"
    assert df.iloc[0]["news_raw_score"] == pytest.approx(0.55)


def test_backfill_upsert_idempotent(tmp_db):
    from src.db import repo
    r = {"ticker": "AAPL", "signal_date": "2026-07-01", "catalyst": "earnings",
         "headline_count": 3, "top_headline": "h", "classifier_version": "bf1",
         "classified_at": "2026-08-15T00:00:00+00:00"}
    repo.insert_news_event_backfill([r])
    repo.insert_news_event_backfill([{**r, "catalyst": "guidance"}])   # re-classify
    df = repo.fetch_df("SELECT * FROM news_event_backfill", read_only=False)
    assert len(df) == 1
    assert df.iloc[0]["catalyst"] == "guidance"                        # replaced, not duplicated


# ── event-study aggregation ─────────────────────────────────────────────────

def _ev(catalyst, direction, ret, magnitude=0.5):
    return {"ticker": "T", "signal_date": "2026-08-01", "catalyst": catalyst,
            "catalyst_source": "live", "direction": direction,
            "magnitude": magnitude, "fwd_ret_pivot": ret, "news": 0.0,
            "news_raw_score": None, "era": "current"}


def test_catalyst_return_table_orients_by_direction():
    from src.analysis.news_events import catalyst_return_table
    ev = pd.DataFrame(
        [_ev("earnings", "bull", 2.0)] * 5        # bullish news → +2% = oriented +2
        + [_ev("earnings", "bear", 2.0)] * 5      # bearish news → +2% = oriented −2
        + [_ev("distress", "bear", -3.0)] * 10)   # bearish news → −3% = oriented +3
    out = catalyst_return_table(ev, min_n=5)
    e = out[out.catalyst == "earnings"].iloc[0]
    assert e["oriented_mean"] == pytest.approx(0.0)            # +2 and −2 cancel
    assert e["bull_mean"] == pytest.approx(2.0)
    assert e["bear_mean"] == pytest.approx(2.0)                # raw, unoriented
    assert e["hit_pct"] == pytest.approx(50.0)
    d = out[out.catalyst == "distress"].iloc[0]
    assert d["oriented_mean"] == pytest.approx(3.0)
    assert d["hit_pct"] == pytest.approx(100.0)


def test_catalyst_return_table_min_n_and_untyped():
    from src.analysis.news_events import catalyst_return_table
    ev = pd.DataFrame([_ev("ma_deal", "bull", 1.0)] * 3        # below min_n → dropped
                      + [_ev(None, "bull", 1.0)] * 8)          # untyped bucket kept
    out = catalyst_return_table(ev, min_n=5)
    assert "ma_deal" not in set(out.catalyst)
    assert "(untyped)" in set(out.catalyst)


def test_magnitude_table_bands():
    from src.analysis.news_events import magnitude_table
    ev = pd.DataFrame([_ev("earnings", "bull", 1.0, magnitude=0.1)] * 8
                      + [_ev("earnings", "bull", 3.0, magnitude=0.9)] * 8)
    out = magnitude_table(ev, min_n=5)
    assert list(out.band) == ["minor", "major"]
    assert out[out.band == "major"].iloc[0]["oriented_mean"] == pytest.approx(3.0)


# ── backfill helpers ────────────────────────────────────────────────────────

def test_headlines_for_day_window():
    from src.analysis.news_backfill import headlines_for_day
    arts = [{"t": "2026-07-03T14:00:00Z", "h": "same-day"},
            {"t": "2026-07-02T14:00:00Z", "h": "day-before"},
            {"t": "2026-06-20T14:00:00Z", "h": "way-old"}]
    heads = headlines_for_day(arts, "2026-07-03")
    assert heads == ["same-day", "day-before"]                 # recent first, old excluded
    # empty near-window widens to the 3-day pad, then gives up
    assert headlines_for_day(arts, "2026-06-22") == ["way-old"]
    assert headlines_for_day(arts, "2026-08-01") == []


def test_classify_batch_parses_and_normalizes():
    from src.analysis.news_backfill import classify_batch
    payload = ('[{"i": 1, "catalyst": "earnings"}, {"i": 2, "catalyst": "weird_class"},'
               ' {"i": 3, "catalyst": null}]')
    client = _stub_deepseek([], payload)
    got = classify_batch([{"i": 1, "ticker": "A", "date": "2026-07-01", "headlines": ["h"]},
                          {"i": 2, "ticker": "B", "date": "2026-07-01", "headlines": ["h"]},
                          {"i": 3, "ticker": "C", "date": "2026-07-01", "headlines": ["h"]}],
                         client=client)
    assert got == {1: "earnings", 2: "other"}                  # null → absent (retried)


def test_classify_batch_bad_reply_is_empty_not_fatal():
    from src.analysis.news_backfill import classify_batch
    client = _stub_deepseek([], "sorry, cannot help with that")
    assert classify_batch([{"i": 1, "ticker": "A", "date": "d", "headlines": ["h"]}],
                          client=client) == {}


def test_catalyst_class_reaches_the_synthesis_prompt(monkeypatch):
    """2026-08-16: the v4 catalyst capture rides the synthesis prompt — a
    [class] tag on the news line plus the 1c measured-base-rates instruction."""
    import src.analysis.claude_analyst as ca
    from src.models import TickerSignal
    monkeypatch.setattr(ca.settings, "dual_case_synthesis_share", 0.0)
    monkeypatch.setattr(ca.settings, "blind_synthesis_share", 0.0)
    captured = {}
    monkeypatch.setattr(ca, "_call_claude_analyst",
                        lambda prompt, model=None: captured.update(prompt=prompt) or "[]")
    sig = TickerSignal(ticker="KO", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.42, technical_score=0.0,
                       rationale="guidance raise", news_catalyst="guidance")
    try:
        ca.generate_recommendations([sig], force_engine="anthropic")
    except Exception:
        pass
    p = captured.get("prompt", "")
    assert "News sentiment=+0.42 [guidance]" in p
    assert "1c. News catalyst CLASSES" in p
    # a "none" class is omitted from the tag (no event ≠ a class), and a
    # none-only cross-section doesn't render the 1c instruction at all
    captured.clear()
    sig2 = sig.model_copy(update={"news_catalyst": "none"})
    try:
        ca.generate_recommendations([sig2], force_engine="anthropic")
    except Exception:
        pass
    p2 = captured.get("prompt", "")
    assert "News sentiment=+0.42 |" in p2          # no tag before the pipe
    assert "1c. News catalyst CLASSES" not in p2


def test_news_part_strips_method_fragments():
    from src.analysis.news_backfill import _JUNK_RATIONALE_PREFIXES, _news_part
    full = ("Q2 beat with raised guidance reprices estimates. | "
            "Technical score: +0.38 | Put/call signal: -0.12")
    assert _news_part(full) == "Q2 beat with raised guidance reprices estimates."
    # a rationale with no news part is junk-filtered by prefix
    assert _news_part("Technical score: +0.38").startswith(_JUNK_RATIONALE_PREFIXES)
    assert "Provider sentiment (polygon): 3 articles".startswith(_JUNK_RATIONALE_PREFIXES)
