"""News-family continuity pass (2026-08-14) — quantization fixes + news_shock.

Measured motivation (45-day panel): `insider` had 81% within-run tie mass (331
views sharing 62 distinct values — one value covered 3,027 rows), and `news`
had ~38% (round LLM verdicts × step scalers). The rank transform hands a tie
group one average rank, so most of both methods' cross-sections collapsed to
the median and the payoff-shaping deciles were fit on ties. These tests pin the
continuity mechanisms and the new abnormal-attention method.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from src.models import InsiderTrade, NewsArticle


def _art(hours_ago: float, source: str = "rss", title: str = "t") -> NewsArticle:
    return NewsArticle(
        title=title, summary="s", url=f"http://x/{source}/{title}/{hours_ago}",
        source=source,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def _trade(days_ago: int = 0, amount: str = "$15,001 - $50,000",
           role: str = "Director", tx: str = "purchase") -> InsiderTrade:
    return InsiderTrade(
        ticker="AAA", trader_name="X", trader_type="corporate_insider", role=role,
        transaction_type=tx, amount_range=amount,
        transaction_date=date.today() - timedelta(days=days_ago),
        disclosure_date=date.today() - timedelta(days=max(0, days_ago - 1)),
    )


# ── insider continuity ──────────────────────────────────────────────────────

def test_insider_recency_breaks_ties():
    """Same filing pattern, different WHEN → different scores. This is the tie
    the old scorer could not break (no time term at all)."""
    from src.signals.aggregator import _insider_score
    fresh = _insider_score("AAA", [_trade(days_ago=0)])[0]
    week = _insider_score("AAA", [_trade(days_ago=7)])[0]
    month = _insider_score("AAA", [_trade(days_ago=30)])[0]
    assert fresh > week > month > 0
    assert week == pytest.approx(fresh / 2, rel=0.05)      # 7d half-life
    assert _insider_score("AAA", [_trade(days_ago=60)])[0] == 0.0  # beyond max age


def test_exact_notional_beats_the_bucket():
    """The largest quantizer was self-inflicted: 13F / Form 4 / options-sweep
    rows KNOW the exact dollar value and called `_notional_to_amount_range()`
    to squeeze it into 8 buckets before scoring, so a $60k and a $95k purchase
    produced the identical number. The exact value now rides the model."""
    from src.signals.aggregator import _amount_weight_log
    bucket = "$50,001 - $100,000"
    w60 = _amount_weight_log(bucket, 60_000.0)
    w95 = _amount_weight_log(bucket, 95_000.0)
    assert w60 != w95 and w60 < w95           # same bucket, different weight
    # No exact value → the bucket midpoint (congressional/13D-G keep this path).
    assert _amount_weight_log(bucket) == _amount_weight_log(bucket, None)
    assert _amount_weight_log(bucket, 0.0) == _amount_weight_log(bucket)


def test_the_three_known_value_sources_carry_notional():
    """A regression guard: if a source stops setting notional_usd the method
    silently re-quantizes, which looks like nothing at all."""
    import inspect
    from src.data import options_flow, sec_filings
    for mod in (options_flow, sec_filings):
        src = inspect.getsource(mod)
        assert "notional_usd=" in src, f"{mod.__name__} dropped notional_usd"
    assert inspect.getsource(sec_filings).count("notional_usd=") >= 2   # 13F + Form 4


def test_disclosure_lag_weight():
    """Fast disclosure is a stronger tell; same-day sources are unaffected."""
    from src.signals.aggregator import _disclosure_lag_weight
    same = _trade(days_ago=3)                       # disclosure == transaction+... see helper
    fast = InsiderTrade(ticker="A", trader_name="r", trader_type="politician",
                        role="Senator", transaction_type="purchase",
                        amount_range="$15,001 - $50,000",
                        transaction_date=date.today() - timedelta(days=30),
                        disclosure_date=date.today() - timedelta(days=27))   # 3d lag
    slow = InsiderTrade(ticker="A", trader_name="r", trader_type="politician",
                        role="Senator", transaction_type="purchase",
                        amount_range="$15,001 - $50,000",
                        transaction_date=date.today() - timedelta(days=30),
                        disclosure_date=date.today() + timedelta(days=14))   # 44d lag
    assert _disclosure_lag_weight(fast) > _disclosure_lag_weight(slow)
    assert 0.70 < _disclosure_lag_weight(slow) < 1.0        # bounded prior
    # Sources whose two dates are identical are untouched.
    zero_lag = InsiderTrade(ticker="A", trader_name="r", trader_type="institutional",
                            role="x", transaction_type="13f_increase",
                            amount_range="unknown", transaction_date=date.today(),
                            disclosure_date=date.today())
    assert _disclosure_lag_weight(zero_lag) == 1.0


def test_persisted_scores_are_not_requantized_at_3dp():
    """The PERSISTED value is what the rank transform orders on. Both scores
    were rounded to 3dp on the TickerSignal — downstream of all the continuity
    work — which silently re-merged reads that the scorers had just separated
    (insider scores cluster near zero, where 3dp is coarse). Measured live:
    the 3dp value 0.027 alone covered 17% of nonzero insider rows."""
    import ast
    import inspect
    from src.signals import aggregator

    tree = ast.parse(inspect.getsource(aggregator))
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "TickerSignal"):
            continue
        for kw in node.keywords:
            if kw.arg not in ("sentiment_score", "insider_score"):
                continue
            v = kw.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                    and v.func.id == "round" and len(v.args) > 1
                    and isinstance(v.args[1], ast.Constant)
                    and int(v.args[1].value) < 4):
                bad.append((kw.arg, v.args[1].value))
    assert not bad, f"persisted score re-quantized too coarsely: {bad}"


def test_insider_amount_weight_is_log_scaled():
    from src.signals.aggregator import _amount_weight_log
    small = _amount_weight_log("$1,001 - $15,000")
    mid = _amount_weight_log("$100,001 - $250,000")
    big = _amount_weight_log("Over $5,000,000")
    assert 0 < small < mid < big == 1.0
    # log, not linear: the $10M/$4k dollar ratio is ~2,500x, the weight ratio ~7x.
    assert big / small < 8
    assert _amount_weight_log("garbage") == pytest.approx(0.10)


def test_insider_role_weight():
    from src.signals.aggregator import _insider_role_weight
    assert _insider_role_weight("Chief Executive Officer") == pytest.approx(1.30)
    assert _insider_role_weight("CFO") == pytest.approx(1.30)
    assert _insider_role_weight("Director") == 1.0
    assert _insider_role_weight("") == 1.0
    ceo = _insider_score_of(role="CEO")
    dirc = _insider_score_of(role="Director")
    assert ceo > dirc


def _insider_score_of(**kw):
    from src.signals.aggregator import _insider_score
    return _insider_score("AAA", [_trade(**kw)])[0]


def test_insider_epoch_registered():
    from src.signals.method_epochs import METHOD_SCORER_EPOCH
    assert METHOD_SCORER_EPOCH["insider"] >= datetime(2026, 8, 14, tzinfo=timezone.utc)
    assert METHOD_SCORER_EPOCH["news"] >= datetime(2026, 8, 14, tzinfo=timezone.utc)


# ── news continuity ─────────────────────────────────────────────────────────

def test_evidence_scale_is_continuous_in_article_age():
    """3 fresh articles ≠ 3 stale ones — the count scale said they were equal."""
    from src.analysis.sentiment import _evidence_scale, attention_mass
    n_f, fresh_mass = attention_mass([_art(1), _art(2), _art(3)])
    n_s, stale_mass = attention_mass([_art(60), _art(70, "b"), _art(80, "c")])
    assert n_f == n_s == 3
    assert fresh_mass > stale_mass
    assert _evidence_scale(fresh_mass) > _evidence_scale(stale_mass)
    assert _evidence_scale(0.0) == 0.0
    assert _evidence_scale(1e9) == 1.0


def test_diversity_scale_smooth():
    from src.analysis.sentiment import _source_diversity_scale
    vals = [_source_diversity_scale([_art(1, s) for s in
                                     [f"src{i}" for i in range(k)]])
            for k in (1, 2, 3, 4, 6)]
    assert vals[0] == pytest.approx(0.70)
    assert vals[1] == pytest.approx(0.85)
    assert vals == sorted(vals)                 # monotone
    assert 0.90 < vals[2] < vals[3] < 1.0       # no longer a hard 1.0 at k=3


def test_sentiment_cache_key_salted_by_prompt_version(monkeypatch):
    """A prompt edit must invalidate cached raw verdicts — before the salt, the
    cache kept serving old-prompt verdicts for the TTL after a prompt change."""
    import src.analysis.sentiment as sent
    arts = [_art(1)]
    k1 = sent._sentiment_cache_key("AAA", "deepseek", arts)
    monkeypatch.setattr(sent, "_SENT_PROMPT_VERSION", "test-bump")
    k2 = sent._sentiment_cache_key("AAA", "deepseek", arts)
    assert k1 != k2


def test_sentiment_prompt_has_precision_mandate():
    from src.analysis.sentiment import _SENTIMENT_PREFIX
    assert "TWO-decimal" in _SENTIMENT_PREFIX
    assert "CROSS-SECTIONALLY" in _SENTIMENT_PREFIX


# ── news_shock ──────────────────────────────────────────────────────────────

def test_news_shock_scoring():
    from src.signals.news_shock import compute_news_shock
    # 8x normal attention with bullish news → full +1.
    assert compute_news_shock(0.5, 8.0, 1.0) == pytest.approx(1.0)
    # 2x → log2(2)/3 = 1/3, signed by the news read.
    assert compute_news_shock(0.5, 2.0, 1.0) == pytest.approx(1 / 3, abs=1e-3)
    assert compute_news_shock(-0.5, 2.0, 1.0) == pytest.approx(-1 / 3, abs=1e-3)
    # Abstentions: quiet is not a fade, no news is no direction, no baseline is no verdict.
    assert compute_news_shock(0.5, 0.9, 1.0) == 0.0     # below baseline
    assert compute_news_shock(0.0, 8.0, 1.0) == 0.0     # no news direction
    assert compute_news_shock(0.5, 8.0, None) == 0.0    # no baseline yet
    assert compute_news_shock(0.5, 0.0, 1.0) == 0.0     # no articles


def test_news_shock_baseline_failsoft(monkeypatch):
    """A DB failure reads as 'no baseline' — the method abstains, never raises."""
    import src.signals.news_shock as ns
    ns.reset_cache()
    from src.db import repo

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(repo, "fetch_df", boom)
    assert ns.load_attention_baselines(force=True) == {}
    ns.reset_cache()


def test_news_shock_wiring_complete():
    """The full add-method checklist, mechanically."""
    from src.analysis.code_version import METHOD_SOURCES, unmapped_methods
    from src.db.schema import (SIGNAL_BASE_METHOD_COLUMNS,
                               SIGNAL_NEWS_ATTENTION_COLUMNS, _ADD_COLUMNS)
    from src.models import TickerSignal
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    from src.signals.agreement import FAMILY_OF
    from src.signals.aggregator import _BASE_WEIGHTS

    assert "news_shock" in _ALL_METHODS
    assert "news_shock" in SIGNAL_BASE_METHOD_COLUMNS
    assert "news_shock" in METHOD_CATEGORIES["Sentiment"]
    assert "news_shock" in METHOD_LABELS
    assert "news_shock" in METHOD_SOURCES and not unmapped_methods()
    migrated = {(t, c) for t, c, _ in _ADD_COLUMNS}
    assert ("signals", "news_shock") in migrated
    for c in SIGNAL_NEWS_ATTENTION_COLUMNS:
        assert ("signals", c) in migrated
        assert c in TickerSignal.model_fields
    assert "news_shock_score" in TickerSignal.model_fields
    # PANEL-FIRST: not weighted, not a family voter — mirrors squeeze.
    assert "news_shock" not in _BASE_WEIGHTS
    assert "news_shock" not in FAMILY_OF
