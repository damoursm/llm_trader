"""Open-vocabulary ticker extraction (`src/data/ticker_extract.py`).

Feeds discovery from free text (headlines, Reddit). The asymmetry that shapes
every test here: a MISSED ticker costs one candidate, while a FALSE POSITIVE
puts a junk name into the universe, where it is scored, persisted to the panel
and can reach a trade. So the precision controls are the contract:

* `$CASHTAGS` are explicit → accepted on SEC validity alone;
* bare ALL-CAPS tokens must ALSO clear the stopword list, because many common
  English words and finance acronyms are real tickers (ALL, ARE, ON, IT, CASH,
  OPEN, REAL, CEO, ETF …) and would otherwise match in almost every headline.

The SEC universe is faked throughout — `get_valid_tickers` hits the network and
the real map decides half the assertions, which would make this suite both slow
and dependent on what the SEC published today.
"""

from __future__ import annotations

import pytest

from src.data import ticker_extract as tx


@pytest.fixture(autouse=True)
def _fake_universe(monkeypatch):
    """A small SEC universe that deliberately CONTAINS stopword collisions —
    ALL, ARE, ON, IT, CASH, OPEN, REAL and CEO are all genuinely listed symbols,
    which is exactly why the stopword filter exists."""
    universe = frozenset({
        "AAPL", "MSFT", "NVDA", "TSLA", "GME", "F", "T", "SNOW", "ARM",
        "ALL", "ARE", "ON", "IT", "CASH", "OPEN", "REAL", "CEO", "ETF", "BUY",
    })
    monkeypatch.setattr(tx, "_VALID", universe)
    monkeypatch.setattr(tx, "get_valid_tickers", lambda: universe)
    return universe


# ── cashtags ────────────────────────────────────────────────────────────────

def test_cashtags_are_accepted_on_validity_alone():
    assert tx.extract_candidate_tickers("watching $AAPL and $MSFT here") == {"AAPL", "MSFT"}


def test_cashtags_bypass_the_stopword_filter():
    """`$CASH` is an explicit, deliberate reference — the stopword list exists to
    disambiguate BARE tokens, and applying it to cashtags would drop real
    mentions of genuinely-named companies."""
    assert "CASH" in tx.extract_candidate_tickers("long $CASH into earnings")
    assert "CASH" not in tx.extract_candidate_tickers("raising CASH into earnings")


def test_cashtags_are_case_folded():
    assert tx.extract_candidate_tickers("$aapl $Msft") == {"AAPL", "MSFT"}


def test_invalid_cashtags_are_dropped():
    """A $-prefixed token that is not a listed symbol is still junk."""
    assert tx.extract_candidate_tickers("$ZZZZ $AAPL") == {"AAPL"}


# ── bare tokens ─────────────────────────────────────────────────────────────

def test_bare_tokens_must_be_valid_AND_not_stopwords():
    text = "ALL of the CEO commentary ON the ETF was REAL, but NVDA moved"
    assert tx.extract_candidate_tickers(text) == {"NVDA"}


def test_lowercase_words_are_never_candidates():
    """Only ALL-CAPS tokens are considered — 'it', 'on', 'all' in ordinary prose
    must not resolve to tickers."""
    assert tx.extract_candidate_tickers("it is all on the open real cash") == set()


def test_single_letter_tickers_are_not_matched_bare():
    """`F` and `T` are real symbols but a bare single letter is far more often
    an initial or a list marker; the token regex requires 2-5 characters."""
    assert tx.extract_candidate_tickers("F and T reported") == set()
    assert tx.extract_candidate_tickers("$F and $T reported") == {"F", "T"}


def test_tokens_longer_than_five_chars_are_not_matched():
    assert tx.extract_candidate_tickers("MICROSOFT beat") == set()


def test_mixed_cashtag_and_bare_tokens_merge():
    got = tx.extract_candidate_tickers("$GME squeeze while NVDA and TSLA rally")
    assert got == {"GME", "NVDA", "TSLA"}


# ── output shape ────────────────────────────────────────────────────────────

def test_result_is_a_deduplicated_set_per_text():
    """The return is a SET on purpose: callers count how many distinct ITEMS
    mention a ticker, so a headline repeating NVDA five times must not look like
    five independent sources."""
    got = tx.extract_candidate_tickers("NVDA NVDA $NVDA NVDA")
    assert got == {"NVDA"}
    assert isinstance(got, set)


@pytest.mark.parametrize("text", ["", None, "   ", "no tickers here at all 123 %^&"])
def test_empty_or_tickerless_text_yields_nothing(text):
    assert tx.extract_candidate_tickers(text) == set()


def test_unavailable_sec_map_yields_nothing_rather_than_everything(monkeypatch):
    """Fail CLOSED. If the SEC reference set is unavailable, validity cannot be
    checked — emitting the raw ALL-CAPS tokens would flood discovery with junk
    at exactly the moment there is no way to filter it."""
    monkeypatch.setattr(tx, "_VALID", frozenset())
    monkeypatch.setattr(tx, "get_valid_tickers", lambda: frozenset())
    assert tx.extract_candidate_tickers("$AAPL NVDA TSLA") == set()


# ── the stopword list itself ────────────────────────────────────────────────

def test_stopwords_are_uppercase_and_within_the_token_length_window():
    """A lowercase or 6+ character entry can never match `_TOKEN_RE`, so it is a
    silent no-op that reads as protection."""
    for w in tx._STOPWORDS:
        assert w == w.upper(), f"{w!r} can never match the ALL-CAPS token regex"
        assert 2 <= len(w) <= 5, f"{w!r} is outside the 2-5 char token window"


def test_the_notorious_collisions_are_covered():
    """Named explicitly because these are the ones that actually flooded runs —
    each is both an everyday word and a listed symbol."""
    for w in ("ALL", "ARE", "ON", "IT", "CASH", "OPEN", "REAL", "CEO", "ETF", "BUY"):
        assert w in tx._STOPWORDS


def test_valid_set_is_memoised(monkeypatch):
    """`get_valid_tickers` parses the whole SEC map; the module caches it in a
    process global so a per-headline call doesn't re-read it."""
    monkeypatch.setattr(tx, "_VALID", frozenset())
    calls = {"n": 0}

    def _counted():
        calls["n"] += 1
        return frozenset({"AAPL"})

    monkeypatch.setattr(tx, "get_valid_tickers", _counted)
    tx.extract_candidate_tickers("$AAPL")
    tx.extract_candidate_tickers("$AAPL")
    assert calls["n"] == 1
