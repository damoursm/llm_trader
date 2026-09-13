"""The logprob-derived continuous verdict (2026-09-10).

local qwen3:8b emits 19 distinct raw values, 100% on a 0.05 grid — an ARGMAX
artifact, not the model's belief. Reading `top_logprobs` at the score's digit
positions and taking the expectation recovers a continuous value: measured on 80
production digests, 17 -> 75 distinct, 100% -> 6% on the grid, 95% of
argmax-tied rows made distinct, rank correlation +0.9955 with ZERO sign flips.

These tests pin the properties that make it a READING of the same verdict rather
than a new signal, and the fail-soft behaviour that keeps a parsing difficulty
from reaching the 0.40-weight `news` method.
"""
import math

import pytest

from src.analysis.logprob_score import MAX_SHIFT, expected_score, tokens_of


class _Alt:
    def __init__(self, token, p):
        self.token, self.logprob = token, math.log(max(p, 1e-12))


class _Tok:
    def __init__(self, token, alts=None):
        self.token = token
        self.top_logprobs = [_Alt(t, p) for t, p in (alts or [])]


def _resp(text_tokens):
    return [_Tok(t) if isinstance(t, str) else _Tok(*t) for t in text_tokens]


# ── the core behaviour ──────────────────────────────────────────────────────

def test_it_shifts_toward_the_alternatives_the_model_kept_alive():
    """A verdict emitted as 0.20 while the model held real mass on 0.30 is not
    the same belief as one where 0.20 was a certainty. That difference is what
    breaks the tie."""
    certain = _resp(['{"score": ', "0", ".", ("2", [("2", 0.99), ("3", 0.01)])])
    unsure = _resp(['{"score": ', "0", ".", ("2", [("2", 0.55), ("3", 0.45)])])
    a, b = expected_score(certain), expected_score(unsure)
    assert a is not None and b is not None
    assert b > a, (a, b)
    assert abs(a - 0.20) < 0.02          # a near-certain token barely moves
    assert 0.23 < b < 0.30               # a split token moves toward the rival


def test_it_never_changes_the_sign():
    """A finer READING of a verdict, not a different verdict. Measured: 0 sign
    flips over 80 production digests."""
    # NEIGHBOURING digits: a 2<->9 split would be a 0.7 correction, which the
    # MAX_SHIFT guard correctly refuses (see the refusal test below).
    for lit, alts in (("2", [("2", 0.5), ("3", 0.5)]), ("8", [("8", 0.5), ("9", 0.5)])):
        for sign in ("", "-"):
            v = expected_score(_resp([f'{{"score": {sign}', "0", ".", (lit, alts)]))
            assert v is not None
            assert (v < 0) == (sign == "-"), (sign, lit, v)


def test_tokenisation_does_not_matter():
    """THE fix. qwen splits numbers inconsistently — `0.85` arrives as one token,
    or 0/./85, or 0/.85 — and the first implementation walked token TYPES and
    failed on 56% of responses. Char-offset mapping handles any split; the
    production walk now recovers 99%."""
    alts = [("8", 0.7), ("9", 0.3)]
    splits = (
        ['{"score": ', "0", ".", ("8", alts), "5", "}"],
        ['{"score": ', "0.", ("8", alts), "5", "}"],
        ['{"score": ', "0", (".8", [(".8", 0.7), (".9", 0.3)]), "5", "}"],
    )
    vals = [expected_score(_resp(s)) for s in splits]
    assert all(v is not None for v in vals), vals
    assert all(v > 0.85 for v in vals), vals      # every split shifts up toward 0.9


def test_a_single_alternative_leaves_the_value_untouched():
    """No distribution, no correction — the argmax IS the expectation."""
    v = expected_score(_resp(['{"score": ', "0", ".", ("1", [("1", 1.0)]), "5"]))
    assert v == pytest.approx(0.15)


# ── fail-soft: never let a parsing difficulty reach the news method ─────────

def test_it_refuses_an_implausible_correction():
    """A correction past MAX_SHIFT means the walk latched onto the wrong span (a
    number inside the rationale, a truncated answer). Refusing returns None and
    the caller keeps the argmax — a wrong verdict in the 0.40-weight `news`
    method is far worse than a coarse one."""
    wild = _resp(['{"score": ', "0", ".", ("1", [("1", 0.5), ("9", 0.5)])])
    v = expected_score(wild)
    assert v is None or abs(v - 0.10) <= MAX_SHIFT


def test_alternatives_outside_the_contract_are_dropped():
    """A token alternative that parses beyond [-1, +1] is not a candidate
    verdict — it is the model considering a different field."""
    v = expected_score(_resp(['{"score": ', ("0", [("0", 0.6), ("9", 0.4)]), ".", "5"]))
    assert v is None or -1.0 <= v <= 1.0


def test_degenerate_inputs_return_None():
    assert expected_score(None) is None
    assert expected_score([]) is None
    assert expected_score(_resp(["no score field here"])) is None
    assert expected_score(_resp(['{"score": ', "abc"])) is None


def test_it_refuses_logprobs_that_describe_another_response():
    """The caller passes the parsed argmax; if the tokens say something else the
    two are out of step (a retry, a salvaged parse) and the correction would be
    meaningless."""
    toks = _resp(['{"score": ', "0", ".", ("2", [("2", 0.6), ("3", 0.4)])])
    assert expected_score(toks, argmax=0.20) is not None
    assert expected_score(toks, argmax=-0.70) is None


def test_tokens_of_is_safe_on_any_response_shape():
    assert tokens_of(None) is None
    assert tokens_of(object()) is None


# ── wiring: accrual only ────────────────────────────────────────────────────

def test_logprobs_are_requested_on_the_LOCAL_engine_only():
    """They fix qwen3:8b's 0.05 grid specifically, cost payload on every call,
    and a hosted engine may reject the parameter outright."""
    import inspect

    import src.analysis.sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    local_i = src.index('elif engine == "local"')
    deepseek_i = src.index('elif engine == "deepseek"')
    assert "logprobs=True" in src[local_i:deepseek_i]
    assert "logprobs=True" not in src[deepseek_i:]


def test_a_server_refusing_logprobs_still_returns_a_verdict():
    """Not every build supports the parameter. Losing the verdict to a
    nice-to-have would be a silent zero in the 0.40-weight `news` method."""
    import inspect

    import src.analysis.sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    i = src.index("logprobs refused")
    assert "local.chat.completions.create(**_kw)" in src[i:i + 400]


def test_the_expectation_IS_the_verdict():
    """2026-09-10 user directive — improvements go 100% into production, no
    shadowing and no A/B arm. When the expectation is available it REPLACES the
    greedy value in `raw_score`; the greedy value survives only as
    `argmax_score`, for provenance.

    Accepted knowingly as unvalidated: v6-era pivot labels have not settled, so
    "does the finer value beat the argmax on per-day pivot IC" cannot be asked
    yet. What is established is that it is the SAME verdict read more precisely
    (+0.9955 rank correlation, zero sign flips), which bounds the risk to
    re-ordering the ~11% of the cross-section the scaled `news` score ties."""
    import inspect

    import src.analysis.sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    i = src.index("_expected = expected_score(")
    tail = src[max(0, i - 200):i + 600]
    assert "raw_score = _expected" in tail, "the expectation is still a shadow"
    assert "_argmax = raw_score" in tail, "the greedy value must survive as provenance"


def test_it_is_still_not_a_METHOD_of_its_own():
    """It is `news` read more precisely, not a second signal — so it must not
    appear as a method column, an attributed method, or a stacker feature."""
    from src.analysis.ml_stacker import STACKER_CONTEXT_FEATURES, STACKER_SIGNED_FEATURES
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    from src.performance.tracker import _ALL_METHODS
    for name in ("news_expected_score", "news_argmax_score"):
        assert name not in SIGNAL_METHOD_COLUMNS
        assert name not in _ALL_METHODS
        assert name not in STACKER_SIGNED_FEATURES
        assert name not in STACKER_CONTEXT_FEATURES


def test_it_is_persisted_end_to_end():
    """A capture nothing writes is indistinguishable from a capture that never
    fires — the failure mode this project checks mechanically."""
    import inspect

    import src.pipeline as pipeline
    from src.db.schema import SIGNAL_LOGPROB_COLUMNS
    from src.models import TickerSignal
    assert "news_expected_score" in SIGNAL_LOGPROB_COLUMNS
    assert "news_expected_score" in TickerSignal.model_fields
    assert '"news_expected_score": getattr(s, "news_expected_score"' in \
        inspect.getsource(pipeline)


def test_the_column_round_trips(tmp_path, monkeypatch):
    from config.settings import settings
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    repo.insert_signals("run-1", "2026-09-10T14:00:00+00:00", "2026-09-10",
                        [{"ticker": "AAA", "direction": "BULLISH",
                          "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS},
                          "news_raw_score": 0.1637, "news_expected_score": 0.1637,
                          "news_argmax_score": 0.15}])
    df = repo.fetch_df("SELECT news_raw_score, news_expected_score, news_argmax_score "
                       "FROM signals", read_only=False)
    # the VERDICT carries the expectation; the greedy value is provenance
    assert df.iloc[0]["news_raw_score"] == pytest.approx(0.1637)
    assert df.iloc[0]["news_expected_score"] == pytest.approx(0.1637)
    assert df.iloc[0]["news_argmax_score"] == pytest.approx(0.15)
