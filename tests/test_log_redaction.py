"""Tests for src.log_redaction — secrets must never reach a log sink in plaintext."""

import io

import pytest
from loguru import logger

from src.log_redaction import redact_secrets, redaction_filter

# The exact leak vector seen in logs/: httpx echoes the FRED URL (api_key in the
# query string) into its exception, which the codebase logs via f"... {e}".
_FRED_URL = (
    "[fred] DFF fetch failed: Client error '429' for url "
    "'https://api.stlouisfed.org/fred/series/observations?series_id=DFF&"
    "api_key=062d3a054887d0a7daeb67582c43a1e5&file_type=json'"
)
_LIVE_KEY = "062d3a054887d0a7daeb67582c43a1e5"


@pytest.mark.parametrize("text,secret", [
    (_FRED_URL,                                   _LIVE_KEY),
    ("GET /x?apiKey=ABCD1234EFGH5678 done",       "ABCD1234EFGH5678"),
    ('headers={"token": "sk-abc12345xyz9"}',      "sk-abc12345xyz9"),
    ("DEEPSEEK_API_KEY=sk-deadbeefcafe0001 set",  "sk-deadbeefcafe0001"),
    ("?access_token=longlivedtokenvalue123&x=1",  "longlivedtokenvalue123"),
])
def test_secret_is_redacted(text, secret):
    out = redact_secrets(text)
    assert secret not in out
    assert "***REDACTED***" in out


@pytest.mark.parametrize("text", [
    "Found 5 tokens; secret: ok",        # value < 8 chars — prose, not a secret
    "api_key=12 short",                  # value < 8 chars
    "Generated 42 recommendations via deepseek",
    "[move] MOVE=88.4 (NORMAL) | 5d change=+1.2pt",
])
def test_non_secrets_are_left_alone(text):
    assert redact_secrets(text) == text


def test_redaction_is_idempotent():
    once = redact_secrets(_FRED_URL)
    assert redact_secrets(once) == once


def test_filter_scrubs_through_a_real_loguru_sink():
    """The feature hinges on loguru honoring a message-mutating filter."""
    buf = io.StringIO()
    logger.remove()
    try:
        logger.add(buf, format="{message}", level="DEBUG", filter=redaction_filter)
        logger.warning(_FRED_URL)
        captured = buf.getvalue()
    finally:
        logger.remove()
    assert _LIVE_KEY not in captured
    assert "***REDACTED***" in captured
