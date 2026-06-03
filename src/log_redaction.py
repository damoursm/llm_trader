"""Scrub secrets (API keys, tokens) from log output.

Several upstream libraries (httpx, requests, the Anthropic/OpenAI SDKs) place the
API key in the request URL or echo it in an exception, which the codebase then
logs verbatim via ``logger.warning(f"... {e}")``. Left alone, log files end up
with plaintext credentials in them (e.g. ``api_key=062d3a05...`` in a FRED URL).

This module redacts those values *before* any loguru sink writes them, so neither
the console nor ``logs/*.log`` ever contains a live credential. Wire it in via the
``filter=`` argument of ``logger.add`` (see ``main.setup_logging``).
"""
from __future__ import annotations

import re

# Matches a credential-looking key/value pair: a known secret parameter name,
# an ``=`` or ``:`` separator (with optional surrounding quotes/space), then the
# value up to the next delimiter. The value must be at least 8 chars so we don't
# redact prose like "token: 5 found" or "secret: ok".
_SECRET_RE = re.compile(
    r"(?i)"
    r"(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|client[_-]?secret|secret|password)"
    r"(\"?\s*[=:]\s*\"?)"
    r"([^\s&\"'<>),;]{8,})"
)

_MASK = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Return *text* with any credential-looking values masked.

    >>> redact_secrets("GET https://api.x/obs?series_id=DFF&api_key=062d3a054887d0a7daeb67582c43a1e5&f=json")
    'GET https://api.x/obs?series_id=DFF&api_key=***REDACTED***&f=json'
    """
    if not text:
        return text
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", text)


def redaction_filter(record) -> bool:
    """Loguru filter that scrubs secrets from the log message in place.

    Always returns ``True`` (it never drops a record — it only rewrites the
    message). Safe to attach to multiple sinks: redaction is idempotent.
    """
    msg = record.get("message")
    if msg:
        redacted = redact_secrets(msg)
        if redacted != msg:
            record["message"] = redacted
    return True
