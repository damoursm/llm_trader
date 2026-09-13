"""The ib_async → loguru bridge (src/broker/ibkr.py:_install_ib_log_bridge).

ib_async reports API errors (error 10089 "requires additional subscription",
pacing violations, farm outages) through stdlib logging — a channel this
project sinks nowhere, so those errors were invisible (found 2026-08-31: a
whole overnight of refused quote requests left zero trace in the loguru file
while get_quote silently returned None). These tests pin that the bridge
exists, forwards WARNING+ into loguru, and installs idempotently.
"""

from __future__ import annotations

import logging

from loguru import logger


def _capture(fn):
    """Run ``fn`` with a temporary loguru sink; return the captured lines."""
    lines: list = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return lines


def test_ib_async_errors_reach_loguru():
    import src.broker.ibkr  # noqa: F401  (installs the bridge at import)

    lines = _capture(lambda: logging.getLogger("ib_async.wrapper").error(
        "Error 10089, reqId 4: probe message"))
    assert any("10089" in ln and "[ib_async]" in ln for ln in lines), (
        "an ib_async ERROR record must re-emit through loguru — otherwise feed "
        "failures (subscription refusals, pacing) are invisible in the log file")


def test_bridge_forwards_warnings_but_not_info():
    import src.broker.ibkr  # noqa: F401

    lines = _capture(lambda: (
        logging.getLogger("ib_async.client").warning("probe warning"),
        logging.getLogger("ib_async.client").info("probe info"),
    ))
    assert any("probe warning" in ln for ln in lines)
    assert not any("probe info" in ln for ln in lines), (
        "INFO chatter (order status flow) must not flood the loguru file")


def test_bridge_installs_idempotently():
    from src.broker.ibkr import _install_ib_log_bridge

    _install_ib_log_bridge()
    _install_ib_log_bridge()
    handlers = [h for h in logging.getLogger("ib_async").handlers
                if type(h).__name__ == "_LoguruBridge"]
    assert len(handlers) == 1, "re-installing must not stack duplicate handlers"

    lines = _capture(lambda: logging.getLogger("ib_async").error("dupe probe"))
    assert sum("dupe probe" in ln for ln in lines) == 1
