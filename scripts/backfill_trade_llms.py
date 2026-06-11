"""One-time backfill: freeze the LLM engine ids onto pre-existing trades.

New trades are stamped with ``llm_synthesis_model`` / ``llm_sentiment_model`` at
entry (``record_new_trades``). Trades opened before that change are resolved
through their run's recorded provider (runs table) and the CURRENT
provider→model mappings, then written back — freezing today's truth onto
history so a future ANALYST_MODEL change can't retroactively relabel old
trades.

Idempotent: trades already carrying both stamps are left untouched. Run from
the repo root with the venv python:

    .venv\\Scripts\\python.exe -m scripts.backfill_trade_llms
"""

from loguru import logger

from src.performance.tracker import (
    _llm_models_for_trade,
    _llm_run_map,
    _load_trades,
    _save_trades,
)


def main() -> None:
    trades = _load_trades()
    if not trades:
        logger.info("No trades to backfill.")
        return
    run_map = _llm_run_map()
    stamped = 0
    for t in trades:
        if t.get("llm_synthesis_model") and t.get("llm_sentiment_model"):
            continue
        synth, sent = _llm_models_for_trade(t, run_map)
        t["llm_synthesis_model"] = t.get("llm_synthesis_model") or synth
        t["llm_sentiment_model"] = t.get("llm_sentiment_model") or sent
        stamped += 1
    _save_trades(trades)
    logger.info(f"Backfilled LLM engine stamps on {stamped}/{len(trades)} trade(s).")


if __name__ == "__main__":
    main()
