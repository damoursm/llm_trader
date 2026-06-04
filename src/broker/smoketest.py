"""IBKR connectivity smoke test — run BEFORE enabling broker_mode=ibkr_paper.

    python -m src.broker.smoketest             # connect, print account + positions
    python -m src.broker.smoketest --order     # also place + close 1 share of --symbol

Requires IB Gateway (or TWS) running with the API enabled, on the configured
port (paper = 4002). Never touches the internal trade ledger — it talks to the
broker only, so it's safe to run any time to confirm the connection works.
"""
from __future__ import annotations

import argparse

from loguru import logger

from src.broker.base import OrderRequest
from src.broker.ibkr import IBKRBroker


def main() -> None:
    ap = argparse.ArgumentParser(description="IBKR broker connectivity smoke test")
    ap.add_argument("--order", action="store_true",
                    help="place + immediately close 1 share of --symbol (paper account)")
    ap.add_argument("--symbol", default="SPY", help="symbol for the optional test order")
    args = ap.parse_args()

    b = IBKRBroker()
    logger.info(f"Connecting to IB Gateway {b.host}:{b.port} (clientId={b.client_id}) ...")
    if not b.connect():
        logger.error("Could not connect. Is IB Gateway running with the API enabled on that port?")
        return

    logger.info(f"Account : {b.get_account()}")
    positions = b.get_positions()
    logger.info(f"Positions ({len(positions)}): {positions}")

    if args.order:
        logger.info(f"Placing test BUY 1 {args.symbol} ...")
        res = b.submit_order(OrderRequest(args.symbol, "BUY", 1, client_ref="smoketest", intent="ENTRY"))
        logger.info(f"  BUY: ok={res.ok} status={res.status} fill={res.avg_fill_price} id={res.order_id}")
        if res.ok and res.filled_qty:
            logger.info(f"Closing test position: SELL 1 {args.symbol} ...")
            res2 = b.submit_order(OrderRequest(args.symbol, "SELL", 1, client_ref="smoketest-exit", intent="EXIT"))
            logger.info(f"  SELL: ok={res2.ok} status={res2.status} fill={res2.avg_fill_price}")

    b.disconnect()
    logger.info("Smoke test done.")


if __name__ == "__main__":
    main()
