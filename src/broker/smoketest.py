"""IBKR connectivity + order-path smoke test — run BEFORE enabling broker_mode=ibkr_paper.

    python -m src.broker.smoketest            # connect, print account + positions
    python -m src.broker.smoketest --probe    # confirm the ORDER PATH works, any hour (safe)
    python -m src.broker.smoketest --order     # real round trip: BUY+close 1 share (RTH only)

``--probe`` places a deliberately-unfillable limit order (buy 1 share at $1), confirms IBKR
*accepts* it — proving Read-Only API is off and orders flow — then cancels it. It never fills
and leaves nothing behind, so it's safe to run when the market is closed.

``--order`` does a real market BUY + SELL round trip; it only makes sense during regular
trading hours and is refused otherwise (a market order outside RTH would sit pending).

Never touches the internal trade ledger — talks to the broker only.
"""
from __future__ import annotations

import argparse

from loguru import logger

from src.broker.base import OrderRequest
from src.broker.ibkr import IBKRBroker


def _market_open() -> bool:
    from src.utils import now_et
    n = now_et()
    return n.weekday() < 5 and (9 * 60 + 30) <= (n.hour * 60 + n.minute) <= (16 * 60)


def _probe(b: IBKRBroker, symbol: str) -> None:
    """Place an unfillable limit order, confirm acceptance, then cancel. Safe any time."""
    from ib_async import LimitOrder, Stock
    ib = b._ib
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)
    order = LimitOrder("BUY", 1, 1.00)          # $1 limit — never fills for a liquid ETF
    order.orderRef = "smoketest-probe"
    order.tif = "GTC"                           # good-till-cancelled so it rests outside RTH
    order.outsideRth = True                     # allow it to be live outside regular hours
    logger.info(f"Placing unfillable test order: BUY 1 {symbol} LMT $1.00 ...")
    trade = ib.placeOrder(contract, order)
    for _ in range(10):
        ib.sleep(1)
        if trade.orderStatus.status in ("Submitted", "PreSubmitted", "Filled",
                                        "Cancelled", "ApiCancelled", "Inactive"):
            break
    status = trade.orderStatus.status
    accepted = status in ("Submitted", "PreSubmitted", "Filled")
    if accepted:
        logger.info(f"  ✓ ACCEPTED by IBKR (status={status}) — order path works, Read-Only API is off")
    else:
        logger.error(f"  ✗ NOT accepted (status={status}). Is Read-Only API still enabled?")
    ib.cancelOrder(order)
    ib.sleep(1.5)
    logger.info(f"  cancelled test order (status now {trade.orderStatus.status})")


def main() -> None:
    ap = argparse.ArgumentParser(description="IBKR broker smoke test")
    ap.add_argument("--probe", action="store_true",
                    help="confirm the order path with an unfillable+cancelled limit order (safe any time)")
    ap.add_argument("--order", action="store_true",
                    help="real market BUY + close of 1 share (regular trading hours only)")
    ap.add_argument("--symbol", default="SPY", help="symbol for the test order")
    args = ap.parse_args()

    b = IBKRBroker()
    logger.info(f"Connecting to IB Gateway {b.host}:{b.port} (clientId={b.client_id}) ...")
    if not b.connect():
        logger.error("Could not connect. Is IB Gateway running with the API enabled on that port?")
        return

    logger.info(f"Account : {b.get_account()}")
    logger.info(f"Positions ({len(b.get_positions())}): {b.get_positions()}")

    if args.probe:
        _probe(b, args.symbol)

    if args.order:
        if not _market_open():
            logger.warning("Market is CLOSED — skipping the live round trip (a market order would "
                           "sit pending and could fill at the open). Use --probe now, or --order "
                           "during 09:30–16:00 ET.")
        else:
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
