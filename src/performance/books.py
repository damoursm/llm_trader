"""The ledger's LIVE books — the entry mechanisms that trade. Every other open trade belongs to a shadow
book: the legacy exit stack and the signal reversal never touch a live book's trade, the one-shot legacy
flatten skips it, and the broker sends a live book's entries while the legacy books are shadow
(`reconcile.sync`). One definition, read by the tracker and the reconciler."""
from __future__ import annotations

SEL_SHORT = "sel_short"      # the selection short (src/signals/sel_short.py, from 2026-09-28)
DIP_LONG = "dip_long"        # the mega-cap dip long book (src/signals/dip_long.py, from 2026-10-08)
ETF_DM = "etf_dm"            # the ETF dual-momentum book (src/signals/etf_dm.py; trades only with enable_etf_dm)
CRASH_LONG = "crash_long"    # the crash-rebound book (src/signals/crash_long.py, from 2026-10-12)
CAP_LONG = "cap_long"        # the capitulation book (src/signals/cap_long.py, from 2026-10-21)
INSIDER_LONG = "insider_long"    # the insider book (src/signals/insider_long.py, from the 2026-10-26 session)
FEAR_LONG = "fear_long"      # the fear book (src/signals/fear_long.py, from the 2026-10-28 session)
LIVE_MECHANISMS = (SEL_SHORT, DIP_LONG, ETF_DM, CRASH_LONG, CAP_LONG, INSIDER_LONG, FEAR_LONG)
# books that are ONE BOOK for one book per name (PREREG223's M1): each may buy a name the other holds
SHARED_NAMES = {DIP_LONG: (FEAR_LONG,), FEAR_LONG: (DIP_LONG,)}


def same_book(mechanism: str) -> tuple:
    """The mechanisms that count as ``mechanism``'s own book in the one-book-per-name test: itself and the books it
    shares names with (`SHARED_NAMES`)."""
    return (mechanism,) + tuple(SHARED_NAMES.get(mechanism, ()))


def is_live(trade: dict) -> bool:
    """The trade belongs to a live book (it owns its exits; the broker sends its entries)."""
    return trade.get("entry_mechanism") in LIVE_MECHANISMS
