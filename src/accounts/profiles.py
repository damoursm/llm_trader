"""THE EXTRA ACCOUNTS' PROFILES (from 2026-10-23; `docs/SECOND_ACCOUNT.md`).

An extra account runs in its OWN PROCESS each tick (`src.accounts.run`), launched by the main scheduler after the
main account's live path (`src.accounts.launch`). Its profile is the settings it overrides, passed to that process as
environment variables (pydantic reads them over ``.env``): its name and state root (`src.accounts.paths`), its
ledger database, the books it trades, its fresh start and its $50,000, and its broker — none of its own: a CALCULATED
account (the user, 2026-10-23: "There's no second IBKR account creation. The second account will be calculated from
the same account."), its orders a dry run (never sent, its legs costed at their own quote, `spread.resolve_leg_cost`)
and IBKR's what-if answers the main account's (`sim_account.main_account_answer`). Every rule of every book stays the
main account's settings. A profile holds NO secret, and nothing here reads ``.env``.

THE SECOND ACCOUNT (PREREG186 / 189; the user, 2026-10-22: "I want to add a second account of $50,000 that will have
less noisy, more significant trades. Add the market noise floor for the models in that second account.", then
2026-10-23: "What about crash + fall + dip? These are the three that are over t=2."): by default the three models that pass the market-noise floor TOGETHER at their live rules
(PREREG189's T3 — the dip book, the fall arm, the crash book: each t >= 2 in it; 59.8% a year, the drawdown 21.3%, no
margin event in 40 draws), each at its live slice of its own $50,000. Which books: the user's call
(`second_account_books`).
"""
from __future__ import annotations

from typing import Dict, List

from config.settings import settings

# The books a profile can trade: the selection short's arms (through `sel_short_traded_arms`) and the long books
# (each by its switch); every book the profile does not name is off in the account's process.
SHORT_ARMS = ("vol", "etf", "fall")
LONG_BOOKS = ("dip", "crash", "cap")
BOOKS = SHORT_ARMS + LONG_BOOKS
NO_ARM = "none"                                   # a traded-arms list naming no real arm
CALCULATED_BROKER_MODE = "dry_run"                # an extra account's orders: logged, never sent
DRY_RUN_EQUITY = 1_000_000.0                      # the dry run's account value for the broker's exposure cap


def books_of(raw) -> List[str]:
    """A profile's books, checked (a comma string or a list)."""
    if isinstance(raw, str):
        raw = [b for b in raw.split(",")]
    books = [str(b).strip().lower() for b in (raw or []) if str(b).strip()]
    bad = [b for b in books if b not in BOOKS]
    if bad:
        raise ValueError(f"unknown books {bad} (known: {list(BOOKS)})")
    return books


def env_for(name: str) -> Dict[str, str]:
    """The environment overrides of account ``name``'s process (strings, as pydantic reads them)."""
    if name != "second":
        raise ValueError(f"no profile for the account {name!r}")
    books = books_of(settings.second_account_books)
    shorts = [a for a in SHORT_ARMS if a in books]
    return {
        "ACCOUNT_NAME": name,
        "ACCOUNT_STATE_ROOT": str(settings.second_account_state_root),
        "DB_PATH": str(settings.second_account_db_path),
        # the books: the short arms through the traded-arms list — never empty: an empty list means EVERY arm
        # (`sel_short.traded_arms`), so a profile without a short arm names none ("none") and turns the selection
        # short off — each long book by its switch; the rest off
        "SEL_SHORT_TRADED_ARMS": ",".join(shorts) if shorts else NO_ARM,
        "ENABLE_SEL_SHORT": "true" if shorts else "false",
        "ENABLE_DIP_LONG": "true" if "dip" in books else "false",
        "ENABLE_CRASH_LONG": "true" if "crash" in books else "false",
        "ENABLE_CAP_LONG": "true" if "cap" in books else "false",
        "ENABLE_ETF_DM": "false",
        # the main account's shadows and its journals stay the main account's
        "ENABLE_SEL_SHORT_NEWS_HOLD_SHADOW": "false",
        "ENABLE_LIVE_FEATURE_CAPTURE": "false",
        # its own account: its own start, its own $50,000
        "LIVE_ACCOUNT_FRESH_START": str(settings.second_account_fresh_start),
        "SEL_SHORT_ACCOUNT_INITIAL": str(settings.second_account_initial),
        # no broker of its own: a calculated account — its orders a dry run, never sent, never a gateway session
        # (`src.accounts.run.guard` refuses any other mode); IBKR's what-if answers are the main account's
        "BROKER_MODE": CALCULATED_BROKER_MODE,
        # the dry run's equity is only the broker's exposure cap (the ledger sizes from its own $50,000): IBKR's paper
        # balance, as loose as the main account's cap on IBKR's net liquidation value
        "BROKER_PAPER_EQUITY": str(DRY_RUN_EQUITY),
        "BROKER_GATEWAY_AUTO_RESTART": "false",
        "BROKER_GATEWAY_RELOGIN_RESTART": "false",
    }
