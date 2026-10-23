# The second account

Built on 2026-10-23; trades from the 2026-10-26 open. The user asked, on 2026-10-22: "I want to add a second account
of $50,000 that will have less noisy, more significant trades. Add the market noise floor for the models in that
second account." On 2026-10-23 they settled its form: "There's no second IBKR account creation. The second account
will be calculated from the same account."

So it is a CALCULATED account: its own ledger and its own $50,000, trading the main account's decisions at the live
prices, with no IBKR session and no order of its own.

Code: `src/accounts/` (`paths`, `profiles`, `launch`, `run`, `summary`). Settings: `account_name`,
`account_state_root`, `enable_extra_accounts`, `extra_accounts`, `extra_account_timeout_seconds` and
`second_account_*`. Tests: `tests/test_extra_accounts.py`.

## What it trades

By default it trades PREREG189's T3: the three models that pass the market-noise floor together at their live
rules. Each has t >= 2 against the market's noise in that configuration.

| Book | Its rules | Its slice of the second account's $50,000 |
|---|---|---|
| The dip book | The main account's: the daily top pick among the $2B+ signals, RP33's cut, the earnings demotion, the ½-ATR limit, the 3-session exit | 1/2, at most 2 open |
| The fall arm | The main account's: the turn, the 0.7 volatility exit, the 10-session limit | 1/8 |
| The crash book | The main account's: QQQ's decision traded in TQQQ, the after-hours exit | 1/3 |

The research reading (40 draws a set, 2018–2026): 59.8% a year, a 21.3% worst drawdown, no margin event, and a
worst 12 months of +6.7%. The live account reads 74.8%, 25.4% and +9.0%.

`second_account_books` sets the books; the user decides the configuration. Every rule stays the main account's
setting, because both accounts read the same decisions.

## How it runs

- **Its own process each tick.** The main scheduler launches `python -m src.accounts.run --account second` right
  after the main account's live path (`pipeline._launch_extra_accounts`). It launches it again after the
  end-of-tick broker sync. It never waits for it.
  - One runs at a time, held by `<root>/busy.lock`.
  - Each run ends after `extra_account_timeout_seconds` (900).
  - Output goes to `logs/account_second.log` and `logs/account_second_<date>.log`.
- **Its environment is its profile** (`src.accounts.profiles.env_for`). The variables set its name, its state root
  (`cache/accounts/second`), its ledger database (`data/llm_trader_second.db`), its books, its fresh start, its
  $50,000 and its broker mode. pydantic reads them over `.env`.
  - The runner refuses an environment that does not name the account, the main account's database, and any broker
    mode that opens an IBKR session (`run.guard`: only `dry_run` or `off`).
  - A profile without a short arm names the arm "none": an EMPTY `sel_short_traded_arms` would mean every arm.
- **The same steps as the main account's live path, in the same order.** First the marks, the fresh start and the
  books' exits, then the account's check (IBKR's liquidation rule, de-leveraging), the dip fills and the crash buy,
  the dip orders, the fall shorts' entries and the broker sync. They run on its own ledger and state.
- **It reads the main account's decisions and never makes them.** These are the scorer's picks and live-arm
  markers, the dip book's daily signals, the crash book's daily decision, and the capitulation book's breadth and
  signals.
  - It launches no scorer and fetches no borrow file. It reads the borrow archive the main account writes each tick.
  - A decision that is missing raises `paths.NotReady`. That step fails soft, and the next tick reads the decision
    once the main account has made it.
- **Its own state** lives under `account_state_root` (`paths.state`): the picks it consumed, its entry journal and
  trade log, its dip / crash / capitulation orders and journals, its margin check's state and liquidations, and its
  fresh-start marker.
  - The main account's root is "", so every main-account path stays where it was.
- **Calculated from the main account's IBKR answers.**
  - Its orders are a dry run (`DryRunBroker`): logged, never sent. Each leg is costed at its own quote at the
    order's submit, plus IBKR's fees (`spread.resolve_leg_cost`'s rung 1b, AUDIT151's construction), since it has no
    fill of its own.
  - IBKR's what-if for a fall short is the answer IBKR gave the MAIN account for that name in the same ET session
    (`sim_account.main_account_answer`, read from the main account's entry journals). A refusal skips the pick and
    the quoted rates size it. When the main account got no answer, the default house rates apply, as they do for the
    main account when IBKR does not answer.
  - A dry run holds no position, so the broker's buy-in check is skipped for it (`reconcile._holds_positions`).
- **In the email.** The main email's scorer banner carries one note line from the account's `summary.json`
  (`summary.health`). That line gives the equity since its start, the open positions and today's closes. It is never
  an alarm, and the main process never opens the second account's database.

## Deploying (with the user's go-ahead)

Set `SECOND_ACCOUNT_FRESH_START` to its first regular-hours tick (an ISO instant). An empty fresh start means it is
never launched. Then set `ENABLE_EXTRA_ACCOUNTS=true` and restart the scheduler. The first tick at or after the
fresh start closes nothing, since its ledger is empty. It does settle the picks decided before the instant, which
never enter. The account then counts only its own trades, from $50,000.

## Known gaps

- **Costs.** Its legs are costed at their own quote, not at fills: the main account's fills on the same trades are
  not read yet.
- **Margin.** A fall pick the main account did not fund (for example, the main account's room was full) has no IBKR
  answer: the default house rates (2.00 / 2.86) size it, and an IBKR refusal of that name is not seen. PREREG190's
  refusal model would be the stand-in.
- **Trade log.** The daily trade log (`tradelog/`) is the main scorer's prepare, built from the main ledger. The
  second account's entry journal is complete; its end-to-end trade log is not built yet.
- **Live score.** The live calibration and the forward check (EOD) read the main account only.
