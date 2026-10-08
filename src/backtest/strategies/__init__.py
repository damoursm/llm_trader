"""Strategy adapters: each turns one strategy's rules (with tunable parameters) into the account engine's trade
pieces (`src/backtest/account.py`). A piece's entry and exit are the strategy's own; whether the account can take it,
and at what size, is the engine's decision."""
