"""Research backtesting: one simulated IBKR account that runs several strategies at once (user directive
2026-10-08: "Rework our simulator ... so that it can consider multiple strategies ... reuse it to evaluate different
models or combination of models"). Nothing in production imports this package.

* `account` — the account engine: $10,000, longs and shorts, IBKR's fees, Reg T + house margin, available capital,
  slices per strategy, margin calls at bar closes or intrabar extremes, per-strategy attribution, both metrics.
* `strategies` — adapters that turn a strategy's rules into trade "pieces" for the engine (the vol short arm, the
  mega-cap dip long book).
* `study` — weekly walk-forward re-tuning of one or several strategies together, the objective being the account's
  own log growth.

Runbook: docs/BACKTEST.md.
"""
