"""Signal-pipeline internals, split by responsibility -- Phase 8 of
`projectedPlann.md` (see docs/architecture/000-audit.md).

``application/signal_pipeline.py`` (accumulated too many responsibilities
in one 1900-line file) is now a thin backward-compatible facade re-
exporting everything from these submodules -- every existing import of
``trading_scanner.application.signal_pipeline.X`` keeps working unchanged.
New code should import directly from the relevant submodule here instead:

- ``market_data``: provider selection, candle<->DataFrame conversion, Kite
  session/token-expiry helpers.
- ``evaluation``: per-symbol AlphaEngine evaluation (download-based and
  from-already-stored-candles).
- ``entry_decision``: persisting/finalizing one cash-entry candidate's
  outcome (Phase 3's entry_decisions ledger, GTT placement, the "missed
  signal" notification).
- ``capital_allocation``: one scan cycle's ranked capital allocation
  across the paper/futures/cash books (``application/ranking.py``'s
  consumer).
- ``lifecycle``: position/derivatives-shadow close bookkeeping and exit
  notifications.
- ``orchestrator``: ``run_signal_pipeline`` (the cron/CLI entry point) and
  ``_process_symbol`` (per-symbol trade/notification bookkeeping, shared
  by the cron and live-ticker paths) -- ties every other submodule
  together into one scan cycle.

No behavior changed by this split -- every function's body moved as-is;
only which file it lives in changed.
"""
