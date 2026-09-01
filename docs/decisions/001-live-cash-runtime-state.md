# 001: Runtime cash-trading state is its own type, never a config clone

## Status
Accepted, 2026-09-01.

## Context
`live_pipeline.py`'s live-ticker path needs the dashboard's "Go Live"
cash-trading toggle (enabled/symbols/notional/max_positions) refreshed
every scan cycle from the DB, while every other setting is static,
loaded once at process start (`AppConfig`). The original implementation
built a second `AppConfig`-shaped object each cycle
(`effective_config = dataclasses.replace(self._config, live_cash_trading_*=...)`)
merging the DB toggle's live values onto a clone of the static config, so
every call site downstream could keep taking a plain `AppConfig`.

On 2026-09-01, `_collect_and_open_ranked_positions` was passed
`self._config` (the static object, `live_cash_trading_enabled=False` by
default) instead of `effective_config` (the correctly-toggled clone) at
one call site. `execute_cash_entry`'s gate check
(`_is_gated_in`) silently returned `False` with zero log output. A real,
already-vetted BUY signal (HCLTECH.NS) got no order, with no trace in
logs or notifications -- caught only because the user happened to check
by hand.

## Decision
Runtime-adjustable cash settings live in their own type
(`infrastructure/db/live_cash_toggle.py`'s `LiveCashToggleState`), never
merged into a clone of `AppConfig`. Every function that needs them takes
`cash_state: LiveCashToggleState` as its own explicit parameter, built
once per cycle (from the DB toggle in `live_pipeline.py`, or once per run
from static config in the cron path, `run_signal_pipeline`, which has no
per-cycle refresh) and threaded through unchanged. `AppConfig` keeps only
the settings that are genuinely static (e.g.
`live_cash_entry_cutoff_ist`).

## Consequences
- Passing the wrong object is now a `TypeError` at the call site, not a
  silently wrong value -- the two types are no longer shape-compatible.
- `AppConfig`'s own docstring states this invariant explicitly (see
  `config/settings.py`) so a future change can't reintroduce the clone
  pattern without contradicting a comment right where the field is
  defined.
- The gate that failed silently (`_is_gated_in`) now logs its outcome
  unconditionally, so an unexpected `False` here is visible in logs even
  if the root cause were something else next time.

See `application/live_cash_execution.py`'s own module docstring for the
full incident narrative and `docs/architecture/000-audit.md` for where
this sits in the wider refactor.
