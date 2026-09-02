# 007: Batch the hidden-position sweep into one Kite call, not one per symbol

## Status
Accepted, 2026-09-02.

## Context
ADR 006's allowlist-wide sweep (`broker_reconciliation.find_hidden_
positions`) called `KiteOrderExecutor.holding_quantity(tradingsymbol)` once
per symbol in the cash allowlist, every pipeline cycle. `holding_quantity`
itself makes two full Kite API calls (`positions()`, `holdings()`) --
both already account-wide, returning every symbol's data regardless of
which `tradingsymbol` was asked for. So for N allowlist symbols the sweep
made 2N Kite API calls per cycle, all re-fetching and re-filtering the
same two lists.

ADR 006 assumed "two extra Kite API calls... negligible at this system's
cadence" -- true for the per-signal entry preflight (one signal, one
check), but the sweep runs every cycle against the *whole* allowlist, not
per-signal, so its cost scales with N instead of being constant. In
production on 2026-09-02, with a ~20-symbol allowlist, this burst of ~40
near-simultaneous calls tripped Kite's rate limit
(`NetworkException: Too many requests`) partway through the sweep --
19 symbols logged "Hidden-position check failed... skipping this cycle"
in `live.log` around 15:15-15:16 IST. The sweep degraded safely (each
failure was caught, logged, and skipped -- no crash, no bad order), but a
burst like this competing for the same per-second rate-limit budget as a
real order-placement or GTT-status call in the same cycle is a live-
trading risk, not just noise.

## Decision
New `KiteOrderExecutor.holding_quantities() -> dict[str, int]` computes
every CNC symbol's real quantity from one `positions()` call and one
`holdings()` call, full stop -- not one pair per symbol.
`find_hidden_positions` now calls this once per sweep and looks up each
allowlist symbol's quantity from the returned dict, instead of looping
calls to `holding_quantity`. `holding_quantity` itself is unchanged --
its other two callers (`gtt_bracket.reconcile_before_exit`,
`live_cash_execution`'s per-signal preflight) each check exactly one
symbol at one moment; there's nothing to batch there.

Failure handling changed accordingly: previously a single symbol's API
error was caught and skipped, leaving the rest of the sweep to proceed
per-symbol. Now there is only one call to fail -- if it does, the whole
sweep is skipped for that cycle (logged) and retried next cycle, rather
than trying to partially apply a sweep against a stale/incomplete
quantity map.

## Consequences
- The sweep now costs 2 Kite API calls per cycle regardless of allowlist
  size, not 2N -- eliminates the rate-limit risk this fixes and stops
  competing with real order/GTT calls for API quota during the same
  cycle.
- A transient failure now skips the entire cycle's sweep rather than
  just the one symbol that failed -- an acceptable trade since the sweep
  is periodic and debounced already (ADR 006); the next cycle retries
  the whole thing cheaply.
- New test: `tests/test_broker_reconciliation.py::
  test_find_hidden_positions_skips_the_whole_sweep_when_the_bulk_fetch_
  fails` replaces the old per-symbol-failure test to match the new
  all-or-nothing-per-cycle semantics.

See `infrastructure/kite.py`'s `holding_quantities` and
`application/broker_reconciliation.py`'s `find_hidden_positions` for the
full reasoning.
