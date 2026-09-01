# 002: One shared "unclosed leg" lookup for every exit-eligibility check

## Status
Accepted, 2026-09-01.

## Context
A real cash-equity BUY leg is recorded `status="UNKNOWN"` when
`wait_for_fill`'s poll raises mid-check (a real, already-observed Kite API
flakiness -- see `live_cash_execution.py`'s `_place_and_wait` docstring
for the UNIONBANK.NS incident on the exit side). The order may or may not
have actually filled at the broker.

`infrastructure/db/live_orders.py` had two different "is there a real
position" repository methods: `get_open_cash_legs`/`get_all_open_cash_legs`
(COMPLETE-only) and `get_unclosed_cash_legs` (COMPLETE/OPEN/UNKNOWN). The
broader one was used correctly to *block a duplicate entry*. Every
*exit*-eligibility call site -- the strategy SELL branch, `execute_cash_
exit`, `manual_exit.py`, and the dashboard's `/api/live-cash-positions` --
used the narrower COMPLETE-only one instead. An `UNKNOWN` entry leg was
therefore invisible to all of them: no GTT bracket, not shown on the
dashboard, and the strategy's own exit branch and the manual "exit now"
button both silently found nothing to act on. `gtt_bracket.
reconcile_before_exit`'s broker-ground-truth check (`holding_quantity`) --
built specifically to resolve exactly this kind of uncertainty -- never
even ran, because the leg lookup upstream of it returned nothing first.

Found via a full-repository review (not a live incident) on 2026-09-01.

## Decision
Centralize the lookup in `application/broker_reconciliation.py`:
`get_unclosed_entry_leg`/`get_all_unclosed_positions`, both backed by the
same broad status set `get_unclosed_cash_legs` already used for entry-
blocking. Every exit-eligibility and capacity-counting call site now goes
through this module instead of picking its own repository method.
`gtt_bracket.reconcile_before_exit` itself is untouched -- its ground-
truth check was already correct, it just needed to actually be reached.

`domain/order_lifecycle.py`'s `PositionLifecycle` (added alongside this)
gives the same distinction a named, testable value
(`RECONCILIATION_REQUIRED`), and `broker_reconciliation.
get_reconciliation_required_symbols` surfaces it on the dashboard
(`GET /api/reconciliation-status`) so an `UNKNOWN` leg is visible on its
own, not only once something happens to try exiting it.

## Consequences
- An `UNKNOWN` entry leg is now exitable via every path a `COMPLETE` one
  is: strategy signal, manual button, and visible on the positions view.
- `execute_cash_entry`'s `max_positions` capacity count also switched to
  the broader set -- an `UNKNOWN` position now correctly counts as
  capital at risk (previously it could have let more real positions open
  than the cap intended).
- New end-to-end regression test (`tests/test_manual_exit.py`): an
  `UNKNOWN` entry + real `holding_quantity` now exits successfully.

See `docs/architecture/000-audit.md`'s gap #1 and
`application/broker_reconciliation.py`'s own module docstring.

## Addendum (2026-09-01): a leftover call site

A follow-up review (the "final reliability hardening pass") found one call
site this decision's migration missed: `application/pipeline/entry_
decision.py`'s `_finalize_cash_entry`, called immediately after `live_
cash_execution.execute_cash_entry` to decide whether to place the GTT
bracket and whether to notify "opened" or "missed." It still called `get_
open_cash_legs` (COMPLETE-only) directly, so an `UNKNOWN` fill got no GTT
bracket on a possibly-real position, and the "MISSED BUY SIGNAL"
notification fired even though a real order may have gone through -- an
active false signal, not just a blind spot. Switched to `broker_
reconciliation.get_unclosed_entry_leg`, matching every other call site.
`paper_benchmark.record_entry` (an analytics comparison, not safety-
critical) stays scoped to a leg with a confirmed `average_price`. See
`_finalize_cash_entry`'s own docstring.
