# 006: Broker ground truth before a fresh intent's first order

## Status
Accepted, 2026-09-01.

## Context
ADR 003 (`domain/order_intent.py`) gave a logical trade a deterministic
`intent_id`, and `execute_cash_entry` checks `get_legs_by_intent` before
placing an order -- but that check only ever sees what *this app itself*
already wrote to `live_order_legs`. If Kite accepts a real order and the
process dies (crash, OOM kill, deploy restart) between `place_order`
returning and `_record` running, a fresh attempt at the exact same signal
(new process, same deterministic `intent_id`) sees zero local legs and
would place a second real order. ADR 003 named this gap explicitly and
left it open pending a future pass.

Found via the "final reliability hardening pass" review (2026-09-01), the
same review that found ADR 002's leftover call site.

## Decision
Before the *first* placement attempt for a fresh intent (i.e. when the
local `get_legs_by_intent` check found nothing), `execute_cash_entry` now
runs `_broker_ground_truth_preflight`, checking two independent broker
signals:

1. `KiteOrderExecutor.find_todays_order_by_tag` -- every real entry order
   is now placed with `tag=intent_id[:20]` (Kite caps a tag at 20
   alphanumeric characters); a matching order already in today's book
   means a previous attempt reached the broker.
2. `KiteOrderExecutor.holding_quantity` -- the same ground-truth check
   ADR 002 already trusts for exits, catching a filled position even if
   the tag lookup somehow misses it, and independently catching a real
   position with *zero* local record at all (no crash-recovery signal
   needs to recur to surface it).

Either finding reconciles what was found into the local ledger and sends a
`RECONCILIATION REQUIRED` notification instead of ever placing a new
order -- fail closed. A tagged order confirmed `REJECTED`/`CANCELLED` is
recorded for the audit trail but does not block a fresh attempt (Kite's
own response already proves no real position resulted). Both broker calls
degrade to "found nothing" on failure (logged, never raised) -- the same
fallback discipline `gtt_bracket.reconcile_before_exit` already
established; the risk being closed is what happens when the check
*succeeds*, not what happens when it's unavailable.

A second, allowlist-wide sweep (`broker_reconciliation.
find_hidden_positions`, wired into `live_pipeline.py`'s per-cycle loop,
debounced) covers the broader case the per-signal preflight can't reach on
its own: a real position with no local record where the triggering signal
will never recur.

## Consequences
- A process restart after a broker-accepted-but-unrecorded order no longer
  places a duplicate real BUY -- it reconciles and alerts instead.
- Two extra Kite API calls (`orders()`, `positions()`/`holdings()`) per
  fresh entry attempt -- negligible at this system's hourly-scan cadence.
- Every real entry order is now tagged, which also makes Kite's own order
  book directly searchable by intent for manual investigation.
- New tests: `tests/test_live_cash_execution.py` (tag stamped on
  placement, tagged-order block, confirmed-rejected pass-through, hidden-
  holding block, infra-failure fail-open) and `tests/
  test_broker_reconciliation.py` (the allowlist sweep).

See `application/live_cash_execution.py`'s `_broker_ground_truth_
preflight` and `application/broker_reconciliation.py`'s
`find_hidden_positions` for the full reasoning.
