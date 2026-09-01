# 005: Bounded retry for a rejected/cancelled real entry, not "until cutoff"

## Status
Accepted, 2026-09-01.

## Context
A `REJECTED`/`CANCELLED` real BUY placement used to notify
("LIVE CASH ORDER FAILED") and give up immediately -- a genuinely good
signal (already cleared eligibility, quality filter, and conviction
filter) could lose its slot to a single transient placement failure.

The user's first preference was to keep retrying until the entry cutoff.
Rejected: `_rank_and_open_cash_positions` attempts one scan cycle's
candidates sequentially (by design -- see ADR 002-adjacent reasoning in
`capital_allocation.py`), so a retry loop running for potentially hours
would block every other candidate behind it in that same cycle, and could
still be running when the next hourly cycle starts. A true until-cutoff
version would need an independent background task, adding real complexity
(avoiding races with next hour's fresh signal on the same symbol).

## Decision
Bounded retry: up to 10 attempts, 20s apart (~3 minutes of backoff),
inside `execute_cash_entry`'s own retry loop. Only `REJECTED`/`CANCELLED`
outcomes retry -- `COMPLETE`/`OPEN`/`UNKNOWN` all mean a real order state
already exists at the broker and must never be retried (that's exactly
how the earlier PERSISTENT.NS double-buy incident happened). Before each
retry, the entry cutoff and `get_unclosed_cash_legs` are both re-checked,
since either could have changed during the backoff sleep.

## Consequences
- Enough retrying to ride out a real transient failure (a network blip, a
  momentary RMS/rate-limit hiccup -- both typically resolve in seconds to
  low minutes) while still leaving the rest of the cycle's candidates a
  real chance at their own slot.
- If still failing after ~3 minutes, that's very likely a persistent
  rejection reason (bad instrument, insufficient margin, ...) that more
  retrying wouldn't fix anyway -- the final FAILED notification reports
  the attempt count so an exhausted-retries failure reads differently
  from today's old single-shot failure.

See `application/live_cash_execution.py`'s own `_MAX_ENTRY_ATTEMPTS`
comment for the full reasoning.
