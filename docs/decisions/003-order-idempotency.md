# 003: Deterministic order-intent id for retry/restart correlation

## Status
Accepted, 2026-09-01.

## Context
`execute_cash_entry`'s retry loop groups its own attempts under one
`basket_id` -- but `basket_id` is built from `datetime.now(UTC)` at call
time, so two independent calls for what a human would recognize as the
same signal (e.g. one attempt before a process restart, a second attempt
after) get unrelated `basket_id`s. There is no way to look at
`live_order_legs` and correlate them as "the same logical trade."

## Decision
`domain/order_intent.py`'s `compute_intent_id(symbol, side,
signal_timestamp, purpose)` is a deterministic hash -- the same signal
always produces the same id, regardless of how many times or across how
many process restarts it's attempted. `execute_cash_entry` computes one
intent per call (from the candidate's own `signal_timestamp`, not wall-
clock time) and writes it onto every attempt's leg (`live_order_legs.
intent_id`, nullable, additive column). Before the first placement
attempt, `get_legs_by_intent` checks whether this exact intent already has
a real/unconfirmed leg recorded -- a second, intent-keyed line of defense
alongside the existing per-symbol `get_unclosed_cash_legs` check.

## Consequences (and an explicit limit)
- Every attempt at one logical trade, across restarts, is now correlated
  under one stable, queryable key.
- This does **not** close the harder gap: a real order accepted by Kite
  but lost before any local DB write ever happened (crash between
  `place_order` returning and `_record` running). No local key can detect
  that -- only broker ground truth (`holding_quantity`, see ADR 002) can,
  and that check isn't wired into the *entry* path today, only exits.
  Recorded here rather than implied as solved; a future pass could extend
  entry-time reconciliation the same way exits were fixed, if this gap
  ever proves costly in practice.

See `domain/order_intent.py`'s own module docstring for the full reasoning.

## Addendum (2026-09-01): the gap above is now closed

See [006-broker-crash-window.md](006-broker-crash-window.md) --
`execute_cash_entry` now runs a broker-ground-truth preflight (Kite order-
tag lookup + `holding_quantity`) before the first placement attempt for a
fresh intent, closing exactly the gap this decision recorded as open.
