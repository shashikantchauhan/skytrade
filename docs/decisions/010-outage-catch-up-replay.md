# 010: Replay every candle missed during an outage, not just the newest one

## Status
Accepted, 2026-09-02.

## Context
`_evaluate_from_stored_candles` (the incremental "fast predict" evaluation
every live cycle runs) only ever evaluated the single newest stored candle,
advancing the persisted ANN neighbor queue (`QueueState`) by exactly one
bar's contribution each call. If the pipeline was offline across more than
one candle close (Kite session expired -- see ADR 009 -- or the process
itself down/restarting), any candle that closed during that gap was never
evaluated at all, before *or* after coming back online: reconnecting just
resumes from "whatever the newest candle is right now," with no mechanism
anywhere -- including the dashboard's manual "Run pipeline now" button --
that went back and asked what an intervening candle's own signal actually
was. Confirmed in production: on the three token-expiry days in ADR 009,
whatever candle closed during each outage window was silently skipped, and
logging back in later never surfaced it.

This was worse than a missed notification -- the skipped candle's own
contribution to the persisted neighbor queue was lost too, meaning the
model's own memory quietly diverged from what it would have been had the
pipeline never gone offline.

## Decision
`_evaluate_from_stored_candles` now finds every candle that closed since
`engine_state.last_bar_timestamp` (`_find_new_bar_indices`) and replays
each one individually, in chronological order, through `evaluate_latest_
bar` -- carrying `signal_previous`/`queue_state`/`exit_state` forward
between calls exactly as consecutive real-time calls would have, and
persisting `engine_state` after *each* replayed bar (not just the last)
so a crash mid-catch-up doesn't force redoing already-caught-up bars.
Returns a list of `(result, candle)` pairs (almost always length 1) instead
of a single result-or-None.

A gap wider than `_MAX_CATCH_UP_BARS` (50 hourly bars, ~2 trading weeks)
falls back to the old behavior -- jump straight to the newest bar -- rather
than replaying a large backlog inline during a live cycle; a gap that wide
needs a human's attention regardless.

Callers (`orchestrator.py`, `live_pipeline.py`) treat only the *last* entry
in the returned list as current -- it alone competes for ranking and can
lead to a real order, exactly as before. Every earlier entry is stale
catch-up: notified via the new `_notify_stale_catch_up_signals` (one
Telegram message per non-NEUTRAL caught-up bar, explicitly "not acted on")
and nothing else -- no trade recorded, no order placed, no paper/futures
position opened. This was a deliberate, discussed choice: a real order
against a candle that closed an hour or more ago isn't the same trade that
actually fired (price has moved on), so auto-ordering it risks a worse
outcome than the miss itself. The user chose notify-only over auto-order
for exactly this reason.

## Consequences
- No candle is silently skipped anymore, online or on catch-up -- the
  model's own neighbor-queue memory stays correct across an outage instead
  of quietly diverging from it.
- A real BUY/SELL found on a caught-up candle is now visible (Telegram)
  instead of vanishing with no trace, even though it's still not acted on
  automatically.
- `_evaluate_symbol`/`_evaluate_from_stored_candles` both changed their
  return type from `tuple | None` to `list[tuple]` -- every call site
  (`orchestrator.py`'s index/ranking/process paths, `live_pipeline.py`'s
  equivalents) updated to take the list's last entry for existing
  processing and the rest for stale notification. `_process_symbol` itself
  is unchanged -- it still only ever sees one `precomputed_evaluation`
  tuple, exactly as before this change.
- New tests: `tests/test_evaluation_catch_up.py` (no-gap case unchanged,
  no-new-candle case unchanged, a 3-bar gap replays all three in order
  with per-bar state persistence, a gap past the cap falls back safely,
  `_find_new_bar_indices`'s normal/multi-bar/None/unknown-timestamp cases,
  the stale-notification function's BUY/SELL-only/no-notifier/nothing-
  stale cases).

See `application/pipeline/evaluation.py`'s `_evaluate_from_stored_candles`
and `_notify_stale_catch_up_signals` for the full reasoning.
