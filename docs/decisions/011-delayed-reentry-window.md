# 011: Retry a gate-cleared-but-skipped BUY candidate for up to 2 trading days

## Status
Accepted, 2026-09-02.

## Context
On 2026-09-02 two real candidates -- SOLARINDS.NS and UPL.NS -- cleared
every real gate for a cash entry (track record, `entry_quality_filter`,
`conviction_filter`, ranking) and still didn't get a real order, purely on
capacity/cutoff/execution grounds (`entry_decisions.final_decision =
"skipped"`, confirmed via `_finalize_cash_entry`'s generic "no free slot,
past cutoff, or execution failed" note). Until now, that candidate was
simply gone -- nothing looked at it again, even though the strategy's own
BUY thesis for that symbol was very likely still valid an hour or a day
later.

The user's request: if the strategy's own dynamic exit hasn't fired for
that symbol since (the thesis is still alive), and price comes back within
~0.5% of the original signal price within 2 trading days, retry the entry
instead of losing it outright -- re-checking quality/conviction against
the day it actually retries, not trusting a stale approval.

Investigation found this needed less new machinery than it looked:
`entry_decisions` already durably records exactly the candidate set
needed -- a `final_decision="skipped"` row only exists on the one code
path (`_rank_and_open_cash_positions`) reached after all three real gates
already passed, so `track_record_passed`/`quality_passed`/
`conviction_passed` are `True` by construction there. And
`_rank_and_open_cash_positions` is already sequential-by-construction with
no lock (see its own docstring), so a delayed candidate merged into the
same ranked list before that call inherits its existing capital-safety
guarantees for free.

## Decision
New `LiveCashToggleState.delayed_retry_enabled: bool = False`
(`infrastructure/db/live_cash_toggle.py`) -- dashboard-toggleable, same
pattern as the existing `max_positions` field. Off by default: shipping
this code is not the same event as it ever placing a real order.

New `TursoEntryDecisionRepository.get_pending_cash_retries(since)` --
every `final_decision="skipped"`, `ranking_passed=True`, BUY row at or
after `since`, deduped to the most recent per symbol.

New `_collect_delayed_retry_candidates` (`application/pipeline/
capital_allocation.py`), called from `_collect_and_open_ranked_positions`
right before `_rank_and_open_cash_positions`, extending the same
`cash_candidates` list ordinary fresh signals build. Per pending row, in
order: skip if no fresh evaluation this cycle; skip if a fresh BUY already
covers this symbol this cycle (the main loop handles it); skip if
`FastPredictResult.end_long` or `.signal == "SELL"` -- deliberately *not*
just "signal is no longer BUY", since `signal` is transition-only and
reads `NEUTRAL` on every bar after the original fire even while the
thesis is still live (see ADR 008's addendum for the exact bug this would
otherwise repeat); skip if a real position is already open
(`get_unclosed_cash_legs`); skip if today's price
(`_market_price(newest_candle)`, the same `(H+L+O+O)/4` convention used
everywhere else in this pipeline) has moved more than
`_RETRY_PRICE_TOLERANCE` (0.5%) from the original `signal_price`; skip if
`evaluate_track_record_gate`/`evaluate_cash_quality_gates` -- called
against *today's* candle, not the stale one -- don't both pass. A
surviving row becomes a `RankedCandidate` priced and timestamped off
today's candle, carrying the original `signal_timestamp` in a new
traceability-only field, `RankedCandidate.retry_of_signal_timestamp`
(`application/ranking.py`, not read by `score_candidate`).

The 2-trading-day window (`_RETRY_WINDOW_TRADING_DAYS`) is computed by
`_trading_days_ago` -- Mon-Fri only, no exchange-holiday calendar (none
exists anywhere in this codebase today; `is_market_hours` is explicitly
Mon-Fri-only too). A holiday inside the window makes this slightly more
permissive than exactly 2 real sessions, at most a couple of times a year
-- a known, accepted simplification, not worth a new calendar dependency.

A retry that actually fills gets one extra Telegram notification
(`_notify_filled_delayed_retries`, best-effort, never raises) distinct
from an ordinary fresh entry -- original missed price/time plus today's
fill. No notification for a retry that doesn't qualify or doesn't fill --
would be noisy every cycle otherwise. Persistence needed no new code: a
retry candidate flows through the real `_rank_and_open_cash_positions`,
which already unconditionally calls `_persist_entry_decision` for every
candidate it processes.

## Consequences
- A gate-cleared-but-skipped candidate gets a second chance instead of
  being lost outright, but only while the underlying thesis is still
  live and only within a tight price band -- never a stale re-entry days
  after conditions changed.
- Off by default in production; the user turns it on from the dashboard
  (`Retry a missed signal for 2 trading days` checkbox on the existing
  Go Live panel) when ready. Merging this change alone does not affect
  any live order.
- `_collect_and_open_ranked_positions` gained no new required parameters
  -- the retry step reads `cash_state`/`live_order_repository`/
  `entry_decision_repository`, all already threaded through for other
  reasons, and no-ops cleanly when any is `None` or the toggle is off.
- New tests: `tests/test_delayed_retry.py` (`_trading_days_ago`'s weekend
  skip; every exclusion case -- disabled toggle, missing repos, no fresh
  evaluation, fresh BUY already covers it, `end_long`/SELL invalidation, a
  merely-NEUTRAL signal does *not* invalidate, already-open, price outside
  tolerance, price exactly at the tolerance boundary, failing a re-checked
  gate; `_notify_filled_delayed_retries`'s opened/not-opened/best-effort
  cases), plus round-trip tests in `tests/test_live_cash_toggle_repository.py`,
  `tests/test_db_entry_decisions.py`, and `tests/test_ranking.py`, plus two
  full-stack tests in `tests/test_signal_pipeline.py`
  (`test_a_delayed_retry_candidate_flows_end_to_end_into_a_real_cash_entry`,
  `test_delayed_retry_produces_nothing_when_the_toggle_is_off`) reusing
  that file's existing fake `KiteOrderExecutor`/`TursoLiveOrderRepository`
  stack rather than duplicating it.

See `application/pipeline/capital_allocation.py`'s
`_collect_delayed_retry_candidates` for the full reasoning.
