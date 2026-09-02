# 008: Persist every symbol's gate state each cycle, for a dashboard tab

## Status
Accepted, 2026-09-02.

## Context
When the automated system fails to place a real order it should have
(see ADR 007/009/010 for three separate real ways that happened on
2026-09-02), there was no way to see *why* -- or which other symbols were
close to qualifying -- without reading server logs by hand. The gate
values that would answer this (regime/volatility/ADX state, track record,
quality filter, conviction filter) are computed fresh every cycle inside
`evaluate_latest_bar` and the real cash-entry gate checks, then discarded
once that cycle's trading decision is made.

Two constraints shaped the design: recomputing these values on demand for
~220 symbols is too slow for a dashboard request (each one is a multi-
second AlphaEngine pass), and the three real gates (track record, quality
filter, conviction filter) are today only ever evaluated for a symbol that
*already* has an active BUY/SELL signal and has already won ranking --
useless for "how close is a NEUTRAL symbol," which is exactly what matters
for spotting a near-miss before it happens.

## Decision
New `GateStatusSnapshot` (domain/models.py) and `TursoGateStatusRepository`
(one always-overwritten row per symbol, not an append-only log like
`entry_decisions`). `_process_symbol` (orchestrator.py) calls a new
`_record_gate_status` right after every cycle's evaluation -- for *every*
symbol evaluated that cycle, regardless of `result.signal` (BUY, SELL, or
NEUTRAL) -- reusing the exact same `entry_gates.evaluate_track_record_gate`
/`evaluate_cash_quality_gates` calls a real cash order would use, fed by
values (`adx`/`regime_normalized`/`volatility_margin`, already computed by
`evaluate_latest_bar`; OHLC, already on the newest candle) that cost
nothing extra to obtain -- no new Kite calls, no duplicated gate logic, no
separate recomputation path that could drift from what a real decision
actually sees.

New `/api/gate-status` (viewer-level, read-only) and a "Gates" dashboard
tab show every symbol's latest snapshot: signal, the three pass/fail
gates, and the raw ADX/regime/volatility numbers behind the quality gate,
filterable by signal.

Best-effort and silent-on-failure (`_record_gate_status` never raises into
`_process_symbol`) -- a dashboard convenience must never be able to break
real trade processing.

## Consequences
- The Gates tab always matches what a real decision would have seen, since
  it reuses the identical gate calls rather than a parallel implementation.
- One extra local DB write per symbol per cycle -- no Kite API cost, no
  measurable effect on cycle timing.
- `_process_symbol`/`run_signal_pipeline` gained one new optional
  `gate_status_repository` parameter each, defaulting to `None`
  (unwired callers/tests are unaffected).
- New tests: `tests/test_gate_status.py` (repository round-trip, upsert-
  not-append, per-interval filtering, snapshot computed for a NEUTRAL
  symbol, reflects a failed track-record/quality gate, best-effort
  swallows a repository failure).

See `domain/models.py`'s `GateStatusSnapshot`, `infrastructure/db/
gate_status.py`, and `application/pipeline/orchestrator.py`'s
`_record_gate_status` for the full reasoning.

## Addendum, 2026-09-02 (same day): the snapshot alone was misleading

`gate_status` (above) is a single always-overwritten row per symbol -- it
can only ever answer "what does the system see on the very latest candle."
`result.signal` is TRUE only on the one bar where a signal actually
changes; every bar after that reads back NEUTRAL again even though a real
BUY/SELL already fired earlier. Asked "did any signal fire today," the
snapshot-only view answered "1" when the real number (cross-checked
against `entry_decisions` and a manual review) was 15 -- a live,
confirmed-wrong answer given to the user the same day this shipped.

Added `gate_status_events`: a second, append-only table, one permanent
row per actual BUY/SELL (never for NEUTRAL, so it stays small), for both
sides -- unlike `entry_decisions`, which is BUY-only (SELL never reaches
ranking/paper-account code at all in this cash-only, long-only system).
`_record_gate_status` writes to it whenever `result.signal in ("BUY",
"SELL")`, using `ON CONFLICT (symbol, interval, evaluated_at) DO NOTHING`
so re-processing an already-recorded bar (e.g. catch-up replay) can't
duplicate it. New `/api/gate-status/events` (IST-midnight-scoped "today")
and a "Today's signals" section on the dashboard, above the existing
current-state table.

New tests: `tests/test_gate_status.py` (event round-trip, since-cutoff
filtering, replay-safe no-duplicate, `_record_gate_status` logs an event
for BUY/SELL but not NEUTRAL).
