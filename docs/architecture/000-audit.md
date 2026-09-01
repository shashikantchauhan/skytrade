# SkyTrade architecture audit (Phase 0)

Starting point for the architecture/reliability refactor in
`projectedPlann.md`, executed on `refactor/architecture-reliability` per
`.claude` plan `adaptive-sleeping-quiche.md`. This is a snapshot of the
system as of commit `d212a57` (main) -- a map, not a wishlist; changes
happen in the later stages, not here.

## Module map

```text
live_pipeline.py (systemd: p-trade-live, always-on during market hours)
    KiteTicker (background thread) --tick queue--> _drain_ticks_loop
    _boundary_loop -- hourly bucket close --> _process_closed_candles
        --> application/signal_pipeline.py: _evaluate_from_stored_candles (per symbol, concurrent)
        --> application/signal_pipeline.py: _collect_and_open_ranked_positions (ranked, sequential per book)
              paper book:   application/paper_trading.py
              futures book: application/futures_trading.py (paper only, Nifty50-gated)
              cash book:    application/live_cash_execution.py (REAL MONEY)
                  gates: paper_trading.is_eligible, entry_quality_filter, conviction_filter
                  execute_cash_entry -> infrastructure/kite.py: KiteOrderExecutor
                  application/gtt_bracket.py -- OCO target/stop GTT
        --> application/signal_pipeline.py: _process_symbol (per symbol, concurrent) -- exits,
              notifications, derivatives shadow tracking (options_shadow.py/futures_shadow.py,
              analysis-only, never real orders)
    tick-level: stop_loss/trailing_stop checks (paper book only) each tick
    watchdogs: _ticker_watchdog_loop (stale-tick force-reconnect), _token_refresh_loop,
               _heartbeat_loop, _run_until_market_close

signals.py (cron fallback / backfill / dashboard manual trigger -- no per-cycle DB toggle
    refresh, builds one static LiveCashToggleState from AppConfig at the top of the run)
    --> application/signal_pipeline.py: run_signal_pipeline (same functions as above)

webapp.py (systemd: p-trade-dashboard, FastAPI, cookie session auth)
    admin vs viewer roles (_require_admin / _require_session)
    /kite/login, /kite/callback -- Kite OAuth, token stored via TursoKiteSessionRepository
    /api/live-cash-positions, /api/live-cash-positions/{symbol}/exit -- application/manual_exit.py
    /api/config, /api/live-cash-trading -- writes .env / live_cash_toggle table
    /api/trigger, /api/trigger-backtest -- runs signals.py / derivatives_backtest.py as subprocess
    /api/live-pipeline-health, /restart -- systemctl restart p-trade-live

infrastructure/kite.py -- KiteConnect wrapper: KiteInstrumentMap (symbol->token/tick-size),
    KiteProvider (historical candles), KiteDerivativesChain (options/futures analysis, never
    real orders), KiteOrderExecutor (real orders: place_cash_market_order, place_cash_bracket_gtt,
    holding_quantity, wait_for_fill)

infrastructure/db/ -- one module per repository, local SQLite via aiosqlite (_shared.py),
    "Turso*" naming kept for historical reasons only (hosted Turso retired 2026-08-20)
```

## Domain types today (`domain/models.py`)

`Signal`, `Trade`, `PaperPosition`, `FuturesPaperPosition`,
`OptionsShadowTrade`, `FuturesShadowTrade`, `LiveOrderLeg`,
`PaperBenchmarkPosition`, `GttBracket` -- all flat `@dataclass(frozen=True,
slots=True)` (only `GttBracket` is mutable, since its `status` transitions
in place). No shared base type, no explicit lifecycle enum -- `status` is
a bare string (`"open"`/`"closed"` or Kite's own order-status vocabulary)
on nearly every one. This is exactly the gap Phases 1/5/6 address, without
touching any of the existing types' fields.

`infrastructure/db/live_cash_toggle.py`'s `LiveCashToggleState` is the one
existing example of what this project calls "runtime state" separated from
`config/settings.py`'s `AppConfig` ("static config") -- established this
session specifically to fix a real incident (see below). Phase 11 should
generalize this pattern, not invent a new one.

## Known gaps (found this session's review, before this refactor)

1. **[Fixed by stage 6] Unclosed-leg blind spot on the exit side.**
   `infrastructure/db/live_orders.py`'s `get_open_cash_legs`/
   `get_all_open_cash_legs` only recognize `status='COMPLETE'` legs as
   "open." `get_unclosed_cash_legs` (COMPLETE+OPEN+UNKNOWN) is used to
   *block* re-entry, but every exit-eligibility check --
   `signal_pipeline.py`'s strategy-SELL branch, `live_cash_execution.
   execute_cash_exit`, `manual_exit.exit_position`, and `webapp.py`'s
   `/api/live-cash-positions` -- uses the narrower COMPLETE-only view.
   An entry leg that lands `UNKNOWN` (a real `wait_for_fill` API
   exception mid-poll -- already happened once, on a SELL leg, for
   UNIONBANK.NS on 2026-08-28) may represent an actual filled position
   that becomes invisible everywhere and un-exitable by any automated or
   manual path in this codebase. Untested today.
2. **[Addressed incrementally by stages 4-6, not a full rewrite]** No
   explicit `OrderIntent`/idempotency key independent of `basket_id` --
   a process restart between "Kite accepted the order" and "the leg got
   recorded" has no dedicated detection path today (partially covered by
   `get_unclosed_cash_legs`'s per-symbol check on the next call, but not
   verified against the exact intent that was in flight).
3. **[Out of scope, noted for awareness]** `webapp.py`'s session cookie
   omits `secure=True` (relies on the reverse proxy enforcing HTTPS).
   Not part of this refactor's scope (`projectedPlann.md` doesn't ask for
   an auth hardening pass); flagged here so it isn't lost.

## Stage progress notes (updated as the refactor lands)

- **Phase 8 (pipeline decomposition)**: done in full -- `signal_pipeline.py`
  is now a 101-line facade over `application/pipeline/*`.
- **Phase 9 (live ticker pipeline refactor)**: deliberately partial.
  `live_pipeline.py`'s `LiveTickerPipeline` is a single stateful class with
  tightly-coupled mutable state (ticker connection, tick queue,
  aggregators, caches) shared between what would become the
  `KiteTickerAdapter` and the application-level orchestration -- unlike
  `signal_pipeline.py`'s free functions, there is no risk-free "just move
  it" split here; a real extraction needs `KiteTicker`/`KiteConnect`
  construction pulled out from where `__init__`/`_connect_ticker`
  currently build them inline, into an injectable seam. That's real,
  further work, not done in this pass. What *is* done: a genuine
  characterization-test harness (`tests/test_live_pipeline.py`) for every
  piece of `LiveTickerPipeline` testable without that seam -- tick
  callbacks, the tick-level stop-loss/trailing-stop check, and
  `_run_until_first_exit` (the exact mechanism behind the 2026-08-18
  orphaned-task incident) -- so the adapter extraction, if/when attempted,
  has a real regression net for at least those pieces. Full connection/
  reconnect/stale-hang scenarios through `run_forever` remain untested
  (they need the same seam). Be honest about this in the final report.

- **Phase 16 (dashboard decomposition)**: deliberately partial, same
  reasoning as Phase 9. `webapp.py`'s large inline HTML/CSS/JS was
  already externalized into `templates/` before this refactor (not new
  work here). Split out: `web/services/auth.py` (session state +
  `_authenticate`/`_require_session`/`_require_admin`) -- the one piece
  every other route already depends on, and the most security-sensitive
  code in the file, now with its own focused module and dedicated tests
  (`tests/test_web_auth.py`) that didn't exist before. The remaining ~30
  routes stay in `webapp.py`: there is no FastAPI `TestClient`/route-level
  test suite for this file (confirmed absent before this refactor too),
  so a larger mechanical split (into `web/routes/*.py` per group) would
  have no automated safety net against a mistake in a live, already-in-use
  dashboard -- exactly the risk profile this refactor's own ground rules
  say to avoid attempting without characterization tests first.

## Non-goals reaffirmed

Every threshold, filter, and formula below is a read-only input to the new
abstractions built in this refactor -- never edited as part of it:
`AlphaEngine` (alpha_engine.py), ranking score/deciles (application/
ranking.py), `paper_trading.MIN_WIN_RATE`/`MIN_CLOSED_TRADES`,
`entry_quality_filter`'s two floors, `conviction_filter.
CONVICTION_THRESHOLD`, `paper_trading.STOP_LOSS_PCT`/
`TRAILING_STOP_ACTIVATION_PCT`/`TRAILING_STOP_TRAIL_PCT`,
`gtt_bracket.py`'s 10%/3%/8%/15% figures, `AppConfig.
live_cash_entry_cutoff_ist`.
