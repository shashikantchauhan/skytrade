"""Per-symbol AlphaEngine evaluation -- split out of ``signal_pipeline.py``
(Phase 8, see ``application/pipeline/__init__.py``).

2026-09-02: gained the ability to catch up on more than one candle closed
since the last run (see ``_evaluate_from_stored_candles``'s own docstring)
-- everything else moved as-is.
"""

import asyncio
import logging
from collections.abc import Sequence

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.fast_predict import (
    ExitState,
    FastPredictResult,
    QueueState,
    bootstrap_queue_state,
    evaluate_latest_bar,
)
from trading_scanner.application.pipeline.market_data import (
    MarketDataProvider,
    _candles_to_dataframe,
    _dataframe_to_candles,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle
from trading_scanner.domain.ports import (
    CandleRepository,
    EngineState,
    EngineStateRepository,
    Notifier,
)

# AlphaEngine's regime filter needs ~200 bars of warm-up before predictions are
# meaningful; below this the pipeline skips a symbol instead of notifying on
# noise. This mirrors application.validation.validate_candles's own minimum.
_MINIMUM_CANDLES = 200

# Small recent window fetched every run once a symbol already has history --
# just enough to cover any gap since the last hourly run.
_RECENT_WINDOW_DAYS = 5

# Yahoo Finance caps 1h/60m intraday history at 730 days regardless of what is
# requested; 729 is the largest window that reliably stays under that cap. A
# brand-new symbol is backfilled with this once, so AlphaEngine analyzes
# roughly the same depth of history TradingView's chart typically has, rather
# than waiting months for the small recent window to accumulate that far.
_BACKFILL_WINDOW_DAYS = 729

# evaluate_latest_bar's neighbor search always looks at indices [0, max_bars_back-1]
# of whatever candle history is passed in -- i.e. the *oldest* stored candles.
# Fetching anything less than the full accumulated history (e.g. a "most recent
# N" cap) would drop those oldest rows and shift what index 0 means, silently
# breaking the TradingView match this pipeline exists to preserve. See
# fast_predict.py's module docstring for the full explanation.
_FULL_HISTORY = None

# 2026-09-02: a gap wider than this (candles closed while the pipeline was
# offline -- Kite session expired, process down) falls back to jumping
# straight to the newest bar instead of replaying every one individually --
# see _evaluate_from_stored_candles's own docstring for why replaying is the
# normal behavior below this cap. ~50 hourly bars is roughly two trading
# weeks; a gap this wide happening at all means something needs a human's
# attention regardless, and replaying that many bars inline during a live
# cycle risks stalling the whole symbol batch behind one backlog.
_MAX_CATCH_UP_BARS = 50


async def _evaluate_symbol(
    symbol: str,
    config: AppConfig,
    provider: MarketDataProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> list[tuple[FastPredictResult, Candle]]:
    """Download, store, and evaluate every newly-closed bar for one symbol
    since the last run -- the polling/historical-API path (cron, manual
    dashboard trigger, backfill). See ``_evaluate_from_stored_candles``
    (this calls straight through to it) for the catch-up contract: almost
    always returns zero or one result, more than one after an outage gap.

    Empty list if the symbol is still warming up (<200 candles) or if no
    new candle has arrived since the last run (nothing new to evaluate).

    Zerodha's own guidance is that the Historical Data API is for backfill/
    backtesting, not live signals (it can lag the current session's candles
    by hours -- confirmed live and via Kite's dev forum). The live-ticker
    path (``infrastructure/kite_ticker.py`` -> ``live_pipeline.py``) is the
    real-time replacement for market hours; this function's download step
    stays around for backfill, catch-up after downtime, and the dashboard's
    manual "run pipeline now" button.
    """
    logger = logging.getLogger(__name__)
    existing = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_MINIMUM_CANDLES
    )
    needs_backfill = len(existing) < _MINIMUM_CANDLES
    window_days = _BACKFILL_WINDOW_DAYS if needs_backfill else _RECENT_WINDOW_DAYS
    if needs_backfill:
        logger.info("Backfilling %s: downloading %d days of history.", symbol, window_days)

    # get_recent_history is a blocking call (yfinance is synchronous); running
    # it in a thread lets other symbols' downloads proceed concurrently on
    # the event loop instead of serializing behind this one.
    downloaded = await asyncio.to_thread(
        provider.get_recent_history, symbol, config.candle_interval, window_days
    )
    await candle_repository.upsert_candles(
        symbol, config.candle_interval, _dataframe_to_candles(symbol, downloaded)
    )
    return await _evaluate_from_stored_candles(
        symbol, config, engine, candle_repository, engine_state_repository
    )


async def _evaluate_from_stored_candles(
    symbol: str,
    config: AppConfig,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> list[tuple[FastPredictResult, Candle]]:
    """Evaluate every already-*stored* bar closed since the last run for one
    symbol -- shared by both the download-based path above (which upserts
    before calling this) and the live-ticker path (``live_pipeline.py``,
    which upserts one freshly-closed candle from aggregated ticks before
    calling this).

    2026-09-02: used to evaluate only the single newest stored candle,
    silently skipping any *other* candle that closed in between two calls
    (a gap that closes more than one candle -- Kite session expired,
    process down/restarting). In production, a Kite session left expired
    for 30-104 minutes at market open on three consecutive trading days
    meant every candle that closed during those windows was never
    evaluated at all, before *or* after logging back in -- no catch-up
    existed anywhere, including the dashboard's manual "Run pipeline now".
    Worse, the skipped candle's own contribution to the persisted ANN
    neighbor queue (``QueueState``) was lost too, not just its signal.

    Now replays every newly-closed candle since ``engine_state.last_bar_
    timestamp`` individually, in chronological order, carrying the
    ``signal_previous``/``queue_state``/``exit_state`` forward between each
    one exactly as consecutive real-time calls would have -- so the engine
    ends up in the same state it would be in had it never gone offline,
    and every candle's own BUY/SELL is actually computed and returned.
    Persists engine_state after *each* replayed bar (not just at the end)
    so a crash mid-catch-up doesn't force redoing already-caught-up bars
    next run.

    Returns one ``(result, candle)`` pair per newly-closed bar, oldest
    first -- almost always exactly one (the normal, no-gap case), zero if
    the symbol is still warming up (<200 candles) or nothing new has
    closed, and more than one only after an outage gap. Callers must not
    treat every entry as "place a real order for this" -- only the *last*
    entry is current; anything before it is stale catch-up by the time
    it's computed and should be notification-only (see ``live_pipeline.
    py``/``orchestrator.py``'s handling of this list).
    """
    logger = logging.getLogger(__name__)
    accumulated = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_FULL_HISTORY
    )
    if len(accumulated) < _MINIMUM_CANDLES:
        logger.info(
            "Warming up %s: %d/%d candles stored.", symbol, len(accumulated), _MINIMUM_CANDLES
        )
        return []

    engine_state = await engine_state_repository.get_state(symbol, config.candle_interval)
    newest_timestamp = accumulated[-1].timestamp.isoformat()
    if newest_timestamp == engine_state.last_bar_timestamp:
        # The newest stored candle is the same bar already advanced into the
        # queue last run (no new candle has closed since) -- advancing it
        # again would double-count its contribution to the persisted
        # neighbor queue, corrupting future predictions. Nothing changed;
        # nothing to do.
        logger.info("No new candle for %s since the last run; skipping.", symbol)
        return []

    history = _candles_to_dataframe(accumulated)
    if engine_state.queue_json is None:
        logger.info("Bootstrapping ANN neighbor queue for %s (one-time, slower).", symbol)
        bootstrap = bootstrap_queue_state(engine, history.iloc[:-1])
        signal_previous = bootstrap.signal_previous
        queue_state = bootstrap.queue_state
        exit_state = bootstrap.exit_state
        new_bar_indices = [len(accumulated) - 1]
    else:
        signal_previous = engine_state.signal
        queue_state = QueueState.from_json(engine_state.queue_json)
        exit_state = ExitState.from_json(engine_state.exit_state_json)
        new_bar_indices = _find_new_bar_indices(accumulated, engine_state.last_bar_timestamp)
        if len(new_bar_indices) > _MAX_CATCH_UP_BARS:
            logger.warning(
                "%s: %d candles closed since the last run -- gap too wide to "
                "replay safely, jumping straight to the newest bar instead "
                "(their own signals will not be evaluated).",
                symbol, len(new_bar_indices),
            )
            new_bar_indices = [len(accumulated) - 1]

    results: list[tuple[FastPredictResult, Candle]] = []
    for index in new_bar_indices:
        result = evaluate_latest_bar(
            engine, history.iloc[: index + 1], signal_previous, queue_state, exit_state
        )
        await engine_state_repository.set_state(
            symbol,
            config.candle_interval,
            EngineState(
                signal=result.signal_previous,
                queue_json=result.queue_state.to_json(),
                exit_state_json=result.exit_state.to_json(),
                last_bar_timestamp=accumulated[index].timestamp.isoformat(),
            ),
        )
        signal_previous = result.signal_previous
        queue_state = result.queue_state
        exit_state = result.exit_state
        results.append((result, accumulated[index]))
    return results


def _find_new_bar_indices(accumulated: list[Candle], last_bar_timestamp: str | None) -> list[int]:
    """Indices into ``accumulated`` (chronological) for every candle that
    closed after ``last_bar_timestamp`` -- almost always just the last
    index, more than one after an outage gap. Falls back to just the last
    index if ``last_bar_timestamp`` isn't found in ``accumulated`` at all
    (should not happen in practice -- it's always set from a previous run's
    own newest-candle timestamp -- but a rewritten/pruned history must
    never trigger an unbounded replay from wherever the search fails)."""
    if last_bar_timestamp is None:
        return [len(accumulated) - 1]
    for position, candle in enumerate(accumulated):
        if candle.timestamp.isoformat() == last_bar_timestamp:
            new_indices = list(range(position + 1, len(accumulated)))
            return new_indices if new_indices else [len(accumulated) - 1]
    return [len(accumulated) - 1]


async def _notify_stale_catch_up_signals(
    symbol: str,
    stale: Sequence[tuple[FastPredictResult, Candle]],
    notifier: Notifier | None,
) -> None:
    """Tell the user about any BUY/SELL found on a candle caught up from an
    outage gap (see ``_evaluate_from_stored_candles``) -- these are never
    acted on automatically: by the time a stale candle's signal is computed
    (the pipeline was offline when it actually closed), price has already
    moved past it, so a real order placed now wouldn't be the same trade
    that actually fired. Notification-only, matching the 2026-09-02
    decision on how missed signals should be handled -- see
    docs/decisions/010-outage-catch-up-replay.md.

    Silent for a NEUTRAL catch-up bar (nothing was missed) and for the
    normal case (``stale`` empty -- no gap happened)."""
    if notifier is None:
        return
    for result, candle in stale:
        if result.signal == "NEUTRAL":
            continue
        try:
            await notifier.send_text(
                "⏮️ <b>CAUGHT-UP SIGNAL (not acted on)</b>\n"
                f"{symbol}: {result.signal} at {candle.timestamp.isoformat()} -- this "
                "candle closed while the pipeline was offline (Kite session expired, "
                "or the process was down/restarting). Too stale to place a real order "
                "against now that it's been caught up -- shown for your own review only."
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to send caught-up-signal notification for %s", symbol
            )
