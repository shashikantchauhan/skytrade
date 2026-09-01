"""Per-symbol AlphaEngine evaluation -- split out of ``signal_pipeline.py``
(Phase 8, see ``application/pipeline/__init__.py``). No behavior changed;
every function's body moved as-is.
"""

import asyncio
import logging

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
from trading_scanner.domain.ports import CandleRepository, EngineState, EngineStateRepository

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


async def _evaluate_symbol(
    symbol: str,
    config: AppConfig,
    provider: MarketDataProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> tuple[FastPredictResult, Candle] | None:
    """Download, store, and evaluate the newest bar for one symbol -- the
    polling/historical-API path (cron, manual dashboard trigger, backfill).

    Returns None if the symbol is still warming up (<200 candles) or if no
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
) -> tuple[FastPredictResult, Candle] | None:
    """Evaluate the newest already-*stored* bar for one symbol -- shared by
    both the download-based path above (which upserts before calling this)
    and the live-ticker path (``live_pipeline.py``, which upserts one
    freshly-closed candle from aggregated ticks before calling this).

    Returns None if the symbol is still warming up (<200 candles) or if no
    new candle has arrived since the last run (nothing new to evaluate).
    """
    logger = logging.getLogger(__name__)
    accumulated = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_FULL_HISTORY
    )
    if len(accumulated) < _MINIMUM_CANDLES:
        logger.info(
            "Warming up %s: %d/%d candles stored.", symbol, len(accumulated), _MINIMUM_CANDLES
        )
        return None

    engine_state = await engine_state_repository.get_state(symbol, config.candle_interval)
    newest_timestamp = accumulated[-1].timestamp.isoformat()
    if newest_timestamp == engine_state.last_bar_timestamp:
        # The newest stored candle is the same bar already advanced into the
        # queue last run (no new candle has closed since) -- advancing it
        # again would double-count its contribution to the persisted
        # neighbor queue, corrupting future predictions. Nothing changed;
        # nothing to do.
        logger.info("No new candle for %s since the last run; skipping.", symbol)
        return None

    history = _candles_to_dataframe(accumulated)
    if engine_state.queue_json is None:
        logger.info("Bootstrapping ANN neighbor queue for %s (one-time, slower).", symbol)
        bootstrap = bootstrap_queue_state(engine, history.iloc[:-1])
        signal_previous = bootstrap.signal_previous
        queue_state = bootstrap.queue_state
        exit_state = bootstrap.exit_state
    else:
        signal_previous = engine_state.signal
        queue_state = QueueState.from_json(engine_state.queue_json)
        exit_state = ExitState.from_json(engine_state.exit_state_json)

    result = evaluate_latest_bar(engine, history, signal_previous, queue_state, exit_state)
    await engine_state_repository.set_state(
        symbol,
        config.candle_interval,
        EngineState(
            signal=result.signal_previous,
            queue_json=result.queue_state.to_json(),
            exit_state_json=result.exit_state.to_json(),
            last_bar_timestamp=newest_timestamp,
        ),
    )
    return result, accumulated[-1]
