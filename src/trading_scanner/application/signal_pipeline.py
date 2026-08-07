"""Hourly pipeline: accumulate candles per symbol and notify on AlphaEngine signals."""

import logging
from decimal import Decimal

import pandas as pd

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.fast_predict import (
    ExitState,
    FastPredictResult,
    QueueState,
    bootstrap_queue_state,
    evaluate_latest_bar,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, Signal, SignalSide, Trade
from trading_scanner.domain.ports import (
    CandleRepository,
    EngineState,
    EngineStateRepository,
    Notifier,
    SignalRepository,
    TradeRepository,
)
from trading_scanner.infrastructure.yahoo import YahooProvider

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

# AlphaEngine's constructor defaults mirror the Pine script's own stock
# defaults. This deployment's actual TradingView chart has two inputs
# overridden from those defaults -- confirmed by comparing the chart's Inputs
# panel against AlphaEngine's default constructor -- so signals here must be
# generated with the same overrides or they will not match the chart:
#   - includeFullHistory is checked (True): the ANN neighbor search scans the
#     entire stored history rather than only the most recent max_bars_back
#     window, which directly changes `prediction` values.
#   - useDynamicExits is checked (True): affects end_long/end_short only, not
#     BUY/SELL entries, but is included so exit behavior matches too.
_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}

# evaluate_latest_bar's neighbor search always looks at indices [0, max_bars_back-1]
# of whatever candle history is passed in -- i.e. the *oldest* stored candles.
# Fetching anything less than the full accumulated history (e.g. a "most recent
# N" cap) would drop those oldest rows and shift what index 0 means, silently
# breaking the TradingView match this pipeline exists to preserve. See
# fast_predict.py's module docstring for the full explanation.
_FULL_HISTORY = None

_STRATEGY_NAME = "lorentzian"


async def run_signal_pipeline(
    config: AppConfig,
    symbols: list[str],
    candle_repository: CandleRepository,
    signal_repository: SignalRepository,
    engine_state_repository: EngineStateRepository,
    trade_repository: TradeRepository,
    notifier: Notifier,
) -> None:
    """Ingest recent candles and notify on new BUY/SELL signals for each symbol.

    If ``config.index_symbol`` is set, it is evaluated once per run (same
    machinery, no trades/notifications of its own) and its current state is
    attached to every stock signal's rationale -- purely informational, so
    you can judge whether a stock signal lines up with the broader market or
    looks like noise against it. It is never used to suppress a signal.
    """
    logger = logging.getLogger(__name__)
    provider = YahooProvider()
    engine = AlphaEngine(**_ENGINE_SETTINGS)

    index_result = None
    if config.index_symbol:
        try:
            index_evaluated = await _evaluate_symbol(
                config.index_symbol, config, provider, engine, candle_repository,
                engine_state_repository,
            )
            index_result = index_evaluated[0] if index_evaluated is not None else None
        except Exception:
            logger.exception("Unexpected exception while evaluating index %s", config.index_symbol)

    for symbol in symbols:
        try:
            await _process_symbol(
                symbol,
                config,
                provider,
                engine,
                candle_repository,
                signal_repository,
                engine_state_repository,
                trade_repository,
                notifier,
                index_result,
            )
        except Exception:
            logger.exception("Unexpected exception while processing %s", symbol)


async def _evaluate_symbol(
    symbol: str,
    config: AppConfig,
    provider: YahooProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> tuple[FastPredictResult, Candle] | None:
    """Download, store, and evaluate the newest bar for one symbol.

    Returns None if the symbol is still warming up (<200 candles) or if no
    new candle has arrived since the last run (nothing new to evaluate).
    """
    logger = logging.getLogger(__name__)
    existing = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_MINIMUM_CANDLES
    )
    needs_backfill = len(existing) < _MINIMUM_CANDLES
    window_days = _BACKFILL_WINDOW_DAYS if needs_backfill else _RECENT_WINDOW_DAYS
    if needs_backfill:
        logger.info("Backfilling %s: downloading %d days of history.", symbol, window_days)

    downloaded = provider.get_recent_history(symbol, config.candle_interval, window_days)
    await candle_repository.upsert_candles(
        symbol, config.candle_interval, _dataframe_to_candles(symbol, downloaded)
    )

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


async def _process_symbol(
    symbol: str,
    config: AppConfig,
    provider: YahooProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    signal_repository: SignalRepository,
    engine_state_repository: EngineStateRepository,
    trade_repository: TradeRepository,
    notifier: Notifier,
    index_result: FastPredictResult | None,
) -> None:
    """Evaluate the newest bar, record/close trades, and notify for one symbol."""
    evaluated = await _evaluate_symbol(
        symbol, config, provider, engine, candle_repository, engine_state_repository
    )
    if evaluated is None:
        return
    result, newest_candle = evaluated

    if result.end_long:
        await trade_repository.close_open_trade(
            symbol,
            config.candle_interval,
            SignalSide.BUY,
            newest_candle.timestamp,
            newest_candle.close,
        )
    if result.end_short:
        await trade_repository.close_open_trade(
            symbol,
            config.candle_interval,
            SignalSide.SELL,
            newest_candle.timestamp,
            newest_candle.close,
        )

    if result.signal not in ("BUY", "SELL"):
        return

    side = SignalSide.BUY if result.signal == "BUY" else SignalSide.SELL
    await trade_repository.open_trade(
        config.candle_interval,
        Trade(
            symbol=symbol,
            side=side,
            entry_timestamp=newest_candle.timestamp,
            entry_price=newest_candle.close,
            prediction_at_entry=result.prediction,
            is_early_signal_flip=result.is_early_signal_flip,
        ),
    )

    rationale = f"prediction={result.prediction}"
    if index_result is not None:
        rationale += (
            f"; index({config.index_symbol})={index_result.signal},"
            f"pred={index_result.prediction},early_flip={index_result.is_early_signal_flip}"
        )
    signal = Signal(
        symbol=symbol,
        side=side,
        strategy=_STRATEGY_NAME,
        timestamp=newest_candle.timestamp,
        price=newest_candle.close,
        rationale=rationale,
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    await notifier.send_signal(signal)
    await signal_repository.record(signal.fingerprint, signal.timestamp)


def _dataframe_to_candles(symbol: str, data: pd.DataFrame) -> list[Candle]:
    """Convert a downloaded OHLCV DataFrame into domain Candle objects."""
    return [
        Candle(
            symbol=symbol,
            timestamp=timestamp.to_pydatetime(),
            open=Decimal(str(row["Open"])),
            high=Decimal(str(row["High"])),
            low=Decimal(str(row["Low"])),
            close=Decimal(str(row["Close"])),
            volume=int(row["Volume"]),
        )
        for timestamp, row in data.iterrows()
    ]


def _candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert chronological Candle objects into the OHLCV DataFrame AlphaEngine expects."""
    return pd.DataFrame(
        {
            "Open": [float(candle.open) for candle in candles],
            "High": [float(candle.high) for candle in candles],
            "Low": [float(candle.low) for candle in candles],
            "Close": [float(candle.close) for candle in candles],
            "Volume": [candle.volume for candle in candles],
        },
        index=pd.DatetimeIndex([candle.timestamp for candle in candles]),
    )
