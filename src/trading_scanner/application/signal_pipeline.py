"""Hourly pipeline: accumulate candles per symbol and notify on AlphaEngine signals."""

import logging
from datetime import UTC
from decimal import Decimal

import pandas as pd

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application import paper_trading
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
    PaperAccountRepository,
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
    paper_account_repository: PaperAccountRepository,
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
                paper_account_repository,
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
    paper_account_repository: PaperAccountRepository,
    notifier: Notifier,
    index_result: FastPredictResult | None,
) -> None:
    """Evaluate the newest bar, record/close trades, and notify for one symbol.

    Trade bookkeeping mirrors Pine's own ``ml.backtest`` block order and
    scoring exactly (see ``application/backtest.py``'s module docstring): a
    new entry first abandons -- without scoring -- whatever opposite-side
    position was still open, then that side's own exit (if it fires the same
    bar) is applied. Entry/exit price uses Pine's ``(high+low+open+open)/4``
    scoring convention, not the close, so live trades stay consistent with
    the historical backtest.

    BUY entries additionally attempt to open a real paper-trading position
    (see ``application/paper_trading.py``) -- gated on the symbol's own
    BUY-only win-rate track record and on the account having free capital.
    SELL signals never touch the paper account: NSE cash market doesn't
    allow short selling for multi-day holds, so they stay informational only.
    """
    evaluated = await _evaluate_symbol(
        symbol, config, provider, engine, candle_repository, engine_state_repository
    )
    if evaluated is None:
        return
    result, newest_candle = evaluated
    market_price = _market_price(newest_candle)
    paper_note = None

    if result.signal == "BUY":
        await trade_repository.abandon_open_trade(symbol, config.candle_interval, SignalSide.SELL)
        await trade_repository.open_trade(
            config.candle_interval,
            Trade(
                symbol=symbol,
                side=SignalSide.BUY,
                entry_timestamp=newest_candle.timestamp,
                entry_price=market_price,
                prediction_at_entry=result.prediction,
                is_early_signal_flip=result.is_early_signal_flip,
            ),
        )
        paper_note = await _open_paper_position(
            symbol, config, newest_candle.timestamp, market_price,
            trade_repository, paper_account_repository,
        )
    if result.end_long:
        await _notify_exit(
            symbol, config, SignalSide.BUY, newest_candle.timestamp, market_price,
            trade_repository, signal_repository, notifier,
        )
        await trade_repository.close_open_trade(
            symbol, config.candle_interval, SignalSide.BUY, newest_candle.timestamp, market_price
        )
        await _close_paper_position(
            symbol, newest_candle.timestamp, market_price,
            paper_account_repository, signal_repository, notifier,
        )
    if result.signal == "SELL":
        await trade_repository.abandon_open_trade(symbol, config.candle_interval, SignalSide.BUY)
        await trade_repository.open_trade(
            config.candle_interval,
            Trade(
                symbol=symbol,
                side=SignalSide.SELL,
                entry_timestamp=newest_candle.timestamp,
                entry_price=market_price,
                prediction_at_entry=result.prediction,
                is_early_signal_flip=result.is_early_signal_flip,
            ),
        )
    if result.end_short:
        await _notify_exit(
            symbol, config, SignalSide.SELL, newest_candle.timestamp, market_price,
            trade_repository, signal_repository, notifier,
        )
        await trade_repository.close_open_trade(
            symbol, config.candle_interval, SignalSide.SELL, newest_candle.timestamp, market_price
        )

    if result.signal not in ("BUY", "SELL"):
        return

    side = SignalSide.BUY if result.signal == "BUY" else SignalSide.SELL
    rationale = f"prediction={result.prediction}"
    win_rate_summary = await _win_rate_summary(symbol, config, trade_repository)
    if win_rate_summary is not None:
        rationale += f"; {win_rate_summary}"
    if side == SignalSide.SELL:
        rationale += "; informational only -- not tradeable in NSE cash market"
    elif paper_note is not None:
        rationale += f"; {paper_note}"
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


async def _notify_exit(
    symbol: str,
    config: AppConfig,
    side: SignalSide,
    exit_timestamp,
    exit_price: Decimal,
    trade_repository: TradeRepository,
    signal_repository: SignalRepository,
    notifier: Notifier,
) -> None:
    """Notify that an open position closed, with its realized pnl_percent.

    Reads the still-open trade's entry price before ``close_open_trade`` is
    called, so this must run first. A distinct ``strategy`` tag keeps this
    signal's fingerprint from ever colliding with an entry notification at
    the same symbol/side/timestamp (see Signal.fingerprint).
    """
    trades = await trade_repository.get_trades(symbol, config.candle_interval)
    open_trade = next(
        (trade for trade in trades if trade.side == side and trade.status == "open"), None
    )
    if open_trade is None:
        return  # Nothing was actually open (e.g. abandoned by an opposite entry earlier).

    pnl_percent = (
        (exit_price - open_trade.entry_price) / open_trade.entry_price * 100
        if side == SignalSide.BUY
        else (open_trade.entry_price - exit_price) / open_trade.entry_price * 100
    )
    signal = Signal(
        symbol=symbol,
        side=side,
        strategy=f"{_STRATEGY_NAME}-exit",
        timestamp=exit_timestamp,
        price=exit_price,
        rationale=(
            f"exit; entry={open_trade.entry_price}, exit={exit_price}, pnl={pnl_percent:.2f}%"
        ),
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    await notifier.send_signal(signal)
    await signal_repository.record(signal.fingerprint, signal.timestamp)


async def _open_paper_position(
    symbol: str,
    config: AppConfig,
    entry_timestamp,
    entry_price: Decimal,
    trade_repository: TradeRepository,
    paper_account_repository: PaperAccountRepository,
) -> str | None:
    """Attempt to open a real paper position for a BUY entry; return a status note.

    Returns a short human-readable reason whenever no position was opened
    (not yet eligible, or the account is out of free capital) so the caller
    can surface it in the notification instead of silently skipping.
    """
    if not await paper_trading.is_eligible(symbol, config.candle_interval, trade_repository):
        return "paper: not eligible yet (win_rate<55% or insufficient trade history)"
    position = await paper_trading.try_open_position(
        symbol, entry_timestamp, entry_price, paper_account_repository
    )
    if position is None:
        return "paper: SKIPPED (no capital available)"
    return f"paper: opened {position.quantity} qty (₹{position.capital_allocated:.0f})"


async def _close_paper_position(
    symbol: str,
    exit_timestamp,
    exit_price: Decimal,
    paper_account_repository: PaperAccountRepository,
    signal_repository: SignalRepository,
    notifier: Notifier,
) -> None:
    """Close an open paper position (if any) and notify the realized P&L."""
    closed = await paper_account_repository.close_position(symbol, exit_timestamp, exit_price)
    if closed is None:
        return  # Nothing was ever opened for this symbol (not eligible / no capacity).
    signal = Signal(
        symbol=symbol,
        side=SignalSide.BUY,
        strategy=f"{_STRATEGY_NAME}-paper-exit",
        timestamp=exit_timestamp,
        price=exit_price,
        rationale=(
            f"paper CLOSE {closed.quantity} qty; entry={closed.entry_price}, "
            f"exit={exit_price}, pnl=₹{closed.pnl_amount:.0f}"
        ),
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    await notifier.send_signal(signal)
    await signal_repository.record(signal.fingerprint, signal.timestamp)


async def _win_rate_summary(
    symbol: str, config: AppConfig, trade_repository: TradeRepository
) -> str | None:
    """Summarize this symbol's historical closed BUY-trade win rate for a notification.

    BUY-only, matching ``paper_trading._buy_only_win_rate`` exactly -- this is
    the same number the paper account's eligibility gate actually uses, so a
    signal tagged "not eligible yet" is never paired with a rationale showing
    a healthier combined BUY+SELL number that would make the rejection look
    wrong. SELL trades are excluded even for a SELL-signal notification since
    they can never affect the (BUY-only) paper account either way.

    Returns None if there's no closed BUY-trade history yet (a brand-new
    symbol, or one whose only trades are still open) -- nothing meaningful to
    show.
    """
    trades = await trade_repository.get_trades(symbol, config.candle_interval)
    closed = [
        trade for trade in trades if trade.side == SignalSide.BUY and trade.status == "closed"
    ]
    if not closed:
        return None
    wins = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent > 0)
    losses = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent < 0)
    win_rate = 100 * wins / len(closed)
    return f"win_rate={win_rate:.1f}%({wins}W/{losses}L)"


def _market_price(candle: Candle) -> Decimal:
    """Pine's ``ml.backtest`` scoring price: (high + low + open + open) / 4.

    Not the close -- matches ``application/backtest.py``'s historical
    replay so live and backtested trades use the same price convention.
    """
    return (candle.high + candle.low + candle.open + candle.open) / 4


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
    """Convert chronological Candle objects into the OHLCV DataFrame AlphaEngine expects.

    Normalizes every timestamp to UTC before building the index -- candles
    stored across different runs can carry equivalent-offset but distinct
    tzinfo objects (e.g. a fixed +05:30 offset vs. a zoneinfo-based one),
    which pandas refuses to unify into one DatetimeIndex without this
    (raises "Tz-aware datetime.datetime cannot be converted to datetime64
    unless utc=True"). AlphaEngine only depends on chronological order and
    OHLCV values, never the displayed hour, so this is safe -- the original
    Candle objects (with their real tzinfo) are still used everywhere else.
    """
    return pd.DataFrame(
        {
            "Open": [float(candle.open) for candle in candles],
            "High": [float(candle.high) for candle in candles],
            "Low": [float(candle.low) for candle in candles],
            "Close": [float(candle.close) for candle in candles],
            "Volume": [candle.volume for candle in candles],
        },
        index=pd.DatetimeIndex([candle.timestamp.astimezone(UTC) for candle in candles]),
    )
