"""One-time CLI: replay full stored history for every symbol into real trades.

Run once, after candles have already been backfilled (see
``application/signal_pipeline.py``'s bootstrap run). Safe to re-run: it
clears the ``trades`` table first, so re-running never double-inserts --
open trades from live hourly runs are naturally regenerated at the same bar
since the replay covers the full history up to now.
"""

import asyncio
import logging
from decimal import Decimal

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.backtest import (
    PineTrade,
    compute_historical_events,
    replay_pine_backtest,
)
from trading_scanner.application.symbols import SymbolLoader
from trading_scanner.config.settings import AppConfig, load_config
from trading_scanner.domain.models import SignalSide, Trade
from trading_scanner.infrastructure.turso import (
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

# Mirrors application/signal_pipeline.py's production engine settings exactly
# -- the historical backtest must use the same configuration as the live
# pipeline or its trade history would not describe the same strategy.
_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}

_MINIMUM_CANDLES = 200


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()

    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set to run the historical backtest.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    candle_repository = TursoCandleRepository(client)
    trade_repository = TursoTradeRepository(client)
    await candle_repository.ensure_schema()
    await trade_repository.ensure_schema()

    logger.info("Clearing existing trades table before full historical replay.")
    await client.execute("DELETE FROM trades")

    symbols = SymbolLoader().load(config.symbols_file)
    logger.info("Loaded %d symbols", len(symbols))

    engine = AlphaEngine(**_ENGINE_SETTINGS)
    for symbol in symbols:
        try:
            await _backtest_symbol(symbol, config, engine, candle_repository, trade_repository, logger)
        except Exception:
            logger.exception("Unexpected exception while backtesting %s", symbol)

    await client.close()
    logger.info("Historical backtest finished.")


async def _backtest_symbol(
    symbol: str,
    config: AppConfig,
    engine: AlphaEngine,
    candle_repository: TursoCandleRepository,
    trade_repository: TursoTradeRepository,
    logger: logging.Logger,
) -> None:
    """Replay one symbol's full stored history into real trade rows."""
    candles = await candle_repository.get_candles(symbol, config.candle_interval, limit=None)
    if len(candles) < _MINIMUM_CANDLES:
        logger.info("Skipping %s: only %d candles stored.", symbol, len(candles))
        return

    logger.info("Backtesting %s over %d candles.", symbol, len(candles))
    events = compute_historical_events(engine, candles)
    pine_trades = replay_pine_backtest(events)
    for pine_trade in pine_trades:
        await _store_trade(symbol, config, pine_trade, trade_repository)


async def _store_trade(
    symbol: str,
    config: AppConfig,
    pine_trade: PineTrade,
    trade_repository: TursoTradeRepository,
) -> None:
    """Store one Pine-scored trade: open it, then close it if it already exited.

    Safe because `replay_pine_backtest` guarantees at most one active
    position per symbol at a time -- `close_open_trade`'s "most recent open
    trade of this side" lookup is unambiguous here.
    """
    side = SignalSide.BUY if pine_trade.side == "buy" else SignalSide.SELL
    await trade_repository.open_trade(
        config.candle_interval,
        Trade(
            symbol=symbol,
            side=side,
            entry_timestamp=pine_trade.entry_timestamp,
            entry_price=Decimal(str(pine_trade.entry_price)),
            prediction_at_entry=pine_trade.prediction_at_entry,
            is_early_signal_flip=pine_trade.is_early_signal_flip,
        ),
    )
    if pine_trade.status == "closed":
        await trade_repository.close_open_trade(
            symbol,
            config.candle_interval,
            side,
            pine_trade.exit_timestamp,
            Decimal(str(pine_trade.exit_price)),
        )


if __name__ == "__main__":
    asyncio.run(main())
