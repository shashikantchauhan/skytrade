"""One-time CLI: remove corrupted flat-OHLC candles and rebuild trades for
just the affected symbols.

Root cause found by hand during Phase 1 analysis: a small number of candles
(open == high == low == close, ~0.09% of the dataset) are provider-side
garbage -- confirmed by cross-symbol duplicates (e.g. SAIL.NS and RVNL.NS
sharing byte-identical OHLC+volume at the same timestamp, which is
physically impossible for two unrelated real stocks). These bad bars
occasionally produced ~99% fake losses in the backtest replay when a
position's entry or exit landed on one. Ordinary zero-volume bars (the
common, benign case -- Yahoo Finance's day-open bar frequently omits volume
but reports a real price range, and AlphaEngine never uses volume in its
signal computation anyway) are NOT touched by this script; only bars with
zero price range are removed.

Scoped, not a full re-backtest: only the symbols that actually have a bad
candle get their candles cleaned and trades rebuilt. Every other symbol's
trades are left untouched.
"""

import asyncio
import logging

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.backtest import compute_historical_events, replay_pine_backtest
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.turso import (
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}
_MINIMUM_CANDLES = 200


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    candle_repository = TursoCandleRepository(client)
    trade_repository = TursoTradeRepository(client)
    await candle_repository.ensure_schema()
    await trade_repository.ensure_schema()

    result = await client.execute(
        """
        SELECT DISTINCT symbol FROM candles
        WHERE interval = ? AND open = high AND high = low AND low = close
        ORDER BY symbol
        """,
        [config.candle_interval],
    )
    affected_symbols = [row[0] for row in result.rows]
    logger.info("Found %d symbols with corrupted flat-OHLC candles.", len(affected_symbols))

    deleted = await client.execute(
        """
        DELETE FROM candles
        WHERE interval = ? AND open = high AND high = low AND low = close
        """,
        [config.candle_interval],
    )
    logger.info("Deleted corrupted candles (rows_affected=%s).", deleted.rows_affected)

    engine = AlphaEngine(**_ENGINE_SETTINGS)
    for symbol in affected_symbols:
        try:
            await client.execute(
                "DELETE FROM trades WHERE symbol = ? AND interval = ?",
                [symbol, config.candle_interval],
            )
            candles = await candle_repository.get_candles(
                symbol, config.candle_interval, limit=None
            )
            if len(candles) < _MINIMUM_CANDLES:
                logger.info(
                    "Skipping %s: only %d candles remain after cleanup.", symbol, len(candles)
                )
                continue
            events = compute_historical_events(engine, candles)
            pine_trades = replay_pine_backtest(events)
            from trading_scanner.backtest import _store_trade  # reuse the exact storage logic

            for pine_trade in pine_trades:
                await _store_trade(symbol, config, pine_trade, trade_repository)
            logger.info(
                "Rebuilt %s: %d trades from %d clean candles.",
                symbol, len(pine_trades), len(candles),
            )
        except Exception:
            logger.exception("Unexpected exception while rebuilding %s", symbol)

    await client.close()
    logger.info("Corrupted-candle cleanup finished.")


if __name__ == "__main__":
    asyncio.run(main())
