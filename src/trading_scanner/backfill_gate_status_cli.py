"""One-off: populate gate_status for every tracked symbol from today's
already-computed state, without waiting for tomorrow's first live cycle.

2026-09-02: added the same day GateStatusSnapshot/the Gates dashboard tab
shipped (docs/decisions/008-gate-status-snapshot.md) -- every symbol was
already evaluated for real today, before this feature existed to record
it, so the table would otherwise sit empty until the next candle closes.
Read-only against the engine: calls evaluate_latest_bar with each symbol's
already-persisted engine_state (signal/queue_state/exit_state) and full
candle history to get the same FastPredictResult a live cycle would have
produced, but never calls engine_state_repository.set_state -- this must
not advance/mutate the live incremental state, only read what it already
implies. Safe to re-run any time (e.g. after resync_daily_candles_cli.py)
as a manual refresh -- it always overwrites with the current read, never
appends.
"""

import asyncio
import logging

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.fast_predict import ExitState, QueueState, evaluate_latest_bar
from trading_scanner.application.pipeline.evaluation import (
    _FULL_HISTORY,
    _MINIMUM_CANDLES,
    _candles_to_dataframe,
)
from trading_scanner.application.pipeline.orchestrator import _ENGINE_SETTINGS, _record_gate_status
from trading_scanner.application.symbols import SymbolLoader
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoEngineStateRepository,
    TursoGateStatusRepository,
    TursoTradeRepository,
    create_turso_client,
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        candle_repository = TursoCandleRepository(client)
        engine_state_repository = TursoEngineStateRepository(client)
        trade_repository = TursoTradeRepository(client)
        gate_status_repository = TursoGateStatusRepository(client)
        await candle_repository.ensure_schema()
        await engine_state_repository.ensure_schema()
        await trade_repository.ensure_schema()
        await gate_status_repository.ensure_schema()

        engine = AlphaEngine(**_ENGINE_SETTINGS)
        symbols = SymbolLoader().load(config.symbols_file)
        logger.info("Backfilling gate status for %d symbols.", len(symbols))

        written = 0
        skipped = 0
        for symbol in symbols:
            try:
                accumulated = await candle_repository.get_candles(
                    symbol, config.candle_interval, limit=_FULL_HISTORY
                )
                if len(accumulated) < _MINIMUM_CANDLES:
                    skipped += 1
                    continue
                engine_state = await engine_state_repository.get_state(
                    symbol, config.candle_interval
                )
                if engine_state.queue_json is None:
                    # Never evaluated live yet -- nothing to read back, and
                    # bootstrapping here would be the expensive full replay
                    # this script exists to avoid. Tomorrow's live cycle
                    # bootstraps it normally.
                    skipped += 1
                    continue
                history = _candles_to_dataframe(accumulated)
                result = evaluate_latest_bar(
                    engine, history, engine_state.signal,
                    QueueState.from_json(engine_state.queue_json),
                    ExitState.from_json(engine_state.exit_state_json),
                )
                await _record_gate_status(
                    symbol, config, result, accumulated[-1], trade_repository,
                    gate_status_repository,
                )
                written += 1
            except Exception:
                skipped += 1
                logger.exception("Failed to backfill gate status for %s -- skipping.", symbol)

        logger.info("Backfill complete: %d written, %d skipped.", written, skipped)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
