"""Command-line entry point for the one-shot current-month derivatives backtest.

Triggered manually from the dashboard (see ``webapp.py``'s
``/api/trigger-backtest``), not on a schedule -- see
``application/derivatives_backtest.py`` for why it's current-month-only and
how it stays separate from the live shadow-tracking flow.
"""

import asyncio
import logging

from kiteconnect import KiteConnect

from trading_scanner.application.derivatives_backtest import run_current_month_backtest
from trading_scanner.config.settings import AppConfig, load_config
from trading_scanner.infrastructure.kite import KiteDerivativesChain
from trading_scanner.infrastructure.turso import (
    TursoFuturesTradeRepository,
    TursoKiteSessionRepository,
    TursoOptionsTradeRepository,
    TursoTradeRepository,
    create_turso_client,
)

logger = logging.getLogger(__name__)


async def _run(config: AppConfig) -> None:
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL is required to run the backtest.")
    if not config.kite_api_key:
        raise RuntimeError("TRADING_SCANNER_KITE_API_KEY is required to run the backtest.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        kite_session_repository = TursoKiteSessionRepository(client)
        trade_repository = TursoTradeRepository(client)
        options_trade_repository = TursoOptionsTradeRepository(client)
        futures_trade_repository = TursoFuturesTradeRepository(client)
        await kite_session_repository.ensure_schema()
        await options_trade_repository.ensure_schema()
        await futures_trade_repository.ensure_schema()

        token_row = await kite_session_repository.get_token()
        if token_row is None:
            raise RuntimeError("No Kite session -- log in via the dashboard first.")
        access_token, _obtained_at = token_row
        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        derivatives_chain = KiteDerivativesChain(kite)

        notes = await run_current_month_backtest(
            trade_repository,
            derivatives_chain,
            options_trade_repository,
            futures_trade_repository,
            config.candle_interval,
        )
        logger.info("Backtest wrote %d legs:", len(notes))
        for note in notes:
            logger.info("  %s", note)
    finally:
        await client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
