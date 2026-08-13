"""One-time CLI: does sector-index confirmation (the stock's mapped sector
index also entering the same side at the same time) actually predict a
better outcome, across the full trade history -- not two examples.

Run after python -m trading_scanner.fix_corrupted_candles_cli (the
corrupted-candle fix), same reasoning as the other post-cleanup analysis
tools: the corrupted candles hit both stocks and their sector indices, so
running this against dirty data would produce an untrustworthy comparison.
"""

import asyncio
import logging

from trading_scanner.application.sector_confirmation import compare_confirmed_vs_unconfirmed
from trading_scanner.config.settings import load_config
from trading_scanner.domain.models import SignalSide
from trading_scanner.infrastructure.db import TursoTradeRepository, create_turso_client


def _print_row(label: str, stats) -> None:
    if stats.n == 0:
        print(f"  {label:<28} n=0")
        return
    print(
        f"  {label:<28} n={stats.n:>5}  win_rate={stats.win_rate:>6.2f}%  "
        f"avg_win={stats.avg_win:>6.2f}%  avg_loss={stats.avg_loss:>7.2f}%  "
        f"expectancy={stats.expectancy:>7.3f}%"
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    trade_repository = TursoTradeRepository(client)
    trades = await trade_repository.get_trades(None, config.candle_interval)
    await client.close()
    logger.info("Loaded %d trades.", len(trades))

    for side in (SignalSide.BUY, SignalSide.SELL):
        print(f"\n=== {side.value.upper()} ===")
        results = compare_confirmed_vs_unconfirmed(trades, side)
        for label, stats in results.items():
            _print_row(label, stats)


if __name__ == "__main__":
    asyncio.run(main())
