"""One-time CLI: does a hard stop-loss actually help, or hurt?

Compares the strategy's real BUY/SELL trade history as-is against the same
trades replayed with a stop-loss overlay (application/stop_loss_replay.py)
at a few candidate thresholds, using each trade's real intra-trade candle
path -- not a guess.

Run after python -m trading_scanner.fix_corrupted_candles_cli (the
corrupted-candle fix) so these numbers aren't distorted by the same fake
price spikes that corrupted the earlier backtest -- otherwise a stop-loss
would look artificially essential (it would "help" mainly by capping
losses that were never real).
"""

import asyncio
import logging
from decimal import Decimal

from trading_scanner.application.stop_loss_replay import apply_stop_loss, summarize
from trading_scanner.config.settings import load_config
from trading_scanner.domain.models import SignalSide
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

_CANDIDATE_THRESHOLDS = [Decimal("2"), Decimal("3"), Decimal("5"), Decimal("8")]


def _print_row(label: str, stats: dict) -> None:
    if stats["n"] == 0:
        print(f"{label:<18} n=0")
        return
    print(
        f"{label:<18} n={stats['n']:>5}  win_rate={stats['win_rate']:>6.2f}%  "
        f"avg_win={stats['avg_win']:>6.2f}%  avg_loss={stats['avg_loss']:>7.2f}%  "
        f"expectancy={stats['expectancy']:>6.3f}%"
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    trade_repository = TursoTradeRepository(client)
    candle_repository = TursoCandleRepository(client)
    trades = await trade_repository.get_trades(None, config.candle_interval)
    logger.info("Loaded %d trades.", len(trades))

    for side in (SignalSide.BUY, SignalSide.SELL):
        print(f"\n=== {side.value.upper()} ===")
        _print_row("no stop-loss", summarize(trades, side))
        for threshold in _CANDIDATE_THRESHOLDS:
            logger.info("Replaying %s side with a %s%% stop-loss...", side.value, threshold)
            adjusted = await apply_stop_loss(
                trades, candle_repository, config.candle_interval, threshold
            )
            _print_row(f"stop @ {threshold}%", summarize(adjusted, side))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
