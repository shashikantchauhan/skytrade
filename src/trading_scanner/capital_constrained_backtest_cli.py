"""One-time CLI: compare the account's actual capital/slot config against
alternatives, with and without ranking, over the full stored trade history.

Run after ``python -m trading_scanner.backtest`` has populated the ``trades``
table (needs the ADX/regime/volatility columns application/backtest.py now
logs, for the ranked runs to use more than prediction_at_entry alone).

Answers "how many signals is the current capital config actually missing,
and does ranking recover any of that" -- application/backtest.py's replay
only answers what the strategy would do with unlimited capital.
"""

import asyncio
import logging
from decimal import Decimal

from trading_scanner.application.capital_constrained_backtest import (
    SimulationConfig,
    SimulationResult,
    simulate,
)
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.db import TursoTradeRepository, create_turso_client


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()

    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    trade_repository = TursoTradeRepository(client)
    await trade_repository.ensure_schema()

    logger.info("Loading full BUY-side trade history...")
    trades = await trade_repository.get_trades(None, config.candle_interval)
    logger.info("Loaded %d trades total.", len(trades))
    await client.close()

    scenarios = [
        ("Current (10 slots, Rs75k floor, ranked)", SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=10,
            min_position_size=Decimal("75000"), use_ranking=True,
        )),
        ("Current (10 slots, Rs75k floor, unranked)", SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=10,
            min_position_size=Decimal("75000"), use_ranking=False,
        )),
        ("Repo default (32 slots, Rs25k floor, ranked)", SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=32,
            min_position_size=Decimal("25000"), use_ranking=True,
        )),
        ("Repo default (32 slots, Rs25k floor, unranked)", SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=32,
            min_position_size=Decimal("25000"), use_ranking=False,
        )),
    ]

    print(f"\n{'Scenario':<45} {'Taken':>7} {'Skip(elig)':>11} {'Skip(cap)':>10} "
          f"{'WinRate':>8} {'FinalEquity':>13}")
    print("-" * 100)
    results: list[tuple[str, SimulationResult]] = []
    for name, scenario_config in scenarios:
        result = simulate(trades, scenario_config)
        results.append((name, result))
        win_rate = "n/a" if result.win_rate is None else f"{result.win_rate:.1f}%"
        print(
            f"{name:<45} {result.trades_taken:>7} {result.trades_skipped_ineligible:>11} "
            f"{result.trades_skipped_no_capital:>10} {win_rate:>8} "
            f"Rs{result.final_equity:>10,.0f}"
        )
    print()


if __name__ == "__main__":
    asyncio.run(main())
