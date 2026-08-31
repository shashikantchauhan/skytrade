"""CLI for the Breakout Probability port -- see
``trading_scanner.application.breakout_probability`` for the actual logic
and what this measures/how it differs from the original Pine indicator.
Personal-use reference tool only: read-only, prints a table, never wired
into the live pipeline or any real order.

Run (same DB-selection convention as this directory's other scripts):

    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/breakout_probability_report.py RELIANCE.NS
    ... --step 1.0 --levels 5 --interval 1h --show-zero
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application.breakout_probability import (  # noqa: E402
    compute_breakout_stats,
    project_next_bar_levels,
)
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.domain.models import Candle  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoCandleRepository,
    create_turso_client,
)


def _fmt_pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:5.1f}%"


def _print_table(symbol: str, candles: list[Candle], step_percent: float, num_levels: int) -> None:
    stats = compute_breakout_stats(candles, step_percent=step_percent, num_levels=num_levels)
    probabilities = stats.level_probabilities()

    print(
        f"\n=== {symbol} -- {len(candles)} candles, step={step_percent}%, levels={num_levels} ==="
    )
    print(f"green candles: {stats.green_total}   red candles: {stats.red_total}")
    print(
        f"\n{'level':>5}  {'green->high':>11}  {'green->low':>11}  "
        f"{'red->high':>11}  {'red->low':>11}"
    )
    for i, probs in enumerate(probabilities):
        print(
            f"{i:>5}  {_fmt_pct(probs.green_high):>11}  {_fmt_pct(probs.green_low):>11}  "
            f"{_fmt_pct(probs.red_high):>11}  {_fmt_pct(probs.red_low):>11}"
        )

    print(
        f"\ndirectional pick (level 0 only, mirrors the original's own "
        f"backtest panel): {stats.directional_wins}W / {stats.directional_losses}L"
        f" ({_fmt_pct(stats.directional_win_rate)})"
    )

    latest = candles[-1]
    bias = "green" if latest.close > latest.open else "red"
    print(f"\n--- next-bar projection from the latest candle ({latest.timestamp}, {bias}) ---")
    for level in project_next_bar_levels(
        latest, stats, step_percent=step_percent, num_levels=num_levels
    ):
        print(
            f"level {level.index}: up to {level.high_target:.2f} "
            f"({_fmt_pct(level.high_probability)})  |  "
            f"down to {level.low_target:.2f} ({_fmt_pct(level.low_probability)})"
        )


async def _run(args: argparse.Namespace) -> None:
    config = load_config()
    url = f"file:{args.db}" if args.db else config.turso_database_url
    client = create_turso_client(url, None if args.db else config.turso_auth_token)
    try:
        repository = TursoCandleRepository(client)
        for symbol in args.symbols:
            candles = await repository.get_candles(symbol, args.interval)
            if len(candles) < 2:
                print(f"\n=== {symbol} -- not enough candles ({len(candles)}) ===")
                continue
            _print_table(symbol, list(candles), args.step, args.levels)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+", help="Yahoo-style symbol(s), e.g. RELIANCE.NS")
    parser.add_argument("--step", type=float, default=1.0, help="percentage step per level")
    parser.add_argument(
        "--levels", type=int, default=5, help="number of levels (max 5, like the original)"
    )
    parser.add_argument(
        "--interval", default=None, help="defaults to the app's configured candle interval"
    )
    parser.add_argument("--db", default=None, help="local sqlite file path, e.g. local.db")
    args = parser.parse_args()

    if args.interval is None:
        args.interval = load_config().candle_interval
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
