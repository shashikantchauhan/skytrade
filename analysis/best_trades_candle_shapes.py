"""What did the entry candle and the candle right before it actually look
like for the best-performing real trades? Data-driven complement to
``candlestick_pattern_screen.py``'s textbook-pattern screen -- instead of
testing predefined pattern shapes, this prints the raw candles (plus a few
derived numbers: color, body size as % of range, gap %) for the top N
trades by pnl_percent, so a real shape can be spotted by eye if the
textbook patterns don't capture it.

Same population as the other trade backtests: trades that would have
cleared both real gates (55%/5-trade eligibility + entry_quality_filter,
reconstructed causally). "Previous candle" is the one immediately before
the entry candle in that symbol's own candle history -- not a look-ahead
concern here since this is pure inspection, not a live decision.

Run:
    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/best_trades_candle_shapes.py [--top N] [--worst]
"""

import argparse
import asyncio
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application import entry_quality_filter, paper_trading  # noqa: E402
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.domain.models import Candle, SignalSide, Trade  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)


def _eligibility_by_trade(buys: list[Trade]) -> dict[tuple[str, datetime], bool]:
    by_symbol: dict[str, list[Trade]] = defaultdict(list)
    for t in buys:
        by_symbol[t.symbol].append(t)

    result: dict[tuple[str, datetime], bool] = {}
    for symbol, symbol_trades in by_symbol.items():
        symbol_trades.sort(key=lambda t: t.entry_timestamp)
        prior_pnls: list[float] = []
        for t in symbol_trades:
            if len(prior_pnls) >= paper_trading.MIN_CLOSED_TRADES:
                win_rate = 100 * len([p for p in prior_pnls if p > 0]) / len(prior_pnls)
                result[(symbol, t.entry_timestamp)] = win_rate >= float(paper_trading.MIN_WIN_RATE)
            else:
                result[(symbol, t.entry_timestamp)] = False
            prior_pnls.append(float(t.pnl_percent))
    return result


def _color(candle: Candle) -> str:
    if candle.close > candle.open:
        return "green"
    if candle.close < candle.open:
        return "red"
    return "doji"


def _describe(candle: Candle, label: str) -> str:
    body = abs(candle.close - candle.open)
    full_range = candle.high - candle.low
    body_pct = float(body / full_range * 100) if full_range else 0.0
    color = _color(candle)
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    return (
        f"    {label}: {candle.timestamp}  {color:<4}  "
        f"O={candle.open} H={candle.high} L={candle.low} C={candle.close}  "
        f"body={body_pct:4.0f}%ofrange  upper_wick={upper_wick}  lower_wick={lower_wick}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", nargs="?", default=None)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--worst", action="store_true", help="show worst trades instead of best")
    args = parser.parse_args()

    config = load_config()
    url = f"file:{args.db_path}" if args.db_path else config.turso_database_url
    client = create_turso_client(url, None if args.db_path else config.turso_auth_token)
    try:
        trade_repository = TursoTradeRepository(client)
        candle_repository = TursoCandleRepository(client)
        await trade_repository.ensure_schema()
        all_trades = await trade_repository.get_trades(None, config.candle_interval)

        buys = [
            t
            for t in all_trades
            if t.side == SignalSide.BUY
            and t.status == "closed"
            and t.pnl_percent is not None
            and t.volatility_margin_at_entry is not None
            and t.regime_normalized_at_entry is not None
        ]
        eligibility = _eligibility_by_trade(buys)
        real_gate_population = [
            t
            for t in buys
            if eligibility[(t.symbol, t.entry_timestamp)]
            and entry_quality_filter.passes_indicator_filter(
                t.volatility_margin_at_entry, t.regime_normalized_at_entry
            )
        ]
        ranked = sorted(
            real_gate_population, key=lambda t: t.pnl_percent, reverse=not args.worst
        )
        selected = ranked[: args.top]

        label = "WORST" if args.worst else "BEST"
        print(
            f"Top {len(selected)} {label} real-gated trades by pnl_percent "
            f"(out of {len(real_gate_population)} total)\n"
        )

        prev_colors: list[str] = []
        entry_colors: list[str] = []
        for trade in selected:
            candles = sorted(
                await candle_repository.get_candles(trade.symbol, config.candle_interval),
                key=lambda c: c.timestamp,
            )
            index = next(
                (i for i, c in enumerate(candles) if c.timestamp == trade.entry_timestamp), None
            )
            print(f"{trade.symbol}  entry={trade.entry_timestamp}  pnl={trade.pnl_percent:+.2f}%")
            if index is None:
                print("    (entry candle not found in stored history)")
                continue
            entry_candle = candles[index]
            print(_describe(entry_candle, "entry (current)  "))
            entry_colors.append(_color(entry_candle))
            if index > 0:
                prev_candle = candles[index - 1]
                print(_describe(prev_candle, "previous         "))
                prev_colors.append(_color(prev_candle))
            print()

        print("=== Summary across the sample above ===")
        for name, colors in (("previous candle", prev_colors), ("entry candle", entry_colors)):
            if not colors:
                continue
            counts = {c: colors.count(c) for c in ("green", "red", "doji")}
            total = len(colors)
            print(
                f"{name:<16}: "
                + "  ".join(f"{c}={n} ({100 * n / total:.0f}%)" for c, n in counts.items())
            )
        pnls = [float(t.pnl_percent) for t in selected]
        if pnls:
            print(f"\npnl range in this sample: {min(pnls):+.2f}% to {max(pnls):+.2f}%, "
                  f"mean {statistics.mean(pnls):+.2f}%")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
