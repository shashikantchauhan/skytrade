"""Backtest: does the Breakout Probability port's own directional bias call
(``application.breakout_probability.bullish_bias``) say anything useful
about which of our REAL, already-gated BUY trades go on to win?

For every real closed BUY trade already in the ``trades`` table (the same
population ``bad_trade_filters.py`` screens -- signals that already passed
today's live gates: ``entry_quality_filter`` + ``paper_trading.is_eligible``),
this walks that symbol's own candle history bar by bar and snapshots the
breakout-probability bias *causally* -- using only candles strictly before
the trade's own entry_timestamp, exactly as a live system would have seen
it in real time, no look-ahead. No parameter is fit here (step_percent and
num_levels are fixed at the original indicator's own defaults), so unlike
``bad_trade_filters.py`` there is no train/test split to worry about --
every trade's own snapshot is already causal on its own.

Important caveat, stated plainly rather than glossed over: the breakout
probability call is a 1-bar-ahead momentum tendency ("does the very next
candle extend in this direction"), while a real trade's pnl_percent is the
outcome of a GTT bracket that can take many bars (potentially days) to
resolve. This tests a real but looser hypothesis than the original
indicator's own use case: does the very-short-horizon call still correlate
with the eventual, much-longer-horizon trade outcome? Read the "kept %"
column too -- a filter that discards most of the trade population to gain
a couple of win-rate points is a mixed bag.

Run:
    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/breakout_probability_trade_backtest.py
"""

import asyncio
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application.breakout_probability import (  # noqa: E402
    BreakoutStats,
    LevelCounts,
    advance_breakout_stats,
    bullish_bias,
)
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.domain.models import SignalSide, Trade  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

STEP_PERCENT = 1.0  # matches the original indicator's own default
NUM_LEVELS = 1  # only level 0 (the loosest threshold) feeds bullish_bias


@dataclass
class Stats:
    n: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    expectancy: float | None
    total_pnl: float | None


def summarize(pnls: list[float]) -> Stats:
    if not pnls:
        return Stats(0, None, None, None, None, None)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return Stats(
        n=len(pnls),
        win_rate=100 * len(wins) / len(pnls),
        avg_win=statistics.mean(wins) if wins else None,
        avg_loss=statistics.mean(losses) if losses else None,
        expectancy=statistics.mean(pnls),
        total_pnl=sum(pnls),
    )


def fmt(stats: Stats, label: str, baseline_n: int) -> str:
    if stats.n == 0:
        return f"{label:<32} n=0 (nothing in this bucket)"
    kept_pct = 100 * stats.n / baseline_n if baseline_n else 0
    return (
        f"{label:<32} n={stats.n:>5} ({kept_pct:5.1f}% of baseline)  "
        f"win_rate={stats.win_rate:5.1f}%  avg_win={stats.avg_win:6.2f}%  "
        f"avg_loss={stats.avg_loss:6.2f}%  expectancy={stats.expectancy:+.3f}%  "
        f"sum_pnl={stats.total_pnl:+8.1f}%"
    )


async def bias_by_trade(
    trades: list[Trade], candle_repository: TursoCandleRepository, interval: str
) -> dict[tuple[str, object], bool | None]:
    """One causal bullish_bias() call per trade, keyed by (symbol,
    entry_timestamp). Walks each symbol's candle history exactly once
    (O(candles), not O(trades * candles)), snapshotting the bias the
    instant before folding in the bar a trade entered on."""
    by_symbol: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t)

    result: dict[tuple[str, object], bool | None] = {}
    for symbol, symbol_trades in by_symbol.items():
        candles = sorted(
            await candle_repository.get_candles(symbol, interval), key=lambda c: c.timestamp
        )
        entry_lookup = {t.entry_timestamp: t for t in symbol_trades}
        stats = BreakoutStats(levels=[LevelCounts() for _ in range(NUM_LEVELS)])
        for prev, curr in zip(candles, candles[1:], strict=False):
            trade = entry_lookup.get(curr.timestamp)
            if trade is not None:
                # Snapshot BEFORE folding curr's own outcome in -- this is
                # exactly what a live system would have known the instant
                # curr's bar opened, having just seen prev close.
                result[(symbol, trade.entry_timestamp)] = bullish_bias(stats, prev)
            advance_breakout_stats(stats, prev, curr, STEP_PERCENT)
    return result


async def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config()
    url = f"file:{db_path}" if db_path else config.turso_database_url
    client = create_turso_client(url, None if db_path else config.turso_auth_token)
    try:
        trade_repository = TursoTradeRepository(client)
        candle_repository = TursoCandleRepository(client)
        await trade_repository.ensure_schema()  # local.db may predate a feature-column migration
        all_trades = await trade_repository.get_trades(None, config.candle_interval)

        buys = [
            t
            for t in all_trades
            if t.side == SignalSide.BUY and t.status == "closed" and t.pnl_percent is not None
        ]
        print(f"Total closed BUY trades: {len(buys)}")

        bias = await bias_by_trade(buys, candle_repository, config.candle_interval)
    finally:
        await client.close()

    bullish = [t for t in buys if bias.get((t.symbol, t.entry_timestamp)) is True]
    bearish = [t for t in buys if bias.get((t.symbol, t.entry_timestamp)) is False]
    unknown = [t for t in buys if bias.get((t.symbol, t.entry_timestamp)) is None]
    print(
        f"bullish call: {len(bullish)}   bearish call: {len(bearish)}   "
        f"not enough history yet: {len(unknown)}"
    )
    print()

    baseline = summarize([float(t.pnl_percent) for t in buys])
    print("=== baseline (every real BUY trade, no filter) ===")
    print(fmt(baseline, "baseline", baseline.n))
    print()

    print("=== filtered by the breakout-probability bias call at entry ===")
    print(fmt(summarize([float(t.pnl_percent) for t in bullish]), "bullish call only", baseline.n))
    print(fmt(summarize([float(t.pnl_percent) for t in bearish]), "bearish call only", baseline.n))
    print(fmt(summarize([float(t.pnl_percent) for t in unknown]), "no history yet", baseline.n))


if __name__ == "__main__":
    asyncio.run(main())
