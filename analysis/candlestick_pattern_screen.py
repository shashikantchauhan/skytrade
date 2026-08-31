"""Walk-forward screen: does ANY of the candlestick patterns in
``application.candlestick_patterns`` being present right before entry
correlate with better real BUY trade outcomes?

Same population and methodology as the other trade backtests this session:
every closed BUY trade that would have cleared BOTH real gates (55%/5-trade
eligibility + entry_quality_filter, reconstructed causally per trade --
only trades strictly before the one being evaluated feed its own checks),
checked against each pattern using only the candles strictly before that
trade's own entry candle (no look-ahead -- the pattern must have already
fully formed by the time the entry decision is made).

Tests every pattern in one pass (one candle walk per symbol, not one per
pattern) so adding a new pattern to the module is free -- just add its
name to ``_PATTERNS`` below.

Run:
    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/candlestick_pattern_screen.py
"""

import asyncio
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application import (  # noqa: E402
    candlestick_patterns,
    entry_quality_filter,
    paper_trading,
)
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.domain.models import SignalSide, Trade  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

_PATTERNS = {
    "bullish_engulfing": candlestick_patterns.bullish_engulfing,
    "bullish_harami": candlestick_patterns.bullish_harami,
    "piercing_line": candlestick_patterns.piercing_line,
    "three_white_soldiers": candlestick_patterns.three_white_soldiers,
    "morning_star": candlestick_patterns.morning_star,
    "three_outside_up": candlestick_patterns.three_outside_up,
    "three_inside_up": candlestick_patterns.three_inside_up,
}


@dataclass
class Stats:
    n: int
    win_rate: float | None
    expectancy: float | None
    total_pnl: float | None


def summarize(pnls: list[float]) -> Stats:
    if not pnls:
        return Stats(0, None, None, None)
    wins = [p for p in pnls if p > 0]
    return Stats(
        n=len(pnls),
        win_rate=100 * len(wins) / len(pnls),
        expectancy=statistics.mean(pnls),
        total_pnl=sum(pnls),
    )


def fmt(stats: Stats, label: str) -> str:
    if stats.n == 0:
        return f"{label:<24} n=0"
    return (
        f"{label:<24} n={stats.n:>5}  win_rate={stats.win_rate:5.1f}%  "
        f"expectancy={stats.expectancy:+.3f}%  sum_pnl={stats.total_pnl:+8.1f}%"
    )


def _eligibility_by_trade(buys: list[Trade]) -> dict[tuple[str, datetime], bool]:
    """Causal (no look-ahead) reconstruction of paper_trading.is_eligible --
    see the other trade-backtest scripts this session for why this can't
    just call that function directly."""
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


async def patterns_by_trade(
    trades: list[Trade], candle_repository: TursoCandleRepository, interval: str
) -> dict[tuple[str, datetime], dict[str, bool]]:
    """Every pattern's match/no-match for every trade, one pass per
    symbol's candle history -- keyed by (symbol, entry_timestamp) ->
    {pattern_name: matched}. Uses only candles strictly before the entry
    candle for each check."""
    by_symbol: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t)

    result: dict[tuple[str, datetime], dict[str, bool]] = {}
    for symbol, symbol_trades in by_symbol.items():
        candles = sorted(
            await candle_repository.get_candles(symbol, interval), key=lambda c: c.timestamp
        )
        entry_lookup = {t.entry_timestamp: t for t in symbol_trades}
        for index, candle in enumerate(candles):
            trade = entry_lookup.get(candle.timestamp)
            if trade is not None:
                prior = candles[:index]
                result[(symbol, trade.entry_timestamp)] = {
                    name: detector(prior) for name, detector in _PATTERNS.items()
                }
    return result


def _report(
    label: str, population: list[Trade], patterns: dict[tuple[str, datetime], dict[str, bool]]
) -> None:
    baseline = summarize([float(t.pnl_percent) for t in population])
    print(f"\n=== {label}: n={baseline.n} ===")
    print(fmt(baseline, "baseline"))
    for name in _PATTERNS:
        matched = [
            t for t in population
            if patterns.get((t.symbol, t.entry_timestamp), {}).get(name)
        ]
        print(fmt(summarize([float(t.pnl_percent) for t in matched]), f"  {name}"))


async def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config()
    url = f"file:{db_path}" if db_path else config.turso_database_url
    client = create_turso_client(url, None if db_path else config.turso_auth_token)
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
        print(f"Total closed BUY trades with feature columns: {len(buys)}")

        eligibility = _eligibility_by_trade(buys)
        real_gate_population = [
            t
            for t in buys
            if eligibility[(t.symbol, t.entry_timestamp)]
            and entry_quality_filter.passes_indicator_filter(
                t.volatility_margin_at_entry, t.regime_normalized_at_entry
            )
        ]
        print(
            f"Would have cleared BOTH real gates (55%/5-trade eligibility + "
            f"entry_quality_filter): {len(real_gate_population)}"
        )

        patterns = await patterns_by_trade(buys, candle_repository, config.candle_interval)
    finally:
        await client.close()

    _report(
        "trades that WOULD have cleared both real gates (the population that matters)",
        real_gate_population,
        patterns,
    )
    _report(
        "for reference only -- every raw AlphaEngine BUY signal, ungated",
        buys,
        patterns,
    )


if __name__ == "__main__":
    asyncio.run(main())
