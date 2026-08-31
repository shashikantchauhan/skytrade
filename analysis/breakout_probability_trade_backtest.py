"""Backtest: does the Breakout Probability port's own directional bias call
(``application.breakout_probability.bullish_bias``) say anything useful as
an *additional* filter on top of the two gates real cash orders already
have to clear (``paper_trading.is_eligible``'s 55%-win-rate/>=5-trade bar,
and ``entry_quality_filter.passes_indicator_filter``)?

2026-08-31 correction: an earlier version of this script tested the
breakout-probability bias against every raw closed BUY trade in the
``trades`` table -- WRONG population. ``trades`` records every AlphaEngine
BUY signal unconditionally (see ``signal_pipeline.py``'s ``open_trade``
call, which fires before either real-money gate is even checked); the two
gates only decide whether a REAL order gets placed on top of that row, they
never filter what gets recorded. So the earlier run was silently comparing
against signals that would mostly never have reached real money in the
first place -- not what "does this help as an entry filter" needs. This
version reconstructs, causally and per trade (no look-ahead: only trades
whose ``entry_timestamp`` is strictly before the one being evaluated feed
its own eligibility check, and its own feature columns feed the quality
filter), the population that WOULD have cleared both real gates -- then
asks whether the breakout-probability call separates winners from losers
*within* that already-gated population.

No parameter is fit here (step_percent/num_levels are fixed at the
original indicator's own defaults, and the two real gates' thresholds are
imported from their own modules, never re-derived) -- every trade's own
snapshot is already causal on its own, so there's no train/test split
needed the way ``bad_trade_filters.py`` needs one for its percentile cuts.

Important caveat, stated plainly: the breakout-probability call is a
1-bar-ahead momentum tendency ("does the very next candle extend in this
direction"), while a real trade's pnl_percent is the outcome of a GTT
bracket that can take many bars (potentially days) to resolve. This tests
a real but looser hypothesis than the original indicator's own use case --
read the "kept %" column too, a filter that discards most of an
already-scarce gated population for a small win-rate move is a mixed bag.

Run:
    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/breakout_probability_trade_backtest.py
"""

import asyncio
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application import entry_quality_filter, paper_trading  # noqa: E402
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


def _eligibility_by_trade(buys: list[Trade]) -> dict[tuple[str, datetime], bool]:
    """Causal (no look-ahead) reconstruction of ``paper_trading.is_eligible``
    -- that function only ever reads the *current* DB state, so it can't be
    called against a moment in the past. Walks each symbol's own BUY trades
    in entry order, recomputing the win-rate/min-trades bar using only
    trades strictly before the one being evaluated -- same rule
    ``bad_trade_filters.py``'s own ``eligible_at`` applies, just grouped by
    symbol first so this is O(trades) instead of O(trades^2)."""
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


async def bias_by_trade(
    trades: list[Trade], candle_repository: TursoCandleRepository, interval: str
) -> dict[tuple[str, datetime], bool | None]:
    """One causal ``bullish_bias()`` call per trade, keyed by (symbol,
    entry_timestamp). Walks each symbol's candle history exactly once
    (O(candles), not O(trades * candles)), snapshotting the bias the
    instant before folding in the bar a trade entered on."""
    by_symbol: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t)

    result: dict[tuple[str, datetime], bool | None] = {}
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


BiasByTrade = dict[tuple[str, datetime], bool | None]


def _report(label: str, population: list[Trade], bias: BiasByTrade) -> None:
    bullish = [t for t in population if bias.get((t.symbol, t.entry_timestamp)) is True]
    bearish = [t for t in population if bias.get((t.symbol, t.entry_timestamp)) is False]
    unknown = [t for t in population if bias.get((t.symbol, t.entry_timestamp)) is None]

    baseline = summarize([float(t.pnl_percent) for t in population])
    print(f"\n=== {label}: n={baseline.n} ===")
    print(fmt(baseline, "baseline (no breakout filter)", baseline.n))
    print(
        f"bullish call: {len(bullish)}   bearish call: {len(bearish)}   "
        f"not enough history yet: {len(unknown)}"
    )
    bullish_stats = summarize([float(t.pnl_percent) for t in bullish])
    bearish_stats = summarize([float(t.pnl_percent) for t in bearish])
    unknown_stats = summarize([float(t.pnl_percent) for t in unknown])
    print(fmt(bullish_stats, "  bullish call only", baseline.n))
    print(fmt(bearish_stats, "  bearish call only", baseline.n))
    print(fmt(unknown_stats, "  no history yet", baseline.n))


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
            if t.side == SignalSide.BUY
            and t.status == "closed"
            and t.pnl_percent is not None
            # pre-migration rows lack feature columns -- can't quality-gate those
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

        bias = await bias_by_trade(buys, candle_repository, config.candle_interval)
    finally:
        await client.close()

    _report(
        "trades that WOULD have cleared both real gates (the population that matters)",
        real_gate_population,
        bias,
    )
    _report(
        "for reference only -- every raw AlphaEngine BUY signal, ungated "
        "(NOT what real money would have taken)",
        buys,
        bias,
    )


if __name__ == "__main__":
    asyncio.run(main())
