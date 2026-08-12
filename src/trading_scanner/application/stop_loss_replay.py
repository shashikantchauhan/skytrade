"""Replay stored trades against their real intra-trade candle path to test
whether a hard stop-loss would help or hurt -- see NOTES.md's exit-strategy
roadmap.

The strategy has no price-based risk control today: ``use_dynamic_exits``
(see ``application/signal_pipeline.py``'s ``_ENGINE_SETTINGS`` docstring)
only changes *when* the opposing technical signal fires, never caps how far
price can move against an open position first. This module answers, for a
given stop-loss percentage, what would have happened if a position were
force-closed the moment it crossed that threshold, instead of waiting for
the strategy's own exit signal -- using the real candle-by-candle price
path between each trade's entry and exit, not just the entry/exit snapshot
the ``trades`` table stores (which is all ``application/backtest.py``'s
replay keeps).

Deliberately does not touch ``alpha_engine.py`` or ``application/
backtest.py``'s Pine-matching replay -- this is a what-if overlay computed
downstream of that data, exactly like ``application/
capital_constrained_backtest.py``. Not wired into live trading: this
answers "would a stop-loss have helped historically", a prerequisite
question before deciding whether ``paper_trading.py``'s position management
should actually gain one.
"""

from dataclasses import replace
from decimal import Decimal

from trading_scanner.domain.models import SignalSide, Trade
from trading_scanner.domain.ports import CandleRepository


async def apply_stop_loss(
    trades: list[Trade],
    candle_repository: CandleRepository,
    interval: str,
    stop_loss_pct: Decimal,
) -> list[Trade]:
    """Return a copy of ``trades`` where any trade whose real candle path
    breached ``stop_loss_pct`` against the entry price exits early, at the
    first breaching candle, instead of its original (strategy-signal) exit.

    Only closed trades with a known exit are affected -- still-open trades
    pass through unchanged (their future path isn't decided yet). Each
    symbol's candle history is fetched once and reused across all of that
    symbol's trades, not re-fetched per trade.
    """
    candles_by_symbol: dict[str, list] = {}
    adjusted: list[Trade] = []

    for trade in trades:
        if trade.status != "closed" or trade.exit_timestamp is None:
            adjusted.append(trade)
            continue

        if trade.symbol not in candles_by_symbol:
            candles_by_symbol[trade.symbol] = list(
                await candle_repository.get_candles(trade.symbol, interval, limit=None)
            )
        candles = candles_by_symbol[trade.symbol]
        path = [c for c in candles if trade.entry_timestamp <= c.timestamp <= trade.exit_timestamp]

        stop_price = (
            trade.entry_price * (1 - stop_loss_pct / 100)
            if trade.side == SignalSide.BUY
            else trade.entry_price * (1 + stop_loss_pct / 100)
        )
        breach = _first_breach(path, trade.side, stop_price)
        if breach is None:
            adjusted.append(trade)
            continue

        pnl_percent = (
            (stop_price - trade.entry_price) / trade.entry_price * 100
            if trade.side == SignalSide.BUY
            else (trade.entry_price - stop_price) / trade.entry_price * 100
        )
        adjusted.append(
            replace(
                trade,
                exit_timestamp=breach.timestamp,
                exit_price=stop_price,
                pnl_percent=pnl_percent,
            )
        )

    return adjusted


def _first_breach(path, side: SignalSide, stop_price: Decimal):
    """The first candle in ``path`` whose range crosses ``stop_price``
    against ``side``, or None if the stop was never hit.

    BUY is stopped out on the way down (uses the candle's low); SELL is
    stopped out on the way up (uses the candle's high) -- the same
    intrabar-touch convention any real stop order would trigger on.
    """
    for candle in path:
        if side == SignalSide.BUY and candle.low <= stop_price:
            return candle
        if side == SignalSide.SELL and candle.high >= stop_price:
            return candle
    return None


def summarize(trades: list[Trade], side: SignalSide) -> dict:
    """Win rate / average win / average loss / expectancy for one side's
    closed trades -- the same shape of numbers used throughout the
    ranking-model roadmap's analysis, so stop-loss scenarios can be
    compared apples-to-apples against the no-stop-loss baseline."""
    closed = [
        t for t in trades if t.side == side and t.status == "closed" and t.pnl_percent is not None
    ]
    if not closed:
        return {"n": 0, "win_rate": None, "avg_win": None, "avg_loss": None, "expectancy": None}
    wins = [t.pnl_percent for t in closed if t.pnl_percent > 0]
    losses = [t.pnl_percent for t in closed if t.pnl_percent <= 0]
    return {
        "n": len(closed),
        "win_rate": Decimal(100 * len(wins)) / len(closed),
        "avg_win": sum(wins) / len(wins) if wins else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "expectancy": sum(t.pnl_percent for t in closed) / len(closed),
    }
