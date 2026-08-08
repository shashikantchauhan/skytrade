"""Paper-trading account: long-only simulated real-money positions.

NSE cash market does not allow short selling for multi-day (delivery) holds
-- only intraday MIS positions can be short, squared off same day. This
strategy's average holding period is ~3.5 days, so SELL/short signals can
never be executed as real cash-market positions; the paper account only ever
opens a position on a BUY entry. SELL signals still notify (see
``signal_pipeline.py``) but are informational only.

Two gates decide whether a BUY entry actually becomes a paper position:

1. **Eligibility**: the symbol's own closed-trade, BUY-only win rate (see
   ``application/backtest.py``/``signal_pipeline.py``'s trade bookkeeping)
   must be at least ``MIN_WIN_RATE``, and it must have at least
   ``MIN_CLOSED_TRADES`` closed BUY trades to compute a meaningful rate from.
   A symbol with no track record yet, or a poor one, is skipped -- still
   notified, just tagged as not paper-traded.
2. **Capacity**: the account only has ``INITIAL_CAPITAL`` to work with, split
   into fixed-size slots (``POSITION_SIZE``). If the cash balance can't cover
   one more slot, the entry is skipped and tagged accordingly rather than
   silently dropped.

``POSITION_SIZE`` (₹75,000) is chosen from the target average win of
₹2,000-3,000/trade: this strategy's BUY-only average winning trade is ~3.34%,
and 3.34% of ₹75,000 is ~₹2,500 -- squarely in that target range. ₹5,00,000 /
₹75,000 gives ~6-7 concurrent positions, matching the earlier capacity
analysis (average concurrent positions ≈ entries/day × average holding
period, via Little's Law).
"""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import PaperPosition, SignalSide
from trading_scanner.domain.ports import PaperAccountRepository, TradeRepository

INITIAL_CAPITAL = Decimal("500000")
POSITION_SIZE = Decimal("75000")
MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5


async def is_eligible(symbol: str, interval: str, trade_repository: TradeRepository) -> bool:
    """Return whether a symbol's BUY-only track record clears the paper-trading bar.

    Long-only, so only BUY-side closed trades count -- a symbol whose edge is
    entirely on the SELL side is still not tradeable here.
    """
    win_rate = await _buy_only_win_rate(symbol, interval, trade_repository)
    return win_rate is not None and win_rate >= MIN_WIN_RATE


async def try_open_position(
    symbol: str,
    entry_timestamp: datetime,
    entry_price: Decimal,
    paper_account_repository: PaperAccountRepository,
) -> PaperPosition | None:
    """Open a paper position sized at POSITION_SIZE if capital allows.

    Returns None (no position opened) if the remaining cash balance can't
    cover one more slot -- the caller is responsible for notifying that the
    signal was skipped for lack of capital, not silently dropping it.
    """
    cash_balance = await paper_account_repository.get_cash_balance()
    if cash_balance < POSITION_SIZE:
        return None
    quantity = int(POSITION_SIZE / entry_price)
    if quantity < 1:
        return None
    position = PaperPosition(
        symbol=symbol,
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        quantity=quantity,
        capital_allocated=quantity * entry_price,
    )
    await paper_account_repository.open_position(position)
    return position


async def _buy_only_win_rate(
    symbol: str, interval: str, trade_repository: TradeRepository
) -> Decimal | None:
    """Compute the closed BUY-only win rate, or None if too few trades exist."""
    trades = await trade_repository.get_trades(symbol, interval)
    closed_buys = [
        trade for trade in trades if trade.side == SignalSide.BUY and trade.status == "closed"
    ]
    if len(closed_buys) < MIN_CLOSED_TRADES:
        return None
    wins = sum(1 for trade in closed_buys if trade.pnl_percent is not None and trade.pnl_percent > 0)
    return Decimal(100 * wins) / len(closed_buys)
