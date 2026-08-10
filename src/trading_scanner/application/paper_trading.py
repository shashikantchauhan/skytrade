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
   into ``TARGET_SLOTS`` dynamically-sized slots. If the cash balance can't
   cover one more slot, the entry is skipped and tagged accordingly rather
   than silently dropped.

``TARGET_SLOTS`` (32) matches real signal demand: Little's Law
(concurrent positions needed ~= entries/day x average holding period),
computed only over symbols that actually clear the eligibility bar above
(ineligible symbols never reach ``try_open_position`` at all, so they don't
count toward real capacity demand). ``INITIAL_CAPITAL`` (Rs 8,00,000) is
sized so 32 slots at the resulting ~Rs 25,000/slot fully covers that demand
with no capital-driven skips under normal conditions.

Slot size is **dynamic**, not fixed: every entry recomputes
``total_equity / TARGET_SLOTS``, where total_equity is cash plus all open
positions' allocated capital. As the account compounds profit week over
week, each slot grows proportionally -- no manual re-tuning needed. A floor
(``MIN_POSITION_SIZE``, Rs 25,000) keeps the flat per-trade DP charge
(~Rs 18, sell-side only) under ~5% of an average winning trade's profit;
below that floor, flat fees start eating a disproportionate share of returns.
"""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import PaperPosition, SignalSide
from trading_scanner.domain.ports import PaperAccountRepository, TradeRepository

INITIAL_CAPITAL = Decimal("800000")
TARGET_SLOTS = 32
MIN_POSITION_SIZE = Decimal("25000")
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
    """Open a paper position sized off current total equity if capital allows.

    Slot size is recomputed fresh on every call from total_equity /
    TARGET_SLOTS (floored at MIN_POSITION_SIZE) so the account scales
    proportionally as profit compounds in, without a hardcoded slot size
    going stale. Returns None (no position opened) if the remaining cash
    balance can't cover one more slot -- the caller is responsible for
    notifying that the signal was skipped for lack of capital, not silently
    dropping it.
    """
    cash_balance = await paper_account_repository.get_cash_balance()
    open_positions = await paper_account_repository.get_open_positions()
    total_equity = cash_balance + sum(
        (position.capital_allocated for position in open_positions), start=Decimal("0")
    )
    position_size = max(total_equity / TARGET_SLOTS, MIN_POSITION_SIZE)

    if cash_balance < position_size:
        return None
    quantity = int(position_size / entry_price)
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
    wins = sum(
        1 for trade in closed_buys if trade.pnl_percent is not None and trade.pnl_percent > 0
    )
    return Decimal(100 * wins) / len(closed_buys)
