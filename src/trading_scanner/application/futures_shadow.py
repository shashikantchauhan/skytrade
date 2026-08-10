"""Shadow-tracks a hypothetical futures trade for every BUY/SELL signal.

``side="long"`` for BUY, ``side="short"`` for SELL -- futures are the real
short mechanism a real broker connection would use (unlike equity, where
NSE cash market has no short selling -- see ``paper_trading.py``). Always
paired with a protective options hedge (see ``options_shadow.py``,
``purpose="hedge"``): a PUT for a long future, a CALL for a short future.
Analysis only, entirely separate from the paper account's capital, never a
real order.

Best-effort and silent on failure, matching ``options_shadow.py`` -- a
missing futures contract or a quote hiccup never breaks the surrounding
signal handling in ``signal_pipeline.py``.
"""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import FuturesShadowTrade
from trading_scanner.infrastructure.kite import KiteDerivativesChain
from trading_scanner.infrastructure.turso import TursoFuturesTradeRepository


async def try_open_futures_position(
    symbol: str,
    side: str,
    entry_timestamp: datetime,
    entry_price: Decimal,
    derivatives_chain: KiteDerivativesChain,
    futures_trade_repository: TursoFuturesTradeRepository,
    purpose: str = "primary",
) -> str | None:
    """Record a hypothetical futures entry at the nearest expiry.

    ``purpose="primary"`` (default): this future is the trade itself.
    ``purpose="hedge"``: this future hedges a directional option instead
    (see ``options_shadow.py``) -- ``side`` is then the opposite-delta
    side (short hedges a bought CALL, long hedges a bought PUT), decided
    by the caller (``signal_pipeline.py``), not by this function.
    """
    contract = derivatives_chain.nearest_future(symbol)
    if contract is None:
        return None
    trade = FuturesShadowTrade(
        symbol=symbol,
        side=side,
        futures_tradingsymbol=contract["tradingsymbol"],
        expiry=str(contract["expiry"]),
        lot_size=int(contract["lot_size"]),
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        purpose=purpose,
    )
    await futures_trade_repository.open_trade(trade)
    return f"futures-shadow({purpose}): opened {side} {contract['tradingsymbol']} @ {entry_price}"


async def close_futures_position(
    symbol: str,
    exit_timestamp: datetime,
    exit_price: Decimal,
    futures_trade_repository: TursoFuturesTradeRepository,
    purpose: str = "primary",
) -> str | None:
    """Close the matching open shadow futures position, if one exists."""
    open_trade = await futures_trade_repository.get_open_trade(symbol, purpose)
    if open_trade is None:
        return None
    if open_trade.side == "long":
        pnl_amount = (exit_price - open_trade.entry_price) * open_trade.lot_size
    else:
        pnl_amount = (open_trade.entry_price - exit_price) * open_trade.lot_size
    pnl_percent = pnl_amount / (open_trade.entry_price * open_trade.lot_size) * 100
    await futures_trade_repository.close_trade(
        symbol, exit_timestamp, exit_price, pnl_amount, pnl_percent, purpose
    )
    return (
        f"futures-shadow({purpose}): closed {open_trade.side} {open_trade.futures_tradingsymbol} "
        f"@ {exit_price} pnl={pnl_percent:.2f}%"
    )
