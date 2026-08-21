"""Real order execution: plain NSE cash-equity BUY/SELL, one leg -- no
hedge, no futures. This is a small fixed-notional trial (see
``live_cash_trading_notional``) for validating that entry/exit orders
actually fire at the right time, before this ever gates real capital-sized
positions.

Deliberately separate from ``live_execution.py`` (the futures+hedge
basket): different kill switch (``live_cash_trading_enabled`` /
``live_cash_trading_symbols``, not ``live_trading_enabled`` /
``live_trading_symbols``), different product (NSE cash CNC delivery, not
NFO NRML), one leg instead of two. Reuses the same
``TursoLiveOrderRepository`` ledger with ``purpose="cash"`` so both real-
order flows are queryable from the one table, but never touches or reads
``purpose="primary"``/``"hedge"`` rows -- entirely independent bookkeeping.

Gated end-to-end behind ``AppConfig.live_cash_trading_enabled`` +
``live_cash_trading_symbols`` -- every function here checks the gate
itself and no-ops if it's off, so calling these from the signal pipeline
is safe regardless of config, and nothing here ever places an order
unless explicitly turned on for that exact symbol.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import TursoLiveOrderRepository
from trading_scanner.infrastructure.kite import KiteOrderExecutor, to_kite_tradingsymbol

logger = logging.getLogger(__name__)

_FILL_TIMEOUT_SECONDS = 30.0
_PURPOSE = "cash"


@dataclass(frozen=True, slots=True)
class CashLegResult:
    tradingsymbol: str
    transaction_type: str
    quantity: int
    order_id: str
    status: str
    average_price: Decimal | None
    rejection_reason: str | None


def _is_gated_in(symbol: str, config: AppConfig) -> bool:
    return config.live_cash_trading_enabled and symbol in config.live_cash_trading_symbols


async def _place_and_wait(
    order_executor: KiteOrderExecutor, tradingsymbol: str, transaction_type: str, quantity: int
) -> CashLegResult:
    """Places one real cash order and blocks (off the event loop) until it
    fills or times out. Never raises -- same "treat an exception as a
    REJECTED order" discipline as live_execution.py's basket flow."""
    try:
        order_id = await asyncio.to_thread(
            order_executor.place_cash_market_order, tradingsymbol, transaction_type, quantity
        )
    except Exception:
        logger.exception(
            "Live cash order placement failed for %s (%s)", tradingsymbol, transaction_type
        )
        return CashLegResult(
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_id="",
            status="REJECTED",
            average_price=None,
            rejection_reason="place_order raised an exception -- see logs",
        )
    status = await asyncio.to_thread(
        order_executor.wait_for_fill, order_id, _FILL_TIMEOUT_SECONDS
    )
    average_price = status.get("average_price")
    return CashLegResult(
        tradingsymbol=tradingsymbol,
        transaction_type=transaction_type,
        quantity=quantity,
        order_id=order_id,
        status=status["status"],
        average_price=Decimal(str(average_price)) if average_price else None,
        rejection_reason=status.get("status_message"),
    )


async def _record(
    live_order_repository: TursoLiveOrderRepository, basket_id: str, symbol: str, leg: CashLegResult
) -> None:
    await live_order_repository.record_leg(
        LiveOrderLeg(
            basket_id=basket_id,
            symbol=symbol,
            purpose=_PURPOSE,
            tradingsymbol=leg.tradingsymbol,
            transaction_type=leg.transaction_type,
            quantity=leg.quantity,
            order_id=leg.order_id,
            status=leg.status,
            placed_at=datetime.now(UTC),
            average_price=leg.average_price,
            rejection_reason=leg.rejection_reason,
        )
    )


async def execute_cash_entry(
    symbol: str,
    market_price: Decimal,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> str | None:
    """Real cash-equity BUY, sized from ``config.live_cash_trading_notional``
    / ``market_price`` (a fixed rupee amount per symbol, not a fixed share
    count -- a flat share count would mean wildly different real risk
    across a Rs50 stock and a Rs3,000 stock). Returns the basket_id if an
    order was placed, None if the gate was closed, a real position for this
    symbol is already open (refuses to stack a second one), or
    ``live_cash_trading_max_positions`` real positions are already open
    across the whole allowlist (this is what makes a wide allowlist -- e.g.
    the full symbol universe -- safe to run: breadth of what's *eligible*
    to trade doesn't widen how much real capital can be at risk at once)."""
    if not _is_gated_in(symbol, config):
        return None
    already_open = await live_order_repository.get_open_cash_legs(symbol)
    if already_open:
        logger.info("Live cash entry skipped for %s -- a real position is already open.", symbol)
        return None
    all_open = await live_order_repository.get_all_open_cash_legs()
    if len(all_open) >= config.live_cash_trading_max_positions:
        logger.info(
            "Live cash entry skipped for %s -- max_positions (%d) already open.",
            symbol,
            config.live_cash_trading_max_positions,
        )
        return None

    tradingsymbol = to_kite_tradingsymbol(symbol)
    quantity = max(1, int(config.live_cash_trading_notional / market_price))
    basket_id = f"{symbol}-cash-entry-{datetime.now(UTC).isoformat()}"

    leg = await _place_and_wait(order_executor, tradingsymbol, "BUY", quantity)
    await _record(live_order_repository, basket_id, symbol, leg)

    if leg.status != "COMPLETE":
        await notifier.send_text(
            "⚠️ <b>LIVE CASH ORDER FAILED</b>\n"
            f"{symbol}: {tradingsymbol} BUY x{quantity} did not fill (status={leg.status})."
        )
        return basket_id

    await notifier.send_text(
        "✅ <b>LIVE CASH ORDER PLACED</b>\n"
        f"{symbol}: {tradingsymbol} BUY x{quantity}"
        + (f" @ {leg.average_price}" if leg.average_price else "")
    )
    return basket_id


async def execute_cash_exit(
    symbol: str,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> str | None:
    """Squares off whatever real cash position ``execute_cash_entry``
    opened for ``symbol``. Returns None if nothing is actually open.

    Deliberately NOT gated on ``_is_gated_in`` (unlike ``execute_cash_entry``)
    -- disabling the toggle or removing a symbol from the allowlist must
    only ever stop *new* entries, never strand an already-open real
    position with no way to be closed. Whether to exit is a fact about
    what's actually open (``live_order_repository``), not about the
    current toggle state."""
    open_legs = await live_order_repository.get_open_cash_legs(symbol)
    if not open_legs:
        return None
    open_leg = open_legs[0]
    basket_id = f"{symbol}-cash-exit-{datetime.now(UTC).isoformat()}"

    close_txn = "SELL" if open_leg.transaction_type == "BUY" else "BUY"
    leg = await _place_and_wait(
        order_executor, open_leg.tradingsymbol, close_txn, open_leg.quantity
    )
    await _record(live_order_repository, basket_id, symbol, leg)

    if leg.status != "COMPLETE":
        await notifier.send_text(
            "⚠️ <b>LIVE CASH EXIT INCOMPLETE</b>\n"
            f"{symbol}: {open_leg.tradingsymbol} close status={leg.status} -- "
            "check the account directly, this may need manual squaring off."
        )
    else:
        await notifier.send_text(
            "✅ <b>LIVE CASH POSITION CLOSED</b>\n"
            f"{symbol}: {open_leg.tradingsymbol} squared off"
            + (f" @ {leg.average_price}" if leg.average_price else "")
        )
    return basket_id
