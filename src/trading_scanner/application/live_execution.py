"""Real order execution: option-first, then-futures basket entry/exit.

Gated end-to-end behind ``AppConfig.live_trading_enabled`` +
``live_trading_symbols`` (see ``config/settings.py``'s kill-switch
docstring) -- every function here checks the gate itself and no-ops if it's
off, so calling these from the signal pipeline is safe regardless of
config, and nothing here ever places an order unless explicitly turned on
for that exact symbol.

Leg ordering, and why it's asymmetric between entry and exit:

- **Entry: option first, futures second.** The option leg is the smaller,
  defined-risk piece (premium paid = max loss). If it fills but the
  futures leg then fails, you're left holding a naked option -- bounded
  loss, not a crisis. The reverse order risks an unhedged futures position
  (open-ended risk) if the second leg fails -- exactly what the hedge
  exists to prevent. See the project's own history: this was explicitly
  corrected after a session where futures-first was the original guess.
- **Exit: futures first, option second.** Symmetric reasoning -- the
  futures leg carries the open-ended risk, so it's closed first to
  minimize how long that risk is held open; the option leg (bounded risk)
  can follow a beat later without meaningfully increasing exposure.

A failed second leg on entry triggers an immediate rollback (square off
the leg that did fill) rather than leaving a naked position sitting there
silently -- and either way, a Telegram alert fires so a human knows
something needs attention, matching this project's existing pattern of
"notification failures never block state changes, but state changes
always try to notify."
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg, SignalSide
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.kite import KiteDerivativesChain, KiteOrderExecutor
from trading_scanner.infrastructure.turso import TursoLiveOrderRepository

logger = logging.getLogger(__name__)

_FILL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class BasketLegResult:
    tradingsymbol: str
    transaction_type: str
    quantity: int
    order_id: str
    status: str
    average_price: Decimal | None
    rejection_reason: str | None


def _is_gated_in(symbol: str, config: AppConfig) -> bool:
    return config.live_trading_enabled and symbol in config.live_trading_symbols


async def _place_and_wait(
    order_executor: KiteOrderExecutor, tradingsymbol: str, transaction_type: str, quantity: int
) -> BasketLegResult:
    """Places one real order and blocks (off the event loop) until it fills
    or times out. Never raises -- a Kite API exception is treated the same
    as a REJECTED order, so basket sequencing always has a definite status
    to act on instead of an unhandled exception mid-basket."""
    try:
        order_id = await asyncio.to_thread(
            order_executor.place_market_order, tradingsymbol, transaction_type, quantity
        )
    except Exception:
        logger.exception("Live order placement failed for %s (%s)", tradingsymbol, transaction_type)
        return BasketLegResult(
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
    return BasketLegResult(
        tradingsymbol=tradingsymbol,
        transaction_type=transaction_type,
        quantity=quantity,
        order_id=order_id,
        status=status["status"],
        average_price=Decimal(str(average_price)) if average_price else None,
        rejection_reason=status.get("status_message"),
    )


async def _record(
    live_order_repository: TursoLiveOrderRepository,
    basket_id: str,
    symbol: str,
    purpose: str,
    leg: BasketLegResult,
) -> None:
    await live_order_repository.record_leg(
        LiveOrderLeg(
            basket_id=basket_id,
            symbol=symbol,
            purpose=purpose,
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


async def execute_basket_entry(
    symbol: str,
    side: SignalSide,
    hedge_option_type: str,
    hedge_strike_target: Decimal,
    config: AppConfig,
    derivatives_chain: KiteDerivativesChain,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> str | None:
    """Real hedged-futures basket entry: option leg first, futures leg
    second. Returns the basket_id if anything was placed, None if the gate
    was closed (live trading off, or this symbol isn't allowlisted) or the
    option/futures contract couldn't be resolved.

    Refuses to open a second real position for a symbol that already has
    one open (checked via ``live_order_repository``) -- this function is
    meant to be called once per fresh signal, not accumulate positions.
    """
    if not _is_gated_in(symbol, config):
        return None
    already_open = await live_order_repository.get_open_primary_legs(symbol)
    if already_open:
        logger.info("Live basket entry skipped for %s -- a real position is already open.", symbol)
        return None

    option_contract = derivatives_chain.nearest_atm_option(
        symbol, hedge_option_type, float(hedge_strike_target)
    )
    futures_contract = derivatives_chain.nearest_future(symbol)
    if option_contract is None or futures_contract is None:
        logger.warning("Live basket entry skipped for %s -- no option/futures contract.", symbol)
        return None

    lot_size = int(futures_contract["lot_size"])
    quantity = lot_size * config.live_trading_max_lots
    basket_id = f"{symbol}-entry-{datetime.now(UTC).isoformat()}"

    # Leg 1: option hedge, always bought (protection).
    option_leg = await _place_and_wait(
        order_executor, option_contract["tradingsymbol"], "BUY", quantity
    )
    await _record(live_order_repository, basket_id, symbol, "hedge", option_leg)

    if option_leg.status != "COMPLETE":
        await notifier.send_text(
            "⚠️ <b>LIVE ORDER FAILED</b>\n"
            f"{symbol}: option leg ({option_contract['tradingsymbol']}) did not fill "
            f"(status={option_leg.status}). Basket aborted -- no futures leg placed, "
            "no real position opened."
        )
        return basket_id

    # Leg 2: futures -- BUY for a long (BUY signal), SELL for a short (SELL signal).
    futures_transaction_type = "BUY" if side == SignalSide.BUY else "SELL"
    futures_leg = await _place_and_wait(
        order_executor, futures_contract["tradingsymbol"], futures_transaction_type, quantity
    )
    await _record(live_order_repository, basket_id, symbol, "primary", futures_leg)

    if futures_leg.status != "COMPLETE":
        # Rollback: the option leg is real and open -- square it off rather
        # than leave a lone hedge with nothing to hedge.
        unwind_leg = await _place_and_wait(
            order_executor, option_contract["tradingsymbol"], "SELL", quantity
        )
        await _record(live_order_repository, basket_id, symbol, "hedge", unwind_leg)
        await notifier.send_text(
            "⚠️ <b>LIVE ORDER FAILED</b>\n"
            f"{symbol}: futures leg ({futures_contract['tradingsymbol']}) did not fill "
            f"(status={futures_leg.status}). Option leg squared off "
            f"(unwind status={unwind_leg.status}). Basket aborted."
        )
        return basket_id

    await notifier.send_text(
        "✅ <b>LIVE ORDER PLACED</b>\n"
        f"{symbol}: option {option_contract['tradingsymbol']} + futures "
        f"{futures_contract['tradingsymbol']}, qty {quantity}"
    )
    return basket_id


async def execute_basket_exit(
    symbol: str,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> str | None:
    """Closes whatever real basket is open for ``symbol`` -- futures leg
    first (the open-ended-risk side), then the paired option hedge.
    Returns None if the gate is closed or nothing is actually open.
    """
    if not _is_gated_in(symbol, config):
        return None
    open_primary = await live_order_repository.get_open_primary_legs(symbol)
    if not open_primary:
        return None
    primary = open_primary[0]
    basket_id = f"{symbol}-exit-{datetime.now(UTC).isoformat()}"

    close_futures_txn = "SELL" if primary.transaction_type == "BUY" else "BUY"
    futures_close = await _place_and_wait(
        order_executor, primary.tradingsymbol, close_futures_txn, primary.quantity
    )
    await _record(live_order_repository, basket_id, symbol, "primary", futures_close)

    hedge_legs = await live_order_repository.get_legs(primary.basket_id)
    hedge_open = next(
        (leg for leg in hedge_legs if leg.purpose == "hedge" and leg.transaction_type == "BUY"),
        None,
    )
    hedge_close = None
    if hedge_open is not None:
        hedge_close = await _place_and_wait(
            order_executor, hedge_open.tradingsymbol, "SELL", hedge_open.quantity
        )
        await _record(live_order_repository, basket_id, symbol, "hedge", hedge_close)

    hedge_incomplete = hedge_open is not None and hedge_close.status != "COMPLETE"
    if futures_close.status != "COMPLETE" or hedge_incomplete:
        await notifier.send_text(
            "⚠️ <b>LIVE EXIT INCOMPLETE</b>\n"
            f"{symbol}: futures close status={futures_close.status}"
            + (f", hedge close status={hedge_close.status}" if hedge_close else "")
            + " -- check the account directly, this may need manual squaring off."
        )
    else:
        await notifier.send_text(f"✅ <b>LIVE POSITION CLOSED</b>\n{symbol}: basket squared off.")
    return basket_id
