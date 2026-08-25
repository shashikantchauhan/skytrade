"""Real target + trailing stop-loss OCO GTT bracket on a live cash-equity
position -- placed the moment ``live_cash_execution.execute_cash_entry``
fills, reconciled/cancelled the moment a strategy exit signal fires.

Mechanics: 10% target, 3% stop-loss at entry. Once price gets within
``EXTENSION_TRIGGER_PERCENT`` of the original target (checked once per
scan cycle -- see the module-level caveat below), the bracket is replaced
with a wider target (``EXTENDED_TARGET_PERCENT``) and the stop-loss
trailed up to breakeven (entry price), so a winner gets more room to run
while the worst case from that point on is "flat," not a loss. Extension
only ever happens once per position (``GttBracket.status`` "extended" is
terminal for this module -- it is not re-extended a second time).

**Polling caveat**: the exchange-side GTT triggers in real time regardless
of anything here -- that part is solid. But "is price near the target yet"
is only checked once per scan cycle (hourly by default). A sharp move that
jumps straight past the extension threshold and the original target
between two cycles will hit the original (unextended) target first,
closing the position before this module ever gets a chance to widen it.
Grinding moves are handled correctly; single-candle spikes are not.

**Reconciliation before a strategy exit**: a strategy exit signal doesn't
necessarily mean the real position is still open -- the GTT may have
already fired (target or stop-loss hit) since the last scan. Deleting an
already-fired/already-deleted GTT is harmless, but placing a market SELL
against a position that's already flat is not something to do blindly, so
``reconcile_before_exit`` checks the GTT's actual exchange-side status
first and tells the caller whether a market exit is still appropriate.
"""

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import GttBracket
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import TursoGttRepository
from trading_scanner.infrastructure.kite import KiteOrderExecutor, round_to_tick

logger = logging.getLogger(__name__)

TARGET_PERCENT = Decimal("10")
STOP_LOSS_PERCENT = Decimal("3")
EXTENSION_TRIGGER_PERCENT = Decimal("8")  # "near" the original 10% target
EXTENDED_TARGET_PERCENT = Decimal("15")

# Kite GTT statuses that mean "still live, has not fired" -- everything
# else (triggered/deleted/expired/cancelled/rejected) means the bracket is
# no longer actually managing the position.
_LIVE_STATUSES = frozenset({"active"})


def _is_gated_in(symbol: str, config: AppConfig) -> bool:
    return config.live_cash_trading_enabled and symbol in config.live_cash_trading_symbols


async def place_bracket(
    symbol: str,
    tradingsymbol: str,
    quantity: int,
    entry_price: Decimal,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    gtt_repository: TursoGttRepository,
    notifier: Notifier,
) -> None:
    """Places the initial 10%-target/3%-stop-loss OCO GTT for a real cash
    entry that just filled. Best-effort: a failure here notifies but never
    raises -- the real position stays open either way, just unmanaged by a
    GTT until the next entry (there is no automatic retry)."""
    if not _is_gated_in(symbol, config):
        return
    # 2026-08-25: was a flat .quantize(Decimal("0.05")) -- NSE requires the
    # exact tick size for this instrument (not always 0.05), see kite.py's
    # round_to_tick docstring. Off the event loop since tick_size() may hit
    # Kite's instrument-dump API on first use.
    tick = await asyncio.to_thread(order_executor.tick_size, tradingsymbol)
    stop_price = round_to_tick(entry_price * (1 - STOP_LOSS_PERCENT / 100), tick)
    target_price = round_to_tick(entry_price * (1 + TARGET_PERCENT / 100), tick)
    try:
        trigger_id = await asyncio.to_thread(
            order_executor.place_cash_bracket_gtt,
            tradingsymbol, quantity, entry_price, stop_price, target_price,
        )
    except Exception:
        logger.exception("GTT bracket placement failed for %s -- position stays unmanaged.", symbol)
        await notifier.send_text(
            f"⚠️ <b>GTT PLACEMENT FAILED</b>\n{symbol}: real position is open but has no "
            "target/stop-loss GTT -- placing one manually is recommended."
        )
        return
    await gtt_repository.record(
        GttBracket(
            symbol=symbol, trigger_id=trigger_id, tradingsymbol=tradingsymbol, quantity=quantity,
            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
            created_at=datetime.now(UTC),
        )
    )
    await notifier.send_text(
        f"✅ <b>GTT BRACKET PLACED</b>\n{symbol}: target {target_price} (+{TARGET_PERCENT}%), "
        f"stop {stop_price} (-{STOP_LOSS_PERCENT}%)"
    )


async def check_and_extend(
    symbol: str,
    current_price: Decimal,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    gtt_repository: TursoGttRepository,
    notifier: Notifier,
) -> None:
    """Called once per scan cycle for every symbol -- no-ops immediately
    unless there's an active (not yet extended) bracket for ``symbol`` and
    price has reached ``EXTENSION_TRIGGER_PERCENT`` gain. See the module
    docstring's polling caveat."""
    if not _is_gated_in(symbol, config):
        return
    bracket = await gtt_repository.get_active(symbol)
    if bracket is None or bracket.status != "active":
        return
    gain_percent = (current_price - bracket.entry_price) / bracket.entry_price * 100
    if gain_percent < EXTENSION_TRIGGER_PERCENT:
        return

    new_stop = bracket.entry_price  # trail to breakeven
    tick = await asyncio.to_thread(order_executor.tick_size, bracket.tradingsymbol)
    new_target = round_to_tick(bracket.entry_price * (1 + EXTENDED_TARGET_PERCENT / 100), tick)
    try:
        await asyncio.to_thread(
            order_executor.modify_cash_bracket_gtt,
            bracket.trigger_id, bracket.tradingsymbol, bracket.quantity,
            current_price, new_stop, new_target,
        )
    except Exception:
        logger.exception(
            "GTT extension failed for %s -- original bracket left in place, will retry next cycle.",
            symbol,
        )
        return
    await gtt_repository.update_status(
        bracket.trigger_id, "extended", stop_price=new_stop, target_price=new_target
    )
    await notifier.send_text(
        f"📈 <b>GTT EXTENDED</b>\n{symbol}: target raised to {new_target} "
        f"(+{EXTENDED_TARGET_PERCENT}%), stop trailed to breakeven ({new_stop})"
    )


async def reconcile_before_exit(
    symbol: str,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    gtt_repository: TursoGttRepository,
) -> bool:
    """Call before a strategy exit signal's market square-off. Returns
    True if the caller should still place a market exit (no bracket, or
    the bracket was still live and has now been cancelled), False if the
    GTT already fired since the last scan -- the real position is already
    flat, and a market SELL now would be against shares no longer held.
    Deliberately NOT gated on ``_is_gated_in`` (unlike ``place_bracket``/
    ``check_and_extend``) -- disabling the toggle must only ever stop new
    brackets from being placed, never leave an already-live GTT unmanaged
    when a strategy exit fires. Whether to reconcile is a fact about
    whether a bracket is actually active (``gtt_repository``), not about
    the current toggle state.
    """
    bracket = await gtt_repository.get_active(symbol)
    if bracket is None:
        return True

    try:
        status = await asyncio.to_thread(order_executor.gtt_status, bracket.trigger_id)
    except Exception:
        logger.warning(
            "GTT status check failed for %s -- assuming still live, will attempt cancel.",
            symbol, exc_info=True,
        )
        status = "active"

    if status not in _LIVE_STATUSES:
        # Already triggered/deleted/expired at the exchange -- the real
        # position is already flat, nothing left to cancel or sell.
        await gtt_repository.update_status(bracket.trigger_id, "closed")
        return False

    try:
        await asyncio.to_thread(order_executor.delete_gtt, bracket.trigger_id)
    except Exception:
        # Treat "already gone by the time we tried to delete it" the same
        # as the status check catching it -- either way it's not live
        # anymore, and this must never block the market exit that follows.
        logger.warning(
            "GTT delete failed for %s -- treating as already gone.", symbol, exc_info=True
        )
    await gtt_repository.update_status(bracket.trigger_id, "cancelled")
    return True
