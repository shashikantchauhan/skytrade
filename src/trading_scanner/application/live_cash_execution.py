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

Gated end-to-end behind ``LiveCashToggleState.enabled`` +
``LiveCashToggleState.symbols`` -- every function here checks the gate
itself and no-ops if it's off, so calling these from the signal pipeline
is safe regardless of config, and nothing here ever places an order
unless explicitly turned on for that exact symbol.

``cash_state: LiveCashToggleState`` (not fields read off ``AppConfig``)
is the single source of truth for every dashboard-adjustable cash setting
(enabled/symbols/notional/max_positions) -- deliberately its own explicit
parameter, not merged into a cloned ``AppConfig``. 2026-09-01: a real BUY
signal (HCLTECH.NS) cleared every gate and got no order because
``live_pipeline.py`` passed the wrong of two near-identical ``AppConfig``
objects (a static one, cloned dashboard-toggle one) into this module's old
``config``-only signature -- ``_is_gated_in`` silently returned False with
zero log output. Threading ``cash_state`` as its own explicitly-named,
distinctly-typed parameter means passing the wrong thing is a ``TypeError``
at the call site, not a silently wrong value -- see
``infrastructure/db/live_cash_toggle.py``'s ``LiveCashToggleState``,
already built fresh from the DB toggle every scan cycle in
``live_pipeline.py``, or from static config once per run in
``run_signal_pipeline`` (which has no per-cycle DB refresh). ``config``
(``AppConfig``) is still needed here, but now only ever for genuinely
static settings that are NOT dashboard-adjustable --
``live_cash_entry_cutoff_ist`` is the only one this module reads.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import LiveCashToggleState, TursoLiveOrderRepository
from trading_scanner.infrastructure.kite import KiteOrderExecutor, to_kite_tradingsymbol
from trading_scanner.infrastructure.kite_ticker import IST

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


def _is_gated_in(symbol: str, cash_state: LiveCashToggleState) -> bool:
    return cash_state.enabled and symbol in cash_state.symbols


async def _place_and_wait(
    order_executor: KiteOrderExecutor,
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    reference_price: Decimal,
) -> CashLegResult:
    """Places one real cash order and blocks (off the event loop) until it
    fills or times out. Never raises -- same "treat an exception as a
    REJECTED order" discipline as live_execution.py's basket flow.

    ``reference_price`` -- 2026-08-21: Kite now rejects a plain market
    order placed via the API ("Market orders without market protection are
    not allowed via API"), discovered when the very first real order this
    system ever placed failed on it. ``place_cash_market_order`` uses this
    to price a protected limit order instead (see its own docstring)."""
    try:
        order_id = await asyncio.to_thread(
            order_executor.place_cash_market_order,
            tradingsymbol,
            transaction_type,
            quantity,
            reference_price,
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
    try:
        status = await asyncio.to_thread(
            order_executor.wait_for_fill, order_id, _FILL_TIMEOUT_SECONDS
        )
    except Exception:
        # 2026-08-28: confirmed live against UNIONBANK.NS -- a real SELL
        # was placed and filled at the broker, but a transient Kite API
        # error while polling its fill status (order_history raised) meant
        # this exception used to propagate straight out of _place_and_wait
        # and the order was NEVER recorded anywhere -- our own ledger had
        # zero trace of a real trade that had already happened. order_id
        # is always known by this point regardless of what wait_for_fill
        # does, so this must never be allowed to lose it.
        logger.exception(
            "Order %s placed for %s (%s) but its fill status could not be confirmed -- "
            "recording as UNKNOWN, verify directly in Kite.",
            order_id, tradingsymbol, transaction_type,
        )
        return CashLegResult(
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_id=order_id,
            status="UNKNOWN",
            average_price=None,
            rejection_reason="fill status check raised an exception -- see logs, verify in Kite",
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
    cash_state: LiveCashToggleState,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
    now: datetime | None = None,
) -> str | None:
    """Real cash-equity BUY, sized from ``cash_state.notional`` /
    ``market_price`` (a fixed rupee amount per symbol, not a fixed share
    count -- a flat share count would mean wildly different real risk
    across a Rs50 stock and a Rs3,000 stock). Returns the basket_id if an
    order was placed, None if the gate was closed, it's past
    ``config.live_cash_entry_cutoff_ist``, a real position for this symbol
    is already open (refuses to stack a second one), or
    ``cash_state.max_positions`` real positions are already open across
    the whole allowlist (this is what makes a wide allowlist -- e.g. the
    full symbol universe -- safe to run: breadth of what's *eligible* to
    trade doesn't widen how much real capital can be at risk at once).

    ``now`` -- defaults to the real wall clock; overridable for tests. Only
    used for the entry-cutoff check below; the rest of this function's
    gating is unrelated to time-of-day."""
    if not _is_gated_in(symbol, cash_state):
        # 2026-09-01: logged now (previously silent) -- this exact gate
        # returning an unexpected False with zero output is what let a
        # config-threading bug hide a genuinely good signal (HCLTECH.NS)
        # in production with no trace until the user noticed by hand.
        logger.info(
            "Live cash entry skipped for %s -- cash trading gate closed "
            "(enabled=%s, symbol in allowlist=%s).",
            symbol, cash_state.enabled, symbol in cash_state.symbols,
        )
        return None
    if config.live_cash_entry_cutoff_ist is not None:
        current_ist_time = (now or datetime.now(UTC)).astimezone(IST).time()
        if current_ist_time >= config.live_cash_entry_cutoff_ist:
            logger.info(
                "Live cash entry skipped for %s -- past the %s IST entry cutoff "
                "(closing-session liquidity is unreliable for fills).",
                symbol, config.live_cash_entry_cutoff_ist,
            )
            return None
    # 2026-08-28: get_unclosed_cash_legs, not get_open_cash_legs -- the
    # latter only counts a COMPLETE leg as "already open," which let a
    # second real BUY through for PERSISTENT.NS while the first order's
    # fill was still unconfirmed (status OPEN). See that method's own
    # docstring for the incident.
    already_open = await live_order_repository.get_unclosed_cash_legs(symbol)
    if already_open:
        logger.info("Live cash entry skipped for %s -- a real position is already open.", symbol)
        return None
    all_open = await live_order_repository.get_all_open_cash_legs()
    if len(all_open) >= cash_state.max_positions:
        logger.info(
            "Live cash entry skipped for %s -- max_positions (%d) already open.",
            symbol,
            cash_state.max_positions,
        )
        return None

    tradingsymbol = to_kite_tradingsymbol(symbol)
    quantity = max(1, int(cash_state.notional / market_price))
    basket_id = f"{symbol}-cash-entry-{datetime.now(UTC).isoformat()}"

    leg = await _place_and_wait(order_executor, tradingsymbol, "BUY", quantity, market_price)
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


async def record_broker_side_exit(
    symbol: str,
    open_leg: LiveOrderLeg,
    market_price: Decimal,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> str:
    """Closes ``open_leg`` in our own ledger without placing any order --
    for when the real position is already flat at the broker (a GTT that
    fired, or a manual square-off outside this app) but ``live_order_legs``
    still shows it open. Records a synthetic COMPLETE closing leg (there is
    no real order to poll) priced at ``market_price`` -- the latest quote,
    not the true fill, since that already happened outside anything this
    app placed and Kite's own order book doesn't reliably surface it (see
    ``gtt_bracket.reconcile_before_exit``'s 2026-08-28 docstring). Called
    from both ``execute_cash_exit`` (pre-flight ground-truth check) and
    directly by callers when ``reconcile_before_exit`` already determined
    the position is flat, so a real exit order is never attempted at all.

    2026-08-28: added after COCHINSHIP.NS and VMM.NS were found stuck
    "open" for days with no way to exit or re-enter -- see that module's
    own docstring for the incident this fixes."""
    close_txn = "SELL" if open_leg.transaction_type == "BUY" else "BUY"
    basket_id = f"{symbol}-cash-reconciled-{datetime.now(UTC).isoformat()}"
    await _record(
        live_order_repository, basket_id, symbol,
        CashLegResult(
            tradingsymbol=open_leg.tradingsymbol,
            transaction_type=close_txn,
            quantity=open_leg.quantity,
            order_id="RECONCILED",
            status="COMPLETE",
            average_price=market_price,
            rejection_reason=None,
        ),
    )
    await notifier.send_text(
        "🔄 <b>POSITION RECONCILED</b>\n"
        f"{symbol}: already flat at the broker but our records still showed it open -- "
        f"closed it now at an approximate price ({market_price}, the latest quote, not "
        "the exact fill). Check Kite directly for the exact exit price/P&L."
    )
    return basket_id


async def execute_cash_exit(
    symbol: str,
    market_price: Decimal,
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
    current toggle state.

    ``market_price`` -- reference price for the protected limit order
    ``_place_and_wait`` places (see its own docstring); this symbol's
    current bar close/candle price, not the original entry price."""
    open_legs = await live_order_repository.get_open_cash_legs(symbol)
    if not open_legs:
        return None
    open_leg = open_legs[0]

    # 2026-08-28, then reverted same day: a redundant ground-truth
    # holding_quantity() check used to run here too (see
    # record_broker_side_exit's docstring for why it exists at all) --
    # every real caller already calls gtt_bracket.reconcile_before_exit
    # first, which makes the same check, so this doubled the real Kite API
    # calls made per exit. That doubling is what tipped UNIONBANK.NS into
    # Kite's own rate limit mid-cycle ("Too many requests"), which then hit
    # an unrelated fragility in wait_for_fill below and lost track of a
    # real order entirely (see that function's own 2026-08-28 fix). Removed
    # -- reconcile_before_exit is the single ground-truth check now.
    basket_id = f"{symbol}-cash-exit-{datetime.now(UTC).isoformat()}"

    close_txn = "SELL" if open_leg.transaction_type == "BUY" else "BUY"
    leg = await _place_and_wait(
        order_executor, open_leg.tradingsymbol, close_txn, open_leg.quantity, market_price
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
