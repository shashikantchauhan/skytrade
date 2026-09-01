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

from trading_scanner.application import broker_reconciliation
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.order_intent import new_intent
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import LiveCashToggleState, TursoLiveOrderRepository
from trading_scanner.infrastructure.kite import KiteOrderExecutor, to_kite_tradingsymbol
from trading_scanner.infrastructure.kite_ticker import IST

logger = logging.getLogger(__name__)

_FILL_TIMEOUT_SECONDS = 30.0
_PURPOSE = "cash"
# Leg statuses that represent a real or still-unconfirmed order -- same set
# TursoLiveOrderRepository.get_unclosed_cash_legs uses (see its own
# docstring for the incident this excludes REJECTED/CANCELLED for).
_UNCLOSED_ENTRY_STATUSES = frozenset({"COMPLETE", "OPEN", "UNKNOWN"})
# 2026-09-01: a REJECTED/CANCELLED real BUY used to just notify and give
# up -- a genuinely good signal (already cleared eligibility, quality
# filter, and conviction filter to get this far) could lose its slot to a
# transient placement failure alone. Retry up to this many total attempts,
# backing off between them -- see execute_cash_entry's own retry loop for
# exactly what does and doesn't get retried.
#
# Deliberately bounded, not "keep retrying until the entry cutoff" --
# _rank_and_open_cash_positions attempts one scan cycle's candidates
# sequentially, so a retry loop that ran for hours would block every other
# candidate behind it in that same cycle (and likely still be running when
# the next hourly cycle closes). 10 attempts / 20s apart is ~3 minutes of
# backoff -- enough to ride out a real transient failure (a network blip,
# a momentary RMS/rate-limit hiccup, both of which resolve in seconds to
# low minutes), while still leaving the rest of the cycle's candidates a
# real chance. If it's still failing after ~3 minutes of retrying, that's
# very likely a persistent rejection reason (bad instrument, insufficient
# margin, ...) that more retrying wouldn't fix anyway.
_MAX_ENTRY_ATTEMPTS = 10
_ENTRY_RETRY_BACKOFF_SECONDS = 20.0


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
    tag: str | None = None,
) -> CashLegResult:
    """Places one real cash order and blocks (off the event loop) until it
    fills or times out. Never raises -- same "treat an exception as a
    REJECTED order" discipline as live_execution.py's basket flow.

    ``reference_price`` -- 2026-08-21: Kite now rejects a plain market
    order placed via the API ("Market orders without market protection are
    not allowed via API"), discovered when the very first real order this
    system ever placed failed on it. ``place_cash_market_order`` uses this
    to price a protected limit order instead (see its own docstring).

    ``tag`` -- see ``place_cash_market_order``'s own docstring; forwarded
    as-is (None for an exit, an entry's ``intent_id[:20]`` for an entry --
    see ``execute_cash_entry``)."""
    try:
        order_id = await asyncio.to_thread(
            order_executor.place_cash_market_order,
            tradingsymbol,
            transaction_type,
            quantity,
            reference_price,
            tag,
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
    live_order_repository: TursoLiveOrderRepository,
    basket_id: str,
    symbol: str,
    leg: CashLegResult,
    intent_id: str | None = None,
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
            intent_id=intent_id,
        )
    )


def _confirmed_no_position(status: str) -> bool:
    """True only for a status we know for certain never resulted in a real
    position -- Kite's own REJECTED/CANCELLED. Any other status, including
    one this codebase doesn't otherwise branch on (e.g. Kite's own
    "TRIGGER PENDING"), is NOT confirmed-safe: fail closed rather than risk
    a duplicate order against a position that may already be real."""
    return status in ("REJECTED", "CANCELLED")


async def _broker_ground_truth_preflight(
    symbol: str,
    tradingsymbol: str,
    intent,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> bool:
    """Closes the one gap ``intent_id`` alone can't (see ``domain/
    order_intent.py``'s own docstring): Kite accepting an order and this
    process dying before ``_record`` ever ran, so a fresh attempt at the
    exact same intent -- possibly a whole new process, after a restart --
    has zero local memory that anything was ever tried. Called once, right
    before the first placement attempt for a fresh intent; the local
    ``get_legs_by_intent`` check above already covers every case where
    *this* process's own ledger has something to say.

    Checks two independent broker-ground-truth signals and reconciles
    (records locally, notifies) whatever either finds instead of ever
    placing a new order on top of it, per the "prefer blocking a duplicate
    entry over placing another order" discipline this whole function
    exists for:

    1. An order still sitting in *today's* order book tagged with this
       exact intent (``KiteOrderExecutor.find_todays_order_by_tag`` --
       ``place_cash_market_order`` stamps every real entry with
       ``intent.intent_id[:20]`` as its Kite ``tag``, see that function's
       own docstring). Catches the order regardless of whether it has
       filled yet.
    2. A real held quantity for ``tradingsymbol``
       (``KiteOrderExecutor.holding_quantity``, the same ground-truth
       check ``gtt_bracket.reconcile_before_exit`` already trusts) that no
       local leg accounts for -- catches a filled position even when the
       tag lookup above somehow misses it, and independently catches the
       broader "a real position exists with literally zero local record"
       case (e.g. this exact signal will never recur, so the tag alone
       would never be re-checked).

    Returns True if a new order must be blocked.

    Best-effort: if the broker calls themselves fail (a transient Kite API
    error, or in tests, a fake that doesn't implement them), this is
    treated the same as "found nothing" and the caller proceeds --
    matching ``gtt_bracket.reconcile_before_exit``'s own established
    fallback discipline elsewhere in this codebase. The risk being closed
    here is what happens when the check *succeeds* and finds real
    evidence, not what happens when the check itself is unavailable:
    refusing every entry whenever a single broker call hiccups would trade
    a rare, already-mitigated risk (a lost local write) for a routine one
    (blocking genuinely fresh, legitimate signals on ordinary API
    flakiness)."""
    tag = intent.intent_id[:20]
    try:
        broker_order = await asyncio.to_thread(order_executor.find_todays_order_by_tag, tag)
    except Exception:
        logger.warning(
            "Broker order-book lookup by tag failed for %s -- proceeding without it.",
            symbol, exc_info=True,
        )
        broker_order = None

    if broker_order is not None:
        status = str(broker_order.get("status", "UNKNOWN"))
        average_price = broker_order.get("average_price")
        basket_id = f"{symbol}-cash-reconciled-crash-recovery-{datetime.now(UTC).isoformat()}"
        confirmed_safe = _confirmed_no_position(status)
        await _record(
            live_order_repository, basket_id, symbol,
            CashLegResult(
                tradingsymbol=tradingsymbol,
                transaction_type="BUY",
                quantity=int(
                    broker_order.get("quantity") or broker_order.get("filled_quantity") or 0
                ),
                order_id=str(broker_order.get("order_id") or ""),
                status=status,
                average_price=Decimal(str(average_price)) if average_price else None,
                rejection_reason=(
                    "Reconciled from Kite's own order book -- this process had no local "
                    "record of it (likely a crash between the broker accepting the order "
                    "and this app recording it normally)."
                ),
            ),
            intent.intent_id,
        )
        if not confirmed_safe:
            await notifier.send_text(
                "⚠️ <b>RECONCILIATION REQUIRED</b>\n"
                f"{symbol}: found an untracked order for this exact signal already in "
                f"Kite's order book (status={status}) -- recorded it, refusing to place a "
                "duplicate. Verify directly in Kite."
            )
            return True
        # REJECTED/CANCELLED -- confirmed no real position resulted. Worth
        # the audit-trail row above (this process never recorded it), but
        # not a reason to block a fresh attempt.

    try:
        real_quantity = await asyncio.to_thread(order_executor.holding_quantity, tradingsymbol)
    except Exception:
        logger.warning(
            "Broker holding-quantity check failed for %s -- proceeding without it.",
            symbol, exc_info=True,
        )
        real_quantity = None

    if real_quantity:
        basket_id = f"{symbol}-cash-reconciled-hidden-position-{datetime.now(UTC).isoformat()}"
        await _record(
            live_order_repository, basket_id, symbol,
            CashLegResult(
                tradingsymbol=tradingsymbol,
                transaction_type="BUY",
                quantity=real_quantity,
                order_id="RECONCILED-HOLDING",
                status="COMPLETE",
                average_price=None,
                rejection_reason=(
                    "Reconciled from Kite's own real holding quantity -- a real position "
                    "exists with no local ledger row at all (order_id/fill price unknown)."
                ),
            ),
            intent.intent_id,
        )
        await notifier.send_text(
            "⚠️ <b>RECONCILIATION REQUIRED</b>\n"
            f"{symbol}: {real_quantity} real shares are held at the broker with no "
            "matching local record -- recorded it, refusing to place a duplicate. "
            "Verify directly in Kite."
        )
        return True

    return False


async def execute_cash_entry(
    symbol: str,
    market_price: Decimal,
    config: AppConfig,
    cash_state: LiveCashToggleState,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
    now: datetime | None = None,
    signal_timestamp: datetime | None = None,
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
    gating is unrelated to time-of-day.

    ``signal_timestamp`` -- the candle/signal's own timestamp (e.g.
    ``RankedCandidate.entry_timestamp``), used to compute a deterministic
    ``OrderIntent`` (see ``domain/order_intent.py``) shared by every retry
    attempt in this call *and* by a second, independent call for the same
    signal after a process restart -- unlike ``basket_id``, which is fresh
    every call. Defaults to ``now`` (or the wall clock) when not given,
    which keeps intent lookups well-defined but loses the cross-restart
    correlation -- every real caller should pass the actual signal
    timestamp.

    A REJECTED/CANCELLED placement is retried up to ``_MAX_ENTRY_ATTEMPTS``
    times (backing off ``_ENTRY_RETRY_BACKOFF_SECONDS`` between attempts)
    before giving up -- COMPLETE/OPEN/UNKNOWN never retry, since those all
    mean a real order state already exists at the broker. Before the first
    attempt, a leg already recorded under this call's intent (e.g. this
    exact signal was already attempted by a process that has since
    restarted) blocks a second placement the same way an already-open
    position does -- see ``domain/order_intent.py``'s own docstring for
    what this does and doesn't protect against."""
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
    # broker_reconciliation.get_all_unclosed_positions, not get_all_open_
    # cash_legs directly -- an UNKNOWN-status position is real capital at
    # risk too and must count toward the cap (see that module's own
    # docstring); the old COMPLETE-only count could under-report how many
    # real positions were actually open.
    all_open = await broker_reconciliation.get_all_unclosed_positions(live_order_repository)
    if len(all_open) >= cash_state.max_positions:
        logger.info(
            "Live cash entry skipped for %s -- max_positions (%d) already open.",
            symbol,
            cash_state.max_positions,
        )
        return None

    resolved_now = now or datetime.now(UTC)
    intent = new_intent(symbol, "BUY", signal_timestamp or resolved_now, _PURPOSE)
    # Phase 4 (domain/order_intent.py): a leg already recorded under this
    # exact intent means a previous call -- possibly a process that has
    # since crashed and restarted -- already got as far as recording a
    # real or unconfirmed order for this exact signal. get_unclosed_cash_
    # legs above already blocks a new BUY whenever *any* unclosed leg
    # exists for the symbol regardless of intent; this is an independent,
    # more specific check keyed on the signal itself rather than just the
    # symbol, and is the mechanism that makes the intent -- not just the
    # basket -- a real, queryable identity for "this attempt." It does NOT
    # catch a broker-accepted order lost before any local write ever
    # happened -- see new_intent's own docstring.
    existing_intent_legs = await live_order_repository.get_legs_by_intent(intent.intent_id)
    if any(leg.status in _UNCLOSED_ENTRY_STATUSES for leg in existing_intent_legs):
        logger.warning(
            "Live cash entry for %s skipped -- intent %s already has a real/unconfirmed "
            "order recorded (likely a retry after a process restart); refusing to place "
            "a duplicate.",
            symbol, intent.intent_id,
        )
        return None

    tradingsymbol = to_kite_tradingsymbol(symbol)

    # P0 fix (2026-09-01): the intent-legs check above only catches an
    # attempt *this process itself* already recorded -- it says nothing
    # about an order Kite accepted that got lost before ``_record`` ever
    # ran (a crash between ``place_order`` returning and the local write).
    # See ``_broker_ground_truth_preflight``'s own docstring for exactly
    # what this closes and why it's checked here, once, rather than on
    # every retry attempt below (a REJECTED/CANCELLED local record from an
    # earlier attempt in *this* call already proves definitively that no
    # real position resulted, so there is nothing left to reconcile).
    if await _broker_ground_truth_preflight(
        symbol, tradingsymbol, intent, order_executor, live_order_repository, notifier
    ):
        return None

    quantity = max(1, int(cash_state.notional / market_price))
    basket_id = f"{symbol}-cash-entry-{datetime.now(UTC).isoformat()}"

    attempts_made = 0
    for attempt in range(1, _MAX_ENTRY_ATTEMPTS + 1):
        if attempt > 1:
            # Re-validate the two time/state-sensitive checks before
            # retrying -- the backoff sleep below could have crossed the
            # entry cutoff, or something else could have opened a real
            # position for this symbol in the meantime. Never retry past
            # either of those (that's exactly how a second/stacked
            # position gets placed -- see get_unclosed_cash_legs's own
            # docstring for the incident this discipline comes from).
            if config.live_cash_entry_cutoff_ist is not None:
                current_ist_time = datetime.now(UTC).astimezone(IST).time()
                if current_ist_time >= config.live_cash_entry_cutoff_ist:
                    logger.info(
                        "Live cash entry retry for %s abandoned -- now past the %s IST "
                        "entry cutoff.", symbol, config.live_cash_entry_cutoff_ist,
                    )
                    break
            if await live_order_repository.get_unclosed_cash_legs(symbol):
                logger.info(
                    "Live cash entry retry for %s abandoned -- a real position opened "
                    "for it during the retry backoff.", symbol,
                )
                break
            await asyncio.sleep(_ENTRY_RETRY_BACKOFF_SECONDS)

        leg = await _place_and_wait(
            order_executor, tradingsymbol, "BUY", quantity, market_price,
            tag=intent.intent_id[:20],
        )
        attempts_made += 1
        await _record(live_order_repository, basket_id, symbol, leg, intent.intent_id)

        if leg.status not in ("REJECTED", "CANCELLED"):
            # COMPLETE, OPEN, or UNKNOWN -- a real order state now exists
            # at the broker. Never retry past this point.
            break
        if attempt < _MAX_ENTRY_ATTEMPTS:
            logger.info(
                "Live cash entry for %s was %s (attempt %d/%d) -- retrying in %.0fs.",
                symbol, leg.status, attempt, _MAX_ENTRY_ATTEMPTS, _ENTRY_RETRY_BACKOFF_SECONDS,
            )

    if leg.status != "COMPLETE":
        retry_note = f" after {attempts_made} attempts" if attempts_made > 1 else ""
        await notifier.send_text(
            "⚠️ <b>LIVE CASH ORDER FAILED</b>\n"
            f"{symbol}: {tradingsymbol} BUY x{quantity} did not fill{retry_note} "
            f"(status={leg.status})."
        )
        return basket_id

    retry_note = (
        f" (succeeded on attempt {attempts_made}/{_MAX_ENTRY_ATTEMPTS})"
        if attempts_made > 1 else ""
    )
    await notifier.send_text(
        "✅ <b>LIVE CASH ORDER PLACED</b>\n"
        f"{symbol}: {tradingsymbol} BUY x{quantity}"
        + (f" @ {leg.average_price}" if leg.average_price else "")
        + retry_note
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
    # broker_reconciliation.get_unclosed_entry_leg, not get_open_cash_legs
    # directly -- see that module's own docstring. Every real caller
    # already ran gtt_bracket.reconcile_before_exit before reaching here,
    # so by this point "should_exit" is already known true; this lookup
    # just needs the tradingsymbol/quantity to close, which an UNKNOWN
    # leg carries just as reliably as a COMPLETE one (recorded at
    # placement time either way).
    open_leg = await broker_reconciliation.get_unclosed_entry_leg(symbol, live_order_repository)
    if open_leg is None:
        return None

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
