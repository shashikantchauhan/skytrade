"""Centralized "which leg represents this symbol's current real-or-
possibly-real cash position" lookup -- Phase 7 of `projectedPlann.md` (see
docs/architecture/000-audit.md).

Fixes the reviewed bug (audit doc's gap #1): every exit-eligibility call
site independently picked its own repository method for "is there a real
position to act on" -- most used ``get_open_cash_legs``/
``get_all_open_cash_legs`` (COMPLETE-only), which made an entry leg
recorded ``status="UNKNOWN"`` (fill status unconfirmed -- see
``live_cash_execution.py``'s ``wait_for_fill`` docstring for the incident
this status exists for) invisible to the strategy exit branch
(``signal_pipeline.py``), ``execute_cash_exit``, ``manual_exit.py``, and
the dashboard's ``/api/live-cash-positions`` alike -- even though the
entry-blocking check (``get_unclosed_cash_legs``) already correctly
recognized it as real. A real, unprotected position could sit forever
with no automated or manual path able to see or close it.

This module is the one place that "is there something to act on" decision
gets made now, using the broader (COMPLETE/OPEN/UNKNOWN) status set every
exit-eligibility/capacity check needs. It does NOT replace
``gtt_bracket.reconcile_before_exit`` -- that function's broker-ground-
truth check (``KiteOrderExecutor.holding_quantity``) already correctly
resolves an UNKNOWN leg once it's actually reached; the bug was that it
was never being reached. Every call site below should use these functions
instead of calling ``get_open_cash_legs``/``get_all_open_cash_legs``
directly.
"""

import asyncio
import logging
from collections.abc import Sequence

from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.order_lifecycle import PositionLifecycle, derive_position_lifecycle
from trading_scanner.infrastructure.db import LiveCashToggleState, TursoLiveOrderRepository
from trading_scanner.infrastructure.kite import KiteOrderExecutor, to_kite_tradingsymbol

logger = logging.getLogger(__name__)


async def get_unclosed_entry_leg(
    symbol: str, live_order_repository: TursoLiveOrderRepository
) -> LiveOrderLeg | None:
    """The leg every exit-eligibility call site (strategy exit,
    ``execute_cash_exit``, ``manual_exit.py``, the dashboard's manual-exit
    pre-check) should treat as "this symbol's current entry, if any real-
    or-possibly-real position might exist." None only when nothing is
    genuinely unclosed on record (``PositionLifecycle.NONE`` or
    ``CLOSED``) -- callers can then trust ``gtt_bracket.
    reconcile_before_exit``'s own ``holding_quantity`` check to resolve
    the rest, including an ``UNKNOWN`` leg that might represent a real
    fill."""
    legs = await live_order_repository.get_unclosed_cash_legs(symbol)
    return legs[0] if legs else None


async def get_all_unclosed_positions(
    live_order_repository: TursoLiveOrderRepository,
) -> Sequence[LiveOrderLeg]:
    """Every symbol's current unclosed entry leg, across the whole
    allowlist -- the broader analogue of ``get_all_open_cash_legs``, for
    the dashboard's positions view and ``execute_cash_entry``'s
    max_positions capacity check. An ``UNKNOWN`` leg counts as real
    capital at risk here too -- erring toward *overcounting* capacity
    (refusing a new entry sooner) rather than the old undercounting
    (allowing more real positions than ``max_positions`` was ever meant to
    permit)."""
    return await live_order_repository.get_all_unclosed_cash_legs()


async def position_lifecycle(
    symbol: str, live_order_repository: TursoLiveOrderRepository
) -> PositionLifecycle:
    """The full lifecycle state (see ``domain/order_lifecycle.py``) for
    ``symbol``'s real cash position -- surfaces ``RECONCILIATION_
    REQUIRED`` (an ``UNKNOWN`` leg exists anywhere in the still-relevant
    history) as its own distinct value, for dashboard/health-check
    consumers that want to flag it explicitly rather than only silently
    resolving it via broker ground truth the next time an exit is
    attempted."""
    legs = await live_order_repository.get_cash_legs(symbol)
    return derive_position_lifecycle(legs)


async def get_reconciliation_required_symbols(
    live_order_repository: TursoLiveOrderRepository,
) -> list[str]:
    """Every symbol whose real cash position currently needs broker
    ground truth to resolve (``PositionLifecycle.RECONCILIATION_
    REQUIRED``) -- Phase 15 (observability): backs the dashboard's health
    surface, so an ``UNKNOWN``-status leg is visible somewhere even before
    a strategy exit signal or manual action happens to resolve it. Read-
    only -- never places an order or otherwise acts on what it finds."""
    symbols = await live_order_repository.get_cash_symbols()
    flagged = []
    for symbol in symbols:
        lifecycle = await position_lifecycle(symbol, live_order_repository)
        if lifecycle is PositionLifecycle.RECONCILIATION_REQUIRED:
            flagged.append(symbol)
    return flagged


async def find_hidden_positions(
    cash_state: LiveCashToggleState,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
) -> list[str]:
    """Every symbol in the current cash allowlist with a real broker-side
    holding but literally no local unclosed leg accounting for it -- the
    one gap ``execute_cash_entry``'s own broker-ground-truth preflight
    (``live_cash_execution.py``, 2026-09-01) can't reach on its own,
    because that preflight only ever runs when a matching signal fires
    again. A hidden position can persist with no signal ever recurring for
    it (the crash that lost the local write happened on a signal that
    won't repeat, or the position came from something entirely outside
    this app's own order flow) -- this is the periodic, allowlist-wide
    sweep for exactly that case (the spec's own test case: "existing real
    broker position but missing local ledger row").

    Read-only -- never places an order or writes to the ledger itself
    (unlike the entry preflight, which reconciles what it finds because
    it's already mid-decision for that one symbol); this only surfaces the
    symbol list for a caller (``live_pipeline.py``) to alert a human about.
    Best-effort per symbol: a ``holding_quantity`` failure for one symbol
    is logged and skipped, never allowed to abort the sweep for the rest
    of the allowlist."""
    hidden: list[str] = []
    for symbol in sorted(cash_state.symbols):
        tradingsymbol = to_kite_tradingsymbol(symbol)
        try:
            real_quantity = await asyncio.to_thread(order_executor.holding_quantity, tradingsymbol)
        except Exception:
            logger.warning(
                "Hidden-position check failed for %s -- skipping this cycle.",
                symbol, exc_info=True,
            )
            continue
        if real_quantity <= 0:
            continue
        unclosed = await live_order_repository.get_unclosed_cash_legs(symbol)
        if not unclosed:
            hidden.append(symbol)
    return hidden
