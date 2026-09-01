"""Ledger of every real order leg placed on Zerodha -- see application/live_execution.py."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.infrastructure.db._shared import DbClient, add_column_if_missing

_CREATE_LIVE_ORDER_LEGS_TABLE = """
CREATE TABLE IF NOT EXISTS live_order_legs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    basket_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('cash', 'primary', 'hedge')),
    tradingsymbol TEXT NOT NULL,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    average_price REAL,
    rejection_reason TEXT
)
"""
# Phase 12 (DB invariants, projectedPlann.md): deliberately no CHECK on
# `status` -- it's Kite's own order-status vocabulary, passed through
# opaquely (see KiteOrderExecutor.wait_for_fill's own docstring: a timeout
# can return an intermediate status like "TRIGGER PENDING", not just the
# five this codebase's own logic branches on). A closed enum here that
# turns out incomplete would make a real order's own execution record fail
# to INSERT -- exactly the kind of lost-order risk this codebase has
# already been bitten by once (see live_cash_execution.py's UNIONBANK.NS
# incident). Constraining `purpose`/`transaction_type`/`quantity` above is
# safe because this codebase fully controls that vocabulary; `status` is
# not this codebase's vocabulary to close.
#
# SQLite can't add a CHECK constraint to an already-existing table via
# ALTER TABLE (only ADD COLUMN) -- this DDL only takes effect for a
# database where this table doesn't exist yet (a fresh test DB, or a
# brand-new deployment). The already-deployed production `live_order_legs`
# table predates this constraint and is NOT retroactively migrated --
# rebuilding it (CREATE new + copy + drop + rename) is exactly the
# destructive migration this phase is told to avoid without a real need.

# Statuses a cash leg can be in that represent (or might still turn into) a
# real held position -- see get_unclosed_cash_legs's own docstring.
_UNCLOSED_STATUSES = frozenset({"COMPLETE", "OPEN", "UNKNOWN"})


class TursoLiveOrderRepository:
    """Records every real order leg placed on Zerodha -- see
    ``application/live_execution.py``. Purely a ledger/audit trail (every
    row is an already-known outcome from Kite's own order API); nothing
    here decides whether to place an order.
    """

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_LIVE_ORDER_LEGS_TABLE)
        # Phase 4 (domain/order_intent.py) -- additive, nullable, no
        # backfill of historical rows (see LiveOrderLeg.intent_id's own
        # docstring).
        await add_column_if_missing(self._client, "live_order_legs", "intent_id", "TEXT")

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        await self._client.execute(
            """
            INSERT INTO live_order_legs
                (basket_id, symbol, purpose, tradingsymbol, transaction_type,
                 quantity, order_id, status, placed_at, average_price, rejection_reason,
                 intent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                leg.basket_id,
                leg.symbol,
                leg.purpose,
                leg.tradingsymbol,
                leg.transaction_type,
                leg.quantity,
                leg.order_id,
                leg.status,
                leg.placed_at.isoformat(),
                float(leg.average_price) if leg.average_price is not None else None,
                leg.rejection_reason,
                leg.intent_id,
            ],
        )

    async def get_legs_by_intent(self, intent_id: str) -> Sequence[LiveOrderLeg]:
        """Every leg (any status) recorded under ``intent_id`` -- see
        ``domain/order_intent.py``. Used before the first placement
        attempt for a logical trade to detect "a previous attempt at this
        exact signal already got as far as recording a real/unconfirmed
        order," independent of the fresh ``basket_id`` a new call would
        otherwise generate. Does NOT catch a broker-accepted order that
        crashed before any local write at all -- see that module's own
        docstring."""
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason,
                   intent_id
            FROM live_order_legs WHERE intent_id = ? ORDER BY placed_at ASC
            """,
            [intent_id],
        )
        return [_row_to_leg(row) for row in result.rows]

    async def get_legs(self, basket_id: str) -> Sequence[LiveOrderLeg]:
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason,
                   intent_id
            FROM live_order_legs WHERE basket_id = ? ORDER BY placed_at ASC
            """,
            [basket_id],
        )
        return [_row_to_leg(row) for row in result.rows]

    async def _cash_legs(self, symbol: str | None) -> list[LiveOrderLeg]:
        """Every ``purpose='cash'`` leg, optionally scoped to one symbol,
        at every status -- raw material for the net-quantity-aware
        open/unclosed derivations below."""
        query = """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason,
                   intent_id
            FROM live_order_legs WHERE purpose = 'cash'
        """
        params: list = []
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY placed_at ASC"
        result = await self._client.execute(query, params)
        return [_row_to_leg(row) for row in result.rows]

    async def get_cash_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every ``purpose='cash'`` leg for ``symbol``, at every status,
        ``placed_at``-ordered -- raw material for ``domain.order_lifecycle.
        derive_position_lifecycle`` (see ``application/broker_
        reconciliation.py``). Public counterpart of ``_cash_legs`` scoped
        to one symbol."""
        return await self._cash_legs(symbol)

    async def get_all_open_cash_legs(self) -> Sequence[LiveOrderLeg]:
        """Every currently-open (net quantity > 0) ``purpose='cash'`` leg
        across all symbols, COMPLETE fills only -- see ``get_all_unclosed_
        cash_legs`` for the broader (COMPLETE/OPEN/UNKNOWN) view every
        exit-eligibility/capacity check should use instead (see
        ``application/broker_reconciliation.py``). Kept for callers that
        specifically need "a confirmed real fill exists," same reasoning
        as ``get_open_cash_legs``'s own docstring."""
        legs = await self._cash_legs(symbol=None)
        return _net_unclosed_legs(legs, opener_statuses=frozenset({"COMPLETE"}))

    async def get_all_unclosed_cash_legs(self) -> Sequence[LiveOrderLeg]:
        """Every currently-unclosed (net quantity > 0, COMPLETE/OPEN/
        UNKNOWN) ``purpose='cash'`` leg across all symbols -- the broader
        analogue of ``get_all_open_cash_legs``, for the dashboard's real-
        positions view and the max_positions capacity check, both of which
        must count an UNKNOWN-status leg as real capital at risk (see
        ``get_unclosed_cash_legs``'s own docstring for the incident this
        broader status set fixes)."""
        legs = await self._cash_legs(symbol=None)
        return _net_unclosed_legs(legs, opener_statuses=_UNCLOSED_STATUSES)

    async def get_open_cash_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every currently-open (net quantity > 0) ``purpose='cash'`` leg
        for ``symbol`` -- see ``application/live_cash_execution.py``."""
        legs = await self._cash_legs(symbol)
        return _net_unclosed_legs(legs, opener_statuses=frozenset({"COMPLETE"}))

    async def get_unclosed_cash_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every ``purpose='cash'`` leg for ``symbol`` that represents a
        real or still-unconfirmed order (status ``COMPLETE``, ``OPEN``, or
        ``UNKNOWN`` -- excludes ``REJECTED``/``CANCELLED``, which never
        resulted in a real position) not yet fully offset by COMPLETE
        closing quantity.

        2026-08-28: broader than ``get_open_cash_legs`` (COMPLETE only) --
        used solely by ``execute_cash_entry``'s stacking-prevention check.
        Confirmed live against PERSISTENT.NS: a real BUY's fill-status poll
        (``wait_for_fill``) timed out while Kite still reported it as
        ``OPEN`` (it had, in fact, already filled moments later), and
        because ``get_open_cash_legs`` only ever counted ``COMPLETE`` legs
        as "already open," the next cycle's evaluation saw an apparently
        empty position and placed a second real BUY -- 17 shares (~2x the
        intended notional) bought instead of one properly-sized position.
        An ``OPEN``/``UNKNOWN`` leg must block a second entry exactly like
        a ``COMPLETE`` one does."""
        legs = await self._cash_legs(symbol)
        return _net_unclosed_legs(legs, opener_statuses=_UNCLOSED_STATUSES)

    async def get_open_primary_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every currently-open ``purpose='primary'`` leg for ``symbol`` --
        the futures-basket analogue of ``get_open_cash_legs``, scoped to
        the ``'primary'`` purpose instead of ``'cash'``; the two never see
        each other's rows. Still uses the older any-closer-exists check
        (not the quantity-netting fix from 2026-08-31) -- this path is the
        shadow/paper futures simulation, not real capital, so the
        partial-exit failure mode that hit PERSISTENT.NS's real cash
        position doesn't carry the same stakes here. Revisit if this ever
        starts placing real futures orders."""
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason,
                   intent_id
            FROM live_order_legs
            WHERE symbol = ? AND purpose = 'primary' AND status = 'COMPLETE'
              AND tradingsymbol NOT IN (
                  SELECT tradingsymbol FROM live_order_legs AS closer
                  WHERE closer.symbol = live_order_legs.symbol
                    AND closer.tradingsymbol = live_order_legs.tradingsymbol
                    AND closer.purpose = 'primary'
                    AND closer.transaction_type != live_order_legs.transaction_type
                    AND closer.status = 'COMPLETE'
              )
            ORDER BY placed_at ASC
            """,
            [symbol],
        )
        return [_row_to_leg(row) for row in result.rows]


def _row_to_leg(row: Sequence) -> LiveOrderLeg:
    return LiveOrderLeg(
        basket_id=row[0],
        symbol=row[1],
        purpose=row[2],
        tradingsymbol=row[3],
        transaction_type=row[4],
        quantity=int(row[5]),
        order_id=row[6],
        status=row[7],
        placed_at=datetime.fromisoformat(row[8]),
        average_price=Decimal(str(row[9])) if row[9] is not None else None,
        rejection_reason=row[10],
        intent_id=row[11] if len(row) > 11 else None,
    )


def _net_unclosed_legs(
    legs: Sequence[LiveOrderLeg], opener_statuses: frozenset[str]
) -> list[LiveOrderLeg]:
    """Groups ``legs`` by ``tradingsymbol`` and returns whichever opening
    legs (status in ``opener_statuses``, earliest-first) haven't yet been
    fully offset by COMPLETE closing quantity for that same tradingsymbol
    -- a FIFO net, not a binary "does any closer exist" check.

    2026-08-31 regression this replaced: the old SQL treated *any* COMPLETE
    opposite-transaction_type leg for a tradingsymbol as proof the whole
    position was closed, regardless of quantity. PERSISTENT.NS had been
    bought twice (8 + 9 shares, a stacking bug from three days earlier) and
    a single 8-share exit fully squared off only the first leg -- but the
    old query saw "a closer exists for PERSISTENT" and stopped counting
    *either* leg as open, hiding a real 9-share position from the
    dashboard, the exit path, and the max_positions capacity count.

    The "opening" transaction_type for a tradingsymbol is taken from
    whichever leg was placed first -- always BUY for this app's real cash
    trades (NSE cash delivery is long-only, no short selling), so this
    doesn't need to special-case direction."""
    by_tradingsymbol: dict[str, list[LiveOrderLeg]] = {}
    for leg in legs:
        by_tradingsymbol.setdefault(leg.tradingsymbol, []).append(leg)

    unclosed: list[LiveOrderLeg] = []
    for symbol_legs in by_tradingsymbol.values():
        symbol_legs = sorted(symbol_legs, key=lambda leg: leg.placed_at)
        opening_type = symbol_legs[0].transaction_type
        openers = [
            leg for leg in symbol_legs
            if leg.transaction_type == opening_type and leg.status in opener_statuses
        ]
        closed_quantity = sum(
            leg.quantity for leg in symbol_legs
            if leg.transaction_type != opening_type and leg.status == "COMPLETE"
        )
        remaining_to_close = closed_quantity
        for leg in openers:
            if remaining_to_close >= leg.quantity:
                remaining_to_close -= leg.quantity
                continue
            unclosed.append(leg)
            remaining_to_close = 0
    return unclosed
