"""Ledger of every real order leg placed on Zerodha -- see application/live_execution.py."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_LIVE_ORDER_LEGS_TABLE = """
CREATE TABLE IF NOT EXISTS live_order_legs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    basket_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    purpose TEXT NOT NULL,
    tradingsymbol TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    average_price REAL,
    rejection_reason TEXT
)
"""


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

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        await self._client.execute(
            """
            INSERT INTO live_order_legs
                (basket_id, symbol, purpose, tradingsymbol, transaction_type,
                 quantity, order_id, status, placed_at, average_price, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ],
        )

    async def get_legs(self, basket_id: str) -> Sequence[LiveOrderLeg]:
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason
            FROM live_order_legs WHERE basket_id = ? ORDER BY placed_at ASC
            """,
            [basket_id],
        )
        return [
            LiveOrderLeg(
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
            )
            for row in result.rows
        ]

    async def get_all_open_cash_legs(self) -> Sequence[LiveOrderLeg]:
        """Every currently-open ``purpose='cash'`` leg across all symbols --
        same open/closed logic as ``get_open_cash_legs``, just not scoped to
        one symbol. Used by the dashboard's real-positions view (see
        webapp.py's /api/live-cash-positions), which needs to list all of
        them, not check one at a time."""
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason
            FROM live_order_legs
            WHERE purpose = 'cash' AND status = 'COMPLETE'
              AND tradingsymbol NOT IN (
                  SELECT tradingsymbol FROM live_order_legs AS closer
                  WHERE closer.symbol = live_order_legs.symbol
                    AND closer.tradingsymbol = live_order_legs.tradingsymbol
                    AND closer.purpose = 'cash'
                    AND closer.transaction_type != live_order_legs.transaction_type
                    AND closer.status = 'COMPLETE'
              )
            ORDER BY placed_at ASC
            """
        )
        return [
            LiveOrderLeg(
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
            )
            for row in result.rows
        ]

    async def get_open_cash_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every currently-open ``purpose='cash'`` leg for ``symbol`` -- see
        ``application/live_cash_execution.py``. Same open/closed logic as
        ``get_open_primary_legs`` below, just scoped to the cash-order
        purpose instead of the futures-basket ``'primary'`` purpose; the two
        never see each other's rows."""
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason
            FROM live_order_legs
            WHERE symbol = ? AND purpose = 'cash' AND status = 'COMPLETE'
              AND tradingsymbol NOT IN (
                  SELECT tradingsymbol FROM live_order_legs AS closer
                  WHERE closer.symbol = live_order_legs.symbol
                    AND closer.tradingsymbol = live_order_legs.tradingsymbol
                    AND closer.purpose = 'cash'
                    AND closer.transaction_type != live_order_legs.transaction_type
                    AND closer.status = 'COMPLETE'
              )
            ORDER BY placed_at ASC
            """,
            [symbol],
        )
        return [
            LiveOrderLeg(
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
            )
            for row in result.rows
        ]

    async def get_open_primary_legs(self, symbol: str) -> Sequence[LiveOrderLeg]:
        """Every currently-open (no matching closing leg yet) real
        ``purpose="primary"`` futures leg for ``symbol`` -- used to check
        "do we already hold a real position here" before opening another,
        and to know what to square off on exit. A leg counts as open if its
        basket_id has no later leg with the opposite transaction_type for
        the same tradingsymbol (i.e. it was never closed out)."""
        result = await self._client.execute(
            """
            SELECT basket_id, symbol, purpose, tradingsymbol, transaction_type,
                   quantity, order_id, status, placed_at, average_price, rejection_reason
            FROM live_order_legs
            WHERE symbol = ? AND purpose = 'primary' AND status = 'COMPLETE'
              AND tradingsymbol NOT IN (
                  SELECT tradingsymbol FROM live_order_legs AS closer
                  WHERE closer.symbol = live_order_legs.symbol
                    AND closer.tradingsymbol = live_order_legs.tradingsymbol
                    AND closer.transaction_type != live_order_legs.transaction_type
                    AND closer.status = 'COMPLETE'
              )
            ORDER BY placed_at ASC
            """,
            [symbol],
        )
        return [
            LiveOrderLeg(
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
            )
            for row in result.rows
        ]
