"""Persists real target+stop-loss OCO GTT brackets on live cash-equity
positions -- see application/gtt_bracket.py."""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import GttBracket
from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_GTT_BRACKETS_TABLE = """
CREATE TABLE IF NOT EXISTS gtt_brackets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trigger_id INTEGER NOT NULL,
    tradingsymbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
)
"""


class TursoGttRepository:
    """One row per real cash-equity position's bracket -- mutated in place
    as the bracket extends/closes/gets cancelled, not append-only like
    ``TursoLiveOrderRepository``'s ledger."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_GTT_BRACKETS_TABLE)

    async def record(self, bracket: GttBracket) -> None:
        await self._client.execute(
            """
            INSERT INTO gtt_brackets
                (symbol, trigger_id, tradingsymbol, quantity, entry_price,
                 stop_price, target_price, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                bracket.symbol, bracket.trigger_id, bracket.tradingsymbol, bracket.quantity,
                float(bracket.entry_price), float(bracket.stop_price), float(bracket.target_price),
                bracket.created_at.isoformat(), bracket.status,
            ],
        )

    async def get_active(self, symbol: str) -> GttBracket | None:
        """The one bracket for ``symbol`` still live on the exchange
        (status "active" or "extended") -- None if there isn't one."""
        result = await self._client.execute(
            """
            SELECT symbol, trigger_id, tradingsymbol, quantity, entry_price,
                   stop_price, target_price, created_at, status
            FROM gtt_brackets
            WHERE symbol = ? AND status IN ('active', 'extended')
            ORDER BY created_at DESC LIMIT 1
            """,
            [symbol],
        )
        if not result.rows:
            return None
        return _row_to_bracket(result.rows[0])

    async def update_status(
        self, trigger_id: int, status: str,
        stop_price: Decimal | None = None, target_price: Decimal | None = None,
    ) -> None:
        """Moves a bracket to a new status, optionally recording new
        trigger prices (only set on an extension)."""
        if stop_price is not None and target_price is not None:
            await self._client.execute(
                "UPDATE gtt_brackets SET status = ?, stop_price = ?, target_price = ? "
                "WHERE trigger_id = ?",
                [status, float(stop_price), float(target_price), trigger_id],
            )
        else:
            await self._client.execute(
                "UPDATE gtt_brackets SET status = ? WHERE trigger_id = ?", [status, trigger_id]
            )


def _row_to_bracket(row) -> GttBracket:
    return GttBracket(
        symbol=row[0],
        trigger_id=int(row[1]),
        tradingsymbol=row[2],
        quantity=int(row[3]),
        entry_price=Decimal(str(row[4])),
        stop_price=Decimal(str(row[5])),
        target_price=Decimal(str(row[6])),
        created_at=datetime.fromisoformat(row[7]),
        status=row[8],
    )
