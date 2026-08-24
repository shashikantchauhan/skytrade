"""Paper-simulated benchmark run 1:1 alongside every real live-cash trade
-- see application/paper_benchmark.py and application/live_cash_execution.py."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import PaperBenchmarkPosition
from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_PAPER_BENCHMARK_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS paper_benchmark_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    basket_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_timestamp TEXT NOT NULL,
    paper_entry_price REAL NOT NULL,
    real_entry_price REAL NOT NULL,
    exit_timestamp TEXT,
    paper_exit_price REAL,
    real_exit_price REAL,
    paper_pnl_amount REAL,
    real_pnl_amount REAL,
    status TEXT NOT NULL DEFAULT 'open'
)
"""


class TursoPaperBenchmarkRepository:
    """One row per real live-cash entry/exit pair, tracking the paper
    ("no friction") fill alongside Kite's real fill for the same basket."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_PAPER_BENCHMARK_POSITIONS_TABLE)

    async def open_position(self, position: PaperBenchmarkPosition) -> None:
        await self._client.execute(
            """
            INSERT INTO paper_benchmark_positions
                (symbol, basket_id, quantity, entry_timestamp,
                 paper_entry_price, real_entry_price, status)
            VALUES (?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                position.symbol,
                position.basket_id,
                position.quantity,
                position.entry_timestamp.isoformat(),
                float(position.paper_entry_price),
                float(position.real_entry_price),
            ],
        )

    async def close_position(
        self,
        symbol: str,
        basket_id: str,
        exit_timestamp: datetime,
        paper_exit_price: Decimal,
        real_exit_price: Decimal,
    ) -> None:
        """Closes the one open row for ``(symbol, basket_id)`` -- basket_id
        is the real entry's own basket_id, captured by the caller while the
        position was still open, so this is an exact match, not a
        "most recent open row" inference. Silently no-ops if there's no
        matching open row (e.g. already closed, or never opened)."""
        result = await self._client.execute(
            """
            SELECT paper_entry_price, real_entry_price, quantity
            FROM paper_benchmark_positions
            WHERE symbol = ? AND basket_id = ? AND status = 'open'
            """,
            [symbol, basket_id],
        )
        if not result.rows:
            return
        paper_entry_price, real_entry_price, quantity = result.rows[0]
        paper_pnl = (float(paper_exit_price) - paper_entry_price) * quantity
        real_pnl = (float(real_exit_price) - real_entry_price) * quantity
        await self._client.execute(
            """
            UPDATE paper_benchmark_positions
            SET exit_timestamp = ?, paper_exit_price = ?, real_exit_price = ?,
                paper_pnl_amount = ?, real_pnl_amount = ?, status = 'closed'
            WHERE symbol = ? AND basket_id = ? AND status = 'open'
            """,
            [
                exit_timestamp.isoformat(),
                float(paper_exit_price),
                float(real_exit_price),
                paper_pnl,
                real_pnl,
                symbol,
                basket_id,
            ],
        )

    async def get_open_positions(self) -> Sequence[PaperBenchmarkPosition]:
        result = await self._client.execute(
            """
            SELECT symbol, basket_id, quantity, entry_timestamp,
                   paper_entry_price, real_entry_price,
                   exit_timestamp, paper_exit_price, real_exit_price,
                   paper_pnl_amount, real_pnl_amount, status
            FROM paper_benchmark_positions WHERE status = 'open'
            ORDER BY entry_timestamp DESC
            """
        )
        return [_row_to_position(row) for row in result.rows]

    async def get_recent_closed_positions(self, limit: int) -> Sequence[PaperBenchmarkPosition]:
        result = await self._client.execute(
            """
            SELECT symbol, basket_id, quantity, entry_timestamp,
                   paper_entry_price, real_entry_price,
                   exit_timestamp, paper_exit_price, real_exit_price,
                   paper_pnl_amount, real_pnl_amount, status
            FROM paper_benchmark_positions WHERE status = 'closed'
            ORDER BY exit_timestamp DESC LIMIT ?
            """,
            [limit],
        )
        return [_row_to_position(row) for row in result.rows]


def _row_to_position(row) -> PaperBenchmarkPosition:
    return PaperBenchmarkPosition(
        symbol=row[0],
        basket_id=row[1],
        quantity=int(row[2]),
        entry_timestamp=datetime.fromisoformat(row[3]),
        paper_entry_price=Decimal(str(row[4])),
        real_entry_price=Decimal(str(row[5])),
        exit_timestamp=datetime.fromisoformat(row[6]) if row[6] else None,
        paper_exit_price=Decimal(str(row[7])) if row[7] is not None else None,
        real_exit_price=Decimal(str(row[8])) if row[8] is not None else None,
        paper_pnl_amount=Decimal(str(row[9])) if row[9] is not None else None,
        real_pnl_amount=Decimal(str(row[10])) if row[10] is not None else None,
        status=row[11],
    )
