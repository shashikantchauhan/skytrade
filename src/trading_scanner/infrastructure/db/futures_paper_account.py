"""The futures paper account: its own margin pool and combo positions --
separate book from the cash paper account (see paper_account.py)."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import libsql_client

from trading_scanner.domain.models import FuturesPaperPosition

_CREATE_FUTURES_PAPER_ACCOUNT_TABLE = """
CREATE TABLE IF NOT EXISTS futures_paper_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL
)
"""

_CREATE_FUTURES_PAPER_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS futures_paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    futures_entry_price REAL NOT NULL,
    futures_tradingsymbol TEXT NOT NULL,
    hedge_tradingsymbol TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    margin_allocated REAL NOT NULL,
    exit_timestamp TEXT,
    futures_exit_price REAL,
    pnl_amount REAL,
    status TEXT NOT NULL DEFAULT 'open'
)
"""


class TursoFuturesPaperAccountRepository:
    """Persists the futures paper account's own capital pool and combo
    positions -- separate book from ``TursoPaperAccountRepository``'s cash
    pool (see ``domain/models.py``'s ``FuturesPaperPosition`` docstring for
    why). Same one-row-account pattern as the cash paper account."""

    def __init__(self, client: libsql_client.Client, initial_capital: Decimal) -> None:
        self._client = client
        self._initial_capital = initial_capital

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_FUTURES_PAPER_ACCOUNT_TABLE)
        await self._client.execute(_CREATE_FUTURES_PAPER_POSITIONS_TABLE)

    async def get_cash_balance(self) -> Decimal:
        result = await self._client.execute(
            "SELECT cash_balance FROM futures_paper_account WHERE id = 1"
        )
        if not result.rows:
            await self._client.execute(
                "INSERT INTO futures_paper_account (id, cash_balance) VALUES (1, ?)",
                [float(self._initial_capital)],
            )
            return self._initial_capital
        return Decimal(str(result.rows[0][0]))

    async def open_position(self, position: FuturesPaperPosition) -> None:
        await self.get_cash_balance()  # Ensures the account row exists.
        await self._client.execute(
            """
            INSERT INTO futures_paper_positions
                (symbol, side, entry_timestamp, futures_entry_price, futures_tradingsymbol,
                 hedge_tradingsymbol, lot_size, margin_allocated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                position.symbol,
                position.side,
                position.entry_timestamp.isoformat(),
                float(position.futures_entry_price),
                position.futures_tradingsymbol,
                position.hedge_tradingsymbol,
                position.lot_size,
                float(position.margin_allocated),
            ],
        )
        await self._client.execute(
            "UPDATE futures_paper_account SET cash_balance = cash_balance - ? WHERE id = 1",
            [float(position.margin_allocated)],
        )

    async def close_position(
        self, symbol: str, exit_timestamp: datetime, futures_exit_price: Decimal
    ) -> FuturesPaperPosition | None:
        """Close the most recent open combo for ``symbol``, crediting cash
        back its margin plus/minus the futures leg's own P&L only -- the
        hedge option leg's P&L is tracked separately by
        ``options_shadow.py`` (``purpose="hedge"``), not folded in here.
        """
        result = await self._client.execute(
            """
            SELECT id, side, entry_timestamp, futures_entry_price, futures_tradingsymbol,
                   hedge_tradingsymbol, lot_size, margin_allocated
            FROM futures_paper_positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol],
        )
        if not result.rows:
            return None
        (
            position_id, side, entry_timestamp, futures_entry_price, futures_tradingsymbol,
            hedge_tradingsymbol, lot_size, margin_allocated,
        ) = result.rows[0]
        futures_entry_price = Decimal(str(futures_entry_price))
        margin_allocated = Decimal(str(margin_allocated))
        pnl_amount = (
            (futures_exit_price - futures_entry_price) * lot_size
            if side == "long"
            else (futures_entry_price - futures_exit_price) * lot_size
        )
        proceeds = margin_allocated + pnl_amount
        await self._client.execute(
            """
            UPDATE futures_paper_positions
            SET exit_timestamp = ?, futures_exit_price = ?, pnl_amount = ?, status = 'closed'
            WHERE id = ?
            """,
            [
                exit_timestamp.isoformat(), float(futures_exit_price), float(pnl_amount),
                position_id,
            ],
        )
        await self._client.execute(
            "UPDATE futures_paper_account SET cash_balance = cash_balance + ? WHERE id = 1",
            [float(proceeds)],
        )
        return FuturesPaperPosition(
            symbol=symbol,
            side=side,
            entry_timestamp=datetime.fromisoformat(entry_timestamp),
            futures_entry_price=futures_entry_price,
            futures_tradingsymbol=futures_tradingsymbol,
            hedge_tradingsymbol=hedge_tradingsymbol,
            lot_size=lot_size,
            margin_allocated=margin_allocated,
            exit_timestamp=exit_timestamp,
            futures_exit_price=futures_exit_price,
            pnl_amount=pnl_amount,
            status="closed",
        )

    async def get_open_positions(self) -> Sequence[FuturesPaperPosition]:
        result = await self._client.execute(
            """
            SELECT symbol, side, entry_timestamp, futures_entry_price, futures_tradingsymbol,
                   hedge_tradingsymbol, lot_size, margin_allocated
            FROM futures_paper_positions WHERE status = 'open'
            """
        )
        return [
            FuturesPaperPosition(
                symbol=row[0],
                side=row[1],
                entry_timestamp=datetime.fromisoformat(row[2]),
                futures_entry_price=Decimal(str(row[3])),
                futures_tradingsymbol=row[4],
                hedge_tradingsymbol=row[5],
                lot_size=row[6],
                margin_allocated=Decimal(str(row[7])),
            )
            for row in result.rows
        ]
