"""Hypothetical futures trades shadowing BUY/SELL signals -- analysis only."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import libsql_client

from trading_scanner.domain.models import FuturesShadowTrade
from trading_scanner.infrastructure.db._shared import add_column_if_missing

_CREATE_FUTURES_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS futures_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    futures_tradingsymbol TEXT NOT NULL,
    expiry TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_amount REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source TEXT NOT NULL DEFAULT 'live',
    purpose TEXT NOT NULL DEFAULT 'primary'
)
"""


class TursoFuturesTradeRepository:
    """Tracks hypothetical futures trades shadowing BUY/SELL signals.

    Two purposes can be open per symbol at once -- ``purpose="primary"``
    (the futures position is the trade) and ``purpose="hedge"`` (it hedges
    a directional option instead, see ``domain.models.FuturesShadowTrade``)
    -- so every lookup/close is scoped by ``(symbol, purpose)``, not just
    symbol. Analysis only -- see ``application/futures_shadow.py``. Fully
    separate from the paper account's capital.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_FUTURES_TRADES_TABLE)
        await add_column_if_missing(
            self._client, "futures_trades", "source", "TEXT NOT NULL DEFAULT 'live'"
        )
        await add_column_if_missing(
            self._client, "futures_trades", "purpose", "TEXT NOT NULL DEFAULT 'primary'"
        )

    async def open_trade(self, trade: FuturesShadowTrade) -> None:
        await self._client.execute(
            """
            INSERT INTO futures_trades
                (symbol, side, futures_tradingsymbol, expiry, lot_size,
                 entry_timestamp, entry_price, purpose, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                trade.symbol,
                trade.side,
                trade.futures_tradingsymbol,
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.purpose,
            ],
        )

    async def get_open_trade(
        self, symbol: str, purpose: str = "primary"
    ) -> FuturesShadowTrade | None:
        result = await self._client.execute(
            """
            SELECT symbol, side, futures_tradingsymbol, expiry, lot_size,
                   entry_timestamp, entry_price, purpose
            FROM futures_trades WHERE symbol = ? AND purpose = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol, purpose],
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return FuturesShadowTrade(
            symbol=row[0],
            side=row[1],
            futures_tradingsymbol=row[2],
            expiry=row[3],
            lot_size=int(row[4]),
            entry_timestamp=datetime.fromisoformat(row[5]),
            entry_price=Decimal(str(row[6])),
            purpose=row[7],
        )

    async def close_trade(
        self,
        symbol: str,
        exit_timestamp: datetime,
        exit_price: Decimal,
        pnl_amount: Decimal,
        pnl_percent: Decimal,
        purpose: str = "primary",
    ) -> None:
        await self._client.execute(
            """
            UPDATE futures_trades SET
                exit_timestamp = ?, exit_price = ?, pnl_amount = ?, pnl_percent = ?,
                status = 'closed'
            WHERE id = (
                SELECT id FROM futures_trades
                WHERE symbol = ? AND purpose = ? AND status = 'open'
                ORDER BY entry_timestamp DESC LIMIT 1
            )
            """,
            [
                exit_timestamp.isoformat(),
                float(exit_price),
                float(pnl_amount),
                float(pnl_percent),
                symbol,
                purpose,
            ],
        )

    async def delete_backtest_trades(self) -> None:
        """Clears every previous source='backtest' row -- see
        ``TursoOptionsTradeRepository.delete_backtest_trades``."""
        await self._client.execute("DELETE FROM futures_trades WHERE source = 'backtest'")

    async def insert_backtest_trade(self, trade: FuturesShadowTrade) -> None:
        """Inserts one already-closed row directly, source='backtest' --
        see ``TursoOptionsTradeRepository.insert_backtest_trade`` for why
        this bypasses the open/close two-step."""
        await self._client.execute(
            """
            INSERT INTO futures_trades
                (symbol, side, futures_tradingsymbol, expiry, lot_size,
                 entry_timestamp, entry_price, exit_timestamp, exit_price,
                 pnl_amount, pnl_percent, purpose, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', 'backtest')
            """,
            [
                trade.symbol,
                trade.side,
                trade.futures_tradingsymbol,
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                float(trade.exit_price) if trade.exit_price is not None else None,
                float(trade.pnl_amount) if trade.pnl_amount is not None else None,
                float(trade.pnl_percent) if trade.pnl_percent is not None else None,
                trade.purpose,
            ],
        )

    async def get_trades(
        self, symbol: str | None = None, source: str | None = None, purpose: str | None = None
    ) -> Sequence[FuturesShadowTrade]:
        query = """
            SELECT symbol, side, futures_tradingsymbol, expiry, lot_size,
                   entry_timestamp, entry_price, exit_timestamp, exit_price,
                   pnl_amount, pnl_percent, status, source, purpose
            FROM futures_trades
        """
        clauses: list[str] = []
        parameters: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            parameters.append(symbol)
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        if purpose is not None:
            clauses.append("purpose = ?")
            parameters.append(purpose)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY entry_timestamp DESC"
        result = await self._client.execute(query, parameters)
        return [
            FuturesShadowTrade(
                symbol=row[0],
                side=row[1],
                futures_tradingsymbol=row[2],
                expiry=row[3],
                lot_size=int(row[4]),
                entry_timestamp=datetime.fromisoformat(row[5]),
                entry_price=Decimal(str(row[6])),
                exit_timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                exit_price=Decimal(str(row[8])) if row[8] is not None else None,
                pnl_amount=Decimal(str(row[9])) if row[9] is not None else None,
                pnl_percent=Decimal(str(row[10])) if row[10] is not None else None,
                status=row[11],
                source=row[12],
                purpose=row[13],
            )
            for row in result.rows
        ]
