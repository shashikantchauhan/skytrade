"""Hypothetical options trades shadowing BUY/SELL signals -- analysis only."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import libsql_client

from trading_scanner.domain.models import OptionsShadowTrade
from trading_scanner.infrastructure.db._shared import add_column_if_missing

_CREATE_OPTIONS_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS options_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    option_type TEXT NOT NULL,
    purpose TEXT NOT NULL,
    option_tradingsymbol TEXT NOT NULL,
    strike REAL NOT NULL,
    expiry TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    entry_timestamp TEXT NOT NULL,
    underlying_price_at_entry REAL NOT NULL,
    entry_premium REAL NOT NULL,
    exit_timestamp TEXT,
    underlying_price_at_exit REAL,
    exit_premium REAL,
    pnl_amount REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source TEXT NOT NULL DEFAULT 'live'
)
"""


class TursoOptionsTradeRepository:
    """Tracks hypothetical options trades shadowing BUY/SELL signals.

    A symbol can have up to two simultaneously open shadow option trades --
    a "directional" one and a "hedge" one (see ``domain.models.
    OptionsShadowTrade``) -- so every lookup/close is scoped by
    ``(symbol, option_type, purpose)``, not just symbol. Analysis only --
    see ``application/options_shadow.py``. Fully separate from the paper
    account's capital and from ``trades``' own equity-side scoring.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_OPTIONS_TRADES_TABLE)
        await add_column_if_missing(
            self._client, "options_trades", "source", "TEXT NOT NULL DEFAULT 'live'"
        )

    async def open_trade(self, trade: OptionsShadowTrade) -> None:
        await self._client.execute(
            """
            INSERT INTO options_trades
                (symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                 lot_size, entry_timestamp, underlying_price_at_entry, entry_premium, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                trade.symbol,
                trade.option_type,
                trade.purpose,
                trade.option_tradingsymbol,
                float(trade.strike),
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.underlying_price_at_entry),
                float(trade.entry_premium),
            ],
        )

    async def get_open_trade(
        self, symbol: str, option_type: str, purpose: str
    ) -> OptionsShadowTrade | None:
        result = await self._client.execute(
            """
            SELECT symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                   lot_size, entry_timestamp, underlying_price_at_entry, entry_premium
            FROM options_trades
            WHERE symbol = ? AND option_type = ? AND purpose = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol, option_type, purpose],
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return OptionsShadowTrade(
            symbol=row[0],
            option_type=row[1],
            purpose=row[2],
            option_tradingsymbol=row[3],
            strike=Decimal(str(row[4])),
            expiry=row[5],
            lot_size=int(row[6]),
            entry_timestamp=datetime.fromisoformat(row[7]),
            underlying_price_at_entry=Decimal(str(row[8])),
            entry_premium=Decimal(str(row[9])),
        )

    async def close_trade(
        self,
        symbol: str,
        option_type: str,
        purpose: str,
        exit_timestamp: datetime,
        underlying_price_at_exit: Decimal,
        exit_premium: Decimal,
        pnl_amount: Decimal,
        pnl_percent: Decimal,
    ) -> None:
        await self._client.execute(
            """
            UPDATE options_trades SET
                exit_timestamp = ?, underlying_price_at_exit = ?, exit_premium = ?,
                pnl_amount = ?, pnl_percent = ?, status = 'closed'
            WHERE id = (
                SELECT id FROM options_trades
                WHERE symbol = ? AND option_type = ? AND purpose = ? AND status = 'open'
                ORDER BY entry_timestamp DESC LIMIT 1
            )
            """,
            [
                exit_timestamp.isoformat(),
                float(underlying_price_at_exit),
                float(exit_premium),
                float(pnl_amount),
                float(pnl_percent),
                symbol,
                option_type,
                purpose,
            ],
        )

    async def delete_backtest_trades(self) -> None:
        """Clears every previous source='backtest' row before a fresh
        backtest run writes new ones -- without this, re-running the
        backtest (or running it again after logic changes) just appends on
        top of stale rows forever, silently mixing old and new results
        under the same label with no way to tell them apart (this is what
        was actually behind a "derivatives data looks wrong" report)."""
        await self._client.execute("DELETE FROM options_trades WHERE source = 'backtest'")

    async def insert_backtest_trade(self, trade: OptionsShadowTrade) -> None:
        """Inserts one already-closed row directly, source='backtest'.

        Unlike ``open_trade``/``close_trade`` (a two-step open-then-later-
        close flow scoped by the most recent open row for a key), a
        backtest replay knows both entry and exit up front and runs as a
        one-shot batch alongside the always-on live shadow-tracking flow --
        writing the whole row atomically avoids racing live's own
        open/close lookups for the same (symbol, option_type, purpose).
        """
        await self._client.execute(
            """
            INSERT INTO options_trades
                (symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                 lot_size, entry_timestamp, underlying_price_at_entry, entry_premium,
                 exit_timestamp, underlying_price_at_exit, exit_premium,
                 pnl_amount, pnl_percent, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', 'backtest')
            """,
            [
                trade.symbol,
                trade.option_type,
                trade.purpose,
                trade.option_tradingsymbol,
                float(trade.strike),
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.underlying_price_at_entry),
                float(trade.entry_premium),
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                float(trade.underlying_price_at_exit)
                if trade.underlying_price_at_exit is not None
                else None,
                float(trade.exit_premium) if trade.exit_premium is not None else None,
                float(trade.pnl_amount) if trade.pnl_amount is not None else None,
                float(trade.pnl_percent) if trade.pnl_percent is not None else None,
            ],
        )

    async def get_trades(
        self, symbol: str | None = None, source: str | None = None
    ) -> Sequence[OptionsShadowTrade]:
        query = """
            SELECT symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                   lot_size, entry_timestamp, underlying_price_at_entry, entry_premium,
                   exit_timestamp, underlying_price_at_exit, exit_premium,
                   pnl_amount, pnl_percent, status, source
            FROM options_trades
        """
        clauses: list[str] = []
        parameters: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            parameters.append(symbol)
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY entry_timestamp DESC"
        result = await self._client.execute(query, parameters)
        return [
            OptionsShadowTrade(
                symbol=row[0],
                option_type=row[1],
                purpose=row[2],
                option_tradingsymbol=row[3],
                strike=Decimal(str(row[4])),
                expiry=row[5],
                lot_size=int(row[6]),
                entry_timestamp=datetime.fromisoformat(row[7]),
                underlying_price_at_entry=Decimal(str(row[8])),
                entry_premium=Decimal(str(row[9])),
                exit_timestamp=datetime.fromisoformat(row[10]) if row[10] else None,
                underlying_price_at_exit=Decimal(str(row[11])) if row[11] is not None else None,
                exit_premium=Decimal(str(row[12])) if row[12] is not None else None,
                pnl_amount=Decimal(str(row[13])) if row[13] is not None else None,
                pnl_percent=Decimal(str(row[14])) if row[14] is not None else None,
                status=row[15],
                source=row[16],
            )
            for row in result.rows
        ]
