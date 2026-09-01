"""Strategy trade history -- entries/exits for win-rate and backtest analysis."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import SignalSide, Trade
from trading_scanner.infrastructure.db._shared import DbClient, add_column_if_missing

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    prediction_at_entry INTEGER NOT NULL,
    is_early_signal_flip INTEGER NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    adx_at_entry REAL,
    regime_normalized_at_entry REAL,
    volatility_margin_at_entry REAL,
    volatility_filter_passed INTEGER,
    regime_filter_passed INTEGER,
    adx_filter_passed INTEGER
)
"""


class TursoTradeRepository:
    """Tracks entries/exits for win-rate and backtest analysis."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the trades table if it does not already exist, and migrate it forward."""
        await self._client.execute(_CREATE_TRADES_TABLE)
        for column, definition in (
            ("adx_at_entry", "REAL"),
            ("regime_normalized_at_entry", "REAL"),
            ("volatility_margin_at_entry", "REAL"),
            ("volatility_filter_passed", "INTEGER"),
            ("regime_filter_passed", "INTEGER"),
            ("adx_filter_passed", "INTEGER"),
        ):
            await add_column_if_missing(self._client, "trades", column, definition)
        # 2026-09-01: every query below filters by (symbol, interval,
        # side, status='open') or (interval) alone -- this table only
        # grows (every paper/win-rate trade ever taken), unindexed beyond
        # the implicit rowid.
        await self._client.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_interval_side_status "
            "ON trades (symbol, interval, side, status)"
        )

    async def open_trade(self, interval: str, trade: Trade) -> None:
        await self._client.execute(
            """
            INSERT INTO trades
                (symbol, interval, side, entry_timestamp, entry_price,
                 prediction_at_entry, is_early_signal_flip, status,
                 adx_at_entry, regime_normalized_at_entry, volatility_margin_at_entry,
                 volatility_filter_passed, regime_filter_passed, adx_filter_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
            """,
            [
                trade.symbol,
                interval,
                trade.side.value,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.prediction_at_entry,
                int(trade.is_early_signal_flip),
                trade.adx_at_entry,
                trade.regime_normalized_at_entry,
                trade.volatility_margin_at_entry,
                (
                    None
                    if trade.volatility_filter_passed is None
                    else int(trade.volatility_filter_passed)
                ),
                None if trade.regime_filter_passed is None else int(trade.regime_filter_passed),
                None if trade.adx_filter_passed is None else int(trade.adx_filter_passed),
            ],
        )

    async def close_open_trade(
        self,
        symbol: str,
        interval: str,
        side: SignalSide,
        exit_timestamp: datetime,
        exit_price: Decimal,
    ) -> None:
        """Close the most recent open trade for (symbol, interval, side).

        Long (BUY) profit is exit-minus-entry; short (SELL) profit is
        entry-minus-exit -- matches the Pine backtest helper's own
        long/short profit formulas (MLExtensions_v2.pine's ``ml.backtest``).
        """
        result = await self._client.execute(
            """
            SELECT id, entry_price FROM trades
            WHERE symbol = ? AND interval = ? AND side = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol, interval, side.value],
        )
        if not result.rows:
            return
        trade_id, entry_price = result.rows[0]
        entry_price = Decimal(str(entry_price))
        pnl_percent = (
            (exit_price - entry_price) / entry_price * 100
            if side == SignalSide.BUY
            else (entry_price - exit_price) / entry_price * 100
        )
        await self._client.execute(
            """
            UPDATE trades SET exit_timestamp = ?, exit_price = ?, pnl_percent = ?, status = 'closed'
            WHERE id = ?
            """,
            [exit_timestamp.isoformat(), float(exit_price), float(pnl_percent), trade_id],
        )

    async def abandon_open_trade(self, symbol: str, interval: str, side: SignalSide) -> None:
        """Mark a still-open trade abandoned, excluded from win-rate stats.

        Mirrors Pine's ``ml.backtest``: a new opposite-side entry discards
        whatever position was still open without scoring it -- not closed,
        not a win, not a loss.
        """
        await self._client.execute(
            """
            UPDATE trades SET status = 'abandoned'
            WHERE symbol = ? AND interval = ? AND side = ? AND status = 'open'
            """,
            [symbol, interval, side.value],
        )

    async def get_trades(self, symbol: str | None, interval: str) -> Sequence[Trade]:
        query = """
            SELECT symbol, side, entry_timestamp, entry_price, prediction_at_entry,
                   is_early_signal_flip, exit_timestamp, exit_price, pnl_percent, status,
                   adx_at_entry, regime_normalized_at_entry, volatility_margin_at_entry,
                   volatility_filter_passed, regime_filter_passed, adx_filter_passed
            FROM trades WHERE interval = ?
        """
        parameters = [interval]
        if symbol is not None:
            query += " AND symbol = ?"
            parameters.append(symbol)
        query += " ORDER BY entry_timestamp ASC"
        result = await self._client.execute(query, parameters)
        return [
            Trade(
                symbol=row[0],
                side=SignalSide(row[1]),
                entry_timestamp=datetime.fromisoformat(row[2]),
                entry_price=Decimal(str(row[3])),
                prediction_at_entry=int(row[4]),
                is_early_signal_flip=bool(row[5]),
                exit_timestamp=datetime.fromisoformat(row[6]) if row[6] else None,
                exit_price=Decimal(str(row[7])) if row[7] is not None else None,
                pnl_percent=Decimal(str(row[8])) if row[8] is not None else None,
                status=row[9],
                adx_at_entry=row[10],
                regime_normalized_at_entry=row[11],
                volatility_margin_at_entry=row[12],
                volatility_filter_passed=None if row[13] is None else bool(row[13]),
                regime_filter_passed=None if row[14] is None else bool(row[14]),
                adx_filter_passed=None if row[15] is None else bool(row[15]),
            )
            for row in result.rows
        ]
