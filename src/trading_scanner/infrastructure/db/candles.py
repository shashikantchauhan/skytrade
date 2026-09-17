"""Accumulated OHLCV candle storage."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.domain.models import Candle
from trading_scanner.infrastructure.db._shared import DbClient, Statement

_CREATE_CANDLES_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, interval, timestamp)
)
"""

_UPSERT_CANDLE = """
INSERT INTO candles (symbol, interval, timestamp, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (symbol, interval, timestamp) DO UPDATE SET
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume
"""

_SELECT_CANDLES = """
SELECT timestamp, open, high, low, close, volume FROM candles
WHERE symbol = ? AND interval = ?
ORDER BY timestamp DESC
"""


class TursoCandleRepository:
    """Persist and retrieve accumulated OHLCV candles in Turso/libSQL."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the candles table if it does not already exist."""
        await self._client.execute(_CREATE_CANDLES_TABLE)

    async def upsert_candles(self, symbol: str, interval: str, candles: Sequence[Candle]) -> None:
        """Insert new candles or refresh existing ones for the same bar.

        Timestamps are normalized to UTC here as a second line of defense
        (callers should already do this -- see
        ``signal_pipeline._dataframe_to_candles``) -- storing any other
        offset produces a text timestamp that sorts incorrectly against
        UTC-stored rows under this table's plain ``ORDER BY timestamp``,
        scrambling chronological order for every downstream reader.
        """
        if not candles:
            return
        statements = [
            Statement(
                _UPSERT_CANDLE,
                [
                    symbol,
                    interval,
                    candle.timestamp.astimezone(UTC).isoformat(),
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    candle.volume,
                ],
            )
            for candle in candles
        ]
        await self._client.batch(statements)

    async def upsert_many(self, interval: str, candles: Sequence[Candle]) -> None:
        """Store one cross-symbol completed-bucket batch in one transaction."""
        if not candles:
            return
        statements = [
            Statement(
                _UPSERT_CANDLE,
                [
                    candle.symbol,
                    interval,
                    candle.timestamp.astimezone(UTC).isoformat(),
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    candle.volume,
                ],
            )
            for candle in candles
        ]
        await self._client.batch(statements)

    async def get_interval_rows_since(self, interval: str, since: datetime) -> list[tuple]:
        result = await self._client.execute(
            """SELECT symbol, timestamp, open, high, low, close, volume
               FROM candles WHERE interval = ? AND timestamp >= ?
               ORDER BY symbol, timestamp""",
            [interval, since.astimezone(UTC).isoformat()],
        )
        return list(result.rows)

    async def get_daily_aggregates(
        self, interval: str, since: datetime, before: datetime
    ) -> list[tuple]:
        """Aggregate intraday history inside SQLite to keep live-runner RAM bounded."""
        result = await self._client.execute(
            """
            WITH base AS (
                SELECT symbol, substr(timestamp, 1, 10) AS session_date,
                       timestamp, open, high, low, close, volume
                FROM candles
                WHERE interval = ? AND timestamp >= ? AND timestamp < ?
            ), ranked AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY symbol, session_date ORDER BY timestamp
                       ) AS first_row,
                       row_number() OVER (
                           PARTITION BY symbol, session_date ORDER BY timestamp DESC
                       ) AS last_row
                FROM base
            )
            SELECT symbol, session_date,
                   max(CASE WHEN first_row = 1 THEN open END) AS open,
                   max(high), min(low),
                   max(CASE WHEN last_row = 1 THEN close END) AS close,
                   sum(volume), count(*)
            FROM ranked
            GROUP BY symbol, session_date
            HAVING count(*) >= 10
            ORDER BY symbol, session_date
            """,
            [
                interval,
                since.astimezone(UTC).isoformat(),
                before.astimezone(UTC).isoformat(),
            ],
        )
        return list(result.rows)

    async def get_volume_rows_since(
        self, interval: str, since: datetime, before: datetime
    ) -> list[tuple]:
        result = await self._client.execute(
            """SELECT symbol, timestamp, volume FROM candles
               WHERE interval = ? AND timestamp >= ? AND timestamp < ?
               ORDER BY symbol, timestamp""",
            [
                interval,
                since.astimezone(UTC).isoformat(),
                before.astimezone(UTC).isoformat(),
            ],
        )
        return list(result.rows)

    async def get_latest_interval_timestamp(self, interval: str) -> datetime | None:
        result = await self._client.execute(
            "SELECT max(timestamp) FROM candles WHERE interval = ?", [interval]
        )
        value = result.rows[0][0] if result.rows else None
        return datetime.fromisoformat(value) if value else None

    async def count_sessions(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> int:
        result = await self._client.execute(
            """SELECT COUNT(DISTINCT substr(timestamp, 1, 10)) FROM candles
               WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?""",
            [symbol, interval, start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()],
        )
        return int(result.rows[0][0]) if result.rows else 0

    async def get_bucket_opens(self, interval: str, timestamp: datetime) -> dict[str, Decimal]:
        result = await self._client.execute(
            "SELECT symbol, open FROM candles WHERE interval = ? AND timestamp = ?",
            [interval, timestamp.astimezone(UTC).isoformat()],
        )
        return {row[0]: Decimal(str(row[1])) for row in result.rows}

    async def get_candles(
        self, symbol: str, interval: str, limit: int | None = None
    ) -> Sequence[Candle]:
        """Return chronological accumulated candles, most recent `limit` rows."""
        query = _SELECT_CANDLES + (" LIMIT ?" if limit is not None else "")
        parameters = [symbol, interval] + ([limit] if limit is not None else [])
        result = await self._client.execute(query, parameters)
        candles = [
            Candle(
                symbol=symbol,
                timestamp=datetime.fromisoformat(row[0]),
                open=Decimal(str(row[1])),
                high=Decimal(str(row[2])),
                low=Decimal(str(row[3])),
                close=Decimal(str(row[4])),
                volume=int(row[5]),
            )
            for row in result.rows
        ]
        return list(reversed(candles))  # Query is newest-first; callers need chronological order.
