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

    async def upsert_candles(
        self, symbol: str, interval: str, candles: Sequence[Candle]
    ) -> None:
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
