"""Persistent live state and decision ledger for the daily swing strategy."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import DailySwingPosition
from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS daily_swing_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL CHECK (side IN (-1, 1)),
    setup_date TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    initial_stop REAL NOT NULL,
    active_stop REAL NOT NULL,
    target REAL NOT NULL,
    atr REAL NOT NULL,
    risk REAL NOT NULL,
    best_close REAL NOT NULL,
    basket_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('entering', 'open', 'closed', 'rejected')),
    exit_timestamp TEXT,
    exit_price REAL,
    exit_reason TEXT,
    UNIQUE(symbol, setup_date)
)
"""

_CREATE_ATTEMPTS = """
CREATE TABLE IF NOT EXISTS daily_swing_attempts (
    symbol TEXT NOT NULL,
    setup_date TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    outcome TEXT NOT NULL,
    detail TEXT,
    PRIMARY KEY (symbol, setup_date)
)
"""


class DailySwingRepository:
    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_POSITIONS)
        await self._client.execute(_CREATE_ATTEMPTS)
        await self._client.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_swing_status ON daily_swing_positions(status)"
        )

    async def record_attempt(
        self, symbol: str, setup_date: str, attempted_at: datetime, outcome: str, detail: str = ""
    ) -> bool:
        result = await self._client.execute(
            """
            INSERT OR IGNORE INTO daily_swing_attempts
                (symbol, setup_date, attempted_at, outcome, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            [symbol, setup_date, attempted_at.isoformat(), outcome, detail],
        )
        return result.rows_affected > 0

    async def has_attempt(self, symbol: str, setup_date: str) -> bool:
        result = await self._client.execute(
            "SELECT 1 FROM daily_swing_attempts WHERE symbol = ? AND setup_date = ?",
            [symbol, setup_date],
        )
        return bool(result.rows)

    async def create_entering(self, position: DailySwingPosition) -> None:
        await self._client.execute(
            """
            INSERT INTO daily_swing_positions
                (symbol, side, setup_date, entry_timestamp, entry_price,
                 initial_stop, active_stop, target, atr, risk, best_close,
                 basket_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'entering')
            """,
            [
                position.symbol,
                position.side,
                position.setup_date,
                position.entry_timestamp.isoformat(),
                float(position.entry_price),
                float(position.initial_stop),
                float(position.active_stop),
                float(position.target),
                float(position.atr),
                float(position.risk),
                float(position.best_close),
                position.basket_id,
            ],
        )

    async def mark_open(self, symbol: str, setup_date: str, basket_id: str) -> None:
        await self._client.execute(
            """UPDATE daily_swing_positions SET status = 'open', basket_id = ?
               WHERE symbol = ? AND setup_date = ? AND status = 'entering'""",
            [basket_id, symbol, setup_date],
        )

    async def reject_entering(self, symbol: str, setup_date: str, reason: str) -> None:
        await self._client.execute(
            """UPDATE daily_swing_positions SET status = 'rejected', exit_reason = ?
               WHERE symbol = ? AND setup_date = ? AND status = 'entering'""",
            [reason, symbol, setup_date],
        )

    async def get_active(self) -> Sequence[DailySwingPosition]:
        result = await self._client.execute(
            _SELECT + " WHERE status IN ('entering', 'open') ORDER BY id"
        )
        return [_row(row) for row in result.rows]

    async def update_trail(self, symbol: str, active_stop: Decimal, best_close: Decimal) -> None:
        await self._client.execute(
            """UPDATE daily_swing_positions SET active_stop = ?, best_close = ?
               WHERE symbol = ? AND status = 'open'""",
            [float(active_stop), float(best_close), symbol],
        )

    async def close(self, symbol: str, timestamp: datetime, price: Decimal, reason: str) -> None:
        await self._client.execute(
            """UPDATE daily_swing_positions
               SET status = 'closed', exit_timestamp = ?, exit_price = ?, exit_reason = ?
               WHERE symbol = ? AND status = 'open'""",
            [timestamp.isoformat(), float(price), reason, symbol],
        )


_SELECT = """
SELECT symbol, side, setup_date, entry_timestamp, entry_price, initial_stop,
       active_stop, target, atr, risk, best_close, basket_id, status,
       exit_timestamp, exit_price, exit_reason
FROM daily_swing_positions
"""


def _row(row: Sequence) -> DailySwingPosition:
    return DailySwingPosition(
        symbol=row[0],
        side=int(row[1]),
        setup_date=row[2],
        entry_timestamp=datetime.fromisoformat(row[3]),
        entry_price=Decimal(str(row[4])),
        initial_stop=Decimal(str(row[5])),
        active_stop=Decimal(str(row[6])),
        target=Decimal(str(row[7])),
        atr=Decimal(str(row[8])),
        risk=Decimal(str(row[9])),
        best_close=Decimal(str(row[10])),
        basket_id=row[11],
        status=row[12],
        exit_timestamp=datetime.fromisoformat(row[13]) if row[13] else None,
        exit_price=Decimal(str(row[14])) if row[14] is not None else None,
        exit_reason=row[15],
    )
