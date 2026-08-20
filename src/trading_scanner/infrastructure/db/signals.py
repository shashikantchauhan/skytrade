"""Sent-signal fingerprint dedup, so the same signal is never notified twice."""

from datetime import UTC, datetime

from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS sent_signals (
    fingerprint TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
)
"""


class TursoSignalRepository:
    """Track which signal fingerprints have already been notified."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the sent_signals table if it does not already exist."""
        await self._client.execute(_CREATE_SIGNALS_TABLE)

    async def contains(self, fingerprint: str) -> bool:
        result = await self._client.execute(
            "SELECT 1 FROM sent_signals WHERE fingerprint = ?", [fingerprint]
        )
        return len(result.rows) > 0

    async def record(self, fingerprint: str, created_at: datetime) -> None:
        await self._client.execute(
            "INSERT OR IGNORE INTO sent_signals (fingerprint, created_at) VALUES (?, ?)",
            [fingerprint, created_at.astimezone(UTC).isoformat()],
        )
