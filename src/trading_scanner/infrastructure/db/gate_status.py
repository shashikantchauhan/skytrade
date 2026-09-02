"""Every symbol's latest gate state, refreshed every pipeline cycle -- see
domain/models.py's GateStatusSnapshot and docs/decisions/008-gate-status-
snapshot.md for why this exists (the dashboard's gate-transparency table)."""

from datetime import datetime

from trading_scanner.domain.models import GateStatusSnapshot
from trading_scanner.infrastructure.db._shared import DbClient

_CREATE_GATE_STATUS_TABLE = """
CREATE TABLE IF NOT EXISTS gate_status (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    signal TEXT NOT NULL,
    adx REAL NOT NULL,
    regime_normalized REAL NOT NULL,
    volatility_margin REAL NOT NULL,
    track_record_passed INTEGER NOT NULL,
    quality_passed INTEGER NOT NULL,
    conviction_passed INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, interval)
)
"""


class TursoGateStatusRepository:
    """Persists one always-overwritten row per symbol -- see
    GateStatusSnapshot's own docstring for why this is a snapshot table,
    not an append-only log like entry_decisions."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_GATE_STATUS_TABLE)

    async def set_snapshot(self, snapshot: GateStatusSnapshot) -> None:
        await self._client.execute(
            """
            INSERT INTO gate_status
                (symbol, interval, signal, adx, regime_normalized, volatility_margin,
                 track_record_passed, quality_passed, conviction_passed,
                 evaluated_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval) DO UPDATE SET
                signal = excluded.signal,
                adx = excluded.adx,
                regime_normalized = excluded.regime_normalized,
                volatility_margin = excluded.volatility_margin,
                track_record_passed = excluded.track_record_passed,
                quality_passed = excluded.quality_passed,
                conviction_passed = excluded.conviction_passed,
                evaluated_at = excluded.evaluated_at,
                updated_at = excluded.updated_at
            """,
            [
                snapshot.symbol,
                snapshot.interval,
                snapshot.signal,
                snapshot.adx,
                snapshot.regime_normalized,
                snapshot.volatility_margin,
                int(snapshot.track_record_passed),
                int(snapshot.quality_passed),
                int(snapshot.conviction_passed),
                snapshot.evaluated_at.isoformat(),
                snapshot.updated_at.isoformat(),
            ],
        )

    async def get_all_snapshots(self, interval: str) -> list[GateStatusSnapshot]:
        """Every symbol's latest snapshot for ``interval``, most recently
        evaluated first -- the dashboard's whole Gates tab in one query."""
        result = await self._client.execute(
            """
            SELECT symbol, interval, signal, adx, regime_normalized, volatility_margin,
                   track_record_passed, quality_passed, conviction_passed,
                   evaluated_at, updated_at
            FROM gate_status
            WHERE interval = ?
            ORDER BY evaluated_at DESC
            """,
            [interval],
        )
        return [
            GateStatusSnapshot(
                symbol=row[0],
                interval=row[1],
                signal=row[2],
                adx=row[3],
                regime_normalized=row[4],
                volatility_margin=row[5],
                track_record_passed=bool(row[6]),
                quality_passed=bool(row[7]),
                conviction_passed=bool(row[8]),
                evaluated_at=datetime.fromisoformat(row[9]),
                updated_at=datetime.fromisoformat(row[10]),
            )
            for row in result.rows
        ]
