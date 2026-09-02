"""Every symbol's gate state -- see domain/models.py's GateStatusSnapshot
and docs/decisions/008-gate-status-snapshot.md for why this exists (the
dashboard's gate-transparency table)."""

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

# 2026-09-02: gate_status above is a single always-overwritten row per
# symbol -- "what does the system see right now," nothing more. It cannot
# answer "what fired today," since a symbol's BUY/SELL only shows for the
# one bar it actually changed on -- every bar after that reads back as
# NEUTRAL again even though a real signal already happened. Confirmed live
# the same day: a snapshot-only view undercounted 15 real signals down to
# 1. This append-only table is the fix -- one permanent row per actual
# BUY/SELL event (never for NEUTRAL, so it stays small), for both sides
# (unlike entry_decisions, which only ever logs the BUY side).
_CREATE_GATE_STATUS_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS gate_status_events (
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
    PRIMARY KEY (symbol, interval, evaluated_at)
)
"""

_COLUMNS = (
    "symbol, interval, signal, adx, regime_normalized, volatility_margin, "
    "track_record_passed, quality_passed, conviction_passed, evaluated_at, updated_at"
)


def _row_to_snapshot(row) -> GateStatusSnapshot:
    return GateStatusSnapshot(
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


def _snapshot_params(snapshot: GateStatusSnapshot) -> list:
    return [
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
    ]


class TursoGateStatusRepository:
    """``gate_status``: one always-overwritten row per symbol (what the
    system sees right now). ``gate_status_events``: one permanent row per
    actual BUY/SELL that ever fired (what happened, queryable by day) --
    see that table's own comment above for why both exist."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_GATE_STATUS_TABLE)
        await self._client.execute(_CREATE_GATE_STATUS_EVENTS_TABLE)

    async def set_snapshot(self, snapshot: GateStatusSnapshot) -> None:
        await self._client.execute(
            f"""
            INSERT INTO gate_status ({_COLUMNS})
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
            _snapshot_params(snapshot),
        )

    async def get_all_snapshots(self, interval: str) -> list[GateStatusSnapshot]:
        """Every symbol's latest snapshot for ``interval``, most recently
        evaluated first."""
        result = await self._client.execute(
            f"SELECT {_COLUMNS} FROM gate_status WHERE interval = ? ORDER BY evaluated_at DESC",
            [interval],
        )
        return [_row_to_snapshot(row) for row in result.rows]

    async def record_event(self, snapshot: GateStatusSnapshot) -> None:
        """Append one BUY/SELL event -- never call this for a NEUTRAL
        snapshot (the caller's job to filter; this table has no opinion,
        it just stores what it's given). ``ON CONFLICT ... DO NOTHING``:
        the same (symbol, interval, evaluated_at) bar re-evaluated (e.g.
        catch-up replay re-processing something already recorded) must
        not duplicate or clobber the original event."""
        await self._client.execute(
            f"""
            INSERT INTO gate_status_events ({_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval, evaluated_at) DO NOTHING
            """,
            _snapshot_params(snapshot),
        )

    async def get_events_since(self, interval: str, since: datetime) -> list[GateStatusSnapshot]:
        """Every BUY/SELL event at or after ``since`` (typically midnight
        IST of "today") -- most recent first."""
        result = await self._client.execute(
            f"""
            SELECT {_COLUMNS} FROM gate_status_events
            WHERE interval = ? AND evaluated_at >= ?
            ORDER BY evaluated_at DESC
            """,
            [interval, since.isoformat()],
        )
        return [_row_to_snapshot(row) for row in result.rows]
