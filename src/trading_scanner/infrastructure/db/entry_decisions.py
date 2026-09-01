"""Persists "why was this signal traded or not" rows for real cash-equity
BUY candidates -- Phase 3 of `projectedPlann.md` (see domain/models.py's
``EntryDecisionRecord`` and application/entry_gates.py). Append-only, like
``TursoLiveOrderRepository``'s ledger -- one row per candidate per scan
cycle, never mutated."""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import EntryDecisionRecord, SignalSide
from trading_scanner.infrastructure.db._shared import DbClient, add_column_if_missing

_CREATE_ENTRY_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS entry_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    signal_timestamp TEXT NOT NULL,
    signal_side TEXT NOT NULL CHECK (signal_side IN ('buy', 'sell')),
    signal_price REAL NOT NULL,
    track_record_passed INTEGER,
    quality_passed INTEGER,
    conviction_passed INTEGER,
    ranking_score REAL,
    ranking_passed INTEGER,
    capital_passed INTEGER,
    position_limit_passed INTEGER,
    cutoff_passed INTEGER,
    final_decision TEXT NOT NULL
        CHECK (final_decision IN ('opened', 'rejected', 'skipped', 'error')),
    blocked_reason TEXT,
    created_at TEXT NOT NULL
)
"""


def _bool_to_int(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _int_to_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


class TursoEntryDecisionRepository:
    """Audit trail for cash-entry gate outcomes -- see
    ``EntryDecisionRecord``'s own docstring for the exact semantics of
    each nullable field."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_ENTRY_DECISIONS_TABLE)
        # Phase 9 (2026-09-01): links this append-only audit trail to the
        # order it led to (if any) -- nullable, additive, no backfill of
        # historical rows, same pattern as live_order_legs.intent_id.
        await add_column_if_missing(self._client, "entry_decisions", "intent_id", "TEXT")
        # get_recent's own query is (symbol, created_at DESC); a future
        # "why wasn't this exact signal traded" lookup is (symbol,
        # signal_timestamp). Append-only table -- grows every scan cycle.
        await self._client.execute(
            "CREATE INDEX IF NOT EXISTS idx_entry_decisions_symbol_created_at "
            "ON entry_decisions (symbol, created_at)"
        )
        await self._client.execute(
            "CREATE INDEX IF NOT EXISTS idx_entry_decisions_symbol_signal_timestamp "
            "ON entry_decisions (symbol, signal_timestamp)"
        )

    async def record(self, decision: EntryDecisionRecord) -> None:
        await self._client.execute(
            """
            INSERT INTO entry_decisions
                (symbol, strategy, signal_timestamp, signal_side, signal_price,
                 track_record_passed, quality_passed, conviction_passed,
                 ranking_score, ranking_passed, capital_passed,
                 position_limit_passed, cutoff_passed, final_decision,
                 blocked_reason, created_at, intent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision.symbol,
                decision.strategy,
                decision.signal_timestamp.isoformat(),
                decision.signal_side.value,
                float(decision.signal_price),
                _bool_to_int(decision.track_record_passed),
                _bool_to_int(decision.quality_passed),
                _bool_to_int(decision.conviction_passed),
                float(decision.ranking_score) if decision.ranking_score is not None else None,
                _bool_to_int(decision.ranking_passed),
                _bool_to_int(decision.capital_passed),
                _bool_to_int(decision.position_limit_passed),
                _bool_to_int(decision.cutoff_passed),
                decision.final_decision,
                decision.blocked_reason,
                decision.created_at.isoformat(),
                decision.intent_id,
            ],
        )

    async def get_recent(self, symbol: str, limit: int = 50) -> list[EntryDecisionRecord]:
        """Most recent decisions for one symbol, newest first -- backs a
        future "why wasn't this traded" dashboard view (Phase 16)."""
        result = await self._client.execute(
            """
            SELECT symbol, strategy, signal_timestamp, signal_side, signal_price,
                   track_record_passed, quality_passed, conviction_passed,
                   ranking_score, ranking_passed, capital_passed,
                   position_limit_passed, cutoff_passed, final_decision,
                   blocked_reason, created_at, intent_id
            FROM entry_decisions WHERE symbol = ? ORDER BY created_at DESC LIMIT ?
            """,
            [symbol, limit],
        )
        return [_row_to_decision(row) for row in result.rows]


def _row_to_decision(row) -> EntryDecisionRecord:
    return EntryDecisionRecord(
        symbol=row[0],
        strategy=row[1],
        signal_timestamp=datetime.fromisoformat(row[2]),
        signal_side=SignalSide(row[3]),
        signal_price=Decimal(str(row[4])),
        track_record_passed=_int_to_bool(row[5]),
        quality_passed=_int_to_bool(row[6]),
        conviction_passed=_int_to_bool(row[7]),
        ranking_score=Decimal(str(row[8])) if row[8] is not None else None,
        ranking_passed=_int_to_bool(row[9]),
        capital_passed=_int_to_bool(row[10]),
        position_limit_passed=_int_to_bool(row[11]),
        cutoff_passed=_int_to_bool(row[12]),
        final_decision=row[13],
        blocked_reason=row[14],
        created_at=datetime.fromisoformat(row[15]),
        intent_id=row[16] if len(row) > 16 else None,
    )
