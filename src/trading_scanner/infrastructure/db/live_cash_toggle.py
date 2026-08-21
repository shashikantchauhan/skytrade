"""The "Go Live" switch for real cash-equity order execution -- a single
DB-backed row, checked fresh by ``live_pipeline.py`` every scan cycle (not
baked into AppConfig at process start like the rest of settings.py), so
flipping it from the dashboard takes effect on the very next cycle with no
service restart. See ``webapp.py``'s live-cash-trading endpoints and
``AppConfig.live_cash_trading_enabled``'s docstring for the split between
this (the live runtime source of truth) and AppConfig (the startup
default, used as-is by anything that doesn't refresh this every cycle).
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_scanner.infrastructure.db._shared import DbClient, add_column_if_missing

_CREATE_LIVE_CASH_TOGGLE_TABLE = """
CREATE TABLE IF NOT EXISTS live_cash_toggle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL,
    symbols TEXT NOT NULL,
    notional REAL NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class LiveCashToggleState:
    enabled: bool
    symbols: frozenset[str]
    notional: Decimal
    # 2026-08-21: caps how many real positions (across the whole symbol
    # allowlist, not per-symbol) can be open at once -- lets the allowlist
    # be wide (e.g. the full 220-symbol universe, so a trial isn't stuck
    # waiting on one specific symbol's signal) while still bounding real
    # capital at risk to max_positions * notional. Enforced in
    # application/live_cash_execution.py's execute_cash_entry.
    max_positions: int = 8
    updated_at: datetime | None = None


class TursoLiveCashToggleRepository:
    """One row (``id = 1``), lazily initialized from ``defaults`` the first
    time ``get_state`` runs -- same singleton-row pattern as
    ``TursoPaperAccountRepository``."""

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_LIVE_CASH_TOGGLE_TABLE)
        await add_column_if_missing(
            self._client, "live_cash_toggle", "max_positions", "INTEGER NOT NULL DEFAULT 8"
        )

    async def get_state(self, defaults: LiveCashToggleState) -> LiveCashToggleState:
        result = await self._client.execute(
            "SELECT enabled, symbols, notional, max_positions, updated_at "
            "FROM live_cash_toggle WHERE id = 1"
        )
        if not result.rows:
            await self.set_state(defaults)
            return defaults
        enabled, symbols_csv, notional, max_positions, updated_at = result.rows[0]
        return LiveCashToggleState(
            enabled=bool(enabled),
            symbols=frozenset(s for s in symbols_csv.split(",") if s),
            notional=Decimal(str(notional)),
            max_positions=int(max_positions),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
        )

    async def set_state(self, state: LiveCashToggleState) -> None:
        now = datetime.now().isoformat()
        await self._client.execute(
            """
            INSERT INTO live_cash_toggle (id, enabled, symbols, notional, max_positions, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled = excluded.enabled,
                symbols = excluded.symbols,
                notional = excluded.notional,
                max_positions = excluded.max_positions,
                updated_at = excluded.updated_at
            """,
            [
                int(state.enabled),
                ",".join(sorted(state.symbols)),
                float(state.notional),
                state.max_positions,
                now,
            ],
        )
