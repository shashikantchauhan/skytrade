"""AlphaEngine's carry-forward incremental-prediction state."""

import libsql_client

from trading_scanner.domain.ports import EngineState

_CREATE_ENGINE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS engine_state (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    signal INTEGER NOT NULL,
    queue_json TEXT,
    exit_state_json TEXT,
    last_bar_timestamp TEXT,
    PRIMARY KEY (symbol, interval)
)
"""


class TursoEngineStateRepository:
    """Persist the small carry-forward state AlphaEngine's fast incremental
    evaluation needs between hourly runs (see application/fast_predict.py)."""

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the engine_state table if it does not already exist."""
        await self._client.execute(_CREATE_ENGINE_STATE_TABLE)

    async def get_state(self, symbol: str, interval: str) -> EngineState:
        """Return the last persisted state, or defaults if never seen."""
        result = await self._client.execute(
            """
            SELECT signal, queue_json, exit_state_json, last_bar_timestamp FROM engine_state
            WHERE symbol = ? AND interval = ?
            """,
            [symbol, interval],
        )
        if not result.rows:
            return EngineState()
        signal, queue_json, exit_state_json, last_bar_timestamp = result.rows[0]
        return EngineState(
            signal=int(signal),
            queue_json=queue_json,
            exit_state_json=exit_state_json,
            last_bar_timestamp=last_bar_timestamp,
        )

    async def set_state(self, symbol: str, interval: str, state: EngineState) -> None:
        await self._client.execute(
            """
            INSERT INTO engine_state
                (symbol, interval, signal, queue_json, exit_state_json, last_bar_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval) DO UPDATE SET
                signal = excluded.signal,
                queue_json = excluded.queue_json,
                exit_state_json = excluded.exit_state_json,
                last_bar_timestamp = excluded.last_bar_timestamp
            """,
            [
                symbol,
                interval,
                state.signal,
                state.queue_json,
                state.exit_state_json,
                state.last_bar_timestamp,
            ],
        )
