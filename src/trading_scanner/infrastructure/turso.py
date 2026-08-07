"""Turso (hosted libSQL) storage for accumulated candles and sent signals.

The same ``libsql_client`` connection works against a local ``file:`` database
(no account, no network -- used for tests and local development) or a hosted
``libsql://...`` database with an auth token (production). No other code in
this module depends on which one is in use.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import libsql_client

from trading_scanner.domain.models import Candle, SignalSide, Trade
from trading_scanner.domain.ports import EngineState

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

_CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS sent_signals (
    fingerprint TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
)
"""

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
    status TEXT NOT NULL DEFAULT 'open'
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


def create_turso_client(url: str, auth_token: str | None) -> libsql_client.Client:
    """Create one shared libSQL client for both repositories to reuse.

    ``libsql://`` connects over WebSocket (Hrana), which some networks (proxies,
    restrictive firewalls) block at the handshake. This pipeline has no need
    for WebSocket-only features (subscriptions, interactive transactions across
    calls), so hosted URLs are normalized to plain HTTPS -- functionally
    equivalent here, and works anywhere HTTPS does. Local ``file:`` URLs are
    left untouched.
    """
    if url.startswith("libsql://"):
        url = "https://" + url.removeprefix("libsql://")
    return libsql_client.create_client(url=url, auth_token=auth_token)


class TursoCandleRepository:
    """Persist and retrieve accumulated OHLCV candles in Turso/libSQL."""

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the candles table if it does not already exist."""
        await self._client.execute(_CREATE_CANDLES_TABLE)

    async def upsert_candles(
        self, symbol: str, interval: str, candles: Sequence[Candle]
    ) -> None:
        """Insert new candles or refresh existing ones for the same bar."""
        if not candles:
            return
        statements = [
            libsql_client.Statement(
                _UPSERT_CANDLE,
                [
                    symbol,
                    interval,
                    candle.timestamp.isoformat(),
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


class TursoSignalRepository:
    """Track which signal fingerprints have already been notified."""

    def __init__(self, client: libsql_client.Client) -> None:
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


class TursoTradeRepository:
    """Tracks entries/exits for win-rate and backtest analysis."""

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        """Create the trades table if it does not already exist."""
        await self._client.execute(_CREATE_TRADES_TABLE)

    async def open_trade(self, interval: str, trade: Trade) -> None:
        await self._client.execute(
            """
            INSERT INTO trades
                (symbol, interval, side, entry_timestamp, entry_price,
                 prediction_at_entry, is_early_signal_flip, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                trade.symbol,
                interval,
                trade.side.value,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.prediction_at_entry,
                int(trade.is_early_signal_flip),
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

    async def get_trades(self, symbol: str | None, interval: str) -> Sequence[Trade]:
        query = """
            SELECT symbol, side, entry_timestamp, entry_price, prediction_at_entry,
                   is_early_signal_flip, exit_timestamp, exit_price, pnl_percent, status
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
            )
            for row in result.rows
        ]
