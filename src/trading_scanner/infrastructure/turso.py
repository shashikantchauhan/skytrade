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

from trading_scanner.domain.models import (
    Candle,
    FuturesShadowTrade,
    OptionsShadowTrade,
    PaperPosition,
    SignalSide,
    Trade,
)
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

_CREATE_PAPER_ACCOUNT_TABLE = """
CREATE TABLE IF NOT EXISTS paper_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL
)
"""

_CREATE_PAPER_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    capital_allocated REAL NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_amount REAL,
    status TEXT NOT NULL DEFAULT 'open'
)
"""

_CREATE_KITE_SESSION_TABLE = """
CREATE TABLE IF NOT EXISTS kite_session (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    access_token TEXT NOT NULL,
    obtained_at TEXT NOT NULL,
    expiry_notified_date TEXT
)
"""

_CREATE_OPTIONS_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS options_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    option_type TEXT NOT NULL,
    purpose TEXT NOT NULL,
    option_tradingsymbol TEXT NOT NULL,
    strike REAL NOT NULL,
    expiry TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    entry_timestamp TEXT NOT NULL,
    underlying_price_at_entry REAL NOT NULL,
    entry_premium REAL NOT NULL,
    exit_timestamp TEXT,
    underlying_price_at_exit REAL,
    exit_premium REAL,
    pnl_amount REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source TEXT NOT NULL DEFAULT 'live'
)
"""

_CREATE_FUTURES_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS futures_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    futures_tradingsymbol TEXT NOT NULL,
    expiry TEXT NOT NULL,
    lot_size INTEGER NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    pnl_amount REAL,
    pnl_percent REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source TEXT NOT NULL DEFAULT 'live',
    purpose TEXT NOT NULL DEFAULT 'primary'
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


async def _add_column_if_missing(
    client: libsql_client.Client, table: str, column: str, definition: str
) -> None:
    """Migrates an already-deployed table forward -- ``CREATE TABLE IF NOT
    EXISTS`` only helps on a fresh database, it never alters an existing
    one.

    Checks ``PRAGMA table_info`` first rather than blind-ALTER-and-swallow
    the "duplicate column" error: over Turso's HTTP transport, an ALTER
    against an already-existing column doesn't come back as a normal
    exception with that text in it -- ``libsql_client``'s HTTP backend
    raises a raw ``KeyError('result')`` while parsing the error response,
    which silently killed every caller of this function (the derivatives
    backtest CLI, in particular, crashed before doing any work, so
    triggering it from the dashboard looked like a no-op with zero
    feedback). Checking first sidesteps relying on that error shape at all.
    """
    result = await client.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in result.rows}  # row[1] is the column name
    if column in existing_columns:
        return
    await client.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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
            libsql_client.Statement(
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

    async def abandon_open_trade(self, symbol: str, interval: str, side: SignalSide) -> None:
        """Mark a still-open trade abandoned, excluded from win-rate stats.

        Mirrors Pine's ``ml.backtest``: a new opposite-side entry discards
        whatever position was still open without scoring it -- not closed,
        not a win, not a loss.
        """
        await self._client.execute(
            """
            UPDATE trades SET status = 'abandoned'
            WHERE symbol = ? AND interval = ? AND side = ? AND status = 'open'
            """,
            [symbol, interval, side.value],
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


class TursoPaperAccountRepository:
    """Persists the paper-trading account's cash balance and positions.

    One account only -- ``paper_account`` always holds a single row
    (``id = 1``), initialized with ``initial_capital`` the first time
    ``get_cash_balance`` runs and left untouched on every call after that.
    """

    def __init__(self, client: libsql_client.Client, initial_capital: Decimal) -> None:
        self._client = client
        self._initial_capital = initial_capital

    async def ensure_schema(self) -> None:
        """Create the paper_account and paper_positions tables if missing."""
        await self._client.execute(_CREATE_PAPER_ACCOUNT_TABLE)
        await self._client.execute(_CREATE_PAPER_POSITIONS_TABLE)

    async def get_cash_balance(self) -> Decimal:
        result = await self._client.execute("SELECT cash_balance FROM paper_account WHERE id = 1")
        if not result.rows:
            await self._client.execute(
                "INSERT INTO paper_account (id, cash_balance) VALUES (1, ?)",
                [float(self._initial_capital)],
            )
            return self._initial_capital
        return Decimal(str(result.rows[0][0]))

    async def open_position(self, position: PaperPosition) -> None:
        """Record a new open position and deduct its capital from cash."""
        await self.get_cash_balance()  # Ensures the account row exists.
        await self._client.execute(
            """
            INSERT INTO paper_positions
                (symbol, entry_timestamp, entry_price, quantity, capital_allocated, status)
            VALUES (?, ?, ?, ?, ?, 'open')
            """,
            [
                position.symbol,
                position.entry_timestamp.isoformat(),
                float(position.entry_price),
                position.quantity,
                float(position.capital_allocated),
            ],
        )
        await self._client.execute(
            "UPDATE paper_account SET cash_balance = cash_balance - ? WHERE id = 1",
            [float(position.capital_allocated)],
        )

    async def close_position(
        self, symbol: str, exit_timestamp: datetime, exit_price: Decimal
    ) -> PaperPosition | None:
        """Close the most recent open position for symbol, crediting cash back."""
        result = await self._client.execute(
            """
            SELECT id, entry_timestamp, entry_price, quantity, capital_allocated
            FROM paper_positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol],
        )
        if not result.rows:
            return None
        position_id, entry_timestamp, entry_price, quantity, capital_allocated = result.rows[0]
        entry_price = Decimal(str(entry_price))
        capital_allocated = Decimal(str(capital_allocated))
        pnl_amount = (exit_price - entry_price) * quantity
        proceeds = capital_allocated + pnl_amount
        await self._client.execute(
            """
            UPDATE paper_positions
            SET exit_timestamp = ?, exit_price = ?, pnl_amount = ?, status = 'closed'
            WHERE id = ?
            """,
            [exit_timestamp.isoformat(), float(exit_price), float(pnl_amount), position_id],
        )
        await self._client.execute(
            "UPDATE paper_account SET cash_balance = cash_balance + ? WHERE id = 1",
            [float(proceeds)],
        )
        return PaperPosition(
            symbol=symbol,
            entry_timestamp=datetime.fromisoformat(entry_timestamp),
            entry_price=entry_price,
            quantity=quantity,
            capital_allocated=capital_allocated,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            status="closed",
        )

    async def get_open_positions(self) -> Sequence[PaperPosition]:
        result = await self._client.execute(
            """
            SELECT symbol, entry_timestamp, entry_price, quantity, capital_allocated
            FROM paper_positions WHERE status = 'open'
            """
        )
        return [
            PaperPosition(
                symbol=row[0],
                entry_timestamp=datetime.fromisoformat(row[1]),
                entry_price=Decimal(str(row[2])),
                quantity=row[3],
                capital_allocated=Decimal(str(row[4])),
            )
            for row in result.rows
        ]


class TursoKiteSessionRepository:
    """Persists the single day-long Kite Connect access token.

    One session only -- the dashboard's ``/kite/callback`` route is the only
    writer (see ``webapp.py``), the pipeline is a read-only consumer that
    decides whether to use Kite or fall back to Yahoo based on this table.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_KITE_SESSION_TABLE)
        await _add_column_if_missing(self._client, "kite_session", "expiry_notified_date", "TEXT")

    async def set_token(self, access_token: str, obtained_at: str) -> None:
        await self._client.execute(
            """
            INSERT INTO kite_session (id, access_token, obtained_at) VALUES (1, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                access_token = excluded.access_token,
                obtained_at = excluded.obtained_at
            """,
            [access_token, obtained_at],
        )

    async def get_token(self) -> tuple[str, str] | None:
        result = await self._client.execute(
            "SELECT access_token, obtained_at FROM kite_session WHERE id = 1"
        )
        if not result.rows:
            return None
        return result.rows[0][0], result.rows[0][1]

    async def get_expiry_notified_date(self) -> str | None:
        """Last calendar date (YYYY-MM-DD) the "Kite session expired,
        please re-login" alert was sent -- see ``application/
        signal_pipeline.py``'s ``_select_provider``, which sends at most
        one per day so an all-day expired session doesn't spam Telegram
        every hourly run."""
        result = await self._client.execute(
            "SELECT expiry_notified_date FROM kite_session WHERE id = 1"
        )
        if not result.rows:
            return None
        return result.rows[0][0]

    async def set_expiry_notified_date(self, date_str: str) -> None:
        # INSERT ... ON CONFLICT rather than a plain UPDATE -- there may be
        # no row yet at all if Kite has never been logged into on this
        # deployment, and the expiry alert still needs to fire/dedupe in
        # that case, not silently no-op.
        await self._client.execute(
            """
            INSERT INTO kite_session (id, access_token, obtained_at, expiry_notified_date)
            VALUES (1, '', '', ?)
            ON CONFLICT (id) DO UPDATE SET expiry_notified_date = excluded.expiry_notified_date
            """,
            [date_str],
        )


class TursoOptionsTradeRepository:
    """Tracks hypothetical options trades shadowing BUY/SELL signals.

    A symbol can have up to two simultaneously open shadow option trades --
    a "directional" one and a "hedge" one (see ``domain.models.
    OptionsShadowTrade``) -- so every lookup/close is scoped by
    ``(symbol, option_type, purpose)``, not just symbol. Analysis only --
    see ``application/options_shadow.py``. Fully separate from the paper
    account's capital and from ``trades``' own equity-side scoring.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_OPTIONS_TRADES_TABLE)
        await _add_column_if_missing(
            self._client, "options_trades", "source", "TEXT NOT NULL DEFAULT 'live'"
        )

    async def open_trade(self, trade: OptionsShadowTrade) -> None:
        await self._client.execute(
            """
            INSERT INTO options_trades
                (symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                 lot_size, entry_timestamp, underlying_price_at_entry, entry_premium, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                trade.symbol,
                trade.option_type,
                trade.purpose,
                trade.option_tradingsymbol,
                float(trade.strike),
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.underlying_price_at_entry),
                float(trade.entry_premium),
            ],
        )

    async def get_open_trade(
        self, symbol: str, option_type: str, purpose: str
    ) -> OptionsShadowTrade | None:
        result = await self._client.execute(
            """
            SELECT symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                   lot_size, entry_timestamp, underlying_price_at_entry, entry_premium
            FROM options_trades
            WHERE symbol = ? AND option_type = ? AND purpose = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol, option_type, purpose],
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return OptionsShadowTrade(
            symbol=row[0],
            option_type=row[1],
            purpose=row[2],
            option_tradingsymbol=row[3],
            strike=Decimal(str(row[4])),
            expiry=row[5],
            lot_size=int(row[6]),
            entry_timestamp=datetime.fromisoformat(row[7]),
            underlying_price_at_entry=Decimal(str(row[8])),
            entry_premium=Decimal(str(row[9])),
        )

    async def close_trade(
        self,
        symbol: str,
        option_type: str,
        purpose: str,
        exit_timestamp: datetime,
        underlying_price_at_exit: Decimal,
        exit_premium: Decimal,
        pnl_amount: Decimal,
        pnl_percent: Decimal,
    ) -> None:
        await self._client.execute(
            """
            UPDATE options_trades SET
                exit_timestamp = ?, underlying_price_at_exit = ?, exit_premium = ?,
                pnl_amount = ?, pnl_percent = ?, status = 'closed'
            WHERE id = (
                SELECT id FROM options_trades
                WHERE symbol = ? AND option_type = ? AND purpose = ? AND status = 'open'
                ORDER BY entry_timestamp DESC LIMIT 1
            )
            """,
            [
                exit_timestamp.isoformat(),
                float(underlying_price_at_exit),
                float(exit_premium),
                float(pnl_amount),
                float(pnl_percent),
                symbol,
                option_type,
                purpose,
            ],
        )

    async def delete_backtest_trades(self) -> None:
        """Clears every previous source='backtest' row before a fresh
        backtest run writes new ones -- without this, re-running the
        backtest (or running it again after logic changes) just appends on
        top of stale rows forever, silently mixing old and new results
        under the same label with no way to tell them apart (this is what
        was actually behind a "derivatives data looks wrong" report)."""
        await self._client.execute("DELETE FROM options_trades WHERE source = 'backtest'")

    async def insert_backtest_trade(self, trade: OptionsShadowTrade) -> None:
        """Inserts one already-closed row directly, source='backtest'.

        Unlike ``open_trade``/``close_trade`` (a two-step open-then-later-
        close flow scoped by the most recent open row for a key), a
        backtest replay knows both entry and exit up front and runs as a
        one-shot batch alongside the always-on live shadow-tracking flow --
        writing the whole row atomically avoids racing live's own
        open/close lookups for the same (symbol, option_type, purpose).
        """
        await self._client.execute(
            """
            INSERT INTO options_trades
                (symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                 lot_size, entry_timestamp, underlying_price_at_entry, entry_premium,
                 exit_timestamp, underlying_price_at_exit, exit_premium,
                 pnl_amount, pnl_percent, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', 'backtest')
            """,
            [
                trade.symbol,
                trade.option_type,
                trade.purpose,
                trade.option_tradingsymbol,
                float(trade.strike),
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.underlying_price_at_entry),
                float(trade.entry_premium),
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                float(trade.underlying_price_at_exit)
                if trade.underlying_price_at_exit is not None
                else None,
                float(trade.exit_premium) if trade.exit_premium is not None else None,
                float(trade.pnl_amount) if trade.pnl_amount is not None else None,
                float(trade.pnl_percent) if trade.pnl_percent is not None else None,
            ],
        )

    async def get_trades(
        self, symbol: str | None = None, source: str | None = None
    ) -> Sequence[OptionsShadowTrade]:
        query = """
            SELECT symbol, option_type, purpose, option_tradingsymbol, strike, expiry,
                   lot_size, entry_timestamp, underlying_price_at_entry, entry_premium,
                   exit_timestamp, underlying_price_at_exit, exit_premium,
                   pnl_amount, pnl_percent, status, source
            FROM options_trades
        """
        clauses: list[str] = []
        parameters: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            parameters.append(symbol)
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY entry_timestamp DESC"
        result = await self._client.execute(query, parameters)
        return [
            OptionsShadowTrade(
                symbol=row[0],
                option_type=row[1],
                purpose=row[2],
                option_tradingsymbol=row[3],
                strike=Decimal(str(row[4])),
                expiry=row[5],
                lot_size=int(row[6]),
                entry_timestamp=datetime.fromisoformat(row[7]),
                underlying_price_at_entry=Decimal(str(row[8])),
                entry_premium=Decimal(str(row[9])),
                exit_timestamp=datetime.fromisoformat(row[10]) if row[10] else None,
                underlying_price_at_exit=Decimal(str(row[11])) if row[11] is not None else None,
                exit_premium=Decimal(str(row[12])) if row[12] is not None else None,
                pnl_amount=Decimal(str(row[13])) if row[13] is not None else None,
                pnl_percent=Decimal(str(row[14])) if row[14] is not None else None,
                status=row[15],
                source=row[16],
            )
            for row in result.rows
        ]


class TursoFuturesTradeRepository:
    """Tracks hypothetical futures trades shadowing BUY/SELL signals.

    Two purposes can be open per symbol at once -- ``purpose="primary"``
    (the futures position is the trade) and ``purpose="hedge"`` (it hedges
    a directional option instead, see ``domain.models.FuturesShadowTrade``)
    -- so every lookup/close is scoped by ``(symbol, purpose)``, not just
    symbol. Analysis only -- see ``application/futures_shadow.py``. Fully
    separate from the paper account's capital.
    """

    def __init__(self, client: libsql_client.Client) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_FUTURES_TRADES_TABLE)
        await _add_column_if_missing(
            self._client, "futures_trades", "source", "TEXT NOT NULL DEFAULT 'live'"
        )
        await _add_column_if_missing(
            self._client, "futures_trades", "purpose", "TEXT NOT NULL DEFAULT 'primary'"
        )

    async def open_trade(self, trade: FuturesShadowTrade) -> None:
        await self._client.execute(
            """
            INSERT INTO futures_trades
                (symbol, side, futures_tradingsymbol, expiry, lot_size,
                 entry_timestamp, entry_price, purpose, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            [
                trade.symbol,
                trade.side,
                trade.futures_tradingsymbol,
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.purpose,
            ],
        )

    async def get_open_trade(
        self, symbol: str, purpose: str = "primary"
    ) -> FuturesShadowTrade | None:
        result = await self._client.execute(
            """
            SELECT symbol, side, futures_tradingsymbol, expiry, lot_size,
                   entry_timestamp, entry_price, purpose
            FROM futures_trades WHERE symbol = ? AND purpose = ? AND status = 'open'
            ORDER BY entry_timestamp DESC LIMIT 1
            """,
            [symbol, purpose],
        )
        if not result.rows:
            return None
        row = result.rows[0]
        return FuturesShadowTrade(
            symbol=row[0],
            side=row[1],
            futures_tradingsymbol=row[2],
            expiry=row[3],
            lot_size=int(row[4]),
            entry_timestamp=datetime.fromisoformat(row[5]),
            entry_price=Decimal(str(row[6])),
            purpose=row[7],
        )

    async def close_trade(
        self,
        symbol: str,
        exit_timestamp: datetime,
        exit_price: Decimal,
        pnl_amount: Decimal,
        pnl_percent: Decimal,
        purpose: str = "primary",
    ) -> None:
        await self._client.execute(
            """
            UPDATE futures_trades SET
                exit_timestamp = ?, exit_price = ?, pnl_amount = ?, pnl_percent = ?,
                status = 'closed'
            WHERE id = (
                SELECT id FROM futures_trades
                WHERE symbol = ? AND purpose = ? AND status = 'open'
                ORDER BY entry_timestamp DESC LIMIT 1
            )
            """,
            [
                exit_timestamp.isoformat(),
                float(exit_price),
                float(pnl_amount),
                float(pnl_percent),
                symbol,
                purpose,
            ],
        )

    async def delete_backtest_trades(self) -> None:
        """Clears every previous source='backtest' row -- see
        ``TursoOptionsTradeRepository.delete_backtest_trades``."""
        await self._client.execute("DELETE FROM futures_trades WHERE source = 'backtest'")

    async def insert_backtest_trade(self, trade: FuturesShadowTrade) -> None:
        """Inserts one already-closed row directly, source='backtest' --
        see ``TursoOptionsTradeRepository.insert_backtest_trade`` for why
        this bypasses the open/close two-step."""
        await self._client.execute(
            """
            INSERT INTO futures_trades
                (symbol, side, futures_tradingsymbol, expiry, lot_size,
                 entry_timestamp, entry_price, exit_timestamp, exit_price,
                 pnl_amount, pnl_percent, purpose, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', 'backtest')
            """,
            [
                trade.symbol,
                trade.side,
                trade.futures_tradingsymbol,
                trade.expiry,
                trade.lot_size,
                trade.entry_timestamp.isoformat(),
                float(trade.entry_price),
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                float(trade.exit_price) if trade.exit_price is not None else None,
                float(trade.pnl_amount) if trade.pnl_amount is not None else None,
                float(trade.pnl_percent) if trade.pnl_percent is not None else None,
                trade.purpose,
            ],
        )

    async def get_trades(
        self, symbol: str | None = None, source: str | None = None, purpose: str | None = None
    ) -> Sequence[FuturesShadowTrade]:
        query = """
            SELECT symbol, side, futures_tradingsymbol, expiry, lot_size,
                   entry_timestamp, entry_price, exit_timestamp, exit_price,
                   pnl_amount, pnl_percent, status, source, purpose
            FROM futures_trades
        """
        clauses: list[str] = []
        parameters: list[str] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            parameters.append(symbol)
        if source is not None:
            clauses.append("source = ?")
            parameters.append(source)
        if purpose is not None:
            clauses.append("purpose = ?")
            parameters.append(purpose)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY entry_timestamp DESC"
        result = await self._client.execute(query, parameters)
        return [
            FuturesShadowTrade(
                symbol=row[0],
                side=row[1],
                futures_tradingsymbol=row[2],
                expiry=row[3],
                lot_size=int(row[4]),
                entry_timestamp=datetime.fromisoformat(row[5]),
                entry_price=Decimal(str(row[6])),
                exit_timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                exit_price=Decimal(str(row[8])) if row[8] is not None else None,
                pnl_amount=Decimal(str(row[9])) if row[9] is not None else None,
                pnl_percent=Decimal(str(row[10])) if row[10] is not None else None,
                status=row[11],
                source=row[12],
                purpose=row[13],
            )
            for row in result.rows
        ]
