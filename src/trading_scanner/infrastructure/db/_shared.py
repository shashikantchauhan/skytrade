"""Shared helpers used by every repository in this package.

Local SQLite storage via ``aiosqlite`` -- a real async driver, not
``libsql_client``'s ``file:`` mode this project used to run on. That mode's
``execute()``/``batch()`` are declared ``async def`` but internally just
call plain synchronous ``sqlite3`` with a fresh connection per call --
meaning they block the *entire* event loop for their duration. With two
processes (``p-trade-live`` + ``p-trade-dashboard``) hitting the same local
file concurrently, a lock stall in one process froze that process's whole
single-threaded event loop -- ticker, watchdog, heartbeat, everything --
which is the real root cause behind the 2026-08-19 and 2026-08-20 silent
freezes (diagnosed in commit 685e063, not fixed there pending this exact
decision). Hosted Turso was never actually used in production and is not
wanted going forward (explicit 2026-08-20 decision) -- every real
deployment is local ``file:`` SQLite, so ``aiosqlite`` (built specifically
for local SQLite, real async via a background thread) is a strictly better
fit than the client this used to wrap.

Every repository still type-hints its ``client`` param loosely and calls
only ``execute``/``batch``/``close`` -- unchanged from before, so this file
is the only one that needed to change. Class/function names below keep
their historical "Turso" naming (``create_turso_client``, ``Turso*
Repository`` elsewhere in this package) even though Turso itself is gone --
renaming is a purely cosmetic follow-up with zero functional benefit (same
reasoning as this project's own "trading_scanner" package name outliving
the "SkyTrade" rebrand, see pyproject.toml), not bundled into this fix.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import aiosqlite

# 2026-08-17: a burst of concurrent writes (many symbols closing positions
# within the same second, e.g. after a long data outage catches up all at
# once) can contend hard enough on a local SQLite file to raise "database
# is locked" -- previously unhandled, which crashed the entire live
# pipeline process (self-recovered ~30s later via its own top-level retry,
# but a transient, self-resolving lock shouldn't take the whole system
# down). Retries are capped well under _POSITIONS_CACHE_REFRESH_SECONDS
# (5s, see live_pipeline.py) so a retried call never visibly lags behind
# normal polling. Still relevant under aiosqlite -- concurrent writers can
# still collide at the real SQLite level, this just stops that collision
# from blocking the whole process while it resolves.
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY_SECONDS = 0.05


def _is_lock_error(error: Exception) -> bool:
    text = str(error)
    return "database is locked" in text or "SQLITE_BUSY" in text


@dataclass(frozen=True, slots=True)
class Statement:
    """Drop-in replacement for ``libsql_client.Statement`` -- same (sql,
    positional args) shape, used by ``candles.py``'s batch upsert."""

    sql: str
    args: list[Any] | None = None


class _ExecuteResult:
    """Drop-in replacement for libsql_client's execute() result. ``.rows``
    is a list of plain tuples (positionally indexable -- every existing
    call site already does ``row[0]``, ``row[1]``, etc., matching
    ``sqlite3``'s default row shape with no row_factory needed).
    ``.rows_affected`` matches the two attributes ``fix_corrupted_
    candles_cli.py``/``fix_cross_contaminated_candles_cli.py`` already
    read directly."""

    __slots__ = ("rows", "rows_affected")

    def __init__(self, rows: list[tuple], rows_affected: int) -> None:
        self.rows = rows
        self.rows_affected = rows_affected


class _AiosqliteClient:
    """Real async SQLite client -- backs every repository in this package.
    Duck-type compatible with the old ``libsql_client.Client`` interface
    (``execute``/``batch``/``close``) so no repository call site needed to
    change, only this file and the type hints that named the old type.

    Connects lazily, on first real use, rather than in ``__init__`` --
    unlike ``libsql_client.create_client`` (a synchronous constructor),
    ``aiosqlite.connect()`` only actually opens the database once it's
    itself awaited. Since ``create_turso_client`` is called synchronously
    (no ``await``) at all 15 of its existing call sites across this
    codebase, connecting here on first ``execute``/``batch`` instead of at
    construction time keeps every one of those call sites unchanged.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._connection: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection
        async with self._connect_lock:
            if self._connection is None:  # re-check -- another task may have won the race
                self._connection = await aiosqlite.connect(self._path)
        return self._connection

    async def execute(
        self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None
    ) -> _ExecuteResult:
        connection = await self._ensure_connected()
        cursor = await connection.execute(sql, parameters or [])
        try:
            rows = await cursor.fetchall()
        except aiosqlite.Error:
            # Non-SELECT statements (INSERT/UPDATE/DELETE/PRAGMA-without-
            # output) have nothing to fetch -- same as libsql_client
            # returning an empty rows list for those, not an error.
            rows = []
        rows_affected = cursor.rowcount if cursor.rowcount != -1 else 0
        await connection.commit()
        await cursor.close()
        return _ExecuteResult(list(rows), rows_affected)

    async def batch(self, statements: list[Statement]) -> None:
        """Runs every statement in one transaction, matching libsql_client's
        batch semantics (all-or-nothing) -- one commit at the end, not per
        statement, so a mid-batch failure leaves nothing partially applied.
        """
        connection = await self._ensure_connected()
        for statement in statements:
            await connection.execute(statement.sql, statement.args or [])
        await connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()


class _RetryingClient:
    """Wraps a real async SQLite client, retrying ``execute``/``batch`` a
    few times with jittered exponential backoff on a "database is locked"
    error -- see the module-level comment above. Any other exception
    (including a *different* SQLite error) passes through immediately,
    unretried, exactly as it always did; this only ever catches the one
    specific, known-transient failure mode. Duck-type compatible with the
    old ``libsql_client.Client`` for every method this codebase actually
    calls (``execute``, ``batch``, ``close``).
    """

    def __init__(self, client: _AiosqliteClient) -> None:
        self._client = client

    async def _with_retry(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        delay = _LOCK_RETRY_BASE_DELAY_SECONDS
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return await fn(*args, **kwargs)
            except Exception as error:
                if not _is_lock_error(error) or attempt == _LOCK_RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(delay + random.uniform(0, delay))
                delay *= 2

    async def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return await self._with_retry(self._client.execute, *args, **kwargs)

    async def batch(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return await self._with_retry(self._client.batch, *args, **kwargs)

    async def close(self) -> None:
        await self._client.close()


async def add_column_if_missing(client, table: str, column: str, definition: str) -> None:
    """Migrates an already-deployed table forward -- ``CREATE TABLE IF NOT
    EXISTS`` only helps on a fresh database, it never alters an existing
    one.

    Checks ``PRAGMA table_info`` first rather than blind-ALTER-and-swallow
    the "duplicate column" error, matching this function's original
    behavior under libsql_client (kept as-is; real sqlite3 raises a normal
    ``OperationalError`` with "duplicate column name" for this case, so
    the check-first approach is no longer strictly required to avoid a
    weird error shape, but it's cheap and avoids a spurious ALTER either
    way).
    """
    result = await client.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in result.rows}  # row[1] is the column name
    if column in existing_columns:
        return
    await client.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _resolve_local_path(url: str) -> str:
    """Turso URLs are no longer supported (2026-08-20 decision) -- every
    real deployment is local SQLite. Accepts ``file:<path>`` (this
    project's convention everywhere) or a bare path directly."""
    if url.startswith("file:"):
        return url.removeprefix("file:")
    if url.startswith(("libsql://", "https://", "http://")):
        raise ValueError(
            f"Hosted Turso URLs are no longer supported: {url!r}. "
            "Use a local file: URL instead (see .env.example)."
        )
    return url


DbClient = _RetryingClient
"""Public type alias for every repository's ``client`` param -- what
``create_turso_client`` actually returns. A plain alias rather than a
``Protocol`` since every repository already only calls the three concrete
methods ``_RetryingClient`` defines; no need for structural typing here."""


def create_turso_client(url: str, auth_token: str | None) -> _RetryingClient:  # noqa: ARG001
    """Create one shared SQLite client for every repository to reuse.

    ``auth_token`` is accepted (unused) only so every existing call site
    across the codebase -- 15 of them -- keeps working unchanged; it was
    only ever needed for hosted Turso, which is no longer supported.

    Stays a plain synchronous function, exactly like it was before under
    libsql_client -- no call site needs to change to ``await`` it. The
    real ``aiosqlite`` connection isn't opened here; ``_AiosqliteClient``
    connects lazily on first ``execute``/``batch`` instead (see its own
    docstring for why).

    Wrapped in ``_RetryingClient`` (see above) so every repository gets
    lock-retry behavior automatically, with zero changes needed at any
    call site.
    """
    path = _resolve_local_path(url)
    return _RetryingClient(_AiosqliteClient(path))
