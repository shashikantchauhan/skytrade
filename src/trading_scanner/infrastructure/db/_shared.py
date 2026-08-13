"""Shared helpers used by every repository in this package."""

import libsql_client


async def add_column_if_missing(
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
    """Create one shared libSQL client for every repository to reuse.

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
