"""The single day-long Kite Connect access token."""

from trading_scanner.infrastructure.db._shared import DbClient, add_column_if_missing

_CREATE_KITE_SESSION_TABLE = """
CREATE TABLE IF NOT EXISTS kite_session (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    access_token TEXT NOT NULL,
    obtained_at TEXT NOT NULL,
    expiry_notified_date TEXT,
    expiry_notified_at TEXT
)
"""


class TursoKiteSessionRepository:
    """Persists the single day-long Kite Connect access token.

    One session only -- the dashboard's ``/kite/callback`` route is the only
    writer (see ``webapp.py``), the pipeline is a read-only consumer that
    decides whether a valid Kite session exists based on this table (see
    ``application/signal_pipeline.py``'s ``NoValidKiteSession`` -- there is
    no fallback data source; a missing/expired session just skips the run).
    """

    def __init__(self, client: DbClient) -> None:
        self._client = client

    async def ensure_schema(self) -> None:
        await self._client.execute(_CREATE_KITE_SESSION_TABLE)
        await add_column_if_missing(self._client, "kite_session", "expiry_notified_date", "TEXT")
        await add_column_if_missing(self._client, "kite_session", "expiry_notified_at", "TEXT")

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
        """Superseded 2026-09-02 by ``get_expiry_notified_at`` (see that
        method's docstring for why) -- kept only because it's a real,
        already-persisted column; no longer written to by anything."""
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

    async def get_expiry_notified_at(self) -> str | None:
        """UTC ISO timestamp of the last "Kite session expired, please
        re-login" alert -- see ``application/pipeline/market_data.py``'s
        ``_notify_kite_expired_periodically``, which re-sends every 15
        minutes (not once per calendar day, the old ``expiry_notified_
        date`` behavior) while the session stays unfixed.

        2026-09-02: the once-per-day version let a single missed/late-seen
        alert sit unattended for hours -- in production, an expired token
        blocked real trading for 30-104 minutes at market open on three
        consecutive trading days (2026-08-31, 2026-09-01, 2026-09-02)
        because the one alert for the day came and went unnoticed. A
        15-minute re-nudge can't fully prevent that (still requires a
        human to see it and log in), but it stops one missed ping from
        silently costing the rest of the morning."""
        result = await self._client.execute(
            "SELECT expiry_notified_at FROM kite_session WHERE id = 1"
        )
        if not result.rows:
            return None
        return result.rows[0][0]

    async def set_expiry_notified_at(self, timestamp: str) -> None:
        await self._client.execute(
            """
            INSERT INTO kite_session (id, access_token, obtained_at, expiry_notified_at)
            VALUES (1, '', '', ?)
            ON CONFLICT (id) DO UPDATE SET expiry_notified_at = excluded.expiry_notified_at
            """,
            [timestamp],
        )
