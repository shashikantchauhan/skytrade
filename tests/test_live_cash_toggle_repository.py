"""Round-trip tests for the "Go Live" DB toggle (see infrastructure/db/
live_cash_toggle.py) against a real local SQLite file -- same style as
test_db_repository.py's other repository round-trips."""

from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoLiveCashToggleRepository,
    create_turso_client,
)


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


@pytest.mark.asyncio
async def test_get_state_initializes_from_defaults_on_first_call(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()

        defaults = LiveCashToggleState(
            enabled=False, symbols=frozenset({"RELIANCE.NS"}), notional=Decimal("5000")
        )
        state = await repository.get_state(defaults)

        assert state.enabled is False
        assert state.symbols == frozenset({"RELIANCE.NS"})
        assert state.notional == Decimal("5000")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_set_state_then_get_state_round_trips(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()

        defaults = LiveCashToggleState(enabled=False, symbols=frozenset(), notional=Decimal("5000"))
        await repository.get_state(defaults)  # seed the row

        new_state = LiveCashToggleState(
            enabled=True,
            symbols=frozenset({"RELIANCE.NS", "TCS.NS"}),
            notional=Decimal("7500"),
        )
        await repository.set_state(new_state)

        stored = await repository.get_state(defaults)
        assert stored.enabled is True
        assert stored.symbols == frozenset({"RELIANCE.NS", "TCS.NS"})
        assert stored.notional == Decimal("7500")
        assert stored.updated_at is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_set_state_overwrites_not_duplicates(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()

        await repository.set_state(
            LiveCashToggleState(enabled=True, symbols=frozenset({"A.NS"}), notional=Decimal("1000"))
        )
        await repository.set_state(
            LiveCashToggleState(
                enabled=False, symbols=frozenset({"B.NS"}), notional=Decimal("2000")
            )
        )

        result = await client.execute("SELECT COUNT(*) FROM live_cash_toggle")
        assert result.rows[0][0] == 1  # still a single row, not two

        defaults = LiveCashToggleState(enabled=True, symbols=frozenset(), notional=Decimal("0"))
        stored = await repository.get_state(defaults)
        assert stored.enabled is False
        assert stored.symbols == frozenset({"B.NS"})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_max_positions_round_trips(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()

        await repository.set_state(
            LiveCashToggleState(
                enabled=True, symbols=frozenset({"A.NS"}), notional=Decimal("5000"),
                max_positions=8,
            )
        )
        defaults = LiveCashToggleState(enabled=False, symbols=frozenset(), notional=Decimal("0"))
        stored = await repository.get_state(defaults)

        assert stored.max_positions == 8
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_delayed_retry_enabled_defaults_false_and_round_trips(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()

        defaults = LiveCashToggleState(enabled=False, symbols=frozenset(), notional=Decimal("0"))
        seeded = await repository.get_state(defaults)
        assert seeded.delayed_retry_enabled is False  # off by default -- real capital risk

        await repository.set_state(
            LiveCashToggleState(
                enabled=True, symbols=frozenset({"A.NS"}), notional=Decimal("5000"),
                delayed_retry_enabled=True,
            )
        )
        stored = await repository.get_state(defaults)

        assert stored.delayed_retry_enabled is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ensure_schema_migrates_a_table_created_before_max_positions_existed(
    tmp_path: Path,
) -> None:
    """A production DB that already has this table (from before
    max_positions was added) must not break -- ensure_schema has to add
    the column via ALTER TABLE, not just CREATE TABLE IF NOT EXISTS."""
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        await client.execute(
            """
            CREATE TABLE live_cash_toggle (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled INTEGER NOT NULL,
                symbols TEXT NOT NULL,
                notional REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await client.execute(
            "INSERT INTO live_cash_toggle (id, enabled, symbols, notional, updated_at) "
            "VALUES (1, 0, 'RELIANCE.NS', 5000, '2026-08-21T00:00:00')"
        )

        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()  # must not raise

        defaults = LiveCashToggleState(enabled=False, symbols=frozenset(), notional=Decimal("0"))
        stored = await repository.get_state(defaults)
        assert stored.symbols == frozenset({"RELIANCE.NS"})
        assert stored.max_positions == 8  # column default for pre-existing rows
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ensure_schema_migrates_a_table_created_before_delayed_retry_existed(
    tmp_path: Path,
) -> None:
    """Same migration guarantee as max_positions above, for the newer
    delayed_retry_enabled column -- a production row from before this
    existed must read back as False (off), not break."""
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        await client.execute(
            """
            CREATE TABLE live_cash_toggle (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled INTEGER NOT NULL,
                symbols TEXT NOT NULL,
                notional REAL NOT NULL,
                max_positions INTEGER NOT NULL DEFAULT 8,
                updated_at TEXT NOT NULL
            )
            """
        )
        await client.execute(
            "INSERT INTO live_cash_toggle (id, enabled, symbols, notional, max_positions, "
            "updated_at) VALUES (1, 1, 'RELIANCE.NS', 5000, 8, '2026-08-21T00:00:00')"
        )

        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()  # must not raise

        defaults = LiveCashToggleState(enabled=False, symbols=frozenset(), notional=Decimal("0"))
        stored = await repository.get_state(defaults)
        assert stored.symbols == frozenset({"RELIANCE.NS"})
        assert stored.delayed_retry_enabled is False  # column default for pre-existing rows
    finally:
        await client.close()
