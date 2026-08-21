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
