"""Tests for application/broker_reconciliation.py -- proves the exact gap
from the review (an UNKNOWN entry leg invisible to exit-eligibility
checks) is closed by these functions, against a real local SQLite repo
(same style as test_live_order_repository.py)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_scanner.application import broker_reconciliation
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.order_lifecycle import PositionLifecycle
from trading_scanner.infrastructure.db import TursoLiveOrderRepository, create_turso_client


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _leg(**overrides) -> LiveOrderLeg:
    defaults = dict(
        basket_id="RELIANCE.NS-cash-entry-1", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="UNKNOWN", placed_at=datetime.now(UTC), average_price=None,
    )
    defaults.update(overrides)
    return LiveOrderLeg(**defaults)


@pytest.mark.asyncio
async def test_get_unclosed_entry_leg_finds_an_unknown_status_leg(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())

        leg = await broker_reconciliation.get_unclosed_entry_leg("RELIANCE.NS", repository)

        assert leg is not None
        assert leg.tradingsymbol == "RELIANCE"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_unclosed_entry_leg_is_none_when_nothing_is_open(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        leg = await broker_reconciliation.get_unclosed_entry_leg("RELIANCE.NS", repository)

        assert leg is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_all_unclosed_positions_includes_an_unknown_leg(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())

        positions = await broker_reconciliation.get_all_unclosed_positions(repository)

        assert len(positions) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_position_lifecycle_reports_reconciliation_required_for_an_unknown_leg(
    tmp_path: Path,
) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())

        lifecycle = await broker_reconciliation.position_lifecycle("RELIANCE.NS", repository)

        assert lifecycle == PositionLifecycle.RECONCILIATION_REQUIRED
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_position_lifecycle_is_none_for_a_never_traded_symbol(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        lifecycle = await broker_reconciliation.position_lifecycle("RELIANCE.NS", repository)

        assert lifecycle == PositionLifecycle.NONE
    finally:
        await client.close()
