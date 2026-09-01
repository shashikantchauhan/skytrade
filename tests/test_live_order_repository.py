"""Round-trip tests for the real cash-order ledger (see infrastructure/db/
live_orders.py) against a real local SQLite file -- same style as
test_live_cash_toggle_repository.py's other repository round-trips.

Focus: get_open_cash_legs (COMPLETE only) vs get_unclosed_cash_legs
(COMPLETE/OPEN/UNKNOWN) -- the 2026-08-28 PERSISTENT.NS incident, where an
OPEN leg (order placed, fill status unconfirmed) wasn't counted as "already
open" and a second real BUY stacked on top of it.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import aiosqlite
import pytest

from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.infrastructure.db import TursoLiveOrderRepository, create_turso_client


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _leg(**overrides) -> LiveOrderLeg:
    defaults = dict(
        basket_id="RELIANCE.NS-cash-entry-1", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC), average_price=Decimal("1500"),
    )
    defaults.update(overrides)
    return LiveOrderLeg(**defaults)


@pytest.mark.asyncio
async def test_get_unclosed_cash_legs_counts_a_complete_leg(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())

        unclosed = await repository.get_unclosed_cash_legs("RELIANCE.NS")

        assert len(unclosed) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_unclosed_cash_legs_counts_a_still_open_leg(tmp_path: Path) -> None:
    # 2026-08-28 regression: get_open_cash_legs alone missed exactly this
    # case for PERSISTENT.NS.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(status="OPEN", average_price=None))

        open_only = await repository.get_open_cash_legs("RELIANCE.NS")
        unclosed = await repository.get_unclosed_cash_legs("RELIANCE.NS")

        assert open_only == []  # the old, narrower check -- blind to this
        assert len(unclosed) == 1  # the new, stacking-prevention check
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_unclosed_cash_legs_counts_an_unknown_status_leg(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(status="UNKNOWN", average_price=None))

        unclosed = await repository.get_unclosed_cash_legs("RELIANCE.NS")

        assert len(unclosed) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_unclosed_cash_legs_ignores_a_rejected_leg(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(status="REJECTED", average_price=None))

        unclosed = await repository.get_unclosed_cash_legs("RELIANCE.NS")

        assert unclosed == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_unclosed_cash_legs_is_closed_by_a_matching_complete_sell(
    tmp_path: Path,
) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())
        await repository.record_leg(
            _leg(
                basket_id="RELIANCE.NS-cash-exit-1", transaction_type="SELL", order_id="o2",
            )
        )

        unclosed = await repository.get_unclosed_cash_legs("RELIANCE.NS")

        assert unclosed == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_partial_exit_only_closes_the_quantity_actually_sold(tmp_path: Path) -> None:
    # 2026-08-31 regression, confirmed live against PERSISTENT.NS: bought
    # twice (8 then 9 shares, a stacking bug from three days earlier), then
    # a single 8-share exit only squared off the first leg for real -- but
    # the old "does any COMPLETE opposite-type leg exist for this
    # tradingsymbol" check treated the whole tradingsymbol as closed the
    # moment *any* closer existed, hiding a real 9-share position from the
    # dashboard, the exit path, and the max_positions capacity count.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(
            _leg(
                basket_id="PERSISTENT.NS-cash-entry-1", symbol="PERSISTENT.NS",
                tradingsymbol="PERSISTENT", quantity=8, order_id="o1",
                placed_at=datetime(2026, 8, 28, 7, 57, 22, tzinfo=UTC),
                average_price=Decimal("5896.5"),
            )
        )
        await repository.record_leg(
            _leg(
                basket_id="PERSISTENT.NS-cash-entry-2", symbol="PERSISTENT.NS",
                tradingsymbol="PERSISTENT", quantity=9, order_id="o2",
                placed_at=datetime(2026, 8, 28, 8, 18, 50, tzinfo=UTC),
                average_price=Decimal("5898"),
            )
        )
        await repository.record_leg(
            _leg(
                basket_id="PERSISTENT.NS-cash-exit-1", symbol="PERSISTENT.NS",
                tradingsymbol="PERSISTENT", transaction_type="SELL", quantity=8, order_id="o3",
                placed_at=datetime(2026, 8, 31, 12, 20, 3, tzinfo=UTC),
                average_price=Decimal("5615.5"),
            )
        )

        open_legs = await repository.get_open_cash_legs("PERSISTENT.NS")

        assert len(open_legs) == 1
        assert open_legs[0].quantity == 9
        assert open_legs[0].basket_id == "PERSISTENT.NS-cash-entry-2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_fully_covered_multi_leg_position_shows_as_closed(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(
            _leg(basket_id="RELIANCE.NS-cash-entry-1", quantity=8, order_id="o1")
        )
        await repository.record_leg(
            _leg(basket_id="RELIANCE.NS-cash-entry-2", quantity=9, order_id="o2")
        )
        await repository.record_leg(
            _leg(
                basket_id="RELIANCE.NS-cash-exit-1", transaction_type="SELL", quantity=17,
                order_id="o3",
            )
        )

        open_legs = await repository.get_open_cash_legs("RELIANCE.NS")

        assert open_legs == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_intent_id_round_trips_through_record_and_get_legs(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(intent_id="deadbeef"))

        legs = await repository.get_legs("RELIANCE.NS-cash-entry-1")

        assert legs[0].intent_id == "deadbeef"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_leg_recorded_without_an_intent_id_defaults_to_none(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg())

        legs = await repository.get_legs("RELIANCE.NS-cash-entry-1")

        assert legs[0].intent_id is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_legs_by_intent_gathers_every_retry_attempt(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        # Two attempts under the same intent (a retry), one unrelated leg
        # under a different intent.
        await repository.record_leg(
            _leg(basket_id="b1", order_id="o1", status="REJECTED", intent_id="intent-a")
        )
        await repository.record_leg(
            _leg(basket_id="b1", order_id="o2", status="COMPLETE", intent_id="intent-a")
        )
        await repository.record_leg(
            _leg(basket_id="b2", order_id="o3", intent_id="intent-b")
        )

        legs = await repository.get_legs_by_intent("intent-a")

        assert [leg.order_id for leg in legs] == ["o1", "o2"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_cash_legs_returns_every_status_for_the_symbol(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(order_id="o1", status="REJECTED"))
        await repository.record_leg(_leg(order_id="o2", status="UNKNOWN", average_price=None))

        legs = await repository.get_cash_legs("RELIANCE.NS")

        assert [leg.order_id for leg in legs] == ["o1", "o2"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_all_unclosed_cash_legs_counts_an_unknown_leg(tmp_path: Path) -> None:
    # The exact gap this method exists to close: get_all_open_cash_legs
    # (COMPLETE-only) would miss this entirely.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()
        await repository.record_leg(_leg(status="UNKNOWN", average_price=None))

        open_only = await repository.get_all_open_cash_legs()
        unclosed = await repository.get_all_unclosed_cash_legs()

        assert open_only == []
        assert len(unclosed) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_legs_by_intent_is_empty_for_an_unknown_intent(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        legs = await repository.get_legs_by_intent("never-recorded")

        assert legs == []
    finally:
        await client.close()


# --- Phase 12 (DB invariants) -- CHECK constraints on a fresh database ------
# Only meaningful against a fresh DB (this test always uses tmp_path, a new
# file every time) -- see live_orders.py's own comment on why the already-
# deployed production table isn't retroactively migrated.


@pytest.mark.asyncio
async def test_a_non_positive_quantity_is_rejected(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        with pytest.raises(aiosqlite.IntegrityError):
            await repository.record_leg(_leg(quantity=0))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unknown_purpose_is_rejected(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        with pytest.raises(aiosqlite.IntegrityError):
            await repository.record_leg(_leg(purpose="not-a-real-purpose"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unknown_transaction_type_is_rejected(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        with pytest.raises(aiosqlite.IntegrityError):
            await repository.record_leg(_leg(transaction_type="HOLD"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_arbitrary_kite_status_string_is_still_accepted(tmp_path: Path) -> None:
    # Deliberately NOT constrained -- see live_orders.py's own comment.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoLiveOrderRepository(client)
        await repository.ensure_schema()

        await repository.record_leg(_leg(status="TRIGGER PENDING"))  # must not raise

        legs = await repository.get_legs("RELIANCE.NS-cash-entry-1")
        assert legs[0].status == "TRIGGER PENDING"
    finally:
        await client.close()
