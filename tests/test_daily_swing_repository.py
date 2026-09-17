from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.domain.models import DailySwingPosition
from trading_scanner.infrastructure.db import DailySwingRepository, create_turso_client


@pytest.mark.asyncio
async def test_position_state_survives_entry_trail_and_exit(tmp_path: Path) -> None:
    client = create_turso_client(f"file:{tmp_path / 'swing.db'}", None)
    repository = DailySwingRepository(client)
    await repository.ensure_schema()
    position = DailySwingPosition(
        symbol="POWERGRID.NS",
        side=1,
        setup_date="2026-09-16",
        entry_timestamp=datetime(2026, 9, 17, 4, 15, tzinfo=UTC),
        entry_price=Decimal("270"),
        initial_stop=Decimal("260"),
        active_stop=Decimal("260"),
        target=Decimal("300"),
        atr=Decimal("5"),
        risk=Decimal("10"),
        best_close=Decimal("270"),
        basket_id=None,
    )
    try:
        await repository.create_entering(position)
        await repository.mark_open(position.symbol, position.setup_date, "basket-1")
        await repository.update_trail(position.symbol, Decimal("270"), Decimal("282"))

        active = list(await repository.get_active())
        assert len(active) == 1
        assert active[0].basket_id == "basket-1"
        assert active[0].active_stop == Decimal("270.0")

        await repository.close(
            position.symbol,
            datetime(2026, 9, 18, 5, 15, tzinfo=UTC),
            Decimal("285"),
            "target",
        )
        assert list(await repository.get_active()) == []
    finally:
        await client.close()
