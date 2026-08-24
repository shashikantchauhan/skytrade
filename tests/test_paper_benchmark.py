"""Tests for the paper-trade benchmark run 1:1 alongside real live-cash
trades (see application/paper_benchmark.py and infrastructure/db/
paper_benchmark.py). Application-layer tests use a hand-written fake repo
(no real DB); repository-layer tests round-trip against a real local
SQLite file, same style as test_live_cash_toggle_repository.py.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.application import paper_benchmark
from trading_scanner.domain.models import LiveOrderLeg, PaperBenchmarkPosition
from trading_scanner.infrastructure.db import TursoPaperBenchmarkRepository, create_turso_client

_ENTRY_TIME = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)
_EXIT_TIME = datetime(2026, 8, 25, 11, 0, tzinfo=UTC)


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _leg(
    *, basket_id: str, transaction_type: str = "BUY", quantity: int = 5,
    status: str = "COMPLETE", average_price: Decimal | None = Decimal("1010"),
    placed_at: datetime = _ENTRY_TIME,
) -> LiveOrderLeg:
    return LiveOrderLeg(
        basket_id=basket_id, symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type=transaction_type, quantity=quantity,
        order_id="o1", status=status, placed_at=placed_at, average_price=average_price,
    )


class FakePaperBenchmarkRepository:
    def __init__(self) -> None:
        self.opened: list[PaperBenchmarkPosition] = []
        self.closed: list[tuple] = []

    async def open_position(self, position: PaperBenchmarkPosition) -> None:
        self.opened.append(position)

    async def close_position(
        self, symbol, basket_id, exit_timestamp, paper_exit_price, real_exit_price
    ):
        self.closed.append((symbol, basket_id, exit_timestamp, paper_exit_price, real_exit_price))


# -- application/paper_benchmark.py -----------------------------------------


@pytest.mark.asyncio
async def test_record_entry_writes_open_position_with_paper_and_real_prices():
    repo = FakePaperBenchmarkRepository()
    leg = _leg(basket_id="RELIANCE.NS-cash-entry-x", average_price=Decimal("1010"))

    await paper_benchmark.record_entry("RELIANCE.NS", Decimal("1000"), leg, repo)

    assert len(repo.opened) == 1
    position = repo.opened[0]
    assert position.symbol == "RELIANCE.NS"
    assert position.basket_id == "RELIANCE.NS-cash-entry-x"
    assert position.quantity == 5
    assert position.paper_entry_price == Decimal("1000")
    assert position.real_entry_price == Decimal("1010")
    assert position.status == "open"


@pytest.mark.asyncio
async def test_record_entry_falls_back_to_decision_price_when_average_price_missing():
    repo = FakePaperBenchmarkRepository()
    leg = _leg(basket_id="x", average_price=None)

    await paper_benchmark.record_entry("RELIANCE.NS", Decimal("1000"), leg, repo)

    assert repo.opened[0].real_entry_price == Decimal("1000")


@pytest.mark.asyncio
async def test_record_exit_closes_by_the_entrys_basket_id_not_the_exits():
    repo = FakePaperBenchmarkRepository()
    exit_leg = _leg(
        basket_id="RELIANCE.NS-cash-exit-y", transaction_type="SELL",
        average_price=Decimal("1090"),
    )

    await paper_benchmark.record_exit(
        "RELIANCE.NS", "RELIANCE.NS-cash-entry-x", Decimal("1100"), _EXIT_TIME, exit_leg, repo,
    )

    assert repo.closed == [
        ("RELIANCE.NS", "RELIANCE.NS-cash-entry-x", _EXIT_TIME, Decimal("1100"), Decimal("1090")),
    ]


@pytest.mark.asyncio
async def test_record_exit_falls_back_to_decision_price_when_average_price_missing():
    repo = FakePaperBenchmarkRepository()
    exit_leg = _leg(basket_id="y", transaction_type="SELL", average_price=None)

    await paper_benchmark.record_exit(
        "RELIANCE.NS", "entry-basket", Decimal("1100"), _EXIT_TIME, exit_leg, repo,
    )

    assert repo.closed[0][4] == Decimal("1100")  # real_exit_price falls back to decision price


# -- infrastructure/db/paper_benchmark.py (real SQLite round-trips) ---------


@pytest.mark.asyncio
async def test_repository_open_then_close_computes_both_pnls(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoPaperBenchmarkRepository(client)
        await repository.ensure_schema()

        await repository.open_position(
            PaperBenchmarkPosition(
                symbol="RELIANCE.NS", basket_id="basket-1", quantity=5,
                entry_timestamp=_ENTRY_TIME,
                paper_entry_price=Decimal("1000"), real_entry_price=Decimal("1010"),
            )
        )
        await repository.close_position(
            "RELIANCE.NS", "basket-1", _EXIT_TIME, Decimal("1100"), Decimal("1090"),
        )

        open_positions = await repository.get_open_positions()
        assert open_positions == []

        closed = await repository.get_recent_closed_positions(10)
        assert len(closed) == 1
        position = closed[0]
        assert position.status == "closed"
        # paper: (1100 - 1000) * 5 = 500 ; real: (1090 - 1010) * 5 = 400
        assert position.paper_pnl_amount == Decimal("500")
        assert position.real_pnl_amount == Decimal("400")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_repository_close_position_noop_when_basket_id_not_open(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoPaperBenchmarkRepository(client)
        await repository.ensure_schema()

        # No matching open row at all -- must not raise, must not create one.
        await repository.close_position(
            "RELIANCE.NS", "no-such-basket", _EXIT_TIME, Decimal("1100"), Decimal("1090"),
        )

        result = await client.execute("SELECT COUNT(*) FROM paper_benchmark_positions")
        assert result.rows[0][0] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_repository_close_position_only_closes_the_matching_basket(tmp_path: Path) -> None:
    # Two open pairs for the same symbol (sequential real trades) -- closing
    # one by its exact basket_id must never touch the other.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoPaperBenchmarkRepository(client)
        await repository.ensure_schema()

        for basket_id in ("basket-1", "basket-2"):
            await repository.open_position(
                PaperBenchmarkPosition(
                    symbol="RELIANCE.NS", basket_id=basket_id, quantity=5,
                    entry_timestamp=_ENTRY_TIME,
                    paper_entry_price=Decimal("1000"), real_entry_price=Decimal("1010"),
                )
            )

        await repository.close_position(
            "RELIANCE.NS", "basket-1", _EXIT_TIME, Decimal("1100"), Decimal("1090"),
        )

        open_positions = await repository.get_open_positions()
        assert [p.basket_id for p in open_positions] == ["basket-2"]

        closed = await repository.get_recent_closed_positions(10)
        assert [p.basket_id for p in closed] == ["basket-1"]
    finally:
        await client.close()
