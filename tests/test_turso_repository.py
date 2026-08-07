from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.domain.models import Candle, SignalSide, Trade
from trading_scanner.infrastructure.turso import (
    TursoCandleRepository,
    TursoSignalRepository,
    TursoTradeRepository,
    create_turso_client,
)


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _candle(**overrides) -> Candle:
    defaults = dict(
        symbol="AARTIIND.NS",
        timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
        open=Decimal("488.75"),
        high=Decimal("496.70"),
        low=Decimal("487.00"),
        close=Decimal("495.75"),
        volume=1000,
    )
    defaults.update(overrides)
    return Candle(**defaults)


@pytest.mark.asyncio
async def test_upsert_and_get_candles_round_trip(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoCandleRepository(client)
        await repository.ensure_schema()

        await repository.upsert_candles("AARTIIND.NS", "1h", [_candle()])
        stored = await repository.get_candles("AARTIIND.NS", "1h")

        assert stored == [_candle()]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upsert_same_bar_updates_instead_of_duplicating(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoCandleRepository(client)
        await repository.ensure_schema()

        await repository.upsert_candles("AARTIIND.NS", "1h", [_candle(close=Decimal("495.75"))])
        await repository.upsert_candles("AARTIIND.NS", "1h", [_candle(close=Decimal("500.00"))])
        stored = await repository.get_candles("AARTIIND.NS", "1h")

        assert len(stored) == 1
        assert stored[0].close == Decimal("500.00")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_candles_returns_chronological_order_and_respects_limit(
    tmp_path: Path,
) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoCandleRepository(client)
        await repository.ensure_schema()
        candles = [
            _candle(timestamp=datetime(2026, 8, 6, hour, tzinfo=UTC))
            for hour in range(9, 15)
        ]

        await repository.upsert_candles("AARTIIND.NS", "1h", candles)
        stored = await repository.get_candles("AARTIIND.NS", "1h", limit=3)

        assert [candle.timestamp.hour for candle in stored] == [12, 13, 14]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_signal_repository_dedupes_by_fingerprint(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoSignalRepository(client)
        await repository.ensure_schema()

        assert await repository.contains("lorentzian:AARTIIND.NS:buy:x") is False
        await repository.record("lorentzian:AARTIIND.NS:buy:x", datetime.now(UTC))
        assert await repository.contains("lorentzian:AARTIIND.NS:buy:x") is True
    finally:
        await client.close()


def _trade(**overrides) -> Trade:
    defaults = dict(
        symbol="AARTIIND.NS",
        side=SignalSide.BUY,
        entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
        entry_price=Decimal("495.75"),
        prediction_at_entry=6,
        is_early_signal_flip=False,
    )
    defaults.update(overrides)
    return Trade(**defaults)


@pytest.mark.asyncio
async def test_open_and_get_trade_round_trip(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoTradeRepository(client)
        await repository.ensure_schema()

        await repository.open_trade("1h", _trade())
        stored = await repository.get_trades("AARTIIND.NS", "1h")

        assert len(stored) == 1
        assert stored[0].symbol == "AARTIIND.NS"
        assert stored[0].side == SignalSide.BUY
        assert stored[0].status == "open"
        assert stored[0].exit_price is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_open_trade_computes_pnl_percent_for_long(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoTradeRepository(client)
        await repository.ensure_schema()
        await repository.open_trade("1h", _trade(side=SignalSide.BUY, entry_price=Decimal("100")))

        await repository.close_open_trade(
            "AARTIIND.NS",
            "1h",
            SignalSide.BUY,
            datetime(2026, 8, 6, 12, 15, tzinfo=UTC),
            Decimal("110"),
        )
        stored = (await repository.get_trades("AARTIIND.NS", "1h"))[0]

        assert stored.status == "closed"
        assert stored.pnl_percent == Decimal("10")  # long profits when price rises
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_open_trade_computes_pnl_percent_for_short(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoTradeRepository(client)
        await repository.ensure_schema()
        await repository.open_trade("1h", _trade(side=SignalSide.SELL, entry_price=Decimal("100")))

        await repository.close_open_trade(
            "AARTIIND.NS",
            "1h",
            SignalSide.SELL,
            datetime(2026, 8, 6, 12, 15, tzinfo=UTC),
            Decimal("90"),
        )
        stored = (await repository.get_trades("AARTIIND.NS", "1h"))[0]

        assert stored.status == "closed"
        assert stored.pnl_percent == Decimal("10")  # short profits when price falls
    finally:
        await client.close()
