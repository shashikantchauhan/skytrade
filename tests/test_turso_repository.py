from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.domain.models import (
    Candle,
    FuturesPaperPosition,
    PaperPosition,
    SignalSide,
    Trade,
)
from trading_scanner.infrastructure.turso import (
    TursoCandleRepository,
    TursoFuturesPaperAccountRepository,
    TursoPaperAccountRepository,
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


@pytest.mark.asyncio
async def test_abandon_open_trade_marks_it_abandoned_without_scoring(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoTradeRepository(client)
        await repository.ensure_schema()
        await repository.open_trade("1h", _trade(side=SignalSide.BUY, entry_price=Decimal("100")))

        await repository.abandon_open_trade("AARTIIND.NS", "1h", SignalSide.BUY)
        stored = (await repository.get_trades("AARTIIND.NS", "1h"))[0]

        assert stored.status == "abandoned"
        assert stored.exit_price is None
        assert stored.pnl_percent is None  # never scored as a win or a loss
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_paper_account_initializes_cash_balance_on_first_call(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoPaperAccountRepository(client, Decimal("500000"))
        await repository.ensure_schema()

        assert await repository.get_cash_balance() == Decimal("500000")
        # A second call must not re-initialize (and thus not reset) the balance.
        await repository.open_position(
            PaperPosition(
                symbol="AARTIIND.NS",
                entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
                entry_price=Decimal("100"),
                quantity=750,
                capital_allocated=Decimal("75000"),
            )
        )
        assert await repository.get_cash_balance() == Decimal("425000")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_paper_position_open_close_round_trip_credits_pnl_to_cash(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoPaperAccountRepository(client, Decimal("500000"))
        await repository.ensure_schema()
        await repository.open_position(
            PaperPosition(
                symbol="AARTIIND.NS",
                entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
                entry_price=Decimal("100"),
                quantity=750,
                capital_allocated=Decimal("75000"),
            )
        )

        closed = await repository.close_position(
            "AARTIIND.NS", datetime(2026, 8, 7, 10, 15, tzinfo=UTC), Decimal("110")
        )

        assert closed.status == "closed"
        assert closed.pnl_amount == Decimal("7500")  # (110-100) * 750
        # 500000 - 75000 (opened) + 75000 + 7500 (closed: capital back + pnl).
        assert await repository.get_cash_balance() == Decimal("507500")
        assert await repository.get_open_positions() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_futures_paper_account_initializes_cash_balance_on_first_call(
    tmp_path: Path,
) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoFuturesPaperAccountRepository(client, Decimal("400000"))
        await repository.ensure_schema()

        assert await repository.get_cash_balance() == Decimal("400000")
        await repository.open_position(
            FuturesPaperPosition(
                symbol="RELIANCE.NS",
                side="long",
                entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
                futures_entry_price=Decimal("2900"),
                futures_tradingsymbol="RELIANCE26AUGFUT",
                hedge_tradingsymbol="RELIANCE26AUG2800PE",
                lot_size=500,
                margin_allocated=Decimal("18750"),
            )
        )
        assert await repository.get_cash_balance() == Decimal("381250")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_futures_paper_position_open_close_round_trip_credits_pnl_for_long(
    tmp_path: Path,
) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoFuturesPaperAccountRepository(client, Decimal("400000"))
        await repository.ensure_schema()
        await repository.open_position(
            FuturesPaperPosition(
                symbol="RELIANCE.NS",
                side="long",
                entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
                futures_entry_price=Decimal("2900"),
                futures_tradingsymbol="RELIANCE26AUGFUT",
                hedge_tradingsymbol="RELIANCE26AUG2800PE",
                lot_size=500,
                margin_allocated=Decimal("18750"),
            )
        )

        closed = await repository.close_position(
            "RELIANCE.NS", datetime(2026, 8, 7, 10, 15, tzinfo=UTC), Decimal("2920")
        )

        assert closed.status == "closed"
        assert closed.pnl_amount == Decimal("10000")  # (2920-2900) * 500
        # 400000 - 18750 (opened) + 18750 + 10000 (closed: margin back + pnl).
        assert await repository.get_cash_balance() == Decimal("410000")
        assert await repository.get_open_positions() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_futures_paper_position_close_credits_pnl_for_short(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoFuturesPaperAccountRepository(client, Decimal("400000"))
        await repository.ensure_schema()
        await repository.open_position(
            FuturesPaperPosition(
                symbol="RELIANCE.NS",
                side="short",
                entry_timestamp=datetime(2026, 8, 6, 10, 15, tzinfo=UTC),
                futures_entry_price=Decimal("2900"),
                futures_tradingsymbol="RELIANCE26AUGFUT",
                hedge_tradingsymbol="RELIANCE26AUG3000CE",
                lot_size=500,
                margin_allocated=Decimal("18750"),
            )
        )

        # Price fell -- a short profits.
        closed = await repository.close_position(
            "RELIANCE.NS", datetime(2026, 8, 7, 10, 15, tzinfo=UTC), Decimal("2880")
        )

        assert closed.pnl_amount == Decimal("10000")  # (2900-2880) * 500
    finally:
        await client.close()
