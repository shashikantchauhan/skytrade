"""Tests for the dashboard-triggered manual exit (application/manual_exit.py).
Everything here uses fakes -- no real Kite connection, no real money."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import manual_exit
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import GttBracket, LiveOrderLeg


def _config() -> AppConfig:
    return AppConfig(
        scan_interval_hours=1,
        candle_interval="1h",
        candle_history=300,
        symbols_file=None,
        logging_level=20,
        turso_database_url=None,
        turso_auth_token=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        index_symbol=None,
        kite_api_key=None,
        kite_api_secret=None,
        live_trading_enabled=False,
        live_trading_symbols=frozenset(),
        live_trading_max_lots=1,
        futures_paper_symbols_file=None,
        live_cash_trading_enabled=True,
        live_cash_trading_symbols=frozenset({"RELIANCE.NS"}),
        live_cash_trading_notional=Decimal("50000"),
        live_cash_trading_max_positions=8,
    )


class FakeOrderExecutor:
    def __init__(
        self,
        gtt_status: str = "active",
        fill_status: str = "COMPLETE",
        holding_quantity: int | None = None,
    ) -> None:
        self.deleted: list[int] = []
        self._gtt_status = gtt_status
        self._fill_status = fill_status
        self.calls: list[tuple[str, str, int]] = []
        self._order_counter = 0
        # None -- not configured -- makes reconcile_before_exit fall back to
        # GTT-status-only reasoning, same as a real API failure would.
        self._holding_quantity = holding_quantity

    def gtt_status(self, trigger_id):
        return self._gtt_status

    def holding_quantity(self, tradingsymbol):
        if self._holding_quantity is None:
            raise RuntimeError("holding_quantity not configured for this fake")
        return self._holding_quantity

    def delete_gtt(self, trigger_id):
        self.deleted.append(trigger_id)

    def place_cash_market_order(self, tradingsymbol, transaction_type, quantity, reference_price):
        self.calls.append((tradingsymbol, transaction_type, quantity))
        self._order_counter += 1
        return f"order-{self._order_counter}"

    def wait_for_fill(self, order_id, timeout_seconds, poll_interval=1.0):
        return {"status": self._fill_status, "average_price": 1550.0, "status_message": None}


class FakeGttRepository:
    def __init__(self, active: GttBracket | None = None) -> None:
        self._active = active
        self.status_updates: list[tuple] = []

    async def get_active(self, symbol: str):
        return self._active

    async def update_status(self, trigger_id, status, stop_price=None, target_price=None):
        self.status_updates.append((trigger_id, status))
        if self._active is not None and self._active.trigger_id == trigger_id:
            self._active.status = status


class FakeLiveOrderRepository:
    def __init__(self, open_leg: LiveOrderLeg | None = None) -> None:
        self._open_leg = open_leg
        self.recorded: list[LiveOrderLeg] = []

    async def get_open_cash_legs(self, symbol: str):
        return [self._open_leg] if self._open_leg else []

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        self.recorded.append(leg)

    async def get_legs(self, basket_id: str):
        return [leg for leg in self.recorded if leg.basket_id == basket_id]


class FakePaperBenchmarkRepository:
    def __init__(self) -> None:
        self.closed: list[tuple] = []

    async def close_position(
        self, symbol, basket_id, exit_timestamp, paper_exit_price, real_exit_price
    ):
        self.closed.append((symbol, basket_id, real_exit_price))


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_signal(self, signal) -> None:
        pass

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


def _open_leg(basket_id: str = "RELIANCE.NS-cash-entry-1") -> LiveOrderLeg:
    return LiveOrderLeg(
        basket_id=basket_id, symbol="RELIANCE.NS", purpose="cash", tradingsymbol="RELIANCE",
        transaction_type="BUY", quantity=10, order_id="order-entry", status="COMPLETE",
        placed_at=datetime.now(UTC), average_price=Decimal("1500"),
    )


def _bracket(**overrides) -> GttBracket:
    defaults = dict(
        symbol="RELIANCE.NS", trigger_id=101, tradingsymbol="RELIANCE", quantity=10,
        entry_price=Decimal("1500"), stop_price=Decimal("1455"), target_price=Decimal("1650"),
        created_at=datetime.now(UTC), status="active",
    )
    defaults.update(overrides)
    return GttBracket(**defaults)


@pytest.mark.asyncio
async def test_no_op_when_nothing_is_open_for_the_symbol():
    result = await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), FakeOrderExecutor(),
        FakeGttRepository(), FakeLiveOrderRepository(open_leg=None), None, FakeNotifier(),
    )
    assert result.ok is False
    assert "No real open position" in result.message


@pytest.mark.asyncio
async def test_exits_and_cancels_the_live_gtt():
    executor = FakeOrderExecutor(gtt_status="active")
    gtt_repository = FakeGttRepository(active=_bracket())
    live_order_repository = FakeLiveOrderRepository(open_leg=_open_leg())
    notifier = FakeNotifier()

    result = await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), executor,
        gtt_repository, live_order_repository, None, notifier,
    )

    assert result.ok is True
    assert "closed" in result.message
    assert executor.deleted == [101]  # GTT was cancelled before the market exit
    assert gtt_repository.status_updates == [(101, "cancelled")]
    assert executor.calls == [("RELIANCE", "SELL", 10)]
    assert any("LIVE CASH POSITION CLOSED" in text for text in notifier.texts)


@pytest.mark.asyncio
async def test_reports_already_flat_when_the_gtt_already_fired():
    executor = FakeOrderExecutor(gtt_status="triggered")
    gtt_repository = FakeGttRepository(active=_bracket())
    live_order_repository = FakeLiveOrderRepository(open_leg=_open_leg())
    notifier = FakeNotifier()

    result = await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), executor,
        gtt_repository, live_order_repository, None, notifier,
    )

    assert result.ok is True
    assert "already flat" in result.message
    assert executor.calls == []  # never placed a market order against flat shares
    # 2026-08-28 regression: this used to just report the fact and move on,
    # leaving the ledger stuck "open" forever (exactly how COCHINSHIP.NS/
    # VMM.NS got stuck) -- it must now also close the ledger leg.
    assert len(live_order_repository.recorded) == 1
    reconciled = live_order_repository.recorded[0]
    assert reconciled.transaction_type == "SELL"
    assert reconciled.status == "COMPLETE"
    assert reconciled.quantity == 10
    assert any("RECONCILED" in text for text in notifier.texts)


@pytest.mark.asyncio
async def test_records_the_paper_benchmark_close_when_reconciling_a_flat_position():
    executor = FakeOrderExecutor(gtt_status="triggered")
    gtt_repository = FakeGttRepository(active=_bracket())
    entry_leg = _open_leg(basket_id="RELIANCE.NS-cash-entry-42")
    live_order_repository = FakeLiveOrderRepository(open_leg=entry_leg)
    paper_benchmark_repository = FakePaperBenchmarkRepository()

    result = await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), executor,
        gtt_repository, live_order_repository, paper_benchmark_repository, FakeNotifier(),
    )

    assert result.ok is True
    assert len(paper_benchmark_repository.closed) == 1
    symbol, basket_id, real_price = paper_benchmark_repository.closed[0]
    assert symbol == "RELIANCE.NS"
    assert basket_id == "RELIANCE.NS-cash-entry-42"
    assert real_price == Decimal("1550")


@pytest.mark.asyncio
async def test_reports_failure_when_the_exit_order_does_not_complete():
    executor = FakeOrderExecutor(gtt_status="active", fill_status="REJECTED")
    live_order_repository = FakeLiveOrderRepository(open_leg=_open_leg())

    result = await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), executor,
        None, live_order_repository, None, FakeNotifier(),
    )

    assert result.ok is False
    assert "did not complete" in result.message


@pytest.mark.asyncio
async def test_records_the_paper_benchmark_close_on_a_completed_exit():
    entry_leg = _open_leg(basket_id="RELIANCE.NS-cash-entry-42")
    live_order_repository = FakeLiveOrderRepository(open_leg=entry_leg)
    paper_benchmark_repository = FakePaperBenchmarkRepository()

    await manual_exit.exit_position(
        "RELIANCE.NS", Decimal("1550"), _config(), FakeOrderExecutor(),
        None, live_order_repository, paper_benchmark_repository, FakeNotifier(),
    )

    assert len(paper_benchmark_repository.closed) == 1
    symbol, basket_id, real_price = paper_benchmark_repository.closed[0]
    assert symbol == "RELIANCE.NS"
    assert basket_id == "RELIANCE.NS-cash-entry-42"
    assert real_price == Decimal("1550.0")
