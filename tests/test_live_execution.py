"""Tests for the real order-execution basket entry/exit flow.

Everything here uses fakes -- no real Kite connection, no real money.
Focus: the kill switch actually gates, leg sequencing is option-then-future
on entry (futures-then-option on exit), and a failed second leg on entry
triggers a rollback of the first, with a Telegram alert either way.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import live_execution
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg, SignalSide


def _config(*, enabled: bool, symbols: frozenset[str] = frozenset({"RELIANCE.NS"})) -> AppConfig:
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
        live_trading_enabled=enabled,
        live_trading_symbols=symbols,
        live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )


class FakeDerivativesChain:
    def __init__(self, option: dict | None, future: dict | None) -> None:
        self._option = option
        self._future = future
        self.option_calls: list[tuple[str, str, float]] = []

    def nearest_atm_option(self, symbol, option_type, underlying_price):
        self.option_calls.append((symbol, option_type, underlying_price))
        return self._option

    def nearest_future(self, symbol):
        return self._future


class FakeOrderExecutor:
    """Scripted fill outcomes keyed by (tradingsymbol, transaction_type) --
    each call to place_market_order consumes the next scripted status for
    that key, in order."""

    def __init__(self, scripted: dict[tuple[str, str], list[str]]) -> None:
        self._scripted = {key: list(values) for key, values in scripted.items()}
        self.calls: list[tuple[str, str, int]] = []
        self._order_counter = 0

    def place_market_order(self, tradingsymbol, transaction_type, quantity):
        self.calls.append((tradingsymbol, transaction_type, quantity))
        self._order_counter += 1
        return f"order-{self._order_counter}"

    def order_status(self, order_id):
        raise NotImplementedError

    def wait_for_fill(self, order_id, timeout_seconds, poll_interval=1.0):
        index = self._order_counter - 1
        # Match by the most recent call's key.
        tradingsymbol, transaction_type, _ = self.calls[index]
        key = (tradingsymbol, transaction_type)
        status = self._scripted[key].pop(0)
        return {"status": status, "average_price": 100.0, "status_message": None}


class FakeLiveOrderRepository:
    def __init__(self, open_primary: list[LiveOrderLeg] | None = None) -> None:
        self.recorded: list[LiveOrderLeg] = []
        self._open_primary = open_primary or []

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        self.recorded.append(leg)

    async def get_open_primary_legs(self, symbol: str):
        return self._open_primary

    async def get_legs(self, basket_id: str):
        return [leg for leg in self.recorded if leg.basket_id == basket_id]


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_signal(self, signal) -> None:
        pass

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


_OPTION_CONTRACT = {"tradingsymbol": "RELIANCE25AUG1400PE", "strike": 1400, "lot_size": 250}
_FUTURE_CONTRACT = {"tradingsymbol": "RELIANCE25AUGFUT", "lot_size": 250}


@pytest.mark.asyncio
async def test_entry_noop_when_live_trading_disabled():
    config = _config(enabled=False)
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    result = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, notifier,
    )

    assert result is None
    assert executor.calls == []
    assert notifier.texts == []


@pytest.mark.asyncio
async def test_entry_noop_when_symbol_not_allowlisted():
    config = _config(enabled=True, symbols=frozenset({"TCS.NS"}))
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()

    result = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, FakeNotifier(),
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_places_option_before_futures_when_both_fill():
    config = _config(enabled=True)
    executor = FakeOrderExecutor(
        {
            ("RELIANCE25AUG1400PE", "BUY"): ["COMPLETE"],
            ("RELIANCE25AUGFUT", "BUY"): ["COMPLETE"],
        }
    )
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, notifier,
    )

    assert basket_id is not None
    # Option leg placed strictly before the futures leg.
    assert executor.calls == [
        ("RELIANCE25AUG1400PE", "BUY", 250),
        ("RELIANCE25AUGFUT", "BUY", 250),
    ]
    assert [leg.purpose for leg in repo.recorded] == ["hedge", "primary"]
    assert any("LIVE ORDER PLACED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_aborts_without_futures_leg_if_option_fails():
    config = _config(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE25AUG1400PE", "BUY"): ["REJECTED"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, notifier,
    )

    assert basket_id is not None
    assert executor.calls == [("RELIANCE25AUG1400PE", "BUY", 250)]  # futures never placed
    assert any("LIVE ORDER FAILED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_rolls_back_option_leg_if_futures_leg_fails():
    config = _config(enabled=True)
    executor = FakeOrderExecutor(
        {
            ("RELIANCE25AUG1400PE", "BUY"): ["COMPLETE"],
            ("RELIANCE25AUGFUT", "BUY"): ["REJECTED"],
            ("RELIANCE25AUG1400PE", "SELL"): ["COMPLETE"],
        }
    )
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, notifier,
    )

    assert basket_id is not None
    assert executor.calls == [
        ("RELIANCE25AUG1400PE", "BUY", 250),
        ("RELIANCE25AUGFUT", "BUY", 250),
        ("RELIANCE25AUG1400PE", "SELL", 250),  # rollback: unwind the option leg
    ]
    assert any("LIVE ORDER FAILED" in t and "squared off" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_skips_if_already_holding_a_real_position():
    config = _config(enabled=True)
    existing_leg = LiveOrderLeg(
        basket_id="RELIANCE.NS-entry-old",
        symbol="RELIANCE.NS",
        purpose="primary",
        tradingsymbol="RELIANCE25AUGFUT",
        transaction_type="BUY",
        quantity=250,
        order_id="order-old",
        status="COMPLETE",
        placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(open_primary=[existing_leg])

    result = await live_execution.execute_basket_entry(
        "RELIANCE.NS", SignalSide.BUY, "PE", Decimal("1400"),
        config, FakeDerivativesChain(_OPTION_CONTRACT, _FUTURE_CONTRACT),
        executor, repo, FakeNotifier(),
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_exit_closes_futures_before_option():
    config = _config(enabled=True)
    open_future = LiveOrderLeg(
        basket_id="RELIANCE.NS-entry-1",
        symbol="RELIANCE.NS",
        purpose="primary",
        tradingsymbol="RELIANCE25AUGFUT",
        transaction_type="BUY",
        quantity=250,
        order_id="order-1",
        status="COMPLETE",
        placed_at=datetime.now(UTC),
    )
    open_hedge = LiveOrderLeg(
        basket_id="RELIANCE.NS-entry-1",
        symbol="RELIANCE.NS",
        purpose="hedge",
        tradingsymbol="RELIANCE25AUG1400PE",
        transaction_type="BUY",
        quantity=250,
        order_id="order-0",
        status="COMPLETE",
        placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor(
        {
            ("RELIANCE25AUGFUT", "SELL"): ["COMPLETE"],
            ("RELIANCE25AUG1400PE", "SELL"): ["COMPLETE"],
        }
    )
    repo = FakeLiveOrderRepository(open_primary=[open_future])
    repo.recorded = [open_hedge, open_future]  # pre-seed so get_legs finds the hedge
    notifier = FakeNotifier()

    basket_id = await live_execution.execute_basket_exit(
        "RELIANCE.NS", config, executor, repo, notifier
    )

    assert basket_id is not None
    calls = [c for c in executor.calls]
    assert calls[0] == ("RELIANCE25AUGFUT", "SELL", 250)  # futures closed first
    assert calls[1] == ("RELIANCE25AUG1400PE", "SELL", 250)  # then the hedge
    assert any("LIVE POSITION CLOSED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_exit_noop_when_nothing_open():
    config = _config(enabled=True)
    repo = FakeLiveOrderRepository(open_primary=[])

    result = await live_execution.execute_basket_exit(
        "RELIANCE.NS", config, FakeOrderExecutor({}), repo, FakeNotifier()
    )

    assert result is None
