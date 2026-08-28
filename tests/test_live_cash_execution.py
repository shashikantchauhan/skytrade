"""Tests for the real cash-equity order-execution flow (BUY/SELL trial,
see application/live_cash_execution.py). Everything here uses fakes -- no
real Kite connection, no real money.

Focus: the kill switch (its own -- live_cash_trading_enabled/symbols, not
the futures one) actually gates, quantity is sized from a fixed rupee
notional / market price (not a fixed share count), a real position is
never stacked, and a failed order still notifies without raising.
"""

from datetime import UTC, datetime, time
from decimal import Decimal

import pytest

from trading_scanner.application import live_cash_execution
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import LiveOrderLeg

_PRICE = Decimal("1000")  # Rs5,000 notional / Rs1,000 price = 5 shares, clean numbers


def _config(
    *, enabled: bool, symbols: frozenset[str] = frozenset({"RELIANCE.NS"}),
    notional: Decimal = Decimal("5000"), max_positions: int = 8,
    entry_cutoff_ist: time | None = None,
) -> AppConfig:
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
        live_cash_trading_enabled=enabled,
        live_cash_trading_symbols=symbols,
        live_cash_trading_notional=notional,
        live_cash_trading_max_positions=max_positions,
        # None (disabled) by default here -- the entry-cutoff feature is
        # unrelated to what most of these tests exercise, and leaving it at
        # its real default (15:15 IST) would make every test that doesn't
        # pass `now` to execute_cash_entry flaky depending on wall-clock
        # time. See test_entry_cutoff_* below for the dedicated coverage.
        live_cash_entry_cutoff_ist=entry_cutoff_ist,
    )


class FakeOrderExecutor:
    """Scripted fill outcomes keyed by (tradingsymbol, transaction_type) --
    mirrors test_live_execution.py's fake exactly, just against
    place_cash_market_order instead of place_market_order."""

    def __init__(
        self,
        scripted: dict[tuple[str, str], list[str]],
        holding_quantity: int | None = None,
        wait_for_fill_raises: bool = False,
    ) -> None:
        self._scripted = {key: list(values) for key, values in scripted.items()}
        self.calls: list[tuple[str, str, int]] = []
        self._order_counter = 0
        # None -- not configured -- raises, same as a real API error would.
        self._holding_quantity = holding_quantity
        self._wait_for_fill_raises = wait_for_fill_raises

    def holding_quantity(self, tradingsymbol):
        if self._holding_quantity is None:
            raise RuntimeError("holding_quantity not configured for this fake")
        return self._holding_quantity

    def place_cash_market_order(self, tradingsymbol, transaction_type, quantity, reference_price):
        self.calls.append((tradingsymbol, transaction_type, quantity))
        self._order_counter += 1
        return f"order-{self._order_counter}"

    def wait_for_fill(self, order_id, timeout_seconds, poll_interval=1.0):
        if self._wait_for_fill_raises:
            raise RuntimeError("Couldn't find that `order_id`.")
        index = self._order_counter - 1
        tradingsymbol, transaction_type, _ = self.calls[index]
        key = (tradingsymbol, transaction_type)
        status = self._scripted[key].pop(0)
        return {"status": status, "average_price": 1000.0, "status_message": None}


class FakeLiveOrderRepository:
    def __init__(
        self, open_cash: list[LiveOrderLeg] | None = None,
        all_open_cash: list[LiveOrderLeg] | None = None,
    ) -> None:
        self.recorded: list[LiveOrderLeg] = []
        self._open_cash = open_cash or []
        # Defaults to the same list as get_open_cash_legs -- fine for
        # single-symbol tests; pass all_open_cash explicitly to simulate
        # other symbols' positions also being open (max_positions tests).
        self._all_open_cash = all_open_cash if all_open_cash is not None else self._open_cash

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        self.recorded.append(leg)

    async def get_open_cash_legs(self, symbol: str):
        return self._open_cash

    async def get_all_open_cash_legs(self):
        return self._all_open_cash


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_signal(self, signal) -> None:
        pass

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


@pytest.mark.asyncio
async def test_entry_noop_when_cash_trading_disabled():
    config = _config(enabled=False)
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert result is None
    assert executor.calls == []
    assert notifier.texts == []


@pytest.mark.asyncio
async def test_entry_noop_when_symbol_not_allowlisted():
    config = _config(enabled=True, symbols=frozenset({"TCS.NS"}))
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_sizes_quantity_from_notional_over_price():
    config = _config(enabled=True, notional=Decimal("5000"))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    # Rs5,000 / Rs1,000 = 5 shares, .NS stripped to the Kite tradingsymbol.
    assert executor.calls == [("RELIANCE", "BUY", 5)]
    assert [leg.purpose for leg in repo.recorded] == ["cash"]
    assert any("LIVE CASH ORDER PLACED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_sizing_floors_to_at_least_one_share():
    # Rs5,000 notional against a Rs12,000 stock would floor to 0 -- must
    # still buy 1 share rather than skip the trade silently.
    config = _config(enabled=True, notional=Decimal("5000"))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()

    await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", Decimal("12000"), config, executor, repo, FakeNotifier()
    )

    assert executor.calls == [("RELIANCE", "BUY", 1)]


@pytest.mark.asyncio
async def test_entry_refuses_to_stack_a_second_position():
    config = _config(enabled=True)
    already_open = [
        LiveOrderLeg(
            basket_id="x", symbol="RELIANCE.NS", purpose="cash", tradingsymbol="RELIANCE",
            transaction_type="BUY", quantity=5, order_id="o1", status="COMPLETE",
            placed_at=datetime.now(UTC),
        )
    ]
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(open_cash=already_open)

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_blocked_when_max_positions_already_open():
    # A wide allowlist (e.g. the full universe) must not widen real capital
    # at risk -- once max_positions real positions are open *anywhere*,
    # a new symbol (itself not yet open) still gets refused.
    config = _config(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
    other_symbols_open = [
        LiveOrderLeg(
            basket_id=f"x{i}", symbol=f"SYM{i}.NS", purpose="cash", tradingsymbol=f"SYM{i}",
            transaction_type="BUY", quantity=5, order_id=f"o{i}", status="COMPLETE",
            placed_at=datetime.now(UTC),
        )
        for i in range(2)
    ]
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(open_cash=[], all_open_cash=other_symbols_open)

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_allowed_when_under_max_positions():
    config = _config(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
    one_other_open = [
        LiveOrderLeg(
            basket_id="x", symbol="TCS.NS", purpose="cash", tradingsymbol="TCS",
            transaction_type="BUY", quantity=5, order_id="o1", status="COMPLETE",
            placed_at=datetime.now(UTC),
        )
    ]
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository(open_cash=[], all_open_cash=one_other_open)

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, FakeNotifier()
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


@pytest.mark.asyncio
async def test_entry_failure_notifies_but_does_not_raise():
    config = _config(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["REJECTED"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    assert repo.recorded[0].status == "REJECTED"
    assert any("LIVE CASH ORDER FAILED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_exit_still_fires_when_toggle_is_disabled():
    # A "Stop" click (enabled=False) must never strand a real open
    # position with no way to be closed -- exit is gated on "is anything
    # actually open," not on the current toggle state.
    config = _config(enabled=False)
    open_leg = LiveOrderLeg(
        basket_id="RELIANCE.NS-cash-entry-x", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor({("RELIANCE", "SELL"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository(open_cash=[open_leg])
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_exit(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    assert executor.calls == [("RELIANCE", "SELL", 5)]


@pytest.mark.asyncio
async def test_exit_noop_when_nothing_open():
    config = _config(enabled=True)
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(open_cash=[])

    result = await live_cash_execution.execute_cash_exit(
        "RELIANCE.NS", _PRICE, config, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_exit_squares_off_the_open_buy_leg_with_a_sell():
    config = _config(enabled=True)
    open_leg = LiveOrderLeg(
        basket_id="RELIANCE.NS-cash-entry-x", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor({("RELIANCE", "SELL"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository(open_cash=[open_leg])
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_exit(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    assert executor.calls == [("RELIANCE", "SELL", 5)]
    assert any("LIVE CASH POSITION CLOSED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_record_broker_side_exit_closes_the_ledger_without_placing_an_order():
    # See gtt_bracket.reconcile_before_exit's 2026-08-28 fix -- callers use
    # this directly once they've already determined (via real holding
    # quantity) that a position is flat; execute_cash_exit itself no
    # longer duplicates that check (see its 2026-08-28 docstring note --
    # the duplicate check was what tipped Kite's rate limit and cost the
    # UNIONBANK.NS order-tracking incident the same day).
    open_leg = LiveOrderLeg(
        basket_id="VMM.NS-cash-entry-x", symbol="VMM.NS", purpose="cash",
        tradingsymbol="VMM", transaction_type="BUY", quantity=44, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC),
    )
    repo = FakeLiveOrderRepository(open_cash=[open_leg])
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.record_broker_side_exit(
        "VMM.NS", open_leg, _PRICE, repo, notifier
    )

    assert basket_id is not None
    assert repo.recorded[0].status == "COMPLETE"
    assert repo.recorded[0].transaction_type == "SELL"
    assert repo.recorded[0].order_id == "RECONCILED"
    assert any("RECONCILED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_exit_records_unknown_status_when_fill_check_fails_but_never_loses_the_order():
    # 2026-08-28 regression: a real SELL was placed and filled for
    # UNIONBANK.NS, but a transient Kite API error while polling its fill
    # status raised straight out of _place_and_wait -- the order was never
    # recorded anywhere, even though it had genuinely already happened.
    # order_id must always survive to the ledger regardless of what
    # wait_for_fill does.
    config = _config(enabled=True)
    open_leg = LiveOrderLeg(
        basket_id="UNIONBANK.NS-cash-entry-x", symbol="UNIONBANK.NS", purpose="cash",
        tradingsymbol="UNIONBANK", transaction_type="BUY", quantity=26, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor({}, wait_for_fill_raises=True)
    repo = FakeLiveOrderRepository(open_cash=[open_leg])
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_exit(
        "UNIONBANK.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    assert executor.calls == [("UNIONBANK", "SELL", 26)]  # the order WAS placed
    assert len(repo.recorded) == 1
    assert repo.recorded[0].status == "UNKNOWN"
    assert repo.recorded[0].order_id == "order-1"  # never lost, even though unconfirmed
    assert any("LIVE CASH EXIT INCOMPLETE" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_exit_incomplete_notifies_distinctly():
    config = _config(enabled=True)
    open_leg = LiveOrderLeg(
        basket_id="RELIANCE.NS-cash-entry-x", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC),
    )
    executor = FakeOrderExecutor({("RELIANCE", "SELL"): ["REJECTED"]})
    repo = FakeLiveOrderRepository(open_cash=[open_leg])
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_exit(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier
    )

    assert basket_id is not None
    assert any("LIVE CASH EXIT INCOMPLETE" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_noop_when_past_the_cutoff():
    # 2026-08-25 regression: real orders placed in NSE's final ~10-15
    # minutes either took far longer than the fill-timeout to match, or got
    # cancelled outright by the exchange for lack of a counterparty.
    config = _config(enabled=True, entry_cutoff_ist=time(15, 15))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    past_cutoff = datetime(2026, 8, 25, 9, 50, tzinfo=UTC)  # 15:20 IST

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier, now=past_cutoff,
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_allowed_right_up_to_the_cutoff():
    config = _config(enabled=True, entry_cutoff_ist=time(15, 15))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    before_cutoff = datetime(2026, 8, 25, 9, 44, tzinfo=UTC)  # 15:14 IST

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier, now=before_cutoff,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


@pytest.mark.asyncio
async def test_entry_cutoff_disabled_when_none():
    config = _config(enabled=True, entry_cutoff_ist=None)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    late = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)  # 15:30 IST, market close itself

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, executor, repo, notifier, now=late,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]
