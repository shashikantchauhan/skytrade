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
from trading_scanner.domain import order_intent
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.infrastructure.db import LiveCashToggleState

_PRICE = Decimal("1000")  # Rs5,000 notional / Rs1,000 price = 5 shares, clean numbers


@pytest.fixture(autouse=True)
def _no_real_retry_backoff(monkeypatch):
    """execute_cash_entry's retry loop sleeps _ENTRY_RETRY_BACKOFF_SECONDS
    between attempts -- without this, any test whose FakeOrderExecutor
    scripts a REJECTED/CANCELLED outcome (most of them) would incur a real
    ~15s sleep per retry. Real backoff timing isn't what any test here
    verifies -- only behavior/ordering -- so it's zeroed for every test in
    this file."""
    monkeypatch.setattr(live_cash_execution, "_ENTRY_RETRY_BACKOFF_SECONDS", 0)


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


def _cash_state(
    *, enabled: bool, symbols: frozenset[str] = frozenset({"RELIANCE.NS"}),
    notional: Decimal = Decimal("5000"), max_positions: int = 8,
) -> LiveCashToggleState:
    """Mirrors ``_config``'s cash-related kwargs -- ``execute_cash_entry``
    now takes this as its own explicit parameter instead of reading these
    4 fields off ``AppConfig`` (see live_cash_execution.py's module
    docstring for why)."""
    return LiveCashToggleState(
        enabled=enabled, symbols=symbols, notional=notional, max_positions=max_positions
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
        tagged_order: dict | None = None,
        find_by_tag_raises: bool = False,
    ) -> None:
        self._scripted = {key: list(values) for key, values in scripted.items()}
        self.calls: list[tuple[str, str, int]] = []
        self.tags_used: list[str | None] = []
        self._order_counter = 0
        # None -- not configured -- raises, same as a real API error would.
        self._holding_quantity = holding_quantity
        self._wait_for_fill_raises = wait_for_fill_raises
        # A single pre-scripted order for find_todays_order_by_tag to
        # "find" (the P0 broker-ground-truth preflight) -- None means
        # nothing is sitting in the broker's order book under any tag.
        self._tagged_order = tagged_order
        self._find_by_tag_raises = find_by_tag_raises

    def holding_quantity(self, tradingsymbol):
        if self._holding_quantity is None:
            raise RuntimeError("holding_quantity not configured for this fake")
        return self._holding_quantity

    def find_todays_order_by_tag(self, tag):
        if self._find_by_tag_raises:
            raise RuntimeError("Kite order-book lookup failed")
        if self._tagged_order is not None and self._tagged_order.get("tag") == tag:
            return self._tagged_order
        return None

    def place_cash_market_order(
        self, tradingsymbol, transaction_type, quantity, reference_price, tag=None
    ):
        self.calls.append((tradingsymbol, transaction_type, quantity))
        self.tags_used.append(tag)
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
        unclosed_cash: list[LiveOrderLeg] | None = None,
        legs_by_intent: dict[str, list[LiveOrderLeg]] | None = None,
    ) -> None:
        self.recorded: list[LiveOrderLeg] = []
        self._open_cash = open_cash or []
        # Defaults to the same list as get_open_cash_legs -- fine for
        # single-symbol tests; pass all_open_cash explicitly to simulate
        # other symbols' positions also being open (max_positions tests).
        self._all_open_cash = all_open_cash if all_open_cash is not None else self._open_cash
        # Defaults to the same list too -- execute_cash_entry's stacking
        # check now uses get_unclosed_cash_legs, not get_open_cash_legs;
        # pass unclosed_cash explicitly to simulate an OPEN/UNKNOWN leg
        # that get_open_cash_legs itself wouldn't count.
        self._unclosed_cash = unclosed_cash if unclosed_cash is not None else self._open_cash
        # Phase 4 (domain/order_intent.py): intent_id -> legs already
        # recorded under it -- empty by default (no prior attempt), pass
        # explicitly to simulate a crash-then-restart scenario.
        self._legs_by_intent = legs_by_intent or {}

    async def get_unclosed_cash_legs(self, symbol: str):
        return self._unclosed_cash

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        self.recorded.append(leg)

    async def get_open_cash_legs(self, symbol: str):
        return self._open_cash

    async def get_all_open_cash_legs(self):
        return self._all_open_cash

    async def get_all_unclosed_cash_legs(self):
        # execute_cash_entry's max_positions check now goes through
        # broker_reconciliation.get_all_unclosed_positions -- same list as
        # get_all_open_cash_legs for these tests (none script an UNKNOWN
        # leg specifically for the capacity check; that has its own
        # dedicated coverage in test_broker_reconciliation.py).
        return self._all_open_cash

    async def get_legs_by_intent(self, intent_id: str):
        return self._legs_by_intent.get(intent_id, [])


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


@pytest.mark.asyncio
async def test_entry_noop_when_cash_trading_disabled():
    config = _config(enabled=False)
    cash_state = _cash_state(enabled=False)
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
    )

    assert result is None
    assert executor.calls == []
    assert notifier.texts == []


@pytest.mark.asyncio
async def test_entry_noop_when_symbol_not_allowlisted():
    config = _config(enabled=True, symbols=frozenset({"TCS.NS"}))
    cash_state = _cash_state(enabled=True, symbols=frozenset({"TCS.NS"}))
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_sizes_quantity_from_notional_over_price():
    config = _config(enabled=True, notional=Decimal("5000"))
    cash_state = _cash_state(enabled=True, notional=Decimal("5000"))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
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
    cash_state = _cash_state(enabled=True, notional=Decimal("5000"))
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()

    await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", Decimal("12000"), config, cash_state, executor, repo, FakeNotifier()
    )

    assert executor.calls == [("RELIANCE", "BUY", 1)]


@pytest.mark.asyncio
async def test_entry_refuses_to_stack_a_second_position():
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
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
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_refuses_to_stack_when_the_first_orders_fill_is_still_unconfirmed():
    # 2026-08-28 regression: PERSISTENT.NS's first BUY order was placed for
    # real, but wait_for_fill's poll timed out while Kite still reported it
    # as OPEN (not yet COMPLETE) -- get_open_cash_legs only counts COMPLETE
    # legs, so the next cycle saw an apparently-empty position and bought a
    # second time. An OPEN (or UNKNOWN) leg must block a second entry too.
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    still_pending = [
        LiveOrderLeg(
            basket_id="x", symbol="PERSISTENT.NS", purpose="cash", tradingsymbol="PERSISTENT",
            transaction_type="BUY", quantity=8, order_id="o1", status="OPEN",
            placed_at=datetime.now(UTC),
        )
    ]
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(open_cash=[], unclosed_cash=still_pending)

    result = await live_cash_execution.execute_cash_entry(
        "PERSISTENT.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_refuses_a_duplicate_under_the_same_intent_after_a_restart():
    # Phase 4 (domain/order_intent.py): simulates a process that placed a
    # real order, recorded it, then crashed/restarted before the caller
    # saw the result -- the next attempt at the exact same signal (same
    # symbol/side/signal_timestamp) computes the same intent_id and must
    # refuse to place a second real order, even though get_unclosed_cash_
    # legs (a plain per-symbol check) would independently already catch
    # this too -- this is the intent-keyed line of defense, tested in
    # isolation via a repo where the symbol-level checks are empty but the
    # intent-level lookup is not.
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    intent_id = order_intent.compute_intent_id("RELIANCE.NS", "BUY", signal_timestamp, "cash")
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    already_recorded = [
        LiveOrderLeg(
            basket_id="x", symbol="RELIANCE.NS", purpose="cash", tradingsymbol="RELIANCE",
            transaction_type="BUY", quantity=5, order_id="o1", status="COMPLETE",
            placed_at=datetime.now(UTC), intent_id=intent_id,
        )
    ]
    executor = FakeOrderExecutor({})
    repo = FakeLiveOrderRepository(legs_by_intent={intent_id: already_recorded})

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier(),
        signal_timestamp=signal_timestamp,
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_allowed_when_the_same_intent_only_has_rejected_legs():
    # A prior attempt at this exact signal that only ever got REJECTED/
    # CANCELLED legs must NOT block a fresh attempt -- that's a real
    # signal that's simply never resulted in a live order, not evidence of
    # a duplicate in flight.
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    intent_id = order_intent.compute_intent_id("RELIANCE.NS", "BUY", signal_timestamp, "cash")
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    only_rejected = [
        LiveOrderLeg(
            basket_id="x", symbol="RELIANCE.NS", purpose="cash", tradingsymbol="RELIANCE",
            transaction_type="BUY", quantity=5, order_id="o1", status="REJECTED",
            placed_at=datetime.now(UTC), intent_id=intent_id,
        )
    ]
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository(legs_by_intent={intent_id: only_rejected})

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier(),
        signal_timestamp=signal_timestamp,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


# --- P0 broker-crash-window preflight (2026-09-01) -------------------------
# The gap intent_id alone can't close: Kite accepts an order, the process
# dies before _record ever runs, and a fresh attempt at the exact same
# signal (new process, same deterministic intent_id) has zero local memory
# of it. See _broker_ground_truth_preflight's own docstring.


@pytest.mark.asyncio
async def test_entry_tags_the_real_order_with_the_intent_id():
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    intent_id = order_intent.compute_intent_id("RELIANCE.NS", "BUY", signal_timestamp, "cash")
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier(),
        signal_timestamp=signal_timestamp,
    )

    assert result is not None
    assert executor.tags_used == [intent_id[:20]]


@pytest.mark.asyncio
async def test_entry_reconciles_instead_of_duplicating_when_broker_already_has_a_tagged_order():
    # The core P0 scenario: this process has no local record of this intent
    # at all (legs_by_intent is empty, same as a totally fresh process), but
    # Kite's own order book already has a COMPLETE order tagged with this
    # exact intent -- proof a previous attempt got through the broker
    # before whatever recorded it locally crashed. Must reconcile, not
    # place a second real BUY.
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    intent_id = order_intent.compute_intent_id("RELIANCE.NS", "BUY", signal_timestamp, "cash")
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor(
        {("RELIANCE", "BUY"): ["COMPLETE"]},
        tagged_order={
            "tag": intent_id[:20], "order_id": "broker-order-1", "status": "COMPLETE",
            "quantity": 5, "average_price": 1001.5,
        },
    )
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier,
        signal_timestamp=signal_timestamp,
    )

    assert result is None  # never returns a fresh basket_id -- no new order placed
    assert executor.calls == []  # place_cash_market_order was never called
    assert len(repo.recorded) == 1
    reconciled = repo.recorded[0]
    assert reconciled.order_id == "broker-order-1"
    assert reconciled.status == "COMPLETE"
    assert reconciled.quantity == 5
    assert reconciled.intent_id == intent_id
    assert any("RECONCILIATION REQUIRED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_proceeds_when_the_tagged_broker_order_is_confirmed_rejected():
    # A tagged order found in the broker's book with a REJECTED/CANCELLED
    # status is definitive proof no real position resulted -- must not
    # block a fresh, legitimate attempt at the same signal.
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    intent_id = order_intent.compute_intent_id("RELIANCE.NS", "BUY", signal_timestamp, "cash")
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor(
        {("RELIANCE", "BUY"): ["COMPLETE"]},
        tagged_order={
            "tag": intent_id[:20], "order_id": "broker-order-1", "status": "REJECTED",
            "quantity": 5, "average_price": None,
        },
    )
    repo = FakeLiveOrderRepository()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier(),
        signal_timestamp=signal_timestamp,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]  # the fresh attempt was placed
    # both the audit-trail reconciliation row AND the fresh COMPLETE fill
    # are recorded, in that order.
    assert [leg.status for leg in repo.recorded] == ["REJECTED", "COMPLETE"]


@pytest.mark.asyncio
async def test_entry_blocked_when_broker_shows_a_real_holding_with_no_local_record():
    # No tagged order (e.g. the signal that caused it will never recur, or
    # the tag lookup missed it) but Kite's real holding_quantity shows
    # shares are actually held -- the broader "hidden position" case
    # (spec's test case 8: existing real broker position but missing local
    # ledger row).
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]}, holding_quantity=5)
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier,
        signal_timestamp=signal_timestamp,
    )

    assert result is None
    assert executor.calls == []
    assert len(repo.recorded) == 1
    reconciled = repo.recorded[0]
    assert reconciled.order_id == "RECONCILED-HOLDING"
    assert reconciled.status == "COMPLETE"
    assert reconciled.quantity == 5
    assert any("RECONCILIATION REQUIRED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_proceeds_when_broker_preflight_checks_are_unavailable():
    # Both broker calls failing (a transient Kite API error) must NOT block
    # every fresh entry -- the risk being closed is what happens when the
    # check *succeeds* and finds real evidence, not when the check itself
    # is unavailable. Matches gtt_bracket.reconcile_before_exit's own
    # fallback discipline.
    signal_timestamp = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor(
        {("RELIANCE", "BUY"): ["COMPLETE"]}, find_by_tag_raises=True,
        # holding_quantity left unconfigured -- raises, same as a real
        # infrastructure failure would.
    )
    repo = FakeLiveOrderRepository()

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier(),
        signal_timestamp=signal_timestamp,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


@pytest.mark.asyncio
async def test_entry_blocked_when_max_positions_already_open():
    # A wide allowlist (e.g. the full universe) must not widen real capital
    # at risk -- once max_positions real positions are open *anywhere*,
    # a new symbol (itself not yet open) still gets refused.
    config = _config(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
    cash_state = _cash_state(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
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
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier()
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_allowed_when_under_max_positions():
    config = _config(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
    cash_state = _cash_state(enabled=True, symbols=frozenset({"RELIANCE.NS"}), max_positions=2)
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
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, FakeNotifier()
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


@pytest.mark.asyncio
async def test_entry_failure_notifies_but_does_not_raise():
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["REJECTED"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
    )

    assert basket_id is not None
    assert repo.recorded[0].status == "REJECTED"
    assert any("LIVE CASH ORDER FAILED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_retries_a_rejected_order_and_succeeds():
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["REJECTED", "COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
    )

    max_attempts = live_cash_execution._MAX_ENTRY_ATTEMPTS
    assert basket_id is not None
    assert executor.calls == [("RELIANCE", "BUY", 5), ("RELIANCE", "BUY", 5)]  # 2 real attempts
    assert [leg.status for leg in repo.recorded] == ["REJECTED", "COMPLETE"]  # both recorded
    assert any(
        "LIVE CASH ORDER PLACED" in t and f"attempt 2/{max_attempts}" in t for t in notifier.texts
    )
    assert not any("FAILED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_entry_gives_up_after_max_attempts_all_rejected():
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    max_attempts = live_cash_execution._MAX_ENTRY_ATTEMPTS
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["REJECTED"] * max_attempts})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
    )

    assert basket_id is not None
    assert len(executor.calls) == max_attempts  # no more than this
    assert len(repo.recorded) == max_attempts
    assert any(
        "LIVE CASH ORDER FAILED" in t and f"after {max_attempts} attempts" in t
        for t in notifier.texts
    )


@pytest.mark.asyncio
async def test_entry_does_not_retry_an_unconfirmed_order():
    # UNKNOWN means a real order state might already exist at the broker
    # (the fill-status check itself failed, not the order) -- retrying
    # would risk placing a second real order on top of one that may have
    # already filled. Same reasoning as OPEN: never retry past it.
    config = _config(enabled=True)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({}, wait_for_fill_raises=True)
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier
    )

    assert basket_id is not None
    assert len(executor.calls) == 1  # exactly one attempt, no retry
    assert repo.recorded[0].status == "UNKNOWN"


@pytest.mark.asyncio
async def test_entry_retry_abandoned_once_past_the_cutoff(monkeypatch):
    class _AdvancingClock:
        """Stands in for the ``datetime`` name inside
        live_cash_execution.py -- only ``.now(tz)`` is ever called on it
        there. Returns ``before`` on the first call (so the pre-loop
        cutoff check and attempt 1's own timestamps look normal), ``after``
        on every call from the second onward -- simulates real time
        (specifically, crossing the entry cutoff) passing during the
        retry backoff without an actual sleep."""

        def __init__(self, before: datetime, after: datetime) -> None:
            self._before = before
            self._after = after
            self._calls = 0

        def now(self, tz=None):
            self._calls += 1
            return self._before if self._calls == 1 else self._after

    config = _config(enabled=True, entry_cutoff_ist=time(15, 15))
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["REJECTED", "COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    before_cutoff = datetime(2026, 8, 25, 9, 44, tzinfo=UTC)  # 15:14 IST
    after_cutoff = datetime(2026, 8, 25, 9, 50, tzinfo=UTC)  # 15:20 IST
    monkeypatch.setattr(
        live_cash_execution, "datetime", _AdvancingClock(before_cutoff, after_cutoff)
    )

    basket_id = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier, now=before_cutoff,
    )

    assert basket_id is not None
    assert len(executor.calls) == 1  # retry abandoned -- never got a second real attempt
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
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    past_cutoff = datetime(2026, 8, 25, 9, 50, tzinfo=UTC)  # 15:20 IST

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier, now=past_cutoff,
    )

    assert result is None
    assert executor.calls == []


@pytest.mark.asyncio
async def test_entry_allowed_right_up_to_the_cutoff():
    config = _config(enabled=True, entry_cutoff_ist=time(15, 15))
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    before_cutoff = datetime(2026, 8, 25, 9, 44, tzinfo=UTC)  # 15:14 IST

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier, now=before_cutoff,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]


@pytest.mark.asyncio
async def test_entry_cutoff_disabled_when_none():
    config = _config(enabled=True, entry_cutoff_ist=None)
    cash_state = _cash_state(enabled=True)
    executor = FakeOrderExecutor({("RELIANCE", "BUY"): ["COMPLETE"]})
    repo = FakeLiveOrderRepository()
    notifier = FakeNotifier()
    late = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)  # 15:30 IST, market close itself

    result = await live_cash_execution.execute_cash_entry(
        "RELIANCE.NS", _PRICE, config, cash_state, executor, repo, notifier, now=late,
    )

    assert result is not None
    assert executor.calls == [("RELIANCE", "BUY", 5)]
