"""Tests for application/pipeline/entry_decision.py's _finalize_cash_entry
-- specifically the 2026-09-01 fix for a leftover instance of the UNKNOWN-
fill blind spot (see docs/incidents/2026-09-01-unknown-fill-blind-spot.md):
this call site used get_open_cash_legs (COMPLETE-only) right after a real
BUY was attempted, so an UNKNOWN fill (unconfirmed, but possibly real) got
no GTT bracket and triggered a "missed signal" notification even though a
real order may have gone through."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application.pipeline import entry_decision
from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.infrastructure.db import LiveCashToggleState


def _cash_state() -> LiveCashToggleState:
    return LiveCashToggleState(
        enabled=True, symbols=frozenset({"RELIANCE.NS"}),
        notional=Decimal("50000"), max_positions=8,
    )


def _leg(**overrides) -> LiveOrderLeg:
    defaults = dict(
        basket_id="RELIANCE.NS-cash-entry-1", symbol="RELIANCE.NS", purpose="cash",
        tradingsymbol="RELIANCE", transaction_type="BUY", quantity=5, order_id="o1",
        status="COMPLETE", placed_at=datetime.now(UTC), average_price=Decimal("1500"),
    )
    defaults.update(overrides)
    return LiveOrderLeg(**defaults)


class _FakeLiveOrderRepository:
    def __init__(self, unclosed: list[LiveOrderLeg] | None = None) -> None:
        self._unclosed = unclosed or []

    async def get_unclosed_cash_legs(self, symbol: str):
        return self._unclosed

    async def get_all_unclosed_cash_legs(self):
        # _notify_missed_cash_entry's own capacity-count lookup -- reused
        # here for the "nothing open" test below, which only cares that
        # this doesn't raise, not about the exact reason text.
        return self._unclosed


class _FakeOrderExecutor:
    def tick_size(self, tradingsymbol):
        return Decimal("0.05")

    def place_cash_bracket_gtt(self, tradingsymbol, quantity, last_price, stop_price, target_price):
        return 1001


class _FakeGttRepository:
    def __init__(self) -> None:
        self.recorded = []

    async def record(self, bracket) -> None:
        self.recorded.append(bracket)


class _FakePaperBenchmarkRepository:
    def __init__(self) -> None:
        self.opened = []

    async def open_position(self, position) -> None:
        self.opened.append(position)


class _FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


@pytest.mark.asyncio
async def test_an_unknown_status_fill_still_gets_a_gtt_bracket_and_reports_opened():
    # The bug: get_open_cash_legs (COMPLETE-only) saw nothing for an
    # UNKNOWN leg, so no bracket was placed and the caller fell through to
    # "MISSED BUY SIGNAL" even though a real order may have filled.
    unknown_leg = _leg(status="UNKNOWN", average_price=None)
    live_order_repository = _FakeLiveOrderRepository(unclosed=[unknown_leg])
    gtt_repository = _FakeGttRepository()
    notifier = _FakeNotifier()

    note = await entry_decision._finalize_cash_entry(
        "RELIANCE.NS", Decimal("1500"), _cash_state(), live_order_repository, notifier,
        gtt_repository, None, _FakeOrderExecutor(),
    )

    assert "opened" in note
    assert len(gtt_repository.recorded) == 1  # the bracket WAS placed
    assert not any("MISSED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_an_unknown_status_fill_does_not_get_benchmarked_without_a_confirmed_price():
    unknown_leg = _leg(status="UNKNOWN", average_price=None)
    live_order_repository = _FakeLiveOrderRepository(unclosed=[unknown_leg])
    paper_benchmark_repository = _FakePaperBenchmarkRepository()

    await entry_decision._finalize_cash_entry(
        "RELIANCE.NS", Decimal("1500"), _cash_state(), live_order_repository, _FakeNotifier(),
        None, paper_benchmark_repository, _FakeOrderExecutor(),
    )

    assert paper_benchmark_repository.opened == []


@pytest.mark.asyncio
async def test_a_complete_fill_still_gets_a_gtt_bracket_and_is_benchmarked():
    complete_leg = _leg(status="COMPLETE", average_price=Decimal("1502.5"))
    live_order_repository = _FakeLiveOrderRepository(unclosed=[complete_leg])
    gtt_repository = _FakeGttRepository()
    paper_benchmark_repository = _FakePaperBenchmarkRepository()

    note = await entry_decision._finalize_cash_entry(
        "RELIANCE.NS", Decimal("1500"), _cash_state(), live_order_repository, _FakeNotifier(),
        gtt_repository, paper_benchmark_repository, _FakeOrderExecutor(),
    )

    assert "opened" in note
    assert len(gtt_repository.recorded) == 1
    assert len(paper_benchmark_repository.opened) == 1


@pytest.mark.asyncio
async def test_nothing_open_reports_missed():
    live_order_repository = _FakeLiveOrderRepository(unclosed=[])
    notifier = _FakeNotifier()

    note = await entry_decision._finalize_cash_entry(
        "RELIANCE.NS", Decimal("1500"), _cash_state(), live_order_repository, notifier,
        None, None, _FakeOrderExecutor(),
    )

    assert "SKIPPED" in note
    assert any("MISSED BUY SIGNAL" in t for t in notifier.texts)
