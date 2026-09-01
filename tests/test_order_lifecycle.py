"""Tests for domain/order_lifecycle.py -- especially derive_position_
lifecycle's RECONCILIATION_REQUIRED case, which is the type-level fix for
the reviewed bug (an UNKNOWN entry leg invisible to every exit path
because they all used the narrower COMPLETE-only status set)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading_scanner.domain.models import LiveOrderLeg
from trading_scanner.domain.order_lifecycle import (
    LegStatus,
    OrderBasket,
    PositionLifecycle,
    derive_position_lifecycle,
    group_into_baskets,
)

_T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _leg(offset_minutes: int, **overrides) -> LiveOrderLeg:
    defaults = dict(
        basket_id="b1", symbol="RELIANCE.NS", purpose="cash", tradingsymbol="RELIANCE",
        transaction_type="BUY", quantity=5, order_id="o1", status="COMPLETE",
        placed_at=_T0 + timedelta(minutes=offset_minutes), average_price=Decimal("1000"),
    )
    defaults.update(overrides)
    return LiveOrderLeg(**defaults)


def test_no_legs_is_none():
    assert derive_position_lifecycle([]) == PositionLifecycle.NONE


def test_only_rejected_entry_attempts_is_none():
    legs = [
        _leg(0, order_id="o1", status="REJECTED"),
        _leg(1, order_id="o2", status="CANCELLED"),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.NONE


def test_a_single_open_entry_leg_is_opening():
    legs = [_leg(0, status="OPEN", average_price=None)]
    assert derive_position_lifecycle(legs) == PositionLifecycle.OPENING


def test_a_complete_entry_with_no_exit_is_active():
    legs = [_leg(0, status="COMPLETE")]
    assert derive_position_lifecycle(legs) == PositionLifecycle.ACTIVE


def test_a_complete_entry_fully_closed_by_a_complete_exit_is_closed():
    legs = [
        _leg(0, order_id="o1", status="COMPLETE"),
        _leg(1, order_id="o2", transaction_type="SELL", status="COMPLETE"),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.CLOSED


def test_a_complete_entry_with_an_in_flight_exit_is_exit_pending():
    legs = [
        _leg(0, order_id="o1", status="COMPLETE"),
        _leg(1, order_id="o2", transaction_type="SELL", status="OPEN", average_price=None),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.EXIT_PENDING


def test_an_unknown_entry_leg_alone_requires_reconciliation():
    # The exact bug this type exists to fix: today's get_open_cash_legs
    # (COMPLETE-only) would see nothing here and treat the symbol as if it
    # had never been entered -- invisible to every exit path. This type
    # must not make the same mistake.
    legs = [_leg(0, status="UNKNOWN", average_price=None)]
    assert derive_position_lifecycle(legs) == PositionLifecycle.RECONCILIATION_REQUIRED


def test_an_unknown_leg_wins_even_alongside_an_otherwise_closed_position():
    # An UNKNOWN leg anywhere in the relevant history means local records
    # can't be trusted -- must not let a clean net-zero elsewhere mask it.
    legs = [
        _leg(0, order_id="o1", status="COMPLETE"),
        _leg(1, order_id="o2", transaction_type="SELL", status="COMPLETE"),
        _leg(2, order_id="o3", status="UNKNOWN", average_price=None),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.RECONCILIATION_REQUIRED


def test_a_rejected_retry_attempt_is_ignored_once_a_later_attempt_completes():
    legs = [
        _leg(0, order_id="o1", status="REJECTED"),
        _leg(1, order_id="o2", status="COMPLETE"),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.ACTIVE


def test_a_partial_exit_leaves_the_remainder_active():
    legs = [
        _leg(0, order_id="o1", quantity=9, status="COMPLETE"),
        _leg(1, order_id="o2", transaction_type="SELL", quantity=8, status="COMPLETE"),
    ]
    assert derive_position_lifecycle(legs) == PositionLifecycle.ACTIVE


def test_order_basket_outcome_prefers_a_non_terminal_failure_leg():
    basket = OrderBasket(
        "b1",
        (
            _leg(0, order_id="o1", status="REJECTED"),
            _leg(1, order_id="o2", status="COMPLETE"),
        ),
    )
    assert basket.outcome == LegStatus.COMPLETE


def test_order_basket_outcome_falls_back_to_the_last_leg_when_every_attempt_failed():
    basket = OrderBasket(
        "b1",
        (
            _leg(0, order_id="o1", status="REJECTED"),
            _leg(1, order_id="o2", status="CANCELLED"),
        ),
    )
    assert basket.outcome == LegStatus.CANCELLED


def test_group_into_baskets_splits_by_basket_id_preserving_order():
    legs = [
        _leg(0, basket_id="b1", order_id="o1"),
        _leg(1, basket_id="b2", order_id="o2"),
        _leg(2, basket_id="b1", order_id="o3", status="REJECTED"),
    ]
    baskets = group_into_baskets(legs)
    assert [b.basket_id for b in baskets] == ["b1", "b2"]
    assert [leg.order_id for leg in baskets[0].legs] == ["o1", "o3"]
    assert [leg.order_id for leg in baskets[1].legs] == ["o2"]
