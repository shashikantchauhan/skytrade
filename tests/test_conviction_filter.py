"""Tests for the entry-candle conviction (close-location-value) gate --
see application/conviction_filter.py's own docstring for the backtest
evidence behind CONVICTION_THRESHOLD."""

from decimal import Decimal

from trading_scanner.application.conviction_filter import (
    CONVICTION_THRESHOLD,
    close_location_value,
    passes_conviction_filter,
)


def test_close_location_value_is_one_when_close_equals_high():
    assert close_location_value(Decimal("110"), Decimal("100"), Decimal("110")) == 1.0


def test_close_location_value_is_zero_when_close_equals_low():
    assert close_location_value(Decimal("110"), Decimal("100"), Decimal("100")) == 0.0


def test_close_location_value_is_half_at_the_midpoint():
    assert close_location_value(Decimal("110"), Decimal("100"), Decimal("105")) == 0.5


def test_close_location_value_defaults_to_half_for_a_zero_range_bar():
    assert close_location_value(Decimal("100"), Decimal("100"), Decimal("100")) == 0.5


def test_passes_conviction_filter_true_at_the_default_threshold():
    # close-location-value exactly 0.7 -- the deployed floor itself, "at or
    # above" must pass, not just strictly above.
    assert passes_conviction_filter(Decimal("110"), Decimal("100"), Decimal("107")) is True


def test_passes_conviction_filter_false_just_below_the_default_threshold():
    assert passes_conviction_filter(Decimal("110"), Decimal("100"), Decimal("106.9")) is False


def test_passes_conviction_filter_respects_a_custom_threshold():
    # CLV 0.5 fails the deployed 0.7 floor but clears an explicit 0.5 one.
    high, low, close = Decimal("110"), Decimal("100"), Decimal("105")
    assert passes_conviction_filter(high, low, close) is False
    assert passes_conviction_filter(high, low, close, threshold=0.5) is True


def test_default_threshold_constant_is_the_deployed_value():
    # Locks in the value the docstring's backtest evidence justifies --
    # a silent change here would need a fresh backtest, not just a passing
    # test suite.
    assert CONVICTION_THRESHOLD == 0.7
