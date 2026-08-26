from trading_scanner.application.entry_quality_filter import (
    _REGIME_NORMALIZED_FLOOR,
    _VOLATILITY_MARGIN_FLOOR,
    passes_indicator_filter,
)


def test_passes_when_both_readings_clear_the_floor():
    assert passes_indicator_filter(volatility_margin=10.0, regime_normalized=2.0) is True


def test_rejects_when_volatility_margin_is_below_the_floor():
    assert passes_indicator_filter(volatility_margin=0.1, regime_normalized=2.0) is False


def test_rejects_when_regime_normalized_is_below_the_floor():
    assert passes_indicator_filter(volatility_margin=10.0, regime_normalized=0.01) is False


def test_rejects_when_both_readings_are_below_the_floor():
    assert passes_indicator_filter(volatility_margin=0.1, regime_normalized=0.01) is False


def test_accepts_a_reading_exactly_on_the_floor():
    result = passes_indicator_filter(
        volatility_margin=_VOLATILITY_MARGIN_FLOOR, regime_normalized=_REGIME_NORMALIZED_FLOOR
    )
    assert result is True
