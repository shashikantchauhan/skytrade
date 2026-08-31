"""Tests for the candlestick pattern detectors -- see
application/candlestick_patterns.py's own docstring for the exact
definition each one uses."""

from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.candlestick_patterns import (
    bullish_engulfing,
    bullish_harami,
    morning_star,
    piercing_line,
    three_inside_up,
    three_outside_up,
    three_white_soldiers,
)
from trading_scanner.domain.models import Candle


def _candle(open_, high, low, close, day=1) -> Candle:
    return Candle(
        symbol="TEST.NS",
        timestamp=datetime(2026, 1, day, tzinfo=UTC),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_a_textbook_three_inside_up_matches():
    candles = [
        _candle(110, 112, 98, 100, day=1),  # bearish, body 100-110
        _candle(102, 106, 101, 105, day=2),  # bullish, body 102-105, inside 100-110
        _candle(105, 116, 104, 115, day=3),  # bullish, closes above first's open (110)
    ]

    assert three_inside_up(candles) is True


def test_fewer_than_three_candles_never_matches():
    candles = [
        _candle(110, 112, 98, 100, day=1),
        _candle(102, 106, 101, 105, day=2),
    ]

    assert three_inside_up(candles) is False


def test_first_candle_must_be_bearish():
    candles = [
        _candle(100, 112, 98, 110, day=1),  # bullish, not bearish -- disqualifies
        _candle(102, 106, 101, 105, day=2),
        _candle(105, 116, 104, 115, day=3),
    ]

    assert three_inside_up(candles) is False


def test_second_candle_must_be_bullish():
    candles = [
        _candle(110, 112, 98, 100, day=1),
        _candle(105, 106, 101, 102, day=2),  # bearish, not bullish -- disqualifies
        _candle(105, 116, 104, 115, day=3),
    ]

    assert three_inside_up(candles) is False


def test_second_candle_body_must_be_strictly_inside_the_first():
    candles = [
        _candle(110, 112, 98, 100, day=1),  # body 100-110
        _candle(95, 106, 94, 105, day=2),  # opens at 95, below first's body low (100)
        _candle(105, 116, 104, 115, day=3),
    ]

    assert three_inside_up(candles) is False


def test_third_candle_must_close_above_the_first_candles_open():
    candles = [
        _candle(110, 112, 98, 100, day=1),  # first's open is 110
        _candle(102, 106, 101, 105, day=2),
        _candle(105, 109, 104, 108, day=3),  # bullish, but closes at 108 < 110
    ]

    assert three_inside_up(candles) is False


def test_only_the_last_three_candles_are_examined():
    # A matching pattern buried earlier in the history must not trigger a
    # match on its own -- only the most recent 3 candles are examined, and
    # here they don't form one (day 4 is bearish, disqualifying it as the
    # "third" candle of the trailing window days 2-4).
    candles = [
        _candle(110, 112, 98, 100, day=1),
        _candle(102, 106, 101, 105, day=2),
        _candle(105, 116, 104, 115, day=3),  # a real match here (days 1-3)...
        _candle(115, 116, 108, 109, day=4),  # ...but trailing window is now 2-4
    ]

    assert three_inside_up(candles) is False


def test_a_textbook_bullish_engulfing_matches():
    candles = [
        _candle(110, 111, 98, 100, day=1),  # bearish, body 100-110
        _candle(98, 116, 97, 115, day=2),  # bullish, opens below 100, closes above 110
    ]

    assert bullish_engulfing(candles) is True


def test_bullish_engulfing_fails_when_the_body_is_not_fully_engulfed():
    candles = [
        _candle(110, 111, 98, 100, day=1),
        _candle(98, 106, 97, 105, day=2),  # closes at 105, doesn't clear first's open (110)
    ]

    assert bullish_engulfing(candles) is False


def test_a_textbook_bullish_harami_matches():
    candles = [
        _candle(110, 111, 98, 100, day=1),  # bearish, body 100-110
        _candle(102, 109, 101, 108, day=2),  # bullish, body 102-108, inside 100-110
    ]

    assert bullish_harami(candles) is True


def test_bullish_harami_fails_when_the_second_body_is_not_strictly_inside():
    candles = [
        _candle(110, 111, 98, 100, day=1),
        _candle(99, 109, 98, 108, day=2),  # opens at 99, below first's body low (100)
    ]

    assert bullish_harami(candles) is False


def test_a_textbook_piercing_line_matches():
    candles = [
        _candle(110, 111, 97, 100, day=1),  # bearish, body 100-110, midpoint 105
        _candle(98, 109, 97, 108, day=2),  # gaps below 100, closes at 108 (>105, <110)
    ]

    assert piercing_line(candles) is True


def test_piercing_line_fails_when_it_fully_engulfs_instead():
    candles = [
        _candle(110, 113, 97, 100, day=1),
        _candle(98, 113, 97, 112, day=2),  # closes at 112, past first's open (110)
    ]

    assert piercing_line(candles) is False


def test_piercing_line_fails_without_a_gap_down_at_the_open():
    candles = [
        _candle(110, 111, 97, 100, day=1),
        _candle(101, 109, 100, 108, day=2),  # opens at 101, not below first's close (100)
    ]

    assert piercing_line(candles) is False


def test_a_textbook_three_white_soldiers_matches():
    candles = [
        _candle(100, 111, 99, 110, day=1),
        _candle(105, 119, 104, 118, day=2),
        _candle(112, 127, 111, 126, day=3),
    ]

    assert three_white_soldiers(candles) is True


def test_three_white_soldiers_fails_with_a_large_upper_wick():
    candles = [
        _candle(100, 111, 99, 110, day=1),
        _candle(105, 119, 104, 118, day=2),
        _candle(112, 140, 111, 126, day=3),  # high of 140 -- a huge upper wick
    ]

    assert three_white_soldiers(candles) is False


def test_a_textbook_morning_star_matches():
    candles = [
        _candle(120, 121, 99, 100, day=1),  # bearish, body 100-120 (size 20)
        _candle(99, 102, 98, 101, day=2),  # small body (size 2, well under 30% of 20)
        _candle(102, 116, 101, 115, day=3),  # bullish, closes above midpoint (110)
    ]

    assert morning_star(candles) is True


def test_morning_star_fails_when_the_middle_candle_is_not_small():
    candles = [
        _candle(120, 121, 99, 100, day=1),
        _candle(99, 112, 98, 111, day=2),  # body size 12 -- not small vs 20
        _candle(102, 116, 101, 115, day=3),
    ]

    assert morning_star(candles) is False


def test_a_textbook_three_outside_up_matches():
    candles = [
        _candle(110, 111, 98, 100, day=1),  # bearish, body 100-110
        _candle(98, 116, 97, 115, day=2),  # bullish, engulfs day 1
        _candle(115, 121, 114, 120, day=3),  # bullish, closes above day 2's close
    ]

    assert three_outside_up(candles) is True


def test_three_outside_up_fails_without_the_third_confirming_candle():
    candles = [
        _candle(110, 111, 98, 100, day=1),
        _candle(98, 116, 97, 115, day=2),
        _candle(116, 117, 110, 112, day=3),  # bullish, but closes below day 2's close (115)
    ]

    assert three_outside_up(candles) is False
