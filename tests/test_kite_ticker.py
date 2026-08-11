"""Unit tests for the pure tick-to-candle aggregation logic (no live
connection needed -- see infrastructure/kite_ticker.py's module docstring
for why this is kept separate from live_pipeline.py's orchestration)."""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading_scanner.infrastructure.kite_ticker import (
    IST,
    CandleAggregator,
    bucket_start,
    is_market_hours,
)

UTC = ZoneInfo("UTC")


def _ist(y, m, d, h, mi) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=IST)


def test_bucket_start_aligns_to_market_open_not_clock_hour():
    # 10:47 IST falls in the 10:15-11:15 bucket, not 10:00-11:00.
    assert bucket_start(_ist(2026, 8, 11, 10, 47)) == _ist(2026, 8, 11, 10, 15)


def test_bucket_start_at_exact_open():
    assert bucket_start(_ist(2026, 8, 11, 9, 15)) == _ist(2026, 8, 11, 9, 15)


def test_bucket_start_last_short_bucket():
    # 3:20 PM falls in the final (short, 15-minute) 3:15-3:30 bucket.
    assert bucket_start(_ist(2026, 8, 11, 15, 20)) == _ist(2026, 8, 11, 15, 15)


def test_bucket_start_handles_non_ist_input():
    # A UTC timestamp equivalent to 10:47 IST should bucket the same way.
    when_utc = _ist(2026, 8, 11, 10, 47).astimezone(UTC)
    assert bucket_start(when_utc) == _ist(2026, 8, 11, 10, 15)


def test_is_market_hours_true_during_session():
    assert is_market_hours(_ist(2026, 8, 11, 11, 0)) is True


def test_is_market_hours_false_before_open():
    assert is_market_hours(_ist(2026, 8, 11, 9, 0)) is False


def test_is_market_hours_false_after_close():
    assert is_market_hours(_ist(2026, 8, 11, 15, 31)) is False


def test_is_market_hours_false_on_weekend():
    # 2026-08-15 is a Saturday.
    assert is_market_hours(_ist(2026, 8, 15, 11, 0)) is False


def test_candle_aggregator_tracks_ohlc_across_ticks():
    aggregator = CandleAggregator()
    aggregator.start(_ist(2026, 8, 11, 9, 15))
    aggregator.add_tick(_ist(2026, 8, 11, 9, 16), Decimal("100.0"), 1000)
    aggregator.add_tick(_ist(2026, 8, 11, 9, 20), Decimal("105.0"), 1200)
    aggregator.add_tick(_ist(2026, 8, 11, 9, 30), Decimal("98.0"), 1500)
    aggregator.add_tick(_ist(2026, 8, 11, 9, 45), Decimal("102.0"), 1800)

    open_, high, low, close, volume = aggregator.finalize()
    assert open_ == Decimal("100.0")
    assert high == Decimal("105.0")
    assert low == Decimal("98.0")
    assert close == Decimal("102.0")
    assert volume == 800  # 1800 - 1000, cumulative day volume delta


def test_candle_aggregator_no_ticks_finalizes_to_none():
    aggregator = CandleAggregator()
    aggregator.start(_ist(2026, 8, 11, 9, 15))
    assert aggregator.finalize() is None


def test_candle_aggregator_start_resets_state():
    aggregator = CandleAggregator()
    aggregator.start(_ist(2026, 8, 11, 9, 15))
    aggregator.add_tick(_ist(2026, 8, 11, 9, 16), Decimal("100.0"), 1000)
    aggregator.start(_ist(2026, 8, 11, 10, 15))
    assert aggregator.finalize() is None
