"""Live tick-to-candle aggregation for Kite's WebSocket feed.

Zerodha's own guidance (confirmed live and via their dev forum) is that the
Historical Data API -- what ``KiteProvider``/the hourly cron used -- is for
backfill/backtesting only, and can lag the current session's candles by
hours. Their documented fix for live strategies: build candles from the
WebSocket tick feed yourself. This module is that: pure, synchronous,
in-memory aggregation logic with no I/O, kept separate from
``live_pipeline.py`` (the orchestrator that wires this to KiteTicker,
asyncio, and the rest of the signal pipeline) so the bucketing math can be
unit-tested without a live connection.

NSE's hourly candles are aligned to market open (9:15, 10:15, ..., 15:15
IST), not to the top of the clock hour -- ``bucket_start`` reflects that.
"""

from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE equity/derivatives session: 9:15 AM to 3:30 PM IST. Hourly bars are
# aligned to the open, so the last bar of the day (3:15-3:30) is a short
# 15-minute bucket, not a full hour -- matches how the historical API's own
# hourly candles are cut.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
_BUCKET_MINUTES = 60


def is_market_hours(when: datetime) -> bool:
    """Whether ``when`` (any tzinfo) falls within the NSE trading session,
    Mon-Fri. Does not account for exchange holidays -- ticks simply won't
    arrive on a holiday, so a stale/idle connection during one is harmless,
    just not worth specifically optimizing away here."""
    when_ist = when.astimezone(IST)
    if when_ist.weekday() >= 5:  # Saturday/Sunday
        return False
    return MARKET_OPEN <= when_ist.time() < MARKET_CLOSE


def bucket_start(when: datetime) -> datetime:
    """The start (IST, tz-aware) of the hourly bucket ``when`` falls into,
    aligned to market open rather than the clock hour -- e.g. 10:47 IST
    falls in the 10:15-11:15 bucket, whose start is 10:15."""
    when_ist = when.astimezone(IST)
    open_dt = datetime.combine(when_ist.date(), MARKET_OPEN, tzinfo=IST)
    elapsed_minutes = (when_ist - open_dt).total_seconds() / 60
    bucket_index = int(elapsed_minutes // _BUCKET_MINUTES)
    return open_dt + timedelta(minutes=bucket_index * _BUCKET_MINUTES)


class CandleAggregator:
    """Accumulates ticks for one instrument into an OHLCV candle for the
    bucket it's currently tracking. One instance per subscribed symbol.

    Volume: Kite's tick payload carries ``volume_traded``, the exchange's
    running total *for the day*, not a per-tick delta -- so this candle's
    volume is (volume at the last tick of the bucket) minus (volume at the
    first tick of the bucket), snapshotted at ``start()``.
    """

    __slots__ = (
        "current_bucket", "open", "high", "low", "close",
        "_volume_at_bucket_start", "_last_volume",
    )

    def __init__(self) -> None:
        self.current_bucket: datetime | None = None
        self.open: Decimal | None = None
        self.high: Decimal | None = None
        self.low: Decimal | None = None
        self.close: Decimal | None = None
        self._volume_at_bucket_start: int | None = None
        self._last_volume: int | None = None

    def add_tick(self, tick_time: datetime, price: Decimal, cumulative_day_volume: int) -> None:
        """Feed one tick. If it belongs to a bucket newer than the one
        currently being tracked, the caller is responsible for having
        already called ``finalize()`` and ``start()`` for the new bucket --
        this method does not roll buckets itself, since finalizing needs to
        happen at a scheduled boundary check, not the instant a stray tick
        for the next hour happens to arrive slightly early or late.
        """
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
            self._volume_at_bucket_start = cumulative_day_volume
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        self.close = price
        self._last_volume = cumulative_day_volume

    def start(self, bucket: datetime) -> None:
        """Begin tracking a new (empty) bucket."""
        self.current_bucket = bucket
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self._volume_at_bucket_start = None
        self._last_volume = None

    def finalize(self) -> tuple[Decimal, Decimal, Decimal, Decimal, int] | None:
        """The completed (open, high, low, close, volume) for the current
        bucket, or None if no ticks arrived during it (illiquid symbol, or
        the bucket spans outside market hours)."""
        if self.open is None:
            return None
        volume = 0
        if self._volume_at_bucket_start is not None and self._last_volume is not None:
            volume = max(0, self._last_volume - self._volume_at_bucket_start)
        return (self.open, self.high, self.low, self.close, volume)
