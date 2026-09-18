from datetime import UTC, datetime, timedelta

import pytest

from trading_scanner.daily_swing_live import DailySwingLive


class _Notifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_tick_health_notifies_once_then_reports_recovery() -> None:
    runner = DailySwingLive.__new__(DailySwingLive)
    runner.notifier = _Notifier()
    runner.active = None
    runner.stale_tick_alerted = False
    now = datetime(2026, 9, 18, 5, 0, tzinfo=UTC)  # Friday 10:30 IST
    runner.last_tick_at = now - timedelta(minutes=4)

    assert await runner.check_tick_health(now) is True
    assert await runner.check_tick_health(now + timedelta(minutes=1)) is True

    assert len(runner.notifier.messages) == 1
    assert "market feed is stale" in runner.notifier.messages[0]

    runner.last_tick_at = now + timedelta(minutes=1, seconds=55)
    assert await runner.check_tick_health(now + timedelta(minutes=2)) is False

    assert len(runner.notifier.messages) == 2
    assert "RECOVERED" in runner.notifier.messages[1]
    assert runner.stale_tick_alerted is False


@pytest.mark.asyncio
async def test_tick_health_is_silent_outside_market_hours() -> None:
    runner = DailySwingLive.__new__(DailySwingLive)
    runner.notifier = _Notifier()
    runner.active = None
    runner.stale_tick_alerted = False
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)  # Friday 17:30 IST
    runner.last_tick_at = now - timedelta(hours=1)

    assert await runner.check_tick_health(now) is False

    assert runner.notifier.messages == []
