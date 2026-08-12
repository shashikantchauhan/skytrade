from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_scanner.application.stop_loss_replay import apply_stop_loss, summarize
from trading_scanner.domain.models import Candle, SignalSide, Trade


class _FakeCandleRepository:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]):
        self._candles_by_symbol = candles_by_symbol

    async def upsert_candles(self, symbol, interval, candles):
        raise NotImplementedError

    async def get_candles(self, symbol, interval, limit=None):
        return self._candles_by_symbol.get(symbol, [])


def _candle(hour: int, low: float, high: float, start: datetime) -> Candle:
    return Candle(
        symbol="TEST", timestamp=start + timedelta(hours=hour),
        open=Decimal(str((low + high) / 2)), high=Decimal(str(high)), low=Decimal(str(low)),
        close=Decimal(str((low + high) / 2)), volume=1000,
    )


@pytest.mark.asyncio
async def test_buy_trade_stopped_out_when_path_breaches_low():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    # Entry at 100, drifts down to 90 (a -10% low) by hour 3, then recovers
    # and the strategy's own exit doesn't fire until hour 5 at 105 (a win,
    # if the stop-loss never triggered).
    candles = [
        _candle(0, 99, 101, start), _candle(1, 95, 99, start), _candle(2, 90, 96, start),
        _candle(3, 92, 98, start), _candle(4, 100, 106, start), _candle(5, 103, 107, start),
    ]
    trade = Trade(
        symbol="TEST", side=SignalSide.BUY, entry_timestamp=start, entry_price=Decimal("100"),
        prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=start + timedelta(hours=5), exit_price=Decimal("105"),
        pnl_percent=Decimal("5"), status="closed",
    )
    repo = _FakeCandleRepository({"TEST": candles})

    adjusted = await apply_stop_loss([trade], repo, "1h", stop_loss_pct=Decimal("5"))

    assert len(adjusted) == 1
    stopped = adjusted[0]
    assert stopped.exit_price == Decimal("95")  # 100 * (1 - 5/100)
    assert stopped.pnl_percent == Decimal("-5")
    assert stopped.exit_timestamp == start + timedelta(hours=1)  # first bar whose low <= 95


@pytest.mark.asyncio
async def test_trade_never_breaching_stop_is_unchanged():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [_candle(h, 99, 103, start) for h in range(4)]  # never drops below 99
    trade = Trade(
        symbol="TEST", side=SignalSide.BUY, entry_timestamp=start, entry_price=Decimal("100"),
        prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=start + timedelta(hours=3), exit_price=Decimal("102"),
        pnl_percent=Decimal("2"), status="closed",
    )
    repo = _FakeCandleRepository({"TEST": candles})

    adjusted = await apply_stop_loss([trade], repo, "1h", stop_loss_pct=Decimal("5"))

    assert adjusted == [trade]


@pytest.mark.asyncio
async def test_sell_trade_stopped_out_when_path_breaches_high():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        _candle(0, 99, 101, start), _candle(1, 103, 108, start),  # high 108 > 105 stop
        _candle(2, 90, 95, start),
    ]
    trade = Trade(
        symbol="TEST", side=SignalSide.SELL, entry_timestamp=start, entry_price=Decimal("100"),
        prediction_at_entry=-5, is_early_signal_flip=False,
        exit_timestamp=start + timedelta(hours=2), exit_price=Decimal("92"),
        pnl_percent=Decimal("8"), status="closed",
    )
    repo = _FakeCandleRepository({"TEST": candles})

    adjusted = await apply_stop_loss([trade], repo, "1h", stop_loss_pct=Decimal("5"))

    stopped = adjusted[0]
    assert stopped.exit_price == Decimal("105")  # 100 * (1 + 5/100)
    assert stopped.pnl_percent == Decimal("-5")


@pytest.mark.asyncio
async def test_open_trades_pass_through_unchanged():
    trade = Trade(
        symbol="TEST", side=SignalSide.BUY, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    repo = _FakeCandleRepository({})

    adjusted = await apply_stop_loss([trade], repo, "1h", stop_loss_pct=Decimal("5"))

    assert adjusted == [trade]


def test_summarize_computes_win_rate_and_expectancy():
    trades = [
        Trade(
            symbol="A", side=SignalSide.BUY, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10"), status="closed",
        ),
        Trade(
            symbol="B", side=SignalSide.BUY, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("95"),
            pnl_percent=Decimal("-5"), status="closed",
        ),
    ]
    stats = summarize(trades, SignalSide.BUY)
    assert stats["n"] == 2
    assert stats["win_rate"] == Decimal("50")
    assert stats["avg_win"] == Decimal("10")
    assert stats["avg_loss"] == Decimal("-5")
    assert stats["expectancy"] == Decimal("2.5")
