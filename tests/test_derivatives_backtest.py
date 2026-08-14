"""Regression coverage for application/derivatives_backtest.py -- this
module had zero tests before 2026-08-14, which is exactly how a real bug
(every trade silently backtested as a SHORT future regardless of its real
BUY/SELL side -- comparing trade.side.value, lowercase, against the
uppercase literals "BUY"/"SELL") went unnoticed in a live dashboard
feature. These tests pin the fixed behavior down directly.
"""
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application.derivatives_backtest import _backtest_one_trade
from trading_scanner.domain.models import SignalSide, Trade


class FakeDerivativesChain:
    """Stands in for KiteDerivativesChain -- returns a fixed contract/
    premium for any symbol, records what side each future leg was opened
    with so tests can assert on it directly."""

    def __init__(self) -> None:
        self.future_calls: list[str] = []  # nothing here needs the symbol distinguished

    def nearest_future(self, symbol):
        return {"tradingsymbol": f"{symbol.removesuffix('.NS')}FUT", "lot_size": 100,
                "instrument_token": 111, "expiry": "2026-08-25"}

    def nearest_atm_option(self, symbol, option_type, underlying_price):
        return {"tradingsymbol": f"{symbol.removesuffix('.NS')}{option_type}", "lot_size": 100,
                "instrument_token": 222, "strike": underlying_price, "expiry": "2026-08-25"}

    def historical_premium(self, instrument_token, when):
        # Distinct entry/exit values so pnl_amount is nonzero and easy to sanity-check.
        return 100.0 if when.day <= 5 else 110.0


class _RecordingRepo:
    def __init__(self) -> None:
        self.inserted: list = []

    async def insert_backtest_trade(self, trade) -> None:
        self.inserted.append(trade)

    async def delete_backtest_trades(self) -> None:
        self.inserted.clear()


def _closed_trade(side: SignalSide) -> Trade:
    return Trade(
        symbol="RELIANCE.NS", side=side,
        entry_timestamp=datetime(2026, 8, 3, tzinfo=UTC), entry_price=Decimal("100"),
        prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=datetime(2026, 8, 10, tzinfo=UTC), exit_price=Decimal("110"),
        pnl_percent=Decimal("10"), status="closed",
    )


@pytest.mark.asyncio
async def test_buy_signal_backtests_as_a_long_future_not_short():
    # The actual bug: trade.side.value is "buy" (SignalSide is a lowercase
    # StrEnum), but the old code compared it against "BUY" and always fell
    # through to "short" -- this must now come out "long".
    chain = FakeDerivativesChain()
    futures_repo = _RecordingRepo()
    options_repo = _RecordingRepo()

    await _backtest_one_trade(_closed_trade(SignalSide.BUY), chain, options_repo, futures_repo)

    assert len(futures_repo.inserted) == 1
    assert futures_repo.inserted[0].side == "long"
    # A long future is hedged with a PE (see the module's own docstring).
    assert options_repo.inserted[0].option_type == "PE"


@pytest.mark.asyncio
async def test_sell_signal_backtests_as_a_short_future():
    chain = FakeDerivativesChain()
    futures_repo = _RecordingRepo()
    options_repo = _RecordingRepo()

    await _backtest_one_trade(_closed_trade(SignalSide.SELL), chain, options_repo, futures_repo)

    assert len(futures_repo.inserted) == 1
    assert futures_repo.inserted[0].side == "short"
    assert options_repo.inserted[0].option_type == "CE"


@pytest.mark.asyncio
async def test_long_future_pnl_is_price_move_times_lot_size():
    chain = FakeDerivativesChain()  # entry premium 100, exit premium 110
    futures_repo = _RecordingRepo()
    options_repo = _RecordingRepo()

    await _backtest_one_trade(_closed_trade(SignalSide.BUY), chain, options_repo, futures_repo)

    future = futures_repo.inserted[0]
    assert future.pnl_amount == Decimal("1000")  # (110 - 100) * lot_size(100)
