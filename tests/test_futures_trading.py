from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import futures_trading
from trading_scanner.domain.models import SignalSide, Trade


class FakeFuturesPaperAccountRepository:
    """In-memory FuturesPaperAccountRepository fake, mirrors
    test_signal_pipeline.py's FakePaperAccountRepository."""

    def __init__(self, cash_balance=Decimal("400000")) -> None:
        self._cash_balance = cash_balance
        self.opened = []

    async def get_cash_balance(self) -> Decimal:
        return self._cash_balance

    async def open_position(self, position) -> None:
        self.opened.append(position)
        self._cash_balance -= position.margin_allocated

    async def close_position(self, symbol, exit_timestamp, futures_exit_price):
        matching = [p for p in self.opened if p.symbol == symbol and p.status == "open"]
        if not matching:
            return None
        position = matching[-1]
        pnl_amount = (
            (futures_exit_price - position.futures_entry_price) * position.lot_size
            if position.side == "long"
            else (position.futures_entry_price - futures_exit_price) * position.lot_size
        )
        self._cash_balance += position.margin_allocated + pnl_amount
        index = self.opened.index(position)
        self.opened[index] = replace(
            position, exit_timestamp=exit_timestamp, futures_exit_price=futures_exit_price,
            pnl_amount=pnl_amount, status="closed",
        )
        return self.opened[index]

    async def get_open_positions(self):
        return [p for p in self.opened if p.status == "open"]


class FakeTradeRepository:
    def __init__(self, trades) -> None:
        self._trades = trades

    async def open_trade(self, interval, trade) -> None:
        raise NotImplementedError

    async def close_open_trade(self, symbol, interval, side, exit_timestamp, exit_price) -> None:
        raise NotImplementedError

    async def abandon_open_trade(self, symbol, interval, side) -> None:
        raise NotImplementedError

    async def get_trades(self, symbol, interval):
        return [t for t in self._trades if symbol is None or t.symbol == symbol]


class FakeDerivativesChain:
    """Stands in for KiteDerivativesChain -- only margin_benefit is used
    by futures_trading.py."""

    def __init__(self, combined_margin: Decimal | None):
        self._combined_margin = combined_margin

    def margin_benefit(self, legs):
        if self._combined_margin is None:
            return None
        return {
            "primary_only_margin": float(self._combined_margin) * 1.5,
            "combined_margin": float(self._combined_margin),
            "margin_benefit": float(self._combined_margin) * 0.5,
        }


def _closed_trade(symbol, side, win=True):
    return Trade(
        symbol=symbol, side=side, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        exit_price=Decimal("110") if win else Decimal("90"),
        pnl_percent=Decimal("10") if win else Decimal("-10"), status="closed",
    )


@pytest.mark.asyncio
async def test_is_eligible_checks_the_matching_side_only():
    trades = [_closed_trade("RELIANCE.NS", SignalSide.BUY) for _ in range(5)]
    repo = FakeTradeRepository(trades)
    assert await futures_trading.is_eligible("RELIANCE.NS", SignalSide.BUY, "60minute", repo)
    assert not await futures_trading.is_eligible("RELIANCE.NS", SignalSide.SELL, "60minute", repo)


@pytest.mark.asyncio
async def test_try_open_opens_when_margin_fits_the_slot_budget():
    account = FakeFuturesPaperAccountRepository(cash_balance=Decimal("400000"))
    # Default slot budget: max(400000/16, 15000) = 25000. A 15000 combined
    # margin, plus the 25% buffer, is 18750 -- comfortably under that.
    chain = FakeDerivativesChain(combined_margin=Decimal("15000"))
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", 500, chain, account,
    )
    assert position is not None
    assert position.margin_allocated == Decimal("15000") * (
        1 + futures_trading.FUTURES_MARGIN_BUFFER_PCT
    )
    assert account.opened == [position]


@pytest.mark.asyncio
async def test_try_open_skips_when_margin_exceeds_slot_budget():
    account = FakeFuturesPaperAccountRepository(cash_balance=Decimal("400000"))
    # Way beyond any reasonable slot budget for this account size.
    chain = FakeDerivativesChain(combined_margin=Decimal("1000000"))
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", 500, chain, account,
    )
    assert position is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_try_open_skips_when_margin_api_fails():
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=None)
    position = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", 500, chain, account,
    )
    assert position is None


@pytest.mark.asyncio
async def test_try_open_skips_a_symbol_already_open():
    account = FakeFuturesPaperAccountRepository()
    chain = FakeDerivativesChain(combined_margin=Decimal("10000"))
    first = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "long", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG2800PE", 500, chain, account,
    )
    assert first is not None
    second = await futures_trading.try_open_futures_position(
        "RELIANCE.NS", "short", datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "RELIANCE26AUGFUT", "RELIANCE26AUG3000CE", 500, chain, account,
    )
    assert second is None
