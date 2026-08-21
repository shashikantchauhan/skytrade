"""PAPER_TRADING_ENABLED / FUTURES_PAPER_TRADING_ENABLED -- both simulators
are being retired in favor of the real live-cash-order + GTT flow (see
their own module docstrings). Focus here: turning either off stops *new*
positions without touching/needing anything about positions already open
(that's the caller's/DB's job, not this flag's)."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import futures_trading, paper_trading
from trading_scanner.domain.models import SignalSide


class FakePaperAccountRepository:
    def __init__(self, cash_balance: Decimal = Decimal("500000")) -> None:
        self._cash_balance = cash_balance
        self.opened = []

    async def get_cash_balance(self) -> Decimal:
        return self._cash_balance

    async def get_open_positions(self):
        return []

    async def open_position(self, position) -> None:
        self.opened.append(position)


@pytest.mark.asyncio
async def test_try_open_position_noop_when_paper_trading_disabled(monkeypatch):
    monkeypatch.setattr(paper_trading, "PAPER_TRADING_ENABLED", False)
    repo = FakePaperAccountRepository()

    position = await paper_trading.try_open_position(
        "RELIANCE.NS", datetime.now(UTC), Decimal("1400"), repo
    )

    assert position is None
    assert repo.opened == []


@pytest.mark.asyncio
async def test_try_open_position_still_works_when_enabled(monkeypatch):
    monkeypatch.setattr(paper_trading, "PAPER_TRADING_ENABLED", True)
    repo = FakePaperAccountRepository()

    position = await paper_trading.try_open_position(
        "RELIANCE.NS", datetime.now(UTC), Decimal("1400"), repo
    )

    assert position is not None
    assert repo.opened == [position]


@pytest.mark.asyncio
async def test_open_futures_paper_position_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(futures_trading, "FUTURES_PAPER_TRADING_ENABLED", False)

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("is_eligible should never run when the flag is off")

    monkeypatch.setattr(futures_trading, "is_eligible", _should_not_be_called)

    result = await futures_trading.open_futures_paper_position(
        "RELIANCE.NS", SignalSide.BUY, datetime.now(UTC), Decimal("1400"), "1h",
        derivatives_chain=None, trade_repository=None, futures_account_repository=None,
    )

    assert result is None
