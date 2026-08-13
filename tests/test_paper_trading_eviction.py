"""Capital-rotation: evicting a weaker, currently-losing open position to
make room for a stronger new candidate, instead of just skipping it when
capital is full. See application/paper_trading.py's try_evict_and_open
docstring for the eviction rule (both weaker-ranked AND currently losing)
and the market-close cutoff this refuses to act past.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_scanner.application.paper_trading import (
    _before_rotation_cutoff,
    stop_loss_price,
    try_evict_and_open,
)
from trading_scanner.domain.models import Candle, PaperPosition


class _FakePaperAccountRepository:
    def __init__(self, open_positions: list[PaperPosition], cash_balance=Decimal("0")):
        self._open = list(open_positions)
        self._cash_balance = cash_balance
        self.closed: list[tuple[str, object, object]] = []

    async def get_cash_balance(self) -> Decimal:
        return self._cash_balance

    async def get_open_positions(self):
        return [p for p in self._open if p.status == "open"]

    async def open_position(self, position: PaperPosition) -> None:
        self._open.append(position)
        self._cash_balance -= position.capital_allocated

    async def close_position(self, symbol, exit_timestamp, exit_price):
        matching = [p for p in self._open if p.symbol == symbol and p.status == "open"]
        if not matching:
            return None
        position = matching[-1]
        self._open.remove(position)
        closed = replace(position, exit_timestamp=exit_timestamp, exit_price=exit_price, status="closed")
        self._open.append(closed)
        self._cash_balance += position.capital_allocated
        self.closed.append((symbol, exit_timestamp, exit_price))
        return closed


class _FakeCandleRepository:
    def __init__(self, latest_close: dict[str, Decimal]):
        self._latest_close = latest_close

    async def get_candles(self, symbol, interval, limit=None):
        if symbol not in self._latest_close:
            return []
        return [
            Candle(
                symbol=symbol, timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                open=self._latest_close[symbol], high=self._latest_close[symbol],
                low=self._latest_close[symbol], close=self._latest_close[symbol], volume=0,
            )
        ]


def _position(symbol, entry_price, prediction_at_entry) -> PaperPosition:
    # quantity=300 so capital_allocated (Rs 30,000 at entry_price=100) clears
    # try_open_position's real MIN_POSITION_SIZE floor (Rs 25,000) once
    # freed by an eviction -- a tiny test position would free too little
    # capital to actually open the replacement, independent of the
    # eviction logic itself.
    return PaperPosition(
        symbol=symbol, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=entry_price,
        quantity=300, capital_allocated=entry_price * 300, prediction_at_entry=prediction_at_entry,
    )


# 10:00 UTC = 15:30 IST -- exactly market close, well past the 14:30 IST cutoff.
_PAST_CUTOFF = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
# 07:00 UTC = 12:30 IST -- well before the cutoff.
_BEFORE_CUTOFF = datetime(2026, 1, 1, 7, 0, tzinfo=UTC)


def test_stop_loss_price_is_three_percent_below_entry():
    assert stop_loss_price(Decimal("100")) == Decimal("97")


def test_before_rotation_cutoff_true_well_before_close():
    assert _before_rotation_cutoff(_BEFORE_CUTOFF) is True


def test_before_rotation_cutoff_false_at_market_close():
    assert _before_rotation_cutoff(_PAST_CUTOFF) is False


def test_cutoff_boundary_is_exclusive():
    # 09:00 UTC = 14:30 IST, exactly ROTATION_CUTOFF_IST -- not "before" it.
    at_cutoff = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    assert _before_rotation_cutoff(at_cutoff) is False


@pytest.mark.asyncio
async def test_no_eviction_past_market_close_cutoff():
    losing_weak = _position("WEAK", Decimal("100"), prediction_at_entry=Decimal("1"))
    account = _FakePaperAccountRepository([losing_weak])
    candles = _FakeCandleRepository({"WEAK": Decimal("90")})  # -10%, clearly a loss

    result = await try_evict_and_open(
        "STRONG", _PAST_CUTOFF, Decimal("50"), Decimal("9"), account, candles, "1h"
    )

    assert result is None
    assert account.closed == []


@pytest.mark.asyncio
async def test_no_eviction_when_open_position_is_profitable():
    winning_weak = _position("WEAK", Decimal("100"), prediction_at_entry=Decimal("1"))
    account = _FakePaperAccountRepository([winning_weak])
    candles = _FakeCandleRepository({"WEAK": Decimal("110")})  # +10%, currently winning

    result = await try_evict_and_open(
        "STRONG", _BEFORE_CUTOFF, Decimal("50"), Decimal("9"), account, candles, "1h"
    )

    assert result is None
    assert account.closed == []


@pytest.mark.asyncio
async def test_no_eviction_when_score_margin_too_small():
    # New candidate's score (5) is only 0.5 above the open position's (4.5)
    # -- below EVICTION_MIN_SCORE_MARGIN (1), so not worth the churn.
    close_score_losing = _position("WEAK", Decimal("100"), prediction_at_entry=Decimal("4.5"))
    account = _FakePaperAccountRepository([close_score_losing])
    candles = _FakeCandleRepository({"WEAK": Decimal("90")})

    result = await try_evict_and_open(
        "STRONG", _BEFORE_CUTOFF, Decimal("50"), Decimal("5"), account, candles, "1h"
    )

    assert result is None
    assert account.closed == []


@pytest.mark.asyncio
async def test_no_eviction_for_position_with_no_stored_score():
    # Opened before this feature existed -- nothing to compare against.
    no_score = PaperPosition(
        symbol="WEAK", entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=Decimal("100"),
        quantity=10, capital_allocated=Decimal("1000"), prediction_at_entry=None,
    )
    account = _FakePaperAccountRepository([no_score])
    candles = _FakeCandleRepository({"WEAK": Decimal("90")})

    result = await try_evict_and_open(
        "STRONG", _BEFORE_CUTOFF, Decimal("50"), Decimal("9"), account, candles, "1h"
    )

    assert result is None


@pytest.mark.asyncio
async def test_evicts_weaker_losing_position_and_opens_new_one():
    weak_losing = _position("WEAK", Decimal("100"), prediction_at_entry=Decimal("1"))
    account = _FakePaperAccountRepository([weak_losing])
    candles = _FakeCandleRepository({"WEAK": Decimal("90")})  # -10%

    result = await try_evict_and_open(
        "STRONG", _BEFORE_CUTOFF, Decimal("50"), Decimal("9"), account, candles, "1h"
    )

    assert result is not None
    assert result.symbol == "STRONG"
    assert account.closed == [("WEAK", _BEFORE_CUTOFF, Decimal("90"))]
    open_symbols = {p.symbol for p in await account.get_open_positions()}
    assert open_symbols == {"STRONG"}


@pytest.mark.asyncio
async def test_evicts_the_worst_performer_among_multiple_eligible():
    mild_loss = _position("MILD", Decimal("100"), prediction_at_entry=Decimal("1"))
    deep_loss = _position("DEEP", Decimal("100"), prediction_at_entry=Decimal("1"))
    account = _FakePaperAccountRepository([mild_loss, deep_loss])
    candles = _FakeCandleRepository({"MILD": Decimal("97"), "DEEP": Decimal("80")})

    result = await try_evict_and_open(
        "STRONG", _BEFORE_CUTOFF, Decimal("50"), Decimal("9"), account, candles, "1h"
    )

    assert result is not None
    assert account.closed == [("DEEP", _BEFORE_CUTOFF, Decimal("80"))]
    open_symbols = {p.symbol for p in await account.get_open_positions()}
    assert open_symbols == {"MILD", "STRONG"}
