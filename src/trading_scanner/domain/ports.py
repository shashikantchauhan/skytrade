from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from trading_scanner.domain.models import (
    Candle,
    FuturesPaperPosition,
    PaperPosition,
    Signal,
    SignalSide,
    Trade,
)


@dataclass(frozen=True, slots=True)
class EngineState:
    """The small, opaque-to-the-port state AlphaEngine's fast incremental
    evaluation (application/fast_predict.py) needs carried forward between
    hourly runs. ``queue_json`` is a serialized ``fast_predict.QueueState``;
    ports stay ignorant of that type's shape, only the application layer
    (de)serializes it. ``last_bar_timestamp`` records which candle was last
    advanced into the queue, so a run that sees no new candle since the
    previous run can skip re-advancing (advancing the same bar twice would
    double-count its contribution to the persisted neighbor queue)."""

    signal: int = 0
    queue_json: str | None = None
    exit_state_json: str | None = None
    last_bar_timestamp: str | None = None


class MarketDataProvider(Protocol):
    async def get_candles(self, symbol: str, interval: str) -> Sequence[Candle]: ...


class Notifier(Protocol):
    async def send_signal(self, signal: Signal) -> None: ...

    async def send_text(self, message: str) -> None: ...


class SignalRepository(Protocol):
    async def contains(self, fingerprint: str) -> bool: ...

    async def record(self, fingerprint: str, created_at: datetime) -> None: ...


class CandleRepository(Protocol):
    async def upsert_candles(
        self, symbol: str, interval: str, candles: Sequence[Candle]
    ) -> None: ...

    async def get_candles(
        self, symbol: str, interval: str, limit: int | None = None
    ) -> Sequence[Candle]: ...


class EngineStateRepository(Protocol):
    """Persists the small state AlphaEngine's fast incremental evaluation
    needs carried forward between hourly runs (see
    application/fast_predict.py). Returns ``EngineState()`` (defaults) for a
    symbol never seen before, signaling the caller to bootstrap."""

    async def get_state(self, symbol: str, interval: str) -> EngineState: ...

    async def set_state(self, symbol: str, interval: str, state: EngineState) -> None: ...


class TradeRepository(Protocol):
    """Tracks entries/exits for win-rate and backtest analysis (separate
    from ``SignalRepository``, which only dedupes notifications)."""

    async def open_trade(self, interval: str, trade: Trade) -> None: ...

    async def close_open_trade(
        self,
        symbol: str,
        interval: str,
        side: SignalSide,
        exit_timestamp: datetime,
        exit_price: Decimal,
    ) -> None: ...

    async def abandon_open_trade(self, symbol: str, interval: str, side: SignalSide) -> None:
        """Discard a still-open trade without scoring it as a win or a loss.

        Mirrors Pine's ``ml.backtest``: a new opposite-side entry silently
        overwrites the single ``start_long_trade``/``start_short_trade``
        variable it tracks per symbol, so a position that never reached its
        own exit condition is never counted -- not a win, not a loss.
        """
        ...

    async def get_trades(self, symbol: str | None, interval: str) -> Sequence[Trade]: ...


class PaperAccountRepository(Protocol):
    """Tracks the paper-trading account's cash balance and open/closed positions.

    Long-only (see ``application/paper_trading.py``): every ``PaperPosition``
    here represents real capital committed against a BUY entry, never a
    short. One account only -- no ``symbol`` scoping on the cash balance.
    """

    async def get_cash_balance(self) -> Decimal:
        """Return the current cash balance, initializing it on first call."""
        ...

    async def open_position(self, position: PaperPosition) -> None:
        """Record a new open position and deduct its capital from cash."""
        ...

    async def close_position(
        self,
        symbol: str,
        exit_timestamp: datetime,
        exit_price: Decimal,
    ) -> PaperPosition | None:
        """Close the most recent open position for ``symbol``, crediting cash.

        Returns the closed position (with pnl_amount filled in), or None if
        nothing was open (e.g. never eligible, or capacity was full at entry).
        """
        ...

    async def get_open_positions(self) -> Sequence[PaperPosition]: ...


class FuturesPaperAccountRepository(Protocol):
    """Tracks the futures paper account's own capital pool and open/closed
    futures+hedge combo positions -- see ``application/futures_trading.py``.
    Separate book from ``PaperAccountRepository``'s cash pool by design (see
    ``FuturesPaperPosition``'s docstring)."""

    async def get_cash_balance(self) -> Decimal:
        """Return the current cash balance, initializing it on first call."""
        ...

    async def open_position(self, position: FuturesPaperPosition) -> None:
        """Record a new open combo and deduct its margin from cash."""
        ...

    async def close_position(
        self,
        symbol: str,
        exit_timestamp: datetime,
        futures_exit_price: Decimal,
    ) -> FuturesPaperPosition | None:
        """Close the most recent open combo for ``symbol``, crediting cash
        back its margin plus/minus the futures leg's P&L (the hedge option
        leg's own P&L is tracked separately by ``options_shadow.py`` and not
        included here -- see ``application/futures_trading.py``).

        Returns the closed position, or None if nothing was open.
        """
        ...

    async def get_open_positions(self) -> Sequence[FuturesPaperPosition]: ...
