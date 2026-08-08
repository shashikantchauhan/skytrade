from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class SignalSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: SignalSide
    strategy: str
    timestamp: datetime
    price: Decimal
    rationale: str

    @property
    def fingerprint(self) -> str:
        return f"{self.strategy}:{self.symbol}:{self.side}:{self.timestamp.isoformat()}"


@dataclass(frozen=True, slots=True)
class Trade:
    """A recorded entry (and, once closed, exit) for win-rate/backtest tracking.

    ``side`` follows AlphaEngine's own semantics: BUY is a long entry (profit
    when price rises), SELL is a short entry (profit when price falls) --
    confirmed against the Pine script's own backtest helper, which computes
    long profit as ``exit - entry`` and short profit as ``entry - exit``.
    """

    symbol: str
    side: SignalSide
    entry_timestamp: datetime
    entry_price: Decimal
    prediction_at_entry: int
    is_early_signal_flip: bool
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """A simulated real-money position in the paper-trading account.

    Long-only: NSE cash market doesn't allow short selling for multi-day
    holds, so paper positions only ever come from BUY entries (SELL signals
    stay informational -- see ``application/paper_trading.py``). Distinct
    from ``Trade`` (which records the strategy's own raw backtest scoring
    for every symbol regardless of tradability); this tracks actual capital
    committed and returned by the paper account.
    """

    symbol: str
    entry_timestamp: datetime
    entry_price: Decimal
    quantity: int
    capital_allocated: Decimal
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    status: str = "open"  # "open" | "closed"
