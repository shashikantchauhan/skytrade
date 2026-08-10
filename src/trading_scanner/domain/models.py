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


@dataclass(frozen=True, slots=True)
class OptionsShadowTrade:
    """A hypothetical options trade shadowing a BUY/SELL signal.

    Two distinct purposes, both analysis-only, never a real order:

    - ``purpose="directional"``: buying an option as a standalone
      directional bet -- a CALL for a BUY signal, a PUT for a SELL signal
      (NSE cash market has no short selling, so a SELL can never become a
      real/paper equity position -- see ``paper_trading.py`` -- this is
      what a synthetic short via options would have looked like instead).
    - ``purpose="hedge"``: a protective option bought alongside a shadow
      futures position (see ``FuturesShadowTrade``) -- a PUT hedging a long
      future, a CALL hedging a short future.

    ``option_type`` is Kite's own convention: ``"CE"`` (call) or ``"PE"``
    (put). Entirely separate from the paper account's capital.
    """

    symbol: str
    option_type: str  # "CE" | "PE"
    purpose: str  # "directional" | "hedge"
    option_tradingsymbol: str
    strike: Decimal
    expiry: str
    lot_size: int
    entry_timestamp: datetime
    underlying_price_at_entry: Decimal
    entry_premium: Decimal
    exit_timestamp: datetime | None = None
    underlying_price_at_exit: Decimal | None = None
    exit_premium: Decimal | None = None
    pnl_amount: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    source: str = "live"  # "live" (forward shadow-tracking) | "backtest" (current-month replay)


@dataclass(frozen=True, slots=True)
class FuturesShadowTrade:
    """A hypothetical futures trade shadowing a BUY/SELL signal.

    ``side="long"`` for a BUY signal, ``side="short"`` for a SELL signal --
    the real short mechanism a real broker connection would use, unlike
    equity (NSE cash market has no short selling). Always paired with an
    ``OptionsShadowTrade`` hedge (``purpose="hedge"``) opened at the same
    time: a PUT for a long future, a CALL for a short future. Analysis
    only, entirely separate from the paper account's capital, never a real
    order.
    """

    symbol: str
    side: str  # "long" | "short"
    futures_tradingsymbol: str
    expiry: str
    lot_size: int
    entry_timestamp: datetime
    entry_price: Decimal
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    source: str = "live"  # "live" (forward shadow-tracking) | "backtest" (current-month replay)
