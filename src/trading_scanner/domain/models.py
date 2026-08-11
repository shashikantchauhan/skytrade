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
    # Drives Telegram formatting only (see infrastructure/telegram.py) --
    # not persisted, not used for fingerprint/dedup, purely "which kind of
    # message is this so the notifier can header/emoji it distinctly":
    #   "entry"      -- a fresh BUY/SELL strategy signal
    #   "exit"       -- the strategy itself closing a position (informational
    #                    only when no paper position was open for it)
    #   "paper_exit" -- an actual simulated-capital paper position closing,
    #                    the one with real P&L
    category: str = "entry"

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

    Two distinct purposes, two independent strategies tracked side by side
    for comparison (never a real order, both analysis-only):

    - ``purpose="primary"``: the futures position *is* the trade --
      ``side="long"`` for a BUY signal, ``side="short"`` for a SELL signal
      (the real short mechanism a real broker connection would use, unlike
      equity, where the NSE cash market has no short selling). Paired with
      an ``OptionsShadowTrade`` hedge (``purpose="hedge"``) opened at the
      same time: a PUT for a long future, a CALL for a short future.
    - ``purpose="hedge"``: the reverse structure -- the directional option
      (``OptionsShadowTrade``, ``purpose="directional"``) is the trade, and
      *this* future is what hedges it, opposite-delta to the option: a
      short future hedging a bought CALL, a long future hedging a bought
      PUT.

    Both purposes can be open for the same symbol at once, hence every
    lookup/close is scoped by ``(symbol, purpose)``, not just symbol.
    """

    symbol: str
    side: str  # "long" | "short"
    futures_tradingsymbol: str
    expiry: str
    lot_size: int
    entry_timestamp: datetime
    entry_price: Decimal
    purpose: str = "primary"  # "primary" | "hedge"
    exit_timestamp: datetime | None = None
    exit_price: Decimal | None = None
    pnl_amount: Decimal | None = None
    pnl_percent: Decimal | None = None
    status: str = "open"  # "open" | "closed"
    source: str = "live"  # "live" (forward shadow-tracking) | "backtest" (current-month replay)
