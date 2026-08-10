"""Shadow-tracks hypothetical options trades alongside BUY/SELL signals.

Two distinct purposes, both analysis-only, never a real order (see
``domain.models.OptionsShadowTrade`` for the full rationale):

- ``purpose="directional"``: a standalone directional bet -- a CALL for a
  BUY signal, a PUT for a SELL signal. NSE cash market has no short
  selling, so a SELL can never become a real (or paper) equity position
  (see ``paper_trading.py``) -- this shows what buying a PUT instead would
  have looked like.
- ``purpose="hedge"``: a protective option bought alongside a shadow
  futures position (see ``futures_shadow.py``) -- a PUT hedging a long
  future, a CALL hedging a short future.

Every function here is best-effort and silent on failure (returns None
rather than raising) so a missing options chain, a quote hiccup, or Kite
being unavailable never breaks the surrounding signal handling in
``signal_pipeline.py`` -- this is a side observation, not a dependency.
"""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import OptionsShadowTrade
from trading_scanner.infrastructure.kite import KiteDerivativesChain
from trading_scanner.infrastructure.turso import TursoOptionsTradeRepository


async def try_open_option_position(
    symbol: str,
    option_type: str,
    purpose: str,
    entry_timestamp: datetime,
    underlying_price: Decimal,
    derivatives_chain: KiteDerivativesChain,
    options_trade_repository: TursoOptionsTradeRepository,
) -> str | None:
    """Record a hypothetical option entry at the nearest ATM strike/expiry."""
    contract = derivatives_chain.nearest_atm_option(symbol, option_type, float(underlying_price))
    if contract is None:
        return None
    premium = derivatives_chain.ltp(f"NFO:{contract['tradingsymbol']}")
    if premium is None:
        return None
    trade = OptionsShadowTrade(
        symbol=symbol,
        option_type=option_type,
        purpose=purpose,
        option_tradingsymbol=contract["tradingsymbol"],
        strike=Decimal(str(contract["strike"])),
        expiry=str(contract["expiry"]),
        lot_size=int(contract["lot_size"]),
        entry_timestamp=entry_timestamp,
        underlying_price_at_entry=underlying_price,
        entry_premium=Decimal(str(premium)),
    )
    await options_trade_repository.open_trade(trade)
    return (
        f"options-shadow({purpose}): opened {contract['tradingsymbol']} {option_type} @ {premium}"
    )


async def close_option_position(
    symbol: str,
    option_type: str,
    purpose: str,
    exit_timestamp: datetime,
    underlying_price: Decimal,
    derivatives_chain: KiteDerivativesChain,
    options_trade_repository: TursoOptionsTradeRepository,
) -> str | None:
    """Close the matching open shadow option position, if one exists."""
    open_trade = await options_trade_repository.get_open_trade(symbol, option_type, purpose)
    if open_trade is None:
        return None
    premium = derivatives_chain.ltp(f"NFO:{open_trade.option_tradingsymbol}")
    if premium is None:
        return None
    exit_premium = Decimal(str(premium))
    # Buying an option profits when its premium rises above entry --
    # true whether it's a CALL (rises with the underlying) or a PUT
    # (rises when the underlying falls).
    pnl_amount = (exit_premium - open_trade.entry_premium) * open_trade.lot_size
    pnl_percent = (exit_premium - open_trade.entry_premium) / open_trade.entry_premium * 100
    await options_trade_repository.close_trade(
        symbol, option_type, purpose, exit_timestamp, underlying_price,
        exit_premium, pnl_amount, pnl_percent,
    )
    return (
        f"options-shadow({purpose}): closed {open_trade.option_tradingsymbol} {option_type} "
        f"@ {exit_premium} pnl={pnl_percent:.2f}%"
    )
