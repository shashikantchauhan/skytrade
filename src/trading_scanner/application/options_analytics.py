"""Implied volatility + greeks for a stored options trade (shadow or
backtest), computed after the fact from what's already in the database --
strike, premium, underlying price, expiry. No live Kite call needed, which
is why this works identically on months-old backtest rows and today's
shadow trades.

Uses ``vollib`` (Black-Scholes) rather than hand-rolling Newton-Raphson IV
solving -- see the project's own research into this before adding the
dependency: vollib is pure closed-form/numerical Black-Scholes math with no
external data dependency, so "last updated 2023" doesn't matter here the
way it would for a library that has to track a changing API.

None of this feeds back into any trading decision (entry/exit signals,
paper trading, real order execution) -- purely an analysis layer surfaced
on the dashboard so a stored ₹X premium becomes readable as "this was
priced at Y% implied volatility, with a delta of Z."
"""

from datetime import date, datetime
from decimal import Decimal

from vollib.black_scholes.greeks.analytical import delta as _delta
from vollib.black_scholes.greeks.analytical import gamma as _gamma
from vollib.black_scholes.greeks.analytical import theta as _theta
from vollib.black_scholes.greeks.analytical import vega as _vega
from vollib.black_scholes.implied_volatility import implied_volatility as _implied_volatility

# Approximate short-term Indian risk-free rate (91-day T-bill territory) --
# greeks are not very sensitive to this input, so a fixed approximation
# rather than a live rate feed is an acceptable simplification here.
_RISK_FREE_RATE = 0.07

_DAYS_PER_YEAR = 365.0


def _time_to_expiry_years(expiry: str, as_of: datetime) -> float | None:
    """Years between ``as_of`` and ``expiry`` (a "YYYY-MM-DD" string, as
    stored by ``OptionsShadowTrade.expiry``). None if expiry is on/before
    ``as_of`` -- no valid Black-Scholes input for zero/negative time."""
    expiry_date = date.fromisoformat(expiry)
    days = (expiry_date - as_of.date()).days
    if days <= 0:
        return None
    return days / _DAYS_PER_YEAR


def compute_greeks(
    option_type: str,
    underlying_price: Decimal,
    strike: Decimal,
    premium: Decimal,
    expiry: str,
    as_of: datetime,
) -> dict[str, float] | None:
    """IV + delta/theta/gamma/vega for one option at one point in time.

    Returns None rather than raising whenever the inputs don't admit a
    solution -- a premium below intrinsic value, or above the
    theoretical maximum, both of which do happen with real
    (possibly stale/illiquid) recorded premiums, especially near expiry.
    """
    t = _time_to_expiry_years(expiry, as_of)
    if t is None:
        return None
    flag = "c" if option_type == "CE" else "p"
    spot, strike_price, price = float(underlying_price), float(strike), float(premium)
    try:
        sigma = _implied_volatility(price, spot, strike_price, t, _RISK_FREE_RATE, flag)
    except Exception:
        return None
    return {
        "implied_volatility": round(sigma * 100, 2),  # as a percentage
        "delta": round(_delta(flag, spot, strike_price, t, _RISK_FREE_RATE, sigma), 4),
        "theta": round(_theta(flag, spot, strike_price, t, _RISK_FREE_RATE, sigma), 4),
        "gamma": round(_gamma(flag, spot, strike_price, t, _RISK_FREE_RATE, sigma), 6),
        "vega": round(_vega(flag, spot, strike_price, t, _RISK_FREE_RATE, sigma), 4),
    }


def enrich_trade(
    option_type: str,
    strike: Decimal,
    expiry: str,
    entry_timestamp: datetime,
    underlying_price_at_entry: Decimal,
    entry_premium: Decimal,
    exit_timestamp: datetime | None = None,
    underlying_price_at_exit: Decimal | None = None,
    exit_premium: Decimal | None = None,
) -> dict[str, dict[str, float] | None]:
    """Entry greeks always computed; exit greeks only if the trade is
    closed. Shape: {"entry": {...} | None, "exit": {...} | None}."""
    entry = compute_greeks(
        option_type, underlying_price_at_entry, strike, entry_premium, expiry, entry_timestamp
    )
    exit_ = None
    is_closed = (
        exit_timestamp is not None
        and underlying_price_at_exit is not None
        and exit_premium is not None
    )
    if is_closed:
        exit_ = compute_greeks(
            option_type, underlying_price_at_exit, strike, exit_premium, expiry, exit_timestamp
        )
    return {"entry": entry, "exit": exit_}
