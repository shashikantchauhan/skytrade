"""Classic bullish candlestick reversal/continuation patterns, computed
purely from stored candles -- no relation to AlphaEngine (never touches
``alpha_engine.py``, per NOTES.md's hard constraint; these are separate,
disposable overlay signals).

Every function here takes the *last N* candles (chronological order,
oldest first) and returns whether they form that specific pattern --
fewer than N candles always returns False. Only bullish/BUY-side patterns
are implemented: this system is long-only in the cash market (see
``paper_trading.py``'s own docstring), so a bearish pattern would never
gate anything real.

See ``analysis/candlestick_pattern_screen.py`` for the walk-forward
backtest of whether any of these patterns' presence at entry actually
correlates with better real trade outcomes -- this module is only the
pure pattern detectors, no backtest logic.
"""

from collections.abc import Sequence
from decimal import Decimal

from trading_scanner.domain.models import Candle


def _is_bullish(candle: Candle) -> bool:
    return candle.close > candle.open


def _is_bearish(candle: Candle) -> bool:
    return candle.close < candle.open


def _body_range(candle: Candle) -> tuple[Decimal, Decimal]:
    return min(candle.open, candle.close), max(candle.open, candle.close)


def _body_size(candle: Candle) -> Decimal:
    return abs(candle.close - candle.open)


def bullish_engulfing(candles: Sequence[Candle]) -> bool:
    """Last 2 candles: a bearish candle followed by a bullish candle whose
    real body completely engulfs the first's -- opens below the first's
    close, closes above the first's open."""
    if len(candles) < 2:
        return False
    first, second = candles[-2], candles[-1]
    return (
        _is_bearish(first)
        and _is_bullish(second)
        and second.open < first.close
        and second.close > first.open
    )


def bullish_harami(candles: Sequence[Candle]) -> bool:
    """Last 2 candles: a bearish candle followed by a smaller bullish
    candle whose entire real body sits strictly *inside* the first's --
    the 2-candle "inside bar" pattern, without Three Inside Up's third
    confirming candle."""
    if len(candles) < 2:
        return False
    first, second = candles[-2], candles[-1]
    if not (_is_bearish(first) and _is_bullish(second)):
        return False
    low, high = _body_range(first)
    return low < second.open < high and low < second.close < high


def piercing_line(candles: Sequence[Candle]) -> bool:
    """Last 2 candles: a bearish candle followed by a bullish candle that
    opens below the first's close (a gap down at the open), then recovers
    to close above the midpoint of the first candle's real body -- but not
    so far that it fully engulfs it (that would be ``bullish_engulfing``
    instead)."""
    if len(candles) < 2:
        return False
    first, second = candles[-2], candles[-1]
    if not (_is_bearish(first) and _is_bullish(second)):
        return False
    if second.open >= first.close:
        return False
    midpoint = (first.open + first.close) / 2
    return midpoint < second.close < first.open


def three_white_soldiers(candles: Sequence[Candle]) -> bool:
    """Last 3 candles: three consecutive bullish candles, each opening
    within the previous candle's real body, each closing higher than the
    last, each closing near its own high (a small upper wick -- real
    continuation strength, not an indecisive candle that happened to close
    green)."""
    if len(candles) < 3:
        return False
    first, second, third = candles[-3], candles[-2], candles[-1]
    if not (_is_bullish(first) and _is_bullish(second) and _is_bullish(third)):
        return False
    if not (first.close < second.close < third.close):
        return False
    if not (first.open < second.open < first.close):
        return False
    if not (second.open < third.open < second.close):
        return False
    # Small upper wick: the leftover distance to the high is a modest
    # fraction of the candle's own real body, for every candle.
    for candle in (first, second, third):
        body = _body_size(candle)
        if body == 0 or (candle.high - candle.close) > body * Decimal("0.3"):
            return False
    return True


def morning_star(candles: Sequence[Candle]) -> bool:
    """Last 3 candles: a large bearish candle, a small-bodied indecisive
    "star" candle (real body well under a third of the first candle's),
    then a large bullish candle closing back above the midpoint of the
    first candle's real body -- a reversal pattern, not a continuation
    one."""
    if len(candles) < 3:
        return False
    first, second, third = candles[-3], candles[-2], candles[-1]
    if not (_is_bearish(first) and _is_bullish(third)):
        return False
    first_body = _body_size(first)
    if first_body == 0:
        return False
    if _body_size(second) >= first_body * Decimal("0.3"):
        return False
    midpoint = (first.open + first.close) / 2
    return third.close > midpoint


def three_outside_up(candles: Sequence[Candle]) -> bool:
    """Last 3 candles: a ``bullish_engulfing`` pair, followed by a third
    bullish candle that closes even higher, confirming the reversal
    continues rather than immediately fading."""
    if len(candles) < 3:
        return False
    if not bullish_engulfing(candles[-3:-1]):
        return False
    third = candles[-1]
    return _is_bullish(third) and third.close > candles[-2].close


def three_inside_up(candles: Sequence[Candle]) -> bool:
    """Last 3 candles form a bullish "Three Inside Up" reversal:
    ``bullish_harami`` (a bearish candle, then a smaller bullish candle
    whose real body sits strictly inside it), followed by a third bullish
    candle closing above the *first* candle's open -- confirms the
    reversal actually broke back above where the down-day started, not
    just a small bounce still contained within it."""
    if len(candles) < 3:
        return False
    if not bullish_harami(candles[-3:-1]):
        return False
    first, third = candles[-3], candles[-1]
    return _is_bullish(third) and third.close > first.open
