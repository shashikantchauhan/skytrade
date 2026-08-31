"""Hard entry-quality gate for real cash orders: reject a BUY candidate
outright when its own entry candle didn't close with real conviction --
close-location-value (CLV) below ``CONVICTION_THRESHOLD``. On top of (not
instead of) ``entry_quality_filter.passes_indicator_filter`` and
``paper_trading.is_eligible``.

CLV = (close - low) / (high - low): 1.0 means the bar closed at its own
high (maximum bullish conviction -- buyers held the whole move right to
the close, no pullback), 0.0 means it closed at its low (buyers gave the
whole move back before the close, even if the bar still looks "green" on
open vs close alone). 0.5 for a zero-range bar rather than NaN/inf.

Why this threshold: real-gated BUY trades (2025-06-11 -- 2026-08-31,
causally reconstructed 55%/5-trade eligibility + entry_quality_filter, see
``analysis/conviction_threshold_filter.py`` for the full script) split by
entry-candle CLV show a real, MONOTONIC relationship -- every single
threshold step improves both win rate and expectancy, no reversals:

    CLV >=   n      win rate   expectancy
    0.0    1077      60.0%      +1.342%
    0.5     822      65.2%      +1.544%
    0.6     729      66.1%      +1.629%
    0.7     634      67.5%      +1.761%
    0.8     497      68.8%      +1.788%
    0.9     273      71.8%      +2.090%

0.7 chosen as the deployed floor: a real, substantial lift (67.5% vs 60.0%
win rate, +31% better expectancy) while keeping enough trade volume (634 of
1077, -41%) not to starve the 8-slot system of signals. Confirmed at the
real deployed size in ``analysis/capital_simulation_50k_8slots.py``'s
"+ ranking + conviction" scenario: layering this on top of
entry_quality_filter + ranking beat every other combination tested --
higher win rate (61.5% vs 56.1% baseline), higher ROI (+73.4% vs +64.7%),
LOWER drawdown (Rs6,130 vs Rs7,350), and FEWER "no free slot" skips (164 vs
473) than quality-only or quality+ranking alone. That last point matters:
unlike the expectancy hard-filter tested earlier this session (which hurt
via capacity dilution -- rejecting candidates that would have competed for
an already-scarce slot anyway), this filter cuts weak candidates before
they'd have been in that competition, so it doesn't starve capital.

Operates on the entry candle's own OHLC -- the same bar the BUY signal
itself fires on, whose close is already known at signal time. Not
look-ahead.

Only gates real cash order placement (``application/live_cash_execution.py``
-- see its call site in ``signal_pipeline.py``), same as
``entry_quality_filter``/``paper_trading.is_eligible``: a rejected
candidate still gets its ``Trade`` row and feeds eligibility/quality-filter
history, it just never reaches ``execute_cash_entry``. Exits are never
gated by this (squaring off an already-open real position is never blocked
on entry-quality grounds, same reasoning as the other two gates).
"""

from decimal import Decimal

CONVICTION_THRESHOLD = 0.7


def close_location_value(high: Decimal, low: Decimal, close: Decimal) -> float:
    """Where the close landed within its own bar's [low, high] range: 1.0
    at the high, 0.0 at the low, 0.5 for a zero-range bar (rather than
    raising or returning NaN/inf)."""
    bar_range = high - low
    if bar_range == 0:
        return 0.5
    return float((close - low) / bar_range)


def passes_conviction_filter(
    high: Decimal, low: Decimal, close: Decimal, threshold: float = CONVICTION_THRESHOLD
) -> bool:
    """True only when the entry candle's own close-location-value clears
    ``threshold``."""
    return close_location_value(high, low, close) >= threshold
