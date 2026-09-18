# SkyTrade Daily Swing on TradingView

Open `daily_swing_strategy.pine` in TradingView's Pine Editor, save it, and add
it to a regular **30-minute NSE equity chart**. The strategy tests both long and
short underlying signals, enters at the completed confirmation candle's close,
uses the structural stop and 3R target, trails at 1R/2R, and exits after ten
sessions. TradingView commission is set to 0.10% per order side, matching the
Python backtest's 0.20% round-trip friction.

This is a chart-level approximation of the frozen Python strategy. Two inputs
cannot be reproduced exactly in a standalone Pine script:

- The Python setup uses daily breadth calculated across the full Nifty 500.
  TradingView cannot scan that whole universe from one chart, so the Pine port
  omits only that breadth gate.
- Python uses the median volume of the prior 20 observations in the same
  intraday slot. The Pine port uses their mean and assumes 13 chart bars per
  NSE session. Holidays and missing bars can shift those references.

Those differences mean TradingView trade counts and returns are useful for
visual checking and symbol-by-symbol exploration, but they should not be
expected to equal the 180-trade Python research ledger. The Pine strategy tests
the cash/underlying signal; it does not simulate the live futures plus
protective-option basket or its margin and option-premium P&L.

The daily values use the last completed daily candle with a one-bar offset and
`barmerge.lookahead_on`, which is TradingView's documented non-repainting form
for higher-timeframe requests. `process_orders_on_close` is enabled because the
SkyTrade rule enters at the confirmation candle's close.
