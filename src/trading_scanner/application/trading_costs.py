"""Realistic NSE cash-delivery trading costs -- previously never modeled
anywhere in this codebase (flagged in NOTES.md as a known gap). Every
backtest/simulation P&L number produced before 2026-08-14 is gross, not
net of real costs.

This strategy only ever holds delivery positions (~3.16 days average,
never intraday -- see paper_trading.py's own docstring), and Zerodha (the
broker this project integrates with) charges **zero brokerage on equity
delivery trades**. That's easy to assume means "costs are negligible" --
they aren't, because the regulatory charges below apply regardless of
which broker is used, and the flat DP charge is a real, unavoidable cost
per position closed.

Rates (NSE equity delivery segment, standard as of 2026):

* **STT** (Securities Transaction Tax): 0.1% on BOTH buy and sell value --
  the single largest cost component here, and non-negotiable.
* **Stamp duty**: 0.015% on buy value only (state-mandated, standardized
  nationwide for delivery equity since 2020).
* **NSE exchange transaction charge**: ~0.00297% on both buy and sell value.
* **SEBI turnover fee**: Rs 10 per crore (~0.0001%), both sides.
* **GST**: 18% on (brokerage + exchange transaction charge + SEBI fee) --
  brokerage is 0 for delivery, so this only bites on the tiny exchange/SEBI
  slice, not the STT or stamp duty.
* **DP (Depository Participant) charge**: flat ~Rs 15.93 (Rs 13.5 + 18%
  GST), charged once per symbol per sell-side debit from demat, regardless
  of quantity or position size -- this is a fixed rupee cost, not a
  percentage, so it matters most on small positions (exactly why
  paper_trading.py's MIN_POSITION_SIZE floor rationale already accounted
  for it, even though the P&L math itself never deducted it until now).
"""

from decimal import Decimal

STT_RATE = Decimal("0.001")  # 0.1%, both buy and sell
STAMP_DUTY_RATE = Decimal("0.00015")  # 0.015%, buy side only
EXCHANGE_TXN_RATE = Decimal("0.0000297")  # both sides
SEBI_FEE_RATE = Decimal("0.000001")  # Rs 10/crore, both sides
GST_RATE = Decimal("0.18")
DP_CHARGE = Decimal("15.93")  # flat, sell side only, per symbol per day


def round_trip_cost(entry_value: Decimal, exit_value: Decimal) -> Decimal:
    """Total real cost of one complete delivery trade (entry + exit).

    ``entry_value``/``exit_value`` are the trade's notional value on each
    side (quantity x price) -- not percentages, real rupee amounts, so the
    caller doesn't need to know these rates to use this correctly.
    """
    buy_regulatory = entry_value * (STT_RATE + STAMP_DUTY_RATE + EXCHANGE_TXN_RATE + SEBI_FEE_RATE)
    sell_regulatory = exit_value * (STT_RATE + EXCHANGE_TXN_RATE + SEBI_FEE_RATE)
    gst_base = (
        entry_value * (EXCHANGE_TXN_RATE + SEBI_FEE_RATE)
        + exit_value * (EXCHANGE_TXN_RATE + SEBI_FEE_RATE)
    )
    return buy_regulatory + sell_regulatory + gst_base * GST_RATE + DP_CHARGE
