"""Synthetic protective-option replay for the frozen cash-price strategy.

The underlying directional P&L uses each stored cash entry/exit and one current
futures lot. A long future buys a put at the structural stop; a short future
buys a call there. Option prices use constant-volatility Black--Scholes. This is
a sensitivity study for expired contracts, not an actual options-data backtest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from vollib.black_scholes import black_scholes

from analysis.futures_margin_capital_simulation import simulate

RISK_FREE_RATE = 0.07


def option_price(
    option_type: str,
    spot: float,
    strike: float,
    days: float,
    volatility: float,
) -> float:
    if days <= 0:
        return max(strike - spot, 0.0) if option_type == "p" else max(spot - strike, 0.0)
    return float(black_scholes(option_type, spot, strike, days / 365, RISK_FREE_RATE, volatility))


def price_hedges(trades: pd.DataFrame, *, volatility: float, entry_dte: int) -> pd.DataFrame:
    frame = trades.copy()
    holding_days = (frame.exit_timestamp - frame.entry_timestamp).dt.total_seconds() / (
        24 * 60 * 60
    )
    entry_premiums = []
    exit_premiums = []
    for row, held in zip(frame.itertuples(), holding_days, strict=True):
        option_type = "p" if row.side == 1 else "c"
        strike = float(row.stop)
        entry_premiums.append(
            option_price(option_type, float(row.entry), strike, entry_dte, volatility)
        )
        exit_premiums.append(
            option_price(
                option_type,
                float(row.exit),
                strike,
                max(entry_dte - float(held), 0),
                volatility,
            )
        )
    frame["entry_option_premium"] = entry_premiums
    frame["exit_option_premium"] = exit_premiums
    frame["option_pnl"] = (frame.exit_option_premium - frame.entry_option_premium) * frame.lot_size
    frame["future_pnl"] = frame.side * (frame.exit - frame.entry) * frame.lot_size
    frame["estimated_pnl"] = frame.future_pnl + frame.option_pnl
    frame["notional"] = frame.entry * frame.lot_size
    frame["required_margin"] = frame.notional * frame.side.map({1: 0.098331, -1: 0.092004})
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--lots", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trades = pd.read_csv(args.trades, parse_dates=["entry_timestamp", "exit_timestamp"])
    lots = pd.read_csv(args.lots).drop_duplicates("symbol")
    trades = trades.merge(lots, on="symbol", how="inner")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    baseline = None
    for volatility in (0.20, 0.25, 0.30):
        for entry_dte in (21, 30, 45):
            priced = price_hedges(trades, volatility=volatility, entry_dte=entry_dte)
            yearly = priced.groupby(priced.entry_timestamp.dt.year).estimated_pnl.sum()
            summaries.append(
                {
                    "iv_pct": volatility * 100,
                    "entry_dte": entry_dte,
                    "trades": len(priced),
                    "future_pnl": priced.future_pnl.sum(),
                    "option_pnl": priced.option_pnl.sum(),
                    "combined_pnl": priced.estimated_pnl.sum(),
                    "average_pnl": priced.estimated_pnl.mean(),
                    "win_rate_pct": (priced.estimated_pnl > 0).mean() * 100,
                    "worst_trade": priced.estimated_pnl.min(),
                    "positive_years": int((yearly > 0).sum()),
                }
            )
            if volatility == 0.25 and entry_dte == 30:
                baseline = priced

    pd.DataFrame(summaries).to_csv(output_dir / "hedge-sensitivity.csv", index=False)
    assert baseline is not None
    baseline.to_csv(output_dir / "baseline-trades.csv", index=False)

    capital_rows = []
    for capital in (400_000, 600_000, 800_000, 1_000_000):
        result, _selected = simulate(
            baseline,
            capital,
            margin_ratios={1: 0.098331, -1: 0.092004},
        )
        capital_rows.append(result.__dict__)
    pd.DataFrame(capital_rows).to_csv(output_dir / "baseline-capital.csv", index=False)

    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(capital_rows).to_string(index=False))


if __name__ == "__main__":
    main()
