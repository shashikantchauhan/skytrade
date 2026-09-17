"""Temporary capital replay for one-lot directional futures estimates.

Signals and percentage outcomes come from the cash-price strategy backtest.
Current futures lot sizes and empirically observed hedged-margin ratios are
inputs. This does not price futures basis, option P&L, costs, or intratrade MTM.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    starting_capital: float
    signals: int
    accepted: int
    rejected: int
    ending_equity: float
    pnl: float
    total_return_pct: float
    cagr_pct: float
    peak_positions: int
    peak_margin: float
    lowest_free_cash: float
    max_realized_drawdown: float
    max_realized_drawdown_pct: float


def simulate(
    trades: pd.DataFrame,
    starting_capital: float,
    *,
    margin_ratios: dict[int, float],
) -> tuple[SimulationResult, pd.DataFrame]:
    frame = trades.copy()
    frame["required_margin"] = frame.notional * frame.side.map(margin_ratios)
    events: list[tuple[pd.Timestamp, int, str, int]] = []
    for index, row in frame.iterrows():
        events.append((row.exit_timestamp, 0, row.symbol, index))
        events.append((row.entry_timestamp, 1, row.symbol, index))
    events.sort()

    free_cash = starting_capital
    realized_equity = starting_capital
    peak_equity = starting_capital
    lowest_free_cash = starting_capital
    max_drawdown = 0.0
    peak_margin = 0.0
    peak_positions = 0
    open_indices: set[int] = set()
    accepted: list[int] = []

    for _timestamp, event_type, _symbol, index in events:
        row = frame.loc[index]
        if event_type == 0:
            if index not in open_indices:
                continue
            open_indices.remove(index)
            free_cash += float(row.required_margin + row.estimated_pnl)
            realized_equity += float(row.estimated_pnl)
            peak_equity = max(peak_equity, realized_equity)
            max_drawdown = max(max_drawdown, peak_equity - realized_equity)
        elif row.required_margin <= free_cash:
            open_indices.add(index)
            accepted.append(index)
            free_cash -= float(row.required_margin)
            locked_margin = float(frame.loc[list(open_indices), "required_margin"].sum())
            peak_margin = max(peak_margin, locked_margin)
            peak_positions = max(peak_positions, len(open_indices))
            lowest_free_cash = min(lowest_free_cash, free_cash)

    selected = frame.loc[accepted].copy()
    elapsed_years = (frame.entry_timestamp.max() - frame.entry_timestamp.min()).total_seconds() / (
        365.2425 * 24 * 60 * 60
    )
    pnl = float(selected.estimated_pnl.sum())
    ending_equity = starting_capital + pnl
    total_return = pnl / starting_capital * 100
    cagr = ((ending_equity / starting_capital) ** (1 / elapsed_years) - 1) * 100
    result = SimulationResult(
        starting_capital=starting_capital,
        signals=len(frame),
        accepted=len(selected),
        rejected=len(frame) - len(selected),
        ending_equity=ending_equity,
        pnl=pnl,
        total_return_pct=total_return,
        cagr_pct=cagr,
        peak_positions=peak_positions,
        peak_margin=peak_margin,
        lowest_free_cash=lowest_free_cash,
        max_realized_drawdown=max_drawdown,
        max_realized_drawdown_pct=max_drawdown / starting_capital * 100,
    )
    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--lots", required=True)
    parser.add_argument("--output")
    parser.add_argument("--long-margin-ratio", type=float, default=0.098331)
    parser.add_argument("--short-margin-ratio", type=float, default=0.092004)
    args = parser.parse_args()

    trades = pd.read_csv(args.trades, parse_dates=["entry_timestamp", "exit_timestamp"])
    lots = pd.read_csv(args.lots).drop_duplicates("symbol")
    frame = trades.merge(lots, on="symbol", how="inner")
    # The frozen backtest deducted 0.20% friction; add it back for this
    # explicitly cost-free temporary estimate.
    frame["gross_pct"] = frame.net_pct + 0.20
    frame["notional"] = frame.entry * frame.lot_size
    frame["estimated_pnl"] = frame.notional * frame.gross_pct / 100

    outputs = []
    for capital in (400_000, 500_000, 600_000, 800_000, 1_000_000):
        result, _selected = simulate(
            frame,
            capital,
            margin_ratios={1: args.long_margin_ratio, -1: args.short_margin_ratio},
        )
        outputs.append(asdict(result))
    report = pd.DataFrame(outputs)
    print(report.to_string(index=False))
    if args.output:
        report.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
