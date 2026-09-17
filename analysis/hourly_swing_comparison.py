"""Compare the frozen swing strategy with hourly confirmation and execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from analysis.nifty500_swing_backtest import (
    Config,
    backtest,
    daily_features,
    generate_setups,
    load_bars,
    prepare_intraday,
    report,
)


def aggregate_complete_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NSE 09:15-aligned pairs and discard the final half-hour stub."""
    work = bars.copy()
    utc_minutes = work.timestamp.dt.hour * 60 + work.timestamp.dt.minute
    work["hour_slot"] = (utc_minutes - (3 * 60 + 45)) // 60
    hourly = (
        work.groupby(["symbol", "date", "hour_slot"], sort=True)
        .agg(
            timestamp=("timestamp", "max"),
            first_timestamp=("timestamp", "min"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            component_bars=("close", "size"),
        )
        .reset_index()
    )
    separation_seconds = (hourly.timestamp - hourly.first_timestamp).dt.total_seconds()
    complete = (hourly.component_bars == 2) & separation_seconds.eq(30 * 60)
    columns = [
        "symbol",
        "timestamp",
        "hour_slot",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date",
    ]
    return hourly.loc[complete, columns].sort_values(["symbol", "timestamp"])


def run_comparison(
    *, db: str, symbols: str | None, output_directory: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_bars(db, symbols)
    daily, _ = daily_features(raw)
    setups = generate_setups(daily)
    half_hour = prepare_intraday(raw)
    hourly = prepare_intraday(aggregate_complete_hourly(raw))
    config = replace(Config(), confirmation_bars=2)

    hourly_confirmation = backtest(
        hourly,
        daily,
        setups,
        config,
        execution_bars=half_hour,
    )
    fully_hourly = backtest(hourly, daily, setups, config)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    hourly_confirmation.to_csv(destination / "hourly-confirmation-30m-exits.csv", index=False)
    fully_hourly.to_csv(destination / "fully-hourly.csv", index=False)
    return hourly_confirmation, fully_hourly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/skytrade-swing-30m.db")
    parser.add_argument("--symbols", help="optional newline-delimited fixed universe")
    parser.add_argument(
        "--output-directory",
        default="analysis/reports/2026-09-16-hourly-comparison",
    )
    args = parser.parse_args()
    confirmation_result, hourly_result = run_comparison(
        db=args.db,
        symbols=args.symbols,
        output_directory=args.output_directory,
    )
    print("\nHOURLY CONFIRMATION, 30-MINUTE EXITS")
    report(confirmation_result)
    print("\nFULLY HOURLY")
    report(hourly_result)
