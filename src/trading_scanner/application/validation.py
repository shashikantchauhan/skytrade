"""Validation for downloaded OHLCV candle data."""

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class CandleValidationError(ValueError):
    """Raised when downloaded candles cannot be used by a strategy."""


def validate_candles(data: pd.DataFrame, minimum_candles: int = 200) -> None:
    """Ensure candle data has the expected shape and minimum history."""
    if data.empty:
        raise CandleValidationError("Candle data is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise CandleValidationError(f"Candle data is missing required columns: {missing}.")

    if len(data) < minimum_candles:
        raise CandleValidationError(
            f"Candle data has {len(data)} rows; at least {minimum_candles} are required."
        )
