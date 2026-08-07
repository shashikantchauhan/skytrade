import pandas as pd
import pytest

from trading_scanner.application.validation import CandleValidationError, validate_candles


def test_validate_candles_requires_ohlcv_columns_and_history() -> None:
    data = pd.DataFrame(
        {
            "Open": range(200),
            "High": range(200),
            "Low": range(200),
            "Close": range(200),
            "Volume": range(200),
        }
    )

    validate_candles(data)


def test_validate_candles_reports_missing_columns() -> None:
    with pytest.raises(CandleValidationError, match="missing required columns"):
        validate_candles(pd.DataFrame({"Open": [1] * 200}))
