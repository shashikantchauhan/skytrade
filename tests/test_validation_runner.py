from pathlib import Path

import pandas as pd

from trading_scanner.infrastructure.yahoo import YahooFinanceProvider
from trading_scanner.validation.runner import ValidationRunner


def test_validation_runner_writes_one_row_per_downloaded_candle(
    monkeypatch, tmp_path: Path
) -> None:
    downloaded = pd.DataFrame(
        {
            "Open": [100.0 + index for index in range(12)],
            "High": [101.0 + index for index in range(12)],
            "Low": [99.0 + index for index in range(12)],
            "Close": [100.5 + index for index in range(12)],
            "Volume": [1_000 + index for index in range(12)],
        },
        index=pd.date_range("2026-01-01", periods=12, freq="h"),
    )
    monkeypatch.setattr(
        YahooFinanceProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: downloaded,
    )

    output_path = ValidationRunner(tmp_path).run("AARTIIND.NS", "1h", 10)
    exported = pd.read_csv(output_path)

    assert output_path.name == "validation_AARTIIND_1h.csv"
    assert len(exported) == len(downloaded)
    assert list(exported.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "prediction",
        "prediction_direction",
        "is_bullish",
        "is_bearish",
        "is_ema_uptrend",
        "is_ema_downtrend",
        "is_sma_uptrend",
        "is_sma_downtrend",
        "start_long_trade",
        "start_short_trade",
        "end_long_trade",
        "end_short_trade",
        "signal",
    ]
