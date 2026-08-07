"""Market scan workflow for downloading and displaying candle data."""

import logging

import pandas as pd

from trading_scanner.application.validation import CandleValidationError, validate_candles
from trading_scanner.config.settings import AppConfig
from trading_scanner.infrastructure.yahoo import YahooProvider


def scan_symbols(config: AppConfig, symbols: list[str]) -> None:
    """Download, validate, and print candle summaries for all symbols."""
    provider = YahooProvider()
    logger = logging.getLogger(__name__)

    for symbol in symbols:
        logger.info("Downloading %s", symbol)
        try:
            data = provider.get_history(symbol, config.candle_interval, config.candle_history)
            validate_candles(data)
            print_summary(symbol, data)
        except RuntimeError as error:
            logger.error("Download failure for %s: %s", symbol, error)
        except CandleValidationError as error:
            logger.error("Validation failure for %s: %s", symbol, error)
        except Exception:
            logger.exception("Unexpected exception while processing %s", symbol)
        else:
            logger.info("Completed %s", symbol)


def print_summary(symbol: str, data: pd.DataFrame) -> None:
    """Print the newest candle in the requested scanner format."""
    candle = data.iloc[-1]
    print("-" * 50)
    print(f"\n{symbol}\n")
    print(f"Rows:\n{len(data)}\n")
    print("Latest Candle\n")
    print(f"Time: {data.index[-1]}")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        print(f"{column}: {candle[column]}")
    print("-" * 50)
