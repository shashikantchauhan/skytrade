"""Command-line entry point for AlphaEngine CSV validation."""

import argparse
import logging

from trading_scanner.validation import ValidationRunner


def main() -> None:
    """Parse validation options and export a chronological comparison CSV."""
    parser = argparse.ArgumentParser(description="Export AlphaEngine validation data.")
    parser.add_argument("--symbol", default="AARTIIND.NS", help="Yahoo Finance symbol.")
    parser.add_argument("--interval", default="1h", help="Yahoo Finance candle interval.")
    parser.add_argument("--days", type=int, default=10, help="Number of calendar days to download.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timestamp, prediction, and signal for every BUY or SELL bar.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ValidationRunner().run(arguments.symbol, arguments.interval, arguments.days, arguments.verbose)


if __name__ == "__main__":
    main()
