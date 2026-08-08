"""Command-line entry point for the market scanner."""

import logging

from trading_scanner.application.scanner import scan_symbols
from trading_scanner.application.symbols import SymbolLoader, SymbolLoadError
from trading_scanner.config.settings import AppConfig, load_config


def configure_logging(config: AppConfig) -> None:
    """Configure console-only logging from application configuration."""
    logging.basicConfig(level=config.logging_level, format="%(asctime)s %(levelname)s: %(message)s")


def main() -> None:
    """Load infrastructure configuration and run one scanner pass."""
    config = load_config()
    configure_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Application Started")

    try:
        symbols = SymbolLoader().load(config.symbols_file)
        logger.info("Loaded %d symbols", len(symbols))
        scan_symbols(config, symbols)
    except SymbolLoadError as error:
        logger.error("%s", error)
    except Exception:
        logger.exception("Unexpected exception during application startup")
    finally:
        logger.info("Application Finished")


if __name__ == "__main__":
    main()
