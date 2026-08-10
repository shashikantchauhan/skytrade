"""Centralized runtime configuration for the market scanner."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Settings required to run one market scan."""

    scan_interval_hours: int
    candle_interval: str
    candle_history: int
    symbols_file: Path
    logging_level: int
    turso_database_url: str | None
    turso_auth_token: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    index_symbol: str | None
    kite_api_key: str | None
    kite_api_secret: str | None


def load_config() -> AppConfig:
    """Load application settings from environment variables and safe defaults."""
    load_dotenv()
    return AppConfig(
        scan_interval_hours=_positive_int("TRADING_SCANNER_SCAN_INTERVAL_HOURS", 1),
        candle_interval=os.getenv("TRADING_SCANNER_CANDLE_INTERVAL", "1h"),
        candle_history=_positive_int("TRADING_SCANNER_CANDLE_HISTORY", 300),
        symbols_file=Path(os.getenv("TRADING_SCANNER_SYMBOLS_FILE", "config/symbols.txt")),
        logging_level=_logging_level(os.getenv("TRADING_SCANNER_LOGGING_LEVEL", "INFO")),
        turso_database_url=os.getenv("TRADING_SCANNER_TURSO_URL"),
        turso_auth_token=os.getenv("TRADING_SCANNER_TURSO_AUTH_TOKEN"),
        telegram_bot_token=os.getenv("TRADING_SCANNER_TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TRADING_SCANNER_TELEGRAM_CHAT_ID"),
        # NIFTY 50 -- broad NSE benchmark, not tied to any single sector.
        # Evaluated once per run and shown alongside every stock signal so you
        # can judge whether a signal lines up with the broader market or looks
        # like noise against it. Set to "" to disable index tracking entirely.
        index_symbol=os.getenv("TRADING_SCANNER_INDEX_SYMBOL", "^NSEI") or None,
        kite_api_key=os.getenv("TRADING_SCANNER_KITE_API_KEY"),
        kite_api_secret=os.getenv("TRADING_SCANNER_KITE_API_SECRET"),
    )


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer environment setting."""
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {parsed}.")
    return parsed


def _logging_level(value: str) -> int:
    """Convert a configured logging level name into a logging constant."""
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"TRADING_SCANNER_LOGGING_LEVEL is invalid: {value!r}.")
    return level
