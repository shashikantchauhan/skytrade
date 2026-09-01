"""Centralized runtime configuration for the market scanner."""

import logging
import os
from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Settings required to run one market scan.

    Static, process-startup configuration ONLY -- loaded once by
    ``load_config()`` and never mutated or cloned afterward (see Phase 11
    of `projectedPlann.md`, ``docs/architecture/000-audit.md``). Anything
    a human can change while the process is already running (the
    dashboard's "Go Live" cash-trading toggle: enabled/symbols/notional/
    max_positions) is deliberately NOT one of these fields' live value --
    it lives in its own dedicated runtime-state type instead
    (``infrastructure/db/live_cash_toggle.py``'s ``LiveCashToggleState``,
    read fresh from the DB every scan cycle) and is threaded through as
    its own explicit parameter wherever it's needed, alongside this
    (unrelated, still-static) config.

    2026-09-01: this distinction used to be blurred -- a
    ``dataclasses.replace(AppConfig, ...)`` clone merging the DB toggle's
    live values onto a copy of this static config was built, and a bug in
    which of the two nearly-identical ``AppConfig`` objects got passed
    into one call site silently disabled real order execution in
    production with zero log output (see ``application/live_cash_
    execution.py``'s own module docstring for the incident). Never
    reintroduce that pattern -- a dashboard-adjustable setting belongs in
    its own type, not folded into a clone of this one.
    """

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
    # The kill switch for real order execution -- see application/
    # live_execution.py. Defaults fully OFF; every one of these must be
    # explicitly set to place a single real order. live_trading_symbols
    # empty means nothing is allowed regardless of live_trading_enabled --
    # there is no "all symbols" wildcard, on purpose.
    live_trading_enabled: bool
    live_trading_symbols: frozenset[str]
    live_trading_max_lots: int
    # Real, capital-gated futures paper account (application/futures_trading.py)
    # -- restricted to this allowlist (Nifty50 by default) rather than the
    # full 220-symbol universe, so the extra Kite margin-API call per signal
    # this account needs stays bounded. Empty file/no file -> nothing
    # trades on this book, same no-wildcard-default philosophy as
    # live_trading_symbols above.
    futures_paper_symbols_file: Path
    # 2026-08-17: distinguishes which deployment a Telegram message came
    # from -- e.g. this repo's own p-trade vs. the skytrade-smallcap fork,
    # which reuses the exact same bot/chat ID. Shown in every message
    # header (see infrastructure/telegram.py's send_text). Defaulted
    # here (unlike every other field above) so existing AppConfig(...)
    # call sites -- test fixtures mostly -- don't all need updating just
    # for this; load_config() below still sets it explicitly from env.
    notification_label: str = "Cash"
    # A second, independent kill switch for real NSE cash-equity orders
    # (application/live_cash_execution.py) -- deliberately its own flag/
    # allowlist, not reused from live_trading_* above, since that one gates
    # the futures+hedge basket. Same no-wildcard-default philosophy: empty
    # symbols means nothing trades. These three are the *startup* defaults
    # -- live_pipeline.py overrides them every cycle from the DB-backed
    # ``live_cash_toggle`` repository (see webapp.py's "Go Live" endpoint),
    # so the dashboard toggle takes effect immediately, no restart needed.
    # ``signals.py``'s plain hourly CLI has no such per-cycle refresh and
    # just uses these static values as-is.
    # 2026-08-21: quantity is sized from a fixed rupee amount per symbol
    # (``live_cash_trading_notional`` / that bar's price), not a fixed
    # share count -- a flat share count meant wildly different real risk
    # across a Rs50 stock and a Rs3,000 stock. Rs5,000/symbol is this
    # week's trial size; scales up once the trial validates timing.
    # Defaulted (like notification_label above) so existing AppConfig(...)
    # call sites -- test fixtures mostly -- don't all need updating just
    # for this; load_config() below still sets them explicitly from env.
    live_cash_trading_enabled: bool = False
    live_cash_trading_symbols: frozenset[str] = frozenset()
    live_cash_trading_notional: Decimal = Decimal("5000")
    # 2026-08-21: caps real positions open at once *across the whole
    # allowlist*, not per symbol -- lets live_cash_trading_symbols be wide
    # (the full universe) without waiting on one specific symbol's signal,
    # while still bounding total real capital at risk to this * notional.
    # See infrastructure/db/live_cash_toggle.py (the dashboard toggle
    # overrides this at runtime, same as the three fields above it).
    live_cash_trading_max_positions: int = 8
    # 2026-08-25: no new real entries once IST wall-clock time reaches this
    # -- orders placed in NSE's final ~10-15 minutes were observed either
    # taking far longer than the fill-timeout to match, or getting
    # cancelled outright by the exchange for lack of a counterparty in
    # thin closing-session liquidity. Exits are never affected -- squaring
    # off an already-open real position must always be allowed regardless
    # of time. None disables the cutoff entirely.
    live_cash_entry_cutoff_ist: time | None = time(15, 15)


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
        notification_label=os.getenv("TRADING_SCANNER_NOTIFICATION_LABEL", "Cash"),
        # NIFTY 50 -- broad NSE benchmark, not tied to any single sector.
        # Evaluated once per run and shown alongside every stock signal so you
        # can judge whether a signal lines up with the broader market or looks
        # like noise against it. Set to "" to disable index tracking entirely.
        index_symbol=os.getenv("TRADING_SCANNER_INDEX_SYMBOL", "^NSEI") or None,
        kite_api_key=os.getenv("TRADING_SCANNER_KITE_API_KEY"),
        kite_api_secret=os.getenv("TRADING_SCANNER_KITE_API_SECRET"),
        live_trading_enabled=_bool_flag("TRADING_SCANNER_LIVE_TRADING_ENABLED", default=False),
        live_trading_symbols=frozenset(
            s.strip()
            for s in os.getenv("TRADING_SCANNER_LIVE_TRADING_SYMBOLS", "").split(",")
            if s.strip()
        ),
        live_trading_max_lots=_positive_int("TRADING_SCANNER_LIVE_TRADING_MAX_LOTS", 1),
        live_cash_trading_enabled=_bool_flag(
            "TRADING_SCANNER_LIVE_CASH_TRADING_ENABLED", default=False
        ),
        live_cash_trading_symbols=frozenset(
            s.strip()
            for s in os.getenv("TRADING_SCANNER_LIVE_CASH_TRADING_SYMBOLS", "").split(",")
            if s.strip()
        ),
        live_cash_trading_notional=Decimal(
            os.getenv("TRADING_SCANNER_LIVE_CASH_TRADING_NOTIONAL", "5000")
        ),
        live_cash_trading_max_positions=_positive_int(
            "TRADING_SCANNER_LIVE_CASH_TRADING_MAX_POSITIONS", 8
        ),
        live_cash_entry_cutoff_ist=_time_hhmm(
            "TRADING_SCANNER_LIVE_CASH_ENTRY_CUTOFF_IST", time(15, 15)
        ),
        futures_paper_symbols_file=Path(
            os.getenv(
                "TRADING_SCANNER_FUTURES_PAPER_SYMBOLS_FILE", "config/nifty50_symbols.txt"
            )
        ),
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


def _bool_flag(name: str, default: bool) -> bool:
    """Explicit opt-in parsing for the live-trading kill switch -- only the
    exact string "true" (case-insensitive) turns it on; anything else
    (unset, "false", a typo) stays off. No implicit truthiness on a
    setting this consequential."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() == "true"


def _logging_level(value: str) -> int:
    """Convert a configured logging level name into a logging constant."""
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"TRADING_SCANNER_LOGGING_LEVEL is invalid: {value!r}.")
    return level


def _time_hhmm(name: str, default: time | None) -> time | None:
    """Parses an "HH:MM" env setting into a ``time``. Unset -> ``default``;
    explicitly set to "" -> ``None`` (disables whatever this gates)."""
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip() == "":
        return None
    hour, minute = value.strip().split(":")
    return time(int(hour), int(minute))
