from trading_scanner.config.settings import load_config


def test_load_config_uses_milestone_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SCANNER_SCAN_INTERVAL_HOURS", "1")
    monkeypatch.setenv("TRADING_SCANNER_CANDLE_INTERVAL", "1h")
    monkeypatch.setenv("TRADING_SCANNER_CANDLE_HISTORY", "300")
    monkeypatch.setenv("TRADING_SCANNER_SYMBOLS_FILE", "config/symbols.txt")
    monkeypatch.setenv("TRADING_SCANNER_LOGGING_LEVEL", "INFO")

    config = load_config()

    assert config.scan_interval_hours == 1
    assert config.candle_interval == "1h"
    assert config.candle_history == 300
    assert str(config.symbols_file) == "config/symbols.txt"


def test_live_trading_kill_switch_defaults_off(monkeypatch) -> None:
    """Real order execution must never turn itself on -- every one of
    these has to be explicitly set. Unset (the state on a fresh deploy) is
    the safest possible default: no symbols allowed, no real orders."""
    monkeypatch.delenv("TRADING_SCANNER_LIVE_TRADING_ENABLED", raising=False)
    monkeypatch.delenv("TRADING_SCANNER_LIVE_TRADING_SYMBOLS", raising=False)

    config = load_config()

    assert config.live_trading_enabled is False
    assert config.live_trading_symbols == frozenset()


def test_live_trading_enabled_requires_exact_true(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SCANNER_LIVE_TRADING_ENABLED", "yes")

    config = load_config()

    assert config.live_trading_enabled is False  # only the literal "true" turns it on
