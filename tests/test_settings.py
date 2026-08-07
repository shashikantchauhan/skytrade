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
