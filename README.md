# Trading Scanner

This project downloads and validates market candles so that a strategy can be
added in a later milestone.

## Project Structure

```text
src/trading_scanner/
├── alpha_engine.py      # TradingView Lorentzian Classification translation
├── application/
│   ├── scanner.py       # Per-symbol download, validation, and summary output
│   ├── symbols.py       # SymbolLoader for the symbols text file
│   └── validation.py    # OHLCV candle validation
├── config/
│   └── settings.py      # Centralized application configuration
├── infrastructure/
│   └── yahoo.py         # YahooProvider market-data adapter
├── validation/
│   └── runner.py         # Per-bar AlphaEngine CSV validation
├── main.py              # Application orchestration
└── validate.py          # Validation command-line entry point
config/
└── symbols.txt          # One market symbol per line
```

## Running Project

Python 3.12 is required.

```bash
poetry install
poetry run trading-scanner
```

Alternatively, install the project dependencies and run:

```bash
PYTHONPATH=src python -m trading_scanner.main
```

## Configuration

Configuration is centralized in `trading_scanner.config.settings`. Defaults can
be overridden with environment variables or a `.env` file:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Scan interval | `TRADING_SCANNER_SCAN_INTERVAL_HOURS` | `1` hour |
| Candle interval | `TRADING_SCANNER_CANDLE_INTERVAL` | `1h` |
| Candle history | `TRADING_SCANNER_CANDLE_HISTORY` | `300` candles |
| Symbols file | `TRADING_SCANNER_SYMBOLS_FILE` | `config/symbols.txt` |
| Logging level | `TRADING_SCANNER_LOGGING_LEVEL` | `INFO` |

The scan interval is configuration only at this stage; the application runs one
scan and does not schedule recurring work.

## Symbols File

Place one Yahoo Finance symbol per line in `config/symbols.txt`. Blank lines,
whitespace, and duplicate symbols are ignored. A missing file produces a clear
error during startup.

## Yahoo Provider

`YahooProvider` downloads unadjusted historical OHLCV data with `yfinance`.
Results are ordered by datetime, incomplete rows are removed, and the returned
data is limited to the configured candle history. Before display, the scanner
requires at least 200 rows and the `Open`, `High`, `Low`, `Close`, and `Volume`
columns.

## Alpha Engine Validation

Export every candle's Alpha Engine result to compare it bar-by-bar with the
TradingView indicator:

```bash
PYTHONPATH=src python -m trading_scanner.validate \
  --symbol AARTIIND.NS \
  --interval 1h \
  --days 10
```

The command downloads all available hourly candles from the requested calendar
day window, evaluates AlphaEngine once for every historical candle, and writes
the result to `validation/validation_AARTIIND_1h.csv`.

## Current Limitations

The exported validation CSV is intended for manual bar-by-bar comparison
against TradingView; it is separate from the accumulated candle history used
by the hourly signal pipeline.
