# SkyTrade

An NSE market signal scanner built around a Pine Script-derived technical
strategy, with an hourly ingestion pipeline and a simulated (paper) trading
account for tracking performance over time.

## Project Structure

```text
src/trading_scanner/
├── alpha_engine.py      # Strategy signal engine
├── application/         # Pipeline, backtest, and paper-trading logic
├── config/               # Centralized application configuration
├── infrastructure/       # Market-data providers and storage adapters
├── validation/            # Per-bar signal validation tooling
├── main.py                # Application orchestration
├── signals.py              # Hourly pipeline entry point
├── webapp.py                # Web dashboard
└── validate.py               # Validation command-line entry point
config/
└── symbols.txt              # One market symbol per line
```

## Running

Python 3.12 is required.

```bash
poetry install
poetry run trading-scanner
```

Or install dependencies directly:

```bash
PYTHONPATH=src python -m trading_scanner.main
```

## Configuration

Configuration is centralized in `trading_scanner.config.settings` and can be
overridden with environment variables or a `.env` file. See
`config/settings.py` for the full list of settings.

## Symbols File

Place one market symbol per line in `config/symbols.txt`. Blank lines,
whitespace, and duplicate symbols are ignored.

## Signal Validation

Export every candle's signal-engine result to compare it bar-by-bar against
an external reference chart:

```bash
PYTHONPATH=src python -m trading_scanner.validate \
  --symbol SYMBOL.NS \
  --interval 1h \
  --days 10
```

## Hourly Signal Pipeline

`trading_scanner.signals` runs the scanner across every symbol in
`config/symbols.txt` on a schedule, accumulating candles in a database so the
strategy has real warm-up history instead of re-downloading a short window
every run. See `application/signal_pipeline.py` for details.

### Running locally with no account

The storage layer speaks directly to a local SQLite file, so the pipeline can
be developed and tested without a hosted database account:

```bash
TRADING_SCANNER_TURSO_URL="file:local.db" \
PYTHONPATH=src python -m trading_scanner.signals
```

## Web Dashboard

`trading_scanner.webapp` is a small FastAPI app for viewing pipeline/account
status and adjusting configuration. See `webapp.py` for details.

```bash
TRADING_SCANNER_DASHBOARD_PASSWORD=<pick-a-password> \
PYTHONPATH=src python -m trading_scanner.webapp
```

## Tests

```bash
PYTHONPATH=src pytest
```
